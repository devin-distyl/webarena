import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2
import tiktoken
from beartype import beartype

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_stop_action,
    create_scroll_action,
    create_go_back_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
    openai_llm,
)
from llms.tokenizers import Tokenizer
from pydantic import BaseModel

class BrowserAgentOutput(BaseModel):
    action: str
    reasoning: str

class PlanNode:
    """Simple tree node for hierarchical planning"""
    def __init__(self, goal: str, rationale: str = "", parent: Optional['PlanNode'] = None):
        self.goal = goal
        self.rationale = rationale  
        self.parent = parent
        self.children: List['PlanNode'] = []
        self.actions: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        
    def add_child(self, goal: str, rationale: str = "") -> 'PlanNode':
        child = PlanNode(goal, rationale, self)
        self.children.append(child)
        return child
        
    def get_path_to_root(self) -> List[str]:
        """Get goals from root to this node"""
        path = []
        node = self
        while node:
            path.insert(0, node.goal)
            node = node.parent
        return path

class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError

class DistylAgent(Agent):
    
    def __init__(self, lm_config: lm_config.LMConfig, tokenizer: Tokenizer) -> None:

        super().__init__()
        self.lm_config = lm_config
        self.tokenizer = tokenizer
        
        # AgentOccam features
        self.plan_tree_root: Optional[PlanNode] = None
        self.current_plan_node: Optional[PlanNode] = None
        self.valid_actions = {'click', 'type', 'go_back', 'go_home', 'stop', 'scroll', 'note', 'branch', 'prune'}
        
        # Workflow completion tracking
        self.recent_type_actions: List[Dict[str, Any]] = []
        self.navigation_history: List[str] = []
        self.last_observation_hash: str = ""

    def next_action(self, trajectory: Trajectory, intent: str, meta_data: Any):
        logger = logging.getLogger(__name__)
        
        try:
            # Initialize plan tree if needed
            if self.plan_tree_root is None:
                self.plan_tree_root = PlanNode(intent, "Main task goal")
                self.current_plan_node = self.plan_tree_root
            
            # Feature 1: ObservationSimplifier - get cleaned observation 
            current_obs = self._simplify_observation(trajectory)
            
            # Feature 3: SelfPlanningAgentTree - get relevant context
            action_context = self._get_plan_context(trajectory)
            
            # Feature 4: FlatPromptPackager - create minimal prompt
            # Add workflow completion guidance
            workflow_guidance = self._get_workflow_guidance(current_obs, trajectory)
            prompt = self._create_flat_prompt(intent, current_obs, action_context, workflow_guidance)
            
            logger.info(f"=== DISTYL AGENT PROMPT ===")
            logger.debug(f"Flat prompt: {prompt}")
            
            # Get LLM response
            image_data = trajectory[-1].get("observation", {}).get("image") if trajectory and isinstance(trajectory[-1], dict) else None
            response = openai_llm(prompt, image_data, BrowserAgentOutput)
            
            # If structured output fails, try one more time with even more explicit JSON instructions
            if response is None:
                logger.warning("First LLM call failed, trying with enhanced JSON instructions")
                enhanced_prompt = prompt + "\n\nREMINDER: You MUST respond with valid JSON only. Do not include any text before or after the JSON."
                response = openai_llm(enhanced_prompt, image_data, BrowserAgentOutput)
            
            # Handle None response from LLM
            if response is None:
                logger.error("LLM returned None response - likely JSON parsing failure or API error")
                logger.error("This usually means the LLM didn't return valid JSON matching BrowserAgentOutput schema")
                return create_none_action()
            
            # Check if response has required attributes
            if not hasattr(response, 'action') or not hasattr(response, 'reasoning'):
                logger.error(f"LLM response missing required attributes. Got: {type(response)} with attributes: {dir(response) if hasattr(response, '__dict__') else 'N/A'}")
                logger.error(f"Expected: BrowserAgentOutput with 'action' and 'reasoning' fields")
                return create_none_action()
            
            # Handle None or empty action/reasoning
            if response.action is None or response.reasoning is None:
                logger.error(f"LLM response has None values - action: {response.action}, reasoning: {response.reasoning}")
                logger.error("This suggests the JSON was parsed but contained null values")
                return create_none_action()
            
            # Validate action and reasoning are strings
            if not isinstance(response.action, str) or not isinstance(response.reasoning, str):
                logger.error(f"LLM response fields have wrong types - action: {type(response.action)}, reasoning: {type(response.reasoning)}")
                return create_none_action()
            
            logger.info(f"=== DISTYL AGENT RESPONSE ===")
            logger.info(f"Action: {response.action}")
            logger.info(f"Reasoning: {response.reasoning}")
            
            # Feature 2: ActionReducer - parse and validate action
            action = self._parse_and_validate_action(response.action, response.reasoning, current_obs)
            
            # Update workflow tracking
            self._update_workflow_tracking(response.action, action, current_obs)
            
            # Update plan tree based on action
            self._update_plan_tree(response.action, response.reasoning)
            
            return action
                
        except Exception as e:
            logger.error(f"Error in DistylAgent.next_action: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return create_none_action()
    
    def reset(self, test_config_file: str) -> None:
        """Reset planning state"""
        self.plan_tree_root = None
        self.current_plan_node = None
        # Reset workflow tracking
        self.recent_type_actions.clear()
        self.navigation_history.clear()
        self.last_observation_hash = ""
    
    # ==================== AGENTOCCAM FEATURES ====================
    
    def _simplify_observation(self, trajectory: Trajectory) -> str:
        """Feature 1: ObservationSimplifier - Clean and simplify accessibility tree"""
        if not trajectory:
            return "No observation available"
            
        latest_step = trajectory[-1]
        if not isinstance(latest_step, dict):
            return "Invalid observation format"
            
        obs = latest_step.get("observation", {})
        text_tree = obs.get("text", "")
        url = obs.get("url", "")
        
        if not text_tree:
            return f"URL: {url}\nNo text content available"
        
        # Use LLM to clean observation if it's too verbose
        if len(text_tree) > 3000:  # Token threshold
            clean_prompt = f"""Clean this web page accessibility tree by:
                1. Remove redundant StaticText next to links  
                2. Remove visual-only formatting elements (gridcell, columnheader, etc)
                3. Convert tables/lists to clean Markdown
                4. Keep only semantically meaningful interactive elements
                5. Focus on actionable content

                Raw accessibility tree:
                {text_tree[:3000]}...

                Provide a clean, concise text representation:
            """
            
            try:
                clean_obs = openai_llm(clean_prompt, None, str)
                return f"URL: {url}\n{clean_obs}"
            except Exception:
                # Fallback to deterministic cleaning
                pass
        
        # Deterministic cleaning as fallback
        cleaned = self._deterministic_clean_observation(text_tree)
        return f"URL: {url}\n{cleaned}"
    
    def _deterministic_clean_observation(self, text_tree: str) -> str:
        """Deterministic observation cleaning"""
        lines = text_tree.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip visual-only elements
            if any(skip in line.lower() for skip in ['gridcell', 'columnheader', 'row', 'cell']):
                continue
                
            # Skip redundant static text near links
            if 'StaticText' in line and len(cleaned_lines) > 0:
                prev_line = cleaned_lines[-1]
                if 'link' in prev_line.lower() and line.replace('StaticText', '').strip() in prev_line:
                    continue
            
            # Convert common patterns to markdown
            if 'heading' in line.lower():
                text = re.search(r'"([^"]+)"', line)
                if text:
                    level = 1 if 'heading 1' in line.lower() else 2
                    cleaned_lines.append('#' * level + ' ' + text.group(1))
                    continue
            
            if 'button' in line.lower() or 'link' in line.lower():
                text = re.search(r'"([^"]+)"', line)
                id_match = re.search(r'\[(\d+)\]', line)
                if text and id_match:
                    element_type = 'button' if 'button' in line.lower() else 'link'
                    cleaned_lines.append(f"[{id_match.group(1)}] {element_type}: {text.group(1)}")
                    continue
            
            # Keep useful lines as-is
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines[:50])  # Limit to 50 most relevant lines
    
    def _get_plan_context(self, trajectory: Trajectory) -> str:
        """Feature 3: SelfPlanningAgentTree - Get task-relevant action context"""
        if not self.current_plan_node:
            return "No plan context"
        
        # Get current goal path
        goal_path = self.current_plan_node.get_path_to_root()
        goal_context = " > ".join(goal_path)
        
        # Get recent relevant actions (last 5)
        recent_actions = []
        for step in trajectory[-5:]:
            if isinstance(step, dict) and step.get("action_type"):
                action_str = step.get("raw_prediction", "Unknown")
                reasoning = step.get("reasoning", "")
                recent_actions.append(f"{action_str}: {reasoning[:100]}")
        
        # Get notes from current plan node
        notes = "\n".join([f"- {note}" for note in self.current_plan_node.notes[-3:]])
        
        context_str = f"""Current Goal: {self.current_plan_node.goal}
Goal Path: {goal_context}
Recent Actions:
{chr(10).join(recent_actions[-3:])}
Notes:
{notes}"""
        return context_str
    
    def _create_flat_prompt(self, intent: str, observation: str, context: str, workflow_guidance: str = "") -> str:
        """Feature 4: FlatPromptPackager - Create minimal stateless prompt"""
        guidance_section = f"\n{workflow_guidance}\n" if workflow_guidance else ""
        
        prompt_str = f"""You are a web agent. Issue a correct action for the current step.

Task: {intent}
{context}
{guidance_section}
Observation:
{observation}

Actions: 
- click [id] - Click element with given id (e.g., click [123])
- type [id] text - Type text into element (e.g., type [456] Hello World)
- scroll [up/down] - Scroll page (e.g., scroll [down])
- go_back - Go back one page
- go_home - Go to home page
- stop [answer] - Complete task with answer
- note [insight] - Record insight (planning only)
- branch [goal] - Create new subgoal
- prune [reason] - Abandon current plan

IMPORTANT: Use exact bracket format for IDs [123]. Output ONE action only.

You must respond with valid JSON in exactly this format:
{{
    "reasoning": "Your step-by-step reasoning for why you chose this action",
    "action": "The exact action command (e.g., click [123], type [456] Hello World, stop [answer])"
}}

Example response:
{{
    "reasoning": "I need to click the Save button to save the changes I made to the product price.",
    "action": "click [789]"
}}"""
        return prompt_str
    
    def _parse_and_validate_action(self, action_str: str, reasoning: str, observation: str = "") -> Action:
        """Feature 2: ActionReducer - Parse and validate actions"""
        action_str = action_str.strip()
        logger = logging.getLogger(__name__)
        
        # Handle special AgentOccam actions
        if action_str.startswith('note '):
            note_text = action_str[5:].strip()
            if self.current_plan_node:
                self.current_plan_node.notes.append(note_text)
            # Return a benign action since note is just for planning
            return create_none_action()
        
        elif action_str.startswith('branch '):
            goal = action_str[7:].strip()
            if self.current_plan_node:
                self.current_plan_node = self.current_plan_node.add_child(goal, reasoning)
            return create_none_action()
        
        elif action_str.startswith('prune '):
            reason = action_str[6:].strip()
            if self.current_plan_node and self.current_plan_node.parent:
                self.current_plan_node = self.current_plan_node.parent
            return create_none_action()
        
        # Split on first space to get action_type and rest
        parts = action_str.split(None, 1)
        action_type = parts[0].lower() if parts else ""
        action_params = parts[1] if len(parts) > 1 else ""
        
        if action_type not in self.valid_actions:
            logger.warning(f"Invalid action type: {action_type}. Using 'none' action.")
            return create_none_action()
        
        # Parse standard actions with robust format handling
        try:
            if action_type == 'click':
                # Handle both "click 123" and "click [123]" formats
                id_match = re.search(r'(?:\[(\d+)\]|(\d+))', action_params)
                if id_match:
                    element_id = id_match.group(1) or id_match.group(2)
                    # Normalize to bracket format for create_id_based_action
                    normalized_action = f"click [{element_id}]"
                    action = create_id_based_action(normalized_action)
                else:
                    # Try label-to-ID fallback for actions like "click [View]"
                    label_match = re.search(r'\[([^\d][^\]]*)\]', action_params)
                    if label_match and observation:
                        label = label_match.group(1)
                        element_id = self._find_id_by_label(observation, label)
                        if element_id:
                            logger.info(f"Converted label '{label}' to ID '{element_id}'")
                            normalized_action = f"click [{element_id}]"
                            action = create_id_based_action(normalized_action)
                        else:
                            logger.error(f"Could not find ID for label '{label}' in observation")
                            return create_none_action()
                    else:
                        logger.error(f"Could not parse click action: {action_str}")
                        return create_none_action()
                    
            elif action_type == 'type':
                # Handle "type [123] text" format - need to match expected format: type [id] [text] [enter_flag]
                type_match = re.search(r'(?:\[(\d+)\]|(\d+))\s+(.+)', action_params)
                if type_match:
                    element_id = type_match.group(1) or type_match.group(2)
                    text = type_match.group(3).strip()
                    # Remove any existing brackets from text and add proper format
                    text = text.strip('[]')
                    # Format as expected by create_id_based_action: type [id] [text] [1]
                    normalized_action = f"type [{element_id}] [{text}] [1]"
                    action = create_id_based_action(normalized_action)
                else:
                    # Try label-to-ID fallback for actions like "type [price] 27.00"
                    label_match = re.search(r'\[([^\d][^\]]*)\]\s+(.+)', action_params)
                    if label_match and observation:
                        label = label_match.group(1)
                        text = label_match.group(2).strip()
                        element_id = self._find_id_by_label(observation, label)
                        if element_id:
                            logger.info(f"Converted type label '{label}' to ID '{element_id}'")
                            normalized_action = f"type [{element_id}] [{text}] [1]"
                            action = create_id_based_action(normalized_action)
                        else:
                            logger.error(f"Could not find ID for type label '{label}' in observation")
                            return create_none_action()
                    else:
                        logger.error(f"Could not parse type action: {action_str}")
                        return create_none_action()
                    
            elif action_type == 'stop':
                # Handle "stop [answer]" format
                match = re.search(r'\[([^\]]+)\]', action_params)
                answer = match.group(1) if match else action_params.strip()
                action = create_stop_action(answer)
                
            elif action_type == 'scroll':
                # Handle "scroll [direction]" or "scroll direction" format
                direction_match = re.search(r'(?:\[([^\]]+)\]|(\w+))', action_params)
                if direction_match:
                    direction = direction_match.group(1) or direction_match.group(2)
                else:
                    direction = "down"  # default
                action = create_scroll_action(direction)
                
            elif action_type == 'go_back':
                action = create_go_back_action()
                
            elif action_type == 'go_home':
                # Use create_id_based_action with special home action
                action = create_id_based_action("go_home")
                
            else:
                # Fallback: try to use existing parser
                # First normalize the format to ensure brackets
                if not re.search(r'\[\d+\]', action_str) and re.search(r'\d+', action_str):
                    # Convert "action 123" to "action [123]"
                    normalized = re.sub(r'(\w+)\s+(\d+)', r'\1 [\2]', action_str)
                    action = create_id_based_action(normalized)
                else:
                    action = create_id_based_action(action_str)
            
            action["reasoning"] = reasoning
            return action
            
        except (ActionParsingError, Exception) as e:
            logger.error(f"Action parsing error for '{action_str}': {e}")
            return create_none_action()
    
    def _update_plan_tree(self, action_str: str, reasoning: str) -> None:
        """Update plan tree with completed action"""
        if self.current_plan_node:
            self.current_plan_node.actions.append({
                "action": action_str,
                "reasoning": reasoning
            })
    
    # ==================== WORKFLOW INTELLIGENCE FEATURES ====================
    
    def _get_workflow_guidance(self, observation: str, trajectory: Trajectory) -> str:
        """Generate intelligent workflow guidance based on current state"""
        guidance_parts = []
        
        # Check for workflow completion needs
        if self.recent_type_actions:
            guidance_parts.append("⚠️ WORKFLOW INCOMPLETE: You recently typed values but haven't saved. Look for Save/Apply/Submit buttons.")
        
        # Check for "0 records" situations
        if "0 records found" in observation.lower() or "no records" in observation.lower():
            if self._has_active_filters(observation):
                guidance_parts.append("⚠️ EMPTY RESULTS: Grid shows 0 records but has active filters/search. Try clearing filters or changing search terms.")
            elif self._has_search_box(observation):
                guidance_parts.append("⚠️ EMPTY RESULTS: Grid is empty. Try using the search box or applying filters to find data.")
        
        # Check for navigation loops
        if self._detect_navigation_loop():
            guidance_parts.append("⚠️ NAVIGATION LOOP: You're repeating menu clicks. Try search boxes, filters, or breadcrumbs instead.")
        
        # Check for incomplete workflows
        workflow_hint = self._detect_incomplete_workflow(observation)
        if workflow_hint:
            guidance_parts.append(f"⚠️ WORKFLOW HINT: {workflow_hint}")
        
        return "\n".join(guidance_parts) if guidance_parts else ""
    
    def _has_active_filters(self, observation: str) -> bool:
        """Check if there are active filters or search terms"""
        indicators = [
            "filters applied", "search:", "filter by", "showing filtered",
            "clear filters", "reset", "keyword:", "name contains"
        ]
        obs_lower = observation.lower()
        return any(indicator in obs_lower for indicator in indicators)
    
    def _has_search_box(self, observation: str) -> bool:
        """Check if there's a search box available"""
        indicators = ["search", "keyword", "filter", "textbox", "text input"]
        obs_lower = observation.lower()
        return any(f"[{i}" in obs_lower or f"textbox" in obs_lower for i in indicators)
    
    def _detect_navigation_loop(self) -> bool:
        """Detect if agent is clicking the same navigation elements repeatedly"""
        if len(self.navigation_history) < 4:
            return False
        
        # Check for repeated menu clicks
        recent_actions = self.navigation_history[-4:]
        if len(set(recent_actions)) <= 2:  # Only 1-2 unique actions in last 4
            return True
            
        return False
    
    def _detect_incomplete_workflow(self, observation: str) -> str:
        """Detect common incomplete workflow patterns"""
        obs_lower = observation.lower()
        
        # Common workflow completion buttons
        save_buttons = ["save", "apply", "submit", "update", "confirm", "ship", "cancel"]
        
        if self.recent_type_actions and any(btn in obs_lower for btn in save_buttons):
            return "You made changes but haven't saved yet. Look for Save, Apply, or Submit buttons."
        
        if "order #" in obs_lower and ("view" in obs_lower or "edit" in obs_lower):
            if not any(word in obs_lower for word in ["tracking", "ship", "invoice", "cancel"]):
                return "You're viewing an order. Look for Ship, Invoice, Cancel, or Edit buttons to complete the task."
        
        if "product" in obs_lower and "price" in obs_lower and "edit" in obs_lower:
            return "You're editing a product. Make sure to save your changes."
        
        return ""
    
    def _find_id_by_label(self, observation: str, label: str) -> Optional[str]:
        """Find element ID by searching for label text in observation"""
        label_lower = label.lower()
        
        # Look for patterns like: [123] button "View" or [456] link "Save" or [789] textbox "Price"
        patterns = [
            rf'\[(\d+)\][^"]*"[^"]*{re.escape(label_lower)}[^"]*"',  # [123] button "View Order"  
            rf'\[(\d+)\][^"]*{re.escape(label_lower)}[^"]*"',  # [123] "Price" textbox
            rf'\[(\d+)\][^"]*{re.escape(label_lower)}',  # [123] View
            rf'{re.escape(label_lower)}[^"]*\[(\d+)\]',  # Price [123] 
            rf'\[(\d+)\][^"]*textbox[^"]*{re.escape(label_lower)}',  # [123] textbox Price
            rf'\[(\d+)\][^"]*button[^"]*{re.escape(label_lower)}',  # [123] button Save
            # Look for partial matches in case of compound labels like "Price" in "Product Price"
            rf'\[(\d+)\][^"]*["\s]*[^"]*{re.escape(label_lower)}',  # Broader match
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, observation, re.IGNORECASE)
            for match in matches:
                return match.group(1)
        
        return None
    
    def _update_workflow_tracking(self, action_str: str, parsed_action: Action, observation: str) -> None:
        """Update workflow tracking state"""
        action_type = action_str.split()[0].lower() if action_str.split() else ""
        
        # Track type actions for workflow completion
        if action_type == "type":
            self.recent_type_actions.append({
                "action": action_str,
                "step": len(self.recent_type_actions),
                "observation_hash": hash(observation[:500])  # Track context
            })
            # Keep only recent type actions
            if len(self.recent_type_actions) > 3:
                self.recent_type_actions.pop(0)
        
        # Clear type actions if we save/submit/apply
        elif any(save_word in action_str.lower() for save_word in ["save", "apply", "submit", "confirm"]):
            self.recent_type_actions.clear()
        
        # Track navigation for loop detection  
        if action_type == "click":
            # Extract menu-like terms for navigation tracking
            menu_terms = ["catalog", "sales", "customers", "reports", "products", "orders", "invoices"]
            for term in menu_terms:
                if term in action_str.lower():
                    self.navigation_history.append(f"click_{term}")
                    break
            else:
                self.navigation_history.append("click_other")
                
            # Keep only recent navigation
            if len(self.navigation_history) > 6:
                self.navigation_history.pop(0)