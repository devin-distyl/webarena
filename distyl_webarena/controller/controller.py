"""
DistylWebArenaController: Main orchestrator for Distyl-WebArena agent system

This controller adapts Distyl's hierarchical architecture to WebArena's browser environment,
providing WebArena-compatible interface while maintaining Distyl's sophisticated planning
and execution capabilities.
"""

import json
import time
from typing import Any, Dict, List, Tuple

# Import WebArena components directly since environment variables should be set
from agent import Agent
from browser_env import Trajectory
from browser_env.actions import Action, ActionTypes, create_stop_action, create_none_action

from ..planner.web_steps import WebStepPlanner
from ..executor.web_execution import WebExecutor
from ..grounder.web_grounding import AccessibilityTreeGrounder
from ..memory.web_knowledge import WebKnowledgeBase
from ..utils.web_utils import WebUtils
from ..utils.logging import DistylLogger


class DistylWebArenaController(Agent):
    """
    Main controller that adapts Distyl's hierarchical architecture 
    to WebArena's browser environment interface
    
    Implements the WebArena Agent interface with hierarchical planning,
    accessibility tree grounding, and reflection capabilities
    """
    
    def __init__(
        self,
        engine_params: Dict[str, Any],
        action_set_tag: str = "id_accessibility_tree",
        memory_folder_name: str = "distyl_webarena_kb",
        enable_best_of_n: bool = True,
        n_candidates: int = 3,
        grounding_engine_params: Dict[str, Any] = None,
        enable_reflection: bool = True,
        enable_web_knowledge: bool = True,
        log_file: str = None
    ):
        # Initialize parent Agent class
        super().__init__()
        
        self.engine_params = engine_params
        self.action_set_tag = action_set_tag
        self.enable_best_of_n = enable_best_of_n
        self.n_candidates = n_candidates
        self.enable_reflection = enable_reflection
        self.enable_web_knowledge = enable_web_knowledge
        
        # Initialize logger with optional file output
        self.logger = DistylLogger("DistylWebArenaController", log_level="DEBUG", log_file=log_file)
        
        # Initialize components
        self._initialize_components(memory_folder_name, grounding_engine_params)
        
        # State management
        self.current_task_config = None
        self.current_subtasks = []
        self.current_subtask_idx = 0
        self.task_complete = False
        self.failure_count = 0
        self.max_failures = 3
        
        self.logger.info("DistylWebArenaController initialized")
    
    def _initialize_components(self, memory_folder_name: str, grounding_engine_params: Dict[str, Any]):
        """Initialize all Distyl components"""
        
        # Initialize memory system
        self.memory = WebKnowledgeBase(
            memory_folder_name,
            enable_web_knowledge=self.enable_web_knowledge
        )
        
        # Initialize grounder
        self.grounder = AccessibilityTreeGrounder(
            grounding_engine_params or self.engine_params
        )
        
        # Initialize planner
        self.planner = WebStepPlanner(
            self.engine_params,
            self.memory,
            n_candidates=self.n_candidates if self.enable_best_of_n else 1
        )
        
        # Initialize executor
        self.executor = WebExecutor(
            self.engine_params,
            self.grounder,
            self.memory,
            enable_reflection=self.enable_reflection
        )
    
    def next_action(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> Action:
        """
        WebArena-compatible interface that returns single Action
        Internally manages Distyl's multi-step planning and execution
        """
        
        
        try:
            # Convert WebArena trajectory to Distyl internal format
            distyl_context = self._convert_trajectory_to_context(trajectory, meta_data)
            
            # Check if we need to replan or get next subtask
            if self._should_replan(trajectory, intent):
                self.logger.info(f"Replanning for intent: {intent}")
                self._replan(intent, distyl_context)
            
            # Execute current subtask
            if self.current_subtask_idx < len(self.current_subtasks):
                current_subtask = self.current_subtasks[self.current_subtask_idx]
                
                self.logger.info(f"Executing subtask {self.current_subtask_idx + 1}/{len(self.current_subtasks)}: {current_subtask.get('description', '')}")
                self.logger.debug(f"🎯 Current subtask details: {current_subtask}")
                self.logger.debug(f"📋 Distyl context keys: {list(distyl_context.keys())}")
                if 'observation' in distyl_context:
                    obs = distyl_context['observation']
                    self.logger.debug(f"🌐 Current URL: {obs.get('url', 'unknown')}")
                    self.logger.debug(f"📝 Accessibility tree length: {len(obs.get('accessibility_tree', ''))}")
                
                # Debug what data is being sent to executor
                self.logger.debug(f"📡 EXECUTOR_INPUT_DEBUG: Subtask type: {current_subtask.get('type', 'unknown')}")
                self.logger.debug(f"📡 EXECUTOR_INPUT_DEBUG: Site type: {current_subtask.get('site_type', 'unknown')}")
                self.logger.debug(f"📡 EXECUTOR_INPUT_DEBUG: Context contains image: {'observation' in distyl_context and distyl_context['observation'].get('screenshot') is not None}")
                self.logger.debug(f"📡 EXECUTOR_INPUT_DEBUG: Context contains accessibility tree: {'observation' in distyl_context and bool(distyl_context['observation'].get('accessibility_tree', ''))}")
                self.logger.debug(f"📡 EXECUTOR_INPUT_DEBUG: Action history length: {len(distyl_context.get('action_history', []))}")
                
                # Generate action for current subtask
                action_info, actions = self.executor.next_action(
                    subtask=current_subtask,
                    context=distyl_context,
                    trajectory=trajectory
                )
                
                self.logger.debug(f"⚡ Executor returned - Info: {action_info}, Actions: {actions}")
                
                if not actions or not actions[0]:
                    self.logger.warning("No action generated by executor")
                    return create_none_action()
                
                # Convert Distyl action to WebArena Action format
                webarena_action = self._convert_to_webarena_action(actions[0], distyl_context)
                
                # Update trajectory and check subtask completion
                if action_info.get("subtask_complete", False):
                    self.current_subtask_idx += 1
                    self.logger.info(f"Subtask completed, moving to next ({self.current_subtask_idx + 1}/{len(self.current_subtasks)})")
                
                # Reset failure count on successful action
                if webarena_action["action_type"] != ActionTypes.NONE:
                    self.failure_count = 0
                else:
                    self.failure_count += 1
                
                return webarena_action
            else:
                # All subtasks complete
                self.logger.info("All subtasks completed successfully")
                self.task_complete = True
                return create_stop_action("Task completed")
                
        except Exception as e:
            self.logger.error(f"Error in next_action: {str(e)}")
            self.failure_count += 1
            
            if self.failure_count >= self.max_failures:
                return create_stop_action(f"Task failed after {self.max_failures} attempts")
            else:
                return create_none_action()
    
    def reset(self, test_config_file: str) -> None:
        """WebArena-compatible reset for new task"""
        
        self.logger.info(f"Resetting for config: {test_config_file}")
        
        # Load task configuration
        with open(test_config_file) as f:
            config = json.load(f)
        
        # Initialize task context
        self.current_task_config = config
        self.current_subtasks = []
        self.current_subtask_idx = 0
        self.task_complete = False
        self.failure_count = 0
        
        # Reset all Distyl components
        self.planner.reset()
        self.executor.reset()
        self.grounder.reset()
        
        # Initialize memory for this task
        self.memory.initialize_task_trajectory(
            task_id=config.get("task_id"),
            intent=config.get("intent"),
            sites=config.get("sites", [])
        )
        
        self.logger.info(f"Reset complete for task {config.get('task_id')}: {config.get('intent')}")
    
    def finalize_memory(self) -> None:
        """Save accumulated knowledge at task completion"""
        
        if hasattr(self, 'current_task_config') and self.current_task_config:
            self.logger.info("Finalizing memory and saving knowledge")
            
            # Save task-level narrative
            self.memory.save_task_narrative(
                task_config=self.current_task_config,
                subtasks_completed=self.current_subtasks[:self.current_subtask_idx],
                success_status=self.task_complete
            )
            
            # Save successful subtask patterns
            for i, subtask in enumerate(self.current_subtasks[:self.current_subtask_idx]):
                self.memory.save_subtask_experience(subtask, success=True)
    
    def _convert_trajectory_to_context(self, trajectory: Trajectory, meta_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert WebArena trajectory to Distyl internal context"""
        
        if not trajectory:
            return {"observation": None, "action_history": []}
        
        # Extract current observation (last state)
        current_state = trajectory[-1] if trajectory else None
        
        self.logger.debug(f"🔍 TRAJECTORY_DEBUG: Trajectory length: {len(trajectory) if trajectory else 0}")
        self.logger.debug(f"🔍 Current state keys: {list(current_state.keys()) if current_state else None}")
        self.logger.debug(f"🔍 META_DATA_DEBUG: Meta data keys: {list(meta_data.keys()) if meta_data else None}")
        
        if current_state and "observation" in current_state:
            obs = current_state["observation"]
            
            self.logger.debug(f"🔍 OBS_DEBUG: Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'not a dict'}")
            self.logger.debug(f"🔍 OBS_DEBUG: URL value: '{obs.get('url', 'KEY_NOT_FOUND')}'")
            self.logger.debug(f"🔍 OBS_DEBUG: Text length: {len(obs.get('text', ''))}")
            
            # Debug image/screenshot availability
            image_data = obs.get('image')
            if image_data is not None:
                if hasattr(image_data, 'shape'):
                    self.logger.debug(f"🖼️  IMAGE_DEBUG: Screenshot available - Shape: {image_data.shape}, Type: {type(image_data)}")
                else:
                    self.logger.debug(f"🖼️  IMAGE_DEBUG: Image data available - Type: {type(image_data)}, Length: {len(str(image_data))}")
            else:
                self.logger.debug(f"🖼️  IMAGE_DEBUG: No image/screenshot data available")
            
            # Debug accessibility tree content (first 500 chars)
            text_content = obs.get('text', '')
            if text_content:
                preview = text_content[:500].replace('\n', '\\n')
                self.logger.debug(f"🌳 ACCESSIBILITY_TREE_DEBUG: First 500 chars: '{preview}...'")
                
                # Count interactive elements in accessibility tree
                interactive_count = text_content.lower().count('button') + text_content.lower().count('link') + text_content.lower().count('textbox') + text_content.lower().count('combobox')
                self.logger.debug(f"🌳 INTERACTIVE_ELEMENTS_DEBUG: Found ~{interactive_count} interactive elements (button/link/textbox/combobox)")
                
                # Extract and show specific clickable elements (with IDs)
                import re
                element_pattern = r'\[(\d+)\]\s+(button|link|textbox|combobox|menuitem)\s+([^\n]*)'
                matches = re.findall(element_pattern, text_content, re.IGNORECASE)
                if matches:
                    self.logger.debug(f"🎯 AVAILABLE_ACTIONS_DEBUG: Found {len(matches)} actionable elements:")
                    for element_id, element_type, element_text in matches[:10]:  # Show first 10
                        clean_text = element_text.strip()[:50]  # First 50 chars
                        self.logger.debug(f"   [{element_id}] {element_type}: '{clean_text}'")
                    if len(matches) > 10:
                        self.logger.debug(f"   ... and {len(matches) - 10} more elements")
                else:
                    self.logger.debug(f"🎯 AVAILABLE_ACTIONS_DEBUG: No actionable elements found with IDs")
            else:
                self.logger.debug(f"🌳 ACCESSIBILITY_TREE_DEBUG: No accessibility tree text available")
            
            # Extract URL from trajectory info (this is where WebArena stores it)
            found_url = ""
            
            # URL is stored in trajectory state info: trajectory[-1]["info"]["page"].url
            if current_state and "info" in current_state:
                info = current_state["info"]
                if "page" in info:
                    page_info = info["page"]
                    if hasattr(page_info, 'url'):
                        found_url = page_info.url
                    elif isinstance(page_info, dict):
                        found_url = page_info.get("url", "")
            
            # Fallback: check other possible locations
            if not found_url:
                url_candidates = []
                
                # Check meta_data (unlikely but just in case)
                if meta_data:
                    url_candidates.extend([
                        meta_data.get("url", ""),
                        meta_data.get("page_url", ""),
                        meta_data.get("current_url", ""),
                    ])
                
                # Check observation (unlikely but just in case)
                url_candidates.extend([
                    obs.get("url", ""),
                    obs.get("page_url", ""), 
                    obs.get("current_url", ""),
                    getattr(obs, 'url', '') if hasattr(obs, 'url') else ""
                ])
                
                found_url = next((url for url in url_candidates if url), "")
            
            site_context = self._infer_site_context_with_fallback(found_url)
            
            self.logger.debug(f"🔍 URL_EXTRACTION: Found URL from trajectory info: '{found_url}'")
            self.logger.debug(f"🔍 SITE_CONTEXT: Final site context: '{site_context}'")
            
            # Extract key information for Distyl
            context = {
                "observation": {
                    "accessibility_tree": obs.get("text", ""),
                    "url": found_url,
                    "screenshot": obs.get("image") if obs.get("image") is not None else None,
                    "page_title": self._extract_page_title(obs.get("text", ""))
                },
                "action_history": self._extract_action_history(trajectory),
                "site_context": site_context,
                "intent": self.current_task_config.get("intent", "") if self.current_task_config else ""
            }
            
            return context
        
        return {"observation": None, "action_history": []}
    
    def _extract_action_history(self, trajectory: Trajectory) -> List[str]:
        """Extract readable action history from trajectory"""
        actions = []
        for i in range(1, len(trajectory), 2):  # Actions are at odd indices
            if i < len(trajectory):
                action = trajectory[i]
                if isinstance(action, dict) and "action_type" in action:
                    action_str = WebUtils.action_to_string(action, self.action_set_tag)
                    actions.append(action_str)
        return actions[-5:]  # Keep last 5 actions for context
    
    def _extract_page_title(self, accessibility_tree: str) -> str:
        """Extract page title from accessibility tree"""
        lines = accessibility_tree.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if 'title' in line.lower() or 'heading' in line.lower():
                return line.strip()
        return "Unknown Page"
    
    def _infer_site_context(self, url: str) -> str:
        """Infer site type from URL"""
        url_lower = url.lower()
        
        # Check for shopping admin sites first (more specific)
        if "/admin" in url_lower or "7780" in url_lower:
            return "shopping_admin"
        # Check for regular shopping sites
        elif ("shop" in url_lower or "store" in url_lower or "7770" in url_lower):
            return "shopping"
        elif "reddit" in url_lower or "9999" in url_lower:
            return "social"
        elif "gitlab" in url_lower or "8023" in url_lower:
            return "development"
        elif "wikipedia" in url_lower or "8888" in url_lower:
            return "knowledge"
        elif "map" in url_lower or "3000" in url_lower:
            return "mapping"
        else:
            return "general"
    
    def _infer_site_context_with_fallback(self, url: str) -> str:
        """Infer site type from URL with task config fallback"""
        
        # First try URL-based detection
        site_context = self._infer_site_context(url)
        
        if site_context == "general" and self.current_task_config:
            # Fallback to task config sites
            sites = self.current_task_config.get("sites", [])
            self.logger.debug(f"🔍 SITE_FALLBACK: URL gave 'general', checking task sites: {sites}")
            
            if "shopping_admin" in sites:
                self.logger.debug(f"🔍 SITE_FALLBACK: Using 'shopping_admin' from task config")
                return "shopping_admin"
            elif "shopping" in sites:
                self.logger.debug(f"🔍 SITE_FALLBACK: Using 'shopping' from task config")
                return "shopping"
            elif "reddit" in sites:
                return "social"
            elif "gitlab" in sites:
                return "development"
            elif "wikipedia" in sites:
                return "knowledge"
            elif "map" in sites:
                return "mapping"
        
        self.logger.debug(f"🔍 SITE_CONTEXT: URL='{url}' -> Context='{site_context}'")
        return site_context
    
    def _should_replan(self, trajectory: Trajectory, intent: str) -> bool:
        """Determine if we need to replan based on current state"""
        
        # Replan if no current plan
        if not self.current_subtasks:
            return True
        
        # Replan if all subtasks completed
        if self.current_subtask_idx >= len(self.current_subtasks):
            return True
        
        # Replan if too many recent failures
        if self.failure_count >= 2:
            self.logger.warning(f"Replanning due to {self.failure_count} failures")
            return True
        
        # Replan if last few actions failed
        if self._detect_failure_pattern(trajectory):
            return True
        
        return False
    
    def _replan(self, intent: str, context: Dict[str, Any]) -> None:
        """Generate new plan using WebStepPlanner"""
        
        try:
            self.current_subtasks = self.planner.get_action_queue(
                instruction=intent,
                observation=context
            )
            self.current_subtask_idx = 0
            self.failure_count = 0
            
            self.logger.info(f"Generated new plan with {len(self.current_subtasks)} subtasks")
            self.logger.debug(f"📋 FULL PLAN DETAILS:")
            for i, subtask in enumerate(self.current_subtasks):
                self.logger.debug(f"  {i+1}. {subtask.get('description', '')} [Type: {subtask.get('type', 'unknown')}, Site: {subtask.get('site_type', 'unknown')}]")
                
        except Exception as e:
            self.logger.error(f"Error during replanning: {str(e)}")
            self.current_subtasks = []
    
    def _detect_failure_pattern(self, trajectory: Trajectory) -> bool:
        """Detect if recent actions indicate failure"""
        if len(trajectory) < 6:  # Need at least 3 action-observation pairs
            return False
        
        # Check last 3 actions for failure indicators
        recent_actions = trajectory[-6::2]  # Last 3 actions
        
        # Count NONE actions (parsing failures)
        none_count = sum(1 for action in recent_actions 
                        if action.get("action_type") == ActionTypes.NONE)
        
        return none_count >= 2  # Replan if 2+ parsing failures
    
    def _convert_to_webarena_action(self, distyl_action: str, context: Dict[str, Any]) -> Action:
        """Convert Distyl generated action to WebArena Action format"""
        
        try:
            # Parse Distyl action string and convert to WebArena action
            action_str = distyl_action.strip().lower()
            
            if action_str.startswith("click"):
                # Extract element information and create click action
                element_id = self._extract_element_id(distyl_action)
                if element_id:
                    from browser_env.actions import create_click_action
                    return create_click_action(element_id=element_id)
            
            elif action_str.startswith("type"):
                # Extract element and text information
                element_id, text = self._extract_type_params(distyl_action)
                if element_id and text:
                    from browser_env.actions import create_type_action
                    return create_type_action(text=text, element_id=element_id)
            
            elif action_str.startswith("scroll"):
                direction = "down" if "down" in action_str else "up"
                from browser_env.actions import create_scroll_action
                return create_scroll_action(direction=direction)
            
            elif any(keyword in action_str for keyword in ["navigate", "goto", "go to"]):
                url = self._extract_url(distyl_action)
                if url:
                    from browser_env.actions import create_goto_url_action
                    return create_goto_url_action(url=url)
            
            elif any(keyword in action_str for keyword in ["press", "key"]):
                key_combo = self._extract_key_combination(distyl_action)
                if key_combo:
                    from browser_env.actions import create_key_press_action
                    return create_key_press_action(key_combo)
            
            elif any(keyword in action_str for keyword in ["done", "complete", "finished"]):
                return create_stop_action("Task completed")
            
            # Fallback to none action if parsing fails
            self.logger.warning(f"Could not parse action: {distyl_action}")
            return create_none_action()
            
        except Exception as e:
            self.logger.error(f"Error converting action '{distyl_action}': {str(e)}")
            return create_none_action()
    
    def _extract_element_id(self, action_str: str) -> str:
        """Extract element ID from action string"""
        import re
        match = re.search(r'\[(\d+)\]', action_str)
        return match.group(1) if match else ""
    
    def _extract_type_params(self, action_str: str) -> Tuple[str, str]:
        """Extract element ID and text from type action"""
        import re
        
        # Pattern: type [element_id] [text] or similar variations
        match = re.search(r'type.*?\[(\d+)\].*?\[([^\]]+)\]', action_str, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
        
        # Alternative pattern: type "text" in element_id
        match = re.search(r'type\s+["\']([^"\']+)["\'].*?(\d+)', action_str, re.IGNORECASE)
        if match:
            return match.group(2), match.group(1)
        
        return "", ""
    
    def _extract_url(self, action_str: str) -> str:
        """Extract URL from navigation action"""
        import re
        
        # Look for URLs in the action string
        url_match = re.search(r'https?://[^\s\]]+', action_str)
        if url_match:
            return url_match.group(0)
        
        # Look for quoted URLs
        quoted_match = re.search(r'["\']([^"\']*(?:http|www)[^"\']*)["\']', action_str)
        if quoted_match:
            return quoted_match.group(1)
        
        return ""
    
    def _extract_key_combination(self, action_str: str) -> str:
        """Extract key combination from press action"""
        import re
        
        # Look for key combinations in brackets or quotes
        key_match = re.search(r'(?:press|key).*?["\']([^"\']+)["\']', action_str, re.IGNORECASE)
        if key_match:
            return key_match.group(1)
        
        key_match = re.search(r'(?:press|key).*?\[([^\]]+)\]', action_str, re.IGNORECASE)
        if key_match:
            return key_match.group(1)
        
        return ""