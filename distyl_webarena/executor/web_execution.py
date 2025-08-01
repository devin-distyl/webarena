"""
WebExecutor: Web browser action execution with reflection

Converts high-level subtasks into specific web actions using code generation and reflection.
"""

from typing import Dict, List, Any, Tuple
from ..utils.logging import DistylLogger
from ..actions.web_actions import WebActionCodeGenerator
from .reflection import WebReflectionAgent
from .action_validation import ActionValidator


class WebExecutor:
    """
    Converts high-level subtasks into specific actions using code generation and reflection
    """
    
    def __init__(self, engine_params: Dict[str, Any], grounder, memory, enable_reflection: bool = True):
        self.engine_params = engine_params
        self.grounder = grounder
        self.memory = memory
        self.enable_reflection = enable_reflection
        
        self.action_generator = WebActionCodeGenerator(grounder.element_detector if grounder else None)
        self.reflection_agent = WebReflectionAgent(engine_params) if enable_reflection else None
        self.validator = ActionValidator()
        self.logger = DistylLogger("WebExecutor")
        
        # Execution state
        self.current_subtask = None
        self.execution_attempts = 0
        self.max_attempts = 3
    
    def next_action(self, subtask: Dict[str, Any], context: Dict[str, Any], trajectory: List[Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Generate next web action for given subtask"""
        
        try:
            self.current_subtask = subtask
            subtask_description = subtask.get("description", "")
            
            self.logger.info(f"Executing subtask: {subtask_description}")
            
            # Retrieve similar subtask experience
            similar_experience = self._retrieve_subtask_experience(subtask_description)
            
            # Generate action with web context
            action_plan = self._generate_web_action_plan(subtask, context, similar_experience)
            
            if not action_plan:
                self.logger.warning("No action plan generated")
                return {"subtask_complete": False}, ["none"]
            
            # Ground action parameters
            grounded_action = self.grounder.resolve_action_parameters(action_plan, context["observation"])
            
            # Validate action
            is_valid, validation_reason = self.validator.validate_action(grounded_action, context)
            
            if not is_valid:
                self.logger.warning(f"Action validation failed: {validation_reason}")
                
                if self.enable_reflection and self.reflection_agent:
                    # Try reflection-based correction
                    alternative_action = self.reflection_agent.suggest_alternative(
                        subtask, grounded_action, context, validation_reason
                    )
                    return {"subtask_complete": False}, [alternative_action]
                else:
                    return {"subtask_complete": False}, ["none"]
            
            # Check if subtask should be marked complete
            subtask_complete = self._should_complete_subtask(subtask, grounded_action, context)
            
            return {
                "subtask_complete": subtask_complete,
                "action_valid": True
            }, [grounded_action]
            
        except Exception as e:
            self.logger.error(f"Error in action execution: {str(e)}")
            return {"subtask_complete": False, "error": str(e)}, ["none"]
    
    def reset(self):
        """Reset executor state"""
        self.current_subtask = None
        self.execution_attempts = 0
    
    def _retrieve_subtask_experience(self, subtask_description: str) -> str:
        """Retrieve similar subtask experiences from memory"""
        
        try:
            if hasattr(self.memory, 'retrieve_subtask_experience'):
                return self.memory.retrieve_subtask_experience(subtask_description)
            else:
                return ""
        except Exception as e:
            self.logger.warning(f"Could not retrieve subtask experience: {e}")
            return ""
    
    def _generate_web_action_plan(self, subtask: Dict[str, Any], context: Dict[str, Any], experience: str) -> str:
        """Generate detailed action plan for subtask"""
        
        subtask_description = subtask.get("description", "")
        subtask_type = subtask.get("type", "general")
        site_type = subtask.get("site_type", "general")
        
        # Use action generator to create web action
        action_plan = self.action_generator.generate_action_code(subtask_description, context)
        
        if action_plan == "none":
            # Try alternative generation strategies
            action_plan = self._generate_fallback_action(subtask_description, context, site_type)
        
        self.logger.debug(f"Generated action plan: {action_plan}")
        return action_plan
    
    def _generate_fallback_action(self, description: str, context: Dict[str, Any], site_type: str) -> str:
        """Generate fallback action when primary generation fails"""
        
        description_lower = description.lower()
        
        # Simple rule-based fallbacks based on description keywords
        if "navigate" in description_lower:
            if "search" in description_lower:
                return "click [auto_detect_search_field]"
            elif "admin" in description_lower:
                return "click [auto_detect_admin_menu]"
            else:
                return "scroll [down]"  # General navigation
        
        elif "search" in description_lower:
            query = self._extract_search_query(description)
            return f"type [auto_detect_search_field] [{query}] [1]"
        
        elif "click" in description_lower or "access" in description_lower:
            if "button" in description_lower:
                return "click [auto_detect_button]"
            elif "link" in description_lower:
                return "click [auto_detect_link]"
            else:
                return "click [auto_detect_element]"
        
        elif "extract" in description_lower or "get" in description_lower:
            return "scroll [down]"  # Scroll to view content
        
        elif "verify" in description_lower:
            return "scroll [down]"  # Scroll to verify content
        
        else:
            # Ultimate fallback
            return "scroll [down]"
    
    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description"""
        
        import re
        
        # Look for quoted text
        quoted_match = re.search(r'["\']([^"\']+)["\']', description)
        if quoted_match:
            return quoted_match.group(1)
        
        # Look for "search for X" patterns
        search_patterns = [
            r'search for (.+?)(?:\.|$|,)',
            r'find (.+?)(?:\.|$|,)',
            r'look for (.+?)(?:\.|$|,)'
        ]
        
        for pattern in search_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Extract meaningful words (exclude common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = description.lower().split()
        meaningful_words = [word for word in words if word not in common_words and len(word) > 2]
        
        return ' '.join(meaningful_words[:3])  # Take first 3 meaningful words
    
    def _should_complete_subtask(self, subtask: Dict[str, Any], action: str, context: Dict[str, Any]) -> bool:
        """Determine if subtask should be marked as complete"""
        
        subtask_type = subtask.get("type", "general")
        action_lower = action.lower()
        
        # Completion criteria based on subtask type
        if subtask_type == "navigation":
            # Navigation subtasks complete after one successful action
            return action_lower != "none"
        
        elif subtask_type == "search":
            # Search subtasks complete after entering search query
            return "type" in action_lower and "search" in action_lower
        
        elif subtask_type == "click":
            # Click subtasks complete after one click
            return "click" in action_lower
        
        elif subtask_type == "input":
            # Input subtasks complete after typing
            return "type" in action_lower
        
        elif subtask_type == "extraction":
            # Extraction subtasks might need multiple actions
            return False  # Continue until explicit completion
        
        elif subtask_type == "verification":
            # Verification subtasks complete after one action
            return action_lower != "none"
        
        else:
            # General subtasks complete after one successful action
            return action_lower != "none"
    
    def update_from_feedback(self, subtask: Dict[str, Any], action: str, success: bool, context: Dict[str, Any]):
        """Update executor based on action feedback"""
        
        if self.grounder and hasattr(self.grounder, 'update_grounding_from_feedback'):
            # Update grounder with feedback
            description = subtask.get("description", "")
            element_id = self._extract_element_id_from_action(action)
            
            if element_id:
                self.grounder.update_grounding_from_feedback(
                    description, element_id, success, context["observation"]
                )
        
        # Log performance
        status = "SUCCESS" if success else "FAILED"
        self.logger.log_action(action, success=success)
    
    def _extract_element_id_from_action(self, action: str) -> str:
        """Extract element ID from action string"""
        import re
        match = re.search(r'\[(\d+)\]', action)
        return match.group(1) if match else ""