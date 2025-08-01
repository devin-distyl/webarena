"""
WebReflectionAgent: Reflection and error correction for web actions

Provides feedback on action success/failure and suggests alternatives.
"""

from typing import Dict, Any, Optional
from ..utils.logging import DistylLogger


class WebReflectionAgent:
    """
    Provides feedback on action success/failure and suggests alternatives
    """
    
    def __init__(self, engine_params: Dict[str, Any]):
        self.engine_params = engine_params
        self.logger = DistylLogger("WebReflectionAgent")
        
        # Reflection history
        self.reflection_history = []
        self.common_failures = {}
    
    def suggest_alternative(self, subtask: Dict[str, Any], failed_action: str, 
                          context: Dict[str, Any], failure_reason: str = "") -> str:
        """Suggest alternative action when original action fails"""
        
        self.logger.info(f"Reflecting on failed action: {failed_action}")
        
        # Record failure
        self._record_failure(subtask, failed_action, failure_reason)
        
        # Analyze failure type and suggest alternative
        alternative = self._analyze_and_suggest(subtask, failed_action, context, failure_reason)
        
        self.logger.info(f"Suggested alternative: {alternative}")
        return alternative
    
    def _record_failure(self, subtask: Dict[str, Any], failed_action: str, failure_reason: str):
        """Record failure for learning"""
        
        failure_record = {
            "subtask": subtask.get("description", ""),
            "failed_action": failed_action,
            "failure_reason": failure_reason,
            "subtask_type": subtask.get("type", "general"),
            "site_type": subtask.get("site_type", "general")
        }
        
        self.reflection_history.append(failure_record)
        
        # Track common failure patterns
        failure_key = f"{subtask.get('type', 'general')}_{failure_reason}"
        self.common_failures[failure_key] = self.common_failures.get(failure_key, 0) + 1
    
    def _analyze_and_suggest(self, subtask: Dict[str, Any], failed_action: str, 
                           context: Dict[str, Any], failure_reason: str) -> str:
        """Analyze failure and suggest alternative action"""
        
        subtask_description = subtask.get("description", "").lower()
        failed_action_lower = failed_action.lower()
        
        # Failure-specific suggestions
        if "element not found" in failure_reason.lower():
            return self._suggest_for_element_not_found(subtask, failed_action, context)
        
        elif "validation failed" in failure_reason.lower():
            return self._suggest_for_validation_failure(subtask, failed_action, context)
        
        elif "timeout" in failure_reason.lower():
            return self._suggest_for_timeout(subtask, failed_action, context)
        
        else:
            return self._suggest_generic_alternative(subtask, failed_action, context)
    
    def _suggest_for_element_not_found(self, subtask: Dict[str, Any], failed_action: str, 
                                     context: Dict[str, Any]) -> str:
        """Suggest alternatives when element is not found"""
        
        failed_action_lower = failed_action.lower()
        
        if "click" in failed_action_lower:
            # Try different element detection strategies
            if "button" in subtask.get("description", "").lower():
                return "click [auto_detect_submit_button]"  # Try submit button instead
            elif "link" in subtask.get("description", "").lower():
                return "click [auto_detect_link]"
            else:
                # Try scrolling to find element
                return "scroll [down]"
        
        elif "type" in failed_action_lower:
            # Try different input field detection
            if "search" in subtask.get("description", "").lower():
                return "click [auto_detect_search_field]"  # Click first, then type
            else:
                return "click [auto_detect_field]"
        
        else:
            # General fallback - scroll to see more content
            return "scroll [down]"
    
    def _suggest_for_validation_failure(self, subtask: Dict[str, Any], failed_action: str, 
                                      context: Dict[str, Any]) -> str:
        """Suggest alternatives when action validation fails"""
        
        # Try a more general approach
        subtask_type = subtask.get("type", "general")
        
        if subtask_type == "navigation":
            return "scroll [down]"
        elif subtask_type == "search":
            return "click [auto_detect_search_field]"
        elif subtask_type == "click":
            return "click [auto_detect_button]"
        elif subtask_type == "input":
            return "click [auto_detect_field]"
        else:
            return "scroll [down]"
    
    def _suggest_for_timeout(self, subtask: Dict[str, Any], failed_action: str, 
                           context: Dict[str, Any]) -> str:
        """Suggest alternatives when action times out"""
        
        # Wait for page to load, then retry simpler action
        return "scroll [down]"  # Simple action to wait for page load
    
    def _suggest_generic_alternative(self, subtask: Dict[str, Any], failed_action: str, 
                                   context: Dict[str, Any]) -> str:
        """Generic alternative suggestion"""
        
        subtask_description = subtask.get("description", "").lower()
        
        # Context-based suggestions
        if "navigate" in subtask_description:
            return "scroll [down]"
        elif "search" in subtask_description:
            return "click [auto_detect_search_field]"
        elif "admin" in subtask_description:
            return "click [auto_detect_admin_menu]"
        elif "review" in subtask_description:
            return "click [auto_detect_reviews_menu]"
        else:
            return "scroll [down]"
    
    def analyze_trajectory_patterns(self, trajectory_history: list) -> Dict[str, Any]:
        """Analyze patterns in trajectory for improvement suggestions"""
        
        if not trajectory_history:
            return {}
        
        analysis = {
            "common_failures": self.common_failures.copy(),
            "failure_count": len(self.reflection_history),
            "suggestions": []
        }
        
        # Analyze common failure patterns
        if self.common_failures:
            most_common_failure = max(self.common_failures.items(), key=lambda x: x[1])
            analysis["most_common_failure"] = most_common_failure[0]
            analysis["most_common_failure_count"] = most_common_failure[1]
            
            # Generate suggestions based on patterns
            if "element not found" in most_common_failure[0]:
                analysis["suggestions"].append("Consider improving element detection strategies")
            elif "validation" in most_common_failure[0]:
                analysis["suggestions"].append("Review action validation criteria")
        
        return analysis
    
    def get_success_patterns(self) -> Dict[str, Any]:
        """Identify successful patterns from reflection history"""
        
        # This would analyze successful vs failed actions to identify patterns
        # For now, return basic statistics
        
        total_reflections = len(self.reflection_history)
        
        if total_reflections == 0:
            return {"success_rate": 1.0, "total_reflections": 0}
        
        # Simple success pattern analysis
        patterns = {
            "total_reflections": total_reflections,
            "common_failure_types": list(self.common_failures.keys()),
            "reflection_insights": []
        }
        
        # Add insights based on failure patterns
        if "element not found" in str(self.common_failures):
            patterns["reflection_insights"].append(
                "Element detection needs improvement - consider using more robust selectors"
            )
        
        if "validation" in str(self.common_failures):
            patterns["reflection_insights"].append(
                "Action validation may be too strict - review validation criteria"
            )
        
        return patterns
    
    def should_replan(self, consecutive_failures: int) -> bool:
        """Determine if we should trigger replanning based on consecutive failures"""
        
        # Replan if we have multiple consecutive failures
        return consecutive_failures >= 3
    
    def generate_debugging_info(self, subtask: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate debugging information for failed subtasks"""
        
        debug_info = []
        
        # Subtask information
        debug_info.append(f"Subtask: {subtask.get('description', 'Unknown')}")
        debug_info.append(f"Type: {subtask.get('type', 'Unknown')}")
        debug_info.append(f"Site: {subtask.get('site_type', 'Unknown')}")
        
        # Context information
        obs = context.get("observation", {})
        debug_info.append(f"Current URL: {obs.get('url', 'Unknown')}")
        
        # Page analysis
        accessibility_tree = obs.get("accessibility_tree", obs.get("text", ""))
        if accessibility_tree:
            element_count = accessibility_tree.count("[")
            debug_info.append(f"Elements on page: {element_count}")
            
            # Check for common elements
            tree_lower = accessibility_tree.lower()
            if "button" in tree_lower:
                button_count = tree_lower.count("button")
                debug_info.append(f"Buttons found: {button_count}")
            if "link" in tree_lower:
                link_count = tree_lower.count("link")
                debug_info.append(f"Links found: {link_count}")
            if any(word in tree_lower for word in ["textbox", "input"]):
                debug_info.append("Input fields detected")
        
        # Recent failures
        if self.reflection_history:
            recent_failures = self.reflection_history[-3:]  # Last 3 failures
            debug_info.append("Recent failures:")
            for failure in recent_failures:
                debug_info.append(f"  - {failure['failed_action']}: {failure['failure_reason']}")
        
        return "\n".join(debug_info)