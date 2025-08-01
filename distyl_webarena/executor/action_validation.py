"""
ActionValidator: Validation for web actions

Validates if generated actions are feasible given current page state.
"""

import re
from typing import Dict, Any, Tuple
from ..utils.logging import DistylLogger


class ActionValidator:
    """
    Validates if generated actions are feasible given current page state
    """
    
    def __init__(self):
        self.logger = DistylLogger("ActionValidator")
    
    def validate_action(self, action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate if action can be executed
        Returns (is_valid, reason)
        """
        
        if not action or action.strip().lower() in ["none", ""]:
            return False, "Action is empty or 'none'"
        
        action_lower = action.strip().lower()
        
        try:
            # Validate based on action type
            if action_lower.startswith("click"):
                return self._validate_click_action(action, context)
            
            elif action_lower.startswith("type"):
                return self._validate_type_action(action, context)
            
            elif action_lower.startswith("hover"):
                return self._validate_hover_action(action, context)
            
            elif action_lower.startswith("scroll"):
                return self._validate_scroll_action(action, context)
            
            elif action_lower.startswith("press"):
                return self._validate_press_action(action, context)
            
            elif action_lower.startswith("goto"):
                return self._validate_goto_action(action, context)
            
            elif action_lower.startswith("stop"):
                return True, "Stop action is always valid"
            
            elif action_lower in ["new_tab", "close_tab", "go_back", "go_forward"]:
                return True, "Navigation actions are generally valid"
            
            else:
                return False, f"Unknown action type: {action}"
                
        except Exception as e:
            self.logger.error(f"Error validating action '{action}': {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_click_action(self, action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate click actions"""
        
        element_id = self._extract_element_id(action)
        
        if not element_id:
            return False, "No element ID found in click action"
        
        # Check if it's an auto-detect element (these are handled by grounder)
        if element_id.startswith("auto_detect_"):
            return True, "Auto-detect element will be resolved by grounder"
        
        # Check if element exists on page
        if not self._element_exists(element_id, context):
            return False, f"Element {element_id} not found on page"
        
        # Check if element is clickable
        if not self._element_is_clickable(element_id, context):
            return False, f"Element {element_id} may not be clickable"
        
        return True, "Click action is valid"
    
    def _validate_type_action(self, action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate type actions"""
        
        element_id = self._extract_element_id(action)
        text_to_type = self._extract_text_from_action(action)
        
        if not element_id:
            return False, "No element ID found in type action"
        
        if not text_to_type:
            return False, "No text to type found in action"
        
        # Check if it's an auto-detect element
        if element_id.startswith("auto_detect_"):
            return True, "Auto-detect element will be resolved by grounder"
        
        # Check if element exists
        if not self._element_exists(element_id, context):
            return False, f"Element {element_id} not found on page"
        
        # Check if element is an input field
        if not self._element_is_input(element_id, context):
            return False, f"Element {element_id} is not an input field"
        
        return True, "Type action is valid"
    
    def _validate_hover_action(self, action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate hover actions"""
        
        element_id = self._extract_element_id(action)
        
        if not element_id:
            return False, "No element ID found in hover action"
        
        if element_id.startswith("auto_detect_"):
            return True, "Auto-detect element will be resolved by grounder"
        
        if not self._element_exists(element_id, context):
            return False, f"Element {element_id} not found on page"
        
        return True, "Hover action is valid"
    
    def _validate_scroll_action(self, action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate scroll actions"""
        
        direction = self._extract_scroll_direction(action)
        
        if direction not in ["up", "down"]:
            return False, f"Invalid scroll direction: {direction}"
        
        return True, "Scroll action is valid"
    
    def _validate_press_action(self, action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate key press actions"""
        
        key_combo = self._extract_key_combination(action)
        
        if not key_combo:
            return False, "No key combination found in press action"
        
        # Basic validation of key combinations
        valid_keys = [
            "enter", "tab", "escape", "space", "ctrl+c", "ctrl+v", "ctrl+a", "ctrl+z"
        ]
        
        if key_combo.lower() not in valid_keys and not self._is_valid_key_combo(key_combo):
            return False, f"Invalid key combination: {key_combo}"
        
        return True, "Press action is valid"
    
    def _validate_goto_action(self, action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate goto URL actions"""
        
        url = self._extract_url(action)
        
        if not url:
            return False, "No URL found in goto action"
        
        if not self._is_valid_url(url):
            return False, f"Invalid URL: {url}"
        
        return True, "Goto action is valid"
    
    def _extract_element_id(self, action: str) -> str:
        """Extract element ID from action"""
        match = re.search(r'\[([^\]]+)\]', action)
        return match.group(1) if match else ""
    
    def _extract_text_from_action(self, action: str) -> str:
        """Extract text to type from action"""
        # Pattern: type [element_id] [text] [enter_flag]
        matches = re.findall(r'\[([^\]]*)\]', action)
        return matches[1] if len(matches) > 1 else ""
    
    def _extract_scroll_direction(self, action: str) -> str:
        """Extract scroll direction from action"""
        match = re.search(r'\[([^\]]+)\]', action)
        return match.group(1).lower() if match else ""
    
    def _extract_key_combination(self, action: str) -> str:
        """Extract key combination from action"""
        match = re.search(r'\[([^\]]+)\]', action)
        return match.group(1) if match else ""
    
    def _extract_url(self, action: str) -> str:
        """Extract URL from action"""
        match = re.search(r'\[([^\]]+)\]', action)
        return match.group(1) if match else ""
    
    def _element_exists(self, element_id: str, context: Dict[str, Any]) -> bool:
        """Check if element exists in accessibility tree"""
        
        obs = context.get("observation", {})
        tree = obs.get("accessibility_tree", obs.get("text", ""))
        
        return f"[{element_id}]" in tree
    
    def _element_is_clickable(self, element_id: str, context: Dict[str, Any]) -> bool:
        """Check if element is likely clickable"""
        
        obs = context.get("observation", {})
        tree = obs.get("accessibility_tree", obs.get("text", ""))
        
        # Look for clickable keywords near the element ID
        clickable_keywords = ["button", "link", "clickable", "tab", "menu"]
        
        for line in tree.split('\n'):
            if f"[{element_id}]" in line:
                line_lower = line.lower()
                return any(keyword in line_lower for keyword in clickable_keywords)
        
        return True  # Default to clickable if we can't determine
    
    def _element_is_input(self, element_id: str, context: Dict[str, Any]) -> bool:
        """Check if element is an input field"""
        
        obs = context.get("observation", {})
        tree = obs.get("accessibility_tree", obs.get("text", ""))
        
        # Look for input-related keywords near the element ID
        input_keywords = ["textbox", "input", "field", "textarea", "searchbox"]
        
        for line in tree.split('\n'):
            if f"[{element_id}]" in line:
                line_lower = line.lower()
                return any(keyword in line_lower for keyword in input_keywords)
        
        return False
    
    def _is_valid_key_combo(self, key_combo: str) -> bool:
        """Check if key combination is valid"""
        
        # Basic validation - check for common key patterns
        key_combo_lower = key_combo.lower()
        
        # Single keys
        single_keys = ["enter", "tab", "escape", "space", "backspace", "delete"]
        if key_combo_lower in single_keys:
            return True
        
        # Modifier combinations
        if "+" in key_combo_lower:
            parts = key_combo_lower.split("+")
            modifiers = ["ctrl", "alt", "shift", "meta"]
            
            # At least one modifier
            if any(part in modifiers for part in parts[:-1]):
                return True
        
        return False
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        
        if not url:
            return False
        
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return url_pattern.match(url) is not None
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        # This would track validation statistics over time
        # For now, return basic info
        
        return {
            "validator_active": True,
            "supported_actions": [
                "click", "type", "hover", "scroll", "press", "goto", "stop"
            ],
            "validation_features": [
                "element_existence", "element_type_checking", "url_validation", "key_combo_validation"
            ]
        }