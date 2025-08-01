"""
WebActionSystem: Web browser action system for Distyl-WebArena

Replaces Distyl's PyAutoGUI-based action system with WebArena browser actions
Provides unified interface for web interaction patterns
"""

from typing import List, Dict, Any, Optional
from ..utils.logging import DistylLogger


class WebActionSystem:
    """
    Replaces Distyl's PyAutoGUI-based action system with WebArena browser actions
    Provides unified interface for web interaction patterns
    """
    
    def __init__(self, action_set_tag: str = "id_accessibility_tree"):
        self.action_set_tag = action_set_tag
        self.supported_actions = [
            "click", "type", "hover", "scroll", "press", "goto", 
            "new_tab", "close_tab", "go_back", "go_forward", "stop"
        ]
        self.logger = DistylLogger("WebActionSystem")
    
    def generate_web_action(self, action_type: str, **kwargs) -> str:
        """Generate web action string based on type and parameters"""
        
        try:
            if action_type == "click":
                element_id = kwargs.get("element_id", "")
                return f"click [{element_id}]" if element_id else "click [auto_detect_button]"
            
            elif action_type == "type":
                element_id = kwargs.get("element_id", "")
                text = kwargs.get("text", "")
                press_enter = kwargs.get("press_enter", True)
                enter_flag = 1 if press_enter else 0
                return f"type [{element_id}] [{text}] [{enter_flag}]"
            
            elif action_type == "hover":
                element_id = kwargs.get("element_id", "")
                return f"hover [{element_id}]" if element_id else "hover [auto_detect_element]"
            
            elif action_type == "scroll":
                direction = kwargs.get("direction", "down")
                return f"scroll [{direction}]"
            
            elif action_type == "press":
                key_combo = kwargs.get("key_combo", "")
                return f"press [{key_combo}]"
            
            elif action_type == "goto":
                url = kwargs.get("url", "")
                return f"goto [{url}]"
            
            elif action_type == "new_tab":
                return "new_tab"
            
            elif action_type == "close_tab":
                return "close_tab"
            
            elif action_type == "go_back":
                return "go_back"
            
            elif action_type == "go_forward":
                return "go_forward"
            
            elif action_type == "stop":
                answer = kwargs.get("answer", "")
                return f"stop [{answer}]"
            
            else:
                self.logger.warning(f"Unknown action type: {action_type}")
                return "none"
                
        except Exception as e:
            self.logger.error(f"Error generating action {action_type}: {e}")
            return "none"
    
    def validate_action(self, action_str: str) -> bool:
        """Validate if action string is properly formatted"""
        
        action_str = action_str.strip().lower()
        
        # Check if action starts with supported action type
        for action_type in self.supported_actions:
            if action_str.startswith(action_type):
                return True
        
        # Check for special cases
        if action_str in ["none", "new_tab", "close_tab", "go_back", "go_forward"]:
            return True
        
        return False
    
    def parse_action_components(self, action_str: str) -> Dict[str, Any]:
        """Parse action string into components"""
        
        import re
        
        components = {
            "action_type": "",
            "parameters": [],
            "element_id": "",
            "text": "",
            "direction": "",
            "url": "",
            "key_combo": "",
            "answer": ""
        }
        
        # Extract action type (first word)
        action_parts = action_str.strip().split()
        if action_parts:
            components["action_type"] = action_parts[0].lower()
        
        # Extract parameters in brackets
        parameters = re.findall(r'\[([^\]]*)\]', action_str)
        components["parameters"] = parameters
        
        # Parse specific action types
        action_type = components["action_type"]
        
        if action_type == "click" and parameters:
            components["element_id"] = parameters[0]
        
        elif action_type == "type" and len(parameters) >= 2:
            components["element_id"] = parameters[0]
            components["text"] = parameters[1]
            if len(parameters) > 2:
                components["press_enter"] = parameters[2] == "1"
        
        elif action_type == "hover" and parameters:
            components["element_id"] = parameters[0]
        
        elif action_type == "scroll" and parameters:
            components["direction"] = parameters[0]
        
        elif action_type == "press" and parameters:
            components["key_combo"] = parameters[0]
        
        elif action_type == "goto" and parameters:
            components["url"] = parameters[0]
        
        elif action_type == "stop" and parameters:
            components["answer"] = parameters[0]
        
        return components


class WebActionCodeGenerator:
    """
    Generates executable web actions from high-level descriptions
    Replaces Distyl's PyAutoGUI code generation
    """
    
    def __init__(self, element_detector=None):
        self.element_detector = element_detector
        self.logger = DistylLogger("WebActionCodeGenerator")
    
    def generate_action_code(self, description: str, context: Dict[str, Any]) -> str:
        """
        Generate WebArena action code from natural language description
        
        Input: "click the login button"
        Output: "click [123]" (where 123 is the actual element ID)
        """
        
        description = description.lower().strip()
        
        try:
            # Action pattern matching
            if "click" in description:
                element_type = self._extract_element_type(description)
                if self.element_detector:
                    element_id = self.element_detector.resolve_auto_detect_element(f"auto_detect_{element_type}", context)
                    return f"click [{element_id}]" if element_id else f"click [auto_detect_{element_type}]"
                else:
                    return f"click [auto_detect_{element_type}]"
            
            elif "type" in description or "enter" in description:
                text = self._extract_text_content(description)
                element_type = self._extract_element_type(description)
                press_enter = 1 if "enter" in description or "submit" in description else 0
                
                if self.element_detector:
                    element_id = self.element_detector.resolve_auto_detect_element(f"auto_detect_{element_type}", context)
                    if element_id:
                        return f"type [{element_id}] [{text}] [{press_enter}]"
                
                return f"type [auto_detect_{element_type}] [{text}] [{press_enter}]"
            
            elif "scroll" in description:
                direction = "down" if "down" in description else "up"
                return f"scroll [{direction}]"
            
            elif "navigate" in description or "go to" in description:
                url = self._extract_url(description)
                return f"goto [{url}]" if url else "goto [auto_detect_url]"
            
            elif "press" in description:
                keys = self._extract_key_combination(description)
                return f"press [{keys}]" if keys else "press [Enter]"
            
            elif any(word in description for word in ["done", "complete", "finished", "stop"]):
                return "stop [Task completed]"
            
            else:
                self.logger.warning(f"Could not parse description: {description}")
                return "none"
                
        except Exception as e:
            self.logger.error(f"Error generating action code for '{description}': {e}")
            return "none"
    
    def _extract_element_type(self, description: str) -> str:
        """Extract element type from description"""
        element_mappings = {
            "login button": "login_button",
            "sign in button": "login_button",
            "username": "username_field", 
            "password": "password_field",
            "search": "search_field",
            "search button": "search_button",
            "submit": "submit_button",
            "submit button": "submit_button",
            "add to cart": "add_to_cart_button",
            "cart": "add_to_cart_button",
            "checkout": "checkout_button",
            "buy": "checkout_button",
            "title": "title_field",
            "content": "content_field",
            "comment": "comment_field",
            "upload": "upload_button",
            "save": "save_button",
            "cancel": "cancel_button",
            "delete": "delete_button",
            "edit": "edit_button",
            "back": "back_button",
            "next": "next_button",
            "home": "home_link",
            "profile": "profile_link",
            "settings": "settings_link",
            "logout": "logout_link"
        }
        
        for phrase, element_type in element_mappings.items():
            if phrase in description:
                return element_type
        
        # Default fallbacks based on context
        if "button" in description:
            return "button"
        elif any(word in description for word in ["field", "input", "textbox"]):
            return "field"
        elif "link" in description:
            return "link"
        else:
            return "element"
    
    def _extract_text_content(self, description: str) -> str:
        """Extract text to type from description"""
        import re
        
        # Match quoted text
        quoted_match = re.search(r'["\']([^"\']+)["\']', description)
        if quoted_match:
            return quoted_match.group(1)
        
        # Match text after "type" or "enter"
        type_match = re.search(r'(?:type|enter)\s+(.+?)(?:\s+in|$)', description)
        if type_match:
            return type_match.group(1).strip()
        
        return ""
    
    def _extract_url(self, description: str) -> str:
        """Extract URL from navigation description"""
        import re
        
        # Look for URLs in the description
        url_match = re.search(r'https?://[^\s\]]+', description)
        if url_match:
            return url_match.group(0)
        
        # Look for quoted URLs
        quoted_match = re.search(r'["\']([^"\']*(?:http|www)[^"\']*)["\']', description)
        if quoted_match:
            return quoted_match.group(1)
        
        return ""
    
    def _extract_key_combination(self, description: str) -> str:
        """Extract key combination from press action"""
        import re
        
        # Common key combinations
        key_mappings = {
            "enter": "Enter",
            "return": "Enter", 
            "tab": "Tab",
            "escape": "Escape",
            "space": "Space",
            "ctrl+c": "Ctrl+c",
            "ctrl+v": "Ctrl+v",
            "ctrl+a": "Ctrl+a",
            "ctrl+z": "Ctrl+z"
        }
        
        description_lower = description.lower()
        
        for phrase, key_combo in key_mappings.items():
            if phrase in description_lower:
                return key_combo
        
        # Look for key combinations in brackets or quotes
        key_match = re.search(r'(?:press|key).*?["\']([^"\']+)["\']', description, re.IGNORECASE)
        if key_match:
            return key_match.group(1)
        
        key_match = re.search(r'(?:press|key).*?\[([^\]]+)\]', description, re.IGNORECASE)
        if key_match:
            return key_match.group(1)
        
        return "Enter"  # Default fallback