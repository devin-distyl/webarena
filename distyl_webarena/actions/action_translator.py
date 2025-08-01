"""
DesktopToWebActionTranslator: Translation from desktop actions to web actions

Translates Distyl's desktop automation actions to WebArena browser actions
"""

import re
from typing import Dict, Any, List
from ..utils.logging import DistylLogger


class DesktopToWebActionTranslator:
    """
    Translates Distyl's desktop automation actions to WebArena browser actions
    """
    
    def __init__(self):
        self.logger = DistylLogger("DesktopToWebActionTranslator")
    
    @staticmethod
    def translate_action(desktop_action: str, context: Dict[str, Any]) -> str:
        """
        Convert desktop action to web action
        
        Desktop Actions → Web Actions:
        click(x, y) → click [element_id]
        type_text("hello") → type [element_id] [hello] [1]
        hotkey("Ctrl+v") → press [Ctrl+v]
        scroll(direction) → scroll [up|down]
        """
        
        translator = DesktopToWebActionTranslator()
        
        action_patterns = {
            # Click coordinate → Click element
            r'click\((\d+),\s*(\d+)\)': lambda m: translator._coordinate_to_element_click(
                int(m.group(1)), int(m.group(2)), context
            ),
            
            # Type text → Type in field  
            r'type_text\(["\'](.+?)["\']\)': lambda m: f'type [auto_detect_field] [{m.group(1)}] [1]',
            
            # Hotkey → Press key combination
            r'hotkey\(["\'](.+?)["\']\)': lambda m: f'press [{m.group(1)}]',
            
            # Scroll → Scroll page
            r'scroll\((["\']?)(\w+)\1\)': lambda m: f'scroll [{m.group(2)}]',
            
            # Navigate → Goto URL
            r'navigate\(["\'](.+?)["\']\)': lambda m: f'goto [{m.group(1)}]',
            
            # Wait → No direct equivalent (handled by environment)
            r'wait\([\d.]+\)': lambda m: 'none',
            
            # Done → Stop with answer
            r'done\(\)': lambda m: 'stop []',
            r'done\(["\'](.+?)["\']\)': lambda m: f'stop [{m.group(1)}]',
            
            # Fail → Stop with failure
            r'fail\(["\'](.+?)["\']\)': lambda m: f'stop [Failed: {m.group(1)}]',
            r'fail\(\)': lambda m: 'stop [Task failed]'
        }
        
        desktop_action = desktop_action.strip()
        
        for pattern, replacement in action_patterns.items():
            match = re.match(pattern, desktop_action)
            if match:
                try:
                    if callable(replacement):
                        return replacement(match)
                    else:
                        return replacement
                except Exception as e:
                    translator.logger.error(f"Error translating action '{desktop_action}': {e}")
                    return "none"
        
        # If no pattern matches, try to interpret as natural language
        return translator._natural_language_to_web_action(desktop_action, context)
    
    def _coordinate_to_element_click(self, x: int, y: int, context: Dict[str, Any]) -> str:
        """
        Convert click coordinates to element click by finding nearest element
        This requires the accessibility tree context
        """
        # In desktop automation, we have exact coordinates
        # In web automation, we need element IDs
        # Use accessibility tree to find clickable element near coordinates
        
        accessibility_tree = context.get("observation", {}).get("accessibility_tree", "")
        
        if not accessibility_tree:
            self.logger.warning("No accessibility tree available for coordinate translation")
            return "click [auto_detect_button]"
        
        # Simple heuristic: look for clickable elements in tree
        clickable_elements = self._extract_clickable_elements(accessibility_tree)
        
        if clickable_elements:
            # For now, return first clickable element
            # In full implementation, would use spatial reasoning to find closest
            return f"click [{clickable_elements[0]}]"
        
        return "click [auto_detect_button]"
    
    def _extract_clickable_elements(self, accessibility_tree: str) -> List[str]:
        """Extract element IDs of clickable elements from accessibility tree"""
        
        # Find button, link, and clickable elements
        clickable_patterns = [
            r'\[(\d+)\]\s+button',
            r'\[(\d+)\]\s+link', 
            r'\[(\d+)\]\s+clickable',
            r'\[(\d+)\]\s+.*click.*'
        ]
        
        elements = []
        for pattern in clickable_patterns:
            matches = re.findall(pattern, accessibility_tree, re.IGNORECASE)
            elements.extend(matches)
        
        return list(set(elements))  # Remove duplicates
    
    def _natural_language_to_web_action(self, description: str, context: Dict[str, Any]) -> str:
        """Convert natural language description to web action"""
        
        description_lower = description.lower()
        
        # Common action patterns
        if "click" in description_lower:
            if "button" in description_lower:
                return "click [auto_detect_button]"
            elif "link" in description_lower:
                return "click [auto_detect_link]"
            else:
                return "click [auto_detect_element]"
        
        elif any(word in description_lower for word in ["type", "enter", "input"]):
            # Extract text to type
            text = self._extract_text_from_description(description)
            return f"type [auto_detect_field] [{text}] [1]"
        
        elif "scroll" in description_lower:
            direction = "down" if "down" in description_lower else "up"
            return f"scroll [{direction}]"
        
        elif any(word in description_lower for word in ["navigate", "go to", "visit"]):
            return "goto [auto_detect_url]"
        
        else:
            self.logger.warning(f"Could not translate description: {description}")
            return "none"
    
    def _extract_text_from_description(self, description: str) -> str:
        """Extract text content from natural language description"""
        
        # Look for quoted text
        quoted_match = re.search(r'["\']([^"\']+)["\']', description)
        if quoted_match:
            return quoted_match.group(1)
        
        # Look for text after common keywords
        keywords = ["type", "enter", "input", "write"]
        for keyword in keywords:
            pattern = rf'{keyword}\s+(.+?)(?:\s+(?:in|into|on)|$)'
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""


class ActionValidationHelper:
    """
    Helper class for validating translated actions
    """
    
    @staticmethod
    def validate_web_action(action: str, context: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate if translated web action is feasible
        Returns (is_valid, reason)
        """
        
        if not action or action.strip().lower() == "none":
            return False, "Action is 'none' or empty"
        
        action_lower = action.strip().lower()
        
        if action_lower.startswith("click"):
            element_id = ActionValidationHelper._extract_element_id(action)
            if element_id and not ActionValidationHelper._element_exists(element_id, context):
                return False, f"Element {element_id} not found on page"
        
        elif action_lower.startswith("type"):
            element_id = ActionValidationHelper._extract_element_id(action)
            if element_id and not ActionValidationHelper._element_is_input(element_id, context):
                return False, f"Element {element_id} is not an input field"
        
        elif action_lower.startswith("goto"):
            url = ActionValidationHelper._extract_url(action)
            if not ActionValidationHelper._is_valid_url(url):
                return False, f"Invalid URL: {url}"
        
        return True, "Action is valid"
    
    @staticmethod
    def _extract_element_id(action: str) -> str:
        """Extract element ID from action"""
        match = re.search(r'\[(\d+)\]', action)
        return match.group(1) if match else ""
    
    @staticmethod
    def _extract_url(action: str) -> str:
        """Extract URL from action"""
        match = re.search(r'\[([^\]]*(?:http|www)[^\]]*)\]', action)
        return match.group(1) if match else ""
    
    @staticmethod
    def _element_exists(element_id: str, context: Dict[str, Any]) -> bool:
        """Check if element exists in accessibility tree"""
        tree = context.get("observation", {}).get("accessibility_tree", "")
        return f"[{element_id}]" in tree
    
    @staticmethod
    def _element_is_input(element_id: str, context: Dict[str, Any]) -> bool:
        """Check if element is an input field"""
        tree = context.get("observation", {}).get("accessibility_tree", "")
        
        # Look for input-related keywords near the element ID
        input_keywords = ["textbox", "input", "field", "textarea"]
        
        for line in tree.split('\n'):
            if f"[{element_id}]" in line:
                return any(keyword in line.lower() for keyword in input_keywords)
        
        return False
    
    @staticmethod
    def _is_valid_url(url: str) -> bool:
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