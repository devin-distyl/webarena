"""
ElementAutoDetector: Smart element detection for web automation

Automatically detects element IDs based on semantic descriptions
Replaces the need for manual element ID specification
"""

from typing import Dict, List, Any
from ..utils.logging import DistylLogger


class ElementAutoDetector:
    """
    Automatically detects element IDs based on semantic descriptions
    Replaces the need for manual element ID specification
    """
    
    def __init__(self, grounder):
        self.grounder = grounder
        self.element_cache = {}
        self.logger = DistylLogger("ElementAutoDetector")
    
    def resolve_auto_detect_element(self, element_type: str, context: Dict[str, Any]) -> str:
        """
        Resolve auto_detect_* element references to actual element IDs
        
        auto_detect_username_field → actual element ID for username input
        auto_detect_search_button → actual element ID for search button
        """
        
        cache_key = f"{element_type}_{hash(str(context.get('observation', {})))}"
        if cache_key in self.element_cache:
            return self.element_cache[cache_key]
        
        # Remove auto_detect_ prefix
        clean_element_type = element_type.replace("auto_detect_", "")
        
        # Element type patterns
        element_patterns = {
            "username_field": ["username", "email", "login", "user", "account"],
            "password_field": ["password", "pass", "pwd"],
            "login_button": ["login", "sign in", "submit", "enter"],
            "search_field": ["search", "query", "find"],
            "search_button": ["search", "go", "find", "submit"],
            "submit_button": ["submit", "send", "save", "confirm"],
            "add_to_cart_button": ["add to cart", "add", "cart"],
            "checkout_button": ["checkout", "buy", "purchase"],
            "create_post_button": ["create", "new post", "submit"],
            "title_field": ["title", "subject", "name"],
            "content_field": ["content", "body", "text", "message"],
            "comment_field": ["comment", "reply", "message"],
            "upload_button": ["upload", "browse", "choose file"],
            "save_button": ["save", "update", "apply"],
            "cancel_button": ["cancel", "close", "dismiss"],
            "delete_button": ["delete", "remove", "trash"],
            "edit_button": ["edit", "modify", "change"],
            "back_button": ["back", "previous", "return"],
            "next_button": ["next", "continue", "forward"],
            "home_link": ["home", "homepage", "main"],
            "profile_link": ["profile", "account", "user"],
            "settings_link": ["settings", "preferences", "config"],
            "logout_link": ["logout", "sign out", "exit"],
            "reports_link": ["reports", "report"],
            "dashboard_link": ["dashboard", "home"],
            "sales_link": ["sales", "orders"],
            "sales_reports_link": ["sales reports", "sales report", "sales", "reports"],
            "catalog_link": ["catalog", "products"],
            "customers_link": ["customers", "users"],
            "marketing_link": ["marketing", "campaigns"],
            "content_link": ["content", "cms"],
            "stores_link": ["stores", "store"],
            "system_link": ["system", "admin"]
        }
        
        if clean_element_type in element_patterns:
            keywords = element_patterns[clean_element_type]
            element_id = self._find_element_by_keywords(keywords, context.get("observation", {}))
            self.element_cache[cache_key] = element_id
            
            if element_id:
                self.logger.debug(f"Resolved {element_type} to element {element_id}")
            else:
                self.logger.warning(f"Could not resolve {element_type}")
            
            return element_id
        
        self.logger.warning(f"Unknown auto-detect element type: {element_type}")
        return ""
    
    def _find_element_by_keywords(self, keywords: List[str], observation: Dict[str, Any]) -> str:
        """Find element that best matches the given keywords"""
        
        # Try each keyword in order of preference
        for keyword in keywords:
            element_id = self.grounder.ground_element_description(keyword, observation)
            if element_id:
                return element_id
        
        # If no exact matches, try partial matches
        accessibility_tree = observation.get("accessibility_tree", observation.get("text", ""))
        
        for keyword in keywords:
            element_id = self._find_partial_match(keyword, accessibility_tree)
            if element_id:
                return element_id
        
        return ""
    
    def _find_partial_match(self, keyword: str, accessibility_tree: str) -> str:
        """Find element with partial keyword match"""
        import re
        
        lines = accessibility_tree.split('\n')
        keyword_lower = keyword.lower()
        
        for line in lines:
            if keyword_lower in line.lower():
                # Extract element ID from line
                match = re.search(r'\[(\d+)\]', line)
                if match:
                    return match.group(1)
        
        return ""
    
    def get_supported_auto_detect_types(self) -> List[str]:
        """Get list of supported auto-detect element types"""
        return [
            "auto_detect_username_field",
            "auto_detect_password_field", 
            "auto_detect_login_button",
            "auto_detect_search_field",
            "auto_detect_search_button",
            "auto_detect_submit_button",
            "auto_detect_add_to_cart_button",
            "auto_detect_checkout_button",
            "auto_detect_create_post_button",
            "auto_detect_title_field",
            "auto_detect_content_field",
            "auto_detect_comment_field",
            "auto_detect_upload_button",
            "auto_detect_save_button",
            "auto_detect_cancel_button",
            "auto_detect_delete_button",
            "auto_detect_edit_button",
            "auto_detect_back_button",
            "auto_detect_next_button",
            "auto_detect_home_link",
            "auto_detect_profile_link",
            "auto_detect_settings_link",
            "auto_detect_logout_link"
        ]