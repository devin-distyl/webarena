"""
WebArena Interface Adapter

Provides additional compatibility layers and utilities for WebArena integration.
"""

from typing import Any, Dict, List
from browser_env import Trajectory
from browser_env.actions import Action


class WebArenaInterfaceAdapter:
    """
    Adapter to ensure full compatibility with WebArena's interface expectations
    """
    
    @staticmethod
    def validate_trajectory(trajectory: Trajectory) -> bool:
        """Validate that trajectory is in expected WebArena format"""
        if not trajectory:
            return True
        
        # Check alternating state-action pattern
        for i, item in enumerate(trajectory):
            if i % 2 == 0:  # Even indices should be states
                if not isinstance(item, dict) or "observation" not in item:
                    return False
            else:  # Odd indices should be actions
                if not isinstance(item, dict) or "action_type" not in item:
                    return False
        
        return True
    
    @staticmethod
    def validate_action(action: Action) -> bool:
        """Validate that action is in expected WebArena format"""
        required_fields = ["action_type", "coords", "element_role", "element_name", 
                          "text", "page_number", "url", "nth", "element_id", 
                          "direction", "key_comb", "pw_code", "answer", "raw_prediction"]
        
        return all(field in action for field in required_fields)
    
    @staticmethod
    def extract_observation_metadata(observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from WebArena observation"""
        return {
            "url": observation.get("url", ""),
            "page_type": WebArenaInterfaceAdapter._infer_page_type(observation.get("text", "")),
            "has_forms": "form" in observation.get("text", "").lower(),
            "has_buttons": "button" in observation.get("text", "").lower(),
            "element_count": observation.get("text", "").count("[") if observation.get("text") else 0
        }
    
    @staticmethod
    def _infer_page_type(accessibility_tree: str) -> str:
        """Infer page type from accessibility tree content"""
        tree_lower = accessibility_tree.lower()
        
        if any(word in tree_lower for word in ["login", "sign in", "password"]):
            return "login_page"
        elif any(word in tree_lower for word in ["search", "results", "found"]):
            return "search_page"
        elif any(word in tree_lower for word in ["cart", "checkout", "price", "$"]):
            return "shopping_page"
        elif any(word in tree_lower for word in ["post", "comment", "vote", "upvote"]):
            return "social_page"
        elif any(word in tree_lower for word in ["repository", "commit", "issue", "pull"]):
            return "development_page"
        else:
            return "general_page"