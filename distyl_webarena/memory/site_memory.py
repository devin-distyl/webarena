"""
SiteMemoryPatterns: Site-specific memory patterns for web interactions
"""

from typing import Dict, List, Any
from ..utils.logging import DistylLogger


class SiteMemoryPatterns:
    """
    Site-specific memory patterns and interaction histories
    """
    
    def __init__(self):
        self.logger = DistylLogger("SiteMemoryPatterns")
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize site-specific memory patterns"""
        
        return {
            "shopping": {
                "common_workflows": [
                    "search -> filter -> select -> add_to_cart -> checkout",
                    "navigate -> browse -> compare -> purchase",
                    "admin -> reports -> analyze -> extract"
                ],
                "element_patterns": {
                    "search": ["searchbox", "query", "find"],
                    "cart": ["cart", "bag", "basket"],
                    "checkout": ["checkout", "buy", "purchase"]
                }
            },
            
            "social": {
                "common_workflows": [
                    "login -> navigate -> create_post -> publish",
                    "browse -> find_content -> vote -> comment",
                    "search -> select -> interact"
                ],
                "element_patterns": {
                    "post": ["submit", "create", "new post"],
                    "vote": ["upvote", "downvote", "score"],
                    "comment": ["reply", "comment", "respond"]
                }
            },
            
            "development": {
                "common_workflows": [
                    "navigate -> repository -> browse -> edit",
                    "issues -> create -> describe -> submit",
                    "code -> commit -> push -> merge"
                ],
                "element_patterns": {
                    "repository": ["repo", "project", "code"],
                    "issues": ["issue", "bug", "feature"],
                    "commit": ["commit", "save", "push"]
                }
            },
            
            "knowledge": {
                "common_workflows": [
                    "search -> select_article -> read -> extract",
                    "navigate -> browse -> follow_links -> gather_info"
                ],
                "element_patterns": {
                    "search": ["search", "find", "lookup"],
                    "article": ["article", "page", "content"],
                    "navigation": ["link", "section", "category"]
                }
            }
        }
    
    def get_site_patterns(self, site_type: str) -> Dict[str, Any]:
        """Get patterns for specific site type"""
        return self.patterns.get(site_type, {})
    
    def add_successful_pattern(self, site_type: str, workflow: str, success_rate: float):
        """Add a successful interaction pattern"""
        
        if site_type not in self.patterns:
            self.patterns[site_type] = {"common_workflows": [], "element_patterns": {}}
        
        if workflow not in self.patterns[site_type]["common_workflows"]:
            self.patterns[site_type]["common_workflows"].append(workflow)
            self.logger.info(f"Added successful pattern for {site_type}: {workflow}")
    
    def get_element_suggestions(self, site_type: str, action_type: str) -> List[str]:
        """Get element suggestions for site and action type"""
        
        site_patterns = self.patterns.get(site_type, {})
        element_patterns = site_patterns.get("element_patterns", {})
        
        return element_patterns.get(action_type, [])