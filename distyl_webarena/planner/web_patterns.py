"""
WebInteractionPatterns: Common web interaction patterns

Contains reusable patterns for web automation across different sites.
"""

from typing import Dict, List, Any


class WebInteractionPatterns:
    """
    Common web interaction patterns for different sites and tasks
    """
    
    def __init__(self):
        self.patterns = self._load_patterns()
    
    def get_relevant_patterns(self, instruction: str, site_type: str) -> List[Dict[str, Any]]:
        """Get patterns relevant to the instruction and site type"""
        
        instruction_lower = instruction.lower()
        relevant = []
        
        # Get site-specific patterns
        site_patterns = self.patterns.get(site_type, {})
        
        # Match patterns based on keywords in instruction
        for pattern_name, pattern_data in site_patterns.items():
            keywords = pattern_data.get("keywords", [])
            if any(keyword in instruction_lower for keyword in keywords):
                relevant.append(pattern_data)
        
        # Add universal patterns
        universal_patterns = self.patterns.get("universal", {})
        for pattern_name, pattern_data in universal_patterns.items():
            keywords = pattern_data.get("keywords", [])
            if any(keyword in instruction_lower for keyword in keywords):
                relevant.append(pattern_data)
        
        return relevant
    
    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load all interaction patterns"""
        
        return {
            "universal": {
                "login_flow": {
                    "keywords": ["login", "sign in", "authenticate"],
                    "steps": [
                        "Navigate to login page",
                        "Enter username/email",
                        "Enter password", 
                        "Click login button",
                        "Verify successful login"
                    ],
                    "elements": ["username_field", "password_field", "login_button"]
                },
                
                "search_flow": {
                    "keywords": ["search", "find", "look for"],
                    "steps": [
                        "Locate search field",
                        "Enter search query",
                        "Submit search",
                        "Review results"
                    ],
                    "elements": ["search_field", "search_button"]
                },
                
                "form_submission": {
                    "keywords": ["submit", "send", "save"],
                    "steps": [
                        "Fill required fields",
                        "Validate form data",
                        "Submit form",
                        "Confirm submission"
                    ],
                    "elements": ["form_fields", "submit_button"]
                }
            },
            
            "shopping": {
                "product_search": {
                    "keywords": ["product", "search", "find item"],
                    "steps": [
                        "Access search functionality",
                        "Enter product keywords",
                        "Apply filters if needed",
                        "Browse results",
                        "Select product"
                    ],
                    "elements": ["search_field", "filter_options", "product_links"]
                },
                
                "add_to_cart": {
                    "keywords": ["add to cart", "buy", "purchase"],
                    "steps": [
                        "Select product",
                        "Choose options (size, color, etc.)",
                        "Add to shopping cart",
                        "Verify cart contents"
                    ],
                    "elements": ["product_options", "add_to_cart_button", "cart_icon"]
                },
                
                "checkout_process": {
                    "keywords": ["checkout", "order", "payment"],
                    "steps": [
                        "Review cart contents",
                        "Proceed to checkout",
                        "Enter shipping information",
                        "Select payment method",
                        "Complete order"
                    ],
                    "elements": ["cart_button", "checkout_button", "payment_fields"]
                },
                
                "admin_review_analysis": {
                    "keywords": ["admin", "review", "approved", "count"],
                    "steps": [
                        "Access admin panel",
                        "Navigate to reviews section",
                        "Filter by review status",
                        "Count or analyze reviews",
                        "Extract metrics"
                    ],
                    "elements": ["admin_menu", "reviews_section", "status_filters"]
                }
            },
            
            "social": {
                "create_post": {
                    "keywords": ["post", "create", "publish"],
                    "steps": [
                        "Navigate to post creation",
                        "Select post type",
                        "Enter title/subject",
                        "Add content/body",
                        "Publish post"
                    ],
                    "elements": ["create_button", "title_field", "content_field", "publish_button"]
                },
                
                "vote_content": {
                    "keywords": ["vote", "upvote", "downvote"],
                    "steps": [
                        "Locate target content",
                        "Find voting controls",
                        "Click vote button",
                        "Verify vote registered"
                    ],
                    "elements": ["upvote_button", "downvote_button", "vote_count"]
                },
                
                "comment_interaction": {
                    "keywords": ["comment", "reply", "respond"],
                    "steps": [
                        "Find target post/comment",
                        "Click reply/comment button",
                        "Write comment text",
                        "Submit comment"
                    ],
                    "elements": ["comment_button", "comment_field", "submit_button"]
                }
            },
            
            "development": {
                "create_issue": {
                    "keywords": ["issue", "bug", "feature request"],
                    "steps": [
                        "Navigate to issues section",
                        "Click new issue button",
                        "Enter issue title",
                        "Add description",
                        "Set labels/assignees",
                        "Create issue"
                    ],
                    "elements": ["issues_tab", "new_issue_button", "title_field", "description_field"]
                },
                
                "commit_changes": {
                    "keywords": ["commit", "save changes", "push"],
                    "steps": [
                        "Review changes",
                        "Stage files",
                        "Write commit message",
                        "Commit changes",
                        "Push to repository"
                    ],
                    "elements": ["file_checkboxes", "commit_message_field", "commit_button"]
                },
                
                "repository_navigation": {
                    "keywords": ["repository", "repo", "browse code"],
                    "steps": [
                        "Access repository",
                        "Navigate to desired section",
                        "Browse files/directories",
                        "View file contents"
                    ],
                    "elements": ["repo_tabs", "file_tree", "file_links"]
                }
            },
            
            "knowledge": {
                "article_search": {
                    "keywords": ["search", "article", "information"],
                    "steps": [
                        "Use search functionality",
                        "Enter search terms",
                        "Select relevant article",
                        "Navigate to content"
                    ],
                    "elements": ["search_box", "search_results", "article_links"]
                },
                
                "information_extraction": {
                    "keywords": ["extract", "find information", "get data"],
                    "steps": [
                        "Locate relevant section",
                        "Read content carefully",
                        "Extract specific information",
                        "Verify accuracy"
                    ],
                    "elements": ["content_sections", "headings", "data_tables"]
                }
            },
            
            "mapping": {
                "location_search": {
                    "keywords": ["location", "address", "place"],
                    "steps": [
                        "Enter location in search",
                        "Select from suggestions",
                        "View location on map",
                        "Get additional details"
                    ],
                    "elements": ["search_field", "suggestions", "map_view"]
                },
                
                "directions": {
                    "keywords": ["directions", "route", "navigate"],
                    "steps": [
                        "Enter starting location",
                        "Enter destination",
                        "Get directions",
                        "Review route options"
                    ],
                    "elements": ["from_field", "to_field", "directions_button"]
                }
            }
        }
    
    def get_pattern_by_name(self, pattern_name: str, site_type: str = "universal") -> Dict[str, Any]:
        """Get specific pattern by name"""
        return self.patterns.get(site_type, {}).get(pattern_name, {})
    
    def get_common_elements_for_site(self, site_type: str) -> List[str]:
        """Get commonly used elements for a site type"""
        
        common_elements = {
            "shopping": [
                "search_field", "add_to_cart_button", "checkout_button", 
                "product_links", "cart_icon", "admin_menu"
            ],
            "social": [
                "create_post_button", "upvote_button", "downvote_button",
                "comment_field", "submit_button", "title_field"
            ],
            "development": [
                "issues_tab", "new_issue_button", "commit_button",
                "file_tree", "repo_tabs", "pull_request_button"
            ],
            "knowledge": [
                "search_box", "article_links", "content_sections",
                "navigation_menu", "category_links"
            ],
            "mapping": [
                "search_field", "directions_button", "map_view",
                "zoom_controls", "location_markers"
            ]
        }
        
        return common_elements.get(site_type, [])
    
    def suggest_next_steps(self, current_step: str, pattern_context: Dict[str, Any]) -> List[str]:
        """Suggest possible next steps based on current step and pattern"""
        
        # This is a simplified version - a full implementation would use
        # more sophisticated pattern matching and context analysis
        
        step_lower = current_step.lower()
        
        if "navigate" in step_lower:
            return ["Enter information", "Click button", "Search"]
        elif "search" in step_lower:
            return ["Review results", "Click result", "Refine search"]
        elif "click" in step_lower:
            return ["Wait for page load", "Enter text", "Verify action"]
        elif "enter" in step_lower or "type" in step_lower:
            return ["Submit form", "Click next", "Verify input"]
        else:
            return ["Continue task", "Verify completion", "Next action"]