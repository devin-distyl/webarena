"""
SiteSpecificActions: Site-specific action patterns for WebArena environments

Contains action libraries optimized for different WebArena sites:
- Shopping (OneStopShop)
- Social (Reddit)
- Development (GitLab)
- Knowledge (Wikipedia)
- Mapping services
"""

from typing import List, Dict, Any
from ..utils.logging import DistylLogger


class SiteSpecificActions:
    """
    Site-specific action patterns for WebArena environments
    """
    
    def __init__(self):
        self.logger = DistylLogger("SiteSpecificActions")
    
    @staticmethod
    def get_site_actions(site_type: str) -> 'BaseSiteActions':
        """Factory method to get site-specific action class"""
        
        site_classes = {
            "shopping": ShoppingActions(),
            "social": SocialActions(), 
            "development": DevelopmentActions(),
            "knowledge": KnowledgeActions(),
            "mapping": MappingActions()
        }
        
        return site_classes.get(site_type, GeneralActions())


class BaseSiteActions:
    """Base class for site-specific actions"""
    
    def __init__(self):
        self.logger = DistylLogger(self.__class__.__name__)
    
    def get_common_patterns(self) -> Dict[str, List[str]]:
        """Get common interaction patterns for this site type"""
        return {}
    
    def login_flow(self, username: str, password: str) -> List[str]:
        """Generate login action sequence"""
        return [
            "click [auto_detect_username_field]",
            f"type [auto_detect_username_field] [{username}] [0]",
            "click [auto_detect_password_field]", 
            f"type [auto_detect_password_field] [{password}] [0]",
            "click [auto_detect_login_button]"
        ]
    
    def search_flow(self, query: str) -> List[str]:
        """Generate search action sequence"""
        return [
            "click [auto_detect_search_field]",
            f"type [auto_detect_search_field] [{query}] [1]"
        ]


class ShoppingActions(BaseSiteActions):
    """Shopping site-specific actions (OneStopShop, Magento admin)"""
    
    def get_common_patterns(self) -> Dict[str, List[str]]:
        return {
            "product_search": ["search", "find product", "look for"],
            "add_to_cart": ["add to cart", "add", "buy"],
            "checkout": ["checkout", "proceed", "purchase"],
            "view_cart": ["cart", "shopping cart", "view cart"],
            "admin_login": ["admin", "login", "sign in"]
        }
    
    def product_search_flow(self, product_query: str) -> List[str]:
        """Search for a product"""
        return [
            "click [auto_detect_search_field]",
            f"type [auto_detect_search_field] [{product_query}] [1]",
            "scroll [down]"  # Scroll to see results
        ]
    
    def add_to_cart_flow(self, product_element_id: str = "") -> List[str]:
        """Add product to cart"""
        if product_element_id:
            return [
                f"click [{product_element_id}]",  # Click specific product
                "click [auto_detect_add_to_cart_button]"
            ]
        else:
            return [
                "click [auto_detect_add_to_cart_button]"
            ]
    
    def checkout_flow(self) -> List[str]:
        """Complete checkout process"""
        return [
            "click [auto_detect_cart_button]",
            "click [auto_detect_checkout_button]",
            "click [auto_detect_proceed_button]"
        ]
    
    def admin_product_management(self, action: str, product_data: Dict[str, str] = None) -> List[str]:
        """Admin product management actions"""
        
        if action == "view_products":
            return [
                "click [auto_detect_products_menu]",
                "click [auto_detect_catalog_menu]"
            ]
        
        elif action == "add_product" and product_data:
            return [
                "click [auto_detect_products_menu]",
                "click [auto_detect_add_product_button]",
                f"type [auto_detect_product_name_field] [{product_data.get('name', '')}] [0]",
                f"type [auto_detect_product_price_field] [{product_data.get('price', '')}] [0]",
                "click [auto_detect_save_button]"
            ]
        
        elif action == "view_orders":
            return [
                "click [auto_detect_orders_menu]",
                "click [auto_detect_view_all_orders]"
            ]
        
        return []
    
    def review_analysis(self, action: str) -> List[str]:
        """Review analysis actions for admin"""
        
        if action == "count_approved_reviews":
            return [
                "click [auto_detect_reviews_menu]",
                "click [auto_detect_all_reviews]",
                "click [auto_detect_approved_filter]"
            ]
        
        elif action == "view_review_details":
            return [
                "click [auto_detect_reviews_menu]",
                "click [auto_detect_pending_reviews]"
            ]
        
        return []


class SocialActions(BaseSiteActions):
    """Social platform actions (Reddit-like forums)"""
    
    def get_common_patterns(self) -> Dict[str, List[str]]:
        return {
            "create_post": ["create", "new post", "submit"],
            "vote": ["upvote", "downvote", "vote"],
            "comment": ["comment", "reply", "respond"],
            "browse": ["browse", "view", "read"]
        }
    
    def create_post_flow(self, title: str, content: str, subreddit: str = "") -> List[str]:
        """Create a new post"""
        actions = [
            "click [auto_detect_create_post_button]"
        ]
        
        if subreddit:
            actions.extend([
                "click [auto_detect_subreddit_field]",
                f"type [auto_detect_subreddit_field] [{subreddit}] [0]"
            ])
        
        actions.extend([
            f"type [auto_detect_title_field] [{title}] [0]",
            f"type [auto_detect_content_field] [{content}] [0]",
            "click [auto_detect_submit_button]"
        ])
        
        return actions
    
    def vote_on_post(self, post_id: str, vote_type: str) -> List[str]:
        """Vote on a post (upvote/downvote)"""
        return [f"click [auto_detect_{vote_type}_button_{post_id}]"]
    
    def comment_on_post(self, post_id: str, comment_text: str) -> List[str]:
        """Comment on a post"""
        return [
            f"click [auto_detect_comment_button_{post_id}]",
            f"type [auto_detect_comment_field] [{comment_text}] [0]",
            "click [auto_detect_submit_comment_button]"
        ]
    
    def browse_subreddit(self, subreddit_name: str) -> List[str]:
        """Navigate to specific subreddit"""
        return [
            "click [auto_detect_search_field]",
            f"type [auto_detect_search_field] [r/{subreddit_name}] [1]",
            "click [auto_detect_first_result]"
        ]


class DevelopmentActions(BaseSiteActions):
    """Development platform actions (GitLab, GitHub-like)"""
    
    def get_common_patterns(self) -> Dict[str, List[str]]:
        return {
            "repository": ["repo", "repository", "project"],
            "issue": ["issue", "bug", "task"],
            "commit": ["commit", "push", "save"],
            "merge": ["merge", "pull request", "PR"]
        }
    
    def create_issue_flow(self, title: str, description: str, labels: List[str] = None) -> List[str]:
        """Create a new issue"""
        actions = [
            "click [auto_detect_issues_tab]",
            "click [auto_detect_new_issue_button]",
            f"type [auto_detect_issue_title_field] [{title}] [0]",
            f"type [auto_detect_issue_description_field] [{description}] [0]"
        ]
        
        if labels:
            for label in labels:
                actions.extend([
                    "click [auto_detect_labels_field]",
                    f"type [auto_detect_labels_field] [{label}] [0]"
                ])
        
        actions.append("click [auto_detect_create_issue_button]")
        return actions
    
    def commit_changes_flow(self, commit_message: str, files: List[str] = None) -> List[str]:
        """Commit changes to repository"""
        actions = []
        
        if files:
            for file in files:
                actions.extend([
                    f"click [auto_detect_file_checkbox_{file}]"
                ])
        
        actions.extend([
            f"type [auto_detect_commit_message_field] [{commit_message}] [0]",
            "click [auto_detect_commit_button]"
        ])
        
        return actions
    
    def repository_navigation(self, repo_name: str, section: str = "code") -> List[str]:
        """Navigate to repository section"""
        
        sections_map = {
            "code": "auto_detect_code_tab",
            "issues": "auto_detect_issues_tab", 
            "pull_requests": "auto_detect_pr_tab",
            "settings": "auto_detect_settings_tab"
        }
        
        tab_element = sections_map.get(section, "auto_detect_code_tab")
        
        return [
            "click [auto_detect_search_field]",
            f"type [auto_detect_search_field] [{repo_name}] [1]",
            "click [auto_detect_first_result]",
            f"click [{tab_element}]"
        ]


class KnowledgeActions(BaseSiteActions):
    """Knowledge base actions (Wikipedia-like)"""
    
    def get_common_patterns(self) -> Dict[str, List[str]]:
        return {
            "search": ["search", "find", "look up"],
            "navigate": ["go to", "visit", "open"],
            "read": ["read", "view", "browse"]
        }
    
    def search_and_navigate_flow(self, query: str) -> List[str]:
        """Search for topic and navigate to article"""
        return [
            "click [auto_detect_search_box]",
            f"type [auto_detect_search_box] [{query}] [1]",
            "click [auto_detect_first_result]"
        ]
    
    def extract_information_flow(self, section: str = "") -> List[str]:
        """Navigate to specific section of article"""
        actions = ["scroll [down]"]  # Scroll to see content
        
        if section:
            actions.extend([
                "click [auto_detect_search_field]",
                f"type [auto_detect_search_field] [{section}] [1]"
            ])
        
        return actions
    
    def follow_link_flow(self, link_text: str) -> List[str]:
        """Follow a link within an article"""
        return [
            f"click [auto_detect_link_{link_text}]"
        ]


class MappingActions(BaseSiteActions):
    """Mapping service actions"""
    
    def get_common_patterns(self) -> Dict[str, List[str]]:
        return {
            "search": ["search", "find location", "directions"],
            "navigate": ["go to", "navigate", "route"],
            "zoom": ["zoom", "closer", "further"]
        }
    
    def location_search_flow(self, location: str) -> List[str]:
        """Search for a location"""
        return [
            "click [auto_detect_search_field]",
            f"type [auto_detect_search_field] [{location}] [1]"
        ]
    
    def directions_flow(self, from_location: str, to_location: str) -> List[str]:
        """Get directions between locations"""
        return [
            "click [auto_detect_directions_button]",
            f"type [auto_detect_from_field] [{from_location}] [0]",
            f"type [auto_detect_to_field] [{to_location}] [0]",
            "click [auto_detect_get_directions_button]"
        ]


class GeneralActions(BaseSiteActions):
    """General web actions for unknown site types"""
    
    def get_common_patterns(self) -> Dict[str, List[str]]:
        return {
            "navigation": ["menu", "nav", "link"],
            "interaction": ["button", "click", "submit"],
            "input": ["field", "input", "textbox"]
        }
    
    def generic_form_fill(self, form_data: Dict[str, str]) -> List[str]:
        """Fill out a generic form"""
        actions = []
        
        for field_name, value in form_data.items():
            actions.extend([
                f"click [auto_detect_{field_name}_field]",
                f"type [auto_detect_{field_name}_field] [{value}] [0]"
            ])
        
        actions.append("click [auto_detect_submit_button]")
        return actions
    
    def generic_navigation(self, target: str) -> List[str]:
        """Generic navigation actions"""
        return [
            f"click [auto_detect_{target}_link]"
        ]