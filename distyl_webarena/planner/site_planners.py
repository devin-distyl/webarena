"""
SiteSpecificPlanners: Specialized planning logic for different WebArena sites
"""

from typing import List, Dict, Any
from ..utils.logging import DistylLogger


class SiteSpecificPlanners:
    """Factory and container for site-specific planners"""
    
    @staticmethod
    def get_planner(site_type: str):
        """Get appropriate planner for site type"""
        planners = {
            "shopping": ShoppingSitePlanner(),
            "social": SocialPlatformPlanner(),
            "development": DevelopmentPlatformPlanner(),
            "knowledge": KnowledgeBasePlanner(),
            "mapping": MappingServicePlanner()
        }
        
        return planners.get(site_type, GeneralWebPlanner())


class BaseSitePlanner:
    """Base class for site-specific planners"""
    
    def __init__(self):
        self.logger = DistylLogger(self.__class__.__name__)
    
    def plan_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan task for this site type"""
        return [f"Complete task: {instruction}"]


class ShoppingSitePlanner(BaseSitePlanner):
    """Specialized planner for shopping/e-commerce workflows"""
    
    def plan_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        instruction_lower = instruction.lower()
        
        if "admin" in instruction_lower and "review" in instruction_lower:
            return self.plan_admin_review_task(instruction, context)
        elif "search" in instruction_lower and "product" in instruction_lower:
            return self.plan_product_search(instruction, context)
        elif "purchase" in instruction_lower or "buy" in instruction_lower:
            return self.plan_product_purchase(instruction, context)
        else:
            return self.plan_general_shopping_task(instruction, context)
    
    def plan_admin_review_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan admin review analysis tasks"""
        return [
            "Navigate to admin dashboard",
            "Access reviews management section",
            "Apply appropriate filters (if needed)",
            "Analyze review data",
            "Extract requested information",
            "Provide final answer"
        ]
    
    def plan_product_search(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan product search tasks"""
        return [
            "Navigate to search functionality",
            "Enter product search terms",
            "Review search results",
            "Extract product information"
        ]
    
    def plan_product_purchase(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan product purchase workflow"""
        return [
            "Search for desired product",
            "Select appropriate product",
            "Add product to cart",
            "Proceed to checkout",
            "Complete purchase process"
        ]
    
    def plan_general_shopping_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan general shopping tasks"""
        return [
            "Navigate to appropriate section",
            "Perform requested shopping action",
            "Verify action completion"
        ]


class SocialPlatformPlanner(BaseSitePlanner):
    """Specialized planner for social media/forum workflows"""
    
    def plan_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        instruction_lower = instruction.lower()
        
        if "post" in instruction_lower:
            return self.plan_content_creation(instruction, context)
        elif "vote" in instruction_lower:
            return self.plan_voting_action(instruction, context)
        elif "comment" in instruction_lower:
            return self.plan_comment_action(instruction, context)
        else:
            return self.plan_general_social_task(instruction, context)
    
    def plan_content_creation(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan content creation tasks"""
        return [
            "Navigate to content creation area",
            "Select appropriate content type",
            "Fill in required information",
            "Add content details",
            "Publish content",
            "Verify publication success"
        ]
    
    def plan_voting_action(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan voting/rating actions"""
        return [
            "Locate target content",
            "Find voting controls",
            "Perform voting action",
            "Verify vote was recorded"
        ]
    
    def plan_comment_action(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan commenting actions"""
        return [
            "Find target post or content",
            "Access comment functionality", 
            "Write comment content",
            "Submit comment",
            "Verify comment publication"
        ]
    
    def plan_general_social_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan general social platform tasks"""
        return [
            "Navigate to relevant section",
            "Perform social interaction",
            "Confirm action completion"
        ]


class DevelopmentPlatformPlanner(BaseSitePlanner):
    """Specialized planner for development/coding workflows"""
    
    def plan_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        instruction_lower = instruction.lower()
        
        if "issue" in instruction_lower:
            return self.plan_issue_management(instruction, context)
        elif "commit" in instruction_lower:
            return self.plan_commit_workflow(instruction, context)
        elif "repository" in instruction_lower or "repo" in instruction_lower:
            return self.plan_repository_task(instruction, context)
        else:
            return self.plan_general_dev_task(instruction, context)
    
    def plan_issue_management(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan issue management tasks"""
        return [
            "Navigate to issues section",
            "Access issue management interface",
            "Create or modify issue",
            "Fill in issue details",
            "Submit issue changes",
            "Verify issue status"
        ]
    
    def plan_commit_workflow(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan commit/version control tasks"""
        return [
            "Navigate to repository",
            "Review pending changes",
            "Stage appropriate files",
            "Write commit message",
            "Commit changes",
            "Verify commit success"
        ]
    
    def plan_repository_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan repository management tasks"""
        return [
            "Navigate to target repository",
            "Access appropriate section",
            "Perform repository operation",
            "Verify operation completion"
        ]
    
    def plan_general_dev_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan general development tasks"""
        return [
            "Navigate to development environment",
            "Perform requested development action",
            "Verify action completion"
        ]


class KnowledgeBasePlanner(BaseSitePlanner):
    """Specialized planner for knowledge base/wiki workflows"""
    
    def plan_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        instruction_lower = instruction.lower()
        
        if "search" in instruction_lower:
            return self.plan_information_search(instruction, context)
        elif "navigate" in instruction_lower:
            return self.plan_navigation_task(instruction, context)
        else:
            return self.plan_general_knowledge_task(instruction, context)
    
    def plan_information_search(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan information search tasks"""
        return [
            "Access search functionality",
            "Enter search terms",
            "Navigate to relevant article",
            "Extract requested information",
            "Verify information accuracy"
        ]
    
    def plan_navigation_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan navigation tasks"""
        return [
            "Navigate to target section",
            "Browse content structure",
            "Access specific information",
            "Extract relevant details"
        ]
    
    def plan_general_knowledge_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan general knowledge base tasks"""
        return [
            "Navigate to relevant content",
            "Extract requested information",
            "Verify information completeness"
        ]


class MappingServicePlanner(BaseSitePlanner):
    """Specialized planner for mapping service workflows"""
    
    def plan_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        instruction_lower = instruction.lower()
        
        if "directions" in instruction_lower:
            return self.plan_directions_task(instruction, context)
        elif "search" in instruction_lower or "location" in instruction_lower:
            return self.plan_location_search(instruction, context)
        else:
            return self.plan_general_mapping_task(instruction, context)
    
    def plan_directions_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan directions/routing tasks"""
        return [
            "Access directions functionality",
            "Enter starting location",
            "Enter destination",
            "Get route information",
            "Extract requested details"
        ]
    
    def plan_location_search(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan location search tasks"""
        return [
            "Use location search",
            "Enter location details",
            "View location on map",
            "Extract location information"
        ]
    
    def plan_general_mapping_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan general mapping tasks"""
        return [
            "Navigate mapping interface",
            "Perform requested mapping action",
            "Extract relevant information"
        ]


class GeneralWebPlanner(BaseSitePlanner):
    """General planner for unknown or general web tasks"""
    
    def plan_task(self, instruction: str, context: Dict[str, Any]) -> List[str]:
        """Plan general web tasks"""
        return [
            "Analyze current page",
            "Identify required actions",
            "Execute primary task",
            "Verify task completion"
        ]