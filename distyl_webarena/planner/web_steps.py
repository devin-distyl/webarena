"""
WebStepPlanner: Web-specific task decomposition and DAG generation

Adapted StepPlanner for web-specific task decomposition and DAG generation
Understands web interaction patterns and site-specific workflows
"""

import json
import time
from typing import List, Dict, Any, Optional
from ..utils.logging import DistylLogger
from ..actions.site_actions import SiteSpecificActions
from .web_patterns import WebInteractionPatterns


class WebStepPlanner:
    """
    Adapted StepPlanner for web-specific task decomposition and DAG generation
    Understands web interaction patterns and site-specific workflows
    """
    
    def __init__(self, engine_params: Dict[str, Any], memory, n_candidates: int = 1):
        self.engine_params = engine_params
        self.memory = memory
        self.n_candidates = n_candidates
        self.web_patterns = WebInteractionPatterns()
        self.site_actions = SiteSpecificActions()
        self.logger = DistylLogger("WebStepPlanner", log_level="DEBUG")
        
        # Planning state
        self.current_plan = []
        self.planning_context = {}
    
    def get_action_queue(self, instruction: str, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate web-specific action queue with DAG-based planning
        """
        
        try:
            # Analyze current context
            context = self._analyze_web_context(observation)
            site_type = context["site_type"]
            
            self.logger.info(f"Planning for site type: {site_type}")
            
            # Retrieve relevant knowledge
            knowledge_context = self._gather_web_knowledge(instruction, context)
            
            # Generate site-specific plan
            raw_plan = self._generate_web_plan(instruction, context, knowledge_context)
            
            # Convert to executable subtasks
            subtasks = self._convert_plan_to_subtasks(raw_plan, site_type)
            
            self.current_plan = subtasks
            self.logger.log_planning(instruction, len(subtasks))
            
            return subtasks
            
        except Exception as e:
            self.logger.error(f"Error in planning: {str(e)}")
            return self._create_fallback_plan(instruction, observation)
    
    def reset(self):
        """Reset planner state"""
        self.current_plan = []
        self.planning_context = {}
    
    def _analyze_web_context(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current web page context"""
        
        obs_data = observation.get("observation", {})
        url = obs_data.get("url", "")
        accessibility_tree = obs_data.get("accessibility_tree", obs_data.get("text", ""))
        
        return {
            "site_type": self._classify_site_type(url),
            "current_page": self._identify_page_type(accessibility_tree),
            "available_actions": self._extract_available_actions(accessibility_tree),
            "login_status": self._detect_login_status(accessibility_tree),
            "url": url
        }
    
    def _classify_site_type(self, url: str) -> str:
        """Classify the type of website"""
        url_lower = url.lower()
        
        # Check for shopping admin sites first (more specific)
        if "/admin" in url_lower or "7780" in url:
            return "shopping_admin"
        # Check for regular shopping sites
        elif "7770" in url or "shop" in url_lower:
            return "shopping"
        elif "9999" in url or "reddit" in url_lower:
            return "social"
        elif "8023" in url or "gitlab" in url_lower:
            return "development"
        elif "8888" in url or "wikipedia" in url_lower:
            return "knowledge"
        elif "3000" in url or "map" in url_lower:
            return "mapping"
        else:
            return "general"
    
    def _identify_page_type(self, accessibility_tree: str) -> str:
        """Identify the type of current page"""
        tree_lower = accessibility_tree.lower()
        
        if "login" in tree_lower and "password" in tree_lower:
            return "login_page"
        elif tree_lower.count("search") > 2:
            return "search_page"
        elif "cart" in tree_lower or "checkout" in tree_lower:
            return "shopping_page"
        elif "admin" in tree_lower or "dashboard" in tree_lower:
            return "admin_page"
        else:
            return "content_page"
    
    def _extract_available_actions(self, accessibility_tree: str) -> List[str]:
        """Extract available actions from accessibility tree"""
        actions = []
        
        if not accessibility_tree:
            return actions
        
        tree_lower = accessibility_tree.lower()
        
        if "button" in tree_lower:
            actions.append("click_button")
        if any(word in tree_lower for word in ["textbox", "input", "textarea"]):
            actions.append("type_text")
        if "link" in tree_lower:
            actions.append("click_link")
        if "search" in tree_lower:
            actions.append("search")
        if "form" in tree_lower:
            actions.append("submit_form")
        
        return actions
    
    def _detect_login_status(self, accessibility_tree: str) -> str:
        """Detect if user is logged in"""
        tree_lower = accessibility_tree.lower()
        
        if any(word in tree_lower for word in ["logout", "sign out", "profile"]):
            return "logged_in"
        elif any(word in tree_lower for word in ["login", "sign in"]):
            return "logged_out"
        else:
            return "unknown"
    
    def _gather_web_knowledge(self, instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather relevant web knowledge for planning"""
        
        # Get site-specific experience from memory
        site_experience = ""
        similar_tasks = ""
        
        try:
            if hasattr(self.memory, 'retrieve_site_experience'):
                site_experience = self.memory.retrieve_site_experience(
                    context["site_type"], instruction
                )
            
            if hasattr(self.memory, 'retrieve_similar_web_tasks'):
                similar_tasks = self.memory.retrieve_similar_web_tasks(instruction)
        except Exception as e:
            self.logger.warning(f"Could not retrieve memory: {e}")
        
        # Get interaction patterns
        interaction_patterns = self.web_patterns.get_relevant_patterns(
            instruction, context["site_type"]
        )
        
        return {
            "site_experience": site_experience,
            "similar_tasks": similar_tasks,
            "interaction_patterns": interaction_patterns,
            "web_knowledge": ""  # Placeholder for web search results
        }
    
    def _generate_web_plan(self, instruction: str, context: Dict[str, Any], knowledge: Dict[str, Any]) -> List[str]:
        """Generate web-specific plan"""
        
        site_type = context["site_type"]
        current_page = context["current_page"]
        login_status = context["login_status"]
        
        # Use site-specific planning logic
        if site_type == "shopping_admin":
            return self._plan_shopping_admin_task(instruction, context, knowledge)
        elif site_type == "shopping":
            return self._plan_shopping_task(instruction, context, knowledge)
        elif site_type == "social":
            return self._plan_social_task(instruction, context, knowledge)
        elif site_type == "development":
            return self._plan_development_task(instruction, context, knowledge)
        elif site_type == "knowledge":
            return self._plan_knowledge_task(instruction, context, knowledge)
        else:
            return self._plan_general_task(instruction, context, knowledge)
    
    def _plan_shopping_admin_task(self, instruction: str, context: Dict[str, Any], knowledge: Dict[str, Any]) -> List[str]:
        """Plan shopping admin panel tasks"""
        
        instruction_lower = instruction.lower()
        
        # Handle report generation tasks
        if "report" in instruction_lower:
            if "coupon" in instruction_lower:
                return [
                    "Navigate to Reports section",
                    "Access Sales Reports",
                    "Select Coupons report type", 
                    "Set date range parameters",
                    "Generate coupon report"
                ]
            elif "sales" in instruction_lower:
                return [
                    "Navigate to Reports section",
                    "Access Sales Reports",
                    "Configure sales report parameters",
                    "Generate sales report"
                ]
            else:
                return [
                    "Navigate to Reports section",
                    "Select appropriate report type",
                    "Configure report parameters",
                    "Generate requested report"
                ]
        
        # Handle product/catalog management
        elif "product" in instruction_lower and ("add" in instruction_lower or "create" in instruction_lower):
            return [
                "Navigate to Catalog section",
                "Access product management",
                "Create new product entry",
                "Fill product details",
                "Save product"
            ]
        
        # Handle customer management
        elif "customer" in instruction_lower:
            return [
                "Navigate to Customers section",
                "Access customer management",
                "Perform customer-related task"
            ]
        
        # Handle order management
        elif "order" in instruction_lower:
            return [
                "Navigate to Sales section",
                "Access order management",
                "Perform order-related task"
            ]
        
        # General admin task fallback
        else:
            return [
                "Navigate to appropriate admin section",
                "Access relevant management area",
                "Perform requested admin task",
                "Verify task completion"
            ]
    
    def _plan_shopping_task(self, instruction: str, context: Dict[str, Any], knowledge: Dict[str, Any]) -> List[str]:
        """Plan shopping-specific tasks"""
        
        instruction_lower = instruction.lower()
        
        if "search" in instruction_lower and "product" in instruction_lower:
            query = self._extract_search_query(instruction)
            return [
                "Navigate to search functionality",
                f"Search for product: {query}",
                "Review search results",
                "Extract product information"
            ]
        
        elif "admin" in instruction_lower and "review" in instruction_lower:
            return [
                "Navigate to admin panel",
                "Access reviews section", 
                "Filter or search reviews",
                "Count and analyze review data",
                "Extract final answer"
            ]
        
        elif "cart" in instruction_lower or "checkout" in instruction_lower:
            return [
                "Navigate to shopping cart",
                "Review cart contents",
                "Proceed to checkout if needed",
                "Complete purchase process"
            ]
        
        else:
            return [
                "Navigate to appropriate section",
                "Perform requested shopping action",
                "Verify action completion"
            ]
    
    def _plan_social_task(self, instruction: str, context: Dict[str, Any], knowledge: Dict[str, Any]) -> List[str]:
        """Plan social platform tasks"""
        
        instruction_lower = instruction.lower()
        
        if "post" in instruction_lower:
            return [
                "Navigate to post creation",
                "Fill in post details",
                "Submit new post",
                "Verify post creation"
            ]
        
        elif "comment" in instruction_lower:
            return [
                "Find target post or content",
                "Access comment functionality",
                "Write and submit comment"
            ]
        
        elif "vote" in instruction_lower:
            return [
                "Locate target content",
                "Perform voting action",
                "Verify vote was recorded"
            ]
        
        else:
            return [
                "Navigate to relevant section",
                "Perform social interaction",
                "Confirm action completion"
            ]
    
    def _plan_development_task(self, instruction: str, context: Dict[str, Any], knowledge: Dict[str, Any]) -> List[str]:
        """Plan development platform tasks"""
        
        instruction_lower = instruction.lower()
        
        if "issue" in instruction_lower:
            return [
                "Navigate to issues section",
                "Create or manage issue",
                "Fill in issue details",
                "Submit issue"
            ]
        
        elif "commit" in instruction_lower:
            return [
                "Navigate to repository",
                "Access commit functionality",
                "Make code changes",
                "Commit changes"
            ]
        
        elif "repository" in instruction_lower or "repo" in instruction_lower:
            return [
                "Navigate to repository",
                "Browse repository content",
                "Perform requested action"
            ]
        
        else:
            return [
                "Navigate to development section",
                "Perform development task",
                "Verify completion"
            ]
    
    def _plan_knowledge_task(self, instruction: str, context: Dict[str, Any], knowledge: Dict[str, Any]) -> List[str]:
        """Plan knowledge base tasks"""
        
        query = self._extract_search_query(instruction)
        
        return [
            f"Search for information: {query}",
            "Navigate to relevant article",
            "Extract requested information",
            "Verify information accuracy"
        ]
    
    def _plan_general_task(self, instruction: str, context: Dict[str, Any], knowledge: Dict[str, Any]) -> List[str]:
        """Plan general web tasks"""
        
        return [
            "Analyze current page",
            "Identify required actions",
            "Execute primary task",
            "Verify task completion"
        ]
    
    def _extract_search_query(self, instruction: str) -> str:
        """Extract search query from instruction"""
        
        # Simple extraction - look for quoted text or keywords
        import re
        
        # Look for quoted text
        quoted_match = re.search(r'["\']([^"\']+)["\']', instruction)
        if quoted_match:
            return quoted_match.group(1)
        
        # Look for common patterns
        search_patterns = [
            r'search for (.+?)(?:\.|$)',
            r'find (.+?)(?:\.|$)',
            r'look for (.+?)(?:\.|$)',
            r'about (.+?)(?:\.|$)'
        ]
        
        for pattern in search_patterns:
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback - return the instruction itself
        return instruction
    
    def _convert_plan_to_subtasks(self, plan: List[str], site_type: str) -> List[Dict[str, Any]]:
        """Convert high-level plan to executable subtasks"""
        
        subtasks = []
        
        for i, step in enumerate(plan):
            subtask = {
                "id": f"subtask_{i}",
                "description": step,
                "type": self._classify_subtask_type(step),
                "site_type": site_type,
                "dependencies": [],
                "completed": False
            }
            
            # Add dependencies (simple linear for now)
            if i > 0:
                subtask["dependencies"] = [f"subtask_{i-1}"]
            
            subtasks.append(subtask)
        
        return subtasks
    
    def _classify_subtask_type(self, step: str) -> str:
        """Classify the type of subtask"""
        
        step_lower = step.lower()
        
        if "navigate" in step_lower or "go to" in step_lower:
            return "navigation"
        elif "search" in step_lower:
            return "search"
        elif "click" in step_lower:
            return "click"
        elif "type" in step_lower or "fill" in step_lower:
            return "input"
        elif "extract" in step_lower or "get" in step_lower:
            return "extraction"
        elif "verify" in step_lower or "check" in step_lower:
            return "verification"
        else:
            return "general"
    
    def _create_fallback_plan(self, instruction: str, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a simple fallback plan when detailed planning fails"""
        
        self.logger.warning("Using fallback planning")
        
        return [
            {
                "id": "fallback_0",
                "description": f"Attempt to complete: {instruction}",
                "type": "general",
                "site_type": "general",
                "dependencies": [],
                "completed": False
            }
        ]