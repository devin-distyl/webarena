"""
AccessibilityTreeGrounder: Element grounding for web environments

Replaces Distyl's visual coordinate grounding with accessibility tree-based element grounding
Maps natural language element descriptions to WebArena element IDs
"""

import re
import time
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher

from ..utils.logging import DistylLogger
from .element_detection import ElementAutoDetector
from .multimodal_grounding import MultimodalGrounder


class AccessibilityTreeParser:
    """
    Parses WebArena accessibility tree into structured element information
    """
    
    def parse_tree(self, tree_text: str) -> List[Dict[str, Any]]:
        """
        Parse accessibility tree text into structured elements
        
        Input: "[123] button 'Login'\n[124] textbox 'Username'"
        Output: [
            {"id": "123", "role": "button", "name": "Login", "text": "Login"},
            {"id": "124", "role": "textbox", "name": "Username", "text": "Username"}
        ]
        """
        elements = []
        
        for line in tree_text.split('\n'):
            element = self._parse_tree_line(line.strip())
            if element:
                elements.append(element)
        
        return elements
    
    def _parse_tree_line(self, line: str) -> Dict[str, Any]:
        """Parse individual line of accessibility tree"""
        
        # Pattern to match [ID] role 'name' or [ID] role "name"
        patterns = [
            r'\[(\d+)\]\s+(\w+)\s+["\']([^"\']*)["\']',  # [123] button 'Login'
            r'\[(\d+)\]\s+(\w+)\s+(.+)',                 # [123] button Login
            r'\[(\d+)\]\s+(\w+)',                        # [123] button
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                groups = match.groups()
                return {
                    "id": groups[0],
                    "role": groups[1],
                    "name": groups[2] if len(groups) > 2 else "",
                    "text": line,
                    "line": line
                }
        
        return {}
    
    def find_elements_by_role(self, role: str, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find all elements with specific role"""
        return [elem for elem in elements if elem.get("role", "").lower() == role.lower()]
    
    def find_elements_by_name_pattern(self, pattern: str, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find elements whose names match a pattern"""
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        
        return [
            elem for elem in elements 
            if compiled_pattern.search(elem.get("name", "")) or 
               compiled_pattern.search(elem.get("text", ""))
        ]


class SemanticElementMatcher:
    """
    Uses LLM to match element descriptions to accessibility tree elements
    """
    
    def __init__(self, engine_params: Dict[str, Any]):
        self.engine_params = engine_params
        self.semantic_patterns = self._load_semantic_patterns()
        self.logger = DistylLogger("SemanticElementMatcher")
    
    def match_element(self, description: str, elements: List[Dict[str, Any]]) -> str:
        """Use LLM to find best matching element"""
        
        if not elements:
            return ""
        
        # Prepare elements for LLM analysis
        element_candidates = self._format_elements_for_llm(elements)
        
        prompt = f"""
Find the element that best matches the description: "{description}"

Available elements:
{element_candidates}

Return only the element ID (number) of the best match, or "none" if no good match exists.
Consider element roles, names, and typical web UI patterns.

Element ID:"""
        
        try:
            response = self._call_llm(prompt)
            element_id = response.strip().lower()
            
            # Validate response is a valid element ID
            if element_id.isdigit() and any(elem["id"] == element_id for elem in elements):
                return element_id
                
        except Exception as e:
            self.logger.error(f"LLM grounding failed: {e}")
        
        return ""
    
    def _format_elements_for_llm(self, elements: List[Dict[str, Any]]) -> str:
        """Format elements for LLM prompt"""
        formatted = []
        for elem in elements[:20]:  # Limit to avoid token limits
            role = elem.get("role", "unknown")
            name = elem.get("name", "")
            element_id = elem.get("id", "")
            
            formatted.append(f"ID {element_id}: {role} '{name}'")
        
        return "\n".join(formatted)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt - placeholder for actual implementation"""
        # This would integrate with the actual LLM service
        # For now, return a placeholder
        return "none"
    
    def _load_semantic_patterns(self) -> Dict[str, List[str]]:
        """Load common semantic patterns for web elements"""
        return {
            "login_patterns": ["login", "sign in", "log in", "enter", "submit"],
            "search_patterns": ["search", "find", "look up", "query"],
            "button_patterns": ["button", "click", "press", "submit"],
            "input_patterns": ["input", "field", "textbox", "enter", "type"],
            "navigation_patterns": ["menu", "nav", "link", "go to", "navigate"]
        }


class AccessibilityTreeGrounder:
    """
    Main grounding class that maps natural language element descriptions to WebArena element IDs
    """
    
    def __init__(self, engine_params: Dict[str, Any]):
        self.engine_params = engine_params
        self.element_cache = {}
        self.tree_parser = AccessibilityTreeParser()
        self.semantic_matcher = SemanticElementMatcher(engine_params)
        self.multimodal_grounder = MultimodalGrounder(engine_params)
        self.element_detector = ElementAutoDetector(self)
        self.logger = DistylLogger("AccessibilityTreeGrounder", log_level="DEBUG")
        
        # Performance tracking
        self.grounding_feedback = {}
        self.element_penalties = {}
    
    def ground_element_description(self, description: str, observation: Dict[str, Any]) -> str:
        """
        Main grounding method - converts element description to element ID
        
        Args:
            description: Natural language description ("search button", "username field")
            observation: WebArena observation with accessibility tree
        
        Returns:
            element_id: WebArena element ID or empty string if not found
        """
        
        # Check cache first
        cache_key = self._create_cache_key(description, observation)
        if cache_key in self.element_cache:
            return self.element_cache[cache_key]
        
        # Parse accessibility tree
        elements = self.tree_parser.parse_tree(observation.get("accessibility_tree", observation.get("text", "")))
        
        if not elements:
            self.logger.warning("No elements found in accessibility tree")
            return ""
        
        # Try multiple grounding strategies
        element_id = (
            self._exact_text_match(description, elements) or
            self._role_based_matching(description, elements) or
            self._fuzzy_matching(description, elements) or
            self._semantic_matching(description, elements) or
            self._multimodal_grounding(description, observation)
        )
        
        # Cache result
        self.element_cache[cache_key] = element_id
        
        if element_id:
            self.logger.debug(f"Grounded '{description}' to element {element_id}")
        else:
            self.logger.warning(f"Could not ground '{description}'")
        
        return element_id
    
    def ground_multiple_elements(self, descriptions: List[str], observation: Dict[str, Any]) -> Dict[str, str]:
        """Ground multiple element descriptions in one pass for efficiency"""
        results = {}
        elements = self.tree_parser.parse_tree(observation.get("accessibility_tree", observation.get("text", "")))
        
        for desc in descriptions:
            results[desc] = self.ground_element_description(desc, observation)
        
        return results
    
    def resolve_action_parameters(self, action_template: str, observation: Dict[str, Any]) -> str:
        """
        Resolve parameterized actions to concrete WebArena actions
        
        Input: "click [search_button]"
        Output: "click [123]" (where 123 is the actual element ID)
        """
        
        self.logger.debug(f"🔧 RESOLVE_ACTION_PARAMETERS: Input template: {action_template}")
        
        # Find all parameter placeholders
        parameters = re.findall(r'\[([^\]]+)\]', action_template)
        self.logger.debug(f"🎯 Found parameters to resolve: {parameters}")
        
        resolved_action = action_template
        for param in parameters:
            if param.isdigit():
                # Already an element ID, no resolution needed
                self.logger.debug(f"✅ Parameter '{param}' is already an element ID")
                continue
            
            self.logger.debug(f"🔍 Resolving parameter: {param}")
            
            # Check if it's an auto-detect parameter
            if param.startswith("auto_detect_"):
                self.logger.debug(f"🤖 Using auto-detect for: {param}")
                element_id = self.element_detector.resolve_auto_detect_element(param, observation)
            else:
                # Ground the parameter description
                description = param.replace('_', ' ')
                self.logger.debug(f"📝 Grounding description: '{description}'")
                element_id = self.ground_element_description(description, observation)
            
            if element_id:
                self.logger.debug(f"✅ Resolved '{param}' to element ID: {element_id}")
                resolved_action = resolved_action.replace(f'[{param}]', f'[{element_id}]')
            else:
                # Could not resolve - mark as failed
                self.logger.error(f"❌ Could not resolve parameter: {param}")
                self.logger.debug(f"🔍 Available accessibility tree: {observation.get('accessibility_tree', observation.get('text', ''))[:200]}...")
                return f"none  # Could not resolve {param}"
        
        self.logger.debug(f"🎯 Final resolved action: {resolved_action}")
        return resolved_action
    
    def find_element_by_keywords(self, keywords: List[str], observation: Dict[str, Any]) -> str:
        """Find element that matches any of the given keywords"""
        elements = self.tree_parser.parse_tree(observation.get("accessibility_tree", observation.get("text", "")))
        
        for keyword in keywords:
            element_id = self._exact_text_match(keyword, elements)
            if element_id:
                return element_id
        
        return ""
    
    def reset(self) -> None:
        """Reset grounder state for new task"""
        self.element_cache.clear()
    
    def _create_cache_key(self, description: str, observation: Dict[str, Any]) -> str:
        """Create cache key for grounding results"""
        tree_hash = hash(observation.get("accessibility_tree", observation.get("text", "")))
        return f"{description}_{tree_hash}"
    
    def _exact_text_match(self, description: str, elements: List[Dict[str, Any]]) -> str:
        """Find element with exact text match"""
        description_lower = description.lower()
        
        for element in elements:
            element_name = element.get("name", "").lower()
            element_text = element.get("text", "").lower()
            
            if (description_lower in element_name or 
                description_lower in element_text or
                element_name in description_lower):
                return element["id"]
        
        return ""
    
    def _role_based_matching(self, description: str, elements: List[Dict[str, Any]]) -> str:
        """Match based on element roles and common patterns"""
        
        # Define role mappings for common descriptions
        role_mappings = {
            "button": ["button", "submit"],
            "field": ["textbox", "input", "textarea"],
            "link": ["link"],
            "menu": ["menu", "menuitem"],
            "search": ["searchbox", "textbox"],
            "dropdown": ["combobox", "select"]
        }
        
        description_lower = description.lower()
        
        # Find elements matching expected roles
        for desc_keyword, roles in role_mappings.items():
            if desc_keyword in description_lower:
                for role in roles:
                    matching_elements = self.tree_parser.find_elements_by_role(role, elements)
                    if matching_elements:
                        # Return first match, could be enhanced with ranking
                        return matching_elements[0]["id"]
        
        return ""
    
    def _fuzzy_matching(self, description: str, elements: List[Dict[str, Any]]) -> str:
        """Fuzzy string matching with ranking"""
        candidates = []
        
        for element in elements:
            element_text = f"{element.get('role', '')} {element.get('name', '')}"
            
            # Calculate similarity score
            similarity = SequenceMatcher(None, description.lower(), element_text.lower()).ratio()
            
            if similarity > 0.3:  # Minimum similarity threshold
                candidates.append((element["id"], similarity))
        
        if candidates:
            # Return element with highest similarity
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return ""
    
    def _semantic_matching(self, description: str, elements: List[Dict[str, Any]]) -> str:
        """Use semantic matcher for complex matching"""
        return self.semantic_matcher.match_element(description, elements)
    
    def _multimodal_grounding(self, description: str, observation: Dict[str, Any]) -> str:
        """Use multimodal grounder as fallback"""
        if self.multimodal_grounder.vision_enabled:
            return self.multimodal_grounder.ground_with_screenshot(description, observation)
        return ""
    
    def update_grounding_from_feedback(self, description: str, attempted_element_id: str, 
                                     success: bool, observation: Dict[str, Any]):
        """
        Learn from grounding success/failure to improve future grounding
        """
        
        feedback_key = f"{description}_{hash(str(observation))}"
        
        self.grounding_feedback[feedback_key] = {
            'element_id': attempted_element_id,
            'success': success,
            'description': description,
            'timestamp': time.time()
        }
        
        # Use feedback to adjust future grounding decisions
        if not success:
            # Mark this element as less likely for this description
            self._penalize_element_for_description(description, attempted_element_id)
    
    def _penalize_element_for_description(self, description: str, element_id: str):
        """Reduce likelihood of selecting this element for this description type"""
        penalty_key = f"{description}_{element_id}"
        self.element_penalties[penalty_key] = self.element_penalties.get(penalty_key, 0) + 1