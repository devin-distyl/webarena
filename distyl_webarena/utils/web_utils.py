"""
WebUtils: Utility functions for web automation

Contains helper functions for web interaction, URL handling, and WebArena compatibility.
"""

import re
import json
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse


class WebUtils:
    """Utility functions for web automation and WebArena integration"""
    
    @staticmethod
    def action_to_string(action: Dict[str, Any], action_set_tag: str) -> str:
        """Convert WebArena action to readable string"""
        
        try:
            action_type = action.get("action_type", 0)
            
            # Map action types to readable strings
            action_type_names = {
                0: "NONE",
                1: "SCROLL", 
                2: "KEY_PRESS",
                6: "CLICK",
                7: "TYPE",
                8: "HOVER",
                9: "PAGE_FOCUS",
                10: "NEW_TAB",
                11: "GO_BACK",
                12: "GO_FORWARD", 
                13: "GOTO_URL",
                14: "PAGE_CLOSE",
                17: "STOP"
            }
            
            action_name = action_type_names.get(action_type, f"ACTION_{action_type}")
            
            if action_type == 6:  # CLICK
                element_id = action.get("element_id", "")
                return f"click [{element_id}]"
            elif action_type == 7:  # TYPE
                element_id = action.get("element_id", "")
                text = WebUtils._decode_text(action.get("text", []))
                return f"type [{element_id}] [{text}]"
            elif action_type == 1:  # SCROLL
                direction = action.get("direction", "down")
                return f"scroll [{direction}]"
            elif action_type == 13:  # GOTO_URL
                url = action.get("url", "")
                return f"goto [{url}]"
            elif action_type == 17:  # STOP
                answer = action.get("answer", "")
                return f"stop [{answer}]"
            else:
                return action_name.lower()
                
        except Exception:
            return "unknown_action"
    
    @staticmethod
    def _decode_text(text_list: List[int]) -> str:
        """Decode text from WebArena text encoding"""
        try:
            # This would need to match WebArena's text encoding
            # For now, return as string representation
            return ''.join(chr(i) if isinstance(i, int) and 0 < i < 128 else str(i) for i in text_list)
        except Exception:
            return str(text_list)
    
    @staticmethod
    def extract_page_info(accessibility_tree: str) -> Dict[str, Any]:
        """Extract useful information from accessibility tree"""
        
        info = {
            "element_count": 0,
            "buttons": [],
            "links": [],
            "inputs": [],
            "headings": [],
            "forms": []
        }
        
        if not accessibility_tree:
            return info
        
        lines = accessibility_tree.split('\n')
        info["element_count"] = len([line for line in lines if re.match(r'\[\d+\]', line)])
        
        for line in lines:
            line_lower = line.lower()
            element_id_match = re.search(r'\[(\d+)\]', line)
            
            if element_id_match:
                element_id = element_id_match.group(1)
                
                if 'button' in line_lower:
                    info["buttons"].append({"id": element_id, "text": line.strip()})
                elif 'link' in line_lower:
                    info["links"].append({"id": element_id, "text": line.strip()})
                elif any(word in line_lower for word in ['textbox', 'input', 'textarea']):
                    info["inputs"].append({"id": element_id, "text": line.strip()})
                elif any(word in line_lower for word in ['heading', 'h1', 'h2', 'h3']):
                    info["headings"].append({"id": element_id, "text": line.strip()})
                elif 'form' in line_lower:
                    info["forms"].append({"id": element_id, "text": line.strip()})
        
        return info
    
    @staticmethod
    def classify_page_type(url: str, accessibility_tree: str = "") -> str:
        """Classify the type of web page"""
        
        url_lower = url.lower()
        tree_lower = accessibility_tree.lower()
        
        # URL-based classification
        if any(word in url_lower for word in ["login", "signin", "auth"]):
            return "login_page"
        elif any(word in url_lower for word in ["admin", "dashboard"]):
            return "admin_page"
        elif any(word in url_lower for word in ["search", "results"]):
            return "search_page"
        elif any(word in url_lower for word in ["cart", "checkout"]):
            return "shopping_page"
        
        # Content-based classification
        if tree_lower:
            if tree_lower.count("login") + tree_lower.count("password") > 1:
                return "login_page"
            elif tree_lower.count("search") > 2:
                return "search_page"
            elif tree_lower.count("cart") + tree_lower.count("checkout") > 0:
                return "shopping_page"
            elif tree_lower.count("post") + tree_lower.count("comment") > 2:
                return "social_page"
        
        return "general_page"
    
    @staticmethod
    def extract_urls_from_text(text: str) -> List[str]:
        """Extract URLs from text"""
        
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        return url_pattern.findall(text)
    
    @staticmethod
    def normalize_url(url: str, base_url: str = "") -> str:
        """Normalize URL for consistent handling"""
        
        if not url:
            return ""
        
        # Handle relative URLs
        if url.startswith('/') and base_url:
            parsed_base = urlparse(base_url)
            return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
        
        # Ensure protocol
        if not url.startswith(('http://', 'https://')):
            url = f"http://{url}"
        
        return url
    
    @staticmethod
    def is_webarena_site(url: str) -> bool:
        """Check if URL belongs to a WebArena site"""
        
        webarena_ports = ["7770", "7780", "9999", "8023", "8888", "3000", "4399"]
        
        for port in webarena_ports:
            if port in url:
                return True
        
        webarena_domains = ["localhost", "onestopmarket", "reddit", "gitlab", "wikipedia"]
        
        for domain in webarena_domains:
            if domain in url.lower():
                return True
        
        return False
    
    @staticmethod
    def get_site_type_from_url(url: str) -> str:
        """Determine WebArena site type from URL"""
        
        url_lower = url.lower()
        
        if "7770" in url or "7780" in url or "shop" in url_lower:
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
    
    @staticmethod
    def sanitize_text_for_action(text: str) -> str:
        """Sanitize text for use in actions"""
        
        # Remove or escape problematic characters
        text = text.replace('[', '(').replace(']', ')')
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length
        if len(text) > 200:
            text = text[:197] + "..."
        
        return text
    
    @staticmethod
    def parse_task_config(config_file: str) -> Dict[str, Any]:
        """Parse WebArena task configuration file"""
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Add derived information
            config["site_type"] = WebUtils.get_site_type_from_url(config.get("start_url", ""))
            config["requires_auth"] = config.get("require_login", False)
            
            return config
            
        except Exception as e:
            return {"error": f"Could not parse config file: {e}"}
    
    @staticmethod
    def format_trajectory_summary(trajectory: List[Dict[str, Any]]) -> str:
        """Create a readable summary of the trajectory"""
        
        if not trajectory:
            return "Empty trajectory"
        
        summary_parts = []
        action_count = 0
        
        for i, item in enumerate(trajectory):
            if i % 2 == 1:  # Actions are at odd indices
                action_count += 1
                action_str = WebUtils.action_to_string(item, "id_accessibility_tree")
                summary_parts.append(f"{action_count}. {action_str}")
        
        return "\n".join(summary_parts) if summary_parts else "No actions taken"
    
    @staticmethod
    def calculate_success_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate success metrics from task results"""
        
        if not results:
            return {"success_rate": 0.0, "average_score": 0.0}
        
        scores = [result.get("score", 0.0) for result in results]
        successful_tasks = sum(1 for score in scores if score > 0.8)
        
        return {
            "success_rate": successful_tasks / len(results),
            "average_score": sum(scores) / len(scores),
            "total_tasks": len(results),
            "successful_tasks": successful_tasks
        }