"""
WebKnowledgeBase: Memory system for web interactions

Sophisticated knowledge management with episodic and narrative memory for web tasks.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from ..utils.logging import DistylLogger


class WebKnowledgeBase:
    """
    Sophisticated knowledge management with episodic and narrative memory
    """
    
    def __init__(self, memory_folder_name: str = "distyl_webarena_kb", enable_web_knowledge: bool = True):
        self.memory_folder = memory_folder_name
        self.enable_web_knowledge = enable_web_knowledge
        self.logger = DistylLogger("WebKnowledgeBase")
        
        # Create memory directories
        self._ensure_memory_directories()
        
        # Memory stores
        self.episodic_memory = {}  # Subtask-level experiences
        self.narrative_memory = {}  # Task-level experiences
        self.web_knowledge_cache = {}  # Cached web knowledge
        self.site_patterns = {}  # Site-specific patterns
        
        # Current task tracking
        self.current_task_id = None
        self.current_trajectory = []
        
        # Load existing memory
        self._load_memory()
    
    def _ensure_memory_directories(self):
        """Create memory directory structure"""
        
        directories = [
            self.memory_folder,
            os.path.join(self.memory_folder, "episodic"),
            os.path.join(self.memory_folder, "narrative"), 
            os.path.join(self.memory_folder, "web_knowledge"),
            os.path.join(self.memory_folder, "site_patterns")
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.debug(f"Created memory directory: {directory}")
    
    def _load_memory(self):
        """Load existing memory from files"""
        
        try:
            # Load episodic memory
            episodic_file = os.path.join(self.memory_folder, "episodic", "subtask_experiences.json")
            if os.path.exists(episodic_file):
                with open(episodic_file, 'r') as f:
                    self.episodic_memory = json.load(f)
                    self.logger.info(f"Loaded {len(self.episodic_memory)} episodic memories")
            
            # Load narrative memory
            narrative_file = os.path.join(self.memory_folder, "narrative", "task_experiences.json")
            if os.path.exists(narrative_file):
                with open(narrative_file, 'r') as f:
                    self.narrative_memory = json.load(f)
                    self.logger.info(f"Loaded {len(self.narrative_memory)} narrative memories")
            
            # Load site patterns
            patterns_file = os.path.join(self.memory_folder, "site_patterns", "patterns.json")
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    self.site_patterns = json.load(f)
                    self.logger.info(f"Loaded site patterns for {len(self.site_patterns)} sites")
        
        except Exception as e:
            self.logger.warning(f"Could not load existing memory: {e}")
    
    def initialize_task_trajectory(self, task_id: str, intent: str, sites: List[str]):
        """Initialize memory tracking for a new task"""
        
        self.current_task_id = task_id
        self.current_trajectory = []
        
        task_info = {
            "task_id": task_id,
            "intent": intent,
            "sites": sites,
            "start_time": time.time(),
            "trajectory": []
        }
        
        self.current_trajectory.append(task_info)
        self.logger.info(f"Initialized task trajectory for task {task_id}")
    
    def retrieve_site_experience(self, site_type: str, instruction: str) -> str:
        """Retrieve experience for specific site type"""
        
        try:
            site_key = f"{site_type}_experiences"
            if site_key in self.narrative_memory:
                experiences = self.narrative_memory[site_key]
                
                # Find most relevant experience based on instruction similarity
                best_match = self._find_best_experience_match(instruction, experiences)
                
                if best_match:
                    return best_match.get("summary", "")
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error retrieving site experience: {e}")
            return ""
    
    def retrieve_similar_web_tasks(self, instruction: str) -> str:
        """Retrieve similar web task experiences"""
        
        try:
            all_experiences = []
            
            # Collect all narrative experiences
            for site_type, experiences in self.narrative_memory.items():
                if isinstance(experiences, list):
                    all_experiences.extend(experiences)
                elif isinstance(experiences, dict):
                    all_experiences.append(experiences)
            
            # Find best match
            best_match = self._find_best_experience_match(instruction, all_experiences)
            
            if best_match:
                return best_match.get("summary", "")
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error retrieving similar tasks: {e}")
            return ""
    
    def retrieve_subtask_experience(self, subtask_description: str) -> str:
        """Retrieve experience for similar subtasks"""
        
        try:
            # Simple keyword matching for now
            description_lower = subtask_description.lower()
            
            for stored_description, experience in self.episodic_memory.items():
                stored_lower = stored_description.lower()
                
                # Check for keyword overlap
                if self._calculate_similarity(description_lower, stored_lower) > 0.3:
                    return experience.get("summary", "")
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error retrieving subtask experience: {e}")
            return ""
    
    def save_task_narrative(self, task_config: Dict[str, Any], subtasks_completed: List[Dict[str, Any]], 
                          success_status: bool):
        """Save task-level narrative experience"""
        
        try:
            task_id = task_config.get("task_id", "unknown")
            intent = task_config.get("intent", "")
            sites = task_config.get("sites", [])
            
            # Create narrative summary
            narrative = {
                "task_id": task_id,
                "intent": intent,
                "sites": sites,
                "success": success_status,
                "subtask_count": len(subtasks_completed),
                "summary": self._create_task_summary(intent, subtasks_completed, success_status),
                "timestamp": time.time(),
                "subtasks": [st.get("description", "") for st in subtasks_completed]
            }
            
            # Store by site type
            for site in sites:
                site_key = f"{site}_experiences"
                if site_key not in self.narrative_memory:
                    self.narrative_memory[site_key] = []
                
                self.narrative_memory[site_key].append(narrative)
            
            # Save to file
            self._save_narrative_memory()
            
            self.logger.info(f"Saved task narrative for task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving task narrative: {e}")
    
    def save_subtask_experience(self, subtask: Dict[str, Any], success: bool = True):
        """Save subtask-level episodic experience"""
        
        try:
            description = subtask.get("description", "")
            subtask_type = subtask.get("type", "general")
            site_type = subtask.get("site_type", "general")
            
            # Create experience summary
            experience = {
                "description": description,
                "type": subtask_type,
                "site_type": site_type,
                "success": success,
                "summary": self._create_subtask_summary(subtask, success),
                "timestamp": time.time()
            }
            
            # Store experience
            memory_key = f"{site_type}_{subtask_type}_{description}"
            self.episodic_memory[memory_key] = experience
            
            # Save to file
            self._save_episodic_memory()
            
            self.logger.debug(f"Saved subtask experience: {description}")
            
        except Exception as e:
            self.logger.error(f"Error saving subtask experience: {e}")
    
    def search_web_knowledge(self, query: str, site_type: str = "") -> str:
        """Search for web knowledge (placeholder for web search integration)"""
        
        if not self.enable_web_knowledge:
            return ""
        
        try:
            # Check cache first
            cache_key = f"{site_type}_{query}".lower()
            if cache_key in self.web_knowledge_cache:
                cached_result = self.web_knowledge_cache[cache_key]
                if time.time() - cached_result["timestamp"] < 3600:  # 1 hour cache
                    return cached_result["content"]
            
            # In a full implementation, this would call external web search
            # For now, return cached knowledge or empty string
            return ""
            
        except Exception as e:
            self.logger.error(f"Error searching web knowledge: {e}")
            return ""
    
    def _find_best_experience_match(self, instruction: str, experiences: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find best matching experience based on instruction similarity"""
        
        if not experiences:
            return None
        
        instruction_lower = instruction.lower()
        best_match = None
        best_score = 0.0
        
        for experience in experiences:
            if isinstance(experience, dict):
                exp_intent = experience.get("intent", "").lower()
                exp_summary = experience.get("summary", "").lower()
                
                # Calculate similarity score
                score = max(
                    self._calculate_similarity(instruction_lower, exp_intent),
                    self._calculate_similarity(instruction_lower, exp_summary)
                )
                
                if score > best_score:
                    best_score = score
                    best_match = experience
        
        return best_match if best_score > 0.2 else None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on common words"""
        
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _create_task_summary(self, intent: str, subtasks: List[Dict[str, Any]], success: bool) -> str:
        """Create summary of task execution"""
        
        status = "successfully completed" if success else "failed"
        subtask_count = len(subtasks)
        
        # Extract key subtask types
        subtask_types = [st.get("type", "general") for st in subtasks]
        unique_types = list(set(subtask_types))
        
        summary = f"Task '{intent}' {status} with {subtask_count} subtasks involving {', '.join(unique_types)} actions."
        
        if success and subtasks:
            # Add successful pattern information
            successful_pattern = " -> ".join([st.get("description", "")[:30] for st in subtasks[:3]])
            summary += f" Successful pattern: {successful_pattern}"
        
        return summary
    
    def _create_subtask_summary(self, subtask: Dict[str, Any], success: bool) -> str:
        """Create summary of subtask execution"""
        
        description = subtask.get("description", "")
        subtask_type = subtask.get("type", "general")
        status = "succeeded" if success else "failed"
        
        return f"{subtask_type.title()} subtask '{description}' {status}"
    
    def _save_episodic_memory(self):
        """Save episodic memory to file"""
        
        try:
            episodic_file = os.path.join(self.memory_folder, "episodic", "subtask_experiences.json")
            with open(episodic_file, 'w') as f:
                json.dump(self.episodic_memory, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving episodic memory: {e}")
    
    def _save_narrative_memory(self):
        """Save narrative memory to file"""
        
        try:
            narrative_file = os.path.join(self.memory_folder, "narrative", "task_experiences.json")
            with open(narrative_file, 'w') as f:
                json.dump(self.narrative_memory, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving narrative memory: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        
        return {
            "episodic_memories": len(self.episodic_memory),
            "narrative_memories": sum(len(v) if isinstance(v, list) else 1 
                                    for v in self.narrative_memory.values()),
            "site_patterns": len(self.site_patterns),
            "web_knowledge_cached": len(self.web_knowledge_cache),
            "memory_folder": self.memory_folder,
            "current_task": self.current_task_id
        }
    
    def clear_memory(self, memory_type: str = "all"):
        """Clear memory (for testing or cleanup)"""
        
        if memory_type in ["all", "episodic"]:
            self.episodic_memory.clear()
            self.logger.info("Cleared episodic memory")
        
        if memory_type in ["all", "narrative"]:
            self.narrative_memory.clear()
            self.logger.info("Cleared narrative memory")
        
        if memory_type in ["all", "web_knowledge"]:
            self.web_knowledge_cache.clear()
            self.logger.info("Cleared web knowledge cache")
        
        if memory_type in ["all", "site_patterns"]:
            self.site_patterns.clear()
            self.logger.info("Cleared site patterns")