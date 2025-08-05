"""
WebArenaAdapter: Integration with existing WebArena parallel execution system

Provides seamless integration between Distyl-WebArena agent and the existing
run_parallel_distyl.py system while maintaining full compatibility.
"""

import json
import os
import sys
from typing import Dict, Any, List, Optional

# Add the WebArena project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..controller.controller import DistylWebArenaController
from ..utils.logging import DistylLogger


class WebArenaAdapter:
    """
    Adapter to integrate Distyl-WebArena with existing WebArena infrastructure
    """
    
    def __init__(self, model_name: str = "distyl-webarena", config_path: str = None, task_id: int = None):
        self.model_name = model_name
        self.config_path = config_path
        self.task_id = task_id
        
        # Set up logging with task-specific file if possible
        self._setup_logging()
        
        # Initialize Distyl controller
        self.distyl_controller = None
        self._initialize_controller()
    
    def _setup_logging(self):
        """Set up Distyl logging with file output"""
        import time
        from distyl_webarena.utils.logging import configure_logging
        
        # Create logs directory if it doesn't exist
        os.makedirs("distyl_logs", exist_ok=True)
        
        # Set up timestamped log file
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        task_suffix = f"_task_{self.task_id}" if self.task_id else ""
        log_file = f"distyl_logs/distyl_webarena_{timestamp}{task_suffix}.log"
        
        # Configure global Distyl logging FIRST
        configure_logging(
            log_level="DEBUG",  # Changed from INFO to DEBUG
            log_directory="distyl_logs",
            enable_file_logging=True
        )
        
        # Force DEBUG level for Python's root logging module
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("distyl_webarena").setLevel(logging.DEBUG)
        
        # Create logger for this adapter
        self.logger = DistylLogger("WebArenaAdapter", log_file=log_file)
        
        # Store log file path for later reference
        self.log_file_path = log_file
    
    def _initialize_controller(self):
        """Initialize the Distyl WebArena controller"""
        
        try:
            # Default engine parameters for Distyl-WebArena
            # Hardcode to use GPT-4o for internal LLM calls
            engine_params = {
                'model': 'gpt-4o-2024-08-06',  # Use GPT-4o for internal planning
                'temperature': 0.1,
                'max_tokens': 2048,
                'provider': 'openai',
                'distyl_model_name': self.model_name  # Keep original model name for identification
            }
            
            # Load custom config if provided
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    custom_config = json.load(f)
                    engine_params.update(custom_config)
            
            self.distyl_controller = DistylWebArenaController(
                engine_params=engine_params,
                memory_folder_name="distyl_webarena/memory",
                enable_reflection=True,
                enable_web_knowledge=True,
                log_file=getattr(self, 'log_file_path', None)
            )
            
            self.logger.info("Distyl WebArena controller initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Distyl controller: {e}")
            raise
    
    def get_agent_for_webarena(self):
        """
        Get the agent instance compatible with WebArena's browser_env/run.py
        
        Returns the DistylWebArenaController which implements the Agent interface
        """
        return self.distyl_controller
    
    def get_log_file_path(self):
        """Get the path to the Distyl log file for this adapter"""
        return getattr(self, 'log_file_path', None)
    
    def create_agent_config(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create agent configuration compatible with WebArena system
        
        Args:
            task_config: Task configuration from config_files/*.json
            
        Returns:
            Agent configuration for browser_env/run.py
        """
        
        # Extract key information from task config
        task_id = task_config.get("task_id", "unknown")
        sites = task_config.get("sites", [])
        require_login = task_config.get("require_login", False)
        
        agent_config = {
            "agent_type": "distyl_webarena",
            "model": self.model_name,
            "task_id": task_id,
            "sites": sites,
            "require_login": require_login,
            "memory_enabled": True,
            "reflection_enabled": True,
            "max_steps": 50,
            "timeout": 600  # 10 minutes
        }
        
        # Add site-specific configurations
        if "shopping" in sites or "shopping_admin" in sites:
            agent_config["shopping_mode"] = True
            agent_config["admin_access"] = "shopping_admin" in sites
        
        if "reddit" in sites:
            agent_config["social_mode"] = True
        
        if "gitlab" in sites:
            agent_config["development_mode"] = True
        
        if "wikipedia" in sites:
            agent_config["knowledge_mode"] = True
        
        self.logger.debug(f"Created agent config for task {task_id}: {agent_config}")
        return agent_config
    
    def prepare_for_parallel_execution(self, task_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare Distyl-WebArena for parallel execution
        
        Args:
            task_configs: List of task configurations
            
        Returns:
            Preparation summary
        """
        
        # Analyze task distribution
        task_count = len(task_configs)
        site_distribution = {}
        
        for config in task_configs:
            sites = config.get("sites", [])
            for site in sites:
                site_distribution[site] = site_distribution.get(site, 0) + 1
        
        # Prepare memory system for parallel execution
        if self.distyl_controller and hasattr(self.distyl_controller, 'memory'):
            # Pre-load any existing knowledge
            self.distyl_controller.memory.initialize_task_trajectory(
                task_id="parallel_batch",
                intent="Parallel execution batch",
                sites=list(site_distribution.keys())
            )
        
        preparation_summary = {
            "total_tasks": task_count,
            "site_distribution": site_distribution,
            "controller_ready": self.distyl_controller is not None,
            "memory_initialized": True,
            "parallel_compatible": True
        }
        
        self.logger.info(f"Prepared for parallel execution: {preparation_summary}")
        return preparation_summary
    
    def post_execution_cleanup(self, results: Dict[str, Any]):
        """
        Perform cleanup after parallel execution
        
        Args:
            results: Execution results from run_parallel_distyl.py
        """
        
        try:
            # Save execution statistics
            if self.distyl_controller and hasattr(self.distyl_controller, 'memory'):
                # Save memory state
                memory_stats = self.distyl_controller.memory.get_memory_stats()
                self.logger.info(f"Final memory stats: {memory_stats}")
            
            # Log overall performance
            success_rate = results.get("summary", {}).get("success_rate", 0.0)
            total_tasks = results.get("summary", {}).get("total_tasks", 0)
            
            self.logger.info(f"Parallel execution completed: {total_tasks} tasks, {success_rate:.2%} success rate")
            
        except Exception as e:
            self.logger.error(f"Error in post-execution cleanup: {e}")
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about the adapter"""
        
        return {
            "adapter_version": "1.0.0",
            "model_name": self.model_name,
            "controller_initialized": self.distyl_controller is not None,
            "features": [
                "hierarchical_planning",
                "accessibility_tree_grounding", 
                "site_specific_actions",
                "reflection_and_error_recovery",
                "episodic_and_narrative_memory",
                "parallel_execution_compatible"
            ],
            "supported_sites": [
                "shopping",
                "shopping_admin", 
                "reddit",
                "gitlab",
                "wikipedia",
                "map"
            ]
        }


def create_distyl_agent_for_webarena(task_config: Dict[str, Any], 
                                   model_name: str = "distyl-webarena") -> Any:
    """
    Factory function to create Distyl-WebArena agent for use with browser_env/run.py
    
    This function can be imported and used directly by the WebArena system:
    
    from distyl_webarena.integration.webarena_adapter import create_distyl_agent_for_webarena
    agent = create_distyl_agent_for_webarena(task_config, "distyl-webarena")
    
    Args:
        task_config: Task configuration from config_files/*.json
        model_name: Model name for the agent
        
    Returns:
        Agent instance compatible with WebArena's Agent interface
    """
    
    # Extract task_id from config for logging
    task_id = task_config.get("task_id", None)
    
    adapter = WebArenaAdapter(model_name=model_name, task_id=task_id)
    
    # Store adapter reference in the controller for log access
    controller = adapter.get_agent_for_webarena()
    controller._adapter = adapter  # Store reference for log file access
    
    # Return the controller which implements the Agent interface
    return controller


def get_distyl_model_info() -> Dict[str, Any]:
    """
    Get model information for integration with run_parallel_distyl.py
    
    This allows run_parallel_distyl.py to recognize and use Distyl-WebArena as a model option
    """
    
    return {
        "model_name": "distyl-webarena",
        "provider": "distyl",
        "description": "Distyl WebArena Agent with hierarchical planning and reflection",
        "supported_sites": ["shopping", "shopping_admin", "reddit", "gitlab", "wikipedia", "map"],
        "features": [
            "accessibility_tree_grounding",
            "hierarchical_planning", 
            "memory_system",
            "reflection_agent",
            "parallel_execution"
        ],
        "factory_function": "create_distyl_agent_for_webarena",
        "module_path": "distyl_webarena.integration.webarena_adapter"
    }


# For direct integration testing
if __name__ == "__main__":
    # Test the adapter
    print("Testing Distyl-WebArena Adapter...")
    
    # Sample task config
    sample_task = {
        "task_id": 78,
        "sites": ["shopping_admin"],
        "require_login": True,
        "intent": "What is the total count of Approved reviews amongst all the reviews?"
    }
    
    try:
        adapter = WebArenaAdapter("distyl-webarena-test")
        agent_config = adapter.create_agent_config(sample_task)
        agent = adapter.get_agent_for_webarena()
        
        print(f"✓ Adapter created successfully")
        print(f"✓ Agent config: {agent_config}")
        print(f"✓ Agent type: {type(agent).__name__}")
        print(f"✓ Adapter info: {adapter.get_adapter_info()}")
        
    except Exception as e:
        print(f"✗ Adapter test failed: {e}")