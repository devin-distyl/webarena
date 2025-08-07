"""Simplified script to run WebArena evaluation"""
import os
# Set environment variables to suppress warnings before importing other modules
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import List

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")
warnings.filterwarnings("ignore", message=".*beartype.*")

from agent import construct_agent
from browser_env import ScriptBrowserEnv, create_stop_action, ActionTypes
from browser_env.helper_functions import get_action_description
from evaluation_harness import evaluator_router
from debug_report import DebugHelper


class WebArenaRunner:
    """Simplified runner for WebArena evaluation"""
    
    def __init__(self, config_path: str, result_dir: str = None, debug_mode: bool = True):
        self.config_path = config_path
        self.result_dir = result_dir or f"results_{int(time.time())}"
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        self.debug_mode = debug_mode
        
        # Setup logging and debugging
        self._setup_logging()
        self.debug_helper = DebugHelper(self.result_dir, enable_screenshots=debug_mode)
        
    def _setup_logging(self):
        """Setup logging with info level by default"""
        log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{self.result_dir}/run.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set specific loggers to WARNING to reduce noise
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    def _ensure_auth(self):
        """Ensure authentication files are present and valid"""
        import subprocess
        from pathlib import Path
        
        auth_dir = Path(".auth")
        if not auth_dir.exists() or not list(auth_dir.glob("*.json")):
            self.logger.info("Setting up authentication...")
            try:
                subprocess.run(["python", "browser_env/auto_login.py", "--site_list", "shopping_admin"], 
                              check=True, capture_output=True, text=True)
                self.logger.info("Authentication setup completed")
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Auth setup failed: {e}")
            except Exception as e:
                self.logger.warning(f"Could not set up auth automatically: {e}")
    
    def run_single_test(self, config_file: str, agent, env) -> float:
        """Run a single test case"""
        try:
            # Load config
            with open(config_file) as f:
                config = json.load(f)
            
            intent = config["intent"]
            task_id = config["task_id"]
            
            self.logger.info(f"Running task {task_id}: {intent}")
            
            # Reset agent and environment
            agent.reset(config_file)
            obs, info = env.reset(options={"config_file": config_file})
            
            # Initialize trajectory
            trajectory = [{"observation": obs, "info": info}]
            meta_data = {"action_history": ["None"]}
            
            # Main interaction loop
            max_steps = 5
            for step in range(max_steps):
                # Take screenshot
                self.debug_helper.take_screenshot(env.page, step, task_id)
                
                # Get next action from agent
                try:
                    self.logger.info(f"Calling agent.next_action for step {step}")
                    action = agent.next_action(trajectory, intent, meta_data)
                    self.logger.info(f"Agent returned action: {action.get('action_type', 'unknown')}")
                except Exception as e:
                    action = create_stop_action(f"ERROR: {str(e)}")
                    self.logger.error(f"Agent error at step {step}: {e}")
                    import traceback
                    self.logger.error(f"Full traceback: {traceback.format_exc()}")
                
                trajectory.append(action)
                
                # Log action details
                action_str = get_action_description(
                    action, 
                    trajectory[-2]["info"]["observation_metadata"],
                    action_set_tag="id_accessibility_tree", 
                    prompt_constructor=None
                )
                self.logger.info(f"Step {step}: {action_str}")
                meta_data["action_history"].append(action_str)
                
                # Check if done
                if action["action_type"] == ActionTypes.STOP:
                    break
                
                # Execute action
                obs, _, terminated, _, info = env.step(action)
                trajectory.append({"observation": obs, "info": info})
                
                if terminated:
                    trajectory.append(create_stop_action("Task completed"))
                    break
            
            # If we exit loop without stopping, add a final stop action
            if len(trajectory) == 0 or trajectory[-1].get("action_type") != ActionTypes.STOP:
                trajectory.append(create_stop_action("Max steps reached"))
            
            # Evaluate
            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )
            
            self.logger.info(f"Task {task_id}: {'PASS' if score == 1 else 'FAIL'} (score: {score})")
            return score
            
        except Exception as e:
            self.logger.error(f"Error in task {task_id}: {e}")
            return 0.0
    
    def run_evaluation(self, test_files: List[str], agent_type: str = "prompt", 
                      model: str = "gpt-4o", instruction_path: str = None, headless: bool = False):
        """Run evaluation on multiple test files"""
        
        # Ensure auth is set up
        self._ensure_auth()
        
        slow_mo = 300 if not headless else 0  # 1 second delay when visible
        
        # Setup environment
        env = ScriptBrowserEnv(
            headless=headless,
            slow_mo=slow_mo,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720},
        )
        
        # Setup agent
        agent_args = argparse.Namespace(
            agent_type=agent_type,
            provider="openai",
            model=model,
            instruction_path=instruction_path or "agent/prompts/jsons/p_cot_id_actree_2s.json",
            action_set_tag="id_accessibility_tree",
            max_retry=1,
            max_obs_length=1920,
            temperature=1.0,
            max_tokens=384,
            mode="chat"
        )
        
        agent = construct_agent(agent_args)
        
        # Run tests
        scores = []
        for config_file in test_files:
            score = self.run_single_test(config_file, agent, env)
            scores.append(score)
        
        # Report results
        avg_score = sum(scores) / len(scores) if scores else 0
        self.logger.info(f"Average score: {avg_score:.3f} ({sum(scores)}/{len(scores)} passed)")
        
        # Save results
        with open(f"{self.result_dir}/results.json", "w") as f:
            json.dump({
                "scores": scores,
                "average_score": avg_score,
                "total_tests": len(scores),
                "passed": sum(scores)
            }, f, indent=2)
        
        env.close()
        
        # Generate debug report if in debug mode
        if self.debug_mode:
            try:
                from debug_report import DebugReportGenerator
                generator = DebugReportGenerator(self.result_dir)
                report_path = generator.generate_html_report()
                self.logger.info(f"Debug report generated: {report_path}")
            except Exception as e:
                self.logger.warning(f"Failed to generate debug report: {e}")
        
        return avg_score


def main():
    """Main function with simplified argument parsing"""
    parser = argparse.ArgumentParser(description="Run WebArena evaluation")
    parser.add_argument("--config_dir", default="config_files", help="Directory with test configs")
    parser.add_argument("-t", "--test_indices", default="0,1,2,3,4", help="Comma-separated list of test indices (e.g., '0,1,2,3,4')")
    parser.add_argument("--agent_type", default="prompt", choices=["prompt", "teacher_forcing", "distyl"])
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--instruction_path", help="Path to instruction file")
    parser.add_argument("--result_dir", default="results", help="Result directory")
    parser.add_argument('-v', "--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug mode")
    
    args = parser.parse_args()
    
    # Parse test indices
    try:
        test_indices = [int(idx.strip()) for idx in args.test_indices.split(",")]
    except ValueError:
        print(f"Error: Invalid test indices format. Use comma-separated numbers like '0,1,2,3,4'")
        return
    
    # Generate test file list
    test_files = []
    for i in test_indices:
        config_file = f"{args.config_dir}/{i}.json"
        if os.path.exists(config_file):
            test_files.append(config_file)
        else:
            print(f"Warning: Config file {config_file} not found, skipping...")
    
    if not test_files:
        print(f"No test files found for indices: {args.test_indices}")
        return
    
    print(f"Debug mode: {'enabled' if not args.no_debug else 'disabled'}")
    
    # Run evaluation
    runner = WebArenaRunner(args.config_dir, args.result_dir, debug_mode=not args.no_debug)
    runner.run_evaluation(
        test_files=test_files,
        agent_type=args.agent_type,
        model=args.model,
        instruction_path=args.instruction_path,
        headless=args.headless
    )


if __name__ == "__main__":
    main() 