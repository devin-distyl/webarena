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
        # Create timestamped results directory to avoid overwriting
        if result_dir is None:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.result_dir = f"results/{timestamp}"
        else:
            self.result_dir = result_dir
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        self.debug_mode = debug_mode
        
        # Setup logging and debugging
        self._setup_logging()
        self.debug_helper = DebugHelper(self.result_dir, enable_screenshots=debug_mode)
        
    def _setup_logging(self):
        """Setup logging with debug level to see LLM responses"""
        log_level = logging.DEBUG
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
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    
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
        """Run a single test case with enhanced debugging"""
        try:
            start_time = time.time()
            
            # Load config
            with open(config_file) as f:
                config = json.load(f)
            
            intent = config["intent"]
            task_id = config["task_id"]
            
            # Enhanced logging for task intent
            self.logger.info(f"="*80)
            self.logger.info(f"STARTING TASK {task_id}")
            self.logger.info(f"Intent: {intent}")
            self.logger.info(f"Start URL: {config.get('start_url', 'Unknown')}")
            self.logger.info(f"="*80)
            
            # Initialize detailed tracking
            detailed_steps = []
            performance_metrics = {
                "task_id": task_id,
                "intent": intent,
                "start_time": start_time,
                "step_timings": [],
                "agent_responses": [],
                "action_types": [],
                "errors": []
            }
            
            # Reset agent and environment
            agent.reset(config_file)
            obs, info = env.reset(options={"config_file": config_file})
            
            # Initialize trajectory
            trajectory = [{"observation": obs, "info": info}]
            meta_data = {"action_history": ["None"]}
            
            # Log initial state
            page_info = info.get('page')
            initial_url = page_info.url if page_info else 'Unknown'
            initial_title = env.page.title() if hasattr(env, 'page') and env.page else 'Unknown'
            self.logger.info(f"Initial URL: {initial_url}")
            self.logger.info(f"Initial Title: {initial_title}")
            
            # Main interaction loop
            max_steps = 5
            for step in range(max_steps):
                step_start_time = time.time()
                
                # Take screenshot
                self.debug_helper.take_screenshot(env.page, step, task_id)
                
                # Log detailed state information
                self.debug_helper.log_state(self.logger, step, obs, info, task_id)
                
                # Get next action from agent
                try:
                    self.logger.info(f"[STEP {step}] Calling agent.next_action...")
                    action = agent.next_action(trajectory, intent, meta_data)
                    
                    # Extract and log agent reasoning
                    agent_reasoning = action.get('raw_prediction', 'No reasoning available')
                    self.logger.info(f"[STEP {step}] Agent reasoning:")
                    self.logger.info("-" * 40)
                    self.logger.info(agent_reasoning)
                    self.logger.info("-" * 40)
                    
                    action_type = action.get('action_type', 'unknown')
                    self.logger.info(f"[STEP {step}] Agent decided: {action_type}")
                    
                    # Store for detailed tracking
                    performance_metrics["agent_responses"].append(agent_reasoning)
                    performance_metrics["action_types"].append(action_type)
                    
                except Exception as e:
                    error_msg = f"Agent error at step {step}: {e}"
                    action = create_stop_action(f"ERROR: {str(e)}")
                    self.logger.error(error_msg)
                    performance_metrics["errors"].append(error_msg)
                    import traceback
                    self.logger.error(f"Full traceback: {traceback.format_exc()}")
                
                trajectory.append(action)
                
                # Enhanced action logging
                action_str = get_action_description(
                    action, 
                    trajectory[-2]["info"]["observation_metadata"],
                    action_set_tag="id_accessibility_tree", 
                    prompt_constructor=None
                )
                self.logger.info(f"[STEP {step}] Action: {action_str}")
                self.debug_helper.log_action(self.logger, step, action, action_str, task_id)
                
                meta_data["action_history"].append(action_str)
                
                # Store detailed step information
                step_timing = time.time() - step_start_time
                performance_metrics["step_timings"].append(step_timing)
                
                detailed_step = {
                    "step": step,
                    "timestamp": time.time(),
                    "timing": step_timing,
                    "action": action,
                    "action_description": action_str,
                    "agent_reasoning": action.get('raw_prediction', ''),
                    "observation_length": len(str(obs)),
                    "url": env.page.url if hasattr(env, 'page') and env.page else '',
                    "title": env.page.title() if hasattr(env, 'page') and env.page else ''
                }
                detailed_steps.append(detailed_step)
                
                # Check if done
                if action["action_type"] == ActionTypes.STOP:
                    self.logger.info(f"[STEP {step}] Agent decided to STOP")
                    break
                
                # Execute action
                obs, _, terminated, _, info = env.step(action)
                trajectory.append({"observation": obs, "info": info})
                
                if terminated:
                    self.logger.info(f"[STEP {step}] Environment terminated")
                    trajectory.append(create_stop_action("Task completed"))
                    break
            
            # If we exit loop without stopping, add a final stop action
            if len(trajectory) == 0 or trajectory[-1].get("action_type") != ActionTypes.STOP:
                trajectory.append(create_stop_action("Max steps reached"))
                self.logger.info("Task reached maximum steps limit")
            
            # Evaluate
            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )
            
            # Complete performance metrics
            total_time = time.time() - start_time
            performance_metrics["total_time"] = total_time
            performance_metrics["final_score"] = score
            performance_metrics["total_steps"] = len(detailed_steps)
            performance_metrics["success"] = score == 1
            
            # Log final results
            result_status = 'PASS' if score == 1 else 'FAIL'
            self.logger.info(f"="*80)
            self.logger.info(f"TASK {task_id} COMPLETED: {result_status} (score: {score})")
            self.logger.info(f"Total time: {total_time:.2f}s")
            self.logger.info(f"Total steps: {len(detailed_steps)}")
            self.logger.info(f"Average step time: {sum(performance_metrics['step_timings'])/len(performance_metrics['step_timings']):.2f}s" if performance_metrics['step_timings'] else "No timing data")
            self.logger.info(f"="*80)
            
            # Save detailed trajectory and performance data
            self._save_detailed_results(task_id, detailed_steps, performance_metrics, trajectory, config)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error in task {task_id}: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return 0.0
    
    def _save_detailed_results(self, task_id: str, detailed_steps: list, performance_metrics: dict, trajectory: list, config: dict):
        """Save comprehensive debugging information"""
        try:
            # Create task-specific directory
            task_dir = Path(self.result_dir) / f"task_{task_id}"
            task_dir.mkdir(exist_ok=True)
            
            # Save detailed steps with agent reasoning
            steps_file = task_dir / "detailed_steps.json"
            with open(steps_file, 'w') as f:
                json.dump(detailed_steps, f, indent=2, default=str)
            
            # Save performance metrics
            metrics_file = task_dir / "performance_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(performance_metrics, f, indent=2, default=str)
            
            # Save complete trajectory
            trajectory_file = task_dir / "trajectory.json"
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory, f, indent=2, default=str)
            
            # Save original task config for reference
            config_file = task_dir / "task_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create human-readable summary
            summary_file = task_dir / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"TASK {task_id} DEBUGGING SUMMARY\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Intent: {config['intent']}\n")
                f.write(f"Start URL: {config.get('start_url', 'Unknown')}\n")
                f.write(f"Final Score: {performance_metrics['final_score']}\n")
                f.write(f"Success: {'YES' if performance_metrics['success'] else 'NO'}\n")
                f.write(f"Total Time: {performance_metrics['total_time']:.2f} seconds\n")
                f.write(f"Total Steps: {performance_metrics['total_steps']}\n\n")
                
                if performance_metrics['errors']:
                    f.write(f"ERRORS:\n")
                    for i, error in enumerate(performance_metrics['errors'], 1):
                        f.write(f"  {i}. {error}\n")
                    f.write("\n")
                
                f.write(f"STEP-BY-STEP BREAKDOWN:\n")
                f.write(f"{'-'*30}\n")
                for step_data in detailed_steps:
                    f.write(f"\nSTEP {step_data['step']}:\n")
                    f.write(f"  Action: {step_data['action_description']}\n")
                    f.write(f"  Timing: {step_data['timing']:.2f}s\n")
                    f.write(f"  URL: {step_data['url']}\n")
                    f.write(f"  Page Title: {step_data['title']}\n")
                    if step_data['agent_reasoning']:
                        f.write(f"  Agent Reasoning:\n")
                        # Indent the reasoning
                        reasoning_lines = step_data['agent_reasoning'].split('\n')
                        for line in reasoning_lines[:5]:  # Show first 5 lines to keep summary manageable
                            f.write(f"    {line}\n")
                        if len(reasoning_lines) > 5:
                            f.write(f"    ... (see detailed_steps.json for full reasoning)\n")
            
            self.logger.info(f"Detailed results saved to: {task_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save detailed results for task {task_id}: {e}")
    
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
        
        # Run tests with enhanced tracking
        evaluation_start_time = time.time()
        scores = []
        task_results = []
        
        self.logger.info(f"Starting evaluation of {len(test_files)} tasks...")
        
        for i, config_file in enumerate(test_files, 1):
            self.logger.info(f"Running task {i}/{len(test_files)}: {config_file}")
            task_start_time = time.time()
            score = self.run_single_test(config_file, agent, env)
            task_time = time.time() - task_start_time
            
            scores.append(score)
            
            # Load config for task info
            with open(config_file) as f:
                config = json.load(f)
            
            task_results.append({
                "config_file": config_file,
                "task_id": config["task_id"],
                "intent": config["intent"],
                "score": score,
                "success": score == 1,
                "execution_time": task_time
            })
        
        evaluation_total_time = time.time() - evaluation_start_time
        
        # Enhanced results reporting
        avg_score = sum(scores) / len(scores) if scores else 0
        pass_count = sum(scores)
        success_rate = (pass_count / len(scores) * 100) if scores else 0
        
        self.logger.info(f"="*80)
        self.logger.info(f"EVALUATION COMPLETED")
        self.logger.info(f"="*80)
        self.logger.info(f"Total tasks: {len(scores)}")
        self.logger.info(f"Passed: {int(pass_count)}")
        self.logger.info(f"Failed: {len(scores) - int(pass_count)}")
        self.logger.info(f"Success rate: {success_rate:.1f}%")
        self.logger.info(f"Average score: {avg_score:.3f}")
        self.logger.info(f"Total evaluation time: {evaluation_total_time:.2f} seconds")
        self.logger.info(f"Average time per task: {evaluation_total_time/len(scores):.2f} seconds" if scores else "N/A")
        
        # Show individual task results
        self.logger.info(f"\nINDIVIDUAL TASK RESULTS:")
        self.logger.info(f"-" * 40)
        for task_result in task_results:
            status = "PASS" if task_result["success"] else "FAIL"
            self.logger.info(f"Task {task_result['task_id']}: {status} ({task_result['execution_time']:.1f}s)")
        
        self.logger.info(f"="*80)
        
        # Save enhanced results
        results_data = {
            "evaluation_summary": {
                "total_tasks": len(scores),
                "passed": int(pass_count),
                "failed": len(scores) - int(pass_count),
                "success_rate": success_rate,
                "average_score": avg_score,
                "total_evaluation_time": evaluation_total_time,
                "average_time_per_task": evaluation_total_time/len(scores) if scores else 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "agent_type": agent_type,
                "model": model
            },
            "individual_results": task_results,
            "raw_scores": scores
        }
        
        with open(f"{self.result_dir}/results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
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
    parser.add_argument("--model", default="gpt-5-nano-2025-08-07", help="OpenAI model to use (e.g., gpt-4o, gpt-4o-mini, gpt-5-mini-2025-08-07)")
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
    print(f"Model: {args.model}")
    
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