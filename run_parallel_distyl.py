#!/usr/bin/env python3
"""
Distyl-WebArena Parallel Runner

Universal WebArena parallel runner supporting both Distyl-WebArena and standard models.
Maintains full compatibility with the original while adding intelligent agent capabilities.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# Add current directory to Python path for imports
sys.path.insert(0, os.getcwd())

# Environment variables should be set externally:
# export SHOPPING="<your_shopping_site_domain>:7770"
# export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
# export REDDIT="<your_reddit_domain>:9999"
# export GITLAB="<your_gitlab_domain>:8023"
# export MAP="<your_map_domain>:3000"
# export WIKIPEDIA="<your_wikipedia_domain>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
# export HOMEPAGE="<your_homepage_domain>:4399"

# Don't import Distyl-WebArena here - use lazy loading instead
DISTYL_AVAILABLE = None  # Will be determined when needed
_distyl_imports = None  # Cache for distyl imports

# Import existing WebArena components (when available)
try:
    from browser_env.docker_isolation_manager import (
        DockerIsolationManager,
    )

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


def check_distyl_availability():
    """Check if Distyl-WebArena is available (lazy loading)"""
    global DISTYL_AVAILABLE, _distyl_imports

    if DISTYL_AVAILABLE is not None:
        return DISTYL_AVAILABLE, _distyl_imports

    try:
        # Try to import Distyl-WebArena components
        from distyl_webarena.integration.webarena_adapter import (
            create_distyl_agent_for_webarena,
            get_distyl_model_info,
        )

        DISTYL_AVAILABLE = True
        _distyl_imports = {
            "create_distyl_agent_for_webarena": create_distyl_agent_for_webarena,
            "get_distyl_model_info": get_distyl_model_info,
        }
        print("✓ Distyl-WebArena loaded successfully")
        return True, _distyl_imports

    except ImportError as e:
        DISTYL_AVAILABLE = False
        _distyl_imports = None
        print(f"Warning: Distyl-WebArena not available: {e}")
        return False, None


def copy_distyl_logs_to_results(task_id: int, result_dir: str) -> Dict[str, str]:
    """
    Copy Distyl log files to the task result directory
    
    Args:
        task_id: Task identifier
        result_dir: Directory where task results are stored
        
    Returns:
        Dict with log file paths that were copied
    """
    import glob
    
    copied_logs = {}
    
    try:
        # Look for Distyl log files for this task
        log_pattern = f"distyl_logs/distyl_webarena_*_task_{task_id}.log"
        matching_logs = glob.glob(log_pattern)
        
        # Also look for general Distyl logs from around the same time
        if not matching_logs:
            # Get all recent Distyl logs and find ones that might be for this task
            all_distyl_logs = glob.glob("distyl_logs/distyl_webarena_*.log")
            # Sort by modification time and take the most recent ones
            all_distyl_logs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            matching_logs = all_distyl_logs[:2]  # Take the 2 most recent logs
        
        for log_file in matching_logs:
            if os.path.exists(log_file):
                # Copy to result directory
                log_filename = os.path.basename(log_file)
                dest_path = os.path.join(result_dir, f"distyl_{log_filename}")
                
                import shutil
                shutil.copy2(log_file, dest_path)
                copied_logs[log_filename] = dest_path
                
        # Also create a distyl_logs.txt file similar to log_files.txt
        if copied_logs:
            distyl_log_list_path = os.path.join(result_dir, "distyl_logs.txt")
            with open(distyl_log_list_path, "w") as f:
                for log_filename in copied_logs.keys():
                    f.write(f"distyl_logs/{log_filename}\n")
                    
    except Exception as e:
        print(f"Warning: Failed to copy Distyl logs for task {task_id}: {e}")
    
    return copied_logs


def parse_evaluation_results(
    stdout: str, stderr: str
) -> Tuple[float, bool, str]:
    """
    Parse evaluation results from stdout/stderr

    Returns:
        Tuple of (score, task_success, eval_type)
        - score: The evaluation score (0.0 to 1.0)
        - task_success: Whether the task passed evaluation
        - eval_type: PASS/FAIL/ERROR
    """
    score = 0.0
    task_success = False
    eval_type = "ERROR"

    # Parse score from "Average score: X.X" line
    score_match = re.search(r"Average score:\s*([0-9.]+)", stdout)
    if score_match:
        try:
            score = float(score_match.group(1))
        except ValueError:
            pass

    # Parse pass/fail from "[Result] (PASS)" or "[Result] (FAIL)" lines
    if re.search(r"\[Result\]\s*\(PASS\)", stdout):
        task_success = True
        eval_type = "PASS"
    elif re.search(r"\[Result\]\s*\(FAIL\)", stdout):
        task_success = False
        eval_type = "FAIL"

    # Additional check - if score is 1.0, it's a pass
    if score >= 1.0:
        task_success = True
        eval_type = "PASS"

    return score, task_success, eval_type


def get_provider_from_model(model_name: str) -> str:
    """Determine provider from model name"""
    model_lower = model_name.lower()

    # Distyl-WebArena models (check name pattern even if not available)
    if model_lower.startswith("distyl"):
        return "distyl"

    # OpenAI models
    if any(name in model_lower for name in ["gpt", "o1"]):
        return "openai"

    # Google models
    if any(name in model_lower for name in ["gemini", "bard"]):
        return "google"

    # Anthropic models
    if any(name in model_lower for name in ["claude"]):
        return "anthropic"

    return "unknown"


def validate_environment():
    """Validate that required environment variables are set"""

    required_urls = [
        "REDDIT",
        "SHOPPING",
        "SHOPPING_ADMIN",
        "GITLAB",
        "WIKIPEDIA",
        "MAP",
    ]
    missing_urls = []

    for url_name in required_urls:
        if not os.getenv(url_name):
            missing_urls.append(url_name)

    # Check HOMEPAGE separately - warn but don't fail
    if not os.getenv("HOMEPAGE"):
        print("Warning: HOMEPAGE environment variable not set")

    if missing_urls:
        # Try to load from .env file if variables are missing
        env_file_path = ".env"
        if os.path.exists(env_file_path):
            print(f"Loading environment variables from {env_file_path}")
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file_path)
                
                # Re-check after loading .env
                still_missing = []
                for url_name in missing_urls:
                    if not os.getenv(url_name):
                        still_missing.append(url_name)
                
                if not still_missing:
                    print("✅ All required environment variables loaded from .env file")
                    return
                else:
                    missing_urls = still_missing
                    
            except ImportError:
                print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
            except Exception as e:
                print(f"Warning: Failed to load .env file: {e}")
        
        raise EnvironmentError(
            f"Required environment variables not set: {', '.join(missing_urls)}\n"
            f"Please export these variables or ensure they are in your .env file:\n"
            f'export SHOPPING="<your_shopping_site_domain>:7770"\n'
            f'export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"\n'
            f'export REDDIT="<your_reddit_domain>:9999"\n'
            f'export GITLAB="<your_gitlab_domain>:8023"\n'
            f'export MAP="<your_map_domain>:3000"\n'
            f'export WIKIPEDIA="<your_wikipedia_domain>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"\n'
            f'export HOMEPAGE="<your_homepage_domain>:4399"'
        )


def load_task_config(task_id: int) -> Dict[str, Any]:
    """Load task configuration from config file"""

    config_path = f"config_files/{task_id}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Task config not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def run_task_with_distyl(
    task_id: int, model_name: str, docker_manager=None, task_config=None
) -> Dict[str, Any]:
    """Run a single task using Distyl-WebArena"""

    config_backup_path = None

    try:
        # Use provided task_config or load it
        if task_config is None:
            task_config = load_task_config(task_id)

        # Update with isolated URLs if Docker isolation is available
        if docker_manager:
            try:
                # Get modified configuration for isolated environment
                modified_config = docker_manager.get_environment_config(
                    task_id, task_config
                )

                # Backup original config file
                original_config_path = f"config_files/{task_id}.json"
                config_backup_path = (
                    f"config_files/{task_id}.json.backup_{int(time.time())}"
                )

                import shutil

                shutil.copy2(original_config_path, config_backup_path)

                # Write modified config to original location (temporarily)
                with open(original_config_path, "w") as f:
                    json.dump(modified_config, f, indent=2)

                # Use the modified config
                task_config = modified_config

            except Exception as e:
                print(f"Warning: Failed to get Docker environment config: {e}")
                # Continue without Docker isolation

        # Set up environment with task URLs
        env = os.environ.copy()

        # Add task-specific URLs to environment
        for key, value in task_config.items():
            if key.upper() in [
                "REDDIT",
                "SHOPPING",
                "SHOPPING_ADMIN",
                "GITLAB",
                "WIKIPEDIA",
                "MAP",
                "HOMEPAGE",
            ]:
                env[key.upper()] = str(value)

        # Use virtual environment Python if available
        venv_python = "webarena-env/bin/python"
        python_cmd = (
            venv_python if os.path.exists(venv_python) else sys.executable
        )

        # Create result directory (will be moved to proper location after execution)
        temp_result_dir = f"cache/results_{task_id}_{int(time.time())}"
        os.makedirs(temp_result_dir, exist_ok=True)

        # Run browser_env/run.py with distyl provider
        cmd = [
            python_cmd,
            "browser_env/run.py",
            "--instruction_path",
            "agent/prompts/jsons/p_cot_id_actree_2s.json",
            "--test_start_idx",
            str(task_id),
            "--test_end_idx",
            str(task_id + 1),
            "--provider",
            "distyl",
            "--model",
            model_name,
            "--result_dir",
            temp_result_dir,
            "--save_trace_enabled",
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600
        )
        execution_time = time.time() - start_time
        
        # Debug: Print subprocess output to see our debug messages
        print(f"🔥 SUBPROCESS STDOUT for task {task_id}:")
        print(result.stdout)
        print(f"🔥 SUBPROCESS STDERR for task {task_id}:")
        print(result.stderr)

        # Parse evaluation results from output
        score, task_success, eval_type = parse_evaluation_results(
            result.stdout, result.stderr
        )

        # Copy Distyl log files to result directory if they exist
        distyl_log_info = copy_distyl_logs_to_results(task_id, temp_result_dir)

        return {
            "task_id": task_id,
            "process_success": result.returncode
            == 0,  # Did the script run without crashing
            "task_success": task_success,  # Did the agent complete the task successfully
            "success": result.returncode
            == 0,  # Keep for backward compatibility
            "score": score,  # Actual evaluation score
            "eval_type": eval_type,  # PASS/FAIL/ERROR
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "model": model_name,
            "provider": "distyl",
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "process_success": False,
            "task_success": False,
            "success": False,  # Keep for backward compatibility
            "score": 0.0,
            "eval_type": "ERROR",
            "error": str(e),
            "execution_time": 0,
            "stdout": "",
            "stderr": str(e),
            "model": model_name,
            "provider": "distyl",
        }
    finally:
        # Restore original config file if we backed it up
        if config_backup_path and os.path.exists(config_backup_path):
            try:
                original_config_path = f"config_files/{task_id}.json"
                import shutil

                shutil.copy2(config_backup_path, original_config_path)
                os.remove(config_backup_path)  # Remove backup after restoring
                print(f"✅ Restored config for task {task_id}")
            except Exception as cleanup_error:
                print(
                    f"⚠️ Config restore error for task {task_id}: {cleanup_error}"
                )


def run_task_standard(
    task_id: int,
    model_name: str,
    provider: str,
    docker_manager=None,
    task_config=None,
) -> Dict[str, Any]:
    """Run a single task using standard WebArena agents"""

    config_backup_path = None

    try:
        # Use provided task_config or load it
        if task_config is None:
            task_config = load_task_config(task_id)

        # Update with isolated URLs if Docker isolation is available
        if docker_manager:
            try:
                # Get modified configuration for isolated environment
                modified_config = docker_manager.get_environment_config(
                    task_id, task_config
                )

                # Backup original config file
                original_config_path = f"config_files/{task_id}.json"
                config_backup_path = (
                    f"config_files/{task_id}.json.backup_{int(time.time())}"
                )

                import shutil

                shutil.copy2(original_config_path, config_backup_path)

                # Write modified config to original location (temporarily)
                with open(original_config_path, "w") as f:
                    json.dump(modified_config, f, indent=2)

                # Use the modified config
                task_config = modified_config

            except Exception as e:
                print(f"Warning: Failed to get Docker environment config: {e}")
                # Continue without Docker isolation

        # Set up environment with task URLs
        env = os.environ.copy()

        # Add task-specific URLs to environment
        for key, value in task_config.items():
            if key.upper() in [
                "REDDIT",
                "SHOPPING",
                "SHOPPING_ADMIN",
                "GITLAB",
                "WIKIPEDIA",
                "MAP",
                "HOMEPAGE",
            ]:
                env[key.upper()] = str(value)

        # Use virtual environment Python if available
        venv_python = "webarena-env/bin/python"
        python_cmd = (
            venv_python if os.path.exists(venv_python) else sys.executable
        )

        # Create result directory (will be moved to proper location after execution)
        temp_result_dir = f"cache/results_{task_id}_{int(time.time())}"
        os.makedirs(temp_result_dir, exist_ok=True)

        # Run browser_env/run.py with correct arguments
        cmd = [
            python_cmd,
            "browser_env/run.py",
            "--instruction_path",
            "agent/prompts/jsons/p_cot_id_actree_2s.json",
            "--test_start_idx",
            str(task_id),
            "--test_end_idx",
            str(task_id + 1),
            "--provider",
            provider,
            "--model",
            model_name,
            "--result_dir",
            temp_result_dir,
            "--save_trace_enabled",
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600
        )
        execution_time = time.time() - start_time

        # Parse evaluation results from output
        score, task_success, eval_type = parse_evaluation_results(
            result.stdout, result.stderr
        )

        return {
            "task_id": task_id,
            "process_success": result.returncode
            == 0,  # Did the script run without crashing
            "task_success": task_success,  # Did the agent complete the task successfully
            "success": result.returncode
            == 0,  # Keep for backward compatibility
            "score": score,  # Actual evaluation score
            "eval_type": eval_type,  # PASS/FAIL/ERROR
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "model": model_name,
            "provider": provider,
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "process_success": False,
            "task_success": False,
            "success": False,  # Keep for backward compatibility
            "score": 0.0,
            "eval_type": "ERROR",
            "error": str(e),
            "execution_time": 0,
            "stdout": "",
            "stderr": str(e),
            "model": model_name,
            "provider": provider,
        }
    finally:
        # Restore original config file if we backed it up
        if config_backup_path and os.path.exists(config_backup_path):
            try:
                original_config_path = f"config_files/{task_id}.json"
                import shutil

                shutil.copy2(config_backup_path, original_config_path)
                os.remove(config_backup_path)  # Remove backup after restoring
                print(f"✅ Restored config for task {task_id}")
            except Exception as cleanup_error:
                print(
                    f"⚠️ Config restore error for task {task_id}: {cleanup_error}"
                )


def run_single_task(
    task_id: int, model_name: str, provider: str
) -> Dict[str, Any]:
    """Run a single task with the appropriate agent"""

    print(f"Running task {task_id} with {model_name} ({provider})")

    # Load task configuration to get sites for Docker isolation
    task_config = load_task_config(task_id)
    sites = task_config.get("sites", [])

    # Use Docker isolation if available
    docker_manager = None
    if DOCKER_AVAILABLE and sites:
        try:
            docker_manager = DockerIsolationManager()
            # Start isolated environment for this task
            env = docker_manager.start_environment(task_id, sites)
            if env is None:
                print(
                    f"Warning: Docker environment failed to start for task {task_id}"
                )
                docker_manager = None
            else:
                print(
                    f"✓ Docker isolation set up for task {task_id} with sites: {sites}"
                )
        except Exception as e:
            print(f"Warning: Docker isolation failed for task {task_id}: {e}")
            docker_manager = None

    try:
        # Route to appropriate agent
        if provider == "distyl":
            available, _ = check_distyl_availability()
            if available:
                return run_task_with_distyl(
                    task_id, model_name, docker_manager, task_config
                )
            else:
                print(
                    f"Warning: Distyl-WebArena not available, falling back to standard execution"
                )
                # Fallback to a reasonable standard model
                fallback_model = (
                    "gpt-4" if model_name == "distyl-webarena" else model_name
                )
                fallback_provider = get_provider_from_model(fallback_model)
                return run_task_standard(
                    task_id,
                    fallback_model,
                    fallback_provider,
                    docker_manager,
                    task_config,
                )
        else:
            return run_task_standard(
                task_id, model_name, provider, docker_manager, task_config
            )
    finally:
        # Clean up Docker environment
        if docker_manager:
            try:
                docker_manager.stop_environment(task_id)
                print(f"✓ Docker cleanup completed for task {task_id}")
            except Exception as e:
                print(
                    f"Warning: Docker cleanup failed for task {task_id}: {e}"
                )


def parse_task_list(task_string: str) -> List[int]:
    """Parse task string into list of task IDs"""

    tasks = []
    for part in task_string.split(","):
        part = part.strip()
        if "-" in part:
            # Handle ranges like "78-82"
            start, end = map(int, part.split("-"))
            tasks.extend(range(start, end + 1))
        else:
            # Handle single tasks
            tasks.append(int(part))

    return sorted(list(set(tasks)))  # Remove duplicates and sort


def save_results(
    results: List[Dict[str, Any]], model_name: str, provider: str
):
    """Save results to file"""

    # Create results directory in parallel_demo_results format
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Format: parallel_demo_results/20250731_134713_openai_gpt_4_1_2025_04_14
    formatted_model = model_name.replace("-", "_").replace(".", "_")
    results_dirname = f"{timestamp}_{provider}_{formatted_model}"
    results_dir = f"parallel_demo_results/{results_dirname}"
    os.makedirs(results_dir, exist_ok=True)

    # Create individual task directories and move results
    for result in results:
        task_id = result["task_id"]
        task_dir = os.path.join(results_dir, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        # Look for any temp result directories for this task and move them
        import glob

        temp_dirs = glob.glob(f"cache/results_{task_id}_*")
        for temp_dir in temp_dirs:
            # Move files from temp directory to task directory
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    src = os.path.join(temp_dir, file)
                    dst = os.path.join(task_dir, file)
                    if os.path.isfile(src):
                        try:
                            import shutil

                            shutil.move(src, dst)
                        except Exception as e:
                            print(
                                f"Warning: Could not move {src} to {dst}: {e}"
                            )
                # Remove empty temp directory
                try:
                    os.rmdir(temp_dir)
                except:
                    pass

        # Create config.json for this task
        config_path = f"config_files/{task_id}.json"
        if os.path.exists(config_path):
            import shutil

            shutil.copy2(config_path, os.path.join(task_dir, "config.json"))

    # Calculate summary statistics - now with both process and task success
    process_successful_tasks = [
        r for r in results if r.get("process_success", r.get("success", False))
    ]
    task_successful_tasks = [
        r for r in results if r.get("task_success", False)
    ]
    total_tasks = len(results)

    # Process success rate (no crashes)
    process_success_rate = (
        len(process_successful_tasks) / total_tasks if total_tasks > 0 else 0
    )

    # Task success rate (evaluation passed)
    task_success_rate = (
        len(task_successful_tasks) / total_tasks if total_tasks > 0 else 0
    )

    # Average score
    avg_score = (
        sum(r.get("score", 0.0) for r in results) / total_tasks
        if total_tasks > 0
        else 0
    )

    avg_execution_time = (
        sum(r.get("execution_time", 0) for r in results) / total_tasks
        if total_tasks > 0
        else 0
    )

    # Create detailed_results format that viewer expects
    detailed_results = []
    task_info = {}

    for result in results:
        task_id = result["task_id"]

        # Load task config to get intent and sites
        try:
            config_path = f"config_files/{task_id}.json"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    intent = config.get("intent", f"Task {task_id}")
                    sites = config.get("sites", [])
            else:
                intent = f"Task {task_id}"
                sites = []
        except:
            intent = f"Task {task_id}"
            sites = []

        # Create detailed result entry
        detailed_result = {
            "task_id": task_id,
            "success": result.get(
                "task_success", False
            ),  # Use task success for viewer
            "score": result.get("score", 0.0),
            "elapsed_time": result.get("execution_time", 0.0),
            "process_success": result.get("process_success", False),
            "eval_type": result.get("eval_type", "ERROR"),
        }
        detailed_results.append(detailed_result)

        # Task info
        task_info[str(task_id)] = {"intent": intent, "sites": sites}

    summary = {
        "timestamp": timestamp,
        "model": model_name,
        "provider": provider,
        "total_tasks": total_tasks,
        "successful_tasks": len(
            process_successful_tasks
        ),  # Keep for backward compatibility
        "process_successful_tasks": len(process_successful_tasks),
        "task_successful_tasks": len(task_successful_tasks),
        "success_rate": process_success_rate,  # Keep for backward compatibility
        "process_success_rate": process_success_rate,
        "task_success_rate": task_success_rate,
        "avg_score": avg_score,
        "average_execution_time": avg_execution_time,
        "results": results,
        "detailed_results": detailed_results,  # For viewer compatibility
        "task_info": task_info,  # For viewer compatibility
        "summary": {  # Additional summary for viewer
            "success_rate": task_success_rate,
            "avg_score": avg_score,
        },
    }

    # Save detailed results in expected format
    results_file = os.path.join(results_dir, "parallel_demo_results.json")
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Save README summary
    readme_file = os.path.join(results_dir, "README.md")
    with open(readme_file, "w") as f:
        f.write(f"# Distyl-WebArena Parallel Execution Results\n\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Provider:** {provider}\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Total Tasks:** {total_tasks}\n\n")

        f.write("## Success Metrics\n\n")
        f.write(
            f"**Process Success (no crashes):** {len(process_successful_tasks)}/{total_tasks} ({process_success_rate:.2%})\n"
        )
        f.write(
            f"**Task Success (evaluation passed):** {len(task_successful_tasks)}/{total_tasks} ({task_success_rate:.2%})\n"
        )
        f.write(f"**Average Score:** {avg_score:.3f}\n")
        f.write(f"**Average Execution Time:** {avg_execution_time:.2f}s\n\n")

        f.write("## Task Results\n\n")
        for result in results:
            # Show both types of status
            process_status = (
                "✅" if result.get("process_success", False) else "❌"
            )
            task_status = "✅" if result.get("task_success", False) else "❌"
            score = result.get("score", 0.0)
            eval_type = result.get("eval_type", "ERROR")

            f.write(
                f"- **Task {result['task_id']}:** Process {process_status} | Task {task_status} | "
                f"Score: {score:.3f} | {eval_type} | Time: {result.get('execution_time', 0):.2f}s\n"
            )

        f.write(f"\n## Architecture\n\n")
        f.write(
            f"This experiment used Distyl-WebArena with hierarchical planning:\n"
        )
        f.write(
            f"- **Hierarchical Planning:** Multi-step task decomposition with DAG generation\n"
        )
        f.write(
            f"- **Accessibility Tree Grounding:** Intelligent element detection and selection\n"
        )
        f.write(
            f"- **Reflection & Error Recovery:** Self-correction when actions fail\n"
        )
        f.write(
            f"- **Memory System:** Episodic and narrative memory for learning\n"
        )

    print(f"\nResults saved to: {results_dir}")
    print(
        f"Process Success (no crashes): {process_success_rate:.2%} ({len(process_successful_tasks)}/{total_tasks})"
    )
    print(
        f"Task Success (evaluation passed): {task_success_rate:.2%} ({len(task_successful_tasks)}/{total_tasks})"
    )
    print(f"Average Score: {avg_score:.3f}")

    return results_dir


def ensure_authentication_files(task_ids):
    """Ensure authentication files exist for all required sites"""
    print("🔑 Checking authentication files...")

    # Collect all unique sites needed across tasks
    required_sites = set()
    for task_id in task_ids:
        try:
            config_path = f"config_files/{task_id}.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            sites = config.get("sites", [])
            required_sites.update(sites)
        except Exception as e:
            print(f"   Warning: Could not load config for task {task_id}: {e}")

    if not required_sites:
        print("   No sites require authentication")
        return

    print(f"   Required sites: {', '.join(sorted(required_sites))}")

    # Ensure .auth directory exists
    from pathlib import Path

    auth_dir = Path(".auth")
    auth_dir.mkdir(exist_ok=True)

    # Check if we need to run auto_login
    missing_auth = False
    for site in required_sites:
        auth_file = auth_dir / f"{site}_state.json"
        if not auth_file.exists():
            missing_auth = True
            print(f"   Missing: {auth_file}")

    if missing_auth:
        print("   Running auto_login to generate authentication files...")
        try:
            result = subprocess.run(
                [
                    "python",
                    "browser_env/auto_login.py",
                    "--auth_folder",
                    str(auth_dir),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                print("   ✅ Authentication files generated successfully")
            else:
                print(
                    f"   ⚠️  Auto-login completed with warnings: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            print("   ⚠️  Auto-login timed out, but continuing...")
        except Exception as e:
            print(f"   ⚠️  Auto-login failed: {e}, but continuing...")
    else:
        print("   ✅ All authentication files already exist")


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Distyl-WebArena Parallel Runner"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="distyl-webarena",
        help="Model name (e.g., distyl-webarena, gpt-4) (default: distyl-webarena)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        help="Provider name (auto-detected from model if not specified)",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        default="0",
        help="Task IDs (e.g., 78 or 78,79,80 or 78-82) (default: 0)",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=4,
        help="Maximum parallel workers (default: 4)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # Handle list models
    if args.list_models:
        print("Available Models:")
        print("================")
        available, imports = check_distyl_availability()
        if available:
            get_distyl_model_info = imports["get_distyl_model_info"]
            distyl_info = get_distyl_model_info()
            print(f"Distyl-WebArena: {distyl_info['model_name']}")
            print(f"  Features: {', '.join(distyl_info['features'])}")
        print(
            "Standard WebArena models: gpt-4, gpt-3.5-turbo, gemini-1.5-pro, etc."
        )
        return 0

    # Validate environment
    validate_environment()

    # Determine provider
    provider = args.provider or get_provider_from_model(args.model)

    if provider == "distyl":
        available, _ = check_distyl_availability()
        if not available:
            print(
                "Warning: Distyl-WebArena not available, will fallback to standard execution"
            )

    # Parse tasks
    try:
        task_ids = parse_task_list(args.tasks)
        if not task_ids:
            print("❌ No valid task IDs provided")
            return 1
    except ValueError as e:
        print(f"❌ Error parsing task IDs: {e}")
        return 1

    print(f"🎯 Configuration:")
    print(f"   Model: {args.model}")
    print(
        f"   Provider: {provider} {'(auto-detected)' if not args.provider else ''}"
    )
    print(f"   Tasks: {task_ids}")
    print(f"   Max workers: {args.max_workers}")
    print()

    if provider == "distyl":
        available, _ = check_distyl_availability()
        if available:
            print(
                f"Using Distyl-WebArena with hierarchical planning and reflection"
            )
        else:
            print(f"Using standard execution (Distyl-WebArena not available)")

    # Ensure authentication files exist before starting
    print()
    ensure_authentication_files(task_ids)

    print("\nStarting parallel execution...")

    # Run tasks in parallel
    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                run_single_task, task_id, args.model, provider
            ): task_id
            for task_id in task_ids
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)

                # Display both types of success
                process_status = (
                    "✅" if result.get("process_success", False) else "❌"
                )
                task_status = "✅" if result.get("task_success", False) else "❌"
                score = result.get("score", 0.0)
                time_str = f"{result.get('execution_time', 0):.2f}s"
                print(
                    f"Task {task_id}: Process {process_status} | Task {task_status} | Score: {score:.3f} | Time: {time_str}"
                )

            except Exception as e:
                print(f"❌ Task {task_id} failed with exception: {e}")
                results.append(
                    {
                        "task_id": task_id,
                        "process_success": False,
                        "task_success": False,
                        "success": False,  # Keep for backward compatibility
                        "score": 0.0,
                        "eval_type": "ERROR",
                        "error": str(e),
                        "execution_time": 0,
                        "stdout": "",
                        "stderr": str(e),
                        "model": args.model,
                        "provider": provider,
                    }
                )

    total_time = time.time() - start_time

    # Save results
    results_dir = save_results(results, args.model, provider)

    print(f"\nExecution completed in {total_time:.2f}s")
    print(f"Results directory: {results_dir}")
    print(f"\n✅ Parallel execution demo completed successfully!")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
