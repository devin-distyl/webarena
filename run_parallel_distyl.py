#!/usr/bin/env python3
"""
Distyl-WebArena Parallel Runner

Enhanced version of run_parallel.py with Distyl-WebArena agent support.
Maintains full compatibility with the original while adding intelligent agent capabilities.
"""

import argparse
import json
import os
import sys
import time
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

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
    from browser_env.docker_isolation_manager import DockerIsolationManager
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
            get_distyl_model_info
        )
        DISTYL_AVAILABLE = True
        _distyl_imports = {
            'create_distyl_agent_for_webarena': create_distyl_agent_for_webarena,
            'get_distyl_model_info': get_distyl_model_info
        }
        print("✓ Distyl-WebArena loaded successfully")
        return True, _distyl_imports
        
    except ImportError as e:
        DISTYL_AVAILABLE = False
        _distyl_imports = None
        print(f"Warning: Distyl-WebArena not available: {e}")
        return False, None

def get_provider_from_model(model_name: str) -> str:
    """Determine provider from model name"""
    model_lower = model_name.lower()
    
    # Distyl-WebArena models (check name pattern even if not available)
    if model_lower.startswith('distyl'):
        return "distyl"
    
    # OpenAI models
    if any(name in model_lower for name in ['gpt', 'o1']):
        return "openai"
    
    # Google models
    if any(name in model_lower for name in ['gemini', 'bard']):
        return "google"
    
    # Anthropic models
    if any(name in model_lower for name in ['claude']):
        return "anthropic"
    
    return "unknown"

def validate_environment():
    """Validate that required environment variables are set"""
    
    required_urls = ['REDDIT', 'SHOPPING', 'SHOPPING_ADMIN', 'GITLAB', 'WIKIPEDIA', 'MAP']
    missing_urls = []
    
    for url_name in required_urls:
        if not os.getenv(url_name):
            missing_urls.append(url_name)
    
    # Check HOMEPAGE separately - warn but don't fail
    if not os.getenv('HOMEPAGE'):
        print("Warning: HOMEPAGE environment variable not set")
    
    if missing_urls:
        raise EnvironmentError(
            f"Required environment variables not set: {', '.join(missing_urls)}\n"
            f"Please export these variables:\n"
            f"export SHOPPING=\"<your_shopping_site_domain>:7770\"\n"
            f"export SHOPPING_ADMIN=\"<your_e_commerce_cms_domain>:7780/admin\"\n"
            f"export REDDIT=\"<your_reddit_domain>:9999\"\n"
            f"export GITLAB=\"<your_gitlab_domain>:8023\"\n"
            f"export MAP=\"<your_map_domain>:3000\"\n"
            f"export WIKIPEDIA=\"<your_wikipedia_domain>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing\"\n"
            f"export HOMEPAGE=\"<your_homepage_domain>:4399\""
        )

def load_task_config(task_id: int) -> Dict[str, Any]:
    """Load task configuration from config file"""
    
    config_path = f"config_files/{task_id}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Task config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def run_task_with_distyl(task_id: int, model_name: str, docker_manager = None, task_config = None) -> Dict[str, Any]:
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
                modified_config = docker_manager.get_environment_config(task_id, task_config)
                
                # Backup original config file
                original_config_path = f"config_files/{task_id}.json"
                config_backup_path = f"config_files/{task_id}.json.backup_{int(time.time())}"
                
                import shutil
                shutil.copy2(original_config_path, config_backup_path)
                
                # Write modified config to original location (temporarily)
                with open(original_config_path, 'w') as f:
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
            if key.upper() in ['REDDIT', 'SHOPPING', 'SHOPPING_ADMIN', 'GITLAB', 'WIKIPEDIA', 'MAP', 'HOMEPAGE']:
                env[key.upper()] = str(value)
        
        # Use virtual environment Python if available
        venv_python = 'webarena-env/bin/python'
        python_cmd = venv_python if os.path.exists(venv_python) else sys.executable
        
        # Create result directory (will be moved to proper location after execution)
        temp_result_dir = f"cache/results_{task_id}_{int(time.time())}"
        os.makedirs(temp_result_dir, exist_ok=True)
        
        # Run browser_env/run.py with distyl provider
        cmd = [
            python_cmd, 'browser_env/run.py',
            '--instruction_path', 'agent/prompts/jsons/p_cot_id_actree_2s.json',
            '--test_start_idx', str(task_id),
            '--test_end_idx', str(task_id + 1),
            '--provider', 'distyl',
            '--model', model_name,
            '--result_dir', temp_result_dir
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        execution_time = time.time() - start_time
        
        return {
            'task_id': task_id,
            'success': result.returncode == 0,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'model': model_name,
            'provider': 'distyl'
        }
            
    except Exception as e:
        return {
            'task_id': task_id,
            'success': False,
            'error': str(e),
            'execution_time': 0,
            'model': model_name,
            'provider': 'distyl'
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
                print(f"⚠️ Config restore error for task {task_id}: {cleanup_error}")

def run_task_standard(task_id: int, model_name: str, provider: str, docker_manager = None, task_config = None) -> Dict[str, Any]:
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
                modified_config = docker_manager.get_environment_config(task_id, task_config)
                
                # Backup original config file
                original_config_path = f"config_files/{task_id}.json"
                config_backup_path = f"config_files/{task_id}.json.backup_{int(time.time())}"
                
                import shutil
                shutil.copy2(original_config_path, config_backup_path)
                
                # Write modified config to original location (temporarily)
                with open(original_config_path, 'w') as f:
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
            if key.upper() in ['REDDIT', 'SHOPPING', 'SHOPPING_ADMIN', 'GITLAB', 'WIKIPEDIA', 'MAP', 'HOMEPAGE']:
                env[key.upper()] = str(value)
        
        # Use virtual environment Python if available
        venv_python = 'webarena-env/bin/python'
        python_cmd = venv_python if os.path.exists(venv_python) else sys.executable
        
        # Create result directory (will be moved to proper location after execution)
        temp_result_dir = f"cache/results_{task_id}_{int(time.time())}"
        os.makedirs(temp_result_dir, exist_ok=True)
        
        # Run browser_env/run.py with correct arguments
        cmd = [
            python_cmd, 'browser_env/run.py',
            '--instruction_path', 'agent/prompts/jsons/p_cot_id_actree_2s.json',
            '--test_start_idx', str(task_id),
            '--test_end_idx', str(task_id + 1),
            '--provider', provider,
            '--model', model_name,
            '--result_dir', temp_result_dir
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        execution_time = time.time() - start_time
        
        return {
            'task_id': task_id,
            'success': result.returncode == 0,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'model': model_name,
            'provider': provider
        }
            
    except Exception as e:
        return {
            'task_id': task_id,
            'success': False,
            'error': str(e),
            'execution_time': 0,
            'model': model_name,
            'provider': provider
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
                print(f"⚠️ Config restore error for task {task_id}: {cleanup_error}")

def run_single_task(task_id: int, model_name: str, provider: str) -> Dict[str, Any]:
    """Run a single task with the appropriate agent"""
    
    print(f"Running task {task_id} with {model_name} ({provider})")
    
    # Load task configuration to get sites for Docker isolation
    task_config = load_task_config(task_id)
    sites = task_config.get('sites', [])
    
    # Use Docker isolation if available
    docker_manager = None
    if DOCKER_AVAILABLE and sites:
        try:
            docker_manager = DockerIsolationManager()
            # Start isolated environment for this task
            env = docker_manager.start_environment(task_id, sites)
            if env is None:
                print(f"Warning: Docker environment failed to start for task {task_id}")
                docker_manager = None
            else:
                print(f"✓ Docker isolation set up for task {task_id} with sites: {sites}")
        except Exception as e:
            print(f"Warning: Docker isolation failed for task {task_id}: {e}")
            docker_manager = None
    
    try:
        # Route to appropriate agent
        if provider == "distyl":
            available, _ = check_distyl_availability()
            if available:
                return run_task_with_distyl(task_id, model_name, docker_manager, task_config)
            else:
                print(f"Warning: Distyl-WebArena not available, falling back to standard execution")
                # Fallback to a reasonable standard model
                fallback_model = "gpt-4" if model_name == "distyl-webarena" else model_name
                fallback_provider = get_provider_from_model(fallback_model)
                return run_task_standard(task_id, fallback_model, fallback_provider, docker_manager, task_config)
        else:
            return run_task_standard(task_id, model_name, provider, docker_manager, task_config)
    finally:
        # Clean up Docker environment
        if docker_manager:
            try:
                docker_manager.stop_environment(task_id)
                print(f"✓ Docker cleanup completed for task {task_id}")
            except Exception as e:
                print(f"Warning: Docker cleanup failed for task {task_id}: {e}")

def parse_task_list(task_string: str) -> List[int]:
    """Parse task string into list of task IDs"""
    
    tasks = []
    for part in task_string.split(','):
        part = part.strip()
        if '-' in part:
            # Handle ranges like "78-82"
            start, end = map(int, part.split('-'))
            tasks.extend(range(start, end + 1))
        else:
            # Handle single tasks
            tasks.append(int(part))
    
    return sorted(list(set(tasks)))  # Remove duplicates and sort

def save_results(results: List[Dict[str, Any]], model_name: str, provider: str):
    """Save results to file"""
    
    # Create results directory in parallel_demo_results format
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Format: parallel_demo_results/20250731_134713_openai_gpt_4_1_2025_04_14
    formatted_model = model_name.replace('-', '_').replace('.', '_')
    results_dirname = f"{timestamp}_{provider}_{formatted_model}"
    results_dir = f"parallel_demo_results/{results_dirname}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create individual task directories and move results
    for result in results:
        task_id = result['task_id']
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
                            print(f"Warning: Could not move {src} to {dst}: {e}")
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
    
    # Calculate summary statistics
    successful_tasks = [r for r in results if r.get('success', False)]
    total_tasks = len(results)
    success_rate = len(successful_tasks) / total_tasks if total_tasks > 0 else 0
    avg_execution_time = sum(r.get('execution_time', 0) for r in results) / total_tasks if total_tasks > 0 else 0
    
    summary = {
        'timestamp': timestamp,
        'model': model_name,
        'provider': provider,
        'total_tasks': total_tasks,
        'successful_tasks': len(successful_tasks),
        'success_rate': success_rate,
        'average_execution_time': avg_execution_time,
        'results': results
    }
    
    # Save detailed results in expected format
    results_file = os.path.join(results_dir, 'parallel_demo_results.json')
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save README summary
    readme_file = os.path.join(results_dir, 'README.md')
    with open(readme_file, 'w') as f:
        f.write(f"# Distyl-WebArena Parallel Execution Results\n\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Provider:** {provider}\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Total Tasks:** {total_tasks}\n")
        f.write(f"**Successful Tasks:** {len(successful_tasks)}\n")
        f.write(f"**Success Rate:** {success_rate:.2%}\n")
        f.write(f"**Average Execution Time:** {avg_execution_time:.2f}s\n\n")
        
        f.write("## Task Results\n\n")
        for result in results:
            status = "✅ SUCCESS" if result.get('success', False) else "❌ FAILED"
            f.write(f"- **Task {result['task_id']}:** {status} ({result.get('execution_time', 0):.2f}s)\n")
        
        f.write(f"\n## Architecture\n\n")
        f.write(f"This experiment used Distyl-WebArena with hierarchical planning:\n")
        f.write(f"- **Hierarchical Planning:** Multi-step task decomposition with DAG generation\n")
        f.write(f"- **Accessibility Tree Grounding:** Intelligent element detection and selection\n") 
        f.write(f"- **Reflection & Error Recovery:** Self-correction when actions fail\n")
        f.write(f"- **Memory System:** Episodic and narrative memory for learning\n")
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Success rate: {success_rate:.2%} ({len(successful_tasks)}/{total_tasks})")
    
    return results_dir

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="Distyl-WebArena Parallel Runner")
    parser.add_argument("--model", help="Model name (e.g., distyl-webarena, gpt-4)")
    parser.add_argument("--provider", help="Provider name (auto-detected from model if not specified)")
    parser.add_argument("--tasks", help="Task IDs (e.g., 78 or 78,79,80 or 78-82)")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum parallel workers")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # Handle list models
    if args.list_models:
        print("Available Models:")
        print("================")
        available, imports = check_distyl_availability()
        if available:
            get_distyl_model_info = imports['get_distyl_model_info']
            distyl_info = get_distyl_model_info()
            print(f"Distyl-WebArena: {distyl_info['model_name']}")
            print(f"  Features: {', '.join(distyl_info['features'])}")
        print("Standard WebArena models: gpt-4, gpt-3.5-turbo, gemini-1.5-pro, etc.")
        return
    
    # Validate required arguments
    if not args.model:
        print("Error: --model argument is required")
        print("Use --help for usage information")
        sys.exit(1)
    
    if not args.tasks:
        print("Error: --tasks argument is required")
        print("Use --help for usage information")
        sys.exit(1)
    
    # Validate environment
    validate_environment()
    
    # Determine provider
    provider = args.provider or get_provider_from_model(args.model)
    
    if provider == "distyl":
        available, _ = check_distyl_availability()
        if not available:
            print("Warning: Distyl-WebArena not available, will fallback to standard execution")
    
    # Parse tasks
    try:
        task_ids = parse_task_list(args.tasks)
    except ValueError as e:
        print(f"Error parsing tasks: {e}")
        sys.exit(1)
    
    print(f"Distyl-WebArena Parallel Runner")
    print(f"==============================")
    print(f"Model: {args.model}")
    print(f"Provider: {provider}")
    print(f"Tasks: {task_ids}")
    print(f"Max Workers: {args.max_workers}")
    
    if provider == "distyl":
        available, _ = check_distyl_availability()
        if available:
            print(f"Using Distyl-WebArena with hierarchical planning and reflection")
        else:
            print(f"Using standard execution (Distyl-WebArena not available)")
    
    print("\nStarting parallel execution...")
    
    # Run tasks in parallel
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_single_task, task_id, args.model, provider): task_id 
            for task_id in task_ids
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
                status = "✅" if result.get('success', False) else "❌"
                time_str = f"{result.get('execution_time', 0):.2f}s"
                print(f"{status} Task {task_id} completed in {time_str}")
                
            except Exception as e:
                print(f"❌ Task {task_id} failed with exception: {e}")
                results.append({
                    'task_id': task_id,
                    'success': False,
                    'error': str(e),
                    'execution_time': 0,
                    'model': args.model,
                    'provider': provider
                })
    
    total_time = time.time() - start_time
    
    # Save results
    results_dir = save_results(results, args.model, provider)
    
    print(f"\nExecution completed in {total_time:.2f}s")
    print(f"Results directory: {results_dir}")

if __name__ == "__main__":
    main()