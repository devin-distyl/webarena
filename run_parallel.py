#!/usr/bin/env python3
"""
WebArena Parallel Agent Execution Demo
Configurable parallel execution for any model and task set
"""

import subprocess
import os
import sys
import time
import json
import threading
import argparse
import shutil
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# Import Docker isolation manager for isolated container environments
from browser_env.docker_isolation_manager import DockerIsolationManager


def parse_evaluation_results(stdout: str, stderr: str) -> Tuple[float, bool, str]:
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


def setup_environment():
    """Setup environment variables for WebArena"""
    env = os.environ.copy()
    
    # Load from .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env[key] = value
    
    # Force set URLs (override defaults)
    env["SHOPPING"] = "http://localhost:7770"
    env["SHOPPING_ADMIN"] = "http://localhost:7780/admin"
    env["REDDIT"] = "http://localhost:9999"
    env["GITLAB"] = "http://localhost:8023"
    env["WIKIPEDIA"] = "http://localhost:8888"
    env["MAP"] = "http://localhost:3000"
    env["HOMEPAGE"] = "http://localhost:4399"
    
    return env

def load_task_info(task_id):
    """Load task information from config file"""
    config_path = f"config_files/{task_id}.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('intent', f'Task {task_id}'), config.get('sites', [])
    except Exception as e:
        return f'Task {task_id}', []

def run_single_task_isolated(task_id, provider, model, base_env, result_container, lock, run_timestamp, docker_manager):
    """Run a single WebArena task with isolated Docker environment"""
    thread_id = threading.current_thread().ident
    # Create unique result directory for this run with timestamp
    safe_model = model.replace('-', '_').replace('.', '_')
    result_dir = f"./parallel_demo_results/{run_timestamp}_{provider}_{safe_model}/task_{task_id}"
    
    with lock:
        print(f"🚀 [Thread {thread_id}] Starting task {task_id} with {provider}/{model}")
    
    start_time = time.time()
    isolated_env = None
    
    try:
        # Load original task configuration
        config_path = f"config_files/{task_id}.json"
        with open(config_path, 'r') as f:
            original_config = json.load(f)
        
        sites = original_config.get('sites', [])
        
        with lock:
            print(f"🐳 [Thread {thread_id}] Starting isolated Docker environment for task {task_id}")
        
        # Start isolated Docker environment
        isolated_env = docker_manager.start_environment(task_id, sites)
        if not isolated_env:
            raise Exception("Failed to start isolated Docker environment")
        
        with lock:
            print(f"✅ [Thread {thread_id}] Docker environment ready for task {task_id} on ports {isolated_env.base_port}-{isolated_env.base_port + 99}")
        
        # Get modified configuration for isolated environment
        modified_config = docker_manager.get_environment_config(task_id, original_config)
        
        # Create temporary config file for this task and backup original
        original_config_path = f"config_files/{task_id}.json"
        backup_config_path = f"config_files/{task_id}.json.backup_{int(time.time())}"
        
        # Backup original config
        shutil.copy2(original_config_path, backup_config_path)
        
        # Write modified config to original location (temporarily)
        with open(original_config_path, 'w') as f:
            json.dump(modified_config, f, indent=2)
        
        # Create environment with isolated URLs
        task_env = base_env.copy()
        urls = isolated_env.get_urls()
        task_env.update(urls)
        
        cmd = [
            "python", "browser_env/run.py",
            "--instruction_path", "agent/prompts/jsons/p_cot_id_actree_2s.json",
            "--test_start_idx", str(task_id),
            "--test_end_idx", str(task_id + 1),
            "--provider", provider,
            "--model", model,
            "--result_dir", result_dir
        ]
        
        # Config file is already handled by temporarily modifying the original config file
        
        # Activate virtual environment and run with isolated environment
        activate_cmd = "source webarena-env/bin/activate && " + " ".join(cmd)
        result = subprocess.run(
            activate_cmd,
            shell=True,
            cwd="/Users/devin/local_webarena",
            env=task_env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per task (longer due to container startup)
        )
        
        # Note: Config file will be restored in finally block
        
        elapsed = time.time() - start_time
        process_success = result.returncode == 0
        
        # Parse evaluation results from output
        score, task_success, eval_type = parse_evaluation_results(
            result.stdout or "", result.stderr or ""
        )
        
        task_result = {
            'task_id': task_id,
            'process_success': process_success,  # Did the script run without crashing
            'task_success': task_success,  # Did the agent complete the task successfully
            'success': process_success,  # Keep for backward compatibility
            'score': score,  # Actual evaluation score
            'eval_type': eval_type,  # PASS/FAIL/ERROR
            'elapsed_time': elapsed,
            'thread_id': thread_id,
            'returncode': result.returncode,
            'stdout_lines': len(result.stdout.split('\n')) if result.stdout else 0,
            'stderr_lines': len(result.stderr.split('\n')) if result.stderr else 0,
            'stdout': result.stdout if result.stdout else "",
            'stderr': result.stderr if result.stderr else "",
            'model': model,
            'provider': provider,
            'result_dir': result_dir,
            'docker_ports': f"{isolated_env.base_port}-{isolated_env.base_port + 99}" if isolated_env else "none"
        }
        
        with lock:
            result_container.append(task_result)
            status = "✅ PASS" if task_success else "❌ FAIL"
            if not process_success and result.stderr:
                print(f"🏁 [Thread {thread_id}] Task {task_id}: {status} (Score: {score:.2f}, Time: {elapsed:.1f}s)")
                print(f"   ⚠️  Error: {result.stderr.strip()[:200]}..." if len(result.stderr.strip()) > 200 else f"   ⚠️  Error: {result.stderr.strip()}")
            else:
                print(f"🏁 [Thread {thread_id}] Task {task_id}: {status} (Score: {score:.2f}, Time: {elapsed:.1f}s)")
        
        return task_result
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        with lock:
            print(f"⏰ [Thread {thread_id}] Task {task_id}: TIMEOUT after {elapsed:.1f}s")
        
        task_result = {
            'task_id': task_id,
            'process_success': False,
            'task_success': False,
            'success': False,  # Keep for backward compatibility
            'score': 0.0,
            'eval_type': 'ERROR',
            'elapsed_time': elapsed,
            'thread_id': thread_id,
            'returncode': -1,
            'error': 'TIMEOUT',
            'stdout': '',
            'stderr': 'TIMEOUT',
            'model': model,
            'provider': provider,
            'docker_ports': f"{isolated_env.base_port}-{isolated_env.base_port + 99}" if isolated_env else "none"
        }
        
        with lock:
            result_container.append(task_result)
        
        return task_result
    
    except Exception as e:
        elapsed = time.time() - start_time
        with lock:
            print(f"💥 [Thread {thread_id}] Task {task_id}: ERROR after {elapsed:.1f}s - {str(e)}")
        
        task_result = {
            'task_id': task_id,
            'process_success': False,
            'task_success': False,
            'success': False,  # Keep for backward compatibility
            'score': 0.0,
            'eval_type': 'ERROR',
            'elapsed_time': elapsed,
            'thread_id': thread_id,
            'returncode': -1,
            'error': str(e),
            'stdout': '',
            'stderr': str(e),
            'model': model,
            'provider': provider,
            'docker_ports': f"{isolated_env.base_port}-{isolated_env.base_port + 99}" if isolated_env else "none"
        }
        
        with lock:
            result_container.append(task_result)
        
        return task_result
    
    finally:
        # Always cleanup Docker environment and restore original config
        cleanup_errors = []
        
        # Cleanup Docker environment
        try:
            if isolated_env:
                with lock:
                    print(f"🧹 [Thread {thread_id}] Cleaning up Docker environment for task {task_id}")
                docker_manager.stop_environment(task_id)
        except Exception as cleanup_error:
            cleanup_errors.append(f"Docker cleanup: {cleanup_error}")
            with lock:
                print(f"⚠️  [Thread {thread_id}] Docker cleanup error for task {task_id}: {cleanup_error}")
        
        # Cleanup any orphaned directories (independent of Docker cleanup)
        try:
            cleaned_count = docker_manager.cleanup_orphaned_directories()
            if cleaned_count > 0:
                with lock:
                    print(f"🧹 [Thread {thread_id}] Cleaned up {cleaned_count} orphaned directories")
        except Exception as cleanup_error:
            cleanup_errors.append(f"Orphan cleanup: {cleanup_error}")
            with lock:
                print(f"⚠️  [Thread {thread_id}] Orphan cleanup error: {cleanup_error}")
        
        # Restore original config file
        try:
            original_config_path = f"config_files/{task_id}.json"
            if os.path.exists("config_files"):
                backup_files = [f for f in os.listdir("config_files") if f.startswith(f"{task_id}.json.backup_")]
                if backup_files:
                    # Use the most recent backup
                    backup_file = sorted(backup_files)[-1]
                    backup_path = f"config_files/{backup_file}"
                    if os.path.exists(backup_path):
                        shutil.copy2(backup_path, original_config_path)
                        os.remove(backup_path)  # Remove backup after restoring
                        with lock:
                            print(f"✅ [Thread {thread_id}] Restored config for task {task_id}")
                    else:
                        with lock:
                            print(f"⚠️  [Thread {thread_id}] Backup file missing: {backup_path}")
        except Exception as cleanup_error:
            cleanup_errors.append(f"Config restore: {cleanup_error}")
            with lock:
                print(f"⚠️  [Thread {thread_id}] Config restore error for task {task_id}: {cleanup_error}")
        
        # Report any cleanup errors
        if cleanup_errors:
            with lock:
                print(f"⚠️  [Thread {thread_id}] Task {task_id} cleanup completed with {len(cleanup_errors)} errors:")
                for error in cleanup_errors:
                    print(f"     - {error}")
        else:
            with lock:
                print(f"✅ [Thread {thread_id}] Task {task_id} cleanup completed successfully")

def parse_task_ids(task_ids_str):
    """Parse task IDs from various formats"""
    if not task_ids_str:
        return []
    
    # Handle comma-separated list
    if ',' in task_ids_str:
        return [int(x.strip()) for x in task_ids_str.split(',')]
    
    # Handle range notation (e.g., "78-82" or "78:82")
    if '-' in task_ids_str or ':' in task_ids_str:
        separator = '-' if '-' in task_ids_str else ':'
        start, end = task_ids_str.split(separator)
        return list(range(int(start), int(end) + 1))
    
    # Handle single task ID
    return [int(task_ids_str)]

def auto_detect_provider(model):
    """Auto-detect provider based on model name"""
    model_lower = model.lower()
    if 'gemini' in model_lower or 'bard' in model_lower:
        return 'google'
    elif 'gpt' in model_lower or 'o1' in model_lower:
        return 'openai'
    elif 'claude' in model_lower:
        return 'anthropic'  # Future extension
    else:
        return 'openai'  # Default

def run_parallel_demo(task_ids, provider, model):
    """Run the parallel execution demo"""
    # Create unique timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace('-', '_').replace('.', '_')
    run_dir = f"parallel_demo_results/{run_timestamp}_{provider}_{safe_model}"
    
    print("🧪 WebArena Parallel Agent Execution Demo")
    print("=" * 70)
    print(f"Tasks: {task_ids}")
    print(f"Model: {model}")
    print(f"Provider: {provider}")
    print(f"Max concurrent tasks: {len(task_ids)}")
    print(f"Run directory: {run_dir}")
    print("=" * 70)
    
    # Setup
    env = setup_environment()
    results = []
    results_lock = threading.Lock()
    
    # Check API key for the provider
    api_key_var = {
        'openai': 'OPENAI_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY'
    }.get(provider, 'OPENAI_API_KEY')
    
    if api_key_var not in env:
        raise ValueError(f"{api_key_var} not found in environment or .env file")
    
    # Create timestamped results directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Load task information
    print("\n📋 Task Information:")
    task_info = {}
    for task_id in task_ids:
        intent, sites = load_task_info(task_id)
        task_info[task_id] = {'intent': intent, 'sites': sites}
        print(f"  Task {task_id}: {intent[:70]}{'...' if len(intent) > 70 else ''}")
        if sites:
            print(f"           Sites: {', '.join(sites)}")
    
    print(f"\n🚀 Starting {len(task_ids)} tasks in parallel with {provider}/{model}...")
    print("🐳 Each task will run in its own isolated Docker environment")
    print("   - No shared state conflicts")
    print("   - Isolated authentication")
    print("   - Unique port allocations")
    print()
    start_time = time.time()
    
    # Initialize Docker isolation manager
    docker_manager = DockerIsolationManager()
    
    # Clean up any orphaned directories from previous runs
    print("🧹 Cleaning up orphaned directories from previous runs...")
    cleaned_count = docker_manager.cleanup_orphaned_directories()
    if cleaned_count > 0:
        print(f"   Cleaned up {cleaned_count} orphaned directories")
    else:
        print("   No orphaned directories found")
    
    try:
        # Run tasks in parallel using ThreadPoolExecutor with Docker isolation
        with ThreadPoolExecutor(max_workers=len(task_ids)) as executor:
            # Submit all tasks with Docker isolation
            future_to_task = {
                executor.submit(run_single_task_isolated, task_id, provider, model, env, results, results_lock, run_timestamp, docker_manager): task_id 
                for task_id in task_ids
            }
            
            # Monitor progress
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task_id = future_to_task[future]
                
                try:
                    result = future.result()
                    with results_lock:
                        print(f"📊 Progress: {completed}/{len(task_ids)} tasks completed")
                except Exception as exc:
                    with results_lock:
                        print(f"💥 Task {task_id} generated an exception: {exc}")
    
    finally:
        # Ensure all Docker environments are cleaned up
        print("🧹 Cleaning up all Docker environments...")
        docker_manager.cleanup_all()
    
    total_elapsed = time.time() - start_time
    
    # Analyze results
    print(f"\n🏁 All tasks completed in {total_elapsed:.1f}s")
    print("\n" + "=" * 80)
    print(f"📊 PARALLEL EXECUTION RESULTS - {provider.upper()}/{model.upper()}")
    print("=" * 80)
    
    # Sort results by task_id for consistent display
    results.sort(key=lambda x: x['task_id'])
    
    # Summary statistics - now with both process and task success
    process_successful_tasks = sum(1 for r in results if r.get('process_success', r.get('success', False)))
    task_successful_tasks = sum(1 for r in results if r.get('task_success', False))
    total_score = sum(r.get('score', 0) for r in results)
    avg_score = total_score / len(results) if results else 0.0
    avg_time = sum(r['elapsed_time'] for r in results) / len(results) if results else 0.0
    
    # Process success rate (no crashes)
    process_success_rate = process_successful_tasks / len(task_ids) if len(task_ids) > 0 else 0
    
    # Task success rate (evaluation passed)  
    task_success_rate = task_successful_tasks / len(task_ids) if len(task_ids) > 0 else 0
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Model: {model}")
    print(f"  Provider: {provider}")
    print(f"  Process success rate: {process_successful_tasks}/{len(task_ids)} ({process_success_rate*100:.1f}%)")
    print(f"  Task success rate: {task_successful_tasks}/{len(task_ids)} ({task_success_rate*100:.1f}%)")
    print(f"  Average score: {avg_score:.3f}")
    print(f"  Average time per task: {avg_time:.1f}s")
    print(f"  Total wall clock time: {total_elapsed:.1f}s")
    print(f"  Parallel efficiency: {(avg_time * len(task_ids)) / total_elapsed:.1f}x speedup")
    
    # Detailed results
    print(f"\n📋 Task-by-Task Results:")
    print("-" * 85)
    print(f"{'Task':4} {'Intent':45} {'Score':6} {'Time':6} {'Thread':8} {'Status':8}")
    print("-" * 85)
    
    for result in results:
        task_id = result['task_id']
        intent = task_info[task_id]['intent'][:42] + "..." if len(task_info[task_id]['intent']) > 45 else task_info[task_id]['intent']
        score = result.get('score', 0.0)
        elapsed = result['elapsed_time']
        thread_id = str(result.get('thread_id', 'N/A'))[-6:]  # Last 6 chars of thread ID
        status = "PASS" if result['success'] and score > 0.5 else "FAIL"
        
        print(f"{task_id:4} {intent:45} {score:6.2f} {elapsed:6.1f}s {thread_id:8} {status:8}")
    
    # Thread utilization analysis
    print(f"\n🧵 Thread Utilization:")
    thread_tasks = {}
    for result in results:
        thread_id = result.get('thread_id', 'Unknown')
        if thread_id not in thread_tasks:
            thread_tasks[thread_id] = []
        thread_tasks[thread_id].append(result['task_id'])
    
    for thread_id, tasks in thread_tasks.items():
        print(f"  Thread {str(thread_id)[-6:]}: Tasks {tasks}")
    
    # Save detailed results in the timestamped directory
    results_file = f"{run_dir}/parallel_demo_results.json"
    
    output = {
        'timestamp': run_timestamp,
        'task_ids': task_ids,
        'model': model,
        'provider': provider,
        'total_elapsed_time': total_elapsed,
        'parallel_efficiency': (avg_time * len(task_ids)) / total_elapsed if total_elapsed > 0 else 0,
        'summary': {
            'process_success_rate': process_success_rate,
            'task_success_rate': task_success_rate,
            'avg_score': avg_score,
            'avg_time_per_task': avg_time,
            'process_successful_tasks': process_successful_tasks,
            'task_successful_tasks': task_successful_tasks,
            'total_tasks': len(task_ids)
        },
        'task_info': task_info,
        'detailed_results': results,
        'thread_utilization': {str(k): v for k, v in thread_tasks.items()},
        'run_directory': run_dir
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print(f"🔍 Individual task traces and HTML files saved in: {run_dir}/")
    print(f"📁 Each task has its own subdirectory with render_*.html files for debugging")
    
    # Create a summary of what files are available for debugging
    html_files = []
    trace_dirs = []
    for task_id in task_ids:
        task_dir = f"{run_dir}/task_{task_id}"
        if os.path.exists(task_dir):
            # Find HTML render files
            for file in os.listdir(task_dir):
                if file.startswith('render_') and file.endswith('.html'):
                    html_files.append(f"{task_dir}/{file}")
            # Find trace directories
            trace_dir = f"{task_dir}/traces"
            if os.path.exists(trace_dir):
                trace_dirs.append(trace_dir)
    
    if html_files:
        print(f"🌐 Found {len(html_files)} HTML render files for debugging")
        if len(html_files) <= 10:  # Show individual files if not too many
            for html_file in html_files:
                print(f"   📄 {html_file}")
    if trace_dirs:
        print(f"📊 Found {len(trace_dirs)} trace directories with detailed execution logs")
        
    # Create a README file in the run directory for easy access
    readme_content = f"""# WebArena Parallel Execution Results

## Run Details
- **Timestamp**: {run_timestamp}
- **Model**: {model}
- **Provider**: {provider}
- **Tasks**: {task_ids}
- **Process Success Rate**: {process_successful_tasks}/{len(task_ids)} ({process_success_rate*100:.1f}%)
- **Task Success Rate**: {task_successful_tasks}/{len(task_ids)} ({task_success_rate*100:.1f}%)

## File Structure
```
{run_dir}/
├── parallel_demo_results.json    # Main results file
├── README.md                     # This file
"""

    for task_id in task_ids:
        readme_content += f"├── task_{task_id}/\n"
        readme_content += f"│   ├── render_*.html          # HTML snapshots for debugging\n"
        readme_content += f"│   ├── traces/                # Execution traces\n"
        readme_content += f"│   └── config.json            # Task configuration\n"

    readme_content += f"""
## Debugging
- **HTML Files**: Open render_*.html files in your browser to see page snapshots
- **Traces**: Check traces/ directories for detailed execution logs
- **Results**: Main results in parallel_demo_results.json

## Usage
To view HTML files for task {task_ids[0] if task_ids else 'X'}:
```bash
open {run_dir}/task_{task_ids[0] if task_ids else 'X'}/render_*.html
```
"""

    readme_path = f"{run_dir}/README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📖 README with file structure saved to: {readme_path}")
    
    # If no HTML files found, show debugging information
    if not html_files:
        print(f"\n🔍 No HTML files found. Debugging information:")
        failed_tasks = [r for r in results if not r['success']]
        if failed_tasks:
            print(f"   ❌ {len(failed_tasks)} tasks failed. Sample error:")
            sample_error = failed_tasks[0]
            if sample_error.get('stderr'):
                print(f"   📋 Task {sample_error['task_id']} stderr: {sample_error['stderr'][:300]}...")
            print(f"   📁 Expected directory: {sample_error.get('result_dir', 'unknown')}")
            
            # Check if files were created elsewhere
            print(f"   🔍 Checking for render files in current directory...")
            import glob
            render_files = glob.glob("render_*.html")
            if render_files:
                print(f"   📄 Found render files in current directory: {render_files}")
            else:
                print(f"   📄 No render files found in current directory either")
    
    return results

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='WebArena Parallel Agent Execution Demo')
    
    parser.add_argument('--model', '-m', type=str, default='gpt-4.1-2025-04-14',
                        help='Model to use (e.g., gpt-4.1-2025-04-14, gemini-2.5-pro)')
    parser.add_argument('--provider', '-p', type=str, default=None,
                        help='Provider to use (openai, google). Auto-detected if not specified')
    parser.add_argument('--tasks', '-t', type=str, required=True,
                        help='Task IDs to run. Examples: "78,79,80" or "78-82" or "78:82" or "78"')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: number of tasks)')
    
    args = parser.parse_args()
    
    # Parse task IDs
    try:
        task_ids = parse_task_ids(args.tasks)
        if not task_ids:
            print("❌ No valid task IDs provided")
            return 1
    except ValueError as e:
        print(f"❌ Error parsing task IDs: {e}")
        return 1
    
    # Auto-detect provider if not specified
    provider = args.provider or auto_detect_provider(args.model)
    
    print(f"🎯 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Provider: {provider} {'(auto-detected)' if not args.provider else ''}")
    print(f"   Tasks: {task_ids}")
    print(f"   Max workers: {args.max_workers or len(task_ids)}")
    print()
    
    try:
        results = run_parallel_demo(task_ids, provider, args.model)
        print(f"\n✅ Parallel execution demo completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())