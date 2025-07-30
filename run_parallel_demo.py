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
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def run_single_task(task_id, provider, model, env, result_container, lock):
    """Run a single WebArena task with specified model and provider"""
    thread_id = threading.current_thread().ident
    result_dir = f"./parallel_demo_results/{model.replace('-', '_').replace('.', '_')}"
    
    cmd = [
        "python", "run.py",
        "--instruction_path", "agent/prompts/jsons/p_cot_id_actree_2s.json",
        "--test_start_idx", str(task_id),
        "--test_end_idx", str(task_id + 1),
        "--provider", provider,
        "--model", model,
        "--result_dir", result_dir
    ]
    
    with lock:
        print(f"🚀 [Thread {thread_id}] Starting task {task_id} with {provider}/{model}")
    
    start_time = time.time()
    
    try:
        # Activate virtual environment and run
        activate_cmd = "source env/webarena-env/bin/activate && " + " ".join(cmd)
        result = subprocess.run(
            activate_cmd,
            shell=True,
            cwd="/Users/devin/local_webarena",
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per task
        )
        
        elapsed = time.time() - start_time
        success = result.returncode == 0
        
        # Extract score from output
        score = 0.0
        if result.stdout and "Average score:" in result.stdout:
            try:
                score_line = [line for line in result.stdout.split('\n') if 'Average score:' in line][-1]
                score = float(score_line.split('Average score:')[1].strip())
            except:
                pass
        
        task_result = {
            'task_id': task_id,
            'success': success,
            'score': score,
            'elapsed_time': elapsed,
            'thread_id': thread_id,
            'returncode': result.returncode,
            'stdout_lines': len(result.stdout.split('\n')) if result.stdout else 0,
            'stderr_lines': len(result.stderr.split('\n')) if result.stderr else 0,
            'model': model,
            'provider': provider
        }
        
        with lock:
            result_container.append(task_result)
            status = "✅ PASS" if success and score > 0.5 else "❌ FAIL"
            print(f"🏁 [Thread {thread_id}] Task {task_id}: {status} (Score: {score:.2f}, Time: {elapsed:.1f}s)")
        
        return task_result
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        with lock:
            print(f"⏰ [Thread {thread_id}] Task {task_id}: TIMEOUT after {elapsed:.1f}s")
        
        task_result = {
            'task_id': task_id,
            'success': False,
            'score': 0.0,
            'elapsed_time': elapsed,
            'thread_id': thread_id,
            'returncode': -1,
            'error': 'TIMEOUT',
            'model': model,
            'provider': provider
        }
        
        with lock:
            result_container.append(task_result)
        
        return task_result
        
    except Exception as e:
        elapsed = time.time() - start_time
        with lock:
            print(f"💥 [Thread {thread_id}] Task {task_id}: ERROR - {e}")
        
        task_result = {
            'task_id': task_id,
            'success': False,
            'score': 0.0,
            'elapsed_time': elapsed,
            'thread_id': thread_id,
            'returncode': -1,
            'error': str(e),
            'model': model,
            'provider': provider
        }
        
        with lock:
            result_container.append(task_result)
        
        return task_result

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
    print("🧪 WebArena Parallel Agent Execution Demo")
    print("=" * 70)
    print(f"Tasks: {task_ids}")
    print(f"Model: {model}")
    print(f"Provider: {provider}")
    print(f"Max concurrent tasks: {len(task_ids)}")
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
    
    # Create results directory
    os.makedirs("parallel_demo_results", exist_ok=True)
    
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
    start_time = time.time()
    
    # Run tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(task_ids)) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_single_task, task_id, provider, model, env, results, results_lock): task_id 
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
    
    total_elapsed = time.time() - start_time
    
    # Analyze results
    print(f"\n🏁 All tasks completed in {total_elapsed:.1f}s")
    print("\n" + "=" * 80)
    print(f"📊 PARALLEL EXECUTION RESULTS - {provider.upper()}/{model.upper()}")
    print("=" * 80)
    
    # Sort results by task_id for consistent display
    results.sort(key=lambda x: x['task_id'])
    
    # Summary statistics
    successful_tasks = sum(1 for r in results if r['success'] and r.get('score', 0) > 0.5)
    total_score = sum(r.get('score', 0) for r in results)
    avg_score = total_score / len(results) if results else 0.0
    avg_time = sum(r['elapsed_time'] for r in results) / len(results) if results else 0.0
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Model: {model}")
    print(f"  Provider: {provider}")
    print(f"  Success rate: {successful_tasks}/{len(task_ids)} ({successful_tasks/len(task_ids)*100:.1f}%)")
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
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"parallel_demo_{provider}_{model.replace('-', '_').replace('.', '_')}_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'task_ids': task_ids,
        'model': model,
        'provider': provider,
        'total_elapsed_time': total_elapsed,
        'parallel_efficiency': (avg_time * len(task_ids)) / total_elapsed if total_elapsed > 0 else 0,
        'summary': {
            'success_rate': successful_tasks / len(task_ids),
            'avg_score': avg_score,
            'avg_time_per_task': avg_time,
            'successful_tasks': successful_tasks,
            'total_tasks': len(task_ids)
        },
        'task_info': task_info,
        'detailed_results': results,
        'thread_utilization': {str(k): v for k, v in thread_tasks.items()}
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print(f"🔍 Individual task traces saved in: parallel_demo_results/")
    
    return results

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='WebArena Parallel Agent Execution Demo')
    
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Model to use (e.g., gpt-3.5-turbo, gemini-2.5-pro)')
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