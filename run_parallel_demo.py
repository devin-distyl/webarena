#!/usr/bin/env python3
"""
Parallel WebArena Agent Execution Demo
Runs multiple WebArena tasks concurrently to demonstrate parallel execution
"""

import subprocess
import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Task IDs to run in parallel
TASK_IDS = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 41, 42, 43, 62, 63, 64, 65, 77, 78, 79, 94, 95, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 119, 120, 121, 122, 123, 127, 128, 129, 130, 131, 157, 183, 184, 185, 186, 187, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 243, 244, 245, 246, 247, 288, 289, 290, 291, 292, 344, 345, 346, 347, 348, 374, 375, 423, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 470, 471, 472, 473, 474, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 676, 677, 678, 679, 680, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 759, 760, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 790]

# Model to use
MODEL = "gpt-3.5-turbo-0125"

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
    
    # Check for required API key
    if "OPENAI_API_KEY" not in env:
        raise ValueError("OPENAI_API_KEY not found in environment or .env file")
    
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

def run_single_task(task_id, env, result_container, lock):
    """Run a single WebArena task with specified model"""
    thread_id = threading.current_thread().ident
    result_dir = f"./parallel_demo_results/{MODEL.replace('-', '_').replace('.', '_')}"
    
    cmd = [
        "python", "run.py",
        "--instruction_path", "agent/prompts/jsons/p_cot_id_actree_2s.json",
        "--test_start_idx", str(task_id),
        "--test_end_idx", str(task_id + 1),
        "--model", MODEL,
        "--result_dir", result_dir
    ]
    
    with lock:
        print(f"🚀 [Thread {thread_id}] Starting task {task_id}")
    
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
            'stderr_lines': len(result.stderr.split('\n')) if result.stderr else 0
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
            'error': 'TIMEOUT'
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
            'error': str(e)
        }
        
        with lock:
            result_container.append(task_result)
        
        return task_result

def run_parallel_demo():
    """Run the parallel execution demo"""
    print("🧪 WebArena Parallel Agent Execution Demo")
    print("=" * 60)
    print(f"Tasks: {TASK_IDS}")
    print(f"Model: {MODEL}")
    print(f"Max concurrent tasks: {len(TASK_IDS)}")
    print("=" * 60)
    
    # Setup
    env = setup_environment()
    results = []
    results_lock = threading.Lock()
    
    # Create results directory
    os.makedirs("parallel_demo_results", exist_ok=True)
    
    # Load task information
    print("\n📋 Task Information:")
    task_info = {}
    for task_id in TASK_IDS:
        intent, sites = load_task_info(task_id)
        task_info[task_id] = {'intent': intent, 'sites': sites}
        print(f"  Task {task_id}: {intent[:70]}{'...' if len(intent) > 70 else ''}")
        if sites:
            print(f"           Sites: {', '.join(sites)}")
    
    print(f"\n🚀 Starting {len(TASK_IDS)} tasks in parallel...")
    start_time = time.time()
    
    # Run tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(TASK_IDS)) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_single_task, task_id, env, results, results_lock): task_id 
            for task_id in TASK_IDS
        }
        
        # Monitor progress
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            task_id = future_to_task[future]
            
            try:
                result = future.result()
                with results_lock:
                    print(f"📊 Progress: {completed}/{len(TASK_IDS)} tasks completed")
            except Exception as exc:
                with results_lock:
                    print(f"💥 Task {task_id} generated an exception: {exc}")
    
    total_elapsed = time.time() - start_time
    
    # Analyze results
    print(f"\n🏁 All tasks completed in {total_elapsed:.1f}s")
    print("\n" + "=" * 80)
    print("📊 PARALLEL EXECUTION RESULTS")
    print("=" * 80)
    
    # Sort results by task_id for consistent display
    results.sort(key=lambda x: x['task_id'])
    
    # Summary statistics
    successful_tasks = sum(1 for r in results if r['success'] and r.get('score', 0) > 0.5)
    total_score = sum(r.get('score', 0) for r in results)
    avg_score = total_score / len(results) if results else 0.0
    avg_time = sum(r['elapsed_time'] for r in results) / len(results) if results else 0.0
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Success rate: {successful_tasks}/{len(TASK_IDS)} ({successful_tasks/len(TASK_IDS)*100:.1f}%)")
    print(f"  Average score: {avg_score:.3f}")
    print(f"  Average time per task: {avg_time:.1f}s")
    print(f"  Total wall clock time: {total_elapsed:.1f}s")
    print(f"  Parallel efficiency: {(avg_time * len(TASK_IDS)) / total_elapsed:.1f}x speedup")
    
    # Detailed results
    print(f"\n📋 Task-by-Task Results:")
    print("-" * 80)
    print(f"{'Task':4} {'Intent':45} {'Score':6} {'Time':6} {'Thread':8} {'Status':8}")
    print("-" * 80)
    
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
    results_file = f"parallel_demo_results_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'task_ids': TASK_IDS,
        'model': MODEL,
        'total_elapsed_time': total_elapsed,
        'parallel_efficiency': (avg_time * len(TASK_IDS)) / total_elapsed if total_elapsed > 0 else 0,
        'summary': {
            'success_rate': successful_tasks / len(TASK_IDS),
            'avg_score': avg_score,
            'avg_time_per_task': avg_time,
            'successful_tasks': successful_tasks,
            'total_tasks': len(TASK_IDS)
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
    """Main function"""
    try:
        results = run_parallel_demo()
        print(f"\n✅ Parallel execution demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()