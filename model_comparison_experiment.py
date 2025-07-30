#!/usr/bin/env python3
"""
WebArena Model Comparison Experiment
Compares GPT-3.5-turbo-0125 vs GPT-4-1106-preview on shopping admin tasks
"""

import subprocess
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import re

# Task IDs to test
TASK_IDS = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 41, 42, 43, 62, 63, 64, 65, 77]

# Models to compare
MODELS = {
    "gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
    "gpt-4-1106-preview": "GPT-4 Turbo"  # Using available GPT-4 variant
}

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
    
    # Set defaults if not already set
    env.setdefault("SHOPPING", "http://localhost:7770")
    env.setdefault("SHOPPING_ADMIN", "http://localhost:7780/admin")
    env.setdefault("REDDIT", "http://localhost:9999")
    env.setdefault("GITLAB", "http://localhost:8023")
    env.setdefault("WIKIPEDIA", "http://localhost:8888")
    env.setdefault("MAP", "http://localhost:3000")
    env.setdefault("HOMEPAGE", "http://localhost:4399")
    
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
        return config.get('intent', f'Task {task_id}'), config.get('eval', {}).get('reference_answers', {})
    except Exception as e:
        return f'Task {task_id}', {}

def run_single_task(task_id, model, env):
    """Run a single WebArena task with specified model"""
    result_dir = f"./experiment_results/{model.replace('-', '_').replace('.', '_')}"
    
    cmd = [
        "python", "run.py",
        "--instruction_path", "agent/prompts/jsons/p_cot_id_actree_2s.json",
        "--test_start_idx", str(task_id),
        "--test_end_idx", str(task_id + 1),
        "--model", model,
        "--result_dir", result_dir
    ]
    
    print(f"  Running task {task_id} with {MODELS[model]}...")
    start_time = time.time()
    
    try:
        # Activate virtual environment and run
        activate_cmd = "source webarena-env/bin/activate && " + " ".join(cmd)
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
        score = extract_score_from_output(result.stdout)
        
        return {
            'success': success,
            'score': score,
            'elapsed_time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"    ⏰ TIMEOUT after {elapsed:.1f}s")
        return {
            'success': False,
            'score': 0.0,
            'elapsed_time': elapsed,
            'stdout': "",
            'stderr': "TIMEOUT",
            'returncode': -1
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    ❌ ERROR: {e}")
        return {
            'success': False,
            'score': 0.0,
            'elapsed_time': elapsed,
            'stdout': "",
            'stderr': str(e),
            'returncode': -1
        }

def extract_score_from_output(stdout):
    """Extract success score from run output"""
    if not stdout:
        return 0.0
    
    # Look for various success indicators
    patterns = [
        r'Average score:\s*([0-9]*\.?[0-9]+)',
        r'\(PASS\)',
        r'\(FAIL\)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, stdout)
        if match:
            if pattern.endswith('(PASS)'):
                return 1.0
            elif pattern.endswith('(FAIL)'):
                return 0.0
            else:
                try:
                    return float(match.group(1))
                except:
                    pass
    
    return 0.0

def print_progress_bar(completed, total, model_name):
    """Print a progress bar"""
    percent = (completed / total) * 100
    filled = int(50 * completed // total)
    bar = '█' * filled + '-' * (50 - filled)
    print(f'\r{model_name}: |{bar}| {completed}/{total} ({percent:.1f}%)', end='', flush=True)

def run_experiment():
    """Run the full experiment"""
    print("🧪 WebArena Model Comparison Experiment")
    print("=" * 60)
    print(f"Tasks: {len(TASK_IDS)} shopping admin tasks")
    print(f"Models: {', '.join(MODELS.values())}")
    print("=" * 60)
    
    # Setup
    env = setup_environment()
    results = {}
    task_info = {}
    
    # Load task information
    print("\n📋 Loading task information...")
    for task_id in TASK_IDS:
        intent, expected = load_task_info(task_id)
        task_info[task_id] = {'intent': intent, 'expected': expected}
        print(f"  Task {task_id}: {intent[:80]}...")
    
    # Run experiments
    for model_id, model_name in MODELS.items():
        print(f"\n🤖 Testing {model_name} ({model_id})")
        print("-" * 50)
        
        results[model_id] = {}
        completed = 0
        
        for i, task_id in enumerate(TASK_IDS):
            print_progress_bar(completed, len(TASK_IDS), model_name)
            
            result = run_single_task(task_id, model_id, env)
            results[model_id][task_id] = result
            
            # Print result
            status = "✅ PASS" if result['success'] and result['score'] > 0.5 else "❌ FAIL"
            print(f"\n    Task {task_id}: {status} (Score: {result['score']:.2f}, Time: {result['elapsed_time']:.1f}s)")
            
            completed += 1
            
            # Brief pause between tasks
            time.sleep(1)
        
        print_progress_bar(completed, len(TASK_IDS), model_name)
        print()  # New line after progress bar
    
    return results, task_info

def analyze_results(results, task_info):
    """Analyze and display results"""
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT RESULTS")
    print("=" * 80)
    
    # Summary statistics
    print("\n🎯 Overall Performance:")
    print("-" * 40)
    
    summary = {}
    for model_id, model_name in MODELS.items():
        model_results = results[model_id]
        
        # Calculate metrics
        total_tasks = len(TASK_IDS)
        successful_tasks = sum(1 for r in model_results.values() if r['success'] and r['score'] > 0.5)
        total_score = sum(r['score'] for r in model_results.values())
        avg_score = total_score / total_tasks
        avg_time = sum(r['elapsed_time'] for r in model_results.values()) / total_tasks
        
        summary[model_id] = {
            'success_rate': successful_tasks / total_tasks,
            'avg_score': avg_score,
            'avg_time': avg_time,
            'successful_tasks': successful_tasks,
            'total_tasks': total_tasks
        }
        
        print(f"{model_name:15}: {successful_tasks:2d}/{total_tasks} ({successful_tasks/total_tasks*100:5.1f}%) | "
              f"Avg Score: {avg_score:.3f} | Avg Time: {avg_time:.1f}s")
    
    # Task-by-task comparison
    print("\n📋 Task-by-Task Results:")
    print("-" * 80)
    print(f"{'Task':4} {'Intent':50} {'GPT-3.5':10} {'GPT-4':10} {'Winner':8}")
    print("-" * 80)
    
    gpt35_wins = 0
    gpt4_wins = 0
    ties = 0
    
    for task_id in TASK_IDS:
        intent = task_info[task_id]['intent'][:47] + "..." if len(task_info[task_id]['intent']) > 50 else task_info[task_id]['intent']
        
        gpt35_score = results[list(MODELS.keys())[0]][task_id]['score']
        gpt4_score = results[list(MODELS.keys())[1]][task_id]['score']
        
        gpt35_result = f"{gpt35_score:.2f}"
        gpt4_result = f"{gpt4_score:.2f}"
        
        if gpt35_score > gpt4_score:
            winner = "GPT-3.5"
            gpt35_wins += 1
        elif gpt4_score > gpt35_score:
            winner = "GPT-4"
            gpt4_wins += 1
        else:
            winner = "TIE"
            ties += 1
        
        print(f"{task_id:4} {intent:50} {gpt35_result:10} {gpt4_result:10} {winner:8}")
    
    # Final comparison
    print("\n🏆 Head-to-Head Comparison:")
    print("-" * 40)
    print(f"GPT-3.5 wins: {gpt35_wins}")
    print(f"GPT-4 wins:   {gpt4_wins}")
    print(f"Ties:         {ties}")
    
    better_model = "GPT-4" if gpt4_wins > gpt35_wins else "GPT-3.5" if gpt35_wins > gpt4_wins else "TIE"
    print(f"\n🎖️  Overall Winner: {better_model}")
    
    # Failed tasks analysis
    print("\n❌ Failed Tasks Analysis:")
    print("-" * 40)
    
    for model_id, model_name in MODELS.items():
        failed_tasks = [tid for tid in TASK_IDS if results[model_id][tid]['score'] <= 0.5]
        if failed_tasks:
            print(f"{model_name} failed on tasks: {failed_tasks}")
        else:
            print(f"{model_name}: No failures! 🎉")
    
    return summary

def save_detailed_results(results, task_info, summary):
    """Save detailed results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'task_ids': TASK_IDS,
        'models': MODELS,
        'summary': summary,
        'task_info': task_info,
        'detailed_results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    return filename

def main():
    """Main experiment function"""
    try:
        # Run experiment
        results, task_info = run_experiment()
        
        # Analyze results
        summary = analyze_results(results, task_info)
        
        # Save results
        results_file = save_detailed_results(results, task_info, summary)
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {results_file}")
        print(f"🔍 For debugging, check the experiment_results/ directory for individual task traces")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()