#!/usr/bin/env python3

import subprocess
import os
import sys
import time
import json
from pathlib import Path

def run_single_task(task_id, model, result_dir):
    """Run a single WebArena task with specified model"""
    env = os.environ.copy()
    # Load from .env file if it exists
    if os.path.exists(".env"):
        with open(".env", 'r') as f:
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
    
    cmd = [
        "python", "run.py",
        "--instruction_path", "agent/prompts/jsons/p_cot_id_actree_2s.json",
        "--test_start_idx", str(task_id),
        "--test_end_idx", str(task_id + 1),
        "--model", model,
        "--result_dir", result_dir
    ]
    
    print(f"Running task {task_id} with {model}...")
    try:
        result = subprocess.run(
            cmd, 
            cwd="/Users/devin/local_webarena",
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per task
        )
        
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"Task {task_id} with {model} timed out")
        return False, "", "Timeout"
    except Exception as e:
        print(f"Error running task {task_id} with {model}: {e}")
        return False, "", str(e)

def extract_score_from_output(stdout):
    """Extract success score from run output"""
    lines = stdout.split('\n')
    for line in lines:
        if 'Average score:' in line:
            try:
                score = float(line.split('Average score:')[1].strip())
                return score
            except:
                pass
    return None

if __name__ == "__main__":
    # Shopping admin task IDs (first 10)
    task_ids = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13]
    models = ["gpt-3.5-turbo", "gpt-4-1106-preview"]  # Using gpt-4-1106-preview as it's available
    
    results = {}
    
    for model in models:
        results[model] = {}
        result_dir = f"./results_experiment_{model.replace('-', '_').replace('.', '_')}"
        
        print(f"\n=== Running {model} ===")
        
        for task_id in task_ids:
            success, stdout, stderr = run_single_task(task_id, model, result_dir)
            score = extract_score_from_output(stdout) if success else 0.0
            
            results[model][task_id] = {
                'success': success,
                'score': score,
                'stdout': stdout,
                'stderr': stderr
            }
            
            print(f"Task {task_id}: {'SUCCESS' if success else 'FAILED'} (Score: {score})")
            
            # Brief pause between tasks
            time.sleep(2)
    
    # Save results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== EXPERIMENT SUMMARY ===")
    for model in models:
        scores = [r['score'] for r in results[model].values() if r['score'] is not None]
        successes = sum(1 for r in results[model].values() if r['success'])
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        print(f"{model}:")
        print(f"  Success rate: {successes}/{len(task_ids)} ({successes/len(task_ids)*100:.1f}%)")
        print(f"  Average score: {avg_score:.3f}")
        print()