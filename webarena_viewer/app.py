#!/usr/bin/env python3
"""
WebArena Experiment Viewer
A Flask web application for viewing WebArena parallel execution results
"""

import os
import json
import glob
import re
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, send_file, request
from bs4 import BeautifulSoup

app = Flask(__name__)

# Base directory for experiments
RESULTS_DIR = Path(__file__).parent.parent / "parallel_demo_results"

class ExperimentLoader:
    """Loads and parses WebArena experiment data"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
    
    def get_all_experiments(self):
        """Get all experiment directories sorted by timestamp (newest first)"""
        experiments = []
        
        if not self.results_dir.exists():
            return experiments
        
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                # Parse timestamp from directory name
                try:
                    # Expected format: YYYYMMDD_HHMMSS_provider_model
                    parts = exp_dir.name.split('_')
                    if len(parts) >= 2:
                        timestamp_str = f"{parts[0]}_{parts[1]}"
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        
                        # Load experiment summary
                        summary = self.load_experiment_summary(exp_dir)
                        
                        experiments.append({
                            'id': exp_dir.name,
                            'path': str(exp_dir),
                            'timestamp': timestamp,
                            'timestamp_str': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            'summary': summary
                        })
                except (ValueError, IndexError):
                    # Skip directories that don't match expected format
                    continue
        
        # Sort by timestamp (newest first)
        experiments.sort(key=lambda x: x['timestamp'], reverse=True)
        return experiments
    
    def load_experiment_summary(self, exp_dir):
        """Load experiment summary from JSON files"""
        summary = {
            'model': 'Unknown',
            'provider': 'Unknown', 
            'task_count': 0,
            'success_rate': 0.0,
            'avg_score': 0.0,
            'tasks': []
        }
        
        # Try different possible result file names
        result_files = ['parallel_demo_results.json', 'parallel_pooled_results.json']
        
        for filename in result_files:
            result_file = exp_dir / filename
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    summary.update({
                        'model': data.get('model', 'Unknown'),
                        'provider': data.get('provider', 'Unknown'),
                        'task_count': len(data.get('task_ids', [])),
                        'success_rate': data.get('summary', {}).get('success_rate', 0.0),
                        'avg_score': data.get('summary', {}).get('avg_score', 0.0),
                        'execution_time': data.get('total_elapsed_time', data.get('execution_time', 0)),
                        'task_info': data.get('task_info', {}),
                        'detailed_results': data.get('detailed_results', [])
                    })
                    break
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Get list of available tasks
        tasks = []
        for task_dir in exp_dir.iterdir():
            if task_dir.is_dir() and task_dir.name.startswith('task_'):
                try:
                    task_id = int(task_dir.name.split('_')[1])
                    
                    # Check for render HTML file
                    render_files = list(task_dir.glob('render_*.html'))
                    has_render = len(render_files) > 0
                    
                    # Get task info from summary
                    task_info = summary['task_info'].get(str(task_id), {})
                    
                    # Get detailed results
                    task_result = None
                    for result in summary['detailed_results']:
                        if result.get('task_id') == task_id:
                            task_result = result
                            break
                    
                    task_data = {
                        'id': task_id,
                        'intent': task_info.get('intent', f'Task {task_id}'),
                        'sites': task_info.get('sites', []),
                        'has_render': has_render,
                        'render_files': [f.name for f in render_files],
                        'success': task_result.get('success', False) if task_result else False,
                        'score': task_result.get('score', 0.0) if task_result else 0.0,
                        'elapsed_time': task_result.get('elapsed_time', 0.0) if task_result else 0.0
                    }
                    
                    tasks.append(task_data)
                    
                except (ValueError, IndexError):
                    continue
        
        # Sort tasks by ID
        tasks.sort(key=lambda x: x['id'])
        summary['tasks'] = tasks
        
        return summary

    def get_task_config(self, exp_id, task_id):
        """Get task configuration and evaluation data"""
        exp_dir = self.results_dir / exp_id
        task_dir = exp_dir / f"task_{task_id}"
        
        if not task_dir.exists():
            return None
        
        # Load task config
        config_file = task_dir / "config.json"
        task_config = {}
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    task_config = json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Load original task configuration from config_files
        original_config_file = self.results_dir.parent / "config_files" / f"{task_id}.json"
        original_config = {}
        if original_config_file.exists():
            try:
                with open(original_config_file, 'r') as f:
                    original_config = json.load(f)
            except json.JSONDecodeError:
                pass
        
        return {
            'task_config': task_config,
            'original_config': original_config,
            'task_dir': str(task_dir)
        }
    
    def parse_render_html(self, exp_id, task_id):
        """Parse render HTML file to extract thinking, actions, and screenshots"""
        exp_dir = self.results_dir / exp_id
        task_dir = exp_dir / f"task_{task_id}"
        
        if not task_dir.exists():
            return None
        
        # Find render HTML file
        render_files = list(task_dir.glob('render_*.html'))
        if not render_files:
            return {'error': 'No render file found'}
        
        render_file = render_files[0]
        
        try:
            with open(render_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            steps = []
            step_num = 0
            
            # Find all action prediction blocks
            predict_actions = soup.find_all('div', class_='predict_action')
            
            for predict_div in predict_actions:
                step_num += 1
                
                # Extract thinking/reasoning
                thinking_div = predict_div.find('div', class_='raw_parsed_prediction')
                thinking = thinking_div.get_text(strip=True) if thinking_div else ""
                
                # Extract parsed action
                action_div = predict_div.find('div', class_='parsed_action')
                action = action_div.get_text(strip=True) if action_div else ""
                
                # Find associated screenshots (look comprehensively for img tags)
                screenshots = []
                
                # 1. Check within the predict_action div itself
                imgs_within = predict_div.find_all('img')
                for img in imgs_within:
                    if img.get('src', '').startswith('data:image'):
                        screenshots.append(img.get('src'))
                
                # 2. Check previous siblings (original logic, but collect all)
                current = predict_div
                for _ in range(15):  # Increased search range
                    current = current.find_previous_sibling()
                    if not current:
                        break
                    imgs = current.find_all('img')
                    for img in imgs:
                        if img.get('src', '').startswith('data:image'):
                            src = img.get('src')
                            if src not in screenshots:  # Avoid duplicates
                                screenshots.append(src)
                
                # 3. Check following siblings as well
                current = predict_div
                for _ in range(10):  # Look ahead for screenshots
                    current = current.find_next_sibling()
                    if not current:
                        break
                    imgs = current.find_all('img')
                    for img in imgs:
                        if img.get('src', '').startswith('data:image'):
                            src = img.get('src')
                            if src not in screenshots:  # Avoid duplicates
                                screenshots.append(src)
                
                # 4. If no screenshots found yet, search more broadly in surrounding context
                if not screenshots:
                    # Look in parent containers
                    parent = predict_div.parent
                    if parent:
                        all_imgs = parent.find_all('img')
                        for img in all_imgs:
                            if img.get('src', '').startswith('data:image'):
                                src = img.get('src')
                                if src not in screenshots:
                                    screenshots.append(src)
                                    if len(screenshots) >= 3:  # Limit to avoid too many
                                        break
                
                steps.append({
                    'step': step_num,
                    'thinking': thinking,
                    'action': action,
                    'screenshots': screenshots,
                    'screenshot': screenshots[0] if screenshots else None  # Keep backward compatibility
                })
            
            # Extract page title and URL if available
            page_title = ""
            page_url = ""
            h3_url = soup.find('h3', class_='url')
            if h3_url:
                a_tag = h3_url.find('a')
                if a_tag:
                    page_url = a_tag.get('href', '')
                    page_title = a_tag.get_text(strip=True).replace('URL: ', '')
            
            return {
                'page_title': page_title or 'Task Execution',
                'page_url': page_url,
                'steps': steps,
                'total_steps': len(steps)
            }
            
        except Exception as e:
            return {'error': f'Failed to parse HTML: {str(e)}'}

# Create loader instance
loader = ExperimentLoader(RESULTS_DIR)

@app.route('/')
def index():
    """Main page showing experiment list"""
    experiments = loader.get_all_experiments()
    
    # Get most recent experiment as default
    selected_exp = experiments[0] if experiments else None
    
    return render_template('index.html', 
                         experiments=experiments, 
                         selected_exp=selected_exp)

@app.route('/api/experiments')
def api_experiments():
    """API endpoint to get all experiments"""
    experiments = loader.get_all_experiments()
    return jsonify(experiments)

@app.route('/api/experiment/<exp_id>')
def api_experiment(exp_id):
    """API endpoint to get specific experiment details"""
    experiments = loader.get_all_experiments()
    exp = next((e for e in experiments if e['id'] == exp_id), None)
    
    if not exp:
        return jsonify({'error': 'Experiment not found'}), 404
    
    return jsonify(exp)

@app.route('/api/task/<exp_id>/<int:task_id>')
def api_task(exp_id, task_id):
    """API endpoint to get task details"""
    task_data = loader.get_task_config(exp_id, task_id)
    
    if not task_data:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(task_data)

@app.route('/api/task/<exp_id>/<int:task_id>/parsed')
def api_task_parsed(exp_id, task_id):
    """API endpoint to get parsed task execution data"""
    parsed_data = loader.parse_render_html(exp_id, task_id)
    
    if not parsed_data:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(parsed_data)

@app.route('/render/<exp_id>/<int:task_id>')
def render_task(exp_id, task_id):
    """Serve the render HTML file for a task"""
    exp_dir = loader.results_dir / exp_id
    task_dir = exp_dir / f"task_{task_id}"
    
    if not task_dir.exists():
        return f"""
        <html><body style="font-family: Arial; padding: 20px; text-align: center;">
        <h2>❌ Task Directory Not Found</h2>
        <p>Task {task_id} directory not found in experiment {exp_id}</p>
        <p><code>{task_dir}</code></p>
        </body></html>
        """, 404
    
    # Find render HTML file - try multiple patterns
    render_patterns = [f'render_{task_id}.html', 'render_*.html', '*.html']
    render_file = None
    
    for pattern in render_patterns:
        files = list(task_dir.glob(pattern))
        if files:
            render_file = files[0]
            break
    
    if not render_file:
        return f"""
        <html><body style="font-family: Arial; padding: 20px; text-align: center;">
        <h2>📄 No Render File Found</h2>
        <p>No HTML render file found for task {task_id}</p>
        <p>Looking for files matching: {', '.join(render_patterns)}</p>
        <p>Available files: {', '.join([f.name for f in task_dir.iterdir() if f.is_file()])}</p>
        </body></html>
        """, 404
    
    try:
        return send_file(render_file, mimetype='text/html')
    except Exception as e:
        return f"""
        <html><body style="font-family: Arial; padding: 20px; text-align: center;">
        <h2>⚠️ Error Loading Render File</h2>
        <p>Error serving render file: {e}</p>
        <p>File: <code>{render_file}</code></p>
        </body></html>
        """, 500

@app.route('/render/<exp_id>/<int:task_id>/<filename>')
def render_task_file(exp_id, task_id, filename):
    """Serve specific render file for a task"""
    exp_dir = loader.results_dir / exp_id
    task_dir = exp_dir / f"task_{task_id}"
    render_file = task_dir / filename
    
    if not render_file.exists():
        return "File not found", 404
    
    try:
        return send_file(render_file, mimetype='text/html')
    except Exception as e:
        return f"Error serving file: {e}", 500

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon"""
    favicon_path = Path(__file__).parent / 'distyl.ico'
    if favicon_path.exists():
        return send_file(favicon_path, mimetype='image/x-icon')
    else:
        return '', 404

if __name__ == '__main__':
    # Check if results directory exists
    if not RESULTS_DIR.exists():
        print(f"Warning: Results directory {RESULTS_DIR} does not exist")
        print("Creating empty directory...")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting WebArena Experiment Viewer...")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Available at: http://localhost:8080")
    
    app.run(debug=True, host='0.0.0.0', port=8080)