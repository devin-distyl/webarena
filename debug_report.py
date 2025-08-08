"""HTML debug report generator for WebArena"""
import json
import logging
from pathlib import Path
from typing import List, Dict


class DebugHelper:
    """Helper class for debugging and visualization"""
    
    def __init__(self, result_dir: str, enable_screenshots: bool = True):
        self.result_dir = Path(result_dir)
        self.enable_screenshots = enable_screenshots
        self.screenshot_dir = self.result_dir / "screenshots"
        self.screenshot_dir.mkdir(exist_ok=True)
        
    def log_action(self, logger, step: int, action, action_str: str, task_id: str):
        """Log detailed action information"""
        logger.debug(f"=== STEP {step} ACTION ===")
        logger.debug(f"Action Type: {action.get('action_type', 'Unknown')}")
        logger.debug(f"Action Description: {action_str}")
        
    def log_state(self, logger, step: int, obs, info, task_id: str):
        """Log current page state"""
        logger.debug(f"=== STEP {step} STATE ===")
        logger.debug(f"URL: {info.get('url', 'Unknown')}")
        logger.debug(f"Title: {info.get('title', 'Unknown')}")
        logger.debug(f"Observation length: {len(str(obs))} chars")
        
    def take_screenshot(self, page, step: int, task_id: str):
        """Take and save screenshot"""
        if not self.enable_screenshots:
            return
            
        try:
            screenshot_path = self.screenshot_dir / f"{task_id}_step_{step:02d}.png"
            page.screenshot(path=str(screenshot_path))
        except Exception as e:
            logging.warning(f"Failed to take screenshot: {e}")
            
    def log_agent_reasoning(self, logger, step: int, agent_response: str = None):
        """Log agent's reasoning if available"""
        if agent_response:
            logger.info(f"=== STEP {step} AGENT REASONING ===")
            logger.info(f"Response: {agent_response}")


class DebugReportGenerator:
    """Generate HTML reports for debugging WebArena runs"""
    
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.screenshot_dir = self.result_dir / "screenshots"
        
    def generate_html_report(self, task_ids: List[str] = None):
        """Generate an HTML report with screenshots and logs"""
        
        # Find all task IDs if not provided
        if task_ids is None:
            task_ids = self._get_task_ids()
        
        html_content = self._generate_html_header()
        
        for task_id in task_ids:
            html_content += self._generate_task_section(task_id)
        
        html_content += self._generate_html_footer()
        
        # Save the report
        report_path = self.result_dir / "debug_report.html"
        with open(report_path, "w") as f:
            f.write(html_content)
        
        print(f"Debug report generated: {report_path}")
        return report_path
    
    def _get_task_ids(self) -> List[str]:
        """Extract task IDs from screenshot filenames"""
        task_ids = set()
        if self.screenshot_dir.exists():
            for file in self.screenshot_dir.glob("*.png"):
                # Extract task_id from filename like "0_step_01.png"
                parts = file.stem.split("_")
                if len(parts) >= 2:
                    task_ids.add(parts[0])
        return sorted(list(task_ids))
    
    def _generate_html_header(self) -> str:
        """Generate HTML header with CSS styling"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>WebArena Debug Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }
        .task-section { 
            margin: 20px 0; 
            border: 1px solid #ddd; 
            padding: 20px; 
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .task-details {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #007bff;
        }
        .step-section { 
            margin: 15px 0; 
            padding: 15px; 
            background: #f9f9f9; 
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }
        .step-details {
            background: #ffffff;
            padding: 10px;
            margin: 10px 0;
            border-radius: 3px;
            border: 1px solid #dee2e6;
        }
        .agent-reasoning {
            background: #e3f2fd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #2196f3;
        }
        .reasoning-text {
            background: #f5f5f5;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            border: 1px solid #ddd;
            white-space: pre-wrap;
            font-family: Monaco, Consolas, 'Lucida Console', monospace;
            font-size: 12px;
            line-height: 1.4;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
        }
        .screenshot { 
            max-width: 100%; 
            border: 1px solid #ccc; 
            margin: 10px 0;
            border-radius: 3px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .log-entry { font-family: monospace; margin: 5px 0; padding: 5px; background: #f0f0f0; }
        .error { color: #dc3545; font-weight: bold; }
        .success { color: #28a745; font-weight: bold; }
        .debug { color: #007bff; }
        .step-header { 
            font-weight: bold; 
            color: #333; 
            font-size: 16px;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 2px solid #dee2e6;
        }
        h1 { 
            color: #333; 
            text-align: center; 
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 3px solid #007bff;
        }
        h2 { 
            color: #555; 
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #dee2e6;
        }
        p { margin: 5px 0; }
        strong { color: #333; }
    </style>
</head>
<body>
    <h1>WebArena Enhanced Debug Report</h1>
"""
    
    def _generate_task_section(self, task_id: str) -> str:
        """Generate enhanced HTML section for a specific task"""
        html = f'<div class="task-section">\n'
        
        # Try to load detailed results for this task
        task_dir = self.result_dir / f"task_{task_id}"
        detailed_steps = []
        performance_metrics = {}
        task_config = {}
        
        try:
            if (task_dir / "detailed_steps.json").exists():
                with open(task_dir / "detailed_steps.json") as f:
                    detailed_steps = json.load(f)
            
            if (task_dir / "performance_metrics.json").exists():
                with open(task_dir / "performance_metrics.json") as f:
                    performance_metrics = json.load(f)
            
            if (task_dir / "task_config.json").exists():
                with open(task_dir / "task_config.json") as f:
                    task_config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load detailed results for task {task_id}: {e}")
        
        # Task header with performance info
        html += f'<h2>Task {task_id}'
        if performance_metrics:
            success = performance_metrics.get('success', False)
            status_class = 'success' if success else 'error'
            html += f' - <span class="{status_class}">{"PASS" if success else "FAIL"}</span>'
            html += f' ({performance_metrics.get("final_score", "N/A")})'
        html += '</h2>\n'
        
        # Task details
        if task_config:
            html += f'<div class="task-details">\n'
            html += f'<p><strong>Intent:</strong> {task_config.get("intent", "Unknown")}</p>\n'
            html += f'<p><strong>Start URL:</strong> {task_config.get("start_url", "Unknown")}</p>\n'
            if performance_metrics:
                html += f'<p><strong>Total Time:</strong> {performance_metrics.get("total_time", "N/A"):.2f}s</p>\n'
                html += f'<p><strong>Total Steps:</strong> {performance_metrics.get("total_steps", "N/A")}</p>\n'
            html += '</div>\n'
        
        # Find all screenshots for this task
        screenshots = list(self.screenshot_dir.glob(f"{task_id}_step_*.png"))
        screenshots.sort(key=lambda x: int(x.stem.split("_")[-1]))
        
        # Generate step sections with detailed information
        for i, screenshot in enumerate(screenshots):
            step_num_str = screenshot.stem.split("_")[-1]
            step_num = int(step_num_str)
            
            # Find corresponding detailed step data
            step_data = None
            if detailed_steps:
                for ds in detailed_steps:
                    if ds.get('step') == step_num:
                        step_data = ds
                        break
            
            html += f'<div class="step-section">\n'
            html += f'<div class="step-header">Step {step_num_str}'
            if step_data and step_data.get('timing'):
                html += f' ({step_data["timing"]:.2f}s)'
            html += '</div>\n'
            
            # Add step details if available
            if step_data:
                html += f'<div class="step-details">\n'
                html += f'<p><strong>Action:</strong> {step_data.get("action_description", "N/A")}</p>\n'
                html += f'<p><strong>URL:</strong> {step_data.get("url", "N/A")}</p>\n'
                html += f'<p><strong>Page Title:</strong> {step_data.get("title", "N/A")}</p>\n'
                
                # Add agent reasoning if available
                reasoning = step_data.get('agent_reasoning', '')
                if reasoning and reasoning.strip():
                    html += f'<div class="agent-reasoning">\n'
                    html += f'<p><strong>Agent Reasoning:</strong></p>\n'
                    html += f'<pre class="reasoning-text">{reasoning}</pre>\n'
                    html += '</div>\n'
                
                html += '</div>\n'
            
            # Add screenshot
            html += f'<img src="screenshots/{screenshot.name}" class="screenshot" alt="Step {step_num_str}">\n'
            html += '</div>\n'
        
        html += '</div>\n'
        return html
    
    def _generate_html_footer(self) -> str:
        """Generate enhanced HTML footer"""
        return """
    <div style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
        <h3 style="color: #333; margin-top: 0;">How to use this Enhanced Debug Report:</h3>
        <ul style="line-height: 1.6;">
            <li><strong>Task Overview:</strong> Each task section shows the intent, performance metrics, and overall result</li>
            <li><strong>Step-by-Step Analysis:</strong> Each step includes screenshots, timing, and detailed action information</li>
            <li><strong>Agent Reasoning:</strong> Blue boxes show the complete LLM reasoning for each step</li>
            <li><strong>Performance Data:</strong> Check individual task directories for JSON files with detailed metrics</li>
            <li><strong>Trajectory Analysis:</strong> Complete interaction history is saved in trajectory.json files</li>
            <li><strong>Human Readable:</strong> Check summary.txt files for quick text-based overviews</li>
        </ul>
        <p style="margin-bottom: 0; font-style: italic; color: #666;">
            💡 Tip: Use the agent reasoning sections to understand why the agent made specific decisions at each step.
        </p>
    </div>
</body>
</html>
"""


def main():
    """Generate debug report from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate HTML debug report")
    parser.add_argument("--result_dir", default="results", help="Result directory")
    parser.add_argument("--task_ids", help="Comma-separated list of task IDs")
    
    args = parser.parse_args()
    
    generator = DebugReportGenerator(args.result_dir)
    
    task_ids = None
    if args.task_ids:
        task_ids = args.task_ids.split(",")
    
    report_path = generator.generate_html_report(task_ids)
    print(f"Open {report_path} in your browser to view the debug report")


if __name__ == "__main__":
    main() 