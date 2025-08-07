"""HTML debug report generator for WebArena"""
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
            logger.debug(f"=== STEP {step} AGENT REASONING ===")
            logger.debug(f"Response: {agent_response}")


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
        body { font-family: Arial, sans-serif; margin: 20px; }
        .task-section { margin: 20px 0; border: 1px solid #ddd; padding: 15px; }
        .step-section { margin: 10px 0; padding: 10px; background: #f9f9f9; }
        .screenshot { max-width: 800px; border: 1px solid #ccc; margin: 10px 0; }
        .log-entry { font-family: monospace; margin: 5px 0; padding: 5px; background: #f0f0f0; }
        .error { color: red; }
        .success { color: green; }
        .debug { color: blue; }
        .step-header { font-weight: bold; color: #333; }
    </style>
</head>
<body>
    <h1>WebArena Debug Report</h1>
"""
    
    def _generate_task_section(self, task_id: str) -> str:
        """Generate HTML section for a specific task"""
        html = f'<div class="task-section">\n'
        html += f'<h2>Task {task_id}</h2>\n'
        
        # Find all screenshots for this task
        screenshots = list(self.screenshot_dir.glob(f"{task_id}_step_*.png"))
        screenshots.sort(key=lambda x: int(x.stem.split("_")[-1]))
        
        for screenshot in screenshots:
            step_num = screenshot.stem.split("_")[-1]
            html += f'<div class="step-section">\n'
            html += f'<div class="step-header">Step {step_num}</div>\n'
            html += f'<img src="screenshots/{screenshot.name}" class="screenshot" alt="Step {step_num}">\n'
            html += '</div>\n'
        
        html += '</div>\n'
        return html
    
    def _generate_html_footer(self) -> str:
        """Generate HTML footer"""
        return """
    <div style="margin-top: 30px; padding: 10px; background: #f0f0f0;">
        <p><strong>How to use this report:</strong></p>
        <ul>
            <li>Each task shows step-by-step screenshots</li>
            <li>Check the run.log file for detailed action logs</li>
            <li>Look for errors in the log file</li>
            <li>Compare screenshots to understand what the agent is doing</li>
        </ul>
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