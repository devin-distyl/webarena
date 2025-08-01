"""
Parallel Integration: Enhanced run_parallel.py integration for Distyl-WebArena

Provides integration utilities to modify and extend the existing run_parallel.py
system to support Distyl-WebArena agent execution.
"""

import os
import sys
import json
import subprocess
from typing import Dict, Any, List, Optional
from .webarena_adapter import WebArenaAdapter, get_distyl_model_info


class ParallelIntegration:
    """
    Integration utilities for Distyl-WebArena with run_parallel.py
    """
    
    def __init__(self, webarena_root: str = None):
        if webarena_root is None:
            # Auto-detect WebArena root (go up two directories from this file)
            self.webarena_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.webarena_root = webarena_root
        
        self.run_parallel_path = os.path.join(self.webarena_root, "run_parallel.py")
        self.browser_env_path = os.path.join(self.webarena_root, "browser_env", "run.py")
    
    def create_distyl_run_parallel(self, output_path: str = None) -> str:
        """
        Create a modified run_parallel.py that supports Distyl-WebArena
        
        Args:
            output_path: Where to save the modified file (default: run_parallel_distyl.py)
            
        Returns:
            Path to the created file
        """
        
        if output_path is None:
            output_path = os.path.join(self.webarena_root, "run_parallel_distyl.py")
        
        # Read the original run_parallel.py
        if not os.path.exists(self.run_parallel_path):
            raise FileNotFoundError(f"Original run_parallel.py not found at {self.run_parallel_path}")
        
        with open(self.run_parallel_path, 'r') as f:
            original_content = f.read()
        
        # Modify the content to support Distyl-WebArena
        modified_content = self._modify_run_parallel_content(original_content)
        
        # Write the modified version
        with open(output_path, 'w') as f:
            f.write(modified_content)
        
        # Make it executable
        os.chmod(output_path, 0o755)
        
        print(f"Created Distyl-WebArena compatible run_parallel at: {output_path}")
        return output_path
    
    def _modify_run_parallel_content(self, content: str) -> str:
        """Modify run_parallel.py content to support Distyl-WebArena"""
        
        # Add Distyl-WebArena imports at the top
        distyl_imports = '''
# Distyl-WebArena Integration
try:
    from distyl_webarena.integration.webarena_adapter import create_distyl_agent_for_webarena, get_distyl_model_info
    DISTYL_AVAILABLE = True
except ImportError:
    DISTYL_AVAILABLE = False
    print("Warning: Distyl-WebArena not available. Install distyl_webarena package.")
    
'''
        
        # Find where to insert imports (after existing imports)
        import_position = content.find("import") 
        if import_position == -1:
            import_position = 0
        
        # Insert Distyl imports
        lines = content.split('\n')
        
        # Find the last import line
        last_import_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                last_import_line = i
        
        # Insert Distyl imports after last import
        lines.insert(last_import_line + 1, distyl_imports)
        
        # Add Distyl model support to supported models check
        # Find the model validation section and add Distyl support
        model_check_addition = '''
    # Add Distyl-WebArena model support
    if DISTYL_AVAILABLE and model_name.startswith("distyl"):
        return "distyl"
'''
        
        # Find provider detection function and enhance it
        provider_function_start = None
        for i, line in enumerate(lines):
            if "def get_provider" in line or "provider =" in line:
                provider_function_start = i
                break
        
        if provider_function_start:
            # Find a good place to insert the Distyl check
            for i in range(provider_function_start, min(provider_function_start + 20, len(lines))):
                if "return" in lines[i] and "openai" in lines[i]:
                    lines.insert(i, model_check_addition)
                    break
        
        # Add Distyl execution support to the run_task function
        distyl_execution_support = '''
        # Distyl-WebArena execution
        if DISTYL_AVAILABLE and model_name.startswith('distyl'):
            # Create Distyl agent
            distyl_agent = create_distyl_agent_for_webarena(task_config, model_name)
            
            # Use the existing browser_env/run.py with Distyl agent
            result = execute_with_distyl_agent(task_config, distyl_agent, isolated_urls)
            return result
'''
        
        # Find the task execution section
        for i, line in enumerate(lines):
            if "def run_task" in line:
                # Find the function body and add Distyl support
                function_start = i
                indent_level = 0
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and not lines[j].startswith(' '):
                        # End of function
                        break
                    if "# Execute task" in lines[j] or "browser_env/run.py" in lines[j]:
                        lines.insert(j, distyl_execution_support)
                        break
                break
        
        # Add the execute_with_distyl_agent function
        distyl_executor_function = '''

def execute_with_distyl_agent(task_config, distyl_agent, isolated_urls):
    """Execute task using Distyl-WebArena agent"""
    
    import subprocess
    import tempfile
    import json
    
    # Create temporary config file with isolated URLs
    temp_config = task_config.copy()
    
    # Update URLs to use isolated ports
    if isolated_urls:
        temp_config.update(isolated_urls)
    
    # Create temporary agent config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(temp_config, f, indent=2)
        temp_config_path = f.name
    
    try:
        # Use existing browser_env/run.py infrastructure
        env = os.environ.copy()
        
        # Add Distyl-specific environment variables
        env['DISTYL_AGENT_MODE'] = 'true'
        env['DISTYL_MODEL'] = distyl_agent.__class__.__name__
        
        # Execute using browser_env/run.py with virtual environment
        venv_python = os.path.join('env', 'webarena-env', 'bin', 'python')
        python_cmd = venv_python if os.path.exists(venv_python) else 'python'
        
        cmd = [
            python_cmd, 'browser_env/run.py',
            '--config', temp_config_path,
            '--agent_type', 'distyl_webarena'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            return {"success": True, "output": result.stdout}
        else:
            return {"success": False, "error": result.stderr}
            
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)

'''
        
        # Add the function at the end of the file
        lines.append(distyl_executor_function)
        
        # Join lines back into content
        modified_content = '\n'.join(lines)
        
        return modified_content
    
    def create_browser_env_patch(self, output_path: str = None) -> str:
        """
        Create a patch for browser_env/run.py to support Distyl-WebArena
        
        Args:
            output_path: Where to save the patch (default: browser_env_distyl_patch.py)
            
        Returns:
            Path to the created patch file
        """
        
        if output_path is None:
            output_path = os.path.join(self.webarena_root, "browser_env_distyl_patch.py")
        
        patch_content = '''"""
Browser Environment Distyl-WebArena Patch

This patch modifies browser_env/run.py to support Distyl-WebArena agents.
Apply this patch by importing and calling apply_distyl_patch() before running tasks.
"""

import os
import sys

# Add distyl_webarena to path
distyl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'distyl_webarena')
if distyl_path not in sys.path:
    sys.path.insert(0, distyl_path)

try:
    from distyl_webarena.integration.webarena_adapter import create_distyl_agent_for_webarena
    DISTYL_AVAILABLE = True
except ImportError:
    DISTYL_AVAILABLE = False
    print("Warning: Distyl-WebArena not available")

def apply_distyl_patch():
    """Apply Distyl-WebArena patch to browser_env/run.py"""
    
    if not DISTYL_AVAILABLE:
        print("Cannot apply Distyl patch: distyl_webarena not available")
        return False
    
    print("Distyl-WebArena patch applied successfully")
    return True

def create_distyl_compatible_agent(task_config, model_name="distyl-webarena"):
    """Create agent compatible with browser_env/run.py"""
    
    if not DISTYL_AVAILABLE:
        raise ImportError("Distyl-WebArena not available")
    
    return create_distyl_agent_for_webarena(task_config, model_name)

# Auto-apply patch when imported
if __name__ != "__main__":
    apply_distyl_patch()
'''
        
        with open(output_path, 'w') as f:
            f.write(patch_content)
        
        print(f"Created browser_env patch at: {output_path}")
        return output_path
    
    def run_distyl_parallel_tasks(self, model_name: str, tasks: List[int], 
                                provider: str = "distyl") -> Dict[str, Any]:
        """
        Run parallel tasks using Distyl-WebArena
        
        Args:
            model_name: Distyl model name (e.g., "distyl-webarena")
            tasks: List of task IDs to run
            provider: Provider name (default: "distyl")
            
        Returns:
            Execution results
        """
        
        # Create the modified run_parallel if it doesn't exist
        distyl_run_parallel = os.path.join(self.webarena_root, "run_parallel_distyl.py")
        if not os.path.exists(distyl_run_parallel):
            self.create_distyl_run_parallel(distyl_run_parallel)
        
        # Prepare command
        task_string = ','.join(map(str, tasks))
        cmd = [
            'python', distyl_run_parallel,
            '--model', model_name,
            '--provider', provider,
            '--tasks', task_string
        ]
        
        print(f"Running Distyl-WebArena parallel execution: {' '.join(cmd)}")
        
        # Ensure we use the virtual environment
        env = os.environ.copy()
        venv_path = os.path.join(self.webarena_root, 'env', 'webarena-env')
        if os.path.exists(venv_path):
            env['VIRTUAL_ENV'] = venv_path
            env['PATH'] = f"{os.path.join(venv_path, 'bin')}:{env.get('PATH', '')}"
        
        # Execute
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.webarena_root, env=env)
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    
    def install_distyl_integration(self) -> bool:
        """
        Install Distyl-WebArena integration into existing WebArena setup
        
        Returns:
            True if successful, False otherwise
        """
        
        try:
            # Create modified run_parallel
            distyl_run_parallel = self.create_distyl_run_parallel()
            
            # Create browser_env patch
            browser_patch = self.create_browser_env_patch()
            
            # Create integration documentation
            integration_doc = os.path.join(self.webarena_root, "DISTYL_INTEGRATION.md")
            self._create_integration_documentation(integration_doc, distyl_run_parallel, browser_patch)
            
            print("✓ Distyl-WebArena integration installed successfully!")
            print(f"✓ Modified run_parallel: {distyl_run_parallel}")
            print(f"✓ Browser env patch: {browser_patch}")
            print(f"✓ Documentation: {integration_doc}")
            
            return True
            
        except Exception as e:
            print(f"✗ Integration installation failed: {e}")
            return False
    
    def _create_integration_documentation(self, doc_path: str, run_parallel_path: str, 
                                        browser_patch_path: str):
        """Create integration documentation"""
        
        doc_content = f'''# Distyl-WebArena Integration

## Overview

Distyl-WebArena has been integrated with your existing WebArena setup. This integration provides:

- **Hierarchical Planning**: Multi-step task decomposition with DAG-based planning
- **Accessibility Tree Grounding**: Intelligent element detection and selection
- **Site-Specific Actions**: Optimized actions for different WebArena sites
- **Reflection & Error Recovery**: Self-correction when actions fail
- **Memory System**: Episodic and narrative memory for learning from experience

## Usage

### Running Parallel Tasks with Distyl-WebArena

**Method 1: Using the Runner Script (Recommended)**
```bash
# Single task
./run_distyl_webarena.sh --tasks 78

# Multiple tasks  
./run_distyl_webarena.sh --tasks 78,79,80

# Task range
./run_distyl_webarena.sh --tasks 78-82
```

**Method 2: Manual Virtual Environment Activation**
```bash
# Activate virtual environment
source env/webarena-env/bin/activate

# Run tasks
python run_parallel_distyl.py --model distyl-webarena --tasks 78

# Deactivate when done
deactivate
```

### Model Names

- `distyl-webarena`: Standard Distyl-WebArena agent
- `distyl-webarena-reflect`: With enhanced reflection
- `distyl-webarena-memory`: With persistent memory

### Integration Files

1. **Modified run_parallel**: `{run_parallel_path}`
   - Enhanced to support Distyl-WebArena agents
   - Maintains compatibility with existing models

2. **Browser Environment Patch**: `{browser_patch_path}`
   - Patch for browser_env/run.py
   - Automatically applied when using Distyl agents

### Architecture Components

```
distyl_webarena/
├── controller/          # Main agent controller
├── planner/            # Task planning and decomposition
├── grounder/           # Element detection and grounding
├── executor/           # Action execution with reflection
├── memory/             # Knowledge and experience storage
├── actions/            # Site-specific action libraries
└── integration/        # WebArena integration utilities
```

### Memory System

Distyl-WebArena automatically creates a memory directory:
- `distyl_webarena/memory/episodic/` - Subtask-level experiences
- `distyl_webarena/memory/narrative/` - Task-level experiences
- `distyl_webarena/memory/site_patterns/` - Site-specific patterns

### Supported Sites

- **Shopping**: OneStopShop e-commerce site
- **Shopping Admin**: Magento admin panel
- **Reddit**: Forum-style social platform
- **GitLab**: Development and project management
- **Wikipedia**: Knowledge base and search
- **Map**: Interactive mapping service

### Performance Benefits

- **Intelligent Planning**: Reduces trial-and-error through hierarchical decomposition
- **Element Auto-Detection**: Finds elements by semantic description
- **Experience Learning**: Improves performance over time through memory
- **Error Recovery**: Automatically suggests alternatives for failed actions
- **Site Optimization**: Uses site-specific knowledge for better performance

### Troubleshooting

1. **Import Errors**: Ensure distyl_webarena is in Python path
2. **Memory Issues**: Check write permissions for memory directory
3. **Action Failures**: Review logs in memory/episodic/ directory
4. **Performance**: Enable reflection and memory for best results

### Example Usage

```python
from distyl_webarena.integration.webarena_adapter import create_distyl_agent_for_webarena

# Create agent for specific task
task_config = {{"task_id": 78, "sites": ["shopping_admin"]}}
agent = create_distyl_agent_for_webarena(task_config)

# Agent is compatible with existing WebArena infrastructure
```

For more details, see the main CLAUDE.md documentation.
'''
        
        with open(doc_path, 'w') as f:
            f.write(doc_content)


# Test integration
if __name__ == "__main__":
    print("Testing Distyl-WebArena Parallel Integration...")
    
    try:
        integration = ParallelIntegration()
        success = integration.install_distyl_integration()
        
        if success:
            print("✓ Integration test successful")
            print("✓ Ready to run: python run_parallel_distyl.py --model distyl-webarena --tasks 78")
        else:
            print("✗ Integration test failed")
            
    except Exception as e:
        print(f"✗ Integration test error: {e}")