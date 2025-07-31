# WebArena Parallel Execution System - CLAUDE.md

This document provides comprehensive information about the WebArena parallel execution system for Claude AI assistant.

## System Overview

WebArena is a realistic web environment for building autonomous agents, featuring a sophisticated parallel execution system that allows multiple tasks to run simultaneously with isolated Docker environments.

### Key Components

1. **Main Execution Script**: `run_parallel.py` - Orchestrates parallel task execution
2. **Browser Environment**: `browser_env/run.py` - Core agent execution environment  
3. **Docker Isolation**: `browser_env/docker_isolation_manager.py` - Manages isolated container environments
4. **Agent System**: `agent/agent.py` - AI agent implementations
5. **Configuration**: `config_files/*.json` - Task definitions (812 total tasks)

## Running Parallel Tasks with `run_parallel.py`

### Command Structure
```bash
python run_parallel.py --model <model_name> --provider <provider> --tasks <task_ids>
```

### Examples
```bash
# Single task
python run_parallel.py --model gpt-4.1-2025-04-14 --provider openai --tasks 78

# Multiple tasks (comma-separated)
python run_parallel.py --model gpt-4.1-2025-04-14 --provider openai --tasks 78,79,80

# Task range
python run_parallel.py --model gpt-4.1-2025-04-14 --provider openai --tasks 78-82

# Gemini model (auto-detects Google provider)
python run_parallel.py --model gemini-2.5-pro --tasks 78-80
```

### Supported Providers & Models
- **OpenAI**: `gpt-4.1-2025-04-14`, `gpt-4`, `gpt-3.5-turbo`, `o1-preview`
- **Google**: `gemini-2.5-pro`, `gemini-1.5-pro`
- **Anthropic**: Future support planned

### Environment Variables Required
- `OPENAI_API_KEY` - For OpenAI models
- `GOOGLE_API_KEY` - For Google/Gemini models
- `ANTHROPIC_API_KEY` - For Anthropic models (future)

## How the System Works

### 1. Parallel Execution Flow

When you run `run_parallel.py`:

1. **Setup Phase**: 
   - Loads environment variables from `.env` file if present
   - Sets up isolated URLs for each WebArena service
   - Creates timestamped result directory
   - Loads task configurations from `config_files/{task_id}.json`

2. **Docker Isolation**:
   - Each task gets unique port allocation (base_port + task_id * 100)
   - Isolated container environments prevent conflicts
   - Automatic cleanup of resources after execution

3. **Agent Execution**:
   - Activates virtual environment: `env/webarena-env/bin/activate`
   - Runs `browser_env/run.py` for each task with isolated config
   - Uses prompt template: `agent/prompts/jsons/p_cot_id_actree_2s.json`
   - 10-minute timeout per task

4. **Results Collection**:
   - Individual task results in timestamped directories
   - HTML render files for debugging: `render_*.html`
   - Execution traces in `traces/` subdirectories
   - Comprehensive JSON results file

### 2. Task Configuration Structure

Each task is defined in `config_files/{task_id}.json`:

```json
{
  "sites": ["shopping_admin"],
  "task_id": 78,
  "require_login": true,
  "storage_state": "./.auth/shopping_admin_state.json",
  "start_url": "http://localhost:7780/admin",
  "intent": "What is the total count of Approved reviews amongst all the reviews?",
  "eval": {
    "eval_types": ["string_match"],
    "reference_answers": {
      "must_include": ["346"]
    }
  }
}
```

### 3. Agent Execution Process

When `browser_env/run.py` is called:

1. **Environment Setup**:
   - Creates `ScriptBrowserEnv` with Playwright browser
   - Loads task configuration and intent
   - Handles authentication via storage state files

2. **Agent Loop**:
   - Agent analyzes current page state (accessibility tree)
   - Generates next action using LLM
   - Executes action in browser
   - Continues until task completion or timeout

3. **Evaluation**:
   - Compares final result against reference answers
   - Scores tasks (0.0 = fail, 1.0 = pass)
   - Saves trajectory and HTML snapshots

### 4. Docker Isolation System

The `DockerIsolationManager` provides:

- **Port Allocation**: Each task gets 100-port range (e.g., task 78 = ports 17800-17899)
- **Container Management**: Isolated instances of shopping, reddit, gitlab, wikipedia
- **Authentication Isolation**: Separate auth directories per task
- **Automatic Cleanup**: Removes containers and temp files after execution

Container mappings:
- Shopping: `localhost:{base_port}` 
- Shopping Admin: `localhost:{base_port+10}/admin`
- Reddit: `localhost:{base_port+20}`
- GitLab: `localhost:{base_port+30}`
- Wikipedia: `localhost:{base_port+40}`

## Results and Storage

### Directory Structure
```
parallel_demo_results/
└── 20250731_120820_openai_gpt_4_1_2025_04_14/
    ├── parallel_demo_results.json    # Main results
    ├── README.md                     # Generated documentation
    └── task_78/
        ├── config.json               # Task configuration
        ├── render_0.html            # Browser snapshots
        ├── traces/                  # Execution traces
        └── log_files.txt           # Execution logs
```

### Results Analysis

The system provides comprehensive analytics:
- **Success Rate**: Percentage of tasks passing evaluation
- **Average Score**: Mean score across all tasks
- **Parallel Efficiency**: Speedup compared to sequential execution
- **Thread Utilization**: Which tasks ran on which threads
- **Detailed Logs**: Per-task execution traces and errors

### Sample Results JSON
```json
{
  "timestamp": "20250731_120820",
  "model": "gpt-4.1-2025-04-14",
  "provider": "openai",
  "summary": {
    "success_rate": 0.67,
    "avg_score": 0.67,
    "successful_tasks": 2,
    "total_tasks": 3
  },
  "parallel_efficiency": 2.8,
  "detailed_results": [...],
  "thread_utilization": {...}
}
```

## WebArena Sites and Tasks

The system supports 6 different web applications:

1. **Shopping** (OneStopShop) - E-commerce site with product catalog
2. **Shopping Admin** - Magento admin panel for e-commerce management  
3. **Reddit** - Forum-style social platform
4. **GitLab** - Version control and project management
5. **Wikipedia** - Knowledge base with search functionality
6. **Map** - Interactive mapping service

### Task Types by ID Ranges
- **0-99**: Basic navigation and interaction tasks
- **100-299**: Shopping and e-commerce tasks
- **300-499**: Forum and social tasks  
- **500-699**: Development and GitLab tasks
- **700-812**: Wikipedia and research tasks

## Agent Architecture

### Agent Types
1. **PromptAgent**: Uses LLM with structured prompts
2. **TeacherForcingAgent**: Follows predefined action sequences
3. **Agent**: Base class for custom implementations

### Action Types
- **Click**: Click on elements by ID
- **Type**: Enter text in input fields
- **Key**: Send keyboard inputs
- **Scroll**: Scroll page content
- **Stop**: End task execution

### Observation Types
- **Accessibility Tree**: Structured representation of page elements
- **HTML**: Raw HTML content
- **Image**: Visual screenshots (optional)

## Key Environment URLs

Default local URLs (overridden in parallel execution):
```bash
SHOPPING="http://localhost:7770"
SHOPPING_ADMIN="http://localhost:7780/admin" 
REDDIT="http://localhost:9999"
GITLAB="http://localhost:8023"
WIKIPEDIA="http://localhost:8888"
MAP="http://localhost:3000"
HOMEPAGE="http://localhost:4399"
```

## Troubleshooting

### Common Issues
1. **Docker Permission Errors**: Ensure Docker daemon is running and user has permissions
2. **Port Conflicts**: Check if ports in range are already in use
3. **API Key Issues**: Verify environment variables are set correctly
4. **Virtual Environment**: Ensure `env/webarena-env` exists and is activated
5. **Storage State**: Authentication files may need renewal

### Debugging Tools
- **HTML Renders**: Check `render_*.html` files for page snapshots
- **Execution Traces**: Review `traces/` directories for detailed logs
- **Error Logs**: Check `error.txt` files in result directories
- **Console Output**: Monitor real-time execution progress

### Performance Optimization
- **Concurrent Tasks**: System auto-detects optimal parallelization
- **Container Warmup**: Shopping admin containers need ~3 minutes startup time
- **Resource Cleanup**: Automatic cleanup prevents resource leaks
- **Timeout Management**: 10-minute timeout per task prevents hanging

## File Locations Summary

- **Main Scripts**: `run_parallel.py`, `browser_env/run.py`
- **Agent Code**: `agent/agent.py`, `agent/prompts/`
- **Docker Management**: `browser_env/docker_isolation_manager.py`
- **Task Configs**: `config_files/*.json` (812 files)
- **Results**: `parallel_demo_results/` (timestamped directories)
- **Authentication**: `.auth/` (login state files)
- **Virtual Environment**: `env/webarena-env/`
- **Dependencies**: `requirements.txt`

This system provides a robust, scalable platform for evaluating AI agents across realistic web tasks with proper isolation and comprehensive result tracking.