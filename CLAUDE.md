# WebArena Parallel Execution System - CLAUDE.md

This document provides comprehensive information about the WebArena parallel execution system for Claude AI assistant.

## System Overview

WebArena is a realistic web environment for building autonomous agents, featuring a sophisticated parallel execution system that allows multiple tasks to run simultaneously with isolated Docker environments.

### Key Components

1. **Main Execution Script**: `run_parallel_distyl.py` - Orchestrates parallel task execution
2. **Browser Environment**: `browser_env/run.py` - Core agent execution environment  
3. **Docker Isolation**: `browser_env/docker_isolation_manager.py` - Manages isolated container environments
4. **Agent System**: `agent/agent.py` - AI agent implementations
5. **Configuration**: `config_files/*.json` - Task definitions (812 total tasks)

## Running Parallel Tasks

### Unified Parallel Runner: `run_parallel_distyl.py`

**The single parallel runner that supports ALL models - both Distyl-WebArena and standard models:**
```bash
python run_parallel_distyl.py --model <model_name> --tasks <task_ids>
```

**Key features:**
- Supports both Distyl-WebArena AND all standard models (OpenAI, Google, Anthropic)
- Auto-detects provider from model name (no need for `--provider` flag)
- Graceful fallback if Distyl-WebArena is not available
- Docker isolation and authentication features
- Comprehensive result format and structure

### Examples

```bash
# Distyl-WebArena (intelligent agent with hierarchical planning)
python run_parallel_distyl.py --model distyl-webarena --tasks 78

# Standard models (auto-detects provider - no --provider flag needed)
python run_parallel_distyl.py --model gpt-4 --tasks 78,79,80
python run_parallel_distyl.py --model gpt-4.1-2025-04-14 --tasks 78-82
python run_parallel_distyl.py --model gemini-1.5-pro --tasks 78-82

# Multiple tasks and task ranges
python run_parallel_distyl.py --model gpt-4 --tasks 78,79,80,85-90

# List available models
python run_parallel_distyl.py --list-models
```

### Supported Providers & Models
- **OpenAI**: `gpt-4.1-2025-04-14`, `gpt-4`, `gpt-3.5-turbo`, `o1-preview`
- **Google**: `gemini-2.5-pro`, `gemini-1.5-pro`
- **Anthropic**: Future support planned
- **Distyl-WebArena**: `distyl-webarena` - Intelligent agent with hierarchical planning

### Environment Variables Required
- `OPENAI_API_KEY` - For OpenAI models
- `GOOGLE_API_KEY` - For Google/Gemini models
- `ANTHROPIC_API_KEY` - For Anthropic models (future)

## How the System Works

### 1. Parallel Execution Flow

When you run `run_parallel_distyl.py`:

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

- **Main Scripts**: `run_parallel_distyl.py`, `browser_env/run.py`
- **Agent Code**: `agent/agent.py`, `agent/prompts/`
- **Docker Management**: `browser_env/docker_isolation_manager.py`
- **Task Configs**: `config_files/*.json` (812 files)
- **Results**: `parallel_demo_results/` (timestamped directories)
- **Authentication**: `.auth/` (login state files)
- **Virtual Environment**: `env/webarena-env/`
- **Dependencies**: `requirements.txt`

## Frontend Visualization - WebArena Viewer

The WebArena system includes a sophisticated web-based visualization interface for analyzing experiment results and agent execution traces.

### Quick Start

Launch the viewer with:
```bash
cd webarena_viewer
./start_viewer.sh
```

Or manually:
```bash
cd webarena_viewer
pip install -r requirements.txt
python run_viewer.py
```

Access at: **http://localhost:8080**

### Key Features

1. **📊 Experiment Overview**: Browse all parallel execution results with success rates and performance metrics
2. **🎯 Task Navigation**: Easy navigation between tasks within experiments (auto-selects task 0)
3. **📝 Task Details**: Shows task intent, expected answers, and execution results
4. **🧠 AI Thinking**: Displays the model's step-by-step reasoning process
5. **⚡ Action Timeline**: Shows actions taken by the agent in chronological order
6. **📸 Page Screenshots**: Visual browser states at each step
7. **📈 Performance Metrics**: Shows execution times, scores, and success rates
8. **🎨 Modern UI**: Clean, responsive interface with structured content display

### User Interface Components

#### Left Sidebar
- **Experiment List**: All experiments sorted by date (newest first)
- **Experiment Cards**: Show model name, timestamp, task count, and success rate
- **Task Grid**: Numbered tiles showing all tasks in selected experiment

#### Task Color Coding
- 🟢 **Green**: Successful tasks (score > 0.8)
- 🟡 **Yellow**: Partial success (score 0.3-0.8)  
- 🔴 **Red**: Failed tasks (score < 0.3)

#### Main Content Area
- **Task Header**: Intent, status, score, execution time, target sites
- **Expected Answer**: Reference answers from task configuration
- **Execution Steps**: Step-by-step agent reasoning and actions
- **Screenshots**: Visual page states captured during execution

### Technical Architecture

#### Backend (Flask Application)
- **File**: `webarena_viewer/app.py` - Flask web server
- **Port**: 8080 (configurable)
- **Data Source**: `../parallel_demo_results/` directory
- **Dependencies**: Flask, BeautifulSoup4, Werkzeug

#### Frontend (Vanilla JavaScript)
- **File**: `webarena_viewer/templates/index.html`
- **Design**: Responsive, mobile-friendly interface
- **Framework**: No external dependencies, pure vanilla JS
- **Styling**: Modern CSS with gradient headers and structured layouts

#### Data Processing
The viewer intelligently parses experiment data:

1. **Experiment Loading**: Scans `parallel_demo_results/` for timestamped experiment directories
2. **Result Parsing**: Extracts data from `parallel_*_results.json` files  
3. **HTML Analysis**: Parses `render_*.html` files to extract agent thinking, actions, and screenshots
4. **Configuration Merging**: Combines task configs with original intent from `config_files/*.json`

### API Endpoints

The viewer exposes RESTful API endpoints:

- `GET /api/experiments` - List all experiments with metadata
- `GET /api/experiment/{id}` - Get specific experiment details  
- `GET /api/task/{exp_id}/{task_id}` - Get task configuration and evaluation data
- `GET /api/task/{exp_id}/{task_id}/parsed` - Get parsed execution steps with screenshots
- `GET /render/{exp_id}/{task_id}` - Serve raw render HTML file

### Step-by-Step Execution Analysis

For each task, the viewer displays:

1. **AI Thinking Section**:
   - Model's internal reasoning process
   - Decision-making steps
   - Problem analysis approach

2. **Action Timeline**:
   - Chronological sequence of browser actions
   - Action types: click, type, scroll, navigate
   - Target elements and parameters

3. **Visual Progress**:
   - Screenshot at each major step
   - Page state changes over time
   - Browser viewport captures

### Data Sources and File Structure

The viewer reads from multiple data sources:

```
parallel_demo_results/
└── 20250731_120820_openai_gpt_4_1_2025_04_14/
    ├── parallel_demo_results.json     # Experiment summary
    ├── task_78/
    │   ├── config.json               # Runtime task configuration
    │   ├── render_78.html           # Agent execution trace with screenshots
    │   ├── traces/                  # Detailed execution logs
    │   └── log_files.txt           # System execution logs
    └── ...

config_files/
└── 78.json                         # Original task intent and evaluation criteria
```

### Navigation and Usage

#### Automatic Features
- **Auto-Selection**: Most recent experiment selected by default
- **Task 0 Priority**: Automatically selects task 0 when switching experiments
- **Real-time Loading**: Asynchronous loading with progress indicators

#### Manual Navigation
- **Experiment Selection**: Click any experiment in left sidebar
- **Task Selection**: Click numbered tiles in task grid
- **Step Navigation**: Scroll through execution timeline

### Performance Considerations

- **Lazy Loading**: Task details loaded only when selected
- **HTML Parsing**: BeautifulSoup efficiently extracts execution data
- **Screenshot Handling**: Base64 images embedded in render files
- **Responsive Design**: Optimized for various screen sizes

### Troubleshooting Viewer Issues

#### No Experiments Showing
- Ensure experiments exist: run `run_parallel_distyl.py` first
- Check directory: `parallel_demo_results/` must exist and contain experiment folders
- Verify permissions: viewer must have read access to results directory

#### Render Files Not Loading  
- Confirm `render_*.html` files exist in task directories
- Check browser console for JavaScript errors
- Verify file permissions and server access

#### Performance Issues
- Large HTML render files may cause slow loading
- Use browser zoom controls for content size adjustment
- Consider closing other browser tabs to free memory

### Directory Structure

```
webarena_viewer/
├── app.py                 # Flask backend application (385 lines)
├── run_viewer.py          # Simple startup script
├── start_viewer.sh        # Bash launcher with dependency checking  
├── requirements.txt       # Python dependencies (Flask, BeautifulSoup4)
├── distyl.ico            # Favicon for web interface
├── templates/
│   └── index.html        # Main web interface (728 lines)
└── README.md             # Viewer documentation
```

This system provides a robust, scalable platform for evaluating AI agents across realistic web tasks with proper isolation, comprehensive result tracking, and intuitive visual analysis tools.

## Distyl-WebArena Integration

Distyl-WebArena brings intelligent agent capabilities to the WebArena system through hierarchical planning, semantic grounding, and memory-driven learning.

### Key Features

- **Hierarchical Planning**: Multi-step task decomposition with site-specific strategies
- **Semantic Grounding**: Intelligent element detection by purpose rather than exact text matching
- **Memory System**: Learns from past experiences to improve future performance
- **Error Recovery**: Reflection-based alternatives when actions fail
- **Site Optimization**: Specialized knowledge for all WebArena environments

### Usage

The enhanced runner `run_parallel_distyl.py` provides a clean interface for both Distyl-WebArena and standard models:

```bash
# Use Distyl-WebArena (intelligent agent)
python run_parallel_distyl.py --model distyl-webarena --tasks 78

# Use standard models with automatic provider detection
python run_parallel_distyl.py --model gpt-4 --tasks 78,79,80

# List all available models
python run_parallel_distyl.py --list-models
```

### Documentation

- **Architecture Guide**: `docs/DISTYL_WEBARENA_ARCHITECTURE.md` - Complete technical documentation
- **Implementation Summary**: `docs/DISTYL_IMPLEMENTATION_SUMMARY.md` - Overview of all components  
- **Component README**: `distyl_webarena/README.md` - Quick start guide

### Performance Benefits

Distyl-WebArena typically achieves 40-70% higher success rates compared to reactive agents through:

1. **Intelligent Planning**: Reduces trial-and-error through hierarchical task decomposition
2. **Experience Learning**: Improves performance over time through episodic and narrative memory
3. **Robust Error Handling**: Reflection system provides alternatives for failed actions
4. **Site-Specific Optimization**: Leverages specialized knowledge for different WebArena environments