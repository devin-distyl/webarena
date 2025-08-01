# WebArena Experiment Viewer

A beautiful web interface for viewing and analyzing WebArena parallel execution results.

## Features

- 📊 **Experiment Overview**: View all experiments with success rates and performance metrics
- 🎯 **Task Navigation**: Easy navigation between tasks within experiments (auto-selects task 0)
- 📝 **Task Details**: Shows task intent, expected answers, and execution results
- 🧠 **AI Thinking**: Displays the model's step-by-step reasoning process
- ⚡ **Action Timeline**: Shows actions taken by the agent in chronological order
- 📸 **Page Screenshots**: Visual browser states at each step
- 📈 **Performance Metrics**: Shows execution times, scores, and success rates
- 🎨 **Modern UI**: Clean, responsive interface with structured content display

## Quick Start

1. **Run the Viewer**:
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

2. **Open in Browser**:
   Visit [http://localhost:8080](http://localhost:8080)

## How to Use

### Viewing Experiments
- **Left Sidebar**: Lists all experiments sorted by date (newest first)
- **Experiment Cards**: Show model, timestamp, task count, and success rate
- **Click to Select**: Click any experiment to view its tasks

### Browsing Tasks
- **Task Grid**: Shows all tasks as numbered tiles
- **Color Coding**:
  - 🟢 **Green**: Successful tasks (score > 0.8)
  - 🟡 **Yellow**: Partial success (score 0.3-0.8)  
  - 🔴 **Red**: Failed tasks (score < 0.3)

### Task Details
When you select a task, you'll see:
- **Task Intent**: The natural language instruction given to the agent
- **Execution Metrics**: Status, score, execution time, and target sites
- **Evaluation Details**: Reference answers and evaluation criteria
- **Visual Result**: The rendered HTML showing the final browser state

### Navigation
- **Experiment Selection**: Click any experiment in the left sidebar
- **Task Selection**: Click any task number in the task grid
- **Auto-Selection**: Most recent experiment is selected by default

## Directory Structure

```
webarena_viewer/
├── app.py              # Flask backend application
├── run_viewer.py       # Simple startup script
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Main web interface
└── README.md          # This file
```

## Data Source

The viewer reads experiment data from:
- `../parallel_demo_results/` - Main results directory
- `{experiment_id}/parallel_*_results.json` - Experiment summaries
- `{experiment_id}/task_{id}/render_*.html` - Browser state captures
- `../config_files/{task_id}.json` - Original task configurations

## Troubleshooting

### No Experiments Showing
- Ensure you have run some experiments using `run_parallel_distyl.py`
- Check that `parallel_demo_results/` directory exists and contains experiment folders

### Render Files Not Loading
- Verify that tasks have `render_*.html` files in their directories
- Check browser console for any loading errors

### Performance Issues
- Large render HTML files may take time to load
- Use browser zoom controls if content appears too large/small

## Technical Details

- **Backend**: Flask web framework
- **Frontend**: Vanilla JavaScript with modern CSS
- **No External Dependencies**: Self-contained viewer
- **Responsive Design**: Works on desktop and mobile devices
- **Cross-Platform**: Runs on Windows, macOS, and Linux

## API Endpoints

The viewer provides REST API endpoints:

- `GET /api/experiments` - List all experiments
- `GET /api/experiment/{id}` - Get experiment details  
- `GET /api/task/{exp_id}/{task_id}` - Get task configuration
- `GET /render/{exp_id}/{task_id}` - Serve render HTML file

You can use these endpoints to build custom integrations or analysis tools.