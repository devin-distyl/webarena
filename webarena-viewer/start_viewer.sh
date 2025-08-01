#!/bin/bash
# WebArena Experiment Viewer Launcher

echo "🌐 WebArena Experiment Viewer"
echo "============================="

# Get the script directory and change to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing Flask..."
    pip3 install flask
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Flask. Please install manually:"
        echo "   pip install flask"
        exit 1
    fi
fi

echo "🚀 Starting WebArena Experiment Viewer..."
echo "📁 Scanning for experiments in ../parallel_demo_results/"
echo "🌐 Open your browser to: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the viewer"
echo ""

# Run the viewer
python3 run_viewer.py