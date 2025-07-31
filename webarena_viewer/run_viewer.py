#!/usr/bin/env python3
"""
Simple script to run the WebArena Experiment Viewer
"""

import os
import sys
from pathlib import Path

def main():
    # Change to the viewer directory
    viewer_dir = Path(__file__).parent
    os.chdir(viewer_dir)
    
    # Check if Flask is installed
    try:
        import flask
    except ImportError:
        print("❌ Flask not found. Please install requirements:")
        print(f"   pip install -r {viewer_dir}/requirements.txt")
        return 1
    
    # Import and run the app
    try:
        from app import app, RESULTS_DIR
        
        print("🚀 Starting WebArena Experiment Viewer")
        print(f"📁 Results directory: {RESULTS_DIR}")
        print(f"🌐 Open in browser: http://localhost:8080")
        print("   Press Ctrl+C to stop")
        print()
        
        app.run(debug=False, host='0.0.0.0', port=8080)
        
    except KeyboardInterrupt:
        print("\n✅ Viewer stopped")
        return 0
    except Exception as e:
        print(f"❌ Error starting viewer: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())