"""
Distyl-WebArena: Hierarchical Web Automation Agent

This package adapts the Distyl agent architecture from OSWorld desktop automation
to WebArena web browser automation, providing sophisticated planning, execution,
and learning capabilities for web-based tasks.

Key Components:
- Controller: Main orchestrator with WebArena interface compatibility
- Planner: Web-specific task decomposition with DAG generation
- Executor: Browser action generation with reflection
- Grounder: Accessibility tree-based element grounding
- Actions: Web action system with site-specific patterns
- Memory: Knowledge base for web interaction learning
"""

from .controller.controller import DistylWebArenaController

__version__ = "1.0.0"
__author__ = "Distyl WebArena Team"

__all__ = [
    "DistylWebArenaController"
]