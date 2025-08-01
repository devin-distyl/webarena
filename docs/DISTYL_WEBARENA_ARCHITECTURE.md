# Distyl-WebArena Architecture

## Overview

Distyl-WebArena is a sophisticated adaptation of the Distyl agent architecture from OSWorld (desktop automation) to WebArena (browser automation). This document provides comprehensive details about the agent architecture, implementation, and integration with the WebArena parallel execution system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Key Adaptations from OSWorld](#key-adaptations-from-osworld)
4. [Component Details](#component-details)
5. [Integration with WebArena](#integration-with-webarena)
6. [Memory System](#memory-system)
7. [Performance Optimizations](#performance-optimizations)
8. [Usage Instructions](#usage-instructions)
9. [Directory Structure](#directory-structure)
10. [API Reference](#api-reference)

## Architecture Overview

Distyl-WebArena maintains the hierarchical planning approach from the original Distyl system while adapting all components for web browser automation. The architecture consists of six main components working together to achieve intelligent web automation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DistylWebArenaController                     │
│                    (Agent Interface)                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      WebStepPlanner                             │
│               (Hierarchical Task Planning)                      │
│  • Site-specific planning logic                                │
│  • DAG-based task decomposition                               │
│  • Multi-step workflow generation                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      WebExecutor                                │
│                (Action Generation & Execution)                  │
│  • Subtask-to-action conversion                               │
│  • Reflection-based error recovery                            │
│  • Action validation and feedback                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                AccessibilityTreeGrounder                        │
│                 (Element Detection & Grounding)                 │
│  • Semantic element detection                                 │
│  • Auto-element resolution                                    │
│  • Context-aware element matching                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   WebActionSystem                               │
│                    (Action Translation)                         │
│  • High-level action to WebArena action conversion            │
│  • Site-specific action libraries                             │
│  • Action parameter resolution                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  WebKnowledgeBase                               │
│               (Memory & Experience Storage)                     │
│  • Episodic memory (subtask-level)                            │
│  • Narrative memory (task-level)                              │
│  • Site pattern recognition                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. DistylWebArenaController (`controller/controller.py`)

The main agent controller that implements the WebArena `Agent` interface while internally managing the sophisticated Distyl architecture.

**Key Features:**
- **WebArena Compatibility**: Implements `next_action(trajectory, intent, meta_data)` interface
- **Hierarchical Management**: Orchestrates planner, executor, grounder, and memory components  
- **State Management**: Tracks current task progress and subtask execution
- **Best-of-N Selection**: Generates multiple action candidates and selects the best one

**Interface Adaptation:**
```python
class DistylWebArenaController(Agent):
    def next_action(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> Action:
        # Convert WebArena interface to internal Distyl workflow
        # Returns single Action compatible with browser_env/run.py
```

### 2. WebStepPlanner (`planner/web_steps.py`)

Hierarchical task planner that decomposes high-level intents into executable subtasks with understanding of web interaction patterns.

**Key Features:**
- **Site-Aware Planning**: Different planning strategies for shopping, social, development, knowledge sites
- **DAG Generation**: Creates directed acyclic graphs for task dependencies
- **Pattern Recognition**: Uses web interaction patterns for better planning
- **Context Analysis**: Analyzes current page state to inform planning decisions

**Planning Strategies by Site:**
- **Shopping**: Product search → Filter → Selection → Cart → Checkout workflows
- **Social**: Navigation → Content creation → Interaction workflows  
- **Development**: Repository → Issue management → Code workflows
- **Knowledge**: Search → Article navigation → Information extraction workflows

### 3. WebExecutor (`executor/web_execution.py`)

Converts planned subtasks into specific web actions with reflection-based error recovery.

**Key Features:**
- **Action Generation**: Converts subtask descriptions to specific web actions
- **Validation Pipeline**: Validates actions before execution
- **Reflection System**: Learns from failures and suggests alternatives
- **Memory Integration**: Uses past experiences to improve action selection

**Execution Flow:**
1. Retrieve similar subtask experiences from memory
2. Generate action plan using WebActionCodeGenerator
3. Ground action parameters through AccessibilityTreeGrounder
4. Validate action using ActionValidator
5. Execute with reflection-based error recovery if needed

### 4. AccessibilityTreeGrounder (`grounder/web_grounding.py`)

Intelligent element detection system that converts semantic element descriptions to specific element IDs from the accessibility tree.

**Key Features:**
- **Semantic Matching**: Finds elements by semantic description rather than exact IDs
- **Auto-Detection**: Resolves `auto_detect_*` placeholders to actual elements
- **Context Awareness**: Uses page context and site type for better matching
- **Fuzzy Matching**: Handles variations in element text and attributes

**Element Detection Types:**
- **Direct Matching**: Exact text or attribute matches
- **Semantic Matching**: Understanding of element purpose (e.g., "submit button")
- **Pattern Recognition**: Site-specific element patterns
- **Fallback Strategies**: Multiple detection approaches for robustness

### 5. WebActionSystem (`actions/web_actions.py`)

Translates high-level actions into WebArena-compatible action strings with site-specific optimizations.

**Key Features:**
- **Action Translation**: Converts descriptions to WebArena action format
- **Site-Specific Libraries**: Optimized actions for different WebArena sites
- **Parameter Resolution**: Handles action parameters and element IDs
- **Action Validation**: Ensures generated actions are properly formatted

**Supported Action Types:**
- **Click Actions**: `click [element_id]`
- **Type Actions**: `type [element_id] [text] [enter_flag]`
- **Navigation**: `goto [url]`, `scroll [direction]`
- **Keyboard**: `press [key_combination]`

### 6. WebKnowledgeBase (`memory/web_knowledge.py`)

Sophisticated memory system that stores and retrieves experiences from web task execution.

**Key Features:**
- **Episodic Memory**: Subtask-level experiences and patterns
- **Narrative Memory**: Task-level summaries and outcomes
- **Site Patterns**: Site-specific interaction patterns and workflows
- **Knowledge Persistence**: File-based storage with JSON serialization

## Key Adaptations from OSWorld

The original Distyl system was designed for desktop automation using PyAutoGUI and pixel coordinates. Distyl-WebArena adapts these concepts for web browser automation:

### Environment Differences

| Aspect | OSWorld (Original) | WebArena (Adapted) |
|--------|-------------------|-------------------|
| **Environment** | Desktop applications | Web browsers |
| **Input Method** | PyAutoGUI coordinates | Element IDs from accessibility tree |
| **Observation** | Screenshots | HTML/Accessibility tree text |
| **Actions** | Pixel-based clicks/types | Element-based interactions |
| **Grounding** | Visual object detection | Semantic element matching |
| **Context** | Desktop windows | Web pages and sites |

### Architecture Adaptations

1. **Visual → Semantic Grounding**
   - **Original**: Object detection on screenshots to find clickable regions
   - **Adapted**: Accessibility tree parsing to find semantic elements

2. **Coordinate → Element-Based Actions**
   - **Original**: `click(x, y)` pixel coordinates
   - **Adapted**: `click [element_id]` using accessibility tree IDs

3. **Screenshot → HTML Analysis**
   - **Original**: Computer vision on desktop screenshots
   - **Adapted**: Text parsing of accessibility tree structures

4. **Desktop → Web Site Awareness**
   - **Original**: General desktop application patterns
   - **Adapted**: Site-specific web patterns (shopping, social, development, etc.)

## Component Details

### DistylWebArenaController

**File**: `distyl_webarena/controller/controller.py`

The controller acts as the bridge between WebArena's simple `Agent` interface and Distyl's sophisticated multi-component architecture.

**Initialization Parameters:**
```python
DistylWebArenaController(
    engine_params: Dict[str, Any],           # Model/engine configuration
    memory_folder_name: str = "distyl_kb",   # Memory storage location
    enable_reflection: bool = True,          # Enable reflection system
    enable_memory: bool = True,              # Enable memory system
    n_candidates: int = 1                    # Number of action candidates
)
```

**Internal Workflow:**
1. **Task Analysis**: Analyzes the intent and current trajectory
2. **Planning Phase**: Uses WebStepPlanner to decompose task into subtasks
3. **Execution Phase**: Uses WebExecutor to generate specific actions
4. **Memory Integration**: Stores experiences and retrieves relevant knowledge
5. **Action Selection**: Returns best action to WebArena system

### WebStepPlanner

**File**: `distyl_webarena/planner/web_steps.py`

The planner creates hierarchical plans by analyzing the task intent and current web context.

**Context Analysis:**
```python
def _analyze_web_context(self, observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "site_type": self._classify_site_type(url),           # shopping, social, etc.
        "current_page": self._identify_page_type(tree),       # login, search, admin, etc.
        "available_actions": self._extract_available_actions(tree),  # click_button, type_text, etc.
        "login_status": self._detect_login_status(tree),      # logged_in, logged_out, unknown
        "url": url
    }
```

**Site-Specific Planning:**
- **Shopping Sites**: Focus on product search, cart management, checkout flows
- **Social Sites**: Emphasize content creation, voting, commenting workflows  
- **Development Sites**: Repository browsing, issue management, code operations
- **Knowledge Sites**: Article search, information extraction, navigation

### AccessibilityTreeGrounder

**File**: `distyl_webarena/grounder/web_grounding.py`

The grounder converts natural language element descriptions into specific element IDs.

**Element Detection Strategies:**
1. **Direct Text Matching**: Find elements containing specific text
2. **Attribute Matching**: Match based on element attributes (placeholder, aria-label, etc.)
3. **Type-Based Matching**: Find elements by type (button, textbox, link)
4. **Context-Aware Matching**: Use surrounding elements for disambiguation
5. **Semantic Understanding**: Interpret element purpose (e.g., "submit button", "search field")

**Auto-Detection Resolution:**
```python
# Input: "click [auto_detect_submit_button]"
# Process: Find buttons with submit-related text/attributes
# Output: "click [245]" (actual element ID)
```

### WebActionSystem

**File**: `distyl_webarena/actions/web_actions.py`

Translates high-level action descriptions into WebArena-compatible action strings.

**Site-Specific Action Libraries:**
- **Shopping Actions**: Product search, cart operations, checkout procedures
- **Social Actions**: Post creation, voting, commenting, navigation
- **Development Actions**: Repository browsing, issue management, code editing
- **Knowledge Actions**: Article search, link following, information extraction

**Action Generation Pipeline:**
1. **Description Analysis**: Parse the action description for intent
2. **Context Integration**: Consider current page state and site type
3. **Template Matching**: Match against site-specific action patterns
4. **Parameter Extraction**: Extract relevant parameters (search terms, URLs, etc.)
5. **Format Conversion**: Convert to WebArena action string format

### WebKnowledgeBase

**File**: `distyl_webarena/memory/web_knowledge.py`

The memory system stores and retrieves experiences to improve future performance.

**Memory Types:**

1. **Episodic Memory** (`episodic/subtask_experiences.json`):
   - Stores individual subtask execution experiences
   - Includes success/failure information and strategies used
   - Enables learning from past subtask attempts

2. **Narrative Memory** (`narrative/task_experiences.json`):
   - Stores high-level task completion summaries
   - Organized by site type for relevant retrieval
   - Includes successful workflow patterns

3. **Site Patterns** (`site_patterns/patterns.json`):
   - Common interaction patterns for each site type
   - Element naming conventions and layout patterns
   - Workflow templates for typical tasks

**Memory Storage Structure:**
```
distyl_webarena_memory/
├── episodic/
│   └── subtask_experiences.json    # Individual subtask experiences
├── narrative/
│   └── task_experiences.json       # Task-level summaries
├── site_patterns/
│   └── patterns.json              # Site-specific patterns
└── web_knowledge/
    └── cached_knowledge.json      # Web search results (future)
```

## Integration with WebArena

### WebArena Compatibility

Distyl-WebArena maintains full compatibility with the existing WebArena infrastructure:

1. **Agent Interface**: Implements the standard `Agent` class with `next_action()` method
2. **Action Format**: Returns actions in WebArena's expected string format
3. **Observation Handling**: Processes WebArena's accessibility tree observations
4. **Trajectory Compatibility**: Works with WebArena's trajectory tracking system

### Parallel Execution Integration

**File**: `distyl_webarena/integration/parallel_integration.py`

Provides seamless integration with `run_parallel.py`:

```bash
# Create enhanced run_parallel that supports Distyl-WebArena
python -m distyl_webarena.integration.parallel_integration

# Run parallel tasks with Distyl-WebArena
python run_parallel_distyl.py --model distyl-webarena --tasks 78-80
```

**Integration Features:**
- **Automatic Setup**: Creates modified `run_parallel_distyl.py` with Distyl support
- **Configuration Compatibility**: Works with existing task configurations
- **Docker Isolation**: Compatible with WebArena's Docker isolation system
- **Result Tracking**: Integrates with WebArena's result tracking and analysis

### Browser Environment Adapter

**File**: `distyl_webarena/integration/webarena_adapter.py`

Provides factory functions for creating Distyl agents compatible with `browser_env/run.py`:

```python
from distyl_webarena.integration.webarena_adapter import create_distyl_agent_for_webarena

# Create agent for WebArena task
task_config = {"task_id": 78, "sites": ["shopping_admin"]}
agent = create_distyl_agent_for_webarena(task_config, "distyl-webarena")

# Agent implements WebArena Agent interface and can be used directly
```

## Memory System

### Episodic Memory

Stores detailed experiences from individual subtask executions:

```json
{
  "shopping_click_Navigate to admin panel": {
    "description": "Navigate to admin panel",
    "type": "navigation",
    "site_type": "shopping",
    "success": true,
    "summary": "Navigation subtask 'Navigate to admin panel' succeeded",
    "timestamp": 1690876543.21
  }
}
```

### Narrative Memory

Stores high-level task completion summaries organized by site:

```json
{
  "shopping_experiences": [
    {
      "task_id": 78,
      "intent": "What is the total count of Approved reviews?",
      "sites": ["shopping_admin"],
      "success": true,
      "subtask_count": 4,
      "summary": "Task successfully completed with 4 subtasks involving navigation, click actions.",
      "timestamp": 1690876543.21,
      "subtasks": ["Navigate to admin panel", "Access reviews section", ...]
    }
  ]
}
```

### Site Patterns

Stores common interaction patterns for each site type:

```json
{
  "shopping": {
    "common_workflows": [
      "search -> filter -> select -> add_to_cart -> checkout",
      "admin -> reports -> analyze -> extract"
    ],
    "element_patterns": {
      "search": ["searchbox", "query", "find"],
      "cart": ["cart", "bag", "basket"]
    }
  }
}
```

## Performance Optimizations

### 1. Site-Specific Optimization

- **Custom Planning**: Different planning strategies for each WebArena site
- **Action Libraries**: Pre-built action patterns for common workflows
- **Element Patterns**: Site-specific element naming and layout understanding

### 2. Memory-Driven Learning

- **Experience Reuse**: Leverages past successful experiences for similar tasks
- **Pattern Recognition**: Identifies and reuses successful workflow patterns
- **Error Avoidance**: Learns from past failures to avoid similar mistakes

### 3. Intelligent Element Detection

- **Semantic Grounding**: Finds elements by purpose rather than exact text matching
- **Context Awareness**: Uses page context to disambiguate similar elements
- **Fallback Strategies**: Multiple detection approaches for robustness

### 4. Reflection-Based Error Recovery

- **Failure Analysis**: Analyzes why actions fail and suggests alternatives
- **Alternative Generation**: Automatically generates alternative actions for failures
- **Learning Integration**: Incorporates failure experiences into memory system

## Usage Instructions

### Basic Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Single Task**:
   ```bash
   python run_parallel_distyl.py --model distyl-webarena --tasks 78
   ```

3. **Run Multiple Tasks**:
   ```bash
   python run_parallel_distyl.py --model distyl-webarena --tasks 78,79,80
   ```

### Advanced Configuration

1. **Custom Memory Location**:
   ```python
   controller = DistylWebArenaController(
       engine_params={"model": "distyl-webarena"},
       memory_folder_name="custom_memory_location"
   )
   ```

2. **Disable Reflection**:
   ```python
   controller = DistylWebArenaController(
       engine_params={"model": "distyl-webarena"},
       enable_reflection=False
   )
   ```

3. **Multiple Action Candidates**:
   ```python
   controller = DistylWebArenaController(
       engine_params={"model": "distyl-webarena"},
       n_candidates=3  # Generate 3 action candidates, select best
   )
   ```

### Integration with Custom WebArena Setup

1. **Direct Agent Creation**:
   ```python
   from distyl_webarena.integration.webarena_adapter import create_distyl_agent_for_webarena
   
   task_config = {"task_id": 78, "sites": ["shopping_admin"]}
   agent = create_distyl_agent_for_webarena(task_config)
   
   # Use with browser_env/run.py or custom execution system
   ```

2. **Custom Integration**:
   ```python
   from distyl_webarena.controller.controller import DistylWebArenaController
   
   # Create controller with custom parameters
   controller = DistylWebArenaController(engine_params=custom_params)
   
   # Use with WebArena trajectory system
   action = controller.next_action(trajectory, intent, meta_data)
   ```

## Directory Structure

```
distyl_webarena/
├── __init__.py                     # Package initialization
├── controller/                     # Main agent controller
│   ├── __init__.py
│   ├── controller.py              # DistylWebArenaController (main entry point)
│   └── interface_adapter.py       # WebArena interface compatibility
├── planner/                       # Task planning and decomposition
│   ├── __init__.py
│   ├── web_steps.py              # WebStepPlanner (hierarchical planning)
│   ├── web_patterns.py           # WebInteractionPatterns (pattern recognition)
│   └── site_planners.py          # Site-specific planning strategies
├── executor/                      # Action execution and reflection
│   ├── __init__.py
│   ├── web_execution.py          # WebExecutor (main execution logic)
│   ├── reflection.py             # WebReflectionAgent (error recovery)
│   └── action_validation.py      # ActionValidator (pre-execution validation)
├── grounder/                      # Element detection and grounding
│   ├── __init__.py
│   ├── web_grounding.py          # AccessibilityTreeGrounder (main grounding)
│   ├── element_detection.py      # ElementDetector (detection strategies)
│   └── multimodal_grounding.py   # MultimodalGrounder (future extension)
├── actions/                       # Action translation and execution
│   ├── __init__.py
│   ├── web_actions.py            # WebActionSystem (action generation)
│   ├── action_translator.py      # ActionTranslator (format conversion)
│   └── site_actions.py           # SiteSpecificActions (site-optimized actions)
├── memory/                        # Knowledge storage and retrieval
│   ├── __init__.py
│   ├── web_knowledge.py          # WebKnowledgeBase (main memory system)
│   ├── site_memory.py            # SiteMemoryPatterns (site-specific patterns)
│   └── web_rag.py                # WebRAGSystem (web knowledge retrieval)
├── integration/                   # WebArena system integration
│   ├── __init__.py
│   ├── webarena_adapter.py       # WebArenaAdapter (compatibility layer)
│   └── parallel_integration.py   # ParallelIntegration (run_parallel.py integration)
├── utils/                         # Utilities and helper functions
│   ├── __init__.py
│   ├── logging.py                # DistylLogger (logging system)
│   └── web_utils.py              # WebUtils (web-specific utilities)
├── config/                        # Configuration management
│   └── __init__.py
└── tests/                         # Test suite
    └── __init__.py
```

## API Reference

### DistylWebArenaController

**Main entry point for the Distyl-WebArena agent system.**

```python
class DistylWebArenaController(Agent):
    def __init__(self, engine_params: Dict[str, Any], 
                 memory_folder_name: str = "distyl_webarena_kb",
                 enable_reflection: bool = True,
                 enable_memory: bool = True, 
                 n_candidates: int = 1):
        """Initialize Distyl-WebArena controller"""
    
    def next_action(self, trajectory: Trajectory, intent: str, 
                   meta_data: Dict[str, Any]) -> Action:
        """Generate next action for WebArena system"""
    
    def reset(self):
        """Reset agent state for new task"""
```

### WebStepPlanner

**Hierarchical task planning with site-specific strategies.**

```python
class WebStepPlanner:
    def __init__(self, engine_params: Dict[str, Any], memory, n_candidates: int = 1):
        """Initialize web step planner"""
    
    def get_action_queue(self, instruction: str, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate action queue with DAG-based planning"""
    
    def reset(self):
        """Reset planner state"""
```

### AccessibilityTreeGrounder

**Intelligent element detection and grounding system.**

```python
class AccessibilityTreeGrounder:
    def __init__(self, enable_multimodal: bool = False):
        """Initialize accessibility tree grounder"""
    
    def ground_element_description(self, description: str, observation: Dict[str, Any]) -> str:
        """Ground element description to specific element ID"""
    
    def resolve_action_parameters(self, action: str, observation: Dict[str, Any]) -> str:
        """Resolve action parameters with element grounding"""
```

### WebExecutor

**Action execution with reflection and error recovery.**

```python
class WebExecutor:
    def __init__(self, engine_params: Dict[str, Any], grounder, memory, 
                 enable_reflection: bool = True):
        """Initialize web executor"""
    
    def next_action(self, subtask: Dict[str, Any], context: Dict[str, Any], 
                   trajectory: List[Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Generate next action for subtask"""
    
    def update_from_feedback(self, subtask: Dict[str, Any], action: str, 
                           success: bool, context: Dict[str, Any]):
        """Update executor based on action feedback"""
```

### WebKnowledgeBase

**Memory system for storing and retrieving web task experiences.**

```python
class WebKnowledgeBase:
    def __init__(self, memory_folder_name: str = "distyl_webarena_kb", 
                 enable_web_knowledge: bool = True):
        """Initialize web knowledge base"""
    
    def retrieve_site_experience(self, site_type: str, instruction: str) -> str:
        """Retrieve experience for specific site type"""
    
    def save_task_narrative(self, task_config: Dict[str, Any], 
                          subtasks_completed: List[Dict[str, Any]], 
                          success_status: bool):
        """Save task-level narrative experience"""
    
    def save_subtask_experience(self, subtask: Dict[str, Any], success: bool = True):
        """Save subtask-level episodic experience"""
```

### Integration Functions

**Factory functions for WebArena integration.**

```python
def create_distyl_agent_for_webarena(task_config: Dict[str, Any], 
                                   model_name: str = "distyl-webarena") -> Any:
    """Create Distyl agent compatible with browser_env/run.py"""

def get_distyl_model_info() -> Dict[str, Any]:
    """Get model information for run_parallel.py integration"""
```

## Conclusion

Distyl-WebArena successfully adapts the sophisticated Distyl architecture from desktop automation to web browser automation while maintaining the core benefits of hierarchical planning, intelligent grounding, and memory-driven learning. The system provides significant improvements over simple reactive agents through:

1. **Intelligent Planning**: Multi-step task decomposition with site-specific strategies
2. **Semantic Grounding**: Element detection by purpose rather than exact matching  
3. **Experience Learning**: Memory system that improves performance over time
4. **Error Recovery**: Reflection-based alternatives for failed actions
5. **Site Optimization**: Specialized knowledge for different WebArena environments

The architecture is designed for extensibility and can be enhanced with additional components like multimodal grounding, web search integration, and advanced reflection strategies as needed.