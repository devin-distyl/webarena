# Distyl Agent Implementation: Comprehensive Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Agent Workflow](#agent-workflow)
5. [OSWorld Integration](#osworld-integration)
6. [Memory System](#memory-system)
7. [Best-of-N Selection](#best-of-n-selection)
8. [Key Design Decisions](#key-design-decisions)
9. [Reproduction Guide](#reproduction-guide)
10. [Future Enhancements](#future-enhancements)

## Overview

Distyl is a sophisticated desktop GUI automation agent designed for complex multi-step task execution in the OSWorld environment. The system implements a **modular, hierarchical architecture** that combines high-level planning with low-level action execution, enhanced by comprehensive memory management and best-of-N candidate selection.

### Key Characteristics
- **Multi-layer Architecture**: Planning → Execution → Grounding → Actions
- **Memory-Enhanced**: Episodic and narrative memory with RAG (Retrieval-Augmented Generation)
- **Self-Improving**: Best-of-N selection with parallel candidate evaluation
- **OSWorld Native**: Purpose-built for 3-layer virtualization environment
- **Knowledge Fusion**: Combines web search, past experience, and contextual reasoning

## Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────┐
│                     DISTYL AGENT SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│  Controller (Orchestration Layer)                          │
│  ├─ State Management & Flow Control                        │
│  ├─ Multi-turn Dialog Management                           │
│  └─ Best-of-N Configuration                                │
├─────────────────────────────────────────────────────────────┤
│  Planner (Strategic Layer)                                 │
│  ├─ StepPlanner: Natural language → DAG structure         │
│  ├─ Web Search & Knowledge Retrieval                       │
│  ├─ Experience-based Planning                              │
│  └─ DAG Translation & Topological Sorting                  │
├─────────────────────────────────────────────────────────────┤
│  Executor (Tactical Layer)                                 │
│  ├─ Action Generation from Subtasks                        │
│  ├─ Context-aware Code Generation                          │
│  ├─ Reflection-based Error Correction                      │
│  └─ Trajectory Management                                  │
├─────────────────────────────────────────────────────────────┤
│  Grounder (Perceptual Layer)                              │
│  ├─ Visual Element Identification                          │
│  ├─ OCR-based Text Grounding                              │
│  ├─ Coordinate Translation                                 │
│  └─ Multi-modal Understanding                              │
├─────────────────────────────────────────────────────────────┤
│  Memory System (Knowledge Layer)                           │
│  ├─ Episodic Memory (Subtask-level)                       │
│  ├─ Narrative Memory (Task-level)                         │
│  ├─ Web Knowledge RAG                                      │
│  └─ Experience Fusion                                      │
├─────────────────────────────────────────────────────────────┤
│  Actions (Execution Layer)                                 │
│  ├─ PyAutoGUI Command Generation                           │
│  ├─ Platform-specific Implementations                      │
│  └─ Multi-platform Support (Linux/Windows/macOS)          │
└─────────────────────────────────────────────────────────────┘
```

### Integration with OSWorld
```
┌─────────────────────────────────────────────────────────────┐
│                      HOST SYSTEM                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              DOCKER CONTAINER                           │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │               QEMU VM                               │ │ │
│  │  │  ┌─────────────────────────────────────────────────┐ │ │ │
│  │  │  │            Ubuntu 22.04 + GNOME                │ │ │ │
│  │  │  │  • Flask API Server (:5000)                   │ │ │ │
│  │  │  │  • X Server (DISPLAY=:0)                      │ │ │ │
│  │  │  │  • Target Applications                        │ │ │ │
│  │  │  └─────────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  DISTYL AGENT ←→ DesktopEnv ←→ HTTP API ←→ VM               │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Controller (`distyl/controller/controller.py`)

**Purpose**: Main orchestrator that coordinates all agent components and manages the execution flow.

**Key Responsibilities**:
- **State Management**: Tracks current subtask, completed tasks, and failure states
- **Flow Control**: Implements the main predict() loop for incremental task execution
- **Component Coordination**: Integrates Planner → Executor → Environment workflow
- **Memory Management**: Handles trajectory tracking and knowledge base operations

**Core Methods**:
```python
def predict(instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
    """Main execution loop that returns info dict and actions"""
    
def reset() -> None:
    """Reset all component states for new task"""
    
def finalize_memory() -> None:
    """Save all accumulated knowledge and trajectories"""
```

**Design Motivation**: 
- **Agent-Distyl Compatibility**: Maintains identical interface for easy integration
- **Incremental Control**: Allows external systems to control execution step-by-step
- **Robust State Management**: Handles complex multi-turn dialogues and replanning

### 2. StepPlanner (`distyl/planner/steps.py`)

**Purpose**: Converts natural language instructions into structured execution plans using sophisticated DAG-based planning.

**Key Features**:
- **Multi-source Knowledge Fusion**: Combines web search, past experience, and contextual reasoning
- **DAG Generation**: Creates directed acyclic graphs for complex task dependencies
- **Screenshot-aware Planning**: Uses vision-language models for context-aware planning
- **Best-of-N Planning**: Generates multiple plan candidates and selects the best

**Planning Pipeline**:
```python
1. Knowledge Retrieval:
   - Formulate search queries from instruction + screenshot
   - Retrieve web knowledge via search_web()
   - Retrieve similar past experiences via memory system
   
2. Knowledge Fusion:
   - Combine web knowledge with past experiences
   - Use LLM to integrate information intelligently
   - Create enhanced instruction with context
   
3. Plan Generation:
   - Generate step-by-step natural language plan
   - Translate plan to JSON DAG structure
   - Perform topological sort for execution order
```

**Design Motivation**:
- **Context Awareness**: Screenshot analysis ensures plans are relevant to current state
- **Experience Integration**: Leverages past successes to improve planning quality
- **Dependency Management**: DAG structure handles complex task interdependencies

### 3. Executor (`distyl/executor/execution.py`)

**Purpose**: Converts high-level subtasks into specific actions using code generation and reflection.

**Core Architecture**:
- **Generator Agent**: Creates detailed execution plans with code
- **Reflection Agent**: Provides feedback on action success/failure
- **Grounder Integration**: Converts element descriptions to coordinates
- **Knowledge Retrieval**: Accesses episodic memory for similar subtasks

**Action Generation Process**:
```python
1. Context Building:
   - Retrieve similar subtask experiences
   - Perform web search for subtask-specific knowledge
   - Fuse knowledge sources for comprehensive context
   
2. Code Generation:
   - Generate structured response with analysis
   - Create Python code using agent API
   - Parse and validate generated code
   
3. Execution:
   - Ground element descriptions to coordinates
   - Execute PyAutoGUI commands
   - Handle errors and retries
```

**Design Motivation**:
- **Code-based Actions**: Provides precise, debuggable action specifications
- **Context-rich Generation**: Uses all available knowledge for better actions
- **Error Recovery**: Reflection system enables self-correction

### 4. Grounder (`distyl/grounder/grounding.py`)

**Purpose**: Bridges the gap between semantic element descriptions and pixel coordinates.

**Key Capabilities**:
- **Visual Grounding**: Uses vision-language models to locate UI elements
- **OCR Integration**: Leverages Tesseract for text-based element location
- **Multi-modal Understanding**: Combines visual and textual cues
- **Coordinate Translation**: Handles different screen resolutions

**Grounding Methods**:
```python
def generate_coords(phrase: str, obs: Dict) -> List[int]:
    """Visual grounding using LMM for general UI elements"""
    
def generate_text_coords(phrase: str, obs: Dict, alignment: str) -> List[int]:
    """OCR-based grounding for text elements with precise alignment"""
    
def assign_coordinates(plan: str, obs: Dict):
    """Parse action code and assign appropriate coordinates"""
```

**Design Motivation**:
- **Robust Element Location**: Multiple strategies ensure reliable element finding
- **Precision Control**: Different methods for different element types
- **Resolution Independence**: Coordinate scaling for different display sizes

### 5. Memory System (`distyl/memory/knowledge.py`)

**Purpose**: Sophisticated knowledge management with episodic and narrative memory.

**Memory Types**:

#### Episodic Memory (Subtask-level)
- **Storage**: Individual subtask execution trajectories  
- **Retrieval**: Embedding-based similarity search
- **Purpose**: Learn from similar subtask experiences
- **Format**: Summarized action sequences with outcomes

#### Narrative Memory (Task-level)  
- **Storage**: Complete task execution trajectories
- **Retrieval**: Contextual similarity matching
- **Purpose**: High-level task planning guidance
- **Format**: Strategic approach summaries

#### Web Knowledge RAG
- **Storage**: Cached web search results
- **Retrieval**: Query-based knowledge retrieval
- **Purpose**: External knowledge integration
- **Format**: Instructional content and tutorials

**Knowledge Operations**:
```python
def retrieve_episodic_experience(instruction: str) -> Tuple[str, str]:
    """Find most similar subtask experience"""
    
def retrieve_narrative_experience(instruction: str) -> Tuple[str, str]:  
    """Find most similar task experience"""
    
def knowledge_fusion(observation, instruction, web_knowledge, 
                    similar_task, experience) -> str:
    """Intelligently combine knowledge sources"""
```

**Design Motivation**:
- **Experience-driven Learning**: Improves performance through accumulated experience
- **Multi-granularity**: Both task and subtask level learning
- **Knowledge Integration**: Combines multiple information sources effectively

### 6. Action System (`distyl/executor/actions.py`)

**Purpose**: Platform-specific action implementations with consistent API.

**Action Categories**:

#### Basic Interactions
```python
def click(x: int, y: int, button: str = "left", clicks: int = 1) -> str
def type_text(text: str, enter: bool = False, overwrite: bool = False) -> str  
def scroll(x: int, y: int, clicks: int, shift: bool = False) -> str
def hotkey(*keys: str) -> str
```

#### Advanced Operations
```python
def drag_and_drop(x1: int, y1: int, x2: int, y2: int) -> str
def highlight_text_span(x1: int, y1: int, x2: int, y2: int) -> str
def switch_applications(app_code: str) -> str
def set_cell_values(cell_values: dict, app_name: str, sheet_name: str) -> str
```

#### Control Flow
```python
def wait(seconds: float) -> str
def done(value: Any = None) -> str  
def fail(reason: str = None) -> str
```

**Design Motivation**:
- **PyAutoGUI Integration**: Leverages mature automation library
- **Platform Abstraction**: Consistent API across different operating systems
- **Code Generation**: Returns executable Python strings for flexibility

## Agent Workflow

### Complete Execution Flow

```mermaid
graph TD
    A[Task Input] --> B[Controller.predict()]
    B --> C{Need Replan?}
    C -->|Yes| D[StepPlanner.get_action_queue()]
    C -->|No| E{Need Next Subtask?}
    
    D --> D1[Knowledge Retrieval]
    D1 --> D2[Plan Generation]  
    D2 --> D3[DAG Translation]
    D3 --> D4[Topological Sort]
    D4 --> E
    
    E -->|Yes| F[Get Next Subtask]
    E -->|No| G[Executor.next_action()]
    F --> G
    
    G --> G1[Retrieve Subtask Experience]
    G1 --> G2[Generate Action Plan]
    G2 --> G3[Ground Elements]
    G3 --> G4[Execute Action]
    
    G4 --> H{Action Result}
    H -->|DONE| I[Mark Subtask Complete]
    H -->|FAIL| J[Mark for Replan]
    H -->|Continue| K[Update Trajectory]
    
    I --> L{More Subtasks?}
    J --> C
    K --> M[Return Action]
    L -->|Yes| E
    L -->|No| N[Task Complete]
    M --> O[Environment Step]
    O --> P[Update Memory]
    P --> B
```

### Memory Integration Points

1. **Task Initialization**: `knowledge_base.initialize_task_trajectory()`
2. **Planning Phase**: Web search + experience retrieval + knowledge fusion
3. **Execution Phase**: Subtask experience retrieval + trajectory updates
4. **Completion Phase**: Memory summarization and storage

## OSWorld Integration

### Communication Protocol

**Environment Interface**:
```python
# OSWorld DesktopEnv provides:
obs = env.reset(task_config=example)  # Initialize task
obs, reward, done, info = env.step(action)  # Execute action
```

**Agent Interface**:
```python  
# Distyl Controller provides:
info, actions = agent.predict(instruction, obs)  # Generate action
agent.reset()  # Reset for new task
agent.finalize_memory()  # Save knowledge
```

**Data Flow**:
```
Instruction → Controller → Planner → DAG → Executor → Actions → OSWorld → Screenshot → Controller
```

### Key Integration Features

1. **Incremental Execution**: Step-by-step control allows OSWorld to monitor progress
2. **Screenshot Processing**: Vision-language models analyze OSWorld screenshots
3. **Error Handling**: Robust failure recovery and replanning
4. **Memory Persistence**: Knowledge accumulates across task executions

## Memory System

### Episodic Memory (Subtask Level)

**Storage Format**:
```json
{
  "Task: search for 'python tutorial'\nSubtask: Open web browser": {
    "summary": "Successfully opened Firefox browser using hotkey Ctrl+Alt+T, then typed 'firefox' and pressed Enter",
    "embedding": [0.1, 0.2, ...],
    "timestamp": "2025-08-01T10:30:00Z"
  }
}
```

**Retrieval Process**:
1. Embed current subtask instruction
2. Compare with stored embeddings using cosine similarity  
3. Return most similar experience excluding exact matches
4. Integrate retrieved knowledge into execution context

### Narrative Memory (Task Level)

**Storage Format**:
```json
{
  "search for python tutorial and bookmark best result": {
    "summary": "Task completed in 3 subtasks: opened browser, searched python tutorial, bookmarked top result. Key insight: use Ctrl+D for bookmarking",
    "embedding": [0.3, 0.1, ...],
    "timestamp": "2025-08-01T10:35:00Z"
  }
}
```

**Usage**:
- Retrieved during initial planning phase
- Provides high-level strategic guidance
- Influences overall task decomposition

### Web Knowledge RAG

**Query Formulation**:
```python
def formulate_query(instruction: str, observation: Dict) -> str:
    """Uses vision-language model to create targeted search queries"""
    # Analyzes screenshot + instruction to formulate specific search
    # Example: "calculator task" + screenshot → "how to use gnome calculator Ubuntu"
```

**Knowledge Integration**:
```python  
def knowledge_fusion(observation, instruction, web_knowledge, 
                    similar_task, experience) -> str:
    """Intelligently combines multiple knowledge sources"""
    # Weighs relevance of each source
    # Resolves conflicts between sources  
    # Creates coherent integrated guidance
```

## Best-of-N Selection

### Parallel Candidate Generation

**Implementation**:
```python
# Generate N candidates in parallel using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_candidates) as executor:
    future_to_index = {
        executor.submit(generate_candidate, i): i 
        for i in range(self.n_candidates)
    }
```

**Benefits**:
- **Improved Quality**: Select best from multiple attempts
- **Error Resilience**: Fallback candidates if primary fails
- **Diversity**: Temperature >0.3 ensures varied approaches
- **Efficiency**: Parallel generation minimizes latency

### Evaluation Strategies

**Format Compliance**:
- JSON validity checking
- Code syntax validation  
- Structure completeness scoring

**LLM Judge**:
- Uses separate evaluator agent
- Considers accuracy, relevance, completeness
- Provides objective candidate ranking

**Heuristic Fallback**:
- Length-based scoring (avoid too short/long)
- Structure indicators (headers, code blocks)
- Error pattern detection

## Key Design Decisions

### 1. Modular Architecture
**Decision**: Separate planner, executor, grounder, and memory components
**Motivation**: 
- **Maintainability**: Each component has clear responsibilities
- **Testability**: Components can be tested in isolation
- **Flexibility**: Easy to swap implementations or add features
- **Scalability**: Parallel processing of different components

### 2. DAG-based Planning  
**Decision**: Convert natural language plans to directed acyclic graphs
**Motivation**:
- **Dependency Handling**: Complex tasks often have interdependencies
- **Parallel Execution**: DAG enables parallel subtask execution
- **Robust Planning**: Structured representation vs. linear lists
- **Reusability**: DAG patterns can be cached and reused

### 3. Vision-Language Integration
**Decision**: Use multimodal models throughout the system
**Motivation**:
- **Context Awareness**: Screenshot analysis improves all decisions
- **Accurate Grounding**: Visual understanding for element location  
- **Adaptive Planning**: Plans adapt to current screen state
- **Error Detection**: Visual feedback enables better error handling

### 4. Comprehensive Memory System
**Decision**: Multi-level memory with embeddings and summarization
**Motivation**:
- **Experience Learning**: Improves performance over time
- **Knowledge Transfer**: Experience from similar tasks helps new tasks
- **Efficient Storage**: Summarization prevents memory bloat
- **Fast Retrieval**: Embedding-based search scales well

### 5. Best-of-N Selection
**Decision**: Generate multiple candidates and select best
**Motivation**:
- **Quality Improvement**: Best-of-N consistently outperforms single-shot
- **Error Mitigation**: Multiple attempts reduce random failures
- **Robustness**: System less sensitive to individual generation failures
- **Adaptability**: Can adjust N based on task complexity

## Reproduction Guide

### System Requirements

**Host Environment**:
- Ubuntu 20.04+ or compatible Linux distribution
- Docker with privileged container support
- 16GB+ RAM, 4+ CPU cores recommended
- NVIDIA GPU optional but recommended for vision models

**Software Dependencies**:
```bash
# Core Python packages
pip install openai anthropic requests pillow pytesseract
pip install scikit-learn numpy tiktoken modelx backoff
pip install docker wrapt-timeout-decorator tqdm

# Computer vision and GUI automation  
pip install pyautogui opencv-python
sudo apt-get install tesseract-ocr

# OSWorld environment
git clone https://github.com/xlang-ai/OSWorld
# Follow OSWorld setup instructions
```

### Configuration Setup

**1. API Keys**:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional for grounding
```

**2. OSWorld Container**:
```bash
# Start OSWorld container (follow OSWorld documentation)
docker run -d --privileged -p 5000:5000 happysixd/osworld-docker

# Verify container is running
curl http://localhost:5000/platform
```

**3. Knowledge Base Directory**:
```bash
mkdir -p kb_distyl/linux
# This will store episodic and narrative memory
```

### Basic Usage

**Simple Task Execution**:
```python
from distyl.controller.controller import Controller

# Initialize agent with OpenAI GPT-4
engine_params = {
    "engine_type": "openai",
    "model": "gpt-4o",
    "api_key": os.getenv("OPENAI_API_KEY")
}

agent = Controller(
    engine_params=engine_params,
    platform="linux",
    memory_folder_name="kb_distyl",
    enable_best_of_n=True,
    n_candidates=3
)

# Execute task
instruction = "Open calculator and compute 15 * 23"
obs = env.reset()

while True:
    info, actions = agent.predict(instruction, obs)
    if actions[0] == "DONE":
        break
    obs, reward, done, env_info = env.step(actions[0])
    if done:
        break

agent.finalize_memory()
```

**Advanced Configuration**:
```python
# Use different models for different components
grounding_params = {
    "engine_type": "anthropic", 
    "model": "claude-3-sonnet-20240229"
}

agent = Controller(
    engine_params=engine_params,
    grounding_engine_params=grounding_params,
    platform="linux",
    enable_best_of_n=True,
    n_candidates=5,  # More candidates for better quality
    memory_folder_name="kb_distyl_advanced"
)
```

### Customization Points

**1. Memory Configuration**:
```python
# Modify knowledge base behavior
agent.knowledge_base.save_knowledge = False  # Disable persistence
agent.knowledge_base.embedding_engine = None  # Disable embeddings
```

**2. Planning Customization**:
```python
# Access planner for custom configuration
agent.planner.max_history = 10  # Longer conversation history
agent.planner.n_candidates = 5  # More planning candidates
```

**3. Execution Customization**:
```python
# Modify executor behavior
agent.executor.enable_reflection = False  # Disable reflection
agent.executor.use_subtask_experience = False  # Disable experience retrieval
```

### Testing and Validation

**Unit Testing**:
```bash
# Test individual components
python -m pytest distyl/tests/test_planner.py
python -m pytest distyl/tests/test_executor.py
python -m pytest distyl/tests/test_grounder.py
```

**Integration Testing**:
```bash
# Run working test suite
python working_tests/test2_working.py

# Full OSWorld evaluation
python run_distyl.py --domain os --max_steps 15
```

**Performance Monitoring**:
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Monitor costs and tokens
print(f"Planning cost: ${agent.planner.cost}")
print(f"Execution cost: ${agent.executor.cost}")
```

## Future Enhancements

### Short-term Improvements

**1. Enhanced Grounding**:
- Integration with SOM (Set-of-Mark) for precise element identification
- Multi-scale screenshot analysis for better element detection
- Confidence scoring for grounding decisions

**2. Improved Memory**:
- Cross-task knowledge transfer mechanisms
- Hierarchical memory organization (domain → task → subtask)
- Memory consolidation and forgetting strategies

**3. Better Planning**:
- Dynamic replanning based on execution feedback
- Risk assessment and contingency planning
- Multi-modal plan representation (visual + textual)

### Medium-term Enhancements

**1. Multi-Agent Coordination**:
- Specialist agents for different application domains
- Collaborative task decomposition and execution
- Shared knowledge base across agent instances

**2. Advanced Learning**:
- Reinforcement learning for action optimization
- Meta-learning for rapid adaptation to new domains
- Curriculum learning for progressive skill development

**3. Robustness Improvements**:
- Formal verification of critical action sequences
- Automated testing and validation pipelines
- Error prediction and proactive recovery

### Long-term Vision

**1. General Desktop Intelligence**:
- Support for arbitrary desktop applications
- Cross-platform consistency (Linux/Windows/macOS)
- Natural language query interface for complex workflows

**2. Collaborative Human-AI Interaction**:
- Real-time collaboration with human users
- Explanation and teaching capabilities
- Preference learning and personalization

**3. Ecosystem Integration**:
- API for third-party tool integration
- Plugin architecture for domain-specific extensions
- Cloud-based knowledge sharing across deployments

---

*This documentation represents the current state of the Distyl agent implementation as of August 2025. The system continues to evolve with ongoing research and development.*