# WebArena Docker Isolation System

## Overview

The Docker Isolation System solves a critical problem in WebArena parallel execution: **race conditions and state conflicts between concurrent tasks**. When multiple agents run simultaneously against shared backend services, they interfere with each other's operations, leading to unreliable results.

## Problem Solved

### Before Docker Isolation ❌
```bash
# Multiple tasks sharing same backend
Task 78: http://localhost:7780/admin  ← Shared database
Task 79: http://localhost:7780/admin  ← Shared database  
Task 80: http://localhost:7780/admin  ← Shared database

# Race conditions occur:
- Agent A modifies review data while Agent B counts reviews
- Authentication state gets corrupted
- Database transactions conflict
- Inconsistent results and failures
```

### After Docker Isolation ✅
```bash
# Each task gets isolated environment
Task 78: http://localhost:10780/admin  ← Isolated container
Task 79: http://localhost:10880/admin  ← Isolated container
Task 80: http://localhost:10980/admin  ← Isolated container

# No interference:
- Separate databases per task
- Isolated authentication 
- No shared state conflicts
- Reliable parallel execution
```

## Architecture

### Key Components

1. **DockerIsolationManager** (`docker_isolation_manager.py`)
   - Manages container lifecycle for each task
   - Allocates unique port ranges
   - Handles authentication isolation
   - Provides robust cleanup

2. **Modified Parallel Execution** (`run_parallel_demo.py`)
   - Integrates Docker isolation into task execution
   - Creates isolated environments before task start
   - Ensures cleanup after task completion
   - Handles errors gracefully

### Port Allocation Strategy

Each task gets a dedicated port range of 100 ports:

```python
Task ID  | Port Range     | Services
---------|----------------|------------------
78       | 17800-17899    | Shopping: 17800, Admin: 17810, Reddit: 17820
79       | 17900-17999    | Shopping: 17900, Admin: 17910, Reddit: 17920  
80       | 18000-18099    | Shopping: 18000, Admin: 18010, Reddit: 18020
```

Base formula: `base_port = 10000 + (task_id * 100)`

## Usage

### Basic Parallel Execution

```bash
# Run multiple tasks with automatic Docker isolation
python run_parallel_demo.py --model gemini-2.5-pro --tasks 78,79,80

# The system automatically:
# 1. Creates isolated containers for each task
# 2. Allocates unique ports
# 3. Modifies task configs for isolation
# 4. Runs tasks in parallel
# 5. Cleans up all containers when done
```

### Advanced Usage

```bash
# Single task (still gets isolated environment)
python run_parallel_demo.py --model gpt-4 --tasks 78

# Task ranges
python run_parallel_demo.py --model claude-3-sonnet --tasks 78-82

# Control parallelism
python run_parallel_demo.py --model gemini-2.5-pro --tasks 78,79,80 --max-workers 2
```

### Output Example

```
🎯 Configuration:
   Model: gemini-2.5-pro
   Provider: google (auto-detected)
   Tasks: [78, 79, 80]
   Max workers: 3

🚀 Starting 3 tasks in parallel with google/gemini-2.5-pro...
🐳 Each task will run in its own isolated Docker environment
   - No shared state conflicts
   - Isolated authentication
   - Unique port allocations

🐳 [Thread 123] Starting isolated Docker environment for task 78
✅ [Thread 123] Docker environment ready for task 78 on ports 17800-17899
🐳 [Thread 456] Starting isolated Docker environment for task 79  
✅ [Thread 456] Docker environment ready for task 79 on ports 17900-17999
...
🧹 Cleaning up all Docker environments...
```

## Docker Images Required

The system uses official WebArena Docker images:

```bash
# Download required images (see environment_docker/README.md)
docker load --input shopping_final_0712.tar
docker load --input shopping_admin_final_0719.tar  
docker load --input postmill-populated-exposed-withimg.tar
docker load --input gitlab-populated-final-port8023.tar
```

For testing without images, the system gracefully handles missing images with warnings.

## File Structure

```
WebArena/
├── docker_isolation_manager.py    # Core isolation logic
├── run_parallel_demo.py           # Updated parallel execution
├── DOCKER_ISOLATION.md           # This documentation
├── config_files/                 # Task configurations
│   ├── 78.json                  # Modified for isolated URLs
│   └── 79.json                  # Modified for isolated URLs
└── parallel_demo_results/        # Results with Docker info
    └── 20250730_185420_google_gemini_2_5_pro/
        ├── task_78/
        │   ├── render_78.html    # Full execution trajectories
        │   └── traces/
        └── task_79/
            ├── render_79.html
            └── traces/
```

## Technical Details

### Container Lifecycle

1. **Startup**
   ```python
   # For each task:
   env = docker_manager.start_environment(task_id, sites)
   # Creates containers: shopping_admin, shopping, forum, etc.
   # Waits for readiness
   # Configures internal URLs
   ```

2. **Execution**
   ```python
   # Modified config points to isolated environment
   modified_config = {
       'start_url': 'http://localhost:17810/admin',  # Isolated port
       'storage_state': './temp_auth_task_78/shopping_admin_state.json'
   }
   ```

3. **Cleanup**
   ```python
   # Always executed in finally block
   docker_manager.stop_environment(task_id)
   # Stops containers
   # Removes temp files  
   # Cleans up auth directories
   ```

### Authentication Isolation

Each task gets isolated authentication:

```python
# Before: Shared auth (race conditions)
'storage_state': './.auth/shopping_admin_state.json'  # ❌ Shared

# After: Isolated auth (safe)  
'storage_state': './temp_auth_task_78_123456/shopping_admin_state.json'  # ✅ Isolated
```

### Error Handling

Robust error handling ensures cleanup even on failures:

```python
try:
    # Start containers
    # Run task
    # Get results
except ContainerStartupError:
    # Log error, cleanup containers
except TaskExecutionError:
    # Log error, preserve logs, cleanup containers  
except Exception as e:
    # Log unexpected error, cleanup containers
finally:
    # Always cleanup containers and temp files
    docker_manager.stop_environment(task_id)
```

## Benefits

1. **🛡️ Eliminates Race Conditions**
   - No shared state between parallel tasks
   - Each task operates in complete isolation

2. **🔒 Secure Authentication**
   - Isolated auth files per task
   - No auth state corruption

3. **🚀 True Parallelism**
   - Tasks can run simultaneously without interference
   - Linear scaling with number of workers

4. **🧹 Automatic Cleanup**
   - Containers automatically stopped and removed
   - Temporary files cleaned up
   - No resource leaks

5. **📊 Better Debugging**
   - Each task has isolated logs and HTML traces
   - Port information preserved in results
   - Clear separation of task artifacts

## Comparison: Before vs After

| Aspect | Before (Shared) | After (Isolated) |
|--------|----------------|------------------|
| **Parallel Safety** | ❌ Race conditions | ✅ Fully isolated |
| **Authentication** | ❌ Shared/corrupted | ✅ Per-task isolation |
| **Database State** | ❌ Conflicts | ✅ Independent |
| **Debugging** | ❌ Mixed logs | ✅ Clear separation |
| **Scalability** | ❌ Limited by conflicts | ✅ Linear scaling |
| **Reliability** | ❌ Inconsistent results | ✅ Deterministic |

## Performance Impact

- **Startup time**: +10-30 seconds per task (container startup)
- **Memory usage**: +500MB-1GB per task (container overhead)  
- **Network ports**: 100 ports per task (10000+ range)
- **Disk space**: Temporary auth files (<1MB per task)

**Trade-off**: Slightly higher resource usage for dramatically improved reliability and true parallel execution.

## Future Enhancements

1. **Container Pooling**: Pre-start containers to reduce startup time
2. **Resource Limits**: Add CPU/memory limits per container
3. **Network Isolation**: Use Docker networks for additional isolation
4. **Health Monitoring**: Advanced container health checks
5. **Image Optimization**: Smaller, faster container images

## Troubleshooting

### Common Issues

1. **Docker images not found**
   ```bash
   # Download from WebArena repository
   # See environment_docker/README.md
   ```

2. **Port conflicts**
   ```bash
   # Check if ports 10000+ are available
   netstat -tuln | grep 10000
   ```

3. **Container startup failures**
   ```bash
   # Check Docker daemon is running
   docker ps
   ```

4. **Authentication issues**
   ```bash
   # Ensure original auth files exist
   ls -la ./.auth/
   ```

## Conclusion

The Docker Isolation System transforms WebArena from a single-task execution environment into a robust parallel execution platform. By eliminating shared state conflicts, it enables reliable, scalable testing of web navigation agents with true parallelism.

This architecture provides the foundation for large-scale agent evaluation and comparison studies that were previously impossible due to race conditions and state conflicts.