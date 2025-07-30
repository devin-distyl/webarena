#!/opt/homebrew/bin/bash

# WebArena Model Comparison Script
# Compares GPT-3.5-turbo-0125 vs GPT-4-1106-preview

set -e

# Configuration
TASK_IDS=(0 1 2 3 4 5 6 11 12 13 14 15 41 42 43 62 63 64 65 77)
MODELS=("gpt-3.5-turbo-0125" "gpt-4-1106-preview")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Setup environment
setup_env() {
    # Load .env file if it exists
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
    
    # Set defaults if not already set
    export SHOPPING="${SHOPPING:-http://localhost:7770}"
    export SHOPPING_ADMIN="${SHOPPING_ADMIN:-http://localhost:7780/admin}"
    export REDDIT="${REDDIT:-http://localhost:9999}"
    export GITLAB="${GITLAB:-http://localhost:8023}"
    export WIKIPEDIA="${WIKIPEDIA:-http://localhost:8888}"
    export MAP="${MAP:-http://localhost:3000}"
    export HOMEPAGE="${HOMEPAGE:-http://localhost:4399}"
    
    # Check for required API key
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${RED}❌ ERROR: OPENAI_API_KEY not found in environment or .env file${NC}"
        exit 1
    fi
}

# Run single task
run_task() {
    local task_id=$1
    local model=$2
    local result_dir="./experiment_results/${model//[-.]/_}"
    
    echo -e "    ${BLUE}Running task $task_id with $model...${NC}"
    
    # Activate virtual environment and run
    source webarena-env/bin/activate
    
    timeout 300 python run.py \
        --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
        --test_start_idx $task_id \
        --test_end_idx $((task_id + 1)) \
        --model $model \
        --result_dir $result_dir \
        > "temp_output_${model}_${task_id}.log" 2>&1
    
    local exit_code=$?
    
    # Check results
    if [ $exit_code -eq 0 ]; then
        # Extract score from log
        local score=$(grep "Average score:" "temp_output_${model}_${task_id}.log" | tail -1 | grep -o '[0-9]*\.[0-9]*' || echo "0.0")
        if (( $(echo "$score > 0.5" | bc -l) )); then
            echo -e "      ${GREEN}✅ PASS${NC} (Score: $score)"
            echo "1" # Return success
        else
            echo -e "      ${RED}❌ FAIL${NC} (Score: $score)"
            echo "0" # Return failure
        fi
    else
        echo -e "      ${RED}❌ ERROR${NC} (Exit code: $exit_code)"
        echo "0" # Return failure
    fi
}

# Main experiment
main() {
    echo -e "${BLUE}🧪 WebArena Model Comparison Experiment${NC}"
    echo "============================================"
    echo "Tasks: ${#TASK_IDS[@]} shopping admin tasks"
    echo "Models: ${MODELS[*]}"
    echo "============================================"
    
    setup_env
    
    # Create results directory
    mkdir -p experiment_results
    
    # Initialize counters
    declare -A success_counts
    declare -A total_counts
    
    for model in "${MODELS[@]}"; do
        success_counts[$model]=0
        total_counts[$model]=0
    done
    
    # Run experiments
    for model in "${MODELS[@]}"; do
        echo -e "\n${YELLOW}🤖 Testing $model${NC}"
        echo "----------------------------------------"
        
        for task_id in "${TASK_IDS[@]}"; do
            total_counts[$model]=$((total_counts[$model] + 1))
            
            # Run task and capture result
            result=$(run_task $task_id $model)
            if [ "$result" = "1" ]; then
                success_counts[$model]=$((success_counts[$model] + 1))
            fi
            
            # Brief pause
            sleep 1
        done
        
        echo -e "${YELLOW}$model completed: ${success_counts[$model]}/${total_counts[$model]} tasks${NC}"
    done
    
    # Display final results
    echo -e "\n${GREEN}📊 FINAL RESULTS${NC}"
    echo "=================================="
    
    for model in "${MODELS[@]}"; do
        local success=${success_counts[$model]}
        local total=${total_counts[$model]}
        local percentage=$(echo "scale=1; $success * 100 / $total" | bc)
        echo -e "${model}: ${success}/${total} (${percentage}%)"
    done
    
    # Determine winner
    model1=${MODELS[0]}
    model2=${MODELS[1]}
    
    if [ ${success_counts[$model1]} -gt ${success_counts[$model2]} ]; then
        echo -e "\n🏆 Winner: ${GREEN}$model1${NC}"
    elif [ ${success_counts[$model2]} -gt ${success_counts[$model1]} ]; then
        echo -e "\n🏆 Winner: ${GREEN}$model2${NC}"
    else
        echo -e "\n🤝 Result: ${YELLOW}TIE${NC}"
    fi
    
    # Cleanup temporary files
    rm -f temp_output_*.log
    
    echo -e "\n✅ Experiment completed!"
    echo "📁 Detailed results saved in experiment_results/ directory"
}

# Run the experiment
main "$@"