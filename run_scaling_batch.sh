#!/bin/bash
# Batch script to run scaling experiments with different process counts

# Configuration
EXPERIMENT_DIR="scaling_experiments_$(date +%s)"
MAX_PROCESSES=8  # Maximum number of processes to test
SEQUENTIAL_RUN=true  # Whether to run sequential version for comparison
NUM_STEPS=10  # Number of simulation steps

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR"
echo "Created experiment directory: $EXPERIMENT_DIR"

# Function to run experiment with specific number of processes
run_experiment() {
    local num_processes=$1
    local output_file="$EXPERIMENT_DIR/parallel_${num_processes}.json"
    
    echo "Running parallel simulation with $num_processes processes..."
    mpirun -n $num_processes python3 main.py --parallel --timing_output="$output_file" --num_steps=$NUM_STEPS
    
    if [ $? -eq 0 ]; then
        echo "✅ Experiment with $num_processes processes completed successfully"
        echo "Timing results saved to: $output_file"
    else
        echo "❌ Experiment with $num_processes processes failed"
    fi
}

# Run sequential version if enabled
if [ "$SEQUENTIAL_RUN" = true ]; then
    echo "Running sequential simulation..."
    python3 main.py --timing_output="$EXPERIMENT_DIR/sequential.json" --num_steps=$NUM_STEPS
    
    if [ $? -eq 0 ]; then
        echo "✅ Sequential experiment completed successfully"
        echo "Timing results saved to: $EXPERIMENT_DIR/sequential.json"
    else
        echo "❌ Sequential experiment failed"
    fi
fi

# Run parallel experiments with increasing process counts
for ((p=2; p<=MAX_PROCESSES; p+=1)); do
    run_experiment $p
done

# Generate scaling analysis if analyze_scaling.py exists
if [ -f "analyze_scaling.py" ]; then
    echo "Generating scaling analysis..."
    python3 analyze_scaling.py "$EXPERIMENT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Scaling analysis completed successfully"
    else
        echo "❌ Scaling analysis failed"
    fi
else
    echo "analyze_scaling.py not found. Skipping analysis."
fi

echo "All experiments completed. Results are in: $EXPERIMENT_DIR"
