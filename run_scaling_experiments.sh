#!/usr/bin/env bash
# Script to run COVID-19 simulation with different numbers of processes
# and analyze scaling performance

# Directory to store timing results
TIMING_DIR="./timing_results"
mkdir -p "$TIMING_DIR"

# Run experiments with different numbers of processes
run_scaling_experiments() {
    echo "Starting scaling experiments..."
    echo "This will run the simulation multiple times with different process counts"
    
    # Define the number of processes to test
    # Modify this array based on your system's capabilities
    PROCESS_COUNTS=(1 2 4 8)
    
    # Number of days to simulate (keep this small for quick experiments)
    NUM_DAYS=3
    
    # Create timestamp for this experiment set
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXPERIMENT_DIR="${TIMING_DIR}/scaling_${TIMESTAMP}"
    mkdir -p "$EXPERIMENT_DIR"
    
    # Store experiment parameters
    echo "Scaling experiments run at $(date)" > "${EXPERIMENT_DIR}/experiment_info.txt"
    echo "Process counts: ${PROCESS_COUNTS[@]}" >> "${EXPERIMENT_DIR}/experiment_info.txt"
    echo "Days simulated: $NUM_DAYS" >> "${EXPERIMENT_DIR}/experiment_info.txt"
    
    # Run sequential version first (as baseline)
    echo "Running sequential baseline..."
    python3 run_sequential.py --days $NUM_DAYS --output "${EXPERIMENT_DIR}/sequential.json"
    
    # Run parallel versions with different process counts
    for NPROCS in "${PROCESS_COUNTS[@]}"; do
        echo "Running with $NPROCS processes..."
        mpirun -n $NPROCS python3 run_parallel.py --days $NUM_DAYS --output "${EXPERIMENT_DIR}/parallel_${NPROCS}.json"
        sleep 2  # Small delay to ensure file system operations complete
    done
    
    echo "All experiments completed."
    echo "Results saved in ${EXPERIMENT_DIR}"
    
    # Generate scaling analysis
    python3 analyze_scaling.py "$EXPERIMENT_DIR"
}

# Main execution
if [ "$1" == "scaling" ]; then
    run_scaling_experiments
else
    echo "COVID-19 Simulation Performance Testing"
    echo "Usage:"
    echo "  $0 scaling    - Run scaling experiments with different process counts"
    echo ""
    echo "For regular simulations, use:"
    echo "  mpirun -n <processes> python3 main.py   - Run parallel simulation"
    echo "  python3 main.py                        - Run sequential simulation (using rank 0 only)"
fi
