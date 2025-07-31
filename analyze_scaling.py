#!/usr/bin/env python3
"""
Analyze scaling performance from multiple experiments.
This script processes multiple timing files from different process counts 
and analyzes how performance scales with the number of processes.
"""

import os
import sys
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_timing_data(file_path):
    """Load timing data from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_process_count(filename):
    """Extract process count from filename like 'parallel_4.json'"""
    base = os.path.basename(filename)
    if base.startswith("parallel_"):
        try:
            return int(base.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return None
    return None

def analyze_scaling(experiment_dir):
    """Analyze scaling performance from experiment results"""
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory {experiment_dir} not found.")
        return
        
    # Find the sequential baseline file
    sequential_file = os.path.join(experiment_dir, "sequential.json")
    if not os.path.exists(sequential_file):
        print(f"Error: Sequential baseline file not found at {sequential_file}")
        return
        
    sequential_data = load_timing_data(sequential_file)
    sequential_time = sequential_data["timings"].get("total_simulation", {}).get("total", 0)
    
    if sequential_time == 0:
        print("Error: Invalid sequential timing data.")
        return
        
    # Find all parallel files
    parallel_files = glob.glob(os.path.join(experiment_dir, "parallel_*.json"))
    
    if not parallel_files:
        print("Error: No parallel timing files found.")
        return
        
    # Extract data for analysis
    scaling_data = []
    
    for file in parallel_files:
        process_count = extract_process_count(file)
        if process_count is None:
            continue
            
        data = load_timing_data(file)
        parallel_time = data["timings"].get("total_simulation", {}).get("total", 0)
        
        if parallel_time > 0:
            speedup = sequential_time / parallel_time
            efficiency = speedup / process_count
            
            scaling_data.append({
                'processes': process_count,
                'time': parallel_time,
                'speedup': speedup,
                'efficiency': efficiency
            })
    
    # Add sequential baseline
    scaling_data.append({
        'processes': 1,  # Sequential is equivalent to 1 process
        'time': sequential_time,
        'speedup': 1.0,  # By definition
        'efficiency': 1.0  # By definition
    })
    
    # Convert to DataFrame and sort by process count
    df = pd.DataFrame(scaling_data)
    df = df.sort_values('processes')
    
    # Save results to CSV
    csv_path = os.path.join(experiment_dir, "scaling_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Create text report
    report_path = os.path.join(experiment_dir, "scaling_analysis.txt")
    with open(report_path, 'w') as f:
        f.write("=== COVID-19 Simulation Scaling Analysis ===\n")
        f.write(f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== Sequential Baseline ===\n")
        f.write(f"Execution time: {sequential_time:.4f} seconds\n\n")
        
        f.write("=== Scaling Results ===\n")
        for _, row in df.iterrows():
            f.write(f"Processes: {row['processes']}\n")
            f.write(f"  Execution time: {row['time']:.4f} seconds\n")
            f.write(f"  Speedup: {row['speedup']:.4f}x\n")
            f.write(f"  Efficiency: {row['efficiency']:.4f}\n\n")
            
        # Calculate Amdahl's Law parameters
        try:
            # Simple model: T(n) = T(1) * (s + p/n) where s + p = 1
            # We can estimate s (sequential fraction) by regression
            processes = df['processes'].values
            normalized_times = df['time'].values / sequential_time
            
            # Exclude sequential baseline from fit
            processes_fit = processes[processes > 1]
            times_fit = normalized_times[processes > 1]
            
            if len(processes_fit) >= 2:  # Need at least 2 points for meaningful fit
                # Fit: T(n)/T(1) = s + (1-s)/n
                # This becomes: T(n)/T(1) * n = s*n + (1-s)
                from scipy.optimize import curve_fit
                
                def amdahl(n, s):
                    return s + (1-s)/n
                
                params, _ = curve_fit(amdahl, processes_fit, times_fit)
                s_estimate = params[0]
                
                f.write("=== Amdahl's Law Analysis ===\n")
                f.write(f"Estimated sequential fraction: {s_estimate:.4f}\n")
                f.write(f"Estimated parallel fraction: {1-s_estimate:.4f}\n")
                
                # Theoretical maximum speedup
                max_speedup = 1 / s_estimate if s_estimate > 0 else float('inf')
                f.write(f"Theoretical maximum speedup: {max_speedup:.2f}x\n\n")
        except:
            f.write("Could not perform Amdahl's Law analysis. Need more data points.\n\n")
    
    print(f"Analysis saved to {report_path}")
    
    # Create plots
    create_scaling_plots(df, experiment_dir)

def create_scaling_plots(df, experiment_dir):
    """Create scaling analysis plots"""
    # Create execution time plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['processes'], df['time'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Processes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs. Number of Processes')
    plt.grid(True)
    plt.savefig(os.path.join(experiment_dir, "execution_time.png"))
    plt.close()
    
    # Create speedup plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['processes'], df['speedup'], 'go-', linewidth=2, markersize=8, label='Actual')
    
    # Add ideal speedup line
    max_procs = df['processes'].max()
    ideal_x = np.array([1, max_procs])
    ideal_y = ideal_x  # Ideal speedup is equal to number of processes
    plt.plot(ideal_x, ideal_y, 'r--', linewidth=1.5, label='Ideal')
    
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Number of Processes')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "speedup.png"))
    plt.close()
    
    # Create efficiency plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['processes'], df['efficiency'], 'mo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Processes')
    plt.ylabel('Efficiency (Speedup/Processes)')
    plt.title('Parallel Efficiency vs. Number of Processes')
    plt.grid(True)
    plt.ylim(0, 1.1)  # Efficiency is typically between 0 and 1
    plt.savefig(os.path.join(experiment_dir, "efficiency.png"))
    plt.close()
    
    print(f"Plots saved to {experiment_dir}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_scaling.py <experiment_directory>")
        return
        
    experiment_dir = sys.argv[1]
    analyze_scaling(experiment_dir)

if __name__ == "__main__":
    main()
