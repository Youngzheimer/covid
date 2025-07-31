#!/usr/bin/env python3
"""
Advanced timing analysis for COVID-19 simulation.
This script provides a detailed breakdown of timing metrics for different simulation components.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

def load_timing_data(file_path):
    """Load timing data from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def find_all_timing_files():
    """Find all timing files"""
    parallel_files = glob.glob('./timing_results/parallel_*.json')
    sequential_files = glob.glob('./timing_results/sequential_*.json')
    return parallel_files, sequential_files

def extract_component_timings(data):
    """Extract timings for different simulation components"""
    components = {}
    total_time = 0
    
    # Extract the total simulation time
    if "total_simulation" in data["timings"]:
        total_time = data["timings"]["total_simulation"]["total"]
    
    # Extract timings for various components
    for key, timing in data["timings"].items():
        if key == "total_simulation":
            continue
            
        if key.startswith("day_") and "_" in key[4:]:
            day, component = key.split("_", 1)[1].split("_", 1)
            if component not in components:
                components[component] = []
            
            # Store (day, time) pair
            components[component].append((int(day), timing["total"]))
    
    # Sort each component's timings by day
    for component in components:
        components[component].sort(key=lambda x: x[0])
    
    return components, total_time

def analyze_detailed_breakdown(parallel_file, sequential_file, output_dir="./timing_results"):
    """Perform a detailed breakdown analysis of timing components"""
    parallel_data = load_timing_data(parallel_file)
    sequential_data = load_timing_data(sequential_file)
    
    parallel_components, parallel_total = extract_component_timings(parallel_data)
    sequential_components, sequential_total = extract_component_timings(sequential_data)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/detailed_analysis_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"=== Detailed COVID-19 Simulation Timing Analysis ===\n")
        f.write(f"Parallel file: {parallel_file}\n")
        f.write(f"Sequential file: {sequential_file}\n")
        f.write(f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== Overall Simulation Time ===\n")
        f.write(f"Parallel simulation: {parallel_total:.4f} seconds\n")
        f.write(f"Sequential simulation: {sequential_total:.4f} seconds\n")
        speedup = sequential_total / parallel_total if parallel_total > 0 else 0
        f.write(f"Speedup: {speedup:.2f}x\n\n")
        
        f.write("=== Component Analysis ===\n")
        
        # Find common components
        common_components = sorted(list(set(parallel_components.keys()) & set(sequential_components.keys())))
        
        component_data = {}
        
        for component in common_components:
            f.write(f"\n--- {component} ---\n")
            
            par_days = [d for d, _ in parallel_components[component]]
            seq_days = [d for d, _ in sequential_components[component]]
            
            # Find common days
            common_days = sorted(list(set(par_days) & set(seq_days)))
            
            component_data[component] = {
                'days': common_days,
                'parallel': [],
                'sequential': [],
                'speedup': []
            }
            
            for day in common_days:
                par_time = next((t for d, t in parallel_components[component] if d == day), 0)
                seq_time = next((t for d, t in sequential_components[component] if d == day), 0)
                
                day_speedup = seq_time / par_time if par_time > 0 else 0
                
                component_data[component]['parallel'].append(par_time)
                component_data[component]['sequential'].append(seq_time)
                component_data[component]['speedup'].append(day_speedup)
                
                f.write(f"Day {day}: Parallel = {par_time:.4f}s, Sequential = {seq_time:.4f}s, Speedup = {day_speedup:.2f}x\n")
            
            avg_speedup = np.mean(component_data[component]['speedup'])
            f.write(f"Average Speedup: {avg_speedup:.2f}x\n")
        
        # Calculate bottleneck analysis
        f.write("\n=== Bottleneck Analysis ===\n")
        
        par_bottlenecks = {}
        seq_bottlenecks = {}
        
        for component in common_components:
            par_times = [t for _, t in parallel_components[component]]
            seq_times = [t for _, t in sequential_components[component]]
            
            par_bottlenecks[component] = np.mean(par_times)
            seq_bottlenecks[component] = np.mean(seq_times)
        
        # Sort by time (descending) to find bottlenecks
        par_sorted = sorted(par_bottlenecks.items(), key=lambda x: x[1], reverse=True)
        seq_sorted = sorted(seq_bottlenecks.items(), key=lambda x: x[1], reverse=True)
        
        f.write("Top Parallel Bottlenecks:\n")
        for component, time in par_sorted[:5]:  # Top 5 bottlenecks
            f.write(f"{component}: {time:.4f} seconds (avg)\n")
            
        f.write("\nTop Sequential Bottlenecks:\n")
        for component, time in seq_sorted[:5]:  # Top 5 bottlenecks
            f.write(f"{component}: {time:.4f} seconds (avg)\n")
    
    print(f"Detailed analysis saved to {output_file}")
    return component_data

def plot_detailed_comparison(component_data, output_dir="./timing_results"):
    """Create detailed comparison plots for components"""
    if not component_data:
        return
        
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot stacked bar chart showing components
    components = list(component_data.keys())
    days = component_data[components[0]]['days']
    
    # Prepare data for stacked bar chart
    par_data = {}
    seq_data = {}
    
    for component in components:
        for i, day in enumerate(days):
            if day not in par_data:
                par_data[day] = {}
                seq_data[day] = {}
            par_data[day][component] = component_data[component]['parallel'][i]
            seq_data[day][component] = component_data[component]['sequential'][i]
    
    # Convert to DataFrame for easier plotting
    par_df = pd.DataFrame(par_data).T
    seq_df = pd.DataFrame(seq_data).T
    
    # Plot component breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    par_df.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
    ax1.set_title('Parallel Simulation Component Breakdown')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend(title='Components')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    seq_df.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis')
    ax2.set_title('Sequential Simulation Component Breakdown')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Time (seconds)')
    ax2.legend(title='Components')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/component_breakdown_{timestamp}.png")
    plt.close()
    
    # Plot speedup comparison for each component
    plt.figure(figsize=(12, 8))
    
    for component in components:
        plt.plot(component_data[component]['days'], 
                 component_data[component]['speedup'],
                 marker='o', label=component)
    
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5)
    plt.xlabel('Simulation Day')
    plt.ylabel('Speedup (Sequential/Parallel)')
    plt.title('Component Speedup Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"{output_dir}/component_speedup_{timestamp}.png")
    plt.close()
    
    # Plot heatmap of component speedups
    plt.figure(figsize=(14, 10))
    
    # Prepare data for heatmap
    heatmap_data = []
    for component in components:
        for i, day in enumerate(component_data[component]['days']):
            heatmap_data.append({
                'Component': component,
                'Day': day,
                'Speedup': component_data[component]['speedup'][i]
            })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_pivot = heatmap_df.pivot_table(values='Speedup', index='Component', columns='Day')
    
    sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
    plt.title('Component Speedup Heatmap')
    plt.savefig(f"{output_dir}/component_heatmap_{timestamp}.png")
    plt.close()
    
    print(f"Detailed plots saved to {output_dir}")

def main():
    # Check if specific files are provided as arguments
    import sys
    if len(sys.argv) == 3:
        parallel_file = sys.argv[1]
        sequential_file = sys.argv[2]
        if os.path.exists(parallel_file) and os.path.exists(sequential_file):
            component_data = analyze_detailed_breakdown(parallel_file, sequential_file)
            plot_detailed_comparison(component_data)
            return
    
    # Otherwise, find the latest files
    parallel_files, sequential_files = find_all_timing_files()
    
    if not parallel_files or not sequential_files:
        print("Error: Could not find both parallel and sequential timing files.")
        print("Make sure to run both simulations first or provide the file paths as arguments.")
        print("Usage: python detailed_analysis.py [parallel_file] [sequential_file]")
        return
    
    latest_parallel = max(parallel_files, key=os.path.getmtime)
    latest_sequential = max(sequential_files, key=os.path.getmtime)
    
    component_data = analyze_detailed_breakdown(latest_parallel, latest_sequential)
    plot_detailed_comparison(component_data)

if __name__ == "__main__":
    main()
