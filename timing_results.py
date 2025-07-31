"""
Timing utility functions for COVID-19 simulation
"""
import time
import os
import json
from datetime import datetime

class TimingManager:
    def __init__(self, output_file=None, rank=0):
        self.timings = {}
        self.start_times = {}
        self.output_file = output_file
        self.rank = rank
        
        # Create the output file directory if it doesn't exist
        if output_file and rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    def start_timing(self, section_name):
        """Start timing for a section"""
        key = f"{section_name}"
        self.start_times[key] = time.time()
        return self.start_times[key]
    
    def end_timing(self, section_name):
        """End timing for a section and record the result"""
        key = f"{section_name}"
        if key in self.start_times:
            end_time = time.time()
            elapsed = end_time - self.start_times[key]
            
            # Store timing
            if section_name not in self.timings:
                self.timings[section_name] = []
            
            self.timings[section_name].append(elapsed)
            
            # Remove start time
            del self.start_times[key]
            
            return elapsed
        else:
            return None
    
    def get_avg_timing(self, section_name):
        """Get the average time for a section"""
        if section_name in self.timings and len(self.timings[section_name]) > 0:
            return sum(self.timings[section_name]) / len(self.timings[section_name])
        return None
    
    def get_total_timing(self, section_name):
        """Get the total time for a section"""
        if section_name in self.timings and len(self.timings[section_name]) > 0:
            return sum(self.timings[section_name])
        return None
    
    def print_timing(self, section_name, index=None):
        """Print timing for a specific run of a section"""
        if section_name in self.timings:
            if index is None:
                elapsed = self.timings[section_name][-1]  # Get the most recent timing
            else:
                if 0 <= index < len(self.timings[section_name]):
                    elapsed = self.timings[section_name][index]
                else:
                    print(f"[TIMING][Rank {self.rank}] Invalid index {index} for section {section_name}")
                    return
                
            print(f"[TIMING][Rank {self.rank}] {section_name}: {elapsed:.4f} seconds")
    
    def print_all_timings(self):
        """Print all timings"""
        print(f"\n[TIMING][Rank {self.rank}] === TIMING SUMMARY ===")
        for section_name in self.timings:
            avg_time = self.get_avg_timing(section_name)
            total_time = self.get_total_timing(section_name)
            counts = len(self.timings[section_name])
            print(f"[TIMING][Rank {self.rank}] {section_name}: {avg_time:.4f} seconds (avg), {total_time:.4f} seconds (total), {counts} runs")
        print(f"[TIMING][Rank {self.rank}] === END OF TIMING SUMMARY ===\n")
    
    def save_timings(self):
        """Save timings to file"""
        if not self.output_file or self.rank != 0:  # Only rank 0 saves
            return
        
        # Prepare data for saving
        timing_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timings": {}
        }
        
        # Calculate statistics for each section
        for section_name, times in self.timings.items():
            timing_data["timings"][section_name] = {
                "runs": len(times),
                "total": sum(times),
                "average": sum(times) / len(times) if times else 0,
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
                "all_times": times
            }
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(timing_data, f, indent=4)
        
        print(f"[TIMING][Rank {self.rank}] Timing results saved to {self.output_file}")
