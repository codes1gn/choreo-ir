#!/usr/bin/env python3
"""
Performance Regression Testing Script for Choreo-IR

This script:
1. Runs comprehensive benchmarks
2. Compares with historical baselines  
3. Detects performance regressions
4. Generates HTML reports
5. Integrates with CI/CD pipelines
"""

import os
import sys
import json
import subprocess
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PerformanceRegression:
    def __init__(self, build_dir: str = "build", baseline_file: str = "performance_baseline.json"):
        self.build_dir = Path(build_dir)
        self.baseline_file = Path(baseline_file)
        self.results_dir = Path("performance_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Performance thresholds
        self.regression_threshold = 0.05  # 5% regression threshold
        self.improvement_threshold = 0.02  # 2% improvement threshold
        
    def run_benchmarks(self, benchmark_filter: str = "", iterations: int = 5) -> Dict:
        """Run the benchmark suite and return results"""
        print(f"Running benchmarks with filter: '{benchmark_filter}'")
        
        benchmark_exe = self.build_dir / "benchmark" / "performance" / "benchmark_suite"
        if not benchmark_exe.exists():
            raise FileNotFoundError(f"Benchmark executable not found: {benchmark_exe}")
        
        # Prepare benchmark command
        cmd = [str(benchmark_exe)]
        if benchmark_filter:
            cmd.extend(["--benchmark_filter", benchmark_filter])
        
        cmd.extend([
            "--benchmark_repetitions", str(iterations),
            "--benchmark_report_aggregates_only=true",
            "--benchmark_format=json",
            "--benchmark_out=benchmark_results.json"
        ])
        
        print(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=self.build_dir, capture_output=True, text=True, check=True)
            print("Benchmark completed successfully")
            
            # Load JSON results
            with open(self.build_dir / "benchmark_results.json", 'r') as f:
                results = json.load(f)
                
            return self._process_benchmark_results(results)
            
        except subprocess.CalledProcessError as e:
            print(f"Benchmark failed with return code {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
    def _process_benchmark_results(self, raw_results: Dict) -> Dict:
        """Process raw benchmark JSON into structured data"""
        processed = {
            "timestamp": datetime.datetime.now().isoformat(),
            "context": raw_results.get("context", {}),
            "benchmarks": {}
        }
        
        for benchmark in raw_results.get("benchmarks", []):
            name = benchmark["name"]
            
            # Extract key metrics
            metrics = {
                "time_ms": benchmark.get("real_time", 0),
                "cpu_time_ms": benchmark.get("cpu_time", 0),
                "iterations": benchmark.get("iterations", 0),
                "items_per_second": benchmark.get("items_per_second", 0),
                "bytes_per_second": benchmark.get("bytes_per_second", 0),
                "tflops": benchmark.get("TFLOPS", 0),
                "bandwidth_gb_s": benchmark.get("BandwidthGB/s", 0),
                "M": benchmark.get("M", 0),
                "N": benchmark.get("N", 0), 
                "K": benchmark.get("K", 0),
                "use_tensor_cores": benchmark.get("UseTensorCores", 0),
                "batch_size": benchmark.get("BatchSize", 0)
            }
            
            processed["benchmarks"][name] = metrics
            
        return processed
    
    def load_baseline(self) -> Optional[Dict]:
        """Load baseline performance data"""
        if not self.baseline_file.exists():
            print(f"No baseline file found: {self.baseline_file}")
            return None
            
        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading baseline: {e}")
            return None
    
    def save_baseline(self, results: Dict):
        """Save current results as new baseline"""
        with open(self.baseline_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Baseline saved to {self.baseline_file}")
    
    def compare_results(self, current: Dict, baseline: Dict) -> Dict:
        """Compare current results with baseline"""
        comparison = {
            "timestamp": current["timestamp"],
            "summary": {
                "total_benchmarks": 0,
                "regressions": 0,
                "improvements": 0,
                "no_change": 0,
                "new_benchmarks": 0,
                "removed_benchmarks": 0
            },
            "details": {}
        }
        
        current_benchmarks = set(current["benchmarks"].keys())
        baseline_benchmarks = set(baseline["benchmarks"].keys())
        
        # New and removed benchmarks
        new_benchmarks = current_benchmarks - baseline_benchmarks
        removed_benchmarks = baseline_benchmarks - current_benchmarks
        common_benchmarks = current_benchmarks & baseline_benchmarks
        
        comparison["summary"]["new_benchmarks"] = len(new_benchmarks)
        comparison["summary"]["removed_benchmarks"] = len(removed_benchmarks)
        comparison["summary"]["total_benchmarks"] = len(common_benchmarks)
        
        for name in new_benchmarks:
            comparison["details"][name] = {
                "status": "NEW",
                "current": current["benchmarks"][name],
                "baseline": None,
                "change_ratio": None
            }
        
        for name in removed_benchmarks:
            comparison["details"][name] = {
                "status": "REMOVED", 
                "current": None,
                "baseline": baseline["benchmarks"][name],
                "change_ratio": None
            }
        
        # Compare common benchmarks
        for name in common_benchmarks:
            current_metrics = current["benchmarks"][name]
            baseline_metrics = baseline["benchmarks"][name]
            
            # Use TFLOPS as primary metric, fall back to time
            current_perf = current_metrics.get("tflops", 0)
            baseline_perf = baseline_metrics.get("tflops", 0)
            
            if current_perf == 0 and baseline_perf == 0:
                # Fall back to inverse time (higher is better)
                current_time = current_metrics.get("time_ms", float('inf'))
                baseline_time = baseline_metrics.get("time_ms", float('inf'))
                if baseline_time > 0:
                    current_perf = 1000.0 / current_time if current_time > 0 else 0
                    baseline_perf = 1000.0 / baseline_time
            
            if baseline_perf > 0:
                change_ratio = (current_perf - baseline_perf) / baseline_perf
            else:
                change_ratio = 0
            
            # Determine status
            if abs(change_ratio) < self.improvement_threshold:
                status = "NO_CHANGE"
                comparison["summary"]["no_change"] += 1
            elif change_ratio >= self.improvement_threshold:
                status = "IMPROVEMENT"
                comparison["summary"]["improvements"] += 1
            elif change_ratio <= -self.regression_threshold:
                status = "REGRESSION"
                comparison["summary"]["regressions"] += 1
            else:
                status = "NO_CHANGE"
                comparison["summary"]["no_change"] += 1
            
            comparison["details"][name] = {
                "status": status,
                "current": current_metrics,
                "baseline": baseline_metrics,
                "change_ratio": change_ratio,
                "change_percent": change_ratio * 100
            }
        
        return comparison
    
    def generate_report(self, comparison: Dict, output_file: str = None) -> str:
        """Generate HTML performance report"""
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"performance_report_{timestamp}.html"
        
        html_content = self._generate_html_report(comparison)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        print(f"Report generated: {output_file}")
        return str(output_file)
    
    def _generate_html_report(self, comparison: Dict) -> str:
        """Generate HTML content for the report"""
        summary = comparison["summary"]
        
        # Status colors
        status_colors = {
            "REGRESSION": "#ff4444",
            "IMPROVEMENT": "#44ff44", 
            "NO_CHANGE": "#888888",
            "NEW": "#4444ff",
            "REMOVED": "#ff8844"
        }
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Choreo-IR Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .regression {{ color: {status_colors["REGRESSION"]}; font-weight: bold; }}
        .improvement {{ color: {status_colors["IMPROVEMENT"]}; font-weight: bold; }}
        .no-change {{ color: {status_colors["NO_CHANGE"]}; }}
        .new {{ color: {status_colors["NEW"]}; font-weight: bold; }}
        .removed {{ color: {status_colors["REMOVED"]}; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-family: monospace; }}
    </style>
</head>
<body>
    <h1>Choreo-IR Performance Report</h1>
    <p><strong>Generated:</strong> {comparison["timestamp"]}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <ul>
            <li><strong>Total Benchmarks:</strong> {summary["total_benchmarks"]}</li>
            <li class="regression"><strong>Regressions:</strong> {summary["regressions"]}</li>
            <li class="improvement"><strong>Improvements:</strong> {summary["improvements"]}</li>
            <li class="no-change"><strong>No Change:</strong> {summary["no_change"]}</li>
            <li class="new"><strong>New Benchmarks:</strong> {summary["new_benchmarks"]}</li>
            <li class="removed"><strong>Removed Benchmarks:</strong> {summary["removed_benchmarks"]}</li>
        </ul>
    </div>
"""
        
        if summary["regressions"] > 0:
            html += """
    <div style="background: #ffe6e6; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <h2 style="color: #cc0000;">⚠️ Performance Regressions Detected</h2>
        <p>Some benchmarks show significant performance degradation. Please review the detailed results below.</p>
    </div>
"""
        
        # Detailed results table
        html += """
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Status</th>
            <th>Current TFLOPS</th>
            <th>Baseline TFLOPS</th>
            <th>Change %</th>
            <th>Current Time (ms)</th>
            <th>Baseline Time (ms)</th>
            <th>Matrix Size</th>
        </tr>
"""
        
        # Sort by status (regressions first)
        status_priority = {"REGRESSION": 0, "IMPROVEMENT": 1, "NEW": 2, "REMOVED": 3, "NO_CHANGE": 4}
        sorted_benchmarks = sorted(
            comparison["details"].items(),
            key=lambda x: (status_priority.get(x[1]["status"], 5), x[0])
        )
        
        for name, details in sorted_benchmarks:
            status = details["status"]
            status_class = status.lower().replace("_", "-")
            
            current = details.get("current", {})
            baseline = details.get("baseline", {})
            change_percent = details.get("change_percent", 0)
            
            # Format metrics
            current_tflops = f"{current.get('tflops', 0):.2f}" if current else "N/A"
            baseline_tflops = f"{baseline.get('tflops', 0):.2f}" if baseline else "N/A"
            current_time = f"{current.get('time_ms', 0):.2f}" if current else "N/A"
            baseline_time = f"{baseline.get('time_ms', 0):.2f}" if baseline else "N/A"
            
            # Matrix size
            if current:
                M, N, K = current.get('M', 0), current.get('N', 0), current.get('K', 0)
                if M > 0 and N > 0 and K > 0:
                    matrix_size = f"{M}×{N}×{K}"
                else:
                    matrix_size = "N/A"
            else:
                matrix_size = "N/A"
            
            change_str = f"{change_percent:+.1f}%" if change_percent != 0 else "0%"
            
            html += f"""
        <tr>
            <td class="metric">{name}</td>
            <td class="{status_class}">{status}</td>
            <td class="metric">{current_tflops}</td>
            <td class="metric">{baseline_tflops}</td>
            <td class="metric">{change_str}</td>
            <td class="metric">{current_time}</td>
            <td class="metric">{baseline_time}</td>
            <td class="metric">{matrix_size}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        return html
    
    def create_performance_plots(self, comparison: Dict):
        """Create performance visualization plots"""
        # Extract data for plotting
        benchmark_names = []
        current_tflops = []
        baseline_tflops = []
        
        for name, details in comparison["details"].items():
            if details["status"] in ["REGRESSION", "IMPROVEMENT", "NO_CHANGE"]:
                benchmark_names.append(name.replace("_", "\n"))  # Wrap long names
                current_tflops.append(details["current"].get("tflops", 0))
                baseline_tflops.append(details["baseline"].get("tflops", 0))
        
        if not benchmark_names:
            return
        
        # Create comparison plot
        plt.figure(figsize=(15, 8))
        x = np.arange(len(benchmark_names))
        width = 0.35
        
        plt.bar(x - width/2, baseline_tflops, width, label='Baseline', alpha=0.8)
        plt.bar(x + width/2, current_tflops, width, label='Current', alpha=0.8)
        
        plt.xlabel('Benchmarks')
        plt.ylabel('Performance (TFLOPS)')
        plt.title('Choreo-IR Performance Comparison')
        plt.xticks(x, benchmark_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plot_file = self.results_dir / f"performance_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plot saved: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="Choreo-IR Performance Regression Testing")
    parser.add_argument("--build-dir", default="build", help="Build directory path")
    parser.add_argument("--baseline", default="performance_baseline.json", help="Baseline file path")
    parser.add_argument("--filter", default="", help="Benchmark filter pattern")
    parser.add_argument("--iterations", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument("--save-baseline", action="store_true", help="Save current results as new baseline")
    parser.add_argument("--no-compare", action="store_true", help="Skip comparison with baseline")
    parser.add_argument("--output", help="Output report file path")
    parser.add_argument("--plots", action="store_true", help="Generate performance plots")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit with error code if regressions detected")
    
    args = parser.parse_args()
    
    try:
        regression_tester = PerformanceRegression(args.build_dir, args.baseline)
        
        # Run benchmarks
        print("Starting performance regression tests...")
        current_results = regression_tester.run_benchmarks(args.filter, args.iterations)
        
        if args.save_baseline:
            regression_tester.save_baseline(current_results)
            print("Baseline updated successfully")
            return 0
        
        if not args.no_compare:
            # Load baseline and compare
            baseline_results = regression_tester.load_baseline()
            if baseline_results is None:
                print("No baseline found. Use --save-baseline to create one.")
                return 1
            
            comparison = regression_tester.compare_results(current_results, baseline_results)
            
            # Generate report
            report_file = regression_tester.generate_report(comparison, args.output)
            
            # Generate plots if requested
            if args.plots:
                regression_tester.create_performance_plots(comparison)
            
            # Print summary
            summary = comparison["summary"]
            print(f"\nPerformance Test Summary:")
            print(f"  Total benchmarks: {summary['total_benchmarks']}")
            print(f"  Regressions: {summary['regressions']}")
            print(f"  Improvements: {summary['improvements']}")
            print(f"  No change: {summary['no_change']}")
            print(f"  New benchmarks: {summary['new_benchmarks']}")
            print(f"  Removed benchmarks: {summary['removed_benchmarks']}")
            
            if summary["regressions"] > 0:
                print(f"\n⚠️  WARNING: {summary['regressions']} performance regressions detected!")
                print(f"📊 See detailed report: {report_file}")
                
                if args.fail_on_regression:
                    return 1
            else:
                print("✅ No performance regressions detected")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 