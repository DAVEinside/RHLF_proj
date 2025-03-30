"""
Reporting module for generating evaluation reports.
"""

import logging
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.metrics import EvaluationResult

logger = logging.getLogger(__name__)

def generate_evaluation_report(results: List[Tuple[str, EvaluationResult]]) -> str:
    """
    Generate a text report from evaluation results.
    
    Args:
        results: List of (prompt, evaluation_result) tuples
        
    Returns:
        Report text
    """
    if not results:
        return "No evaluation results to report."
    
    # Start building the report
    report_lines = ["# Content Evaluation Report", ""]
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Number of content items evaluated: {len(results)}")
    report_lines.append("")
    
    # Calculate average scores across all content
    all_metrics = set()
    for _, result in results:
        all_metrics.update(result.metrics.keys())
    
    metric_scores = {metric: [] for metric in all_metrics}
    overall_scores = []
    
    for _, result in results:
        if result.overall_score is not None:
            overall_scores.append(result.overall_score)
            
        for metric in all_metrics:
            if metric in result.metrics:
                metric_scores[metric].append(result.metrics[metric]["score"])
    
    # Add overall score summary
    if overall_scores:
        avg_overall = sum(overall_scores) / len(overall_scores)
        report_lines.append(f"## Overall Score Summary")
        report_lines.append(f"Average overall score: {avg_overall:.2f}/10")
        report_lines.append(f"Highest overall score: {max(overall_scores):.2f}/10")
        report_lines.append(f"Lowest overall score: {min(overall_scores):.2f}/10")
        report_lines.append("")
    
    # Add metric score summaries
    report_lines.append("## Metric Summaries")
    
    for metric in sorted(all_metrics):
        scores = metric_scores[metric]
        if not scores:
            continue
            
        avg_score = sum(scores) / len(scores)
        
        # Format the metric name for display
        display_name = metric.replace('_', ' ').title()
        
        report_lines.append(f"### {display_name}")
        report_lines.append(f"Average score: {avg_score:.2f}/10")
        report_lines.append(f"Highest score: {max(scores):.2f}/10")
        report_lines.append(f"Lowest score: {min(scores):.2f}/10")
        report_lines.append("")
    
    # Add individual content summaries
    report_lines.append("## Individual Content Evaluations")
    
    for i, (prompt, result) in enumerate(results, 1):
        # Truncate long prompts
        if len(prompt) > 50:
            display_prompt = prompt[:47] + "..."
        else:
            display_prompt = prompt
            
        report_lines.append(f"### Content {i}: {display_prompt}")
        report_lines.append(f"Overall Score: {result.overall_score:.2f}/10")
        report_lines.append("")
        report_lines.append("Metric scores:")
        
        # Sort metrics by score (descending)
        sorted_metrics = sorted(
            result.metrics.items(), 
            key=lambda x: x[1]["score"], 
            reverse=True
        )
        
        for metric_name, metric_data in sorted_metrics:
            display_name = metric_name.replace('_', ' ').title()
            report_lines.append(f"- {display_name}: {metric_data['score']:.2f}/10")
        
        report_lines.append("")
    
    # Join all lines
    return "\n".join(report_lines)


def save_evaluation_report(report: str, file_path: str = "data/evaluation_report.md"):
    """
    Save an evaluation report to a file.
    
    Args:
        report: The report text
        file_path: Path to save the report
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved evaluation report to {file_path}")
    return file_path


def generate_evaluation_charts(results: List[Tuple[str, EvaluationResult]], 
                             output_dir: str = "data/charts"):
    """
    Generate charts visualizing evaluation results.
    
    Args:
        results: List of (prompt, evaluation_result) tuples
        output_dir: Directory to save charts
        
    Returns:
        List of generated chart file paths
    """
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    chart_files = []
    
    if not results:
        return chart_files
    
    # Extract data for charts
    all_metrics = set()
    for _, result in results:
        all_metrics.update(result.metrics.keys())
    
    metric_scores = {metric: [] for metric in all_metrics}
    prompt_labels = []
    
    for i, (prompt, result) in enumerate(results):
        # Create short prompt label
        if len(prompt) > 20:
            label = prompt[:17] + "..."
        else:
            label = prompt
        
        prompt_labels.append(f"Content {i+1}: {label}")
        
        for metric in all_metrics:
            if metric in result.metrics:
                metric_scores[metric].append(result.metrics[metric]["score"])
            else:
                metric_scores[metric].append(0)  # Use 0 for missing metrics
    
    # 1. Overall metric comparison chart
    plt.figure(figsize=(10, 6))
    
    avg_scores = []
    metric_labels = []
    
    for metric in sorted(all_metrics):
        scores = metric_scores[metric]
        if scores:
            avg = sum(scores) / len(scores)
            avg_scores.append(avg)
            metric_labels.append(metric.replace('_', ' ').title())
    
    y_pos = np.arange(len(metric_labels))
    
    plt.barh(y_pos, avg_scores, align='center', alpha=0.7)
    plt.yticks(y_pos, metric_labels)
    plt.xlabel('Average Score (0-10)')
    plt.title('Average Scores by Metric')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(chart_path)
    plt.close()
    
    chart_files.append(chart_path)
    
    # 2. Content comparison chart
    # Only create if we have more than one content item
    if len(results) > 1:
        plt.figure(figsize=(12, 8))
        
        # Get overall scores
        overall_scores = [result.overall_score for _, result in results]
        
        y_pos = np.arange(len(prompt_labels))
        
        plt.barh(y_pos, overall_scores, align='center', alpha=0.7)
        plt.yticks(y_pos, prompt_labels)
        plt.xlabel('Overall Score (0-10)')
        plt.title('Overall Scores by Content')
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, "content_comparison.png")
        plt.savefig(chart_path)
        plt.close()
        
        chart_files.append(chart_path)
    
    # 3. Radar chart of metrics for each content
    # Only create if we have up to 5 content items (gets messy with more)
    if len(results) <= 5:
        metrics_list = sorted(all_metrics)
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(metrics_list)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw one line per content item
        for i, (prompt, result) in enumerate(results):
            values = [result.metrics.get(metric, {}).get("score", 0) for metric in metrics_list]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Content {i+1}")
            ax.fill(angles, values, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in metrics_list])
        
        # Draw y-axis label at 0, 2, ..., 10
        ax.set_rlabel_position(0)
        plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=7)
        plt.ylim(0, 10)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Metric Comparison Across Content", y=1.1)
        
        chart_path = os.path.join(output_dir, "radar_comparison.png")
        plt.savefig(chart_path)
        plt.close()
        
        chart_files.append(chart_path)
    
    logger.info(f"Generated {len(chart_files)} evaluation charts")
    return chart_files


def create_html_report(results: List[Tuple[str, EvaluationResult]], 
                      output_file: str = "data/evaluation_report.html"):
    """
    Create an HTML report with evaluation results and charts.
    
    Args:
        results: List of (prompt, evaluation_result) tuples
        output_file: Path to save the HTML report
        
    Returns:
        Path to the generated HTML file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Generate charts
    chart_dir = os.path.join(os.path.dirname(output_file), "charts")
    chart_files = generate_evaluation_charts(results, chart_dir)
    
    # Start building HTML
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>Content Evaluation Report</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }",
        "        h1 { color: #333366; }",
        "        h2 { color: #336699; margin-top: 30px; }",
        "        h3 { color: #339999; }",
        "        .metric { margin-bottom: 5px; }",
        "        .score { font-weight: bold; }",
        "        .good { color: green; }",
        "        .moderate { color: orange; }",
        "        .poor { color: red; }",
        "        .chart { margin: 20px 0; max-width: 800px; }",
        "        table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }",
        "        th { background-color: #f2f2f2; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Content Evaluation Report</h1>",
        f"    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        f"    <p>Number of content items evaluated: {len(results)}</p>"
    ]
    
    # Add chart section if charts were generated
    if chart_files:
        html_lines.append("    <h2>Evaluation Charts</h2>")
        
        for chart_file in chart_files:
            # Get relative path to chart
            rel_path = os.path.relpath(chart_file, os.path.dirname(output_file))
            html_lines.append(f'    <div class="chart"><img src="{rel_path}" alt="Evaluation Chart" width="800"></div>')
    
    # Overall score summary
    overall_scores = [result.overall_score for _, result in results]
    
    if overall_scores:
        avg_overall = sum(overall_scores) / len(overall_scores)
        html_lines.append("    <h2>Overall Score Summary</h2>")
        html_lines.append(f"    <p>Average overall score: <span class='score'>{avg_overall:.2f}/10</span></p>")
        html_lines.append(f"    <p>Highest overall score: <span class='score'>{max(overall_scores):.2f}/10</span></p>")
        html_lines.append(f"    <p>Lowest overall score: <span class='score'>{min(overall_scores):.2f}/10</span></p>")
    
    # Calculate average scores across all content
    all_metrics = set()
    for _, result in results:
        all_metrics.update(result.metrics.keys())
    
    metric_scores = {metric: [] for metric in all_metrics}
    
    for _, result in results:
        for metric in all_metrics:
            if metric in result.metrics:
                metric_scores[metric].append(result.metrics[metric]["score"])
    
    # Metric summaries
    html_lines.append("    <h2>Metric Summaries</h2>")
    html_lines.append("    <table>")
    html_lines.append("        <tr><th>Metric</th><th>Average Score</th><th>Highest Score</th><th>Lowest Score</th></tr>")
    
    for metric in sorted(all_metrics):
        scores = metric_scores[metric]
        if not scores:
            continue
            
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        # Determine score class
        avg_class = "good" if avg_score >= 7 else "moderate" if avg_score >= 5 else "poor"
        
        # Format the metric name for display
        display_name = metric.replace('_', ' ').title()
        
        html_lines.append(f"        <tr>")
        html_lines.append(f"            <td>{display_name}</td>")
        html_lines.append(f"            <td class='{avg_class}'>{avg_score:.2f}/10</td>")
        html_lines.append(f"            <td>{max_score:.2f}/10</td>")
        html_lines.append(f"            <td>{min_score:.2f}/10</td>")
        html_lines.append(f"        </tr>")
    
    html_lines.append("    </table>")
    
    # Individual content evaluations
    html_lines.append("    <h2>Individual Content Evaluations</h2>")
    
    for i, (prompt, result) in enumerate(results, 1):
        html_lines.append(f"    <h3>Content {i}</h3>")
        html_lines.append(f"    <p><strong>Prompt:</strong> {prompt}</p>")
        
        # Overall score
        overall_class = "good" if result.overall_score >= 7 else "moderate" if result.overall_score >= 5 else "poor"
        html_lines.append(f"    <p><strong>Overall Score:</strong> <span class='score {overall_class}'>{result.overall_score:.2f}/10</span></p>")
        
        # Metric scores
        html_lines.append("    <table>")
        html_lines.append("        <tr><th>Metric</th><th>Score</th><th>Explanation</th></tr>")
        
        # Sort metrics by score (descending)
        sorted_metrics = sorted(
            result.metrics.items(), 
            key=lambda x: x[1]["score"], 
            reverse=True
        )
        
        for metric_name, metric_data in sorted_metrics:
            score = metric_data["score"]
            explanation = metric_data.get("explanation", "")
            
            # Determine score class
            score_class = "good" if score >= 7 else "moderate" if score >= 5 else "poor"
            
            # Format the metric name for display
            display_name = metric_name.replace('_', ' ').title()
            
            html_lines.append(f"        <tr>")
            html_lines.append(f"            <td>{display_name}</td>")
            html_lines.append(f"            <td class='{score_class}'>{score:.2f}/10</td>")
            html_lines.append(f"            <td>{explanation}</td>")
            html_lines.append(f"        </tr>")
        
        html_lines.append("    </table>")
    
    # Close HTML
    html_lines.extend([
        "</body>",
        "</html>"
    ])
    
    # Write HTML to file
    with open(output_file, 'w') as f:
        f.write("\n".join(html_lines))
    
    logger.info(f"Created HTML evaluation report at {output_file}")
    return output_file


def export_evaluation_to_csv(results: List[Tuple[str, EvaluationResult]], 
                           output_file: str = "data/evaluation_results.csv"):
    """
    Export evaluation results to CSV format.
    
    Args:
        results: List of (prompt, evaluation_result) tuples
        output_file: Path to save the CSV file
        
    Returns:
        Path to the generated CSV file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Collect all metrics
    all_metrics = set()
    for _, result in results:
        all_metrics.update(result.metrics.keys())
    
    # Prepare data for DataFrame
    data = []
    
    for i, (prompt, result) in enumerate(results, 1):
        row = {
            "content_id": i,
            "prompt": prompt,
            "overall_score": result.overall_score
        }
        
        # Add metric scores
        for metric in all_metrics:
            if metric in result.metrics:
                row[f"{metric}_score"] = result.metrics[metric]["score"]
            else:
                row[f"{metric}_score"] = None
        
        data.append(row)
    
    # Create DataFrame and export to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Exported evaluation results to CSV at {output_file}")
    return output_file