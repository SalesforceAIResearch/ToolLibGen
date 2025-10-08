#!/usr/bin/env python3
"""
Reusable reporting utilities for evaluation results.
This module can be imported and used across different evaluation scripts.
"""

import sys
import os
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime

# Add src directory to path for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import save_json


def calculate_evaluation_stats(results_lst, model_nickname, passk, original_questions_count, 
                             save_path, current_time, tool_retrieval_enabled=False):
   """
   Calculate comprehensive evaluation statistics including pass@k accuracy, failure rates,
   and tool retrieval statistics.
   
   This is the core statistics calculation function that can be reused across different
   evaluation scripts.
   
   Args:
       results_lst: List of evaluation results
       model_nickname: Name/nickname of the model being evaluated
       passk: Number of passes per question
       original_questions_count: Number of original questions
       save_path: Path to save statistics
       current_time: Current timestamp
       tool_retrieval_enabled: Whether tool retrieval was used
   
   Returns:
       dict: Dictionary containing all calculated statistics
   """
   print("\n" + "="*60)
   print("EVALUATION STATISTICS")
   print("="*60)
   
   # Basic statistics
   total_samples = len(results_lst)
   
   print(f"Original questions count: {original_questions_count}")
   print(f"Total samples count: {total_samples}")
   print(f"Pass@K value: {passk}")
   print(f"Tool retrieval enabled: {tool_retrieval_enabled}")
   
   # Group by original question ID
   question_groups = defaultdict(list)
   for result in results_lst:
      original_id = result["id"].rsplit("_", 1)[0]  # Remove the last "_number" suffix
      correctness = result.get(f"{model_nickname}_correctness", 0)
      question_groups[original_id].append(correctness)
   
   # Calculate overall accuracy
   all_correctness = [result.get(f"{model_nickname}_correctness", 0) for result in results_lst]
   overall_accuracy = np.mean(all_correctness) * 100
   
   # Calculate failure rate
   failure_rate = (1 - np.mean(all_correctness)) * 100
   
   print(f"\nOverall accuracy: {overall_accuracy:.2f}%")
   print(f"Overall failure rate: {failure_rate:.2f}%")
   
   # Calculate Pass@K statistics
   pass_at_k_results = {}
   for k in range(1, passk + 1):
      correct_questions = 0
      for question_id, correctness_list in question_groups.items():
         # Pass@K: at least one success in K attempts
         if len(correctness_list) >= k:
            if sum(correctness_list[:k]) >= 1:
               correct_questions += 1
      
      pass_at_k = (correct_questions / len(question_groups)) * 100
      pass_at_k_results[k] = pass_at_k
      print(f"Pass@{k}: {pass_at_k:.2f}%")
   
   # Calculate success count distribution
   success_distribution = defaultdict(int)
   for question_id, correctness_list in question_groups.items():
      success_count = sum(correctness_list)
      success_distribution[success_count] += 1
   
   print(f"\nSuccess count distribution:")
   for success_count in sorted(success_distribution.keys()):
      count = success_distribution[success_count]
      percentage = (count / len(question_groups)) * 100
      print(f"  {success_count} successes: {count} questions ({percentage:.2f}%)")
   
   # Calculate average success rate
   avg_success_rate = np.mean([sum(correctness_list) / len(correctness_list) 
                              for correctness_list in question_groups.values()]) * 100
   print(f"\nAverage success rate: {avg_success_rate:.2f}%")
   
   # Tool retrieval specific statistics
   tool_stats = {}
   if tool_retrieval_enabled:
      print(f"\n" + "="*40)
      print("TOOL RETRIEVAL STATISTICS")
      print("="*40)
      
      # Collect tool usage data
      tool_counts = []
      retrieved_tools_all = []
      
      for result in results_lst:
         tool_count = result.get(f"{model_nickname}_available_tools_count", 0)
         retrieved_tools = result.get(f"{model_nickname}_retrieved_tools", [])
         
         tool_counts.append(tool_count)
         retrieved_tools_all.extend(retrieved_tools)
      
      # Tool usage statistics
      avg_tools_per_sample = np.mean(tool_counts) if tool_counts else 0
      max_tools_used = max(tool_counts) if tool_counts else 0
      min_tools_used = min(tool_counts) if tool_counts else 0
      
      print(f"Average tools per sample: {avg_tools_per_sample:.2f}")
      print(f"Max tools used: {max_tools_used}")
      print(f"Min tools used: {min_tools_used}")
      
      # Tool usage distribution
      tool_count_distribution = defaultdict(int)
      for count in tool_counts:
         tool_count_distribution[count] += 1
      
      print(f"\nTool count distribution:")
      for tool_count in sorted(tool_count_distribution.keys()):
         count = tool_count_distribution[tool_count]
         percentage = (count / len(tool_counts)) * 100 if tool_counts else 0
         print(f"  {tool_count} tools: {count} samples ({percentage:.2f}%)")
      
      # Most frequently retrieved tools
      tool_frequency = Counter(retrieved_tools_all)
      
      print(f"\nMost frequently retrieved tools:")
      for tool_name, frequency in tool_frequency.most_common(10):
         percentage = (frequency / len(results_lst)) * 100 if results_lst else 0
         print(f"  {tool_name}: {frequency} times ({percentage:.2f}%)")
      
      # Tool effectiveness analysis
      tool_effectiveness = {}
      for result in results_lst:
         tool_count = result.get(f"{model_nickname}_available_tools_count", 0)
         correctness = result.get(f"{model_nickname}_correctness", 0)
         
         if tool_count not in tool_effectiveness:
            tool_effectiveness[tool_count] = []
         tool_effectiveness[tool_count].append(correctness)
      
      print(f"\nTool effectiveness (accuracy by tool count):")
      for tool_count in sorted(tool_effectiveness.keys()):
         accuracies = tool_effectiveness[tool_count]
         avg_accuracy = np.mean(accuracies) * 100
         sample_count = len(accuracies)
         print(f"  {tool_count} tools: {avg_accuracy:.2f}% accuracy ({sample_count} samples)")
      
      tool_stats = {
         "average_tools_per_sample": avg_tools_per_sample,
         "max_tools_used": max_tools_used,
         "min_tools_used": min_tools_used,
         "tool_count_distribution": dict(tool_count_distribution),
         "tool_frequency": dict(tool_frequency.most_common(20)),
         "tool_effectiveness": {k: {"accuracy": np.mean(v) * 100, "count": len(v)} 
                              for k, v in tool_effectiveness.items()}
      }
   
   # Create statistics dictionary
   stats_dict = {
      "evaluation_time": current_time,
      "model_nickname": model_nickname,
      "passk": passk,
      "original_questions": original_questions_count,
      "total_samples": total_samples,
      "overall_accuracy": overall_accuracy,
      "failure_rate": failure_rate,
      "pass_at_k": pass_at_k_results,
      "average_success_rate": avg_success_rate,
      "success_distribution": dict(success_distribution),
      "tool_retrieval_enabled": tool_retrieval_enabled,
      "tool_statistics": tool_stats
   }
   
   # Save statistics to file
   stats_path = save_path.replace(".json", "_stats.json")
   save_json(data=stats_dict, file_path=stats_path)
   print(f"\nStatistics saved to: {stats_path}")
   
   print("="*60)
   
   return stats_dict


def generate_detailed_report(stats_dict, save_path):
   """
   Generate a detailed markdown report from statistics.
   
   Args:
       stats_dict: Statistics dictionary from calculate_evaluation_stats
       save_path: Path to save the report
       
   Returns:
       str: Path to the generated report file
   """
   report_path = save_path.replace(".json", "_report.md")
   
   report_content = f"""# Evaluation Report

## Basic Information
- **Model**: {stats_dict['model_nickname']}
- **Evaluation Time**: {stats_dict['evaluation_time']}
- **Original Questions**: {stats_dict['original_questions']}
- **Total Samples**: {stats_dict['total_samples']}
- **Pass@K Value**: {stats_dict['passk']}
- **Tool Retrieval Enabled**: {stats_dict['tool_retrieval_enabled']}

## Performance Metrics
- **Overall Accuracy**: {stats_dict['overall_accuracy']:.2f}%
- **Failure Rate**: {stats_dict['failure_rate']:.2f}%
- **Average Success Rate**: {stats_dict['average_success_rate']:.2f}%

## Pass@K Results
"""
   
   for k, accuracy in stats_dict['pass_at_k'].items():
      report_content += f"- **Pass@{k}**: {accuracy:.2f}%\n"
   
   report_content += "\n## Success Distribution\n"
   for success_count, count in stats_dict['success_distribution'].items():
      percentage = (count / stats_dict['original_questions']) * 100
      report_content += f"- **{success_count} successes**: {count} questions ({percentage:.2f}%)\n"
   
   # Tool statistics if available
   if stats_dict['tool_retrieval_enabled'] and stats_dict['tool_statistics']:
      tool_stats = stats_dict['tool_statistics']
      report_content += f"""
## Tool Retrieval Statistics
- **Average Tools per Sample**: {tool_stats['average_tools_per_sample']:.2f}
- **Max Tools Used**: {tool_stats['max_tools_used']}
- **Min Tools Used**: {tool_stats['min_tools_used']}

### Tool Usage Distribution
"""
      for tool_count, count in tool_stats['tool_count_distribution'].items():
         percentage = (count / stats_dict['total_samples']) * 100
         report_content += f"- **{tool_count} tools**: {count} samples ({percentage:.2f}%)\n"
      
      report_content += "\n### Most Frequently Retrieved Tools\n"
      for tool_name, frequency in list(tool_stats['tool_frequency'].items())[:10]:
         percentage = (frequency / stats_dict['total_samples']) * 100
         report_content += f"- **{tool_name}**: {frequency} times ({percentage:.2f}%)\n"
      
      report_content += "\n### Tool Effectiveness\n"
      for tool_count, effectiveness in tool_stats['tool_effectiveness'].items():
         accuracy = effectiveness['accuracy']
         count = effectiveness['count']
         report_content += f"- **{tool_count} tools**: {accuracy:.2f}% accuracy ({count} samples)\n"
   
   report_content += f"""
## Summary
This evaluation was conducted on {stats_dict['evaluation_time']} using the {stats_dict['model_nickname']} model.
The model achieved an overall accuracy of {stats_dict['overall_accuracy']:.2f}% across {stats_dict['total_samples']} samples.
"""
   
   if stats_dict['tool_retrieval_enabled']:
      report_content += f"Tool retrieval was enabled, with an average of {stats_dict['tool_statistics']['average_tools_per_sample']:.2f} tools used per sample."
   
   # Save report
   with open(report_path, 'w', encoding='utf-8') as f:
      f.write(report_content)
   
   print(f"Detailed report saved to: {report_path}")
   return report_path


def generate_comparison_report(stats_list, save_path="comparison_report.md"):
   """
   Generate a comparison report for multiple evaluation results.
   
   Args:
       stats_list: List of statistics dictionaries to compare
       save_path: Path to save the comparison report
       
   Returns:
       str: Path to the generated comparison report
   """
   if not stats_list:
      print("No statistics provided for comparison")
      return None
   
   report_content = f"""# Evaluation Comparison Report

## Models Compared
"""
   
   for i, stats in enumerate(stats_list):
      report_content += f"- **Model {i+1}**: {stats['model_nickname']} (evaluated on {stats['evaluation_time']})\n"
   
   report_content += "\n## Performance Comparison\n\n"
   
   # Create comparison table
   report_content += "| Model | Overall Accuracy | Pass@1 | Pass@4 | Avg Success Rate | Tool Retrieval |\n"
   report_content += "|-------|------------------|--------|--------|------------------|----------------|\n"
   
   for stats in stats_list:
      model = stats['model_nickname']
      accuracy = stats['overall_accuracy']
      pass_1 = stats['pass_at_k'].get(1, 0)
      pass_4 = stats['pass_at_k'].get(4, 0)
      avg_success = stats['average_success_rate']
      tool_enabled = "Yes" if stats['tool_retrieval_enabled'] else "No"
      
      report_content += f"| {model} | {accuracy:.2f}% | {pass_1:.2f}% | {pass_4:.2f}% | {avg_success:.2f}% | {tool_enabled} |\n"
   
   # Tool usage comparison if applicable
   tool_enabled_models = [stats for stats in stats_list if stats['tool_retrieval_enabled']]
   if tool_enabled_models:
      report_content += "\n## Tool Usage Comparison\n\n"
      report_content += "| Model | Avg Tools/Sample | Max Tools | Most Used Tool |\n"
      report_content += "|-------|------------------|-----------|----------------|\n"
      
      for stats in tool_enabled_models:
         tool_stats = stats['tool_statistics']
         model = stats['model_nickname']
         avg_tools = tool_stats['average_tools_per_sample']
         max_tools = tool_stats['max_tools_used']
         
         # Get most used tool
         most_used = "N/A"
         if tool_stats['tool_frequency']:
            most_used = list(tool_stats['tool_frequency'].keys())[0]
         
         report_content += f"| {model} | {avg_tools:.2f} | {max_tools} | {most_used} |\n"
   
   # Save comparison report
   with open(save_path, 'w', encoding='utf-8') as f:
      f.write(report_content)
   
   print(f"Comparison report saved to: {save_path}")
   return save_path


# Convenience function for easy importing
def quick_report(results_lst, model_nickname, passk, original_count, save_path, current_time, tool_retrieval_enabled=False):
    """
    Generate a comprehensive evaluation report with tool retrieval statistics.
    
    Args:
        results_lst: List of evaluation results
        model_nickname: Model name for reporting
        passk: Pass@K value used
        original_count: Original number of questions
        save_path: Path where results are saved
        current_time: Timestamp string
        tool_retrieval_enabled: Whether tool retrieval was enabled
    
    Returns:
        Tuple of (stats_dict, report_path)
    """
    # Basic statistics
    total_results = len(results_lst)
    correctness_key = f"{model_nickname}_correctness"
    correct_count = sum(1 for r in results_lst if r.get(correctness_key, 0) == 1)
    accuracy = correct_count / total_results if total_results > 0 else 0
    
    # Pass@K calculation
    unique_questions = {}
    for result in results_lst:
        base_id = result["id"].split("_")[0] if "_" in result["id"] else result["id"]
        if base_id not in unique_questions:
            unique_questions[base_id] = []
        unique_questions[base_id].append(result.get(correctness_key, 0))
    
    pass_at_k_count = 0
    for question_id, attempts in unique_questions.items():
        if any(attempt == 1 for attempt in attempts):
            pass_at_k_count += 1
    
    pass_at_k = pass_at_k_count / len(unique_questions) if unique_questions else 0
    
    # Tool usage statistics
    tool_stats = {}
    recall_stats = {}
    
    if tool_retrieval_enabled:
        tool_retrieval_key = f"{model_nickname}_tool_retrieval"
        
        # Collect recall metrics
        recall_at_1_values = []
        recall_at_3_values = []
        recall_at_5_values = []
        recall_at_10_values = []
        
        successful_retrievals = 0
        total_ground_truth_tools = 0
        total_retrieved_tools = 0
        questions_with_ground_truth = 0
        
        for result in results_lst:
            retrieval_info = result.get(tool_retrieval_key, {})
            
            if retrieval_info.get('success', False):
                successful_retrievals += 1
                
                # Collect recall values
                recall_at_1_values.append(retrieval_info.get('recall_at_1', 0.0))
                recall_at_3_values.append(retrieval_info.get('recall_at_3', 0.0))
                recall_at_5_values.append(retrieval_info.get('recall_at_5', 0.0))
                recall_at_10_values.append(retrieval_info.get('recall_at_10', 0.0))
                
                # Count ground truth and retrieved tools
                ground_truth_tools = retrieval_info.get('ground_truth_tools', [])
                if ground_truth_tools:
                    questions_with_ground_truth += 1
                    total_ground_truth_tools += len(ground_truth_tools)
                
                retrieved_tools = retrieval_info.get('retrieved_tools', [])
                total_retrieved_tools += len(retrieved_tools)
        
        # Calculate recall statistics
        if recall_at_1_values:
            recall_stats = {
                "recall_at_1": {
                    "mean": np.mean(recall_at_1_values),
                    "std": np.std(recall_at_1_values),
                    "min": np.min(recall_at_1_values),
                    "max": np.max(recall_at_1_values),
                    "count_perfect": sum(1 for r in recall_at_1_values if r == 1.0),
                    "count_zero": sum(1 for r in recall_at_1_values if r == 0.0)
                },
                "recall_at_3": {
                    "mean": np.mean(recall_at_3_values),
                    "std": np.std(recall_at_3_values),
                    "min": np.min(recall_at_3_values),
                    "max": np.max(recall_at_3_values),
                    "count_perfect": sum(1 for r in recall_at_3_values if r == 1.0),
                    "count_zero": sum(1 for r in recall_at_3_values if r == 0.0)
                },
                "recall_at_5": {
                    "mean": np.mean(recall_at_5_values),
                    "std": np.std(recall_at_5_values),
                    "min": np.min(recall_at_5_values),
                    "max": np.max(recall_at_5_values),
                    "count_perfect": sum(1 for r in recall_at_5_values if r == 1.0),
                    "count_zero": sum(1 for r in recall_at_5_values if r == 0.0)
                },
                "recall_at_10": {
                    "mean": np.mean(recall_at_10_values),
                    "std": np.std(recall_at_10_values),
                    "min": np.min(recall_at_10_values),
                    "max": np.max(recall_at_10_values),
                    "count_perfect": sum(1 for r in recall_at_10_values if r == 1.0),
                    "count_zero": sum(1 for r in recall_at_10_values if r == 0.0)
                }
            }
        
        tool_stats = {
            "successful_retrievals": successful_retrievals,
            "total_attempts": total_results,
            "retrieval_success_rate": successful_retrievals / total_results if total_results > 0 else 0,
            "questions_with_ground_truth": questions_with_ground_truth,
            "avg_ground_truth_tools_per_question": total_ground_truth_tools / questions_with_ground_truth if questions_with_ground_truth > 0 else 0,
            "avg_retrieved_tools_per_question": total_retrieved_tools / successful_retrievals if successful_retrievals > 0 else 0,
            "total_ground_truth_tools": total_ground_truth_tools,
            "total_retrieved_tools": total_retrieved_tools
        }
    
    # Compile comprehensive statistics
    stats_dict = {
        "model_nickname": model_nickname,
        "timestamp": current_time,
        "total_results": total_results,
        "original_questions": original_count,
        "pass_k": passk,
        "unique_questions": len(unique_questions),
        "correct_answers": correct_count,
        "accuracy": accuracy,
        "pass_at_k": pass_at_k,
        "tool_retrieval_enabled": tool_retrieval_enabled,
        "tool_statistics": tool_stats,
        "recall_statistics": recall_stats
    }
    
    # Save statistics
    stats_path = save_path.replace('.json', '_stats.json')
    save_json(data=stats_dict, file_path=stats_path)
    
    # Generate detailed report
    report_path = save_path.replace('.json', '_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"EVALUATION REPORT - {model_nickname}\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {current_time}\n")
        f.write(f"Results file: {save_path}\n")
        f.write(f"Statistics file: {stats_path}\n\n")
        
        # Basic Performance
        f.write("BASIC PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Results: {total_results:,}\n")
        f.write(f"Original Questions: {original_count:,}\n")
        f.write(f"Unique Questions: {len(unique_questions):,}\n")
        f.write(f"Pass@{passk}: {pass_at_k:.4f} ({pass_at_k_count}/{len(unique_questions)})\n")
        f.write(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_results})\n\n")
        
        # Tool Retrieval Performance
        if tool_retrieval_enabled and recall_stats:
            f.write("TOOL RETRIEVAL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Retrieval Success Rate: {tool_stats['retrieval_success_rate']:.4f} ({tool_stats['successful_retrievals']}/{tool_stats['total_attempts']})\n")
            f.write(f"Questions with Ground Truth: {tool_stats['questions_with_ground_truth']:,}\n")
            f.write(f"Avg Ground Truth Tools per Question: {tool_stats['avg_ground_truth_tools_per_question']:.2f}\n")
            f.write(f"Avg Retrieved Tools per Question: {tool_stats['avg_retrieved_tools_per_question']:.2f}\n\n")
            
            f.write("RECALL METRICS\n")
            f.write("-" * 40 + "\n")
            for k in [1, 3, 5, 10]:
                recall_key = f"recall_at_{k}"
                if recall_key in recall_stats:
                    stats = recall_stats[recall_key]
                    f.write(f"Recall@{k}:\n")
                    f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                    f.write(f"  Perfect Recall: {stats['count_perfect']}/{len(recall_at_1_values)} ({stats['count_perfect']/len(recall_at_1_values)*100:.1f}%)\n")
                    f.write(f"  Zero Recall: {stats['count_zero']}/{len(recall_at_1_values)} ({stats['count_zero']/len(recall_at_1_values)*100:.1f}%)\n\n")
        
        elif tool_retrieval_enabled:
            f.write("TOOL RETRIEVAL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write("Tool retrieval was enabled but no successful retrievals found.\n\n")
        
        else:
            f.write("TOOL RETRIEVAL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write("Tool retrieval was disabled for this evaluation.\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"✓ Evaluated {total_results:,} responses from {len(unique_questions):,} unique questions\n")
        f.write(f"✓ Overall Pass@{passk}: {pass_at_k:.4f}\n")
        f.write(f"✓ Overall Accuracy: {accuracy:.4f}\n")
        
        if tool_retrieval_enabled and recall_stats:
            f.write(f"✓ Tool Retrieval Success: {tool_stats['retrieval_success_rate']:.4f}\n")
            f.write(f"✓ Average Recall@5: {recall_stats['recall_at_5']['mean']:.4f}\n")
            f.write(f"✓ Average Recall@10: {recall_stats['recall_at_10']['mean']:.4f}\n")
    
    # Print summary to console
    print(f"\n📊 EVALUATION SUMMARY")
    print(f"   Pass@{passk}: {pass_at_k:.4f} ({pass_at_k_count}/{len(unique_questions)})")
    print(f"   Accuracy: {accuracy:.4f} ({correct_count}/{total_results})")
    
    if tool_retrieval_enabled and recall_stats:
        print(f"   Tool Retrieval Success: {tool_stats['retrieval_success_rate']:.4f}")
        print(f"   Recall@1: {recall_stats['recall_at_1']['mean']:.4f}")
        print(f"   Recall@3: {recall_stats['recall_at_3']['mean']:.4f}")
        print(f"   Recall@5: {recall_stats['recall_at_5']['mean']:.4f}")
        print(f"   Recall@10: {recall_stats['recall_at_10']['mean']:.4f}")
    
    return stats_dict, report_path


if __name__ == "__main__":
   print("Report utils module - import this to use the reporting functions")
   print("Available functions:")
   print("  - calculate_evaluation_stats()")
   print("  - generate_detailed_report()")
   print("  - generate_comparison_report()")
   print("  - quick_report() - convenience function") 