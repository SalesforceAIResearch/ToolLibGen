#!/usr/bin/env python3
"""
Example of how to use the report_utils module for evaluation analysis.
This demonstrates various reporting capabilities.
"""

import sys
import os
from datetime import datetime

# Add eval directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from report_utils import (
    calculate_evaluation_stats,
    generate_detailed_report,
    generate_comparison_report,
    quick_report
)
from utils import read_json


def example_single_report():
    """Example of generating a report for a single evaluation result."""
    print("=== Single Evaluation Report Example ===")
    
    # This would typically be the path to your evaluation results
    result_file = "/export/home/data/all_science_data_Qwen3_answer_20250101_120000.json"
    
    # For demo purposes, create mock results
    if not os.path.exists(result_file):
        print("Demo with mock data (adjust paths for real data)")
        mock_results = create_mock_results("Qwen3_8b", 40, 4)
        model_nickname = "Qwen3_8b"
        passk = 4
        original_count = 10
        save_path = "demo_results.json"
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate report
        stats_dict, report_path = quick_report(
            results_lst=mock_results,
            model_nickname=model_nickname,
            passk=passk,
            original_count=original_count,
            save_path=save_path,
            current_time=current_time,
            tool_retrieval_enabled=True
        )
        
        print(f"📊 Statistics: demo_results_stats.json")
        print(f"📝 Report: {report_path}")
    else:
        # Real data processing
        results = read_json(result_file)
        # Process real results...
        print("Processing real evaluation results...")


def example_comparison_report():
    """Example of generating a comparison report between different models."""
    print("\n=== Model Comparison Report Example ===")
    
    # Create mock comparison data
    model_stats = []
    
    # Mock stats for different models
    models = ["Qwen3_8b", "Qwen3_8b_with_tools", "GPT4_baseline"]
    
    for i, model in enumerate(models):
        mock_results = create_mock_results(model, 40, 4, accuracy_base=0.6 + i*0.1)
        stats = calculate_evaluation_stats(
            results_lst=mock_results,
            model_nickname=model,
            passk=4,
            original_questions_count=10,
            save_path=f"mock_{model}_results.json",
            current_time=datetime.now().strftime("%Y%m%d_%H%M%S"),
            tool_retrieval_enabled=(i == 1)  # Only second model uses tools
        )
        model_stats.append(stats)
    
    # Generate comparison report
    comparison_path = generate_comparison_report(model_stats, "model_comparison.md")
    print(f"📊 Comparison report: {comparison_path}")


def create_mock_results(model_nickname, total_samples, passk, accuracy_base=0.65):
    """Create mock evaluation results for demonstration."""
    import random
    random.seed(42)  # For reproducible results
    
    results = []
    original_questions = total_samples // passk
    
    for q_id in range(original_questions):
        for attempt in range(passk):
            # Simulate varying accuracy
            correctness = 1 if random.random() < accuracy_base else 0
            
            result = {
                "id": f"question_{q_id}_{attempt}",
                "question": f"Mock question {q_id}",
                "answer": "mock_answer",
                f"{model_nickname}_correctness": correctness,
                f"{model_nickname}_response": f"Mock response for attempt {attempt}",
                f"{model_nickname}_pred_answer": "mock_pred"
            }
            
            # Add tool retrieval data for demonstration
            if "tools" in model_nickname.lower():
                result[f"{model_nickname}_available_tools_count"] = random.randint(1, 8)
                result[f"{model_nickname}_retrieved_tools"] = [
                    f"tool_{random.randint(1, 20)}" for _ in range(random.randint(0, 3))
                ]
            
            results.append(result)
    
    return results


def example_advanced_analysis():
    """Example of advanced analysis using the reporting utilities."""
    print("\n=== Advanced Analysis Example ===")
    
    # Create mock data with different characteristics
    base_results = create_mock_results("base_model", 80, 4, 0.6)
    tool_results = create_mock_results("tool_model", 80, 4, 0.75)
    
    print("Analyzing base model...")
    base_stats = calculate_evaluation_stats(
        base_results, "base_model", 4, 20, "base_results.json",
        datetime.now().strftime("%Y%m%d_%H%M%S"), False
    )
    
    print("\nAnalyzing tool-enhanced model...")
    tool_stats = calculate_evaluation_stats(
        tool_results, "tool_model", 4, 20, "tool_results.json",
        datetime.now().strftime("%Y%m%d_%H%M%S"), True
    )
    
    # Compare key metrics
    print(f"\n📊 Performance Comparison:")
    print(f"Base model accuracy: {base_stats['overall_accuracy']:.2f}%")
    print(f"Tool model accuracy: {tool_stats['overall_accuracy']:.2f}%")
    print(f"Improvement: {tool_stats['overall_accuracy'] - base_stats['overall_accuracy']:.2f}%")
    
    print(f"\nPass@1 comparison:")
    print(f"Base model: {base_stats['pass_at_k'][1]:.2f}%")
    print(f"Tool model: {tool_stats['pass_at_k'][1]:.2f}%")
    
    # Generate individual reports
    base_report = generate_detailed_report(base_stats, "base_results.json")
    tool_report = generate_detailed_report(tool_stats, "tool_results.json")
    
    # Generate comparison report
    comparison_report = generate_comparison_report([base_stats, tool_stats], "advanced_comparison.md")
    
    print(f"\n📝 Reports generated:")
    print(f"  Base model: {base_report}")
    print(f"  Tool model: {tool_report}")
    print(f"  Comparison: {comparison_report}")


if __name__ == "__main__":
    print("🔍 Evaluation Reporting Examples")
    print("=" * 50)
    
    try:
        example_single_report()
        example_comparison_report()
        example_advanced_analysis()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\n📚 To use in your own evaluations:")
        print("1. Import: from report_utils import quick_report")
        print("2. Use: stats_dict, report_path = quick_report(results, model, passk, count, save_path)")
        print("3. For comparisons: generate_comparison_report([stats1, stats2, ...])")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Note: This is a demo. Adjust file paths for your actual data.") 