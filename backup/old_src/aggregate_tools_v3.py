import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from ToolAggregationAgent import ToolAggregationAgent, setup_logging, ToolAggregationResult

from utils import read_jsonl, call_openai_api, map_with_progress, save_json, read_json, call_openai_api_multi_turn, validate_code_syntax, execute_code, apply_patch

def load_clusters_data(clusters_file_path: str, debug: bool = False) -> Dict[str, List[Dict]]:
    """Load and parse clusters data from JSON file"""
    try:
        data = read_json(clusters_file_path)
        data = data[:50]
        
        if 'clusters' in data:
            clusters_list = data['clusters']
            if isinstance(clusters_list, list):
                clusters_data = {}
                for cluster in clusters_list:
                    cluster_name = cluster.get('cluster_name', f'cluster_{len(clusters_data)}')
                    tools = cluster.get('tools', [])
                    clusters_data[cluster_name] = tools
            else:
                clusters_data = clusters_list
        else:
            if isinstance(data, list):
                clusters_data = {}
                for i, cluster in enumerate(data):
                    if isinstance(cluster, dict) and 'cluster_name' in cluster:
                        cluster_name = cluster['cluster_name']
                        tools = cluster.get('tools', [])
                    else:
                        cluster_name = f'cluster_{i}'
                        tools = cluster if isinstance(cluster, list) else []
                    clusters_data[cluster_name] = tools
            else:
                clusters_data = data
        
        if debug:
            # Debug mode: select first cluster and first 20 tools
            if clusters_data:
                debug_cluster_count = 3
                debug_tool_count = 20
                selected_cluster_names = list(clusters_data.keys())[:debug_cluster_count]
                debug_clusters = {}
                for cluster_name in selected_cluster_names:
                    cluster_tools = clusters_data[cluster_name][:debug_tool_count]
                    debug_clusters[cluster_name] = cluster_tools
                clusters_data = debug_clusters
        
        return clusters_data
        
    except Exception as e:
        raise Exception(f"Failed to load clusters data: {e}")

def process_single_cluster_wrapper(args: Tuple) -> ToolAggregationResult:
    """Wrapper function for parallel cluster processing"""
    cluster_index, cluster_name, tools, total_clusters, output_dir, model_name, debug = args
    
    # Create a new agent instance for this cluster
    agent = ToolAggregationAgent(model_name=model_name, debug=debug)
    
    # Create cluster-specific output directory
    cluster_output_dir = output_dir / f"{cluster_name}_output"
    cluster_output_dir.mkdir(exist_ok=True)
    
    print(f"[{cluster_index}/{total_clusters}] Processing {cluster_name}...")
    
    # Process the cluster
    result = agent.process_single_cluster(cluster_name, tools, cluster_output_dir)
    
    print(f"[{cluster_index}/{total_clusters}] {'✅' if result.success else '❌'} {cluster_name} completed")
    
    return result

def process_all_clusters_parallel(clusters_data: Dict[str, List[Dict]], output_dir: Path, 
                                model_name: str, debug: bool = False, max_threads: int = 10) -> Dict[str, Any]:
    """Process all clusters using parallel processing"""
    
    total_clusters = len(clusters_data)
    print(f"\n🚀 Starting parallel processing for {total_clusters} clusters")
    print(f"🔧 Model: {model_name}")
    print(f"🧵 Max threads: {min(total_clusters, max_threads)}")
    
    # List all clusters to be processed
    print(f"\n📋 Clusters to process:")
    for i, (cluster_name, tools) in enumerate(clusters_data.items(), 1):
        print(f"  {i}. {cluster_name} ({len(tools)} tools)")
    
    summary = {
        'total_clusters': total_clusters,
        'successful_clusters': 0,
        'failed_clusters': 0,
        'cluster_results': [],
        'processing_time': 0,
        'total_tools': sum(len(tools) for tools in clusters_data.values())
    }
    
    start_time = time.time()
    
    try:
        # Prepare cluster items with index information
        cluster_items = []
        for i, (cluster_name, tools) in enumerate(clusters_data.items(), 1):
            cluster_items.append((i, cluster_name, tools, total_clusters, output_dir, model_name, debug))
        
        print(f"\n🎯 Starting parallel processing with {min(len(cluster_items), max_threads)} threads...")
        
        # Use map_with_progress for multi-threaded cluster processing
        cluster_results = map_with_progress(
            process_single_cluster_wrapper,
            cluster_items,
            num_threads=min(len(cluster_items), max_threads),
            pbar=True
        )
        
        # Process results
        for result in cluster_results:
            summary['cluster_results'].append(result.__dict__)
            
            if result.success:
                summary['successful_clusters'] += 1
                
                # Save final code if available
                if result.final_code:
                    final_code_path = output_dir / f"{result.cluster_name}.py"
                    with open(final_code_path, 'w', encoding='utf-8') as f:
                        f.write(result.final_code)
                    print(f"📁 Saved final code: {result.cluster_name}.py")
            else:
                summary['failed_clusters'] += 1
    
    except Exception as e:
        print(f"❌ Error in parallel cluster processing: {e}")
    
    summary['processing_time'] = time.time() - start_time
    
    print(f"\n🏁 Parallel processing completed!")
    print(f"✅ {summary['successful_clusters']}/{summary['total_clusters']} clusters processed successfully")
    
    return summary

def main():
    """Main function"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Tool Aggregation Agent')
    parser.add_argument('--local', action='store_true', default=False, help='Enable local mode')
    parser.add_argument('--file', default='/export/home/data/adaptive_merged_tool_clusters_with_QA.json', help='Path to the clusters JSON file')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--model-name', type=str, default='gpt-5', help='Model name to use')
    parser.add_argument('--max-threads', type=int, default=5, help='Maximum number of parallel threads')
    debug_suffix = "_debug" if '--debug' in os.sys.argv else ""
    default_output_dir = f'/export/home/temp_lib/lib_{timestamp}{debug_suffix}'
    parser.add_argument('--output-dir', type=str, default=default_output_dir, help='Output directory for generated code')
    parser.add_argument('--log-folder', type=str, default=default_output_dir, help='Log folder for generated code')
    args = parser.parse_args()
    
    if args.local:
        args.file = "/Users/murong.yue/Desktop/data/Nemotron_science_data_tools_saved_all.json"
        args.output_dir = f"/Users/murong.yue/Desktop/temp_lib/phy_lib_{timestamp}"
        args.log_folder = f"/Users/murong.yue/Desktop/log/phy_lib_{timestamp}"
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file, progress_logger = setup_logging(debug=args.debug, log_folder=args.log_folder)
    
    try:
        # Load clusters data
        clusters_data = load_clusters_data(args.file, debug=args.debug)
        
        # Process all clusters in parallel
        summary = process_all_clusters_parallel(
            clusters_data=clusters_data,
            output_dir=output_dir,
            model_name=args.model_name,
            debug=args.debug,
            max_threads=args.max_threads
        )
        
        # Final summary
        print("\n" + "="*60)
        print("🎯 PARALLEL PROCESSING SUMMARY")
        print("="*60)
        print(f"✅ Successful: {summary['successful_clusters']}/{summary['total_clusters']}")
        print(f"📦 Total tools processed: {summary['total_tools']}")
        print(f"⏱️ Time: {summary['processing_time']:.1f}s")
        print(f"📁 Output: {args.output_dir}")
        
        # Show failed clusters
        failed_clusters = [r for r in summary['cluster_results'] if not r['success']]
        if failed_clusters:
            print(f"\n❌ Failed clusters:")
            for cluster_result in failed_clusters:
                print(f"  • {cluster_result['cluster_name']}: {cluster_result.get('error_message', 'Unknown error')}")
        
        # Show successful clusters
        successful_clusters = [r for r in summary['cluster_results'] if r['success']]
        if successful_clusters:
            print(f"\n✅ Successful clusters:")
            for cluster_result in successful_clusters:
                print(f"  • {cluster_result['cluster_name']}: {cluster_result.get('total_tools', 0)} tools processed")
        
        print(f"\n📝 Log: {log_file}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()

