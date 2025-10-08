import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from ToolAggregationAgent_v3 import ToolAggregationAgent, setup_logging, ToolAggregationResult

from utils import read_jsonl, call_openai_api, map_with_progress, save_json, read_json, call_openai_api_multi_turn, validate_code_syntax, execute_code, apply_patch
import random

def load_clusters_data(clusters_file_path: str, debug: bool = False) -> Dict[str, List[Dict]]:
    """Load and parse clusters data from JSON file"""
    try:
        data = read_json(clusters_file_path)
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
                debug_cluster_count = 1
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
    """Two-phase processing: (1) blueprint all clusters, (2) process all SIBs independently."""
    total_clusters = len(clusters_data)
    print(f"\n🚀 Starting processing for {total_clusters} clusters (two-phase)")
    print(f"🔧 Model: {model_name}")
    print(f"🧵 Max threads: {min(total_clusters, max_threads)}")

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
        # Phase 1: Blueprint all clusters to obtain SIBs
        print("\n🎯 Phase 1: Designing blueprints for all clusters...")
        blueprint_items = []
        cluster_names = list(clusters_data.keys())
        for i, cluster_name in enumerate(cluster_names, 1):
            cluster_tools = clusters_data[cluster_name]
            cluster_output_dir = output_dir / f"{cluster_name}_output"
            cluster_output_dir.mkdir(exist_ok=True)
            blueprint_items.append((i, cluster_name, cluster_tools, total_clusters, cluster_output_dir))

        def _blueprint_wrapper(args: Tuple[int, str, List[Dict], int, Path]):
            idx, cname, ctools, tcount, cdir = args
            agent = ToolAggregationAgent(model_name=model_name, debug=debug)
            print(f"[{idx}/{tcount}] Blueprinting {cname}...")
            success, sibs, error = agent.design_blueprint_only(cname, ctools, model_name="gpt-5")
            # Save blueprint SIBs along with their covered tools
            try:
                enriched_sibs = []
                for sib in (sibs or []):
                    covered_indices = sib.get('covered_tools', []) or []
                    covered_tools = []
                    for ti in covered_indices:
                        if isinstance(ti, int) and 0 <= ti < len(ctools):
                            covered_tools.append(ctools[ti])
                    enriched_sibs.append({
                        'sib': sib,
                        'covered_tools': covered_tools
                    })
                save_json({
                    'cluster_name': cname,
                    'timestamp': datetime.now().isoformat(),
                    'total_sibs': len(sibs or []),
                    'sibs': enriched_sibs
                }, str(cdir / 'blueprint_sibs.json'))
            except Exception as e:
                print(f"⚠️ Failed to save blueprint SIBs for {cname}: {e}")
            return {
                'cluster_index': idx,
                'cluster_name': cname,
                'tools': ctools,
                'sibs': sibs if success else [],
                'blueprint_success': success,
                'blueprint_error': error,
                'output_dir': str(cdir)
            }

        blueprint_results = map_with_progress(
            _blueprint_wrapper,
            blueprint_items,
            num_threads=100,
            pbar=True
        )

        # Phase 2: Flatten all SIBs across clusters and process independently
        print("\n🧩 Phase 2: Processing all SIBs independently across clusters...")
        sib_tasks = []
        for bres in blueprint_results:
            cname = bres['cluster_name']
            ctools = bres['tools']
            cdir = Path(bres['output_dir'])
            sibs = bres['sibs'] or []
            for sib in sibs:
                sib_tasks.append((cname, ctools, sib, cdir))

        print(f"Total SIBs to process: {len(sib_tasks)}")

        def _sib_wrapper(args: Tuple[str, List[Dict], Dict, Path]):
            cname, ctools, sib, cdir = args
            agent = ToolAggregationAgent(model_name=model_name, debug=debug)
            success, final_tool, processing_result = agent.process_single_sib(cname, ctools, sib, cdir)
            return {
                'cluster_name': cname,
                'sib_index': processing_result.get('sib_index', sib.get('index', 0)) if isinstance(processing_result, dict) else sib.get('index', 0),
                'success': bool(success),
                'final_tool': final_tool,
                'processing_result': processing_result
            }

        sib_results = []
        if sib_tasks:
            sib_results = map_with_progress(
                _sib_wrapper,
                sib_tasks,
                num_threads=50,
                pbar=True
            )

        # Aggregate results by cluster
        from collections import defaultdict
        cluster_to_tools = defaultdict(list)
        cluster_to_processing = defaultdict(list)

        for r in sib_results:
            if r.get('success') and r.get('final_tool'):
                cluster_to_tools[r['cluster_name']].append(r['final_tool'])
            if r.get('processing_result'):
                cluster_to_processing[r['cluster_name']].append(r['processing_result'])

        # Build cluster-level ToolAggregationResult summaries (minimal)
        for bres in blueprint_results:
            cname = bres['cluster_name']
            tools = clusters_data[cname]
            result = ToolAggregationResult(
                cluster_name=cname,
                total_tools=len(tools)
            )
            if bres['blueprint_success']:
                result.steps_completed.append("blueprint_design")
            final_tools = cluster_to_tools.get(cname, [])
            result.openai_tools = final_tools
            result.success = len(final_tools) > 0
            result.final_code = None
            result.error_message = bres['blueprint_error'] if not bres['blueprint_success'] else None
            summary['cluster_results'].append(result.__dict__)
            if result.success:
                summary['successful_clusters'] += 1
            else:
                summary['failed_clusters'] += 1

    except Exception as e:
        print(f"❌ Error in two-phase processing: {e}")

    summary['processing_time'] = time.time() - start_time

    print(f"\n🏁 Processing completed!")
    print(f"✅ {summary['successful_clusters']}/{summary['total_clusters']} clusters processed successfully")

    return summary

def main():
    """Main function"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Tool Aggregation Agent')
    parser.add_argument('--local', action='store_true', default=False, help='Enable local mode')
    parser.add_argument('--file', default='/export/home/data/math_103k_tools_saved_all.json', help='Path to the clusters JSON file')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--model-name', type=str, default='o4-mini', help='Model name to use')
    parser.add_argument('--max-threads', type=int, default=20, help='Maximum number of parallel threads')
    debug_suffix = "_debug" if '--debug' in os.sys.argv else ""
    default_output_dir = f'/export/home/temp_lib/lib_{timestamp}{debug_suffix}'
    parser.add_argument('--output-dir', type=str, default=default_output_dir, help='Output directory for generated code')
    parser.add_argument('--log-folder', type=str, default=default_output_dir, help='Log folder for generated code')
    args = parser.parse_args()
    
    if args.local:
        args.file = "/Users/murong.yue/Desktop/data/Nemotron_science_data_tools_saved_all.json"
        args.output_dir = f"/Users/murong.yue/Desktop/temp_lib/math_lib_{timestamp}"
        args.log_folder = f"/Users/murong.yue/Desktop/log/math_lib_{timestamp}"
    
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

