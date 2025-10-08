import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from ToolAggregationAgent import ToolAggregationAgent, setup_logging, ToolAggregationResult
import json
import sys
from typing import List, Dict, Any
import os
import glob
import re

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
                debug_cluster_count = 5
                debug_tool_count = 10
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
                                model_name: str, debug: bool = False, max_threads: int = 10, verification_model_name: str = 'gpt-4.1') -> Dict[str, Any]:
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
            success, final_tool, processing_result = agent.process_single_sib(cname, ctools, sib, cdir,verification_model_name=verification_model_name)
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


def get_public_function_name(tool_code: str) -> str:
    """
    Get the public function name from tool code
    """
    if not tool_code or not tool_code.strip():
        return ""

    # Pre-clean common imports to avoid parsing interference
    cleaned = tool_code.replace("from __future__ import annotations\n\n", "")
    # cleaned = cleaned.replace("from typing import List, Dict, Any\n\n", "")

    try:
        import ast
        tree = ast.parse(cleaned)
        # Only functions at module top level (not inside class/if/def)
        candidates = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if not candidates:
            return ""

        # If only one, return directly
        if len(candidates) == 1:
            return candidates[0]

        # Filter out private and magic methods
        filtered = [n for n in candidates if not n.startswith("_") and not (n.startswith("__") and n.endswith("__"))]
        if len(filtered) == 1:
            return filtered[0]

        # If execute already exists, return execute
        if "execute" in candidates:
            return "execute"

        # Fallback to first candidate
        return candidates[0]
    except Exception:
        # When parsing fails, use regex to match def lines at top level (ignore indentation inside class)
        # Match def/async def starting at line beginning without indentation
        pattern = r"^(?:def|async\s+def)\s+([a-zA-Z_]\w*)\s*\("
        for line in cleaned.splitlines():
            if line.lstrip() != line:
                # Has indentation, skip (might be inside class/if)
                continue
            m = re.match(pattern, line)
            if m:
                return m.group(1)
        return ""


def rename_public_function_to_execute(tool_code: str) -> tuple[str, str]:
    """
    Rename the unique public function at module top level to execute.

    Returns (new_code, original_function_name). If not found, returns (original_code, "").
    """
    if not tool_code or not tool_code.strip():
        return tool_code, ""

    original_name = get_public_function_name(tool_code)
    if not original_name:
        return tool_code, ""
    if original_name == "execute":
        return tool_code, original_name

    # Only replace def declaration lines to avoid affecting other identifiers with same name
    # Handle "def name(" and "async def name("
    def_pattern = rf"^(\s*def\s+){re.escape(original_name)}(\s*\()"
    async_def_pattern = rf"^(\s*async\s+def\s+){re.escape(original_name)}(\s*\())"

    # Process line by line, prioritize async def
    changed = False
    lines = tool_code.splitlines()
    for i, line in enumerate(lines):
        # async def has priority
        if re.match(async_def_pattern, line):
            lines[i] = re.sub(async_def_pattern, r"\1execute\2", line, count=1)
            changed = True
            break
        if re.match(def_pattern, line):
            lines[i] = re.sub(def_pattern, r"\1execute\2", line, count=1)
            changed = True
            break

    new_code = "\n".join(lines)
    return (new_code if changed else tool_code), original_name


def collect_tool_info_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Collect tool_info and tool_code from all xxx_final_openai_tools.json files in the specified folder
    
    Args:
        folder_path: Target folder path
        
    Returns:
        List containing all tool information, each element contains tool_info and tool_code
    """
    all_tools = []
    
    # Use glob to recursively search for all *_final_openai_tools.json files
    pattern = os.path.join(folder_path, "**", "*_final_openai_tools.json")
    json_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            function_name_freq = {}
            # Check if there is a tools field
            if 'tools' in data and isinstance(data['tools'], list):
                for tool in data['tools']:
                    # Extract tool_info and tool_code
                    if 'tool_info' in tool and 'tool_code' in tool:
                        # Unify function name to execute, and keep original function name
                        function_name = tool['tool_info']['function']['name']                            
                        function_name_freq_num = tool['tool_code'].count(f"def {function_name}(")
                        if function_name_freq_num!=1:
                            continue
                        all_tools.append({"tool_info": tool['tool_info'], "tool_code": tool['tool_code'].replace("from __future__ import annotations\n\n", "")})

                        # all_tools.append({"tool_info": tool['tool_info'], "tool_code": tool['tool_code'].replace("from __future__ import annotations\n\n", "").replace(f"def {function_name}(", "def execute(")})
                        
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
            continue
    
    print(f"Total collected {len(all_tools)} tools")
    print(function_name_freq)
    return all_tools




def main():
    """Main function"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Tool Aggregation Agent')
    parser.add_argument('--file', default='', help='Path to the clusters JSON file')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')
    parser.add_argument('--model_name', type=str, default='o4-mini', help='Model name to use')
    parser.add_argument('--verification_model_name', type=str, default='gpt-4.1', help='Model name to use')
    parser.add_argument('--max-threads', type=int, default=20, help='Maximum number of parallel threads')
    debug_suffix = "_debug" if '--debug' in os.sys.argv else ""
    default_output_dir = f'log/lib_{timestamp}{debug_suffix}'
    parser.add_argument('--output-dir', type=str, default=default_output_dir, help='Output directory for generated code')
    parser.add_argument('--log-folder', type=str, default=default_output_dir, help='Log folder for generated code')
    parser.add_argument('--save_json', action='store_true', default=True, help='Enable local mode')
    args = parser.parse_args()
        
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
            max_threads=args.max_threads,
            verification_model_name=args.verification_model_name
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
    
    folder = args.output_dir
    
    # Collect all tool information
    all_tools = collect_tool_info_from_folder(folder)
    
    # Save results
    output_file = args.output_dir + "/collected_tools_info.json"
    save_json(all_tools, output_file)
    print(f"Results saved to {output_file}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"- Total tool count: {len(all_tools)}")
    
    # Count different tool names
    tool_names = set()
    for tool in all_tools:
        if 'tool_info' in tool and 'function' in tool['tool_info'] and 'name' in tool['tool_info']['function']:
            tool_names.add(tool['tool_info']['function']['name'])
    
    print(f"- Number of different tool names: {len(tool_names)}")
    print(f"- First 10 tool names: {list(tool_names)[:10]}")

    

if __name__ == "__main__":
    main()

