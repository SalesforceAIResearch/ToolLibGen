import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import multiprocessing as mp

from ToolAggregationAgent_v3 import ToolAggregationAgent, setup_logging
from utils import read_json, save_json, map_with_progress


def find_cluster_output_dirs(root: Path) -> List[Path]:
    """Return all subdirectories that contain a blueprint_sibs.json file.
    If the root itself contains blueprint_sibs.json, return [root].
    """
    dirs: List[Path] = []
    if (root / "blueprint_sibs.json").exists():
        dirs.append(root)
        return dirs
    for child in root.iterdir():
        if child.is_dir() and (child / "blueprint_sibs.json").exists():
            dirs.append(child)
    return sorted(dirs)


def load_blueprint_sibs(cluster_dir: Path) -> Dict[str, Any]:
    """Load blueprint_sibs.json from a cluster output directory."""
    bp_path = cluster_dir / "blueprint_sibs.json"
    if not bp_path.exists():
        raise FileNotFoundError(f"Missing blueprint_sibs.json in {cluster_dir}")
    return read_json(str(bp_path))


def compute_completed_sib_indices(cluster_dir: Path) -> List[int]:
    """Detect completed SIBs by '*_sib_{idx}_final_openai_tools.json' presence."""
    completed: List[int] = []
    for p in cluster_dir.iterdir():
        name = p.name
        if not p.is_file():
            continue
        if name.endswith("_final_openai_tools.json") and "_sib_" in name:
            try:
                # match pattern '*_sib_{idx}_final_openai_tools.json'
                middle = name.split("_sib_")[-1]
                idx_str = middle.split("_")[0]
                completed.append(int(idx_str))
            except Exception:
                continue
    return sorted(set(completed))


def build_sparse_tools_list_from_blueprint(blueprint: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    """Construct a sparse tools list such that tools[i] exists for each covered index i across all SIBs.
    We use the enriched entry 'covered_tools' stored alongside each SIB in blueprint_sibs.json.
    """
    # blueprint['sibs'] entries are objects: { 'sib': <sib_dict>, 'covered_tools': [tool_dict, ...] }
    max_index = -1
    index_to_tool: Dict[int, Dict[str, Any]] = {}

    for entry in blueprint.get("sibs", []):
        sib_obj = entry.get("sib", {})
        covered_indices = sib_obj.get("covered_tools", []) or []
        covered_tools = entry.get("covered_tools", []) or []
        for i, tool_index in enumerate(covered_indices):
            if isinstance(tool_index, int):
                max_index = max(max_index, tool_index)
                # Prefer first occurrence; assume consistent across SIBs
                if tool_index not in index_to_tool and i < len(covered_tools):
                    index_to_tool[tool_index] = covered_tools[i]

    tools: List[Optional[Dict[str, Any]]] = [None] * (max_index + 1 if max_index >= 0 else 0)
    for tool_index, tool in index_to_tool.items():
        if tool_index < len(tools):
            tools[tool_index] = tool
    return tools


def list_pending_sibs(cluster_dir: Path) -> Tuple[str, List[Dict[str, Any]], List[int]]:
    """Return (cluster_name, sib_dicts, pending_indices) for a cluster output dir.
    sib_dicts are the raw SIB dicts (not enriched), as expected by ToolAggregationAgent.
    """
    blueprint = load_blueprint_sibs(cluster_dir)
    cluster_name = blueprint.get("cluster_name", cluster_dir.name.replace("_output", ""))

    # Extract raw SIB dicts
    sib_wrappers: List[Dict[str, Any]] = blueprint.get("sibs", [])
    sib_dicts: List[Dict[str, Any]] = [w.get("sib", {}) for w in sib_wrappers]

    completed = set(compute_completed_sib_indices(cluster_dir))
    all_indices = []
    for sib in sib_dicts:
        if isinstance(sib, dict):
            sib_index = sib.get("index", None)
            if isinstance(sib_index, int):
                all_indices.append(sib_index)

    pending = sorted([i for i in all_indices if i not in completed])
    return cluster_name, sib_dicts, pending


def _sib_runner_proc(cluster_name: str, tools: List[Dict[str, Any]], sib: Dict[str, Any], cluster_dir: Path, debug: bool, out_q: mp.Queue) -> None:
    """Subprocess runner to execute a single SIB with isolation.
    Puts (success, final_tool, processing_result) into out_q.
    """
    try:
        agent = ToolAggregationAgent(model_name="o4-mini", debug=debug)
        success, final_tool, processing_result = agent.process_single_sib(cluster_name, tools, sib, cluster_dir)
        out_q.put((success, final_tool, processing_result))
    except Exception as e:
        out_q.put((False, None, {"error": f"exception: {str(e)}"}))


def _sib_resume_worker(args: Tuple[str, Path, Dict[str, Any], List[Optional[Dict[str, Any]]], bool]) -> Dict[str, Any]:
    """Worker to resume a single SIB."""
    cluster_name, cluster_dir, sib, sparse_tools, debug = args

    # Ensure tools list contains at least needed covered indices
    covered = sib.get("covered_tools", []) or []
    needed_len = (max(covered) + 1) if covered else 0
    tools: List[Dict[str, Any]] = []
    if needed_len > 0:
        tools = [None] * needed_len  # type: ignore
        for idx in range(needed_len):
            tools[idx] = sparse_tools[idx] if idx < len(sparse_tools) and sparse_tools[idx] is not None else {}

    # Run the SIB processing in a subprocess with timeout protection
    timeout_seconds = int(os.getenv("SIB_TIMEOUT_SECONDS", "500"))
    out_q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_sib_runner_proc, args=(cluster_name, tools, sib, cluster_dir, debug, out_q))
    proc.daemon = True
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        try:
            proc.terminate()
        finally:
            proc.join(1)
        success = False
        final_tool = None
        processing_result = {"error": "timeout", "timeout_seconds": timeout_seconds}
    else:
        try:
            success, final_tool, processing_result = out_q.get_nowait()
        except Exception:
            success = False
            final_tool = None
            processing_result = {"error": "no_result_from_subprocess"}
    return {
        "cluster": cluster_name,
        "sib_index": sib.get("index", -1),
        "success": bool(success),
        "processing_result": processing_result,
        "has_tool": final_tool is not None,
    }


def resume_from_root(root_dir: Path, max_threads: int = 10, debug: bool = False, only_indices: Optional[List[int]] = None, dry_run: bool = False) -> Dict[str, Any]:
    """Resume processing across all cluster output directories under root_dir."""
    cluster_dirs = find_cluster_output_dirs(root_dir)
    if not cluster_dirs:
        raise RuntimeError(f"No cluster output directories with blueprint_sibs.json found under: {root_dir}")

    summary: Dict[str, Any] = {
        "root": str(root_dir),
        "timestamp": datetime.now().isoformat(),
        "clusters": [],
        "total_pending_sibs": 0,
        "executed_jobs": 0,
        "successful": 0,
        "failed": 0,
        "debug_mode": bool(debug),
        "debug_selection": None,
    }

    sib_jobs: List[Tuple[str, Path, Dict[str, Any], List[Optional[Dict[str, Any]]], bool]] = []

    for cluster_dir in cluster_dirs:
        cluster_name, sib_dicts, pending = list_pending_sibs(cluster_dir)
        if only_indices is not None:
            pending = [i for i in pending if i in set(only_indices)]

        # Prepare sparse tools lookup from blueprint
        blueprint = load_blueprint_sibs(cluster_dir)
        sparse_tools = build_sparse_tools_list_from_blueprint(blueprint)

        # Prepare jobs
        for sib in sib_dicts:
            if not isinstance(sib, dict):
                continue
            sib_index = sib.get("index", None)
            if not isinstance(sib_index, int):
                continue
            if sib_index in pending:
                sib_jobs.append((cluster_name, cluster_dir, sib, sparse_tools, debug))

        summary["clusters"].append({
            "cluster_name": cluster_name,
            "dir": str(cluster_dir),
            "total_sibs": len(sib_dicts),
            "pending_indices": pending,
        })
        summary["total_pending_sibs"] += len(pending)

    # Print flat pending list for visibility
    print("\nPENDING SIB TASKS (cluster :: sib_index :: dir):")
    for c in summary["clusters"]:
        for idx in c["pending_indices"]:
            print(f"- {c['cluster_name']} :: {idx} :: {c['dir']}")

    # In debug mode, restrict execution to one cluster and first three pending SIBs
    if debug and not dry_run and sib_jobs:
        # Choose first cluster with pending
        chosen_cluster: Optional[str] = None
        chosen_dir: Optional[str] = None
        chosen_indices: List[int] = []
        for c in summary["clusters"]:
            if c["pending_indices"]:
                chosen_cluster = c["cluster_name"]
                chosen_dir = c["dir"]
                chosen_indices = c["pending_indices"][:3]
                break
        if chosen_cluster is not None and chosen_indices:
            print("\n[DEBUG] Restricting run to:")
            print(f"Cluster: {chosen_cluster}")
            print(f"SIB indices: {chosen_indices}")
            sib_jobs = [j for j in sib_jobs if j[0] == chosen_cluster and j[2].get("index") in set(chosen_indices)]
            summary["debug_selection"] = {
                "cluster_name": chosen_cluster,
                "dir": chosen_dir,
                "sib_indices": chosen_indices,
            }

    if dry_run:
        return summary

    if sib_jobs:
        results = map_with_progress(
            _sib_resume_worker,
            sib_jobs,
            num_threads=min(max_threads, len(sib_jobs)),
            pbar=True
        )
        for r in results:
            if r.get("success") and r.get("has_tool"):
                summary["successful"] += 1
            else:
                summary["failed"] += 1
        summary["executed_jobs"] = len(results)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Resume SIB processing from existing blueprints")
    parser.add_argument("--root-output-dir", type=str, required=True, help="Path to root directory containing <cluster>_output folders or a single cluster output folder")
    parser.add_argument("--max-threads", type=int, default=50, help="Maximum parallel SIB jobs")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output for agents and limit to one cluster & 3 SIBs")
    parser.add_argument("--only-indices", type=str, default=None, help="Comma-separated SIB indices to resume (across all clusters)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="List pending SIBs without running")
    args = parser.parse_args()

    root_dir = Path(args.root_output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging under root
    log_file, _ = setup_logging(debug=args.debug, log_folder=str(root_dir / "logs"))

    only_indices: Optional[List[int]] = None
    if args.only_indices:
        try:
            only_indices = [int(x.strip()) for x in args.only_indices.split(",") if x.strip()]
        except Exception:
            raise ValueError("--only-indices must be a comma-separated list of integers")

    summary = resume_from_root(
        root_dir=root_dir,
        max_threads=args.max_threads,
        debug=args.debug,
        only_indices=only_indices,
        dry_run=args.dry_run
    )

    # Save summary
    out_path = root_dir / f"resume_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_json(summary, str(out_path))

    # Human-readable printout
    print("\n" + "=" * 60)
    print("RESUME SUMMARY")
    print("=" * 60)
    print(f"Root: {summary['root']}")
    print(f"Total pending SIBs: {summary['total_pending_sibs']}")
    if not args.dry_run:
        print(f"Executed jobs: {summary['executed_jobs']}")
        print(f"Successful: {summary['successful']}  Failed: {summary['failed']}")
    for c in summary["clusters"]:
        print(f"- {c['cluster_name']}: pending {c['pending_indices']}")
    if summary.get("debug_selection"):
        sel = summary["debug_selection"]
        print(f"\n[DEBUG] Selected cluster: {sel['cluster_name']}  sibs: {sel['sib_indices']}")
    print(f"\nLog file: {log_file}")


if __name__ == "__main__":
    main()
