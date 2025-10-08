import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make eval helpers importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eval')))

from eval_prompt import QUESTION_TEMPLATE_FLATTEN_RETRIEVAL  # noqa: E402
from utils import (  # noqa: E402
    call_sfr_embedding_api_lst,
    call_openai_with_temporary_tool,
    read_json,
    save_jsonl,
    map_with_progress,
)


# Global caches/registries to avoid recomputing per-thread call
_TOOLSET_CACHE: Dict[str, Dict[str, Any]] = {}
_FUNCTION_REGISTRY: Dict[str, Any] = {}


def register_tool_implementation(function_name: str, func: Any) -> None:
    """
    Register a python implementation for a tool by name.
    This enables real execution when the assistant calls the tool.
    """
    if not function_name or not callable(func):
        return
    _FUNCTION_REGISTRY[function_name] = func


def _hash_tools_openai(tools_openai: List[Dict[str, Any]]) -> str:
    # Stable hash based on sorted JSON string of names+descriptions
    try:
        minimal = [
            {
                "name": t.get("function", {}).get("name", ""),
                "description": t.get("function", {}).get("description", ""),
            }
            for t in tools_openai or []
        ]
        payload = json.dumps(minimal, sort_keys=True, ensure_ascii=False)
        import hashlib

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        # Fallback to object id to avoid crashing
        return str(id(tools_openai))


def _ensure_toolset_embeddings(tools_openai: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Build or load embeddings for an OpenAI-format tool list, with in-process caching.
    Returns (normalized_tool_matrix, tool_meta_list).
    tool_meta_list preserves order and includes name/description/parameters.
    """
    cache_key = _hash_tools_openai(tools_openai)
    if cache_key in _TOOLSET_CACHE:
        entry = _TOOLSET_CACHE[cache_key]
        return entry["emb_norm"], entry["tool_meta"]

    # Prepare embedding text for each tool
    tool_meta: List[Dict[str, Any]] = []
    embed_texts: List[str] = []
    for t in tools_openai or []:
        fn = t.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        tool_meta.append({
            "name": name,
            "description": desc,
            "parameters": params,
        })
        embed_texts.append(f"The function name is: {name}. The function description is: {desc}.")

    # Call embedding API
    emb_list = call_sfr_embedding_api_lst(embed_texts)
    if emb_list is None or len(emb_list) != len(embed_texts):
        # Fallback: zeros
        dim = 1024 if not emb_list else len(emb_list[0])
        emb = np.zeros((len(embed_texts), dim), dtype=np.float32)
    else:
        emb = np.array(emb_list, dtype=np.float32)

    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb_norm = emb / norms

    _TOOLSET_CACHE[cache_key] = {"emb_norm": emb_norm, "tool_meta": tool_meta}
    return emb_norm, tool_meta


def _retrieve_topk_tools(question: str, tools_openai: List[Dict[str, Any]], k: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve top-k tools for a question using cosine similarity between query embedding and tool embeddings.
    Returns (topk_tools_openai, topk_details).
    """
    emb_norm, tool_meta = _ensure_toolset_embeddings(tools_openai)

    # Query embedding
    q_emb_list = call_sfr_embedding_api_lst([question], is_query=True)
    if not q_emb_list:
        # If embedding failed, return first k as a fallback
        selected_indices = list(range(min(k, len(tool_meta))))
        details = []
        for rank, idx in enumerate(selected_indices, start=1):
            tm = tool_meta[idx]
            details.append({
                "name": tm["name"],
                "description": tm["description"],
                "parameters": tm["parameters"],
                "rank": rank,
                "relevance_score": 0.0,
                "format": "openai",
            })
        return [tools_openai[i] for i in selected_indices], details

    q_emb = np.array(q_emb_list[0], dtype=np.float32)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # Cosine scores
    scores = emb_norm @ q_emb
    top_indices = np.argsort(-scores)[: min(k, len(scores))]

    details = []
    for rank, idx in enumerate(top_indices.tolist(), start=1):
        tm = tool_meta[idx]
        details.append({
            "name": tm["name"],
            "description": tm["description"],
            "parameters": tm["parameters"],
            "rank": rank,
            "relevance_score": float(scores[idx]),
            "format": "openai",
        })

    return [tools_openai[i] for i in top_indices.tolist()], details


def _extract_pred_answer_from_messages(messages: List[Dict[str, Any]]) -> str:
    # Find last assistant message and parse after "Final Answer:"
    last_assistant_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant_content = msg.get("content") or ""
            break
    if not last_assistant_content:
        return ""
    if "Final Answer:" not in last_assistant_content:
        return ""
    pred = last_assistant_content.split("Final Answer:")[-1].strip()
    # Normalize common trailing punctuation/whitespace
    pred = pred.strip().strip(". ")
    # Only keep the first token/char for choice formats
    if len(pred.split()) > 1:
        pred = pred.split()[0]
    return pred


def _is_correct(pred_answer: str, gt_answer: str) -> bool:
    if not pred_answer or not gt_answer:
        return False
    p = str(pred_answer).strip().lower()
    g = str(gt_answer).strip().lower()
    # Accept simple letter/number/text equality
    return p == g


def generate_sft_record_for_question(
    question: str,
    gt_answer: str,
    tools_openai: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Build a full messages transcript for SFT using gpt-5 with OpenAI-format tools.
    Steps:
    - Retrieve top-5 relevant tools via embeddings
    - Run multi-turn chat with tool execution where possible
    - Parse final answer and filter by ground truth; return None if not correct

    Returns:
        dict with fields: question, ground_truth, messages, retrieved_tools_topk,
        retrieved_tools_details, pred_answer, correctness, model_name
        or None if filtered out.
    """
    if not question or not isinstance(tools_openai, list):
        return None

    # Retrieve tools
    topk_tools, topk_details = _retrieve_topk_tools(question, tools_openai, k=5)

    # Prepare prompt and run chat with tools
    user_prompt = QUESTION_TEMPLATE_FLATTEN_RETRIEVAL.format(question=question)
    messages = [{"role": "user", "content": user_prompt}]

    # Build function registry subset for the selected tools
    registry_subset = {}
    for t in topk_tools:
        fn = t.get("function", {})
        name = fn.get("name")
        if name in _FUNCTION_REGISTRY:
            registry_subset[name] = _FUNCTION_REGISTRY[name]

    final_messages, _turns = call_openai_with_temporary_tool(
        messages=messages,
        tools=topk_tools,
        function_registry=registry_subset,
        model_name="gpt-5",
        max_turns=10,
        completion_check=lambda content: "Final Answer:" in (content or ""),
        max_turns_prompt="You have reached the maximum number of turns. Please provide the final answer now. Your last line must start with 'Final Answer:'.",
    )

    # Parse and filter
    pred_answer = _extract_pred_answer_from_messages(final_messages)
    correct = 1 if _is_correct(pred_answer, gt_answer) else 0
    if correct != 1:
        return None

    record = {
        "question": question,
        "ground_truth": gt_answer,
        "messages": final_messages,
        "retrieved_tools_topk": [d["name"] for d in topk_details],
        "retrieved_tools_details": topk_details,
        "pred_answer": pred_answer,
        "correctness": correct,
        "model_name": "gpt-5",
    }
    return record


def _load_tools_from_arg_or_env(tools_path: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load OpenAI-format tools from a JSON file path or from PLACEHOLDER_B env var.
    If PLACEHOLDER_B is set and not a path, try to parse it as JSON string.
    """
    # 1) If explicit path is provided
    if tools_path:
        if not os.path.exists(tools_path):
            raise FileNotFoundError(f"Tools path not found: {tools_path}")
        with open(tools_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2) Else check env var PLACEHOLDER_B
    env_val = os.getenv("PLACEHOLDER_B")
    if not env_val:
        raise ValueError("No tools provided. Pass --tools_path or set PLACEHOLDER_B (path or JSON string).")

    # If it's a path
    if os.path.exists(env_val):
        with open(env_val, "r", encoding="utf-8") as f:
            return json.load(f)

    # Try to parse as JSON array string
    try:
        tools = json.loads(env_val)
        if isinstance(tools, list):
            return tools
    except json.JSONDecodeError:
        pass

    raise ValueError("PLACEHOLDER_B must be a valid file path or a JSON array string of OpenAI tools.")


def _worker_build_record(item_and_tools: Tuple[Dict[str, Any], List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    item, tools_openai = item_and_tools
    q = item.get("question") or item.get("query") or ""
    gt = item.get("ground_truth") or item.get("answer") or ""
    return generate_sft_record_for_question(q, gt, tools_openai)


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description="Generate tool-augmented SFT JSONL using gpt-5")
    parser.add_argument(
        "--input_questions_path",
        type=str,
        default="/Users/murong.yue/Desktop/temp_lib/phy_lib_20250806_141249/c_kinematics_unique_questions.json",
        help="Path to input questions JSON",
    )
    parser.add_argument(
        "--tools_path",
        type=str,
        default="/Users/murong.yue/Desktop/temp_lib/phy_lib_20250816_222808/c_kinematics_v3_final_openai_tools.json",
        help="Path to OpenAI-format tools JSON (if omitted, will use PLACEHOLDER_B env)",
    )
    parser.add_argument(
        "--output_jsonl_path",
        type=str,
        default=None,
        help="Where to save the filtered SFT JSONL. If omitted, a timestamped file will be created next to input.",
    )
    parser.add_argument("--threads", type=int, default=50)

    args = parser.parse_args()

    # Load inputs
    data = read_json(args.input_questions_path) or []
    if not isinstance(data, list):
        raise ValueError("Input questions JSON must be a list of objects.")

    tools_openai = _load_tools_from_arg_or_env(args.tools_path)

    # Prepare output path
    if args.output_jsonl_path:
        out_path = args.output_jsonl_path
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(args.input_questions_path)
        out_path = os.path.join(base_dir, f"tool_aug_sft_{ts}.jsonl")

    # Build in parallel
    def fn(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return _worker_build_record((item, tools_openai))

    results = map_with_progress(fn, data, num_threads=args.threads, pbar=True)
    filtered = [r for r in results if r]

    # Save JSONL
    save_jsonl(filtered, out_path)
    print(f"Saved {len(filtered)} SFT records to {out_path}")


