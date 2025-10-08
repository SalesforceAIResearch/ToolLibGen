import json
import random
from typing import List, Dict, Any, Optional
from utils import (
    call_openai_api,
    read_jsonl,
    save_jsonl,
    map_with_progress,
    save_json,
    read_json
)
from prompt import (
    CLUSTERING_INITIAL_PROMPT,
    CLUSTERING_UPDATE_OPERATIONS_PROMPT,
    TOOL_ASSIGNMENT_PROMPT,
    JSON_REPAIR_PROMPT
)
import argparse
from datetime import datetime
import os
from pathlib import Path


def extract_tools_from_jsonl(input_path: str, output_path: str) -> List[Dict]:
    """
    Extracts all tools from a JSON/JSONL file and optionally saves them to JSONL.
    """
    print(f"Reading data from {input_path}...")
    if input_path.endswith(".jsonl"):
        raw_data = read_jsonl(input_path)
    elif input_path.endswith(".json"):
        raw_data = read_json(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path}")

    all_tools: List[Dict[str, Any]] = []
    for entry in raw_data:
        try:
            if not isinstance(entry, dict):
                continue
            agent_result_raw = entry.get("agent_result")
            if isinstance(agent_result_raw, str):
                try:
                    agent_result = json.loads(agent_result_raw)
                except json.JSONDecodeError:
                    continue
            elif isinstance(agent_result_raw, dict):
                agent_result = agent_result_raw
            elif isinstance(agent_result_raw, list):
                agent_result = {"final_tools": agent_result_raw}
            else:
                agent_result = {}

            final_tools = agent_result.get("final_tools", [])
            if not isinstance(final_tools, list):
                continue

            original_question = entry.get("question", "")
            original_answer = entry.get("answer", "")

            for tool in final_tools:
                if not isinstance(tool, dict):
                    continue
                tool_info = tool.get("tool_info", {})
                if not isinstance(tool_info, dict):
                    continue
                tool_code = tool.get("code", "")
                processed_tool: Dict[str, Any] = {"description": tool_info}
                if tool_code:
                    processed_tool["python_code"] = tool_code
                if original_question:
                    processed_tool["original_question"] = original_question
                if original_answer:
                    processed_tool["original_answer"] = original_answer
                all_tools.append(processed_tool)
        except Exception:
            continue

    print(f"Extracted a total of {len(all_tools)} tools.")
    if output_path:
        print(f"Saving extracted tools to {output_path}...")
        save_jsonl(all_tools, output_path)
        print("Tools saved successfully.")
    return all_tools


class ClusteringAgent:
    """
    A clustering agent that can initialize and incrementally update hierarchical tool clusters.
    """
    
    def __init__(self, model_name: str = "o4-mini", log_dir: Optional[str] = None):
        self.model_name = model_name
        self.current_hierarchy: Optional[Dict[str, Any]] = None
        self.log_dir = log_dir

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"LLM interactions will be logged to: {self.log_dir}")
        
        self.INITIAL_PROMPT = CLUSTERING_INITIAL_PROMPT
        self.UPDATE_PROMPT = CLUSTERING_UPDATE_OPERATIONS_PROMPT
        self.ASSIGNMENT_PROMPT = TOOL_ASSIGNMENT_PROMPT
        self.REPAIR_PROMPT = JSON_REPAIR_PROMPT

    def convert_dict_to_str_with_only_name_and_description(self, tool_dict: dict) -> str:
        try:
            filtered_dict = {
                "name": tool_dict["description"]["function"]["name"],
                "tags": tool_dict.get("tag", "No tag provided"),
                "description": tool_dict["description"]["function"]["description"],
            }
            return json.dumps(filtered_dict)
        except (KeyError, TypeError, AttributeError):
            name = "Unknown"
            tags = tool_dict.get("tag", "unknown") if isinstance(tool_dict, dict) else "unknown"
            desc_text = "Error processing tool description"
            try:
                description_field = tool_dict["description"] if isinstance(tool_dict, dict) else {}
                if isinstance(description_field, dict):
                    name = description_field.get("function", {}).get("name", name)
                    desc_text = description_field.get("function", {}).get("description", desc_text)
                elif isinstance(description_field, list) and description_field:
                    first_item = description_field[0]
                    if isinstance(first_item, dict):
                        name = first_item.get("function", {}).get("name", name)
                        desc_text = first_item.get("function", {}).get("description", desc_text)
            except Exception:
                pass
            return json.dumps({"name": name, "tags": tags, "description": desc_text})

    def _log_interaction(self, interaction_name: str, prompt: str, response: str):
        if not self.log_dir:
            return
        now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = os.path.join(self.log_dir, f"{interaction_name}_{now}.log")
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"--- PROMPT ---\n{prompt}\n\n--- RESPONSE ---\n{response}")
        except Exception:
            pass

    def repair_json_response(self, response: str, repair_type: str = "assignment", max_retries: int = 3, error_message: str = "") -> Optional[str]:
        for attempt in range(max_retries):
            try:
                repair_prompt = self.REPAIR_PROMPT.format(response=response, error_message=error_message)
                repaired_response = call_openai_api(content=repair_prompt, model_name="gpt-4o-mini")
                json.loads(repaired_response)
                return repaired_response
            except Exception as e:
                if attempt < max_retries - 1:
                    response = repaired_response if 'repaired_response' in locals() else response
                    error_message = str(e)
        return None

    def initialize_clustering(self, tools: List[Dict[str, Any]], initial_batch_size: int = 500, cluster_depth: int = 6) -> Dict[str, Any]:
        shuffled_tools = tools.copy()
        random.shuffle(shuffled_tools)
        initial_tools = shuffled_tools[:min(initial_batch_size, len(shuffled_tools))]
        tool_strings = [self.convert_dict_to_str_with_only_name_and_description(tool) for tool in initial_tools]
        prompt = self.INITIAL_PROMPT.format(tool_lst="\n".join(tool_strings), cluster_depth=cluster_depth)
        response = call_openai_api(content=prompt, model_name=self.model_name)
        self._log_interaction("initialize_clustering", prompt, response)
        try:
            hierarchy = json.loads(response)
            self.current_hierarchy = hierarchy
            return hierarchy
        except json.JSONDecodeError as e:
            repaired = self.repair_json_response(response, "hierarchy", error_message=str(e))
            if repaired:
                try:
                    hierarchy = json.loads(repaired)
                    self.current_hierarchy = hierarchy
                    return hierarchy
                except json.JSONDecodeError:
                    pass
            raise ValueError("Failed to parse initial clustering response after repair attempts.")

    def update_clustering_with_operations(self, new_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.current_hierarchy is None:
            raise ValueError("No existing hierarchy. Initialize first.")

        normalized_tools: List[Dict[str, Any]] = []
        for idx, tool in enumerate(new_tools):
            if not isinstance(tool, dict):
                continue
            desc = tool.get("description")
            if isinstance(desc, list):
                first_valid = next((d for d in desc if isinstance(d, dict)), None)
                if first_valid is not None:
                    tool = {**tool, "description": first_valid}
                else:
                    continue
            normalized_tools.append(tool)

        if not normalized_tools:
            return self.current_hierarchy

        tool_strings = [self.convert_dict_to_str_with_only_name_and_description(tool) for tool in normalized_tools]
        prompt = self.UPDATE_PROMPT.format(
            tool_lst="\n".join(tool_strings),
            current_hierarchy=json.dumps(self.current_hierarchy, indent=2)
        )
        response = call_openai_api(content=prompt, model_name=self.model_name)
        self._log_interaction("update_clustering", prompt, response)
        try:
            operations_result = json.loads(response)
            if isinstance(operations_result, dict):
                operations = operations_result.get('operations', [])
            elif isinstance(operations_result, list):
                operations = operations_result
            else:
                operations = []
            if isinstance(operations, list):
                flat_ops = []
                for item in operations:
                    if isinstance(item, list):
                        flat_ops.extend([x for x in item if isinstance(x, dict)])
                    elif isinstance(item, dict):
                        flat_ops.append(item)
                operations = flat_ops
            if operations:
                updated = self.apply_hierarchy_operations(operations)
                self.current_hierarchy = updated
                return updated
            return self.current_hierarchy
        except json.JSONDecodeError as e:
            repaired = self.repair_json_response(response, "operations", error_message=str(e))
            if repaired:
                try:
                    operations_result = json.loads(repaired)
                    if isinstance(operations_result, dict):
                        operations = operations_result.get('operations', [])
                    elif isinstance(operations_result, list):
                        operations = operations_result
                    else:
                        operations = []
                    if isinstance(operations, list):
                        flat_ops = []
                        for item in operations:
                            if isinstance(item, list):
                                flat_ops.extend([x for x in item if isinstance(x, dict)])
                            elif isinstance(item, dict):
                                flat_ops.append(item)
                        operations = flat_ops
                    if operations:
                        updated = self.apply_hierarchy_operations(operations)
                        self.current_hierarchy = updated
                        return updated
                except json.JSONDecodeError:
                    pass
            return self.current_hierarchy

    def apply_hierarchy_operations(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        updated_hierarchy = json.loads(json.dumps(self.current_hierarchy))
        clusters = updated_hierarchy.get('clusters', [])
        cluster_map = {cluster['id']: cluster for cluster in clusters if isinstance(cluster, dict) and 'id' in cluster}
        for op in operations:
            if not isinstance(op, dict):
                continue
            action = op.get('action')
            if action == 'ADD_NODE':
                new_node = {'id': op['node_id'], 'level': op['level'], 'parent': op['parent'], 'children': []}
                clusters.append(new_node)
                cluster_map[new_node['id']] = new_node
                if op['parent'] in cluster_map:
                    cluster_map[op['parent']]['children'].append(new_node['id'])
            elif action == 'MODIFY_NODE':
                node_id = op.get('node_id')
                if node_id in cluster_map:
                    node = cluster_map[node_id]
                    if 'add_children' in op.get('changes', {}):
                        for child_id in op['changes']['add_children']:
                            if child_id not in node['children']:
                                node['children'].append(child_id)
        return updated_hierarchy

    def incremental_clustering(self, tools: List[Dict[str, Any]], initial_batch_size: int, update_step_size: int, cluster_depth: int, hierarchy_path: str, max_iterations: int) -> Optional[Dict[str, Any]]:
        print(f"Initializing clustering with {min(initial_batch_size, len(tools))} tools...")
        try:
            self.initialize_clustering(tools, initial_batch_size, cluster_depth)
        except Exception as e:
            print(f"Clustering initialization failed: {e}. Aborting.")
            return None
        
        if self.current_hierarchy:
            self.save_hierarchy(hierarchy_path)
        else:
            print("Clustering initialization did not return a hierarchy. Aborting.")
            return None
        
        remaining_tools = tools[initial_batch_size:]
        index = 0
        for i in range(0, len(remaining_tools), update_step_size):
            if index >= max_iterations:
                break
            batch = remaining_tools[i:i + update_step_size]
            print(f"Updating clustering with {len(batch)} new tools...")
            try:
                self.update_clustering_with_operations(batch)
                self.save_hierarchy(hierarchy_path)
            except Exception as e:
                print(f"Failed to update clustering for a batch, skipping. Error: {e}")
                continue
            index += 1
        return self.current_hierarchy

    def _get_leaf_cluster_ids(self) -> List[str]:
        if not self.current_hierarchy or not isinstance(self.current_hierarchy, dict):
            return []
        clusters = self.current_hierarchy.get('clusters', [])
        if not isinstance(clusters, list):
            return []
        leaf_ids: List[str] = []
        for c in clusters:
            if not isinstance(c, dict):
                continue
            children = c.get('children', [])
            if not children:
                cid = c.get('id')
                if isinstance(cid, str):
                    leaf_ids.append(cid)
        return leaf_ids

    def assign_tools_to_clusters(self, tools: List[Dict[str, Any]], assignment_model: str, batch_size: int = 50) -> List[Dict[str, Any]]:
        if self.current_hierarchy is None:
            raise ValueError("No hierarchy available.")

        leaf_ids = self._get_leaf_cluster_ids()
        tags_text = ", ".join(leaf_ids)

        def process_single(args):
            idx, tool = args
            if not isinstance(tool, dict):
                return tool
            desc = tool.get('description')
            if isinstance(desc, list):
                first_valid = next((d for d in desc if isinstance(d, dict)), None)
                if first_valid is not None:
                    tool = {**tool, 'description': first_valid}
            tool_str = self.convert_dict_to_str_with_only_name_and_description(tool)
            prompt = self.ASSIGNMENT_PROMPT.format(hierarchy=tags_text, tools=tool_str)
            response = call_openai_api(content=prompt, model_name=assignment_model)
            self._log_interaction(f"assign_tool_single_{idx}", prompt, response)
            clusters = [x.strip() for x in response.split(',') if x.strip()]
            assignment = {
                'cluster_id': clusters[0] if clusters else None,
                'cluster_ids': clusters,
                'reasoning': ''
            }
            out_tool = tool.copy()
            out_tool['cluster_assignment'] = assignment
            return out_tool

        assigned_tools = list(map_with_progress(
            process_single,
            list(enumerate(tools)),
            num_threads=50,
            pbar=True
        ))

        for tool in assigned_tools:
            if isinstance(tool, dict) and 'cluster_assignment' not in tool:
                tool['cluster_assignment'] = {'cluster_id': None, 'cluster_ids': [], 'reasoning': 'Tool was not assigned by the API.'}

        return assigned_tools

    def save_hierarchy(self, file_path: str):
        if self.current_hierarchy:
            save_json(self.current_hierarchy, file_path)
            print(f"Hierarchy saved to {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Clustering of Tools (final)")
    parser.add_argument("--input_file", type=str, default="/Users/murong.yue/Desktop/data/ReasonMed_tools_100k_20250915_091638.json")
    parser.add_argument("--output_dir", type=str, default="/Users/murong.yue/Desktop/data")
    parser.add_argument("--log_dir", type=str, default="/Users/murong.yue/Desktop/log")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--remote_mode", action='store_true', default=False)
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir
    log_dir = args.log_dir
    if args.remote_mode:
        input_file = "/export/home/data/ReasonMed_tools_100k_20250915_091638.json"
        output_dir = "/export/home/data"
        log_dir = "/export/home/log"

    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_file).stem
    extracted_tools_path = os.path.join(output_dir, f"{base_name}_extracted_tools_{now}.jsonl")

    print("--- Step 1: Extracting Tools ---")
    all_tools = extract_tools_from_jsonl(input_file, extracted_tools_path)
    all_tools = all_tools[:len(all_tools)//3]
    if not all_tools:
        print("No tools were extracted. Exiting.")
        return
    if args.debug:
        print("Debug mode enabled: using a subset of 100 tools for clustering.")
        all_tools = all_tools[:100]

    print("\n--- Step 2: Clustering Tools ---")
    hierarchy_path = os.path.join(output_dir, f"{base_name}_hierarchy_{now}.json")
    assigned_tools_path = os.path.join(output_dir, f"{base_name}_assigned_tools_{now}.jsonl")

    if args.debug:
        hierarchy_path = os.path.join(output_dir, f"{base_name}_hierarchy_debug_{now}.json")
        assigned_tools_path = os.path.join(output_dir, f"{base_name}_assigned_tools_debug_{now}.jsonl")
        initial_batch_size = 30
        update_step_size = 20
        cluster_depth = 3
        assignment_batch_size = 10
        max_iterations = 1
    else:
        initial_batch_size = 1000
        update_step_size = 200
        cluster_depth = 4
        assignment_batch_size = 50
        max_iterations = 1

    agent = ClusteringAgent(model_name="gpt-5", log_dir=log_dir)

    print("Building hierarchy...")
    _ = agent.incremental_clustering(
        tools=all_tools,
        initial_batch_size=initial_batch_size,
        update_step_size=update_step_size,
        cluster_depth=cluster_depth,
        hierarchy_path=hierarchy_path,
        max_iterations=max_iterations
    )

    print("\nAssigning tools to clusters...")
    assigned_tools = agent.assign_tools_to_clusters(
        tools=all_tools,
        assignment_model="gpt-4o-mini",
        batch_size=assignment_batch_size
    )
    save_jsonl(assigned_tools, assigned_tools_path)
    print(f"Assigned tools saved to {assigned_tools_path}")

    print("\nScript finished.")


if __name__ == "__main__":
    main()


