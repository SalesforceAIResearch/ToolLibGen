import json
import random
from typing import List, Dict, Any, Optional
import ast
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
    Extracts all tools from a JSONL file and saves them to another JSONL file.

    Each line in the input file is a JSON object containing an 'agent_result' key.
    The 'agent_result' contains 'final_tools'. We extract each tool from 'final_tools'.
    """
    print(f"Reading data from {input_path}...")
    if input_path.endswith(".jsonl"):
        raw_data = read_jsonl(input_path)
    elif input_path.endswith(".json"):
        raw_data = read_json(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path}")
    all_tools = []

    for entry in raw_data:
        try:
            agent_result_raw = entry.get("agent_result")
            if isinstance(agent_result_raw, str):
                agent_result = json.loads(agent_result_raw)
            elif isinstance(agent_result_raw, dict):
                agent_result = agent_result_raw  # It's already a dict
            else:
                agent_result = {}

            final_tools = agent_result.get("final_tools", [])
            original_question = entry.get("question", "")
            original_answer = entry.get("answer", "")

            for tool in final_tools:
                tool_info = tool.get("tool_info", {})
                tool_code = tool.get("code", "")
                # Ensure description is a dictionary, not a string
                # if isinstance(tool_info.get("description"), str):
                #     try:
                #         # Use ast.literal_eval to safely parse Python dict-like strings
                #         tool_info["description"] = ast.literal_eval(tool_info["description"])
                #     except (ValueError, SyntaxError) as e:
                #         print(f"Warning: Could not parse description string for tool: {tool_info.get('name')}. Error: {e}")
                #         # If parsing fails, keep it as a string under a different key
                #         tool_info["description_str"] = tool_info.pop("description")


                processed_tool = {
                    "description": tool_info,
                }

                if tool_code:
                    processed_tool["python_code"] = tool_code
                if original_question:
                    processed_tool["original_question"] = original_question
                if original_answer:
                    processed_tool["original_answer"] = original_answer

                all_tools.append(processed_tool)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Skipping an entry due to parsing error: {e}")
    
    print(f"Extracted a total of {len(all_tools)} tools.")
    
    # Save the extracted tools to the output file
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
        self.current_hierarchy = None
        self.log_dir = log_dir

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"LLM interactions will be logged to: {self.log_dir}")
        
        # Prompt templates
        self.INITIAL_PROMPT = CLUSTERING_INITIAL_PROMPT
        self.UPDATE_PROMPT = CLUSTERING_UPDATE_OPERATIONS_PROMPT
        self.ASSIGNMENT_PROMPT = TOOL_ASSIGNMENT_PROMPT
        self.REPAIR_PROMPT = JSON_REPAIR_PROMPT

    def convert_dict_to_str_with_only_name_and_description(self, tool_dict: dict) -> str:
        try:
            # The tool structure from extraction is already quite flat
            filtered_dict = {
                "name": tool_dict["description"]["function"]["name"], 
                "tags": tool_dict.get("tag", "No tag provided"), 
                "description": tool_dict["description"]["function"]["description"]
            }
            return json.dumps(filtered_dict)
        except (KeyError, TypeError, AttributeError) as e:
            print(f"Error processing tool structure: {e}")
            # Robust fallback handling for various unexpected shapes
            name = "Unknown"
            tags = "unknown"
            desc_text = "Error processing tool description"

            # Tags
            try:
                if isinstance(tool_dict, dict):
                    tags = tool_dict.get("tag", tags)
            except Exception:
                pass

            # Description can be dict, list, or something else
            try:
                description_field = tool_dict["description"] if isinstance(tool_dict, dict) else {}
                if isinstance(description_field, dict):
                    name = description_field.get("function", {}).get("name", name)
                    desc_text = description_field.get("function", {}).get("description", desc_text)
                elif isinstance(description_field, list) and len(description_field) > 0:
                    first_item = description_field[0]
                    if isinstance(first_item, dict):
                        name = first_item.get("function", {}).get("name", name)
                        desc_text = first_item.get("function", {}).get("description", desc_text)
            except Exception:
                pass

            fallback_dict = {
                "name": name,
                "tags": tags,
                "description": desc_text
            }
            return json.dumps(fallback_dict)

    def get_tool_name(self, tool_dict: dict) -> str:
        try:
            return tool_dict["description"]["function"]["name"]
        except (KeyError, TypeError):
            if isinstance(tool_dict, dict):
                return tool_dict.get('name', 'Unknown')
            return 'Unknown'
            
    def _log_interaction(self, interaction_name: str, prompt: str, response: str):
        if not self.log_dir:
            return
        
        now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = os.path.join(self.log_dir, f"{interaction_name}_{now}.log")
        
        log_content = f"--- PROMPT ---\n{prompt}\n\n--- RESPONSE ---\n{response}"
        
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
        except Exception as e:
            print(f"Warning: Failed to write to log file {log_file_path}. Error: {e}")
            
    def repair_json_response(self, response: str, repair_type: str = "assignment", max_retries: int = 3, error_message: str = "") -> str:
        for attempt in range(max_retries):
            try:
                print(f"Attempting to repair malformed JSON (type: {repair_type}, attempt: {attempt + 1}/{max_retries})...")
                repair_prompt = self.REPAIR_PROMPT.format(response=response, error_message=error_message)
                repaired_response = call_openai_api(content=repair_prompt, model_name="gpt-4o-mini")
                
                # Log the repair interaction
                self._log_interaction(f"repair_json_{repair_type}_attempt_{attempt + 1}", repair_prompt, repaired_response)

                json.loads(repaired_response)
                print(f"JSON repair successful.")
                return repaired_response
            except Exception as e:
                print(f"JSON repair attempt {attempt + 1} failed: {e}")
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
            print(f"JSON parsing failed in initialize_clustering: {e}")
            repaired_response = self.repair_json_response(response, "hierarchy", error_message=str(e))
            if repaired_response:
                try:
                    hierarchy = json.loads(repaired_response)
                    self.current_hierarchy = hierarchy
                    return hierarchy
                except json.JSONDecodeError:
                    print("Repaired JSON still not parseable.")
            raise ValueError("Failed to parse initial clustering response after repair attempts.")

    def update_clustering_with_operations(self, new_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.current_hierarchy is None:
            raise ValueError("No existing hierarchy. Initialize first.")
        
        # Filter out clearly invalid tool entries and normalize structures
        normalized_tools: List[Dict[str, Any]] = []
        for idx, tool in enumerate(new_tools):
            if not isinstance(tool, dict):
                print(f"Warning: Skipping tool at index {idx} due to invalid type: {type(tool)}")
                continue
            # Ensure 'description' is usable (dict). If it's a list, take first dict item.
            desc = tool.get("description")
            if isinstance(desc, list):
                first_valid = next((d for d in desc if isinstance(d, dict)), None)
                if first_valid is not None:
                    tool = {**tool, "description": first_valid}
                else:
                    print(f"Warning: Skipping tool at index {idx} due to list description without dict items")
                    continue
            normalized_tools.append(tool)

        if not normalized_tools:
            print("No valid tools to update from; returning current hierarchy.")
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
            
            # Handle cases where the API returns a list directly or a dict with an 'operations' key
            if isinstance(operations_result, dict):
                operations = operations_result.get('operations', [])
            elif isinstance(operations_result, list):
                operations = operations_result
            else:
                print(f"Warning: Unexpected type for operations_result: {type(operations_result)}. Assuming no operations.")
                operations = []

            # Normalize operations: flatten one level and keep only dicts
            if isinstance(operations, list):
                flat_ops = []
                for item in operations:
                    if isinstance(item, list):
                        flat_ops.extend([x for x in item if isinstance(x, dict)])
                    elif isinstance(item, dict):
                        flat_ops.append(item)
                operations = flat_ops

            if operations:
                updated_hierarchy = self.apply_hierarchy_operations(operations)
                self.current_hierarchy = updated_hierarchy
                print(f"Applied {len(operations)} operations to update hierarchy.")
                return updated_hierarchy
            else:
                print("No operations needed.")
                return self.current_hierarchy
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed in update_clustering: {e}")
            repaired_response = self.repair_json_response(response, "operations", error_message=str(e))
            if repaired_response:
                try:
                    operations_result = json.loads(repaired_response)
                    
                    if isinstance(operations_result, dict):
                        operations = operations_result.get('operations', [])
                    elif isinstance(operations_result, list):
                        operations = operations_result
                    else:
                        print(f"Warning: Unexpected type for repaired operations_result: {type(operations_result)}. Assuming no operations.")
                        operations = []

                    # Normalize operations again after repair
                    if isinstance(operations, list):
                        flat_ops = []
                        for item in operations:
                            if isinstance(item, list):
                                flat_ops.extend([x for x in item if isinstance(x, dict)])
                            elif isinstance(item, dict):
                                flat_ops.append(item)
                        operations = flat_ops

                    if operations:
                        updated_hierarchy = self.apply_hierarchy_operations(operations)
                        self.current_hierarchy = updated_hierarchy
                        return updated_hierarchy
                except json.JSONDecodeError:
                    print("Repaired JSON still not parseable after update.")
            
            print("Skipping this update round due to persistent JSON parsing failures.")
            return self.current_hierarchy

    def apply_hierarchy_operations(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        updated_hierarchy = json.loads(json.dumps(self.current_hierarchy))
        clusters = updated_hierarchy.get('clusters', [])
        cluster_map = {cluster['id']: cluster for cluster in clusters}
        
        for op in operations:
            action = op.get('action')
            if action == 'ADD_NODE':
                new_node = {'id': op['node_id'], 'level': op['level'], 'parent': op['parent'], 'children': []}
                clusters.append(new_node)
                cluster_map[new_node['id']] = new_node
                if op['parent'] in cluster_map:
                    cluster_map[op['parent']]['children'].append(new_node['id'])
            elif action == 'MODIFY_NODE':
                if op['node_id'] in cluster_map:
                    node = cluster_map[op['node_id']]
                    if 'add_children' in op.get('changes', {}):
                        for child_id in op['changes']['add_children']:
                            if child_id not in node['children']:
                                node['children'].append(child_id)
        return updated_hierarchy

    def incremental_clustering(self, tools: List[Dict[str, Any]], initial_batch_size: int, update_step_size: int, cluster_depth: int, hierarchy_path: str, max_iterations: int) -> Dict[str, Any]:
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

    def assign_tools_to_clusters(self, tools: List[Dict[str, Any]], assignment_model: str, batch_size: int = 50, save_interval: int = 1000, output_path: str = None) -> List[Dict[str, Any]]:
        if self.current_hierarchy is None:
            raise ValueError("No hierarchy available.")
        
      #   print(f"Assigning {len(tools)} tools to clusters using {assignment_model} with batch size {batch_size}...")
        
        # Split tools into batches for processing
        tool_batches = [tools[i:i + batch_size] for i in range(0, len(tools), batch_size)]
        
        # Define the function to be mapped
        def process_batch(batch):
            try:
                return self._process_tool_batch(batch, assignment_model)
            except Exception as e:
                print(f"Error processing batch: {e}. Skipping this batch.")
                return None  # Return None to indicate this batch should be skipped
            
        # Process batches with periodic saving
        assigned_tools = []
        processed_count = 0
        
        for i in range(0, len(tool_batches), save_interval):
            # Process a chunk of batches (up to save_interval batches)
            current_chunk = tool_batches[i:i + save_interval]
            
            print(f"Processing batches {i+1} to {min(i + save_interval, len(tool_batches))} of {len(tool_batches)}...")
            
            # Use map_with_progress for parallel processing of current chunk
            processed_batches = map_with_progress(
                process_batch,
                current_chunk,
                num_threads=50,
                pbar=True
            )
            
            # Flatten the current chunk results and add to assigned_tools
            chunk_tools = [tool for batch in processed_batches if batch is not None for tool in batch]
            assigned_tools.extend(chunk_tools)
            
            processed_count += len(current_chunk)
            
            # Save intermediate results if output_path is provided
            if output_path and processed_count % save_interval == 0:
                # Ensure every tool has an assignment key before saving
                for tool in assigned_tools:
                    if 'cluster_assignment' not in tool:
                        tool['cluster_assignment'] = {'cluster_id': None, 'reasoning': 'Tool was not assigned by the API.'}
                
                # Create intermediate save path
                base_path, ext = os.path.splitext(output_path)
                intermediate_path = f"{base_path}_intermediate_{processed_count}_batches{ext}"
                
                from utils import save_jsonl
                save_jsonl(assigned_tools, intermediate_path)
                print(f"Intermediate results saved to {intermediate_path} (processed {processed_count} batches, {len(assigned_tools)} tools)")
        
        # Ensure every tool has an assignment key, even if the API failed for some.
        for tool in assigned_tools:
            if 'cluster_assignment' not in tool:
                tool['cluster_assignment'] = {'cluster_id': None, 'reasoning': 'Tool was not assigned by the API.'}
                
        return assigned_tools

    def _process_tool_batch(self, batch: List[Dict[str, Any]], assignment_model: str) -> List[Dict[str, Any]]:
        tool_strings = [f"Tool {i}: {self.convert_dict_to_str_with_only_name_and_description(tool)}" for i, tool in enumerate(batch)]
        prompt = self.ASSIGNMENT_PROMPT.format(
            hierarchy=json.dumps(self.current_hierarchy, indent=2),
            tools="\n".join(tool_strings)
        )
        response = call_openai_api(content=prompt, model_name=assignment_model)
        self._log_interaction(f"assign_tools_batch", prompt, response)
        
        try:
            assignment_result = json.loads(response)
        except json.JSONDecodeError as e:
            repaired_response = self.repair_json_response(response, "assignment", error_message=str(e))
            if repaired_response:
                try:
                    assignment_result = json.loads(repaired_response)
                except json.JSONDecodeError:
                    assignment_result = {"assignments": []}
            else:
                assignment_result = {"assignments": []}

        assignments = assignment_result.get('assignments', [])
        
        # Add assignment info back to the tools in the batch
        batch_with_assignments = batch[:]
        for assignment in assignments:
            tool_index = assignment.get('tool_index')
            if tool_index is not None and 0 <= tool_index < len(batch_with_assignments):
                batch_with_assignments[tool_index]['cluster_assignment'] = {
                    'cluster_id': assignment.get('assigned_cluster_id'),
                    'reasoning': assignment.get('reasoning', '')
                }
        return batch_with_assignments

    def save_hierarchy(self, file_path: str):
        if self.current_hierarchy:
            save_json(self.current_hierarchy, file_path)
            print(f"Hierarchy saved to {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Clustering of Tools")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file (e.g., Nemotron_science_data_tools.jsonl)", default="/Users/murong.yue/Desktop/data/Nemotron_science_data_tools_20250723_071519.jsonl")
    parser.add_argument("--output_dir", type=str, default="/Users/murong.yue/Desktop/data", help="Directory to save the output files.")
    parser.add_argument("--log_dir", type=str, default="/Users/murong.yue/Desktop/log", help="Directory to save LLM interaction logs.")
    parser.add_argument("--debug", action='store_true', default=False, help="Enable debug mode (use smaller batches and subsets of data)")
    parser.add_argument("--remote_mode", action='store_true', default=False, help="Enable remote mode (use remote LLM API)")
    parser.add_argument("--initial_batch_size", type=int, default=1000, help="Initial batch size for clustering")
    args = parser.parse_args()
    if args.remote_mode:
         input_file = "/export/home/data/ReasonMed_tools_100k_20250915_091638.json"
         output_dir = "/export/home/data"
         log_dir = "/export/home/log"
    else:
         input_file = "/Users/murong.yue/Desktop/data/ReasonMed_tools_100k_20250915_091638.json"
         output_dir = "/Users/murong.yue/Desktop/data"
         log_dir = "/Users/murong.yue/Desktop/log"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define file paths
    base_name = Path(input_file).stem
    extracted_tools_path = os.path.join(output_dir, f"{base_name}_extracted_tools_{now}.jsonl")
    
    # --- Step 1: Extract all tools ---
    print("--- Step 1: Extracting Tools ---")
    all_tools = extract_tools_from_jsonl(input_file, extracted_tools_path)
    import random
    random.shuffle(all_tools)
    all_tools = all_tools[:len(all_tools)]
    if not all_tools:
        print("No tools were extracted. Exiting.")
        return

    if args.debug:
        print("Debug mode enabled: using a subset of 100 tools for clustering.")
        all_tools = all_tools[:100]

    # --- Step 2: Perform Clustering ---
    print("\n--- Step 2: Clustering Tools ---")
    
    hierarchy_path = os.path.join(output_dir, f"{base_name}_hierarchy_{now}.json")
    assigned_tools_path = os.path.join(output_dir, f"{base_name}_assigned_tools_{now}.jsonl")

    # Determine batch sizes based on debug mode
    if args.debug:
        hierarchy_path = os.path.join(output_dir, f"{base_name}_hierarchy_debug_{now}.json")
        assigned_tools_path = os.path.join(output_dir, f"{base_name}_assigned_tools_debug_{now}.jsonl")
        initial_batch_size = 30
        update_step_size = 20
        cluster_depth = 3
        assignment_batch_size = 10
        max_iterations = 1
    else:
        initial_batch_size = args.initial_batch_size
        update_step_size = 100
        cluster_depth = 4
        assignment_batch_size = 20
        max_iterations = 10

    agent = ClusteringAgent(model_name="o4-mini", log_dir=log_dir)
    
    # 1. Build hierarchy
    print("Building hierarchy...")
    final_hierarchy = agent.incremental_clustering(
        tools=all_tools,
        initial_batch_size=initial_batch_size,
        update_step_size=update_step_size,
        cluster_depth=cluster_depth,
        hierarchy_path=hierarchy_path,
        max_iterations=max_iterations
    )
    
    # 2. Assign tools to clusters
    print("\nAssigning tools to clusters...")
    assigned_tools = agent.assign_tools_to_clusters(
        tools=all_tools,
        assignment_model="gpt-4o-mini",
        batch_size=assignment_batch_size,
        save_interval=1000,
        output_path=assigned_tools_path
    )
    save_jsonl(assigned_tools, assigned_tools_path)
    print(f"Assigned tools saved to {assigned_tools_path}")
    
    print("\nScript finished.")


if __name__ == "__main__":
    main()
