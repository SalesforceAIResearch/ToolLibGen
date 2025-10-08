from utils import *
from prompt import *
import argparse
import random
from datetime import datetime
from ToolExtractionAgent import ToolExtractionAgent
from ClusteringAgent import ClusteringAgent


def process_extracted_tools(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
   all_tools = []
   for entry in raw_data:
      try:
         agent_result_raw = entry.get("agent_result")
         if isinstance(agent_result_raw, str):
               agent_result = json.loads(agent_result_raw)
         elif isinstance(agent_result_raw, dict):
               agent_result = agent_result_raw
         else:
               agent_result = {}

         final_tools = agent_result.get("final_tools", [])
         original_question = entry.get("question", "")
         original_answer = entry.get("answer", "")

         for tool in final_tools:
               tool_info = tool.get("tool_info", {})
               tool_code = tool.get("code", "")
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
         pass
         # print(f"Skipping an entry due to parsing error: {e}")
   
   print(f"Extracted a total of {len(all_tools)} tools.")      
   return all_tools



def merge_tools(hierarchy: Dict[str, Any], assigned_tools: List[Dict[str, Any]], tool_per_node: int=100) -> List[Dict[str, Any]]:
   data = hierarchy["clusters"]
   cluster_dict = {}
   for d in data:
      id = d["id"]
      parent = d["parent"]
      child = d["children"]
      if id not in cluster_dict:
         cluster_dict[id] = {"parent":parent,"tools_index":[], "level":d["level"]}
      for c in child:
         if c not in cluster_dict:
            cluster_dict[c] = {"parent": id,"tools_index":[], "level":d["level"]+1}
   for index, tool in enumerate(assigned_tools):
      assign = tool.get("cluster_assignment", {}) if isinstance(tool, dict) else {}
      tool_cluster = assign.get("cluster_id")
      cluster_ids = assign.get("cluster_ids", [])
      target_id = None
      if tool_cluster in cluster_dict:
         target_id = tool_cluster
      else:
         if isinstance(cluster_ids, list) and cluster_ids:
            for cid in reversed(cluster_ids):
               if cid in cluster_dict:
                  target_id = cid
                  break
            if target_id is None:
               target_id = cluster_ids[-1]
         else:
            target_id = tool_cluster

         if target_id not in cluster_dict:
            cluster_dict[target_id] = {"parent": None, "tools_index": [], "level": 0}

      if target_id is not None:
         cluster_dict[target_id]["tools_index"].append(index)

   max_level = 0
   for cluster_id in cluster_dict:
      if "level" in cluster_dict[cluster_id] and cluster_dict[cluster_id]["level"] > max_level:
         max_level = cluster_dict[cluster_id]["level"]

   for level in range(max_level, -1, -1):
      clusters_at_this_level = [cid for cid, cinfo in cluster_dict.items() if cinfo.get("level") == level]
      
      for cluster_id in clusters_at_this_level:
         if len(cluster_dict[cluster_id]["tools_index"]) < tool_per_node:
            continue

   lowest_cluster_number = 50
   highest_cluster_number = 300
   if args.debug:
      lowest_cluster_number = 1
      highest_cluster_number = 300
   print("Merging clusters")
   cluster_lst = []
   for k, v in sorted(cluster_dict.items()):
      if len(v["tools_index"]) > 0:
         parent = v.get('parent', None)
         level = v.get('level', 0)
         tool_count = len(v['tools_index'])
         if tool_count < lowest_cluster_number:
            continue
         if "}" in k:
            continue
         if tool_count > highest_cluster_number:
            num_chunks = (tool_count + highest_cluster_number - 1) // highest_cluster_number
            base_size = tool_count // num_chunks
            remainder = tool_count % num_chunks
            start = 0
            for chunk_index in range(num_chunks):
               chunk_size = base_size + (1 if chunk_index < remainder else 0)
               end = start + chunk_size
               indices_slice = v["tools_index"][start:end]
               # print(f"Cluster ID: {k}_{chunk_index} parent: {parent} level: {level} tool_count: {len(indices_slice)}")
               tools = []
               for index in indices_slice:
                  tools.append(assigned_tools[index])
               cluster_lst.append({"cluster_name":f"{k}_{chunk_index}","tools":tools})
               start = end
            continue
         # print(f"Cluster ID: {k} parent: {parent} level: {level} tool_count: {tool_count}")
         tools = []
         for index in v["tools_index"]:
            tools.append(assigned_tools[index])
         cluster_lst.append({"cluster_name":k,"tools":tools})
   save_json(data=cluster_lst,file_path=merged_tools_save_path)
   print(f"Merged clusters saved to {merged_tools_save_path}")
   return cluster_lst



def extract_tools(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
   agent = ToolExtractionAgent(
      generation_model_name=args.generation_model_name,
      verification_model_name_lst=args.verification_model_name_lst,
      skip_on_validation_failure=False,
      code_validation_timeout=1
   )
   res_lst = []
   def fn(d):
      nonlocal res_lst
      try:
         question = d["question"]
         reasoning = d["reasoning"][0] if isinstance(d["reasoning"], list) else d["reasoning"]
         answer = d["answer"]
         # Use ToolExtractionAgent for agentic tool generation and validation with a 10-minute timeout
         executor = ThreadPoolExecutor(max_workers=1)
         future = executor.submit(
            agent.run_agent_system,
            question=question,
            CoT=reasoning,
            answer=answer,
            max_iterations=0,
            evaluation_repeats=1
         )
         _timed_out = False
         try:
            result = future.result(timeout=600)
         except TimeoutError:
            _timed_out = True
            executor.shutdown(wait=False, cancel_futures=True)
            return
         finally:
            if not _timed_out:
               executor.shutdown(wait=True, cancel_futures=True)
         
         # Store the complete agent result (full interaction history)
         d['agent_result'] = result
         # d['trajectory'] = result['trajectory']
         
         # Extract tools if the agent was successful
         if result['status'] == 'success' and result['final_tools']:
            d["tools"] = [tool.get('code', '') for tool in result['final_tools']]
            d["error_reason"] = ""
         else:
            d["tools"] = []
            d["error_reason"] = f"Agent failed: {result.get('message', 'Unknown error')}"
         res_lst.append(d)
      except Exception as e:
         if args.debug:
            print(f"Error processing item: {e}")
         d["tools"] = []
         d["error_reason"] = f"Exception: {str(e)}"
         res_lst.append(d)

      if len(res_lst) % 200 == 0:
         save_json(res_lst, extracted_tool_save_path)


   # Use lower thread count to prevent potential issues, but keep it simple
   thread_count = 20 if args.debug else 50  # Reduced from 200
   print(f"Starting processing with {thread_count} threads...")
   
   # Use map_with_progress for parallel processing
   map_with_progress(fn, data, num_threads=thread_count)
   save_json(res_lst, extracted_tool_save_path)
   print(f"Processing completed. Saved {len(res_lst)} results to {extracted_tool_save_path}")
   return res_lst



def clustering(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
   initial_batch_size = args.initial_batch_size
   update_step_size = 100
   cluster_depth = 4
   assignment_batch_size = 20
   max_iterations = 100

   agent = ClusteringAgent(model_name=args.generation_model_name, log_dir=args.log_dir)
   print("Building hierarchy...")
   final_hierarchy = agent.incremental_clustering(
      tools=data,
      initial_batch_size=initial_batch_size,
      update_step_size=update_step_size,
      cluster_depth=cluster_depth,
      hierarchy_path=clustered_hierarchy_save_path,
      max_iterations=max_iterations
   )
   save_json(final_hierarchy, clustered_hierarchy_save_path)
   print("Assigning tools to clusters...")
   assigned_tools = agent.assign_tools_to_clusters(
      tools=data,
      assignment_model=args.generation_model_name,
      batch_size=assignment_batch_size,
      save_interval=1000,
      output_path=clustered_assigned_tools_save_path
      )
   save_jsonl(assigned_tools, clustered_assigned_tools_save_path)
   print(f"Assigned tools saved to {clustered_assigned_tools_save_path}")
   return final_hierarchy, assigned_tools


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--file_path",default="data/example_test.json", type=str)
   parser.add_argument("--save_folder",default="save_folder", type=str)
   parser.add_argument("--generation_model_name", type=str, default="o4-mini")
   parser.add_argument("--verification_model_name_lst", type=list, default=["gpt-4.1"])
   parser.add_argument("--debug", action="store_true", default=True)
   parser.add_argument("--initial_batch_size", type=int, default=100)
   parser.add_argument("--log_dir", type=str, default="log")
   args = parser.parse_args()
   datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
   if not os.path.exists(args.save_folder):
      os.makedirs(args.save_folder, exist_ok=True)
   if not os.path.exists(args.log_dir):
      os.makedirs(args.log_dir, exist_ok=True)
   file_path = args.file_path
   save_folder = args.save_folder
   print(f"The current read file path is {file_path} and the current save path is {save_folder}")
   extracted_tool_save_path = os.path.join(save_folder, "extracted_tools.json")
   clustered_hierarchy_save_path = os.path.join(save_folder, "clustered_hierarchy.json")
   clustered_assigned_tools_save_path = os.path.join(save_folder, "clustered_assigned_tools.jsonl")
   merged_tools_save_path = os.path.join(save_folder, "merged_tools.json")
   data = read_json(file_path)
   if args.debug:
      n = 30
      random.seed(42)
      random.shuffle(data)
      data = data[:n]
      extracted_tool_save_path = extracted_tool_save_path.replace(".json", f"_{datetime_str}_debug.json")
      clustered_hierarchy_save_path = clustered_hierarchy_save_path.replace(".json", f"_{datetime_str}_debug.json")
      clustered_assigned_tools_save_path = clustered_assigned_tools_save_path.replace(".jsonl", f"_{datetime_str}_debug.jsonl")
      merged_tools_save_path = merged_tools_save_path.replace(".json", f"_{datetime_str}_debug.json")
   else:
      extracted_tool_save_path = extracted_tool_save_path.replace(".json", f"_{datetime_str}.json")
      clustered_hierarchy_save_path = clustered_hierarchy_save_path.replace(".json", f"_{datetime_str}.json")
      clustered_assigned_tools_save_path = clustered_assigned_tools_save_path.replace(".jsonl", f"_{datetime_str}.jsonl")
      merged_tools_save_path = merged_tools_save_path.replace(".json", f"_{datetime_str}.json")
   
   if os.path.exists(extracted_tool_save_path):
      extracted_tools = read_json(extracted_tool_save_path)
   else:
      extracted_tools = extract_tools(data)
      save_json(extracted_tools, extracted_tool_save_path)
   data = process_extracted_tools(extracted_tools)
   if os.path.exists(clustered_hierarchy_save_path):
      final_hierarchy = read_json(clustered_hierarchy_save_path)
   else:
      final_hierarchy, assigned_tools = clustering(data)
      save_json(final_hierarchy, clustered_hierarchy_save_path)
      save_jsonl(assigned_tools, clustered_assigned_tools_save_path)
   merged_tools = merge_tools(final_hierarchy, assigned_tools, tool_per_node=100)
   save_json(merged_tools, merged_tools_save_path)

