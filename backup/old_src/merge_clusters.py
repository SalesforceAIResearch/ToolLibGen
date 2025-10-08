from utils import read_jsonl, read_json, save_json
import argparse


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--hierarchy_file", type=str, default="/Users/murong.yue/Desktop/data/ReasonMed_tools_100k_20250915_091638_hierarchy_20250924_154037.json")
   parser.add_argument("--assigned_tools_file", type=str, default="/Users/murong.yue/Desktop/data/ReasonMed_tools_100k_20250915_091638_assigned_tools_20250923_053601.jsonl")
   parser.add_argument("--saved_tools_file", type=str, default="/Users/murong.yue/Desktop/data/ReasonMed_tools_saved_all.json")
   parser.add_argument("--tool_per_node", type=int, default=100)
   args = parser.parse_args()
   hierarchy = read_json(args.hierarchy_file)
   assigned_tools = read_jsonl(args.assigned_tools_file)
   tool_per_node = args.tool_per_node
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

   incorrect_classification_num = 0

   for index, tool in enumerate(assigned_tools):
      assign = tool.get("cluster_assignment", {}) if isinstance(tool, dict) else {}
      tool_cluster = assign.get("cluster_id")
      cluster_ids = assign.get("cluster_ids", [])

      # Resolve target cluster id:
      # 1) Prefer exact cluster_id if it exists in hierarchy
      # 2) Otherwise, try deepest id from cluster_ids path that exists
      # 3) Otherwise, create a new node using the deepest id (or cluster_id as fallback)
      target_id = None
      if tool_cluster in cluster_dict:
         target_id = tool_cluster
      else:
         # Try reversed path to find an existing node
         if isinstance(cluster_ids, list) and cluster_ids:
            for cid in reversed(cluster_ids):
               if cid in cluster_dict:
                  target_id = cid
                  break
            # If none exists, pick the deepest one
            if target_id is None:
               target_id = cluster_ids[-1]
         else:
            target_id = tool_cluster

         # Create a new flat node if still missing
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
            # No parent merging required in flat mode; skip upward merge
            continue

   print("Merged clusters:")
   cluster_lst = []
   for k, v in sorted(cluster_dict.items()):
      if len(v["tools_index"]) > 0:
         parent = v.get('parent', None)
         level = v.get('level', 0)
         tool_count = len(v['tools_index'])
         # If tool_count is very large, split this cluster into evenly sized chunks (<=1000)
         if tool_count < 50:
            continue
         if "}" in k:
            continue
         if tool_count > 300:
            num_chunks = (tool_count + 299) // 300
            base_size = tool_count // num_chunks
            remainder = tool_count % num_chunks
            start = 0
            for chunk_index in range(num_chunks):
               chunk_size = base_size + (1 if chunk_index < remainder else 0)
               end = start + chunk_size
               indices_slice = v["tools_index"][start:end]
               print(f"Cluster ID: {k}_{chunk_index} parent: {parent} level: {level} tool_count: {len(indices_slice)}")
               tools = []
               for index in indices_slice:
                  tools.append(assigned_tools[index])
               cluster_lst.append({"cluster_name":f"{k}_{chunk_index}","tools":tools})
               start = end
            continue
         # Keep small clusters as well in flat mode
         print(f"Cluster ID: {k} parent: {parent} level: {level} tool_count: {tool_count}")
         tools = []
         for index in v["tools_index"]:
            tools.append(assigned_tools[index])
         cluster_lst.append({"cluster_name":k,"tools":tools})
   save_json(data=cluster_lst,file_path=args.saved_tools_file)
   
