import sys
import os
from eval_prompt import QUESTION_TEMPLATE_FLATTEN_RETRIEVAL
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import read_json, save_json, map_with_progress, call_vllm_with_temporary_tool, call_sfr_embedding_api_lst, save_pkl,read_pkl,call_vllm_wo_tool
from report_utils import quick_report  # Import the reusable reporting function
import argparse
from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


# OpenAI Function format for tool retrieval
TOOL_RETRIEVE_FUNCTION = {
    "type": "function",
    "function": {
        "name": "retrieve_relevant_tools",
        "description": "Search and retrieve the most relevant tools based on user query. This function helps find appropriate tools from a large tool library to assist with specific tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user query or task description to search for relevant tools. Be specific about what kind of functionality you need."
                }
            },
            "required": ["query"]
        }
    }
}

# Prompt template for LLM to generate tool retrieval queries
QUERY_GENERATION_PROMPT = """Given the following user question, analyze what type of computational tools or functions would be most helpful to answer it. 

User Question: {question}

Please generate a concise search query that would help find the most relevant tools from a scientific/technical tool library. Focus on:
- The specific domain (chemistry, physics, materials science, etc.)
- The type of calculation or analysis needed
- Key technical terms or concepts

Your search query should be optimized for finding relevant tools, not for answering the question directly.

Search Query:"""


class ToolRetriever:
    """
    A reusable tool retrieval system using FAISS for efficient similarity search.
    """
    
    def __init__(self, tool_data: List[Dict], tool_embedding: np.ndarray):
        """
        Initialize the tool retriever.
        
        Args:
            tool_data: List of tool dictionaries with descriptions and code
            tool_embedding: Pre-computed embeddings for the tools
        """
        self.tool_data = tool_data
        self.tool_embedding = tool_embedding
        self.dimension = tool_embedding.shape[1]
        
        # Build FAISS index for cosine similarity
        self.index = self._build_faiss_index()
        
    def _build_faiss_index(self) -> faiss.Index:
        """Build FAISS index for efficient similarity search."""
        # Normalize embeddings for cosine similarity
        normalized_embedding = self.tool_embedding.astype('float32')
        faiss.normalize_L2(normalized_embedding)
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(self.dimension)
        index.add(normalized_embedding)
        return index
    
    def retrieve_top_k(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant tools for the given query.
        
        Args:
            query: User query string
            k: Number of tools to retrieve
            
        Returns:
            List of tool dictionaries with relevance scores
        """
        # Get query embedding
        query_embedding_result = call_sfr_embedding_api_lst([query],is_query=True)
        if query_embedding_result is None or len(query_embedding_result) == 0:
            print(f"Error: Failed to get query embedding for: {query}")
            return []
        
        query_embedding = query_embedding_result[0]
        query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Ensure we don't request more tools than available
        max_available = len(self.tool_data)
        actual_k = min(k, max_available)
        
        print(f"Requesting {actual_k} tools (requested: {k}, available: {max_available})")
        
        # Search for similar tools
        scores, indices = self.index.search(query_embedding, actual_k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.tool_data):  # Ensure valid index
                tool = self.tool_data[idx].copy()
                tool['relevance_score'] = float(score)
                tool['rank'] = i + 1
                results.append(tool)
            else:
                print(f"Warning: Invalid tool index {idx}, skipping")
        
        print(f"Successfully retrieved {len(results)} tools")
        return results
    
    def convert_tool_to_openai_format(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert tool data to OpenAI function calling format.
        
        Args:
            tool: Tool dictionary from tool_data
            
        Returns:
            OpenAI function format dictionary
        """
        return {
            "type": "function",
            "function": tool["description"]["function"]
        }


class DynamicToolManager:
    """
    A reusable manager for dynamic tool pools in conversational systems.
    """
    
    def __init__(self, tool_retriever: Optional[ToolRetriever] = None, 
                 model_name: Optional[str] = None, api_base: Optional[str] = None):
        """
        Initialize the dynamic tool manager.
        
        Args:
            tool_retriever: ToolRetriever instance for fetching tools
            model_name: LLM model name for query generation (optional)
            api_base: API base URL for query generation (optional)
        """
        self.tool_retriever = tool_retriever
        self.model_name = model_name
        self.api_base = api_base
        self.available_tools = [TOOL_RETRIEVE_FUNCTION]  # Start with retrieval function
        self.retrieved_tool_names: Set[str] = set()  # Track retrieved tools
        self.tool_name_to_data: Dict[str, Dict] = {}  # Map tool names to full data
        self.last_retrieval_info: Dict[str, Any] = {}  # Store last retrieval information
        
    def retrieve_relevant_tools(self, query: str, k: int = 5, 
                              use_llm_query_generation: bool = True) -> Dict[str, Any]:
        """
        OpenAI function implementation for retrieving relevant tools.
        
        Args:
            query: User query for tool search
            k: Number of tools to retrieve
            use_llm_query_generation: Whether to use LLM to generate optimized search query
            
        Returns:
            Dictionary with retrieved tools and metadata
        """
        if not self.tool_retriever:
            return {
                "success": False,
                "message": "Tool retriever not initialized",
                "tools": []
            }
        
        try:
            # Generate optimized search query using LLM if enabled and configured
            search_query = query
            if (use_llm_query_generation and 
                self.model_name and self.api_base and 
                len(query) > 10):  # Only for substantial queries
                
                print(f"Generating optimized search query for: {query[:50]}...")
                search_query = generate_tool_search_query(query, self.model_name, self.api_base)
                print(f"Generated query: {search_query}")
            
            # Retrieve relevant tools using the search query
            retrieved_tools = self.tool_retriever.retrieve_top_k(search_query, k)
            
            # Store retrieval information before processing
            self.last_retrieval_info = {
                "original_query": query,
                "search_query": search_query,
                "used_llm_generation": use_llm_query_generation and search_query != query,
                "requested_k": k,
                "retrieved_count": len(retrieved_tools),
                "retrieved_tools_info": [
                    {
                        "name": tool["description"]["function"]["name"],
                        "relevance_score": tool["relevance_score"],
                        "rank": tool["rank"]
                    } for tool in retrieved_tools
                ]
            }
            
            # Convert to OpenAI format and add to available tools
            new_tools = []
            for tool in retrieved_tools:
                tool_name = tool["description"]["function"]["name"]
                
                # Avoid duplicates
                if tool_name not in self.retrieved_tool_names:
                    openai_tool = self.tool_retriever.convert_tool_to_openai_format(tool)
                    self.available_tools.append(openai_tool)
                    self.retrieved_tool_names.add(tool_name)
                    self.tool_name_to_data[tool_name] = tool
                    new_tools.append({
                        "name": tool_name,
                        "description": tool["description"]["function"]["description"],
                        "relevance_score": tool["relevance_score"]
                    })
            
            # Update retrieval info with final results
            self.last_retrieval_info.update({
                "new_tools_added": len(new_tools),
                "total_available_tools": len(self.available_tools)
            })
            
            return {
                "success": True,
                "message": f"Retrieved {len(new_tools)} new tools (total available: {len(self.available_tools)})",
                "tools": new_tools,
                "original_query": query,
                "search_query": search_query,
                "used_llm_generation": use_llm_query_generation and search_query != query,
                "requested_k": k,
                "actual_retrieved": len(retrieved_tools)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error retrieving tools: {str(e)}",
                "tools": []
            }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all currently available tools in OpenAI format."""
        return self.available_tools
    
    def get_tool_code(self, tool_name: str) -> Optional[str]:
        """Get the Python code for a specific tool."""
        if tool_name in self.tool_name_to_data:
            return self.tool_name_to_data[tool_name].get("python", "")
        return None
    
    def reset_tool_pool(self):
        """Reset the tool pool to initial state."""
        self.available_tools = [TOOL_RETRIEVE_FUNCTION]
        self.retrieved_tool_names.clear()
        self.tool_name_to_data.clear()


def eval_response(response, answer):   
   # Extract only the letter after "Final Answer:"
   if "Final Answer:" in response:
      pred_answer = response.split("Final Answer:")[-1].strip()
      # Extract only the first letter/character that appears to be an answer choice
      import re
      match = re.search(r'[a-d]', pred_answer)
      if match:
         pred_answer = match.group(0)
      else:
         # If no clear letter found, take the first non-whitespace character
         pred_answer = pred_answer.split()[0] if pred_answer.split() else ""
   else:
      pred_answer = ""
   
   if answer.lower() in pred_answer.lower():
      correctness = 1
   else:
      correctness = 0
   return correctness, pred_answer


def test_tool_data_format(tool_data, num_samples=3):
    """Test and print the format of tool data for debugging."""
    print(f"Total tools: {len(tool_data)}")
    print("Sample tool data format:")
    for i, tool in enumerate(tool_data[:num_samples]):
        print(f"Tool {i+1}:")
        print(f"  Type: {type(tool)}")
        if isinstance(tool, dict):
            print(f"  Keys: {list(tool.keys())}")
            if 'description' in tool:
                print(f"  Description type: {type(tool['description'])}")
                if isinstance(tool['description'], dict) and 'function' in tool['description']:
                    func = tool['description']['function']
                    print(f"    Function keys: {list(func.keys()) if isinstance(func, dict) else 'Not a dict'}")
                    if isinstance(func, dict):
                        print(f"    Name: {func.get('name', 'Missing')}")
                        print(f"    Description: {func.get('description', 'Missing')[:100]}...")
        print()


def get_tool_embedding(tool_data, tool_embedding_path):
    """
    Generate embeddings for tool data with improved error handling.
    
    Args:
        tool_data: List of tool dictionaries
        tool_embedding_path: Path to save the embedding file
    
    Returns:
        numpy.ndarray: Array of tool embeddings
    """
    print("Testing tool data format...")
    test_tool_data_format(tool_data)
    
    tool_embedding = []
    batch_size = 500  # Smaller batch size for more reliable processing
    print(f"Getting tool embedding for {len(tool_data)} tools...")
    
    failed_indices = []
    
    for i in range(0, len(tool_data), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(tool_data) + batch_size - 1)//batch_size} ({i}/{len(tool_data)})")
        batch_data = tool_data[i:i+batch_size]
        
        # Create batch text descriptions with error handling
        batch_data_str = []
        for j, d in enumerate(batch_data):
            try:
                if isinstance(d, dict) and 'description' in d:
                    desc = d['description']
                    if isinstance(desc, dict) and 'function' in desc:
                        func = desc['function']
                        if isinstance(func, dict):
                            name = func.get('name', f'tool_{i+j}')
                            description = func.get('description', 'No description available')
                            text = f"The function name is: {name}. The function description is: {description}."
                            batch_data_str.append(text)
                        else:
                            batch_data_str.append(f"Tool {i+j}: Invalid function format")
                    else:
                        batch_data_str.append(f"Tool {i+j}: Invalid description format")
                else:
                    batch_data_str.append(f"Tool {i+j}: Invalid tool format")
            except Exception as e:
                print(f"Error processing tool {i+j}: {e}")
                batch_data_str.append(f"Tool {i+j}: Error in processing")
        
        # Call embedding API with error handling
        try:
            batch_embedding = call_sfr_embedding_api_lst(batch_data_str)
            
            if batch_embedding is None:
                print(f"Warning: API returned None for batch {i//batch_size + 1}")
                # Create zero vectors as fallback
                embedding_dim = 1024  # Assumed dimension, adjust as needed
                batch_embedding = [[0.0] * embedding_dim for _ in range(len(batch_data_str))]
                failed_indices.extend(range(i, i + len(batch_data_str)))
            elif len(batch_embedding) != len(batch_data_str):
                print(f"Warning: Embedding count mismatch for batch {i//batch_size + 1}")
                print(f"Expected: {len(batch_data_str)}, Got: {len(batch_embedding)}")
                # Pad with zero vectors if needed
                embedding_dim = len(batch_embedding[0]) if batch_embedding else 1024
                while len(batch_embedding) < len(batch_data_str):
                    batch_embedding.append([0.0] * embedding_dim)
                    failed_indices.append(i + len(batch_embedding) - 1)
            
            tool_embedding.extend(batch_embedding)
            
        except Exception as e:
            print(f"Error calling embedding API for batch {i//batch_size + 1}: {e}")
            # Create zero vectors as fallback
            embedding_dim = 1024 if not tool_embedding else len(tool_embedding[0])
            fallback_embeddings = [[0.0] * embedding_dim for _ in range(len(batch_data_str))]
            tool_embedding.extend(fallback_embeddings)
            failed_indices.extend(range(i, i + len(batch_data_str)))
    
    if failed_indices:
        print(f"Warning: {len(failed_indices)} tools failed to get embeddings and were replaced with zero vectors")
        print(f"Failed indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
    
    # Convert to numpy array
    tool_embedding = np.array(tool_embedding)
    print(f"Final embedding shape: {tool_embedding.shape}")
    
    # Save embeddings
    save_pkl(data=tool_embedding, file_path=tool_embedding_path)
    print(f"Embeddings saved to: {tool_embedding_path}")
    
    return tool_embedding


def create_tool_retrieval_system(tool_data: List[Dict], tool_embedding: np.ndarray, 
                                model_name: Optional[str] = None, 
                                api_base: Optional[str] = None) -> DynamicToolManager:
    """
    Factory function to create a complete tool retrieval system.
    This is the main entry point for other files to use this functionality.
    
    Args:
        tool_data: List of tool dictionaries
        tool_embedding: Pre-computed embeddings for tools
        model_name: LLM model name for query generation (optional)
        api_base: API base URL for query generation (optional)
        
    Returns:
        Configured DynamicToolManager instance
    """
    retriever = ToolRetriever(tool_data, tool_embedding)
    manager = DynamicToolManager(retriever, model_name, api_base)
    return manager


def generate_tool_search_query(question: str, model_name: str, api_base: str) -> str:
    """
    Use LLM to generate an optimized search query for tool retrieval.
    
    Args:
        question: The original user question
        model_name: LLM model name
        api_base: API base URL
        
    Returns:
        Generated search query string
    """
    prompt = QUERY_GENERATION_PROMPT.format(question=question)
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = call_vllm_wo_tool(
            messages=messages,
            model_name=model_name,
            openai_api_base=api_base,
        )
        
        # Extract the generated query
        generated_query = response.get("answer", "").strip()
        
        # Clean up common response patterns
        if "Search Query:" in generated_query:
            generated_query = generated_query.split("Search Query:")[-1].strip()
        if generated_query.startswith('"') and generated_query.endswith('"'):
            generated_query = generated_query[1:-1]
            
        # Fallback to original question if generation fails
        if not generated_query or len(generated_query) < 3:
            generated_query = question
            
        return generated_query
        
    except Exception as e:
        print(f"Error generating search query: {e}")
        return question  # Fallback to original question


def calculate_tool_recall(retrieved_tools, ground_truth_tools):
    """
    Calculate recall for retrieved tools against ground truth tools.
    
    Args:
        retrieved_tools: List of retrieved tool names
        ground_truth_tools: List of ground truth tool names
    
    Returns:
        Dict with recall metrics
    """
    if not ground_truth_tools:
        return {
            "recall": 0.0,
            "retrieved_count": len(retrieved_tools),
            "ground_truth_count": 0,
            "matched_tools": []
        }
    
    # Convert to sets for easier comparison
    retrieved_set = set(retrieved_tools)
    ground_truth_set = set(ground_truth_tools)
    
    # Find matches
    matched_tools = retrieved_set.intersection(ground_truth_set)
    
    # Calculate recall
    recall = len(matched_tools) / len(ground_truth_set)
    
    return {
        "recall": recall,
        "retrieved_count": len(retrieved_tools),
        "ground_truth_count": len(ground_truth_tools),
        "matched_tools": list(matched_tools)
    }


def extract_ground_truth_tools(extracted_art_info):
    """
    Extract ground truth tool names from extracted_art_info.
    
    Args:
        extracted_art_info: List of ground truth tool information
    
    Returns:
        List of ground truth tool names
    """
    if not extracted_art_info:
        return []
    
    ground_truth_tools = []
    for tool_info in extracted_art_info:
        if isinstance(tool_info, dict) and "function_name" in tool_info:
            ground_truth_tools.append(tool_info["function_name"])
    
    return ground_truth_tools


def calculate_recall_at_k(tool_retrieval_info, ground_truth_tools, k_values=[1, 3, 5, 10]):
    """
    Calculate recall@k for different k values.
    
    Args:
        tool_retrieval_info: Tool retrieval information containing retrieved tools
        ground_truth_tools: List of ground truth tool names
        k_values: List of k values to calculate recall for
    
    Returns:
        Dict with recall@k for each k value
    """
    # Check if we have retrieved tools data - could be in 'tools' or 'retrieved_tools'
    retrieved_tools = None
    if tool_retrieval_info:
        if 'tools' in tool_retrieval_info:
            retrieved_tools = tool_retrieval_info['tools']
        elif 'retrieved_tools' in tool_retrieval_info:
            retrieved_tools = tool_retrieval_info['retrieved_tools']
    
    if not retrieved_tools:
        return {f"recall@{k}": 0.0 for k in k_values}
    
    recall_results = {}
    
    for k in k_values:
        # Get top-k retrieved tools
        top_k_tools = [tool['name'] for tool in retrieved_tools[:k]]
        
        # Calculate recall@k
        recall_info = calculate_tool_recall(top_k_tools, ground_truth_tools)
        recall_results[f"recall@{k}"] = recall_info['recall']
        
        # Also save detailed info for top-5
        if k == 5:
            recall_results["recall_details"] = recall_info
    
    return recall_results


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--input_data_path", type=str, default="/export/home/data/dev_physics_322.json")
   parser.add_argument("--model_nickname", type=str, default="Qwen3_8b")
   parser.add_argument("--model_name", type=str, default="/export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model")
   parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
   parser.add_argument("--tool_path", type=str, default="/export/home/data/valid_science_toolset_for_hard_examples_tagged_tools_20250619_054608.json")
   parser.add_argument("--tool_embedding_path", type=str, default="/export/home/data/all_tagged_tools_20250619_054608_embedding.pkl")
   parser.add_argument("--save_path", type=str, default="/export/home/data/dev_flatten.json")
   parser.add_argument("--save_step", type=int, default=500)
   parser.add_argument("--passk", type=int, default=4)
   parser.add_argument("--debug", action='store_true', default=False)
   parser.add_argument("--enable_tool_retrieval", action='store_true', default=True, help="Enable dynamic tool retrieval functionality")
   args = parser.parse_args()
   save_path = args.save_path.replace(".json", f"_{current_time}.json")
   data = read_json(args.input_data_path)
   
   if args.debug:
      data = data[:10]
      save_path = save_path.replace(".json", "_debug.json")
   else:
      data = data
   
   results_lst = []
   tool_data = read_json(args.tool_path)
   tool_embedding_path = args.tool_embedding_path
   
   if os.path.exists(tool_embedding_path):
      tool_embedding = read_pkl(tool_embedding_path)
   else:
      tool_embedding = get_tool_embedding(tool_data, tool_embedding_path)
   assert len(tool_embedding) == len(tool_data), f"Tool embedding length {len(tool_embedding)} != tool data length {len(tool_data)}"

   # Create tool retrieval system
   tool_manager = None
   if args.enable_tool_retrieval:
      print("Initializing tool retrieval system...")
      tool_manager = create_tool_retrieval_system(tool_data, tool_embedding, args.model_name, args.api_base)
      print(f"Tool retrieval system initialized with {len(tool_data)} tools")
      
      # Demonstrate tool retrieval functionality
      print("\n=== Tool Retrieval Demo ===")
      demo_query = "calculate chemical reaction properties"
      demo_result = tool_manager.retrieve_relevant_tools(demo_query, k=5)
      print(f"Demo query: '{demo_query}'")
      print(f"Result: {demo_result['message']}")
      if demo_result['success']:
         for tool in demo_result['tools']:
            print(f"  - {tool['name']}: {tool['description'][:100]}... (score: {tool['relevance_score']:.3f})")
      print("=== End Demo ===\n")
      
      # Reset tool manager after demo to start fresh for each question
      tool_manager.reset_tool_pool()
      print("Tool manager reset after demo")

   def fn(d):
      global results_lst
      # Create a separate tool manager instance for each thread to avoid shared state
      local_tool_manager = None
      if args.enable_tool_retrieval:
         local_tool_manager = create_tool_retrieval_system(tool_data, tool_embedding, args.model_name, args.api_base)
      
      # try:
      for i in range(args.passk):
         id = str(d["id"])+f"_{i}"
         question = d["question"]
         answer = d["answer"]
         prompt = QUESTION_TEMPLATE_FLATTEN_RETRIEVAL.format(question=question)
         
         # Prepare messages and tools
         messages = [
            {"role": "user", "content": prompt}
         ]
         
         # Add tool retrieval capability if enabled
         available_tools = []
         if local_tool_manager:
            # Start fresh for each question
            local_tool_manager.reset_tool_pool()
            available_tools = local_tool_manager.get_available_tools()
            
            # Extract ground truth tools for recall calculation
            ground_truth_tools = extract_ground_truth_tools(d.get("extracted_art_info", []))
            
            # Auto-retrieve relevant tools for each question
            if args.enable_tool_retrieval:
               print(f"Retrieving tools for question: {question[:80]}...")
               retrieval_result = local_tool_manager.retrieve_relevant_tools(
                  query=question, 
                  k=10,  # Retrieve top-10 for comprehensive recall analysis
                  use_llm_query_generation=True
               )
               
               # Calculate recall@k for different k values
               recall_metrics = calculate_recall_at_k(
                  tool_retrieval_info=retrieval_result,
                  ground_truth_tools=ground_truth_tools,
                  k_values=[1, 3, 5, 10]
               )
               
               # Save retrieval query information with recall metrics
               d[f"{args.model_nickname}_tool_retrieval"] = {
                  "original_question": question,
                  "original_query": question,
                  "search_query": retrieval_result.get('search_query', question),
                  "used_llm_generation": retrieval_result.get('used_llm_generation', False),
                  "requested_k": retrieval_result.get('requested_k', 10),
                  "actual_retrieved": retrieval_result.get('actual_retrieved', 0),
                  "retrieved_tools": retrieval_result.get('tools', []),
                  "success": retrieval_result.get('success', False),
                  "message": retrieval_result.get('message', ''),
                  "ground_truth_tools": ground_truth_tools,
                  "recall_at_1": recall_metrics.get('recall@1', 0.0),
                  "recall_at_3": recall_metrics.get('recall@3', 0.0),
                  "recall_at_5": recall_metrics.get('recall@5', 0.0),
                  "recall_at_10": recall_metrics.get('recall@10', 0.0),
                  "recall_details": recall_metrics.get('recall_details', {})
               }
               
               if retrieval_result['success']:
                  print(f"Retrieved {len(retrieval_result['tools'])} tools")
                  print(f"Ground truth tools: {len(ground_truth_tools)} - {ground_truth_tools}")
                  print(f"Recall@1: {recall_metrics.get('recall@1', 0.0):.3f}, Recall@3: {recall_metrics.get('recall@3', 0.0):.3f}, Recall@5: {recall_metrics.get('recall@5', 0.0):.3f}, Recall@10: {recall_metrics.get('recall@10', 0.0):.3f}")
                  print(f"Requested: {retrieval_result.get('requested_k', 'unknown')}, Actually retrieved: {retrieval_result.get('actual_retrieved', 'unknown')}")
                  if retrieval_result.get('used_llm_generation', False):
                     print(f"  Original query: {retrieval_result['original_query'][:60]}...")
                     print(f"  Generated query: {retrieval_result['search_query'][:60]}...")
                  else:
                     print(f"  Used direct query: {retrieval_result['search_query'][:60]}...")
                  for tool in retrieval_result['tools']:
                     print(f"    - {tool['name']} (score: {tool['relevance_score']:.3f})")
                  available_tools = local_tool_manager.get_available_tools()
               else:
                  print(f"Tool retrieval failed: {retrieval_result['message']}")
            else:
               # If tool retrieval is disabled, save empty retrieval info
               d[f"{args.model_nickname}_tool_retrieval"] = {
                  "original_question": question,
                  "original_query": question,
                  "search_query": question,
                  "used_llm_generation": False,
                  "requested_k": 0,
                  "actual_retrieved": 0,
                  "retrieved_tools": [],
                  "success": False,
                  "message": "Tool retrieval disabled",
                  "ground_truth_tools": ground_truth_tools,
                  "recall_at_1": 0.0,
                  "recall_at_3": 0.0,
                  "recall_at_5": 0.0,
                  "recall_at_10": 0.0,
                  "recall_details": {}
               }
         
         # Call model with tools if available
         if available_tools and len(available_tools) > 1:  # More than just retrieval function
            # Create a simple function registry for the retrieved tools
            function_registry = {}
            for tool in available_tools:
                if tool["type"] == "function":
                    func_name = tool["function"]["name"]
                    # Create a dummy function that returns the tool name
                    function_registry[func_name] = lambda **kwargs: f"Tool {func_name} called with args: {kwargs}"
            
            final_messages, turns = call_vllm_with_temporary_tool(
               messages=messages, 
               model_name=args.model_name, 
               openai_api_base=args.api_base,
               tools=available_tools,
               function_registry=function_registry,
               completion_check=lambda content: "final answer:" in content.lower(),
               max_turns_prompt="Please provide your final answer now. Your last line should start with 'Final Answer: YOUR_ALPHABETICAL_CHOICE'."
            )
            
            # Extract reasoning and response content from the final messages
            reasoning_content = ""
            response_content = ""
            
            # Find the last assistant message
            for msg in reversed(final_messages):
               if msg.get("role") == "assistant":
                  response_content = msg.get("content", "")
                  # Check if there's reasoning content (this depends on the model's response format)
                  if hasattr(msg, "reasoning_content"):
                     reasoning_content = msg.reasoning_content
                  break
                  
            # Save complete message history
            d[f"{args.model_nickname}_full_messages"] = final_messages
            d[f"{args.model_nickname}_conversation_turns"] = turns
         else:
            # Use call_vllm_wo_tool when no tools are available for consistent return format
            response = call_vllm_wo_tool(
               messages=messages, 
               model_name=args.model_name, 
               openai_api_base=args.api_base
            )
            reasoning_content = response.get("thinking", "")
            response_content = response.get("answer", "")
            
            # Save the simple message exchange
            d[f"{args.model_nickname}_full_messages"] = messages + [
               {"role": "assistant", "content": response_content, "thinking": reasoning_content}
            ]
            d[f"{args.model_nickname}_conversation_turns"] = 1
         
         correctness, pred_answer = eval_response(response_content, answer)
         
         d[f"{args.model_nickname}_response"] = f"<think>{reasoning_content}</think>{response_content}"
         d[f"{args.model_nickname}_correctness"] = correctness
         d[f"{args.model_nickname}_pred_answer"] = pred_answer
         d["id"] = id
         
         # Add tool retrieval information if enabled
         if local_tool_manager:
            d[f"{args.model_nickname}_available_tools_count"] = len(available_tools)
            # Note: retrieved_tool_names contains cumulative tools across all processed questions
            # For per-question retrieved tools, check the tool_retrieval field above
         
         results_lst.append(d)
         if len(results_lst) % args.save_step == 0:
            save_json(data=results_lst, file_path=save_path)
      # except Exception as e:
      #    print(f"Error processing item {d.get('id', 'unknown')}: {e}")
      #    pass
   
   map_with_progress(f=fn, xs=data, num_threads=50)
   save_json(data=results_lst, file_path=save_path)
   
   if tool_manager:
      print(f"\nFinal tool retrieval stats:")
      print(f"  Total tools retrieved: {len(tool_manager.retrieved_tool_names)}")
      print(f"  Retrieved tool names: {list(tool_manager.retrieved_tool_names)}")

   # Generate comprehensive evaluation report using the reusable reporting utility
   stats_dict, report_path = quick_report(
      results_lst=results_lst,
      model_nickname=args.model_nickname,
      passk=args.passk,
      original_count=len(data),
      save_path=save_path,
      current_time=current_time,
      tool_retrieval_enabled=args.enable_tool_retrieval
   )
   
   print(f"\n🎉 Evaluation complete!")
   print(f"📊 Results saved to: {save_path}")
   print(f"📈 Statistics: {save_path.replace('.json', '_stats.json')}")
   print(f"📝 Report: {report_path}")
