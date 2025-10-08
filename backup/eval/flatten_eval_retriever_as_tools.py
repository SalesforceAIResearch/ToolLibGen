import sys
import os
from eval_prompt import QUESTION_TEMPLATE_FLATTEN_RETRIEVAL
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Set, Callable
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
        "description": "Search and retrieve the most relevant tools based on user query. This function helps find appropriate tools from a large tool library to assist with specific tasks. When you need additional computational tools or functions to solve a problem, use this function to search for relevant tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user query or task description to search for relevant tools. Be specific about what kind of functionality you need, including the domain (chemistry, physics, materials science, etc.) and the type of calculation or analysis needed."
                },
                "k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of relevant tools to retrieve. Default is 5. Increase if you need more options.",
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query"]
        }
    }
}


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
        
        # Create a shared state for dynamic tool updates
        self.dynamic_tools = []  # Tools that can be dynamically added
        self.dynamic_function_registry = {}  # Functions that can be dynamically added
        
    def retrieve_relevant_tools_impl(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Actual implementation for retrieving relevant tools.
        This method will be bound to the retrieve_relevant_tools function in the registry.
        
        Args:
            query: User query for tool search
            k: Number of tools to retrieve
            
        Returns:
            Dictionary with retrieved tools and metadata
        """
        if not self.tool_retriever:
            return {
                "success": False,
                "message": "Tool retriever not initialized",
                "retrieved_tools": [],
                "new_tools_available": []
            }
        
        try:
            # Retrieve relevant tools using the search query
            retrieved_tools = self.tool_retriever.retrieve_top_k(query, k)
            
            # Convert to OpenAI format and prepare for dynamic addition
            new_tools_info = []
            newly_added_tools = []
            
            for tool in retrieved_tools:
                tool_name = tool["description"]["function"]["name"]
                
                # Avoid duplicates
                if tool_name not in self.retrieved_tool_names:
                    # Convert to OpenAI format
                    openai_tool = self.tool_retriever.convert_tool_to_openai_format(tool)
                    
                    # Create executable function from python code
                    python_code = tool.get("python", "")
                    if python_code:
                        try:
                            # Extract python code from markdown if needed
                            if "```python" in python_code:
                                python_code = python_code.split("```python")[1].split("```")[0].strip()
                            
                            # Create local namespace and exec the code
                            local_namespace = {}
                            exec(python_code, local_namespace)
                            
                            # Find the main function (should match tool name)
                            if tool_name in local_namespace:
                                # Add to dynamic registries
                                self.dynamic_tools.append(openai_tool)
                                self.dynamic_function_registry[tool_name] = local_namespace[tool_name]
                                
                                # Update internal tracking
                                self.retrieved_tool_names.add(tool_name)
                                self.tool_name_to_data[tool_name] = tool
                                
                                new_tools_info.append({
                                    "name": tool_name,
                                    "description": tool["description"]["function"]["description"],
                                    "relevance_score": tool["relevance_score"]
                                })
                                newly_added_tools.append(tool_name)
                                
                                print(f"✅ Successfully added tool: {tool_name}")
                            else:
                                print(f"⚠️ Function {tool_name} not found in executed code")
                        except Exception as e:
                            print(f"❌ Error executing code for tool {tool_name}: {str(e)}")
                    else:
                        print(f"⚠️ No Python code found for tool {tool_name}")
            
            # Store retrieval information
            self.last_retrieval_info = {
                "original_query": query,
                "requested_k": k,
                "retrieved_count": len(retrieved_tools),
                "new_tools_added": len(newly_added_tools),
                "total_available_tools": len(self.retrieved_tool_names) + 1  # +1 for retrieval tool itself
            }
            
            result_message = f"Successfully retrieved and added {len(newly_added_tools)} new tools"
            if newly_added_tools:
                result_message += f": {', '.join(newly_added_tools)}"
            
            return {
                "success": True,
                "message": result_message,
                "retrieved_tools": new_tools_info,
                "new_tools_available": newly_added_tools,
                "total_tools_now_available": len(self.retrieved_tool_names) + 1,
                "original_query": query,
                "requested_k": k,
                "retrieved_count": len(retrieved_tools),
                "new_tools_added": len(newly_added_tools),
                "total_available_tools": len(self.retrieved_tool_names) + 1
            }
            
        except Exception as e:
            error_msg = f"Error retrieving tools: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "message": error_msg,
                "retrieved_tools": [],
                "new_tools_available": []
            }
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all currently available tools including dynamically added ones."""
        return self.available_tools + self.dynamic_tools
    
    def get_all_function_registry(self) -> Dict[str, Any]:
        """Get complete function registry including dynamically added functions."""
        # Create the base registry with the retrieval function
        complete_registry = {
            "retrieve_relevant_tools": self.retrieve_relevant_tools_impl
        }
        # Add dynamically retrieved functions
        complete_registry.update(self.dynamic_function_registry)
        
        # Add a special dispatcher function for dynamic tool resolution
        def dynamic_tool_dispatcher(function_name: str, **kwargs):
            """Dispatcher that can find dynamically added tools"""
            if function_name in self.dynamic_function_registry:
                return self.dynamic_function_registry[function_name](**kwargs)
            else:
                return {"error": f"Function {function_name} not found"}
        
        complete_registry["__dynamic_dispatcher__"] = dynamic_tool_dispatcher
        return complete_registry
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all currently available tools in OpenAI format (backward compatibility)."""
        return self.get_all_tools()
    
    def retrieve_relevant_tools(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Public interface for retrieving relevant tools (for demo and testing)."""
        return self.retrieve_relevant_tools_impl(query, k)
    
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
        self.dynamic_tools.clear()
        self.dynamic_function_registry.clear()
    
    def call_model_with_dynamic_tools(
        self, 
        messages: List[Dict[str, Any]], 
        model_name: str,
        openai_api_base: str,
        max_turns: int = 6,
        completion_check: Optional[Callable[[str], bool]] = None,
        max_turns_prompt: Optional[str] = None,
        **kwargs
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Call model with dynamic tool support that updates after each tool retrieval.
        
        Args:
            messages: Initial messages
            model_name: Model name
            openai_api_base: API base URL
            max_turns: Maximum conversation turns
            completion_check: Function to check if conversation is complete
            max_turns_prompt: Prompt to send when max turns reached
            **kwargs: Additional API parameters
            
        Returns:
            Tuple of (final_messages, actual_turns_used)
        """
        import openai
        import json
        
        # Initialize OpenAI client
        client = openai.OpenAI(
            api_key="EMPTY",
            base_url=openai_api_base,
        )
        
        # Default completion check
        if completion_check is None:
            completion_check = lambda content: "final answer:" in content.lower() if content else False
        
        # Make a copy of messages to avoid modifying the original
        working_messages = messages.copy()
        
        turn = 0
        
        while turn < max_turns:
            turn += 1
            
            # Get current tools and function registry (this updates dynamically!)
            current_tools = self.get_all_tools()
            current_function_registry = self.get_all_function_registry()
            
            print(f"Turn {turn}: Using {len(current_tools)} tools, {len(current_function_registry)} functions")
            
            # Prepare API call parameters
            api_params = {
                "model": model_name,
                "messages": working_messages,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.8),
                "max_tokens": kwargs.get("max_tokens", 8192),
                "extra_body": {
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
                    "chat_template_kwargs": {"enable_thinking": kwargs.get("enable_thinking", True)}
                },
            }
            
            # Add tools if available
            if current_tools:
                api_params["tools"] = current_tools
                
            # Make API call
            response = client.chat.completions.create(**api_params)
            
            # Add the assistant's response to messages
            working_messages.append(response.choices[0].message.model_dump())
            
            # Check if there are tool calls to handle
            if tool_calls := working_messages[-1].get("tool_calls", None):
                # Handle each tool call
                for tool_call in tool_calls:
                    call_id = tool_call["id"]
                    if fn_call := tool_call.get("function"):
                        fn_name = fn_call["name"]
                        try:
                            fn_args = json.loads(fn_call["arguments"])
                        except json.JSONDecodeError as e:
                            fn_args = {}
                            print(f"Warning: Failed to parse tool arguments: {e}")
                        
                        # Get the updated function registry and execute
                        updated_function_registry = self.get_all_function_registry()
                        func = updated_function_registry.get(fn_name, None)
                        if func:
                            try:
                                fn_result = func(**fn_args)
                                fn_res = json.dumps(fn_result)
                                print(f"✅ Successfully executed tool: {fn_name}")
                            except Exception as e:
                                fn_res = json.dumps({"error": f"Function execution failed: {str(e)}"})
                                print(f"❌ Error executing tool {fn_name}: {str(e)}")
                        else:
                            fn_res = json.dumps({"error": f"Function {fn_name} not found"})
                            print(f"❌ Function {fn_name} not found in registry")
                            print(f"Available functions: {list(updated_function_registry.keys())}")
                        
                        # Add tool result to messages
                        working_messages.append({
                            "role": "tool",
                            "content": fn_res,
                            "tool_call_id": call_id,
                        })
                
                # Continue the loop to let the model process tool results
                continue
            else:
                # No tool calls, check if conversation is complete
                current_content = working_messages[-1].get("content", "")
                if completion_check(current_content):
                    # Completion condition met, break the loop
                    break
                elif turn >= max_turns:
                    # Reached max turns, send custom prompt if provided
                    if max_turns_prompt:
                        working_messages.append({
                            "role": "user", 
                            "content": max_turns_prompt
                        })
                        
                        # One more API call with the custom prompt
                        final_api_params = {
                            "model": model_name,
                            "messages": working_messages,
                            "temperature": kwargs.get("temperature", 0.7),
                            "top_p": kwargs.get("top_p", 0.8),
                            "max_tokens": kwargs.get("max_tokens", 8192),
                            "extra_body": {
                                "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
                                "chat_template_kwargs": {"enable_thinking": kwargs.get("enable_thinking", True)}
                            },
                        }
                        
                        response = client.chat.completions.create(**final_api_params)
                        working_messages.append(response.choices[0].message.model_dump())
                    break
                else:
                    # Model didn't call tools and completion condition not met, continue
                    continue
        
        return working_messages, turn


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
   parser.add_argument("--passk", type=int, default=1)
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
         for tool in demo_result['retrieved_tools']:
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

         
         # Call model with tools if available
         if local_tool_manager:
            # Get all available tools and function registry
            available_tools = local_tool_manager.get_all_tools()
            function_registry = local_tool_manager.get_all_function_registry()
            
            print(f"Starting with {len(available_tools)} tools available")
            
            final_messages, turns = local_tool_manager.call_model_with_dynamic_tools(
               messages=messages, 
               model_name=args.model_name, 
               openai_api_base=args.api_base,
               max_turns=6,
               completion_check=lambda content: "final answer:" in content.lower() if content else False,
               max_turns_prompt="Please provide your final answer now. Your last line should start with 'Final Answer: YOUR_ALPHABETICAL_CHOICE'."
            )
            
            # Extract reasoning and response content from the final messages
            reasoning_content = ""
            response_content = ""
            
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
            
            # Calculate final tool retrieval metrics
            ground_truth_tools = extract_ground_truth_tools(d.get("extracted_art_info", []))
            final_retrieved_tools = list(local_tool_manager.retrieved_tool_names)
            recall_metrics = calculate_tool_recall(final_retrieved_tools, ground_truth_tools)
            
            # Update tool retrieval information with final results
            d[f"{args.model_nickname}_tool_retrieval"] = {
               "original_question": question,
               "ground_truth_tools": ground_truth_tools,
               "final_retrieved_tools": final_retrieved_tools,
               "total_retrieved_count": len(final_retrieved_tools),
               "recall": recall_metrics.get('recall', 0.0),
               "matched_tools": recall_metrics.get('matched_tools', []),
               "tool_calls_made": sum(1 for msg in final_messages if msg.get("tool_calls")),
               "retrieval_calls_made": sum(1 for msg in final_messages 
                                         if msg.get("tool_calls") and 
                                         any(call.get("function", {}).get("name") == "retrieve_relevant_tools" 
                                             for call in msg["tool_calls"]))
            }
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
            
            # Save empty tool retrieval info
            d[f"{args.model_nickname}_tool_retrieval"] = {
               "original_question": question,
               "ground_truth_tools": extract_ground_truth_tools(d.get("extracted_art_info", [])),
               "final_retrieved_tools": [],
               "total_retrieved_count": 0,
               "recall": 0.0,
               "matched_tools": [],
               "tool_calls_made": 0,
               "retrieval_calls_made": 0
            }
         
         correctness, pred_answer = eval_response(response_content, answer)
         
         d[f"{args.model_nickname}_response"] = f"<think>{reasoning_content}</think>{response_content}"
         d[f"{args.model_nickname}_correctness"] = correctness
         d[f"{args.model_nickname}_pred_answer"] = pred_answer
         d["id"] = id
         
         # Add tool retrieval information if enabled
         if local_tool_manager:
            d[f"{args.model_nickname}_available_tools_count"] = len(local_tool_manager.get_all_tools())
            d[f"{args.model_nickname}_final_retrieved_tools"] = list(local_tool_manager.retrieved_tool_names)
         
         results_lst.append(d)
         if len(results_lst) % args.save_step == 0:
            save_json(data=results_lst, file_path=save_path)
   
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
