#!/usr/bin/env python3
"""
Unified Dynamic Tool Retrieval and Execution System
支持 valid_science_toolset 和 v_2.json 两种格式
使用 GPT-4.1 进行测试
"""

import sys
import os
from eval_prompt import QUESTION_TEMPLATE_FLATTEN_RETRIEVAL
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Set, Callable
from collections import defaultdict
import importlib.util
import tempfile
import traceback
import openai
import json
import inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import read_json, save_json, map_with_progress, call_sfr_embedding_api_lst, save_pkl, read_pkl, call_openai_api, execute_code
from report_utils import quick_report
import argparse
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI Function format for tool retrieval
TOOL_RETRIEVE_FUNCTION = {
    "type": "function",
    "function": {
        "name": "retrieve_relevant_tools",
        "description": "Search and retrieve the most relevant tools based on user query. When you need additional computational tools or functions to solve a problem, use this function to search for relevant tools.",
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


class ToolFormatDetector:
    """
    自动检测工具文件格式
    """
    
    @staticmethod
    def detect_format(tool_data: List[Dict]) -> str:
        """
        检测工具数据格式
        
        Args:
            tool_data: 工具数据列表
            
        Returns:
            'v2' for v_2.json format, 'valid_science' for valid_science_toolset format
        """
        if not tool_data:
            return "unknown"
        
        sample = tool_data[0]
        
        # 检查 v_2.json 格式特征
        if "tool_info" in sample and "tool_code" in sample:
            return "v2"
        
        # 检查 valid_science_toolset 格式特征
        if "description" in sample and "python" in sample:
            return "valid_science"
        
        return "unknown"


class UnifiedToolRetriever:
    """
    统一的工具检索系统，支持多种格式
    """
    
    def __init__(self, tool_data: List[Dict], tool_embedding: np.ndarray):
        """
        Initialize the unified tool retriever.
        
        Args:
            tool_data: List of tool dictionaries with descriptions and code
            tool_embedding: Pre-computed embeddings for the tools
        """
        self.tool_data = tool_data
        self.tool_embedding = tool_embedding
        self.dimension = tool_embedding.shape[1]
        self.format = ToolFormatDetector.detect_format(tool_data)
        
        # print(f"🔍 Detected tool format: {self.format}")
        
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
        query_embedding_result = call_sfr_embedding_api_lst([query], is_query=True)
        if query_embedding_result is None or len(query_embedding_result) == 0:
            print(f"Error: Failed to get query embedding for: {query}")
            return []
        
        query_embedding = query_embedding_result[0]
        query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Ensure we don't request more tools than available
        max_available = len(self.tool_data)
        actual_k = min(k, max_available)
        
        # print(f"Requesting {actual_k} tools (requested: {k}, available: {max_available})")
        
        # Search for similar tools
        scores, indices = self.index.search(query_embedding, actual_k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.tool_data):  # Ensure valid index
                tool = self.tool_data[idx].copy()
                tool['relevance_score'] = float(score)
                tool['rank'] = i + 1
                tool['format'] = self.format  # Add format information
                results.append(tool)
            else:
                print(f"Warning: Invalid tool index {idx}, skipping")
        
        # print(f"Successfully retrieved {len(results)} tools")
        return results
    
    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一获取工具信息
        
        Args:
            tool: Tool dictionary
            
        Returns:
            Standardized tool info dictionary
        """
        if self.format == "v2":
            return tool.get("tool_info", {})
        elif self.format == "valid_science":
            return tool.get("description", {})
        else:
            return {}
    
    def get_tool_code(self, tool: Dict[str, Any]) -> str:
        """
        统一获取工具代码
        
        Args:
            tool: Tool dictionary
            
        Returns:
            Tool code string
        """
        if self.format == "v2":
            return tool.get("tool_code", "")
        elif self.format == "valid_science":
            return tool.get("python", "")
        else:
            return ""
    
    def convert_tool_to_openai_format(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert tool data to OpenAI function calling format.
        
        Args:
            tool: Tool dictionary from tool_data
            
        Returns:
            OpenAI function format dictionary
        """
        tool_info = self.get_tool_info(tool)
        
        return {
            "type": "function",
            "function": tool_info.get("function", {})
        }


class UnifiedDynamicToolManager:
    """
    统一的动态工具管理器，支持多种格式
    """
    
    def __init__(self, tool_retriever: Optional[UnifiedToolRetriever] = None, 
                 model_name: Optional[str] = None, api_base: Optional[str] = None):
        """
        Initialize the unified dynamic tool manager.
        
        Args:
            tool_retriever: UnifiedToolRetriever instance for fetching tools
            model_name: LLM model name (optional)
            api_base: API base URL (optional)
        """
        self.tool_retriever = tool_retriever
        self.model_name = model_name
        self.api_base = api_base
        self.available_tools = [TOOL_RETRIEVE_FUNCTION]  # Start with retrieval function
        self.retrieved_tool_names: Set[str] = set()  # Track retrieved tools
        self.tool_name_to_data: Dict[str, Dict] = {}  # Map tool names to full data
        self.tool_name_to_function: Dict[str, callable] = {}  # Map tool names to executable functions
        self.last_retrieval_info: Dict[str, Any] = {}  # Store last retrieval information
        
        # Register the retrieval function
        self.tool_name_to_function["retrieve_relevant_tools"] = self.retrieve_relevant_tools
        
    def _normalize_code_block(self, tool_code: str) -> str:
        """
        Normalize code block by removing markdown fences and prefixes.
        Similar to langchain's code processing approach.
        """
        s = (tool_code or "").lstrip()
        if "```python" in s:
            idx_open = s.find("```python") + 9
            idx_close = s.rfind("```")
            if idx_close != -1 and idx_close > idx_open:
                actual_code = s[idx_open:idx_close].lstrip("\n").rstrip()
            else:
                actual_code = s[idx_open:].lstrip("\n").rstrip()
        elif s.startswith("```"):
            idx_open = s.find("```") + 3
            idx_close = s.rfind("```")
            if idx_close != -1 and idx_close > idx_open:
                actual_code = s[idx_open:idx_close].lstrip("\n").rstrip()
            else:
                actual_code = s[idx_open:].lstrip("\n").rstrip()
        elif s.startswith("python\n"):
            actual_code = s[len("python\n"):].rstrip()
        else:
            actual_code = s
        return actual_code

    def _validate_function_signature(self, func: Callable, kwargs: Dict[str, Any], func_name: str) -> bool:
        """
        Validate function signature compatibility with provided kwargs.
        Inspired by langchain's signature validation approach.
        """
        try:
            sig = inspect.signature(func)
            # Try to bind the arguments to see if they're compatible
            bound_args = sig.bind(**kwargs)
            bound_args.apply_defaults()
            return True
        except (TypeError, ValueError) as e:
            print(f"Signature validation failed for {func_name}: {e}")
            return False

    def _get_function_candidates(self, env: Dict[str, Any], preferred_name: str = "") -> List[tuple]:
        """
        Get function candidates from environment, prioritizing 'execute' and preferred_name.
        Returns list of (name, function, priority) tuples.
        """
        candidates = []
        
        # Priority 1: 'execute' function
        if 'execute' in env and callable(env['execute']) and inspect.isfunction(env['execute']):
            candidates.append(('execute', env['execute'], 1))
        
        # Priority 2: preferred function name from schema
        if preferred_name and preferred_name in env and callable(env[preferred_name]):
            if inspect.isfunction(env[preferred_name]):
                candidates.append((preferred_name, env[preferred_name], 2))
        
        # Priority 3: other public functions (not classes)
        for name, obj in env.items():
            if (not name.startswith('_') and 
                callable(obj) and 
                inspect.isfunction(obj) and 
                name not in ['execute', preferred_name]):
                candidates.append((name, obj, 3))
        
        # Sort by priority (lower number = higher priority)
        candidates.sort(key=lambda x: x[2])
        return candidates

    def create_executable_function(self, tool_data: Dict[str, Any]) -> Optional[callable]:
        """
        Create an executable function from tool data containing Python code.
        Enhanced with langchain-inspired signature validation and function selection.
        
        Args:
            tool_data: Tool data containing tool code with Python implementation
            
        Returns:
            Callable function or None if creation fails
        """
        try:
            tool_code = self.tool_retriever.get_tool_code(tool_data)
            if not tool_code:
                print(f"Warning: No tool code found for tool")
                return None

            # Normalize code block
            actual_code = self._normalize_code_block(tool_code)
            if not actual_code.strip():
                print(f"Warning: Empty code after normalization")
                return None

            # Get preferred function name from schema
            tool_info = self.tool_retriever.get_tool_info(tool_data) if hasattr(self.tool_retriever, 'get_tool_info') else {}
            preferred_name = ""
            try:
                preferred_name = tool_info.get("function", {}).get("name", "") if isinstance(tool_info, dict) else ""
            except Exception:
                preferred_name = ""

            # Pre-validate the code and find suitable function
            def isolated_tool_wrapper(**kwargs):
                import base64, json, inspect
                code_b64 = base64.b64encode((actual_code or "").encode("utf-8")).decode("ascii")
                args_json = json.dumps({"kwargs": kwargs, "preferred_name": preferred_name})
                args_b64 = base64.b64encode(args_json.encode("utf-8")).decode("ascii")

                wrapper_script = (
                    "import base64, json, inspect\n"
                    f"_code = base64.b64decode('{code_b64}').decode('utf-8')\n"
                    f"_args_data = json.loads(base64.b64decode('{args_b64}').decode('utf-8'))\n"
                    "_kwargs = _args_data['kwargs']\n"
                    "_preferred_name = _args_data['preferred_name']\n"
                    "env = {}\n"
                    "try:\n"
                    "    exec(_code, env)\n"
                    "except Exception as e:\n"
                    "    print('ERROR: Code execution failed:', str(e))\n"
                    "    exit()\n"
                    "\n"
                    "# Find function candidates with priority\n"
                    "candidates = []\n"
                    "if 'execute' in env and callable(env['execute']) and inspect.isfunction(env['execute']):\n"
                    "    candidates.append(('execute', env['execute'], 1))\n"
                    "if _preferred_name and _preferred_name in env and callable(env[_preferred_name]) and inspect.isfunction(env[_preferred_name]):\n"
                    "    candidates.append((_preferred_name, env[_preferred_name], 2))\n"
                    "for name, obj in env.items():\n"
                    "    if (not name.startswith('_') and callable(obj) and inspect.isfunction(obj) and \n"
                    "        name not in ['execute', _preferred_name]):\n"
                    "        candidates.append((name, obj, 3))\n"
                    "candidates.sort(key=lambda x: x[2])\n"
                    "\n"
                    "# Try each candidate with signature validation\n"
                    "selected_func = None\n"
                    "selected_name = None\n"
                    "for name, func, priority in candidates:\n"
                    "    try:\n"
                    "        sig = inspect.signature(func)\n"
                    "        bound_args = sig.bind(**_kwargs)\n"
                    "        bound_args.apply_defaults()\n"
                    "        selected_func = func\n"
                    "        selected_name = name\n"
                    "        break\n"
                    "    except (TypeError, ValueError) as e:\n"
                    "        continue\n"
                    "\n"
                    "if not selected_func:\n"
                    "    available_funcs = [(name, str(inspect.signature(func)) if inspect.isfunction(func) else 'not_function') \n"
                    "                      for name, func, _ in candidates]\n"
                    "    available_classes = [(name, 'class') for name, obj in env.items() \n"
                    "                        if not name.startswith('_') and inspect.isclass(obj)]\n"
                    "    print('ERROR: No compatible function found')\n"
                    "    print('Available functions:', available_funcs)\n"
                    "    print('Available classes:', available_classes)\n"
                    "    print('Required kwargs:', list(_kwargs.keys()))\n"
                    "else:\n"
                    "    try:\n"
                    "        result = selected_func(**_kwargs)\n"
                    "        print('RESULT:', result)\n"
                    "        print('FUNCTION_USED:', selected_name)\n"
                    "    except Exception as e:\n"
                    "        print('ERROR: Function execution failed:', str(e))\n"
                    "        print('FUNCTION_USED:', selected_name)\n"
                    "        print('FUNCTION_SIGNATURE:', str(inspect.signature(selected_func)))\n"
                )

                exec_result = execute_code(wrapper_script, timeout=30)
                
                # Parse results
                if "RESULT:" in exec_result:
                    result_line = [line for line in exec_result.split('\n') if line.startswith('RESULT:')][0]
                    return result_line[7:].strip()
                elif "ERROR:" in exec_result:
                    error_lines = [line for line in exec_result.split('\n') if line.startswith('ERROR:')]
                    raise Exception('\n'.join(error_lines))
                else:
                    raise Exception(f"Unexpected execution result: {exec_result}")

            return isolated_tool_wrapper

        except Exception as e:
            print(f"Failed to create executable function: {e}")
            return None 

    def retrieve_relevant_tools(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        OpenAI function implementation for retrieving relevant tools.
        Enhanced version that adds executable tools to the function registry.
        
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
                "tools": []
            }
        
        try:
            # Retrieve relevant tools using the search query
            retrieved_tools = self.tool_retriever.retrieve_top_k(query, k)
            
            # Store retrieval information before processing
            self.last_retrieval_info = {
                "original_query": query,
                "requested_k": k,
                "retrieved_count": len(retrieved_tools),
                "format": self.tool_retriever.format
            }
            
            # Convert to OpenAI format and add to available tools
            new_tools = []
            for tool in retrieved_tools:
                tool_info = self.tool_retriever.get_tool_info(tool)
                tool_name = tool_info.get("function", {}).get("name", "")
                
                # Avoid duplicates
                if tool_name not in self.retrieved_tool_names:
                    # Add to OpenAI format tools
                    openai_tool = self.tool_retriever.convert_tool_to_openai_format(tool)
                    self.available_tools.append(openai_tool)
                    self.retrieved_tool_names.add(tool_name)
                    self.tool_name_to_data[tool_name] = tool
                    
                    # Create executable function
                    executable_func = self.create_executable_function(tool)
                    if executable_func:
                        self.tool_name_to_function[tool_name] = executable_func
                        # print(f"✅ Successfully created executable function for {tool_name}")
                    else:
                        # Create a dummy function that returns error message
                        def dummy_func(**kwargs):
                            return f"Error: Could not execute tool {tool_name}. Tool code execution failed."
                        self.tool_name_to_function[tool_name] = dummy_func
                        # print(f"⚠️  Created dummy function for {tool_name} (execution failed)")
                    
                    new_tools.append({
                        "name": tool_name,
                        "description": tool_info.get("function", {}).get("description", ""),
                        "relevance_score": tool.get("relevance_score", 0.0),
                        "executable": executable_func is not None,
                        "format": tool.get("format", "unknown")
                    })
            
            # Update retrieval info with final results
            self.last_retrieval_info.update({
                "new_tools_added": len(new_tools),
                "total_available_tools": len(self.available_tools)
            })
            
            success_msg = f"Successfully retrieved {len(new_tools)} new tools. Total available: {len(self.available_tools)}"
            if new_tools:
                success_msg += f"\nRetrieved tools: {', '.join([t['name'] for t in new_tools])}"
            
            return {
                "success": True,
                "message": success_msg,
                "tools": new_tools,
                "original_query": query,
                "requested_k": k,
                "actual_retrieved": len(retrieved_tools),
                "format": self.tool_retriever.format
            }
            
        except Exception as e:
            error_msg = f"Error retrieving tools: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "message": error_msg,
                "tools": []
            }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all currently available tools in OpenAI format."""
        return self.available_tools
    
    def get_function_registry(self) -> Dict[str, callable]:
        """Get the complete function registry for tool execution."""
        return self.tool_name_to_function
    
    def reset_tool_pool(self):
        """Reset the tool pool to initial state."""
        self.available_tools = [TOOL_RETRIEVE_FUNCTION]
        self.retrieved_tool_names.clear()
        self.tool_name_to_data.clear()
        # Keep the retrieval function in the registry
        self.tool_name_to_function = {"retrieve_relevant_tools": self.retrieve_relevant_tools}


class GPT41Client:
    """
    GPT-4.1 客户端，支持动态工具调用
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4-1106-preview"):
        """
        Initialize GPT-4.1 client
        
        Args:
            api_key: OpenAI API key
            model_name: GPT-4.1 model name
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name

    def call(self, messages: List[Dict] = None) -> str:
        """
        Call the GPT-4.1 model.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content
        
    def call_with_dynamic_tools(self, messages: List[Dict], tool_manager: UnifiedDynamicToolManager, 
                               max_turns: int = 10) -> tuple:
        """
        Call GPT-4.1 with dynamic tool retrieval and execution capability.
        
        Args:
            messages: Initial messages
            tool_manager: UnifiedDynamicToolManager instance
            max_turns: Maximum conversation turns
            
        Returns:
            Tuple of (final_messages, total_turns)
        """
        current_messages = messages.copy()
        turn_count = 0
        
        while turn_count < max_turns:
            turn_count += 1
            # print(f"\n🔄 Turn {turn_count}/{max_turns}")
            
            # Get current available tools
            available_tools = tool_manager.get_available_tools()
            function_registry = tool_manager.get_function_registry()
            
            # print(f"Available tools: {len(available_tools)}")
            # for tool in available_tools:
            #     if tool["type"] == "function":
            #         print(f"  - {tool['function']['name']}")
            
            # Call GPT-4.1 with current tools
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=current_messages,
                    tools=available_tools,
                    tool_choice="auto",
                    temperature=0.1
                )
                
                message = response.choices[0].message
                current_messages.append(message.model_dump())
                
                # Check if there are tool calls
                if message.tool_calls:
                    # Handle each tool call
                    for tool_call in message.tool_calls:
                        call_id = tool_call.id
                        function_name = tool_call.function.name
                        
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            # print(f"🔧 Executing tool: {function_name} with args {function_args}")
                            
                            # Execute the function
                            if function_name in function_registry:
                                try:
                                    result = function_registry[function_name](**function_args)
                                    result_str = json.dumps(result) if not isinstance(result, str) else result
                                    # print(f"✅ Tool execution successful: {function_name}")
                                except Exception as e:
                                    result_str = f"Error executing tool '{function_name}': {str(e)}\n{traceback.format_exc()}"
                                    # print(f"❌ Tool execution failed: {function_name}: {str(e)}")
                            else:
                                result_str = f"Function {function_name} not found"
                                # print(f"❌ Function not found: {function_name}")

                        except json.JSONDecodeError as e:
                            result_str = f"Error: Invalid arguments for {function_name}. The provided arguments are not a valid JSON object. Please check the format. Error: {e}. Arguments received: '{tool_call.function.arguments}'"
                            # print(f"❌ JSONDecodeError for tool {function_name}: {e}")
                        
                        # Add tool result to messages
                        current_messages.append({
                            "role": "tool",
                            "content": result_str,
                            "tool_call_id": call_id
                        })
                    
                    # Continue to next turn
                    continue
                else:
                    # No tool calls, check if we have a final answer
                    content = message.content or ""
                    if "final answer:" in content.lower():
                        # print(f"✅ Got final answer in turn {turn_count}")
                        break
                    elif turn_count >= max_turns:
                        # print(f"⏰ Reached max turns ({max_turns})")
                        break
                    else:
                        # Continue the conversation
                        continue
                        
            except Exception as e:
                # print(f"❌ Error in turn {turn_count}: {e}")
                # print(f"Traceback: {traceback.format_exc()}")
                
                # Add error message to conversation and continue
                current_messages.append({
                    "role": "assistant", 
                    "content": f"I encountered an error: {str(e)}"
                })
                
                # Continue to next turn instead of breaking
                continue
        
        # If max turns are reached, force a final answer
        if turn_count >= max_turns:
            # print("Max turns reached. Forcing a final answer...")
            force_final_answer_prompt = {
                "role": "user",
                "content": "You have reached the maximum number of turns. Based on the conversation history, please provide the final answer now. Your last line must start with 'Final Answer:'."
            }
            current_messages.append(force_final_answer_prompt)
            
            try:
                # Call one last time without tools
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=current_messages,
                    temperature=0.1
                )
                message = response.choices[0].message
                current_messages.append(message.model_dump())
                turn_count += 1
                # print("✅ Final answer received.")
            except Exception as e:
                print(f"❌ Error while forcing final answer: {e}")
                current_messages.append({
                    "role": "assistant",
                    "content": f"An error occurred while forcing the final answer: {str(e)}"
                })

        return current_messages, turn_count


class VLLMClient:
    """
    vLLM OpenAI-compatible 客户端，支持动态工具调用
    - 使用 base_url 指向 vLLM 的 OpenAI 兼容服务，例如: http://localhost:8000/v1
    - model_name 可以直接传入本地模型路径（例如: /export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/）
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", model_name: str = "/export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/", api_key: Optional[str] = None):
        """
        Initialize vLLM client
        
        Args:
            base_url: vLLM OpenAI-compatible server base URL
            model_name: Model identifier or local model path used by vLLM
            api_key: API key (vLLM 通常不校验，可留空；此处默认使用环境变量或 "EMPTY")
        """
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            base_url=base_url
        )
        self.model_name = model_name
    
    def _message_to_dict(self, message: Any) -> Dict[str, Any]:
        """兼容地将 SDK 返回的 message 转为 dict。"""
        try:
            return message.model_dump()
        except Exception:
            result = {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", "")
            }
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is not None:
                result["tool_calls"] = tool_calls
            return result
    
    def call(self, messages: List[Dict], return_format: str = "content") -> str:
        """
        Call the vLLM model.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    
    def call_with_dynamic_tools(self, messages: List[Dict], tool_manager: UnifiedDynamicToolManager, 
                               max_turns: int = 10) -> tuple:
        """
        使用 vLLM 的 OpenAI 兼容接口进行带动态工具的多轮对话。
        """
        current_messages = messages.copy()
        turn_count = 0
        
        while turn_count < max_turns:
            turn_count += 1
            
            # Get current available tools
            available_tools = tool_manager.get_available_tools()
            function_registry = tool_manager.get_function_registry()
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=current_messages,
                    tools=available_tools,
                    tool_choice="auto",
                    temperature=0.1
                )
                
                message = response.choices[0].message
                current_messages.append(self._message_to_dict(message))
                
                # Check if there are tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        call_id = tool_call.id
                        function_name = tool_call.function.name
                        
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            
                            if function_name in function_registry:
                                try:
                                    result = function_registry[function_name](**function_args)
                                    result_str = json.dumps(result) if not isinstance(result, str) else result
                                except Exception as e:
                                    result_str = f"Error executing tool '{function_name}': {str(e)}\n{traceback.format_exc()}"
                            else:
                                result_str = f"Function {function_name} not found"
                        except json.JSONDecodeError as e:
                            result_str = f"Error: Invalid arguments for {function_name}. The provided arguments are not a valid JSON object. Please check the format. Error: {e}. Arguments received: '{tool_call.function.arguments}'"
                        
                        # Add tool result to messages
                        current_messages.append({
                            "role": "tool",
                            "content": result_str,
                            "tool_call_id": call_id
                        })
                    
                    # Continue to next turn
                    continue
                else:
                    content = message.content or ""
                    if "final answer:" in content.lower():
                        break
                    elif turn_count >= max_turns:
                        break
                    else:
                        continue
                        
            except Exception as e:
                # print(f"❌ Error in turn {turn_count}: {e}")
                # print(f"Traceback: {traceback.format_exc()}")
                current_messages.append({
                    "role": "assistant", 
                    "content": f"I encountered an error: {str(e)}"
                })
                continue
        
        # If max turns are reached, force a final answer
        if turn_count >= max_turns:
            force_final_answer_prompt = {
                "role": "user",
                "content": "You have reached the maximum number of turns. Based on the conversation history, please provide the final answer now. Your last line must start with 'Final Answer:'."
            }
            current_messages.append(force_final_answer_prompt)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=current_messages,
                    temperature=0.1
                )
                message = response.choices[0].message
                current_messages.append(self._message_to_dict(message))
                turn_count += 1
            except Exception as e:
                print(f"❌ Error while forcing final answer: {e}")
                current_messages.append({
                    "role": "assistant",
                    "content": f"An error occurred while forcing the final answer: {str(e)}"
                })
        
        return current_messages, turn_count

def get_unified_tool_embedding(tool_data, tool_embedding_path):
    """
    Generate embeddings for tool data with format auto-detection.
    
    Args:
        tool_data: List of tool dictionaries
        tool_embedding_path: Path to save the embedding file
    
    Returns:
        numpy.ndarray: Array of tool embeddings
    """
    format_type = ToolFormatDetector.detect_format(tool_data)
    # print(f"🔍 Detected format: {format_type}")
    
    tool_embedding = []
    batch_size = 500
    # print(f"Getting tool embedding for {len(tool_data)} tools...")
    
    failed_indices = []
    
    for i in range(0, len(tool_data), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(tool_data) + batch_size - 1)//batch_size} ({i}/{len(tool_data)})")
        batch_data = tool_data[i:i+batch_size]
        
        # Create batch text descriptions with format-aware processing
        batch_data_str = []
        for j, d in enumerate(batch_data):
            try:
                if isinstance(d, dict):
                    # Handle v_2.json format
                    if format_type == "v2" and 'tool_info' in d:
                        tool_info = d['tool_info']
                        if 'function' in tool_info and isinstance(tool_info['function'], dict):
                            func = tool_info['function']
                            name = func.get('name', f'tool_{i+j}')
                            description = func.get('description', 'No description available')
                            text = f"The function name is: {name}. The function description is: {description}."
                            batch_data_str.append(text)
                        else:
                            batch_data_str.append(f"Tool {i+j}: Invalid tool_info function format")
                    # Handle valid_science_toolset format
                    elif format_type == "valid_science" and 'description' in d:
                        desc = d['description']
                        if 'function' in desc and isinstance(desc['function'], dict):
                            func = desc['function']
                            name = func.get('name', f'tool_{i+j}')
                            description = func.get('description', 'No description available')
                            text = f"The function name is: {name}. The function description is: {description}."
                            batch_data_str.append(text)
                        else:
                            batch_data_str.append(f"Tool {i+j}: Invalid description format")
                    else:
                        batch_data_str.append(f"Tool {i+j}: Invalid tool format for detected format {format_type}")
                else:
                    batch_data_str.append(f"Tool {i+j}: Invalid tool format - not a dictionary")
            except Exception as e:
                print(f"Error processing tool {i+j}: {e}")
                batch_data_str.append(f"Tool {i+j}: Error in processing")
        
        # Call embedding API with error handling
        try:
            batch_embedding = call_sfr_embedding_api_lst(batch_data_str)
            
            if batch_embedding is None:
                print(f"Warning: API returned None for batch {i//batch_size + 1}")
                # Create zero vectors as fallback
                embedding_dim = 1024
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
    
    # Convert to numpy array
    tool_embedding = np.array(tool_embedding)
    print(f"Final embedding shape: {tool_embedding.shape}")
    
    # Save embeddings
    save_pkl(data=tool_embedding, file_path=tool_embedding_path)
    print(f"Embeddings saved to: {tool_embedding_path}")
    
    return tool_embedding


EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


def llm_number_judge(pred, answer):
    prompt = EQUALITY_TEMPLATE.format(expression1=pred, expression2=answer)
    response = call_openai_api(model_name="gpt-4o-mini",content=prompt)
    return response.strip() == "Yes"


def eval_response(response, answer, type="multiple_choice"):
    if "Final Answer" in response:
        pred_answer = response.split("Final Answer")[-1].strip()
        if "boxed" in pred_answer:
            pred_answer = pred_answer.split("boxed{")[1].strip()
        # Extract only the first letter/character that appears to be an answer choice
        import re
        if type == "multiple_choice":
            match = re.search(r'[A-J]', pred_answer)
        elif type == "number":
            if llm_number_judge(pred_answer, answer):
                correctness = 1
            else:
                correctness = 0
            return correctness, pred_answer
        elif match:
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

def create_unified_tool_system(tool_data: List[Dict], tool_embedding: np.ndarray) -> UnifiedDynamicToolManager:
    """
    Factory function to create a unified tool system.
    
    Args:
        tool_data: List of tool dictionaries
        tool_embedding: Pre-computed embeddings for tools
        
    Returns:
        Configured UnifiedDynamicToolManager instance
    """
    retriever = UnifiedToolRetriever(tool_data, tool_embedding)
    manager = UnifiedDynamicToolManager(retriever)
    return manager


def extract_ground_truth_tools(extracted_art_info):
    """Extract ground truth tool names from extracted_art_info."""
    if not extracted_art_info:
        return []
    
    ground_truth_tools = []
    for tool_info in extracted_art_info:
        if isinstance(tool_info, dict) and "function_name" in tool_info:
            ground_truth_tools.append(tool_info["function_name"])
    
    return ground_truth_tools


def calculate_tool_recall(retrieved_tools, ground_truth_tools):
    """Calculate recall for retrieved tools against ground truth tools."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Dynamic Tool System with GPT-4.1")
    parser.add_argument("--input_data_path", type=str, default="/Users/murong.yue/Desktop/data/dev_physics_322.json")
    parser.add_argument("--model_nickname", type=str, default="GPT41")
    parser.add_argument("--gpt_model_name", type=str, default="gpt-4.1", help="GPT-4.1 model name")
    
    # Tool data paths - can be either format
    parser.add_argument("--tool_path", type=str, 
                       default="/Users/murong.yue/Desktop/data/valid_science_toolset_for_hard_examples_tagged_tools_20250619_054608.json",
                       help="Path to tool data (supports both v_2.json and valid_science_toolset formats)")
    parser.add_argument("--tool_embedding_path", type=str, 
                       default="/Users/murong.yue/Desktop/data/all_tagged_tools_20250619_054608_embedding.pkl",
                       help="Path to tool embeddings")
    
    # Alternative tool data paths
    parser.add_argument("--alt_tool_path", type=str, 
                       default="/Users/murong.yue/Desktop/data/v_2.json",
                       help="Alternative tool data path")
    parser.add_argument("--alt_tool_embedding_path", type=str, 
                       default="/Users/murong.yue/Desktop/data/v_2_embedding.pkl",
                       help="Alternative tool embedding path")
    
    parser.add_argument("--save_path", type=str, default="/Users/murong.yue/Desktop/data/unified_dynamic_tools_gpt41.json")
    parser.add_argument("--save_step", type=int, default=500)
    parser.add_argument("--passk", type=int, default=1)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--enable_tool_retrieval", action='store_true', default=True)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--use_alt_tools", action='store_true', help="Use alternative tool data (v_2.json)")
    
    args = parser.parse_args()
    
    # Select tool data based on flag
    if args.use_alt_tools:
        tool_path = args.alt_tool_path
        tool_embedding_path = args.alt_tool_embedding_path
        print(f"📁 Using alternative tool data: {tool_path}")
    else:
        tool_path = args.tool_path
        tool_embedding_path = args.tool_embedding_path
        print(f"📁 Using primary tool data: {tool_path}")
    
    save_path = args.save_path.replace(".json", f"_{current_time}.json")
    data = read_json(args.input_data_path)
    
    if args.debug:
        data = data[:100]  # Small sample for testing
        save_path = save_path.replace(".json", "_debug.json")
    
    results_lst = []
    tool_data = read_json(tool_path)
    
    print(f"📊 Loaded {len(tool_data)} tools from {tool_path}")
    
    # Load or generate embeddings
    if os.path.exists(tool_embedding_path):
        tool_embedding = read_pkl(tool_embedding_path)
        print(f"📊 Loaded embeddings from {tool_embedding_path}")
    else:
        print(f"🔄 Generating embeddings for {len(tool_data)} tools...")
        tool_embedding = get_unified_tool_embedding(tool_data, tool_embedding_path)
    
    assert len(tool_embedding) == len(tool_data), f"Tool embedding length {len(tool_embedding)} != tool data length {len(tool_data)}"
    
    # Create unified tool system
    tool_manager = None
    gpt_client = None
    
    if args.enable_tool_retrieval:
        print("🔧 Initializing unified tool system...")
        tool_manager = create_unified_tool_system(tool_data, tool_embedding)
        print(f"✅ Unified tool system initialized with {len(tool_data)} tools")
        
        # Initialize GPT-4.1 client
        gpt_client = GPT41Client(OPENAI_API_KEY, args.gpt_model_name)
        print(f"✅ GPT-4.1 client initialized with model: {args.gpt_model_name}")
        
        # Demonstrate tool retrieval functionality
        print("\n=== 🧪 Unified Tool Retrieval Demo ===")
        demo_query = "calculate molecular orbital energy"
        demo_result = tool_manager.retrieve_relevant_tools(demo_query, k=3)
        print(f"Demo query: '{demo_query}'")
        print(f"Result: {demo_result['message']}")
        print(f"Format detected: {demo_result.get('format', 'unknown')}")
        if demo_result['success']:
            for tool in demo_result['tools']:
                print(f"  - {tool['name']}: {tool['description'][:100]}... (score: {tool['relevance_score']:.3f}, executable: {tool['executable']}, format: {tool['format']})")
        print("=== End Demo ===\n")
        
        # Reset tool manager after demo
        tool_manager.reset_tool_pool()
        print("🔄 Tool manager reset after demo")
    
    def fn(d):
        global results_lst
        # Create a separate tool manager instance for each thread
        local_tool_manager = None
        local_gpt_client = None
        
        if args.enable_tool_retrieval:
            local_tool_manager = create_unified_tool_system(tool_data, tool_embedding)
            local_gpt_client = GPT41Client(OPENAI_API_KEY, args.gpt_model_name)
        
        try:
            for i in range(args.passk):
                id = str(d["id"]) + f"_{i}"
                question = d["question"]
                answer = d["answer"]
                prompt = QUESTION_TEMPLATE_FLATTEN_RETRIEVAL.format(question=question)
                
                print(f"\n🔍 Processing question {id}: {question[:80]}...")
                
                # Prepare initial messages
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                # Extract ground truth tools for recall calculation
                ground_truth_tools = extract_ground_truth_tools(d.get("extracted_art_info", []))
                
                if local_tool_manager and local_gpt_client:
                    # Start fresh for each question
                    local_tool_manager.reset_tool_pool()
                    
                    # Use GPT-4.1 with dynamic tool calling
                    print(f"🚀 Starting GPT-4.1 dynamic tool conversation...")
                    final_messages, total_turns = local_gpt_client.call_with_dynamic_tools(
                        messages=messages,
                        tool_manager=local_tool_manager,
                        max_turns=args.max_turns
                    )
                    
                    # Extract final response
                    reasoning_content = ""
                    response_content = ""
                    
                    # Find the last assistant message
                    for msg in reversed(final_messages):
                        if msg.get("role") == "assistant":
                            response_content = msg.get("content", "")
                            break
                    
                    # Calculate recall metrics
                    retrieved_tool_names = list(local_tool_manager.retrieved_tool_names)
                    recall_metrics = calculate_tool_recall(retrieved_tool_names, ground_truth_tools)
                    
                    # Save complete conversation and tool usage information
                    d[f"{args.model_nickname}_full_messages"] = final_messages
                    d[f"{args.model_nickname}_conversation_turns"] = total_turns
                    d[f"{args.model_nickname}_retrieved_tools"] = retrieved_tool_names
                    d[f"{args.model_nickname}_available_tools_count"] = len(local_tool_manager.get_available_tools())
                    d[f"{args.model_nickname}_ground_truth_tools"] = ground_truth_tools
                    d[f"{args.model_nickname}_tool_recall"] = recall_metrics['recall']
                    d[f"{args.model_nickname}_tool_recall_details"] = recall_metrics
                    d[f"{args.model_nickname}_tool_format"] = local_tool_manager.tool_retriever.format
                    
                    print(f"✅ Completed {total_turns} turns")
                    print(f"🔧 Used {len(retrieved_tool_names)} tools: {retrieved_tool_names}")
                    print(f"📊 Ground truth tools: {ground_truth_tools}")
                    print(f"📈 Tool recall: {recall_metrics['recall']:.3f}")
                    print(f"🔍 Tool format: {local_tool_manager.tool_retriever.format}")
                    
                else:
                    # Fallback: direct GPT-4.1 call without tools
                    client = openai.OpenAI(api_key=OPENAI_API_KEY)
                    response = client.chat.completions.create(
                        model=args.gpt_model_name,
                        messages=messages,
                        temperature=0.1
                    )
                    
                    response_content = response.choices[0].message.content
                    
                    # Save simple response
                    d[f"{args.model_nickname}_full_messages"] = messages + [
                        {"role": "assistant", "content": response_content}
                    ]
                    d[f"{args.model_nickname}_conversation_turns"] = 1
                    d[f"{args.model_nickname}_retrieved_tools"] = []
                    d[f"{args.model_nickname}_available_tools_count"] = 0
                    d[f"{args.model_nickname}_ground_truth_tools"] = ground_truth_tools
                    d[f"{args.model_nickname}_tool_recall"] = 0.0
                    d[f"{args.model_nickname}_tool_recall_details"] = {}
                    d[f"{args.model_nickname}_tool_format"] = "none"
                
                correctness, pred_answer = eval_response(response_content, answer)
                
                d[f"{args.model_nickname}_response"] = response_content
                d[f"{args.model_nickname}_correctness"] = correctness
                d[f"{args.model_nickname}_pred_answer"] = pred_answer
                d["id"] = id
                
                results_lst.append(d)
                if len(results_lst) % args.save_step == 0:
                    save_json(data=results_lst, file_path=save_path)
                    print(f"💾 Saved {len(results_lst)} results to {save_path}")
                    
        except Exception as e:
            print(f"❌ Error processing item {d.get('id', 'unknown')}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
    
    # Process data with threading
    map_with_progress(f=fn, xs=data, num_threads=5)  # Reduced threads for API rate limits
    save_json(data=results_lst, file_path=save_path)
    
    if tool_manager:
        print(f"\n📊 Final tool retrieval stats:")
        print(f"  Total tools in database: {len(tool_data)}")
        print(f"  Tool format: {tool_manager.tool_retriever.format}")
        print(f"  Tools retrieved across all questions: {len(tool_manager.retrieved_tool_names)}")
        print(f"  Retrieved tool names: {list(tool_manager.retrieved_tool_names)}")
    
    # Generate comprehensive evaluation report
    stats_dict, report_path = quick_report(
        results_lst=results_lst,
        model_nickname=args.model_nickname,
        passk=args.passk,
        original_count=len(data),
        save_path=save_path,
        current_time=current_time,
        tool_retrieval_enabled=args.enable_tool_retrieval
    )
    
    print(f"\n🎉 Unified Dynamic Tool System Evaluation Complete!")
    print(f"📊 Results saved to: {save_path}")
    print(f"📈 Statistics: {save_path.replace('.json', '_stats.json')}")
    print(f"📝 Report: {report_path}")
    print(f"🤖 Model: {args.gpt_model_name}")
    print(f"🔧 Tool format: {tool_manager.tool_retriever.format if tool_manager else 'none'}") 