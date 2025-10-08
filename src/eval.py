#!/usr/bin/env python3
from ast import parse
import json
import sys
import os
import datetime
from collections import defaultdict
import numpy as np
import openai
import time
import argparse
import threading
import inspect
from prompt import TAR_FINAL_ANSWER_PROMPT,CoT_FINAL_ANSWER_PROMPT
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass
from utils import read_json, save_json, map_with_progress, call_openai_api, read_pkl, call_llm_with_python_interpreter, LLMCaller
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Import necessary classes and functions from unified_dynamic_tools.py
from unified_dynamic_tools import (
    UnifiedToolRetriever, 
    UnifiedDynamicToolManager, 
    GPT41Client,
    VLLMClient,
    create_unified_tool_system,
    eval_response,
    get_unified_tool_embedding
)

def read_python_code(file_path):
    with open(file_path, 'r') as file:
        return file.read()


@dataclass
class ToolInfo:
    """Tool information data class"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    required: List[str]
    original_tool_data: Dict[str, Any] = None


class StaticToolRegistry:
    """Static tool registry - preload all tool functions"""
    
    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}
        self.function_registry: Dict[str, Callable] = {}
        self.tool_data_map: Dict[str, Dict[str, Any]] = {}
        
    def _normalize_code_block(self, tool_code: str) -> str:
        """Normalize code block, remove markdown fences"""
        if not tool_code:
            return ""
        
        s = tool_code.lstrip()
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
    
    def _create_function_from_code(self, tool_code: str, tool_name: str) -> Optional[Callable]:
        """Create executable function from code"""
        try:
            actual_code = self._normalize_code_block(tool_code)
            if not actual_code.strip():
                return None
            
            # Create execution environment
            exec_globals = {
                '__builtins__': __builtins__,
                'math': __import__('math'),
                'numpy': __import__('numpy'),
                'np': __import__('numpy'),
                'json': __import__('json'),
                'os': __import__('os'),
                'sys': __import__('sys'),
                'datetime': __import__('datetime'),
                'time': __import__('time'),
                'random': __import__('random'),
                'itertools': __import__('itertools'),
                'collections': __import__('collections'),
                'functools': __import__('functools'),
                'typing': __import__('typing'),
            }
            
            # Try to import common scientific computing libraries
            try:
                exec_globals['scipy'] = __import__('scipy')
                exec_globals['sympy'] = __import__('sympy')
                exec_globals['pandas'] = __import__('pandas')
                exec_globals['pd'] = __import__('pandas')
            except ImportError:
                pass
            
            # Execute code
            exec(actual_code, exec_globals)
            
            # Find execute function
            if 'execute' in exec_globals and callable(exec_globals['execute']):
                return exec_globals['execute']
            
            # If no execute function, find other public functions
            for name, obj in exec_globals.items():
                if (not name.startswith('_') and 
                    callable(obj) and 
                    inspect.isfunction(obj) and
                    name not in ['math', 'numpy', 'json', 'os', 'sys', 'datetime', 'time', 'random', 'itertools', 'collections', 'functools', 'typing']):
                    return obj
            
            return None
            
        except Exception as e:
            # print(f"❌ Failed to create function for {tool_name}: {e}")
            return None
    
    def register_tool_from_data(self, tool_data: Dict[str, Any]) -> bool:
        """Register tools from tool data"""
        try:
            # Get tool information
            if "tool_info" in tool_data:
                tool_info = tool_data["tool_info"]
                tool_code = tool_data.get("tool_code", "")
            elif "description" in tool_data:
                tool_info = tool_data["description"]
                tool_code = tool_data.get("python", "")
            else:
                return False
            
            if not isinstance(tool_info, dict) or "function" not in tool_info:
                return False
            
            func_info = tool_info["function"]
            tool_name = func_info.get("name", "")
            tool_desc = func_info.get("description", "")
            
            if not tool_name:
                return False
            
            # Create function
            func = self._create_function_from_code(tool_code, tool_name)
            if func is None:
                return False
            
            # Get parameter information
            parameters = func_info.get("parameters", {}).get("properties", {})
            required = func_info.get("parameters", {}).get("required", [])
            
            # Create tool information
            tool_info_obj = ToolInfo(
                name=tool_name,
                description=tool_desc,
                function=func,
                parameters=parameters,
                required=required,
                original_tool_data=tool_data
            )
            
            # Register tool
            self.tools[tool_name] = tool_info_obj
            self.function_registry[tool_name] = func
            self.tool_data_map[tool_name] = tool_data
            
            return True
            
        except Exception as e:
            return False
    
    def load_all_tools(self, tool_data_list: List[Dict[str, Any]]) -> int:
        """Load all tool data"""
        success_count = 0
        for tool_data in tool_data_list:
            if self.register_tool_from_data(tool_data):
                success_count += 1
        
      #   print(f"✅ Successfully registered {success_count}/{len(tool_data_list)} tools")
        return success_count
    
    def get_openai_tools(self, tool_names: List[str] = None) -> List[Dict[str, Any]]:
        """Get OpenAI format tool list"""
        if tool_names is None:
            tool_names = list(self.tools.keys())
        
        openai_tools = []
        for tool_name in tool_names:
            if tool_name in self.tools:
                tool_info = self.tools[tool_name]
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool_info.name,
                        "description": tool_info.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool_info.parameters,
                            "required": tool_info.required
                        }
                    }
                }
                openai_tools.append(openai_tool)
        return openai_tools
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute tool function"""
        if name not in self.function_registry:
            available_tools = list(self.tools.keys())
            raise ValueError(f"Tool '{name}' not found. Available tools: {available_tools[:10]}{'...' if len(available_tools) > 10 else ''}")
        
        func = self.function_registry[name]
        
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(**kwargs)
            bound_args.apply_defaults()
            result = func(**kwargs)
            return result
            
        except TypeError as e:
            tool_info = self.tools[name]
            raise TypeError(f"Tool '{name}' parameter error: {e}. "
                          f"Required parameters: {tool_info.required}")
    
    def get_function_registry(self) -> Dict[str, Callable]:
        """Get function registry"""
        return self.function_registry.copy()
    
    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self.tools.keys())


class StaticToolManager:
    """Static tool manager - combines retrieval and static function calls"""
    
    def __init__(self, tool_registry: StaticToolRegistry, tool_retriever: UnifiedToolRetriever):
        self.tool_registry = tool_registry
        self.tool_retriever = tool_retriever
        self.retrieved_tool_names: set = set()
        self.available_tools: List[Dict[str, Any]] = []
        
    def reset_tool_pool(self):
        """Reset tool pool"""
        self.retrieved_tool_names.clear()
        self.available_tools.clear()
    
    def retrieve_relevant_tools(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant tools and add to available tool list"""
        try:
            # Use retriever to get relevant tools
            retrieved_tools = self.tool_retriever.retrieve_top_k(query, k)
            
            new_tools = []
            for tool in retrieved_tools:
                tool_info = self.tool_retriever.get_tool_info(tool)
                tool_name = tool_info.get("function", {}).get("name", "")
                
                # Check if tool is already registered
                if tool_name in self.tool_registry.tools and tool_name not in self.retrieved_tool_names:
                    # Add to available tools
                    openai_tool = self.tool_registry.get_openai_tools([tool_name])[0]
                    self.available_tools.append(openai_tool)
                    self.retrieved_tool_names.add(tool_name)
                    
                    new_tools.append({
                        "name": tool_name,
                        "description": tool_info.get("function", {}).get("description", ""),
                        "relevance_score": tool.get("relevance_score", 0.0),
                        "executable": True,
                        "format": tool.get("format", "unknown")
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
            return {
                "success": False,
                "message": error_msg,
                "tools": []
            }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get current available tool list"""
        # Add retrieval tool
        retrieval_tool = {
            "type": "function",
            "function": {
                "name": "retrieve_relevant_tools",
                "description": "Search and retrieve the most relevant tools based on user query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user query or task description to search for relevant tools."
                        },
                        "k": {
                            "type": "integer",
                            "default": 5,
                            "description": "Number of relevant tools to retrieve.",
                            "minimum": 1,
                            "maximum": 20
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        return [retrieval_tool] + self.available_tools
    
    def get_function_registry(self) -> Dict[str, Callable]:
        """Get function registry"""
        registry = {"retrieve_relevant_tools": self.retrieve_relevant_tools}
        registry.update(self.tool_registry.get_function_registry())
        return registry



def format_messages_for_reading(messages):
    """
    Convert OpenAI messages to a more readable format with role icons and clear tool call formatting.
    """
    if not messages:
        return "No messages to display."
        
    formatted_text = []
    
    role_icons = {
        "user": "👤",
        "assistant": "🤖", 
        "tool": "🔧"
    }
    
    for i, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        icon = role_icons.get(role, "❓")
        
        formatted_text.append(f"\n{'='*60}")
        formatted_text.append(f"Message {i+1}: {icon} {role.upper()}")
        formatted_text.append('='*60)
        
        # Handle regular content
        if content:
            formatted_text.append(f"Content:\n{content}")
        
        # Handle tool calls (assistant making tool calls)
        if "tool_calls" in message:
            formatted_text.append("\n🔧 TOOL CALLS:")
            for j, tool_call in enumerate(message["tool_calls"]):
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", "{}")
                tool_id = tool_call.get("id", "unknown")
                
                formatted_text.append(f"  Tool Call {j+1}:")
                formatted_text.append(f"    ID: {tool_id}")
                formatted_text.append(f"    Function: {tool_name}")
                formatted_text.append(f"    Arguments: {tool_args}")
        
        # Handle tool call results (tool responding)
        if role == "tool":
            tool_call_id = message.get("tool_call_id", "unknown")
            formatted_text.append(f"Tool Call ID: {tool_call_id}")
            if content:
                formatted_text.append(f"Tool Result:\n{content}")
        
        formatted_text.append("")  # Add blank line for separation
    
    return "\n".join(formatted_text)


def CoT_eval(question_data, model_name):
    correct_count = 0
    log_dir = args.log_dir
    eval_type = args.eval_type
    if "gpt" in model_name:
        model_name_to_record = model_name
    else:
        model_name_to_record = model_name.split('/')[-2]
    os.makedirs(log_dir, exist_ok=True)
    if args.debug:
        debug_suffix = "_debug"
    else:
        debug_suffix = ""
    log_file_path = os.path.join(log_dir, f"eval_log_cot_{model_name_to_record}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{debug_suffix}.json")
    results_lst = []
    url = args.url
    if "gpt" in model_name and "oss" not in model_name:
        local_gpt_client = GPT41Client(OPENAI_API_KEY, model_name)
    else:
        local_gpt_client = VLLMClient(model_name=model_name,base_url=f"http://localhost:{url}/v1")
    def fn(d):
        try:
            nonlocal results_lst
            nonlocal correct_count
            prompt = QUESTION_TEMPLATE.format(question=d["question"])
            messages = [{"role": "user", "content": prompt}]
            response = local_gpt_client.call(messages=messages)
            answer = response.split("Final Answer: ")[-1].strip()
            if len(answer) > 1:
                answer = answer[0]
            correctness, pred_answer = eval_response(response, d["ground_truth"], type=eval_type)
            if correctness == 1:
                correct_count += 1
            results_lst.append({
                "question": d["question"],
                "ground_truth": d["ground_truth"],
                "raw_final_messages": response,
                "correct": correctness,
                "predicted_answer": pred_answer,
            })
        except Exception as e:
            print(e)
            results_lst.append({
                "question": d["question"],
                "ground_truth": d["ground_truth"],
                "raw_final_messages": f"Error {str(e)}",
                "correct": 0,
                "predicted_answer": "Error " + str(e),
            })

            return
    map_with_progress(fn, question_data, pbar=True, num_threads=50)
    save_json(results_lst, log_file_path)
    return correct_count / len(question_data)


def retriever_tool_eval(question_data, tool_path, tool_embedding_path, model_name):
    correct_count = 0
    eval_type = args.eval_type
    # Create a log file with a timestamp
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"eval_log_static_{timestamp}.json")
    all_logs = []
    url = args.url
    
    # Load tool data and embeddings
    print(f"📊 Loading tools from {tool_path}")
    tool_data = read_json(tool_path)
    if isinstance(tool_data, dict):
        tool_data = tool_data["tools"]
    print(f"📊 Loaded {len(tool_data)} tools")
    
    if os.path.exists(tool_embedding_path):
        tool_embedding = read_pkl(tool_embedding_path)
        print(f"📊 Loaded embeddings from {tool_embedding_path}")
    else:
        tool_embedding = get_unified_tool_embedding(tool_data, tool_embedding_path)
    
    assert len(tool_embedding) == len(tool_data), f"Tool embedding length {len(tool_embedding)} != tool data length {len(tool_data)}"
    
    # Create static tool registry and preload all tools
    print("🔧 Creating static tool registry...")
    static_registry = StaticToolRegistry()
    registered_count = static_registry.load_all_tools(tool_data)
    print(f"✅ Static registry ready with {registered_count} tools")
    
    # Create retriever
    shared_retriever = UnifiedToolRetriever(tool_data, tool_embedding)
    
    if "gpt" in model_name and "oss" not in model_name:
        local_gpt_client = GPT41Client(OPENAI_API_KEY, model_name)
    else:
        local_gpt_client = VLLMClient(model_name=model_name,base_url=f"http://localhost:{url}/v1")
    CoT_count = 0
    def fn(d):
        nonlocal correct_count
        nonlocal CoT_count
        # Use static tool manager
        local_tool_manager = StaticToolManager(static_registry, shared_retriever)
        question = d["question"]
        answer = d["ground_truth"]
        prompt = QUESTION_TEMPLATE_FLATTEN_RETRIEVAL.format(question=question)
        
        # Prepare initial messages
        messages = [
            {"role": "user", "content": prompt}
        ]
        # Start fresh for each question
        local_tool_manager.reset_tool_pool()
        
        # Use GPT-4.1 with dynamic tool calling (with timeout skip)
        result_holder = {"done": False}
        def _worker():
            t0 = time.perf_counter()
            try:
                fm, tt = local_gpt_client.call_with_dynamic_tools(
                    messages=messages,
                    tool_manager=local_tool_manager,
                    max_turns=6
                )
                dt = time.perf_counter() - t0
                result_holder.update({
                    "done": True,
                    "final_messages": fm,
                    "total_turns": tt,
                    "runtime": dt
                })
            except Exception as e:
                dt = time.perf_counter() - t0
                result_holder.update({
                    "done": True,
                    "error": str(e),
                    "final_messages": messages + [{"role": "assistant", "content": f"Error: {e}"}],
                    "total_turns": 0,
                    "runtime": dt
                })

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(30.0)
        cot_prompt = QUESTION_TEMPLATE.format(question=question)
        cot_messages = [{"role": "user", "content": cot_prompt}]
        cot_response = local_gpt_client.call(messages=messages)
        if not result_holder.get("done"):
            # Timeout: fallback to CoT for this case
            cot_start = time.perf_counter()
            cot_runtime_seconds = time.perf_counter() - cot_start
            response_content = cot_response
            final_messages = cot_messages + [{"role": "assistant", "content": cot_response}]
            correctness, pred_answer = eval_response(response_content, answer, type=eval_type)
            correct_count += correctness
            CoT_count += 1
            log_entry = {
                "question": question,
                "ground_truth": answer,
                "raw_final_messages": final_messages,
                "correct": correctness,
                "predicted_answer": pred_answer,
                "client_runtime_seconds": cot_runtime_seconds,
                "tools_used": list(local_tool_manager.retrieved_tool_names) if hasattr(local_tool_manager, 'retrieved_tool_names') else [],
                "used_cot_due_to_timeout": True
            }
            all_logs.append(log_entry)
            return
        else:
            final_messages = result_holder.get("final_messages", [])
            total_turns = result_holder.get("total_turns", 0)
            client_runtime_seconds = result_holder.get("runtime", 0.0)
        
        # Get tool-augmented final answer
        tar_final_answer_prompt = TAR_FINAL_ANSWER_PROMPT.format(question=question, cot_answer=cot_response, aug_answer=json.dumps(final_messages))
        tar_final_answer_messages = [{"role": "user", "content": tar_final_answer_prompt}]
        tar_final_answer_response = local_gpt_client.call(messages=tar_final_answer_messages)
        # Evaluate correctness
        correctness, pred_answer = eval_response(tar_final_answer_response, answer,type=eval_type)
        correct_count += correctness
        
        # Log the full interaction
        log_entry = {
            "question": question,
            "ground_truth": answer,
            "tar_final_answer_messages": tar_final_answer_messages,
            "correct": correctness,
            "predicted_answer": pred_answer,
            "client_runtime_seconds": client_runtime_seconds,
            "tools_used": list(local_tool_manager.retrieved_tool_names) if hasattr(local_tool_manager, 'retrieved_tool_names') else []
        }
        all_logs.append(log_entry)
        
    
    # Process all questions
    map_with_progress(fn, question_data, pbar=True, num_threads=30)  # Reduced threads for API rate limits
    # Save all logs to a single file
    save_json(all_logs, log_file_path)
    print(f"📝 Logs saved to {log_file_path}")
    # print(f"CoT count: {CoT_count}")
    
    return correct_count / len(question_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, default="data/example_test.json")
    parser.add_argument("--tool_path", type=str, default="data/math_lib.json")
    parser.add_argument("--tool_embedding_path", type=str, default="data/math_lib.pkl")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--model_name", type=str, default="gpt-4.1")
    parser.add_argument("--debug", action='store_true', default=True)
    parser.add_argument("--eval_type", type=str, default="multiple_choice")
    parser.add_argument("--url", type=str, default="8000")
    args = parser.parse_args()
    model_name = args.model_name
    if args.eval_type == "multiple_choice":
       QUESTION_TEMPLATE = """
Please answer the following question: {question}
The question is a multiple choice question, your final answer should be a alphabet (A, B, C, D,E,F,G,...).
Your last line should be your final answer and start with "Final Answer: YOUR_FINAL_ALPHABETICAL_CHOICE".
    """
       QUESTION_TEMPLATE_FLATTEN_RETRIEVAL = """
Please answer the following question: {question}
In the analysis, you are required to retrieve the required tools. You can think about calling tools for helping your analysis.
If there is any error in tool calling, reflect the error type. If the error is because you input incorrect arguments, call it again with correct arguments.
Important note is that the tool may not always be useful in answering the question, so you need to think carefully about the situation and the reply of tools then answering the question.
Your last line should be your final answer and start with "Final Answer: YOUR_FINAL_ALPHABETICAL_CHOICE".
"""

    else:
        QUESTION_TEMPLATE = """
Please answer the following question: {question}
The final answer is a number, it should be a number.
Your last line should be your final answer and start with "Final Answer: YOUR_FINAL_NUMBER_ANSWER".
    """
        QUESTION_TEMPLATE_FLATTEN_RETRIEVAL = """
Please answer the following question: {question}
In the analysis, you are required to retrieve the required tools. You can think about calling tools for helping your analysis.
If there is any error in tool calling, reflect the error type. If the error is because you input incorrect arguments, call it again with correct arguments.
Important note is that the tool may not always be useful in answering the question, so you need to think carefully about the situation and the reply of tools then answering the question.
Your last line should be your final answer and start with "Final Answer: YOUR_FINAL_ANSWER".
"""

    #    question_data = read_json("/Users/murong.yue/Desktop/temp_lib/phy_lib_20250801_163240/c_kinematics_unique_questions.json")
    question_data = read_json(args.input_data_path)
    if args.debug:
        question_data = question_data[:100]

    # CoT Eval
    print("CoT Eval Start...")
    cot_acc = CoT_eval(question_data, model_name)
    print(f"{model_name} {args.input_data_path} CoT Accuracy: {cot_acc}")

    #Flatten Retriever Tool Eval after aggregation
    print("Retriever Tool Eval Start...")
    tool_path = args.tool_path
    tool_embedding_path = args.tool_embedding_path
    accuray = retriever_tool_eval(question_data, tool_path, tool_embedding_path, model_name)
    print(f"{model_name} {args.input_data_path} Retriever Tool Accuracy: {accuray}")