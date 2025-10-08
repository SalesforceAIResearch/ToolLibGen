import pandas as pd
import json
from typing import Any, Dict, List, Union
import requests
import os
import dotenv
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from multiprocessing.pool import ThreadPool
import subprocess
import sys
import ast
import numpy as np
import pickle
import random
import openai
import os
import dotenv
import pandas as pd
import requests
import subprocess
import patch
import signal
import re
from typing import Any, Dict, List, Optional, Tuple



dotenv.load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_client = openai.OpenAI(api_key=openai_api_key)

import openai
import random
OPENAI_API_BASE = "http://localhost:8000/v1"
client = openai.OpenAI(api_key="EMPTY", base_url=OPENAI_API_BASE)


class CodeValidationTimeoutError(Exception):
    """Custom timeout exception for code validation."""
    pass

def _validation_timeout_handler(signum, frame):
    """Signal handler for code validation timeout."""
    raise CodeValidationTimeoutError("Code validation timeout")


def validate_code_syntax(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Validate Python code syntax with timeout protection.
    
    Args:
        code: Python code string to validate
        timeout: Maximum time to spend on validation (default: 5 seconds)
        
    Returns:
        Dict containing:
        - is_valid: bool indicating if code is syntactically valid
        - error_type: str type of error if invalid (None if valid)
        - error_message: str error message if invalid (None if valid)
        - line_number: int line number of error if applicable (None otherwise)
    """
    if not code or not code.strip():
        return {
            "is_valid": True,
            "error_type": None,
            "error_message": None,
            "line_number": None
        }
    
    try:
        # Set up signal handler for timeout (Unix-like systems only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, _validation_timeout_handler)
            signal.alarm(timeout)
        
        try:
            # Try to parse the code with AST
            ast.parse(code)
            return {
                "is_valid": True,
                "error_type": None,
                "error_message": None,
                "line_number": None
            }
        except SyntaxError as e:
            return {
                "is_valid": False,
                "error_type": "SyntaxError",
                "error_message": str(e),
                "line_number": getattr(e, 'lineno', None)
            }
        except Exception as e:
            return {
                "is_valid": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "line_number": None
            }
        finally:
            # Cancel the alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                
    except CodeValidationTimeoutError:
        return {
            "is_valid": False,
            "error_type": "TimeoutError", 
            "error_message": f"Code validation timed out after {timeout} seconds",
            "line_number": None
        }
    except Exception as e:
        # Fallback for systems without signal support or other errors
        try:
            ast.parse(code)
            return {
                "is_valid": True,
                "error_type": None,
                "error_message": None,
                "line_number": None
            }
        except SyntaxError as se:
            return {
                "is_valid": False,
                "error_type": "SyntaxError",
                "error_message": str(se),
                "line_number": getattr(se, 'lineno', None)
            }
        except Exception as pe:
            return {
                "is_valid": False,
                "error_type": type(pe).__name__,
                "error_message": str(pe),
                "line_number": None
            }

def validate_code_syntax_exec(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Validate Python code syntax by compiling in a separate subprocess.

    This avoids executing the code while ensuring we match the interpreter's
    behavior for syntax acceptance. It uses base64 to safely transmit code.

    Returns the same schema as validate_code_syntax.
    """
    if not code or not code.strip():
        return {
            "is_valid": True,
            "error_type": None,
            "error_message": None,
            "line_number": None
        }

    try:
        import base64
        encoded = base64.b64encode(code.encode('utf-8')).decode('ascii')
        script = (
            "import base64, sys\n"
            f"_c = base64.b64decode('{encoded}').decode('utf-8')\n"
            "try:\n"
            "    compile(_c, '<string>', 'exec')\n"
            "    print('OK')\n"
            "except SyntaxError as e:\n"
            "    import traceback\n"
            "    tb = traceback.format_exc()\n"
            "    print(tb)\n"
        )
        result = execute_code(script, timeout=timeout)
        text = (result or "").strip()
        if text.startswith("OK"):
            return {
                "is_valid": True,
                "error_type": None,
                "error_message": None,
                "line_number": None
            }
        # Try to extract lineno from traceback
        line_number = None
        try:
            import re
            m = re.search(r"line (\d+)", text)
            if m:
                line_number = int(m.group(1))
        except Exception:
            line_number = None
        return {
            "is_valid": False,
            "error_type": "SyntaxError" if "SyntaxError" in text else "Error",
            "error_message": text,
            "line_number": line_number
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "line_number": None
        }


def validate_tools_syntax(tools: List[Dict[str, Any]], timeout: int = 1) -> List[Dict[str, Any]]:
    """
    Validate syntax for a list of tools with timeout protection.
    
    Args:
        tools: List of tool dictionaries, each should have 'code' key
        timeout: Maximum time to spend on each code validation
        
    Returns:
        List of error dictionaries for tools with syntax errors. Empty list if all valid.
        Each error dict contains:
        - tool_index: int index of the tool in the input list
        - tool_number: str/int tool number from tool data or index+1
        - error_type: str type of error 
        - error_message: str error message
        - line_number: int line number if applicable
    """
    syntax_errors = []
    
    for i, tool in enumerate(tools):
        code = tool.get("code", "")
        if not code.strip():
            continue
            
        validation_result = validate_code_syntax(code, timeout=timeout)
        
        if not validation_result["is_valid"]:
            syntax_errors.append({
                "tool_index": i,
                "tool_number": tool.get("tool_number", i + 1),
                "error_type": validation_result["error_type"],
                "error_message": validation_result["error_message"],
                "line_number": validation_result.get("line_number")
            })
    
    return syntax_errors


def read_parquet_file(file_path: str = '0000.parquet') -> pd.DataFrame:
    """
    Read a parquet file and return it as a pandas DataFrame.
    
    Args:
        file_path (str): Path to the parquet file. Defaults to '0000.parquet'
        
    Returns:
        pd.DataFrame: DataFrame containing the parquet file data
    """
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return None

def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return it as a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the CSV file data
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def save_json(data: Union[List, Dict], file_path: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data (Union[List, Dict]): Data to save (can be a list or dictionary)
        file_path (str): Path where the JSON file will be saved
        indent (int): Number of spaces for indentation. Defaults to 2
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False

def read_json(file_path: str) -> Union[List, Dict, None]:
    """
    Read data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        Union[List, Dict, None]: The loaded JSON data, or None if there was an error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

def read_jsonl(file_path: str) -> List[Dict]:
    """
    Read data from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict]: A list of dictionaries from the JSONL file
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        return []

def save_jsonl(data: List[Dict], file_path: str) -> bool:
    """
    Save data to a JSONL file.
    
    Args:
        data (List[Dict]): A list of dictionaries to save
        file_path (str): Path where the JSONL file will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"Error saving JSONL file: {e}")
        return False



def read_pkl(file_path: str):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


    
def call_openai_api(content: str, model_name: str = "gpt-5", stream: bool = False, stop_tokens_lst: list = None):
    if model_name == "gpt-5":
        try:
            response = openai_client.responses.create(
            model=model_name,
            input=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "input_text",
                    "text": content
                    }
                ]
                }
            ],
            text={
                "format": {
                "type": "text"
                },
                "verbosity": "medium"
            },
            reasoning={
                "effort": "medium",
                "summary": "auto"
            },
            tools=[]
            )
            return response.output_text
        except Exception as e:
            raise RuntimeError(f"Failed to call OpenAI API: {e}")
    else:
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to call OpenAI API: {e}")



# DEPRECATED: Use call_llm(model_name, messages=..., provider="openai") instead
def call_openai_api_multi_turn(model_name: str, messages: list):
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to call OpenAI API: {e}")



def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap(f, xs), total=len(xs)))



def execute_code(code: str, timeout: int = 3):
    """
    Executes Python code in a subprocess and returns the printed output.

    Args:
        code (str): The code to execute.
        timeout (int): The timeout for the code execution.

    Returns:
        str: The output of the code execution.
    """
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except subprocess.TimeoutExpired:
        return "Error: Timeout - code took too long to execute."
    except Exception as e:
        return f"Error executing code: {e}"
    

# DEPRECATED: Use call_llm(model_name, messages=..., provider="vllm", tools=...) instead
def call_vllm(
    messages: list,
    tools: list = None,
    model_name: str = "Qwen/Qwen3-8B",
    tool_choice: str = "auto",
    enable_thinking: bool = False,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 0.8,
    presence_penalty: float = 1.5,
    top_k: int = 20,
    openai_api_base: str = "http://localhost:8000/v1"
):
    """
    Call the deployed Qwen model.

    Args:
        messages (list): A list of message objects.
        tools (list, optional): A list of tool objects. Defaults to None.
        model_name (str): The name of the model to use. Defaults to "Qwen/Qwen3-8B".
        tool_choice (str): The tool choice strategy. Defaults to "auto".
        enable_thinking (bool): Whether to enable thinking. Defaults to False.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 8192.
        temperature (float): The sampling temperature. Defaults to 0.7.
        top_p (float): The nucleus sampling probability. Defaults to 0.8.
        presence_penalty (float): The presence penalty. Defaults to 1.5.
        top_k (int): The top-k sampling parameter. Defaults to 20.
        openai_api_base (str): The base URL for the vLLM API server. Defaults to "http://localhost:8000/v1".

    Returns:
        The response from the Qwen model. If tools are provided, returns the full response object.
        Otherwise, returns the message content.
    """
    openai_api_key = "EMPTY"
    
    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    extra_body = {
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    create_params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "extra_body": extra_body,
    }

    if tools:
        create_params["tools"] = tools
        create_params["tool_choice"] = tool_choice

    try:
        response = client.chat.completions.create(**create_params)
        
        if tools:
            return response
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Qwen model: {e}")
        return None




# DEPRECATED: Use call_llm(model_name, messages=..., provider="vllm", return_format="full") instead
def call_vllm_wo_tool(
        messages,
        model_name="/export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model",
        max_tokens=16000,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        top_k=20,
        openai_api_base="http://localhost:8000/v1",
        enable_thinking=True,
        random_seed_enabled=True,
        ):
    import openai
    import random
    
    client = openai.OpenAI(api_key="EMPTY", base_url=openai_api_base)
    request_params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "extra_body": {"top_k": top_k, "chat_template_kwargs": {"enable_thinking": enable_thinking}}
    }
    
    if random_seed_enabled:
        # Always generate a new random seed for each call, mimicking the successful test script.
        # This is the most reliable way to ensure different outputs.
        request_params["seed"] = random.randint(0, 1_000_000_000)

    response = client.chat.completions.create(**request_params)

    msg = response.choices[0].message
    return {
        "answer": msg.content,
        "thinking": getattr(msg, "reasoning_content", "")
    }


# DEPRECATED: Use call_llm(model_name, messages=..., provider="vllm", return_format="r1_style") instead
def call_vllm_wo_tool_return_r1_style_response(
        messages,
        model_name="Qwen/Qwen3-8B",
        max_tokens=32768,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        top_k=20,
        openai_api_base="http://localhost:8000/v1",
        enable_thinking=True,
        ):

    import openai
    client = openai.OpenAI(api_key="EMPTY", base_url=openai_api_base)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        extra_body={"top_k": top_k, "chat_template_kwargs": {"enable_thinking": enable_thinking}}
    )
    msg = response.choices[0].message
    reasoning_content = getattr(msg, "reasoning_content", "")
    return f"<think>{reasoning_content}</think>{msg.content}"



def convert_art_to_tools(extracted_art_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert extracted ART info to OpenAI tools format"""
    tools = []
    for art in extracted_art_info:
        tool = {
            "type": "function",
            "function": {
                "name": art.get("function_name", ""),
                "description": art.get("function_description", ""),
                "parameters": art.get("parameters", {})
            }
        }
        tools.append(tool)
    return tools


def create_function_registry(extracted_art_info: List[Dict[str, Any]]) -> Dict[str, Callable]:
    """Create a registry of functions from the Python code in extracted_art_info"""
    function_registry = {}
    
    for art in extracted_art_info:
        function_name = art.get("function_name", "")
        python_code = art.get("python_code", "")
        
        if function_name and python_code:
            try:
                # Create a local namespace for executing the function
                local_namespace = {}
                exec(python_code, local_namespace)
                
                # Get the function from the namespace
                if function_name in local_namespace:
                    function_registry[function_name] = local_namespace[function_name]
                else:
                    print(f"Warning: Function {function_name} not found in executed code")
            except Exception as e:
                print(f"Error executing code for function {function_name}: {str(e)}")
    
    return function_registry

def get_function_by_name(name: str, function_registry: Dict[str, Callable]) -> Callable:
    """Get function from registry by name"""
    return function_registry.get(name, None)

def my_completion_check(content):
    return "Final Answer:" in content


# DEPRECATED: Use call_llm(model_name, ..., tool_type="temporary", return_format="messages_turns") instead
def call_vllm_with_temporary_tool(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] = None,
    function_registry: Dict[str, Callable] = None,
    model_name: str = "Qwen/Qwen3-8B",
    openai_api_base: str = "http://localhost:8000/v1",
    max_turns: int = 6,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 8192,
    repetition_penalty: float = 1.05,
    enable_thinking: bool = True,
    completion_check: Callable[[str], bool] = my_completion_check,
    max_turns_prompt: str = None,
    **kwargs
) -> tuple[List[Dict[str, Any]], int]:
    """
    Call vLLM endpoint with temporary tool usage support.
    
    Args:
        messages: Initial messages (e.g., [{"role": "user", "content": "..."}])
        tools: List of tool definitions in OpenAI format (optional)
        function_registry: Dict mapping function names to callable functions (optional)
        model_name: Name/path of the model to use
        openai_api_base: Base URL for the vLLM API server
        max_turns: Maximum number of conversation turns
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        repetition_penalty: Repetition penalty
        enable_thinking: Whether to enable thinking mode
        completion_check: Function to check if conversation is complete (content -> bool)
        max_turns_prompt: Prompt to send when max_turns is reached (optional)
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        Tuple of (final_messages, actual_turns_used)
    """
    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=openai_api_base,
    )
    
    # Use empty defaults if not provided
    if tools is None:
        tools = []
    if function_registry is None:
        function_registry = {}
    
    # Default completion check (no specific completion condition)
    if completion_check is None:
        completion_check = lambda content: False  # Never automatically complete
    
    # Make a copy of messages to avoid modifying the original
    working_messages = messages.copy()
    
    turn = 0
    
    while turn < max_turns:
        turn += 1
        
        # Prepare API call parameters
        api_params = {
            "model": model_name,
            "messages": working_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "extra_body": {
                "repetition_penalty": repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            },
        }
        
        # Add additional kwargs
        api_params.update(kwargs)
        
        # Make API call (with or without tools)
        if tools:
            api_params["tools"] = tools
            response = client.chat.completions.create(**api_params)
        else:
            # If no tools available, make a regular call
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
                    
                    # Get the actual function and execute it
                    func = function_registry.get(fn_name, None)
                    if func:
                        try:
                            # Special handling for any function with code parameter and prerequisite_code
                            if "code" in fn_args and kwargs.get("prerequisite_code") and kwargs["prerequisite_code"].strip():
                                # Modify the code argument to include prerequisite_code
                                original_code = fn_args["code"]
                                combined_code = f"{kwargs['prerequisite_code'].strip()}\n\n# === LLM Generated Code ===\n{original_code.strip()}"
                                fn_args["code"] = combined_code
                            
                            fn_result = func(**fn_args)
                            fn_res = str(fn_result) if not isinstance(fn_result, str) else fn_result
                        except Exception as e:
                            fn_res = f"Function execution failed: {str(e)}"
                    else:
                        fn_res = f"Function {fn_name} not found"
                    
                    # Add tool result to messages
                    working_messages.append({
                        "role": "tool",
                        "content": fn_res,
                        "tool_call_id": call_id,
                    })
            
            # Continue the loop to let the model process tool results
            continue
        else:
            # No tool calls, check if conversation is complete using custom completion check
            current_content = working_messages[-1].get("content", "")
            if completion_check(current_content):
                # Custom completion condition met, break the loop
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
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "extra_body": {
                            "repetition_penalty": repetition_penalty,
                        },
                    }
                    final_api_params.update(kwargs)
                    
                    response = client.chat.completions.create(**final_api_params)
                    working_messages.append(response.choices[0].message.model_dump())
                break
            else:
                # Model didn't call tools and completion condition not met, continue
                continue
    
    return working_messages, turn

# DEPRECATED: Use call_llm(model_name, ..., tool_type="extracted_art", return_format="messages_turns") instead
def call_vllm_with_extracted_art(
    messages: List[Dict[str, Any]],
    extracted_art_info: List[Dict[str, Any]],
    **kwargs
) -> tuple[List[Dict[str, Any]], int]:
    """
    Convenience wrapper for call_vllm_with_temporary_tool that accepts extracted_art_info format.
    
    This function provides backward compatibility with the original extracted_art_info format
    by converting it to the standard tools + function_registry format.
    
    Args:
        messages: Initial messages
        extracted_art_info: List of extracted ART information containing tool definitions and code
        **kwargs: Additional parameters to pass to call_vllm_with_temporary_tool
        
    Returns:
        Tuple of (final_messages, actual_turns_used)
    """
    # Convert extracted_art_info to tools and function_registry
    tools = convert_art_to_tools(extracted_art_info)
    function_registry = create_function_registry(extracted_art_info)
    
    # Call the main function
    return call_vllm_with_temporary_tool(
        messages=messages,
        tools=tools,
        function_registry=function_registry,
        **kwargs
    )

# Convenience function with common completion patterns
def call_vllm_with_final_answer_pattern(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] = None,
    function_registry: Dict[str, Callable] = None,
    final_answer_pattern: str = "Final Answer:",
    max_turns_prompt: str = "Please provide your final answer now.",
    **kwargs
) -> tuple[List[Dict[str, Any]], int]:
    """
    Convenience wrapper for call_vllm_with_temporary_tool with common "Final Answer" pattern.
    
    Args:
        messages: Initial messages
        tools: Tool definitions (optional)
        function_registry: Function implementations (optional)
        final_answer_pattern: Pattern to look for to determine completion
        max_turns_prompt: Prompt to send when max_turns is reached
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (final_messages, actual_turns_used)
    """
    def completion_check(content):
        return final_answer_pattern in content
    
    return call_vllm_with_temporary_tool(
        messages=messages,
        tools=tools,
        function_registry=function_registry,
        completion_check=completion_check,
        max_turns_prompt=max_turns_prompt,
        **kwargs
    )

def call_llm_with_python_interpreter(
    messages: List[Dict[str, Any]],
    prerequisite_code: str = "",
    timeout: int = 2,
    llm_type: str = "vllm",
    model_name: str = None,
    **kwargs
) -> tuple[List[Dict[str, Any]], int]:
    """
    General function to call any LLM with Python interpreter tool.
    
    Args:
        messages: List of messages to send to the model
        prerequisite_code: Python code to execute before the LLM-generated code (e.g., library definitions, imports, helper functions)
        timeout: Timeout for Python code execution (default: 10 seconds)
        llm_type: Type of LLM to use ("vllm" or "openai")
        model_name: Name of the model to use (optional, will use defaults if not specified)
        **kwargs: Additional parameters to pass to the underlying LLM function
        
    Returns:
        Tuple of (final_messages, actual_turns_used)
    """
    def safe_python_interpreter(code: str) -> str:
        """
        Execute Python code safely with timeout and error handling.
        First executes prerequisite code, then the provided code.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution result as string
        """
        if not code or not code.strip():
            return "Error: No code provided"
        
        # Combine prerequisite code with the LLM-generated code
        if prerequisite_code and prerequisite_code.strip():
            combined_code = f"{prerequisite_code.strip()}\n\n# === LLM Generated Code ===\n{code.strip()}"
        else:
            combined_code = code.strip()
        
        # Execute the combined code using the existing execute_code function
        result = execute_code(combined_code, timeout=timeout)
        
        # If no output, indicate successful execution
        if not result or result.strip() == "":
            return "Code executed successfully (no output)"
        
        return result
    
    # Update tool description to mention prerequisite code capabilities
    prerequisite_desc = ""
    if prerequisite_code and prerequisite_code.strip():
        prerequisite_desc = " You can reference any functions, classes, or variables defined in the prerequisite code."
    
    python_tool = {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": f"Execute Python code in a safe environment with {timeout}s timeout. Can run any valid Python code including loops, functions, imports, etc.{prerequisite_desc} Returns the printed output or error messages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Can include multiple lines, imports, function definitions, etc."
                    }
                },
                "required": ["code"]
            }
        }
    }
    
    # Call appropriate LLM function based on llm_type
    if llm_type.lower() == "openai" or "gpt" in model_name.lower() or "o3" in model_name.lower() or "o4" in model_name.lower():
        # For OpenAI GPT models
        return call_openai_with_temporary_tool(
            messages=messages,
            tools=[python_tool],
            function_registry={"python_interpreter": safe_python_interpreter},
            model_name=model_name,
            **kwargs
        )
    elif llm_type.lower() == "vllm":
        # For vLLM models
        return call_vllm_with_temporary_tool(
            messages=messages,
            tools=[python_tool],
            function_registry={"python_interpreter": safe_python_interpreter},
            model_name=model_name,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported llm_type: {llm_type}. Supported types are 'vllm' and 'openai'.")


def call_openai_with_temporary_tool(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] = None,
    function_registry: Dict[str, Callable] = None,
    model_name: str = "gpt-4o",
    max_turns: int = 6,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    completion_check: Callable[[str], bool] = None,
    max_turns_prompt: str = None,
    prerequisite_code: str = None,
    **kwargs
) -> tuple[List[Dict[str, Any]], int]:
    """
    Call OpenAI GPT API with tool usage support, similar to call_vllm_with_temporary_tool.
    
    Args:
        messages: Initial messages (e.g., [{"role": "user", "content": "..."}])
        tools: List of tool definitions in OpenAI format (optional)
        function_registry: Dict mapping function names to callable functions (optional)
        model_name: Name of the OpenAI model to use
        max_turns: Maximum number of conversation turns
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        completion_check: Function to check if conversation is complete (content -> bool)
        max_turns_prompt: Prompt to send when max_turns is reached (optional)
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        Tuple of (final_messages, actual_turns_used)
    """
    # Use global openai_client
    global openai_client
    
    # Use empty defaults if not provided
    if tools is None:
        tools = []
    if function_registry is None:
        function_registry = {}
    
    # Default completion check (no specific completion condition)
    if completion_check is None:
        completion_check = lambda content: False  # Never automatically complete
    
    # Make a copy of messages to avoid modifying the original
    working_messages = messages.copy()
    
    turn = 0
    
    while turn < max_turns:
        turn += 1
        
        # Prepare API call parameters
        if model_name=="o3" or model_name == "o4-mini":
            api_params = {
            "model": model_name,
            "messages": working_messages,
            }
        else:
            api_params = {
                "model": model_name,
                "messages": working_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        
        # Add additional kwargs
        api_params.update(kwargs)
        
        # Make API call (with or without tools)
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
            response = openai_client.chat.completions.create(**api_params)
        else:
            # If no tools available, make a regular call
            response = openai_client.chat.completions.create(**api_params)
        
        # Add the assistant's response to messages
        assistant_message = {
            "role": "assistant",
            "content": response.choices[0].message.content,
        }
        
        # Add tool calls if present
        if response.choices[0].message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in response.choices[0].message.tool_calls
            ]
        
        working_messages.append(assistant_message)
        
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
                    
                    # Get the actual function and execute it
                    func = function_registry.get(fn_name, None)
                    if func:
                        try:
                            # Special handling for any function with code parameter and prerequisite_code
                            if "code" in fn_args and prerequisite_code and prerequisite_code.strip():
                                # Modify the code argument to include prerequisite_code
                                original_code = fn_args["code"]
                                combined_code = f"{prerequisite_code.strip()}\n\n# === LLM Generated Code ===\n{original_code.strip()}"
                                fn_args["code"] = combined_code
                            
                            fn_result = func(**fn_args)
                            fn_res = str(fn_result) if not isinstance(fn_result, str) else fn_result
                        except Exception as e:
                            fn_res = f"Function execution failed: {str(e)}"
                    else:
                        fn_res = f"Function {fn_name} not found"
                    
                    # Add tool result to messages
                    working_messages.append({
                        "role": "tool",
                        "content": fn_res,
                        "tool_call_id": call_id,
                    })
            
            # Continue the loop to let the model process tool results
            continue
        else:
            # No tool calls, check if conversation is complete using custom completion check
            current_content = working_messages[-1].get("content", "") or ""
            if completion_check(current_content):
                # Custom completion condition met, break the loop
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
                    }
                    final_api_params.update(kwargs)
                    
                    response = openai_client.chat.completions.create(**final_api_params)
                    working_messages.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    })
                break
            else:
                # Model didn't call tools and completion condition not met, continue
                continue
    
    return working_messages, turn


def call_vllm_with_python_interpreter(
    messages: List[Dict[str, Any]],
    prerequisite_code: str = "",
    timeout: int = 10,
    **kwargs
) -> tuple[List[Dict[str, Any]], int]:
    """
    Call vLLM endpoint with Python interpreter tool.
    
    Args:
        messages: List of messages to send to the model
        prerequisite_code: Python code to execute before the LLM-generated code (e.g., library definitions, imports, helper functions)
        timeout: Timeout for Python code execution (default: 10 seconds)
        **kwargs: Additional parameters to pass to call_vllm_with_temporary_tool
        
    Returns:
        Tuple of (final_messages, actual_turns_used)
    """
    def safe_python_interpreter(code: str) -> str:
        """
        Execute Python code safely with timeout and error handling.
        First executes prerequisite code, then the provided code.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution result as string
        """
        if not code or not code.strip():
            return "Error: No code provided"
        
        # Combine prerequisite code with the LLM-generated code
        if prerequisite_code and prerequisite_code.strip():
            combined_code = f"{prerequisite_code.strip()}\n\n# === LLM Generated Code ===\n{code.strip()}"
        else:
            combined_code = code.strip()
        
        # Execute the combined code using the existing execute_code function
        result = execute_code(combined_code, timeout=timeout)
        
        # If no output, indicate successful execution
        if not result or result.strip() == "":
            return "Code executed successfully (no output)"
        
        return result
    
    # Update tool description to mention prerequisite code capabilities
    prerequisite_desc = ""
    if prerequisite_code and prerequisite_code.strip():
        prerequisite_desc = " You can reference any functions, classes, or variables defined in the prerequisite code."
    
    python_tool = {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": f"Execute Python code in a safe environment with {timeout}s timeout. Can run any valid Python code including loops, functions, imports, etc.{prerequisite_desc} Returns the printed output or error messages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Can include multiple lines, imports, function definitions, etc."
                    }
                },
                "required": ["code"]
            }
        }
    }
    
    return call_vllm_with_temporary_tool(
        messages=messages,
        tools=[python_tool],
        function_registry={"python_interpreter": safe_python_interpreter},
        model_name="Qwen/Qwen3-8B",
        **kwargs
    )

# Keep the old function name for backward compatibility (deprecated)
def call_vllm_with_tools(
    messages: List[Dict[str, Any]],
    extracted_art_info: List[Dict[str, Any]],
    **kwargs
) -> tuple[List[Dict[str, Any]], int]:
    """
    Deprecated: Use call_vllm_with_temporary_tool or call_vllm_with_extracted_art instead.
    
    This function maintains the original hardcoded behavior for strict backward compatibility.
    """
    print("Warning: call_vllm_with_tools is deprecated. Use call_vllm_with_temporary_tool or call_vllm_with_extracted_art instead.")
    
    # Maintain original hardcoded behavior exactly
    return call_vllm_with_final_answer_pattern(
        messages=messages,
        tools=convert_art_to_tools(extracted_art_info),
        function_registry=create_function_registry(extracted_art_info),
        final_answer_pattern="Final Answer:",
        max_turns_prompt="Please provide your final answer now. Your last line should start with 'Final Answer: YOUR_ALPHABETICAL_CHOICE'.",
        **kwargs
    )

def call_sfr_embedding_api(
    content: str,
    api_key: str = None,
    is_query: bool = False,
    reduce_dim: bool = False,
    model_url: str = ""
) -> list:
    """
    Call the SFR Embedding API with a single text content and return the embedding vector.

    Args:
        content (str): The text content to embed.
        api_key (str): The API key for authentication. If None, will use SALESFORCE_SFR_API_KEY from env.
        is_query (bool): Whether the text is a query. Defaults to False.
        reduce_dim (bool): Whether to reduce dimensions. Defaults to False.
        model_url (str): The API endpoint URL. Defaults to SFR Embedding endpoint.

    Returns:
        list: The embedding vector as a list of floats, or None if error.
    """
    result = call_sfr_embedding_api_lst([content], api_key, is_query, reduce_dim, model_url)
    if result and len(result) > 0:
        return result[0]
    return None


def call_sfr_embedding_api_lst(
    content_lst: list[str],
    api_key: str = None,
    is_query: bool = False,
    reduce_dim: bool = False,
    model_url: str = ""
) -> list:
    """
    Call the SFR Embedding API with the given content and return the response as a list of embeddings.
    This is a wrapper that calls the OpenAI implementation for compatibility.

    Args:
        content_lst (list[str]): List of text content to embed.
        api_key (str): The API key for authentication. If None, will use SALESFORCE_SFR_API_KEY from env.
        is_query (bool): Whether the text is a query. Defaults to False.
        reduce_dim (bool): Whether to reduce dimensions. Defaults to False.
        model_url (str): The API endpoint URL. Defaults to SFR Embedding endpoint.

    Returns:
        list: List of embedding vectors (each vector is a list of floats).
    """
    # For compatibility, call the OpenAI implementation
    return call_openai_embedding_api_lst(
        content_lst=content_lst,
        api_key=api_key,
        is_query=is_query,
        reduce_dim=reduce_dim
    )


def call_openai_embedding_api_lst(
    content_lst: list[str],
    api_key: str = None,
    is_query: bool = False,
    reduce_dim: bool = False,
    model: str = "text-embedding-3-large"
) -> list:
    """
    Call the OpenAI Embedding API with the given content and return the response as a list of embeddings.

    Args:
        content_lst (list[str]): List of text content to embed.
        api_key (str): The API key for authentication. If None, will use OPENAI_API_KEY from env.
        is_query (bool): Whether the text is a query. Note: OpenAI doesn't differentiate query vs document embeddings.
        reduce_dim (bool): Whether to reduce dimensions. Note: OpenAI text-embedding-3-large supports dimensions parameter.
        model (str): The OpenAI embedding model to use. Defaults to "text-embedding-3-large".

    Returns:
        list: List of embedding vectors (each vector is a list of floats), or None if error.
    """
    try:
        # Use the provided API key or fall back to environment variable
        if api_key is None:
            api_key = openai_api_key
        
        if not api_key:
            print("Error: No OpenAI API key provided")
            return None
            
        # Create OpenAI client with the provided API key
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare parameters for the embedding call
        embed_params = {
            "input": content_lst,
            "model": model
        }
        
        # Handle dimension reduction if requested
        # text-embedding-3-large supports dimensions parameter for reduced dimensionality
        if reduce_dim:
            # text-embedding-3-large default is 3072, we can reduce to 1536 for compatibility
            embed_params["dimensions"] = 1536
        
        # Call OpenAI embedding API
        response = client.embeddings.create(**embed_params)
        
        # Extract embeddings from response
        embeddings = [embedding.embedding for embedding in response.data]
        
        return embeddings
        
    except openai.OpenAIError as e:
        print(f"Error calling OpenAI Embedding API: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error calling OpenAI Embedding API: {e}")
        return None


def apply_patch(original_code: str, diff_text: str) -> str | None:
    """
    Apply a unified diff format patch to a code string.
    
    This function can handle applying patches to modify source code. It uses the
    patch library as the primary method and falls back to a simple implementation
    if needed.
    
    Args:
        original_code (str): The original code string to which the patch will be applied.
        diff_text (str): The unified diff format patch string obtained from LLM or other sources.
    
    Returns:
        str | None: The new code string after applying the patch, or None if patch application fails.
    """
    if not diff_text.strip():
        return original_code

    if "```diff" in diff_text:
        diff_text = diff_text.split("```diff")[1].split("```")[0].strip()

    # try:
    #     return _apply_patch_with_library(original_code, diff_text)
    # except Exception as e:
    #     print(f"Library patch failed: {e}")

    try:
        return _apply_patch_manual(original_code, diff_text)
    except Exception as e2:
        print(f"Manual patch also failed: {e2}")
        return None

def _apply_patch_with_library(original_code: str, diff_text: str) -> str:
    """
    Apply a unified diff format patch to a code string using the patch library.
    
    This is an internal implementation that uses the patch library for precise
    diff handling. It raises exceptions on failure and should not be called
    directly from outside code.
    
    Args:
        original_code (str): The original code string to which the patch will be applied.
        diff_text (str): The unified diff format patch string obtained from LLM or other sources.
    
    Returns:
        str: The new code string after applying the patch.
        
    Raises:
        Exception: If the patch cannot be applied (e.g., context mismatch, format error).
    """
    import io
    import tempfile
    
    # Ensure the diff has proper headers
    if not diff_text.strip().startswith('---'):
        # Add minimal diff headers if missing
        diff_text = f"--- a/file.py\n+++ b/file.py\n{diff_text}"
    
    # Write original code to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(original_code)
        temp_file.flush()
        temp_path = temp_file.name
    
    try:
        # Parse the patch
        patch_set = patch.fromstring(diff_text.encode('utf-8'))
        
        # Apply the patch to the temporary file
        # The apply method modifies the file in place and returns success/failure
        success = patch_set.apply(root=tempfile.gettempdir())
        
        if not success:
            raise Exception("Patch application failed")
        
        # Read the modified content
        with open(temp_path, 'r', encoding='utf-8') as f:
            result = f.read()
        
        return result
        
    finally:
        # Clean up temporary file
        import os
        try:
            os.unlink(temp_path)
        except:
            pass

def _apply_patch_manual(original_code: str, diff_text: str) -> str:
    """
    Manually apply a simple unified diff to code.
    
    This is a fallback implementation for when the patch library fails.
    It handles basic unified diff format with simple line additions/removals.
    
    Args:
        original_code (str): The original code string
        diff_text (str): The unified diff string
        
    Returns:
        str: The patched code
        
    Raises:
        Exception: If the patch format is not supported or application fails
    """
    import re
    
    lines = original_code.splitlines(keepends=True)
    
    # Find all diff hunks (sections starting with @@)
    hunk_pattern = r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@'
    
    # Split diff into hunks
    hunks = []
    current_hunk = []
    
    for line in diff_text.splitlines():
        if line.startswith('@@'):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = [line]
        elif line.startswith(('+', '-', ' ')) or not line.strip():
            if current_hunk:
                current_hunk.append(line)
    
    if current_hunk:
        hunks.append(current_hunk)
    
    # Apply hunks in reverse order to preserve line numbers
    for hunk in reversed(hunks):
        hunk_header = hunk[0]
        hunk_lines = hunk[1:]
        
        # Parse hunk header
        match = re.match(hunk_pattern, hunk_header)
        if not match:
            continue
            
        old_start = int(match.group(1)) - 1  # Convert to 0-based index
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3)) - 1  # Convert to 0-based index
        new_count = int(match.group(4)) if match.group(4) else 1
        
        # Collect new lines from the hunk
        new_lines = []
        old_line_idx = old_start
        
        for hunk_line in hunk_lines:
            if hunk_line.startswith(' '):
                # Context line - keep as is
                new_lines.append(hunk_line[1:] + '\n')
                old_line_idx += 1
            elif hunk_line.startswith('+'):
                # Addition - add to new lines
                new_lines.append(hunk_line[1:] + '\n')
            elif hunk_line.startswith('-'):
                # Deletion - skip this line
                old_line_idx += 1
            elif not hunk_line.strip():
                # Empty line
                new_lines.append('\n')
        
        # Replace the section in the original lines
        lines[old_start:old_start + old_count] = new_lines
    
    return ''.join(lines)


def apply_patch_with_llm(original_code: str, diff_text: str, model_name: str = "gpt-4.1") -> str:
    """
    Apply a unified diff format patch to a code string using the LLM.
    """
    TEMPLATE = """
    You are a senior Python library architect with deep knowledge of software-engineering best-practices and scientific-computing design patterns.

    Your Task:
    You are given a code snippet and a unified diff format patch. Your task is to apply the patch to the code snippet and return the whole patched code.

    Input 1: Original Code
    {original_code}

    Input 2: Diff Text
    {diff_text}

    Output:
    The whole patched code without any other text. Start with ```python and end with ```.
    """
    prompt = TEMPLATE.format(original_code=original_code, diff_text=diff_text)
    result = call_openai_api(prompt, model_name=model_name)
    if "```python" in result:
        result = result.split("```python")[1].split("```")[0].strip()
    return result



# def print_messages(messages):
#     role_icons = {
#         "user": "👤",
#         "assistant": "🤖",
#         "tool": "🔧"
#     }

#     for m in messages:
#         role = m.get("role", "unknown")
#         icon = role_icons.get(role, "❓")

#         # 先打印 content，如果有
#         content = m.get('content')
#         if content:
#             print(f"{icon} {role}: {content}")

#         # 若assistant含有tool_calls也要打印
#         if role == "assistant" and "tool_calls" in m:
#             print(f"{icon} {role} (tool_calls): {m['tool_calls']}")

#         # tool 角色如果有 name 字段也可以显示出来
#         if role == "tool" and 'name' in m:
#             print(f"{icon} {role} (name): {m['name']}")


def print_messages(messages):
    """
    Prints a list of OpenAI message objects in a user-friendly format.

    Args:
        messages (list): A list of message dictionaries from the OpenAI API.
    """
    # Define icons for different roles for better visual representation.
    role_icons = {
        "user": "👤",
        "assistant": "🤖",
        "tool": "🔧",
        "system": "⚙️"  # Added system role for completeness.
    }

    # Iterate through each message in the provided list.
    for message in messages:
        # Get the role and content from the message dictionary.
        # .get() is used to avoid errors if a key is missing.
        role = message.get("role", "unknown")
        content = message.get("content")
        
        # Select the appropriate icon, defaulting to a question mark.
        icon = role_icons.get(role, "❓")

        # --- Build the output string for the current message ---
        
        # Start with the basic role information.
        output = f"{icon} {role.capitalize()}:"

        # 1. Handle regular text content.
        # This is the most common part of user and assistant messages.
        if isinstance(content, str) and content.strip():
            output += f"\n{content}"

        # 2. Handle tool calls from the assistant.
        # This occurs when the assistant decides to use a tool.
        if message.get("tool_calls"):
            output += "\nTool Calls:"
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                func_name = function.get("name", "N/A")
                func_args = function.get("arguments", "{}")
                output += f"\n  - ID: {tool_call.get('id')}\n  - Function: {func_name}()\n  - Arguments: {func_args}"

        # 3. Handle the result from a tool.
        # This is a message with role 'tool' and contains the tool's output.
        if role == "tool":
            output += f"\nTool Call ID: {message.get('tool_call_id')}\nResult: {content}"
        
        # --- Print the fully constructed output for the message ---
        print(output)
        print("-" * 50) # Add a separator for readability.



class LLMCaller:
    """
    Simplified unified LLM caller with only 3 core calling patterns.
    
    1. Basic call (no tools)
    2. Static tools call (predefined tools)  
    3. Dynamic tools call (tool retrieval from library)
    """
    
    def __init__(
        self,
        model_name: str,
        provider: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        openai_api_base: str = "http://localhost:8000/v1",
        **default_kwargs
    ):
        """Initialize LLM caller."""
        self.model_name = model_name
        self.provider = self._detect_provider(model_name) if provider == "auto" else provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.openai_api_base = openai_api_base
        self.default_kwargs = default_kwargs
    
    def _detect_provider(self, model_name: str) -> str:
        """Auto-detect provider based on model name."""
        model_name_lower = model_name.lower()
        if any(name in model_name_lower for name in ["gpt", "o1", "o3", "o4", "o4-mini"]):
            return "openai"
        return "vllm"
    
    # Type 1: Basic call without tools
    def call(
        self,
        content: str = None,
        messages: List[Dict[str, Any]] = None,
        return_format: str = "content",
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Type 1: Basic LLM call without any tools.
        
        Args:
            content: Single message content
            messages: List of conversation messages  
            return_format: "content" or "full"
            
        Returns:
            Response string or full response dict
        """
        if messages is None:
            if content is None:
                raise ValueError("Either 'messages' or 'content' must be provided")
            messages = [{"role": "user", "content": content}]
        
        call_kwargs = {**self.default_kwargs, **kwargs}
        
        if self.provider == "openai":
            if len(messages) == 1:
                result = call_openai_api(
                    content=messages[0]["content"],
                    model_name=self.model_name,
                    **{k: v for k, v in call_kwargs.items() if k in ["stream", "stop_tokens_lst"]}
                )
            else:
                result = call_openai_api_multi_turn(model_name=self.model_name, messages=messages)
        else:  # vllm
            result_dict = call_vllm_wo_tool(
                messages=messages,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_base=self.openai_api_base,
                **{k: v for k, v in call_kwargs.items() if k in [
                    "top_p", "presence_penalty", "top_k", "enable_thinking", "random_seed_enabled"
                ]}
            )
            result = result_dict.get("answer", "")
        
        return result if return_format == "content" else {"content": result, "provider": self.provider}
    
    # Type 2: Static tools call
    def call_with_static_tools(
        self,
        content: str = None,
        messages: List[Dict[str, Any]] = None,
        tools: List[Dict[str, Any]] = None,
        function_registry: Dict[str, Callable] = None,
        return_format: str = "messages_turns",
        **kwargs
    ) -> Union[str, tuple[List[Dict[str, Any]], int]]:
        """
        Type 2: Call LLM with predefined static tools.
        
        Supports tools like:
        - Python interpreter
        - File operations  
        - Custom functions
        - Any predefined tool set
        
        Args:
            content: Single message content
            messages: List of conversation messages
            tools: Static tool definitions in OpenAI format
            function_registry: Static function implementations
            return_format: "content", "full", or "messages_turns"
            
        Returns:
            Response based on return_format
        """
        if messages is None:
            if content is None:
                raise ValueError("Either 'messages' or 'content' must be provided")
            messages = [{"role": "user", "content": content}]
        
        if tools is None:
            tools = []
        if function_registry is None:
            function_registry = {}
        
        call_kwargs = {**self.default_kwargs, **kwargs}
        
        if self.provider == "openai":
            messages, turns = call_openai_with_temporary_tool(
                messages=messages,
                tools=tools,
                function_registry=function_registry,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **call_kwargs
            )
        else:  # vllm
            messages, turns = call_vllm_with_temporary_tool(
                messages=messages,
                tools=tools,
                function_registry=function_registry,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_base=self.openai_api_base,
                **call_kwargs
            )
        
        return self._format_return(messages, turns, return_format)
    
    # Type 3: Dynamic tools call
    def call_with_dynamic_tools(
        self,
        content: str = None,
        messages: List[Dict[str, Any]] = None,
        tool_retriever_func: Callable = None,
        initial_tools: List[Dict[str, Any]] = None,
        initial_registry: Dict[str, Callable] = None,
        return_format: str = "messages_turns",
        **kwargs
    ) -> Union[str, tuple[List[Dict[str, Any]], int]]:
        """
        Type 3: Call LLM with dynamic tool retrieval capability.
        
        Flow:
        1. Start with initial tools (usually just a retriever)
        2. LLM can use retriever to get new tools from tool library
        3. New tools are dynamically added to the conversation
        4. LLM can use new tools or continue retrieving
        
        Args:
            content: Single message content
            messages: List of conversation messages
            tool_retriever_func: Function to retrieve tools from library
                                Should accept query string and return list of tool info
            initial_tools: Initial tool set (usually just retriever tool)
            initial_registry: Initial function registry
            return_format: "content", "full", or "messages_turns"
            
        Returns:
            Response based on return_format
        """
        if messages is None:
            if content is None:
                raise ValueError("Either 'messages' or 'content' must be provided")
            messages = [{"role": "user", "content": content}]
        
        if tool_retriever_func is None:
            raise ValueError("tool_retriever_func must be provided for dynamic tools")
        
        # Setup initial tools and registry
        if initial_tools is None:
            initial_tools = []
        if initial_registry is None:
            initial_registry = {}
        
        # Create the dynamic tool retriever wrapper
        dynamic_tools = initial_tools.copy()
        dynamic_registry = initial_registry.copy()
        
        def enhanced_retriever(query: str) -> str:
            """Enhanced retriever that can dynamically add tools to the conversation."""
            try:
                # Call the user's retriever function
                retrieved_tools = tool_retriever_func(query)
                
                if not retrieved_tools:
                    return "No tools found for your query."
                
                # Process retrieved tools
                added_tools = []
                for tool_info in retrieved_tools:
                    tool_name = tool_info.get("function_name", "")
                    tool_description = tool_info.get("function_description", "")
                    tool_parameters = tool_info.get("parameters", {})
                    python_code = tool_info.get("python_code", "")
                    
                    if tool_name and python_code:
                        # Create tool definition
                        tool_def = {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": tool_description,
                                "parameters": tool_parameters
                            }
                        }
                        
                        # Execute python code to create function
                        try:
                            local_namespace = {}
                            exec(python_code, local_namespace)
                            
                            if tool_name in local_namespace:
                                # Add to dynamic sets
                                dynamic_tools.append(tool_def)
                                dynamic_registry[tool_name] = local_namespace[tool_name]
                                added_tools.append(tool_name)
                            else:
                                print(f"Warning: Function {tool_name} not found in executed code")
                        except Exception as e:
                            print(f"Error executing code for {tool_name}: {str(e)}")
                
                if added_tools:
                    return f"Successfully retrieved and added {len(added_tools)} tools: {', '.join(added_tools)}. You can now use these tools directly."
                else:
                    return "Retrieved tools but failed to add them. Please try a different query."
                    
            except Exception as e:
                return f"Error retrieving tools: {str(e)}"
        
        # Add the enhanced retriever to initial tools
        retriever_tool = {
            "type": "function",
            "function": {
                "name": "retrieve_tools",
                "description": "Retrieve tools from the tool library based on a query. Returns tool definitions and code that will be dynamically added to this conversation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to search for relevant tools in the library"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        
        dynamic_tools.append(retriever_tool)
        dynamic_registry["retrieve_tools"] = enhanced_retriever
        
        # Custom completion check for dynamic scenarios
        def dynamic_completion_check(content: str) -> bool:
            return any(phrase in content.lower() for phrase in [
                "final answer:", "finished", "done with task"
            ])
        
        call_kwargs = {**self.default_kwargs, **kwargs}
        call_kwargs["completion_check"] = dynamic_completion_check
        
        if self.provider == "openai":
            messages, turns = call_openai_with_temporary_tool(
                messages=messages,
                tools=dynamic_tools,
                function_registry=dynamic_registry,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **call_kwargs
            )
        else:  # vllm
            messages, turns = call_vllm_with_temporary_tool(
                messages=messages,
                tools=dynamic_tools,
                function_registry=dynamic_registry,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_turns=self.max_turns,
                openai_api_base=self.openai_api_base,
                **call_kwargs
            )
        
        return self._format_return(messages, turns, return_format)
    
    def _format_return(
        self,
        messages: List[Dict[str, Any]], 
        turns: int, 
        return_format: str
    ) -> Union[str, Dict[str, Any], tuple[List[Dict[str, Any]], int]]:
        """Format the return value based on requested format."""
        if return_format == "content":
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    return msg["content"]
            return ""
        elif return_format == "full":
            return {
                "messages": messages,
                "turns": turns,
                "last_content": self._format_return(messages, turns, "content")
            }
        elif return_format == "messages_turns":
            return messages, turns
        else:
            raise ValueError(f"Unsupported return_format: {return_format}")


# Convenience functions for common static tools
def create_python_interpreter_tools(prerequisite_code: str = "", timeout: int = 10):
    """Create Python interpreter tool for static tool usage."""
    def safe_python_interpreter(code: str) -> str:
        if not code or not code.strip():
            return "Error: No code provided"
        
        if prerequisite_code and prerequisite_code.strip():
            combined_code = f"{prerequisite_code.strip()}\n\n# === LLM Generated Code ===\n{code.strip()}"
        else:
            combined_code = code.strip()
        
        result = execute_code(combined_code, timeout=timeout)
        return result if result.strip() else "Code executed successfully (no output)"
    
    python_tool = {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": f"Execute Python code in a safe environment with {timeout}s timeout. Can run any valid Python code including loops, functions, imports, etc. Returns the printed output or error messages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Can include multiple lines, imports, function definitions, etc."
                    }
                },
                "required": ["code"]
            }
        }
    }
    
    return [python_tool], {"python_interpreter": safe_python_interpreter}


# Simplified convenience functions
def call_llm(model_name: str, content: str = None, messages: List[Dict[str, Any]] = None, **kwargs) -> str:
    """Simple convenience function for Type 1: Basic LLM calls."""
    llm = LLMCaller(model_name, **{k: v for k, v in kwargs.items() if k in [
        "provider", "temperature", "max_tokens", "max_turns", "openai_api_base"
    ]})
    return llm.call(content=content, messages=messages, return_format="content")


def call_llm_with_code(model_name: str, content: str, prerequisite_code: str = "", **kwargs) -> tuple[List[Dict[str, Any]], int]:
    """Convenience function for Type 2: LLM with Python interpreter."""
    tools, registry = create_python_interpreter_tools(prerequisite_code)
    
    llm = LLMCaller(model_name, **{k: v for k, v in kwargs.items() if k in [
        "provider", "temperature", "max_tokens", "max_turns", "openai_api_base"
    ]})
    return llm.call_with_static_tools(content=content, tools=tools, function_registry=registry)


def call_llm_with_dynamic_retrieval(
    model_name: str, 
    content: str, 
    tool_retriever_func: Callable,
    **kwargs
) -> tuple[List[Dict[str, Any]], int]:
    """Convenience function for Type 3: LLM with dynamic tool retrieval."""
    llm = LLMCaller(model_name, **{k: v for k, v in kwargs.items() if k in [
        "provider", "temperature", "max_tokens", "max_turns", "openai_api_base"
    ]})
    return llm.call_with_dynamic_tools(content=content, tool_retriever_func=tool_retriever_func)



_TYPE_MAP = {
    "str": {"type": "string"},
    "string": {"type": "string"},
    "int": {"type": "integer"},
    "integer": {"type": "integer"},
    "float": {"type": "number"},
    "number": {"type": "number"},
    "double": {"type": "number"},
    "bool": {"type": "boolean"},
    "boolean": {"type": "boolean"},
}


def _slugify_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_\-\s]", "", text)
    text = re.sub(r"[\s\-]+", "_", text).strip("_")
    return text or "generated_function"


def _parse_signature(comment: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Parse a python-like signature from the text if present."""
    # Handle both regular signatures and those with return type annotations
    signature_match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\((.*?)\)(?:\s*->\s*[^:]+)?\s*:", comment, re.S)
    if not signature_match:
        return None, []

    func_name = signature_match.group(1)
    params_blob = signature_match.group(2)
    params: List[Dict[str, Any]] = []
    if params_blob.strip():
        for raw in re.split(r",\s*", params_blob.strip()):
            raw = raw.strip()
            if not raw or raw in ("*", "**"):
                continue
            # patterns: name: type = default | name: type | name=default | name
            name_type_default = re.match(r"^([a-zA-Z_]\w*)\s*:\s*([^=]+?)(?:\s*=\s*(.+))?$", raw)
            name_default_only = re.match(r"^([a-zA-Z_]\w*)\s*=\s*(.+)$", raw)
            name_only = re.match(r"^([a-zA-Z_]\w*)$", raw)

            if name_type_default:
                name = name_type_default.group(1)
                type_str = name_type_default.group(2).strip()
                default = name_type_default.group(3)
            elif name_default_only:
                name = name_default_only.group(1)
                type_str = None
                default = name_default_only.group(2)
            elif name_only:
                name = name_only.group(1)
                type_str = None
                default = None
            else:
                continue

            params.append({
                "name": name,
                "type_str": type_str,
                "has_default": default is not None,
            })
    return func_name, params


def _parse_doc_args(comment: str) -> Dict[str, Dict[str, Any]]:
    """Parse a Google/NumPy-style Args: section if present."""
    lines = comment.splitlines()
    params: Dict[str, Dict[str, Any]] = {}

    # Locate Args: block
    start_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*Args\s*:\s*$", line):
            start_idx = i + 1
            break
    if start_idx is None:
        return params

    for j in range(start_idx, len(lines)):
        line = lines[j]
        if not line.strip():
            # empty line signifies end (loosely)
            if params:
                break
            else:
                continue
        # Stop if a new section header appears (expanded common headers)
        if re.match(r"^\s*(Returns?|Raises|Examples?|Notes?|See Also|References?)\s*:\s*$", line) and not line.strip().startswith("Args:"):
            break

        # Accept formats: "name (type): desc" | "name: desc" | "- name (type): desc"
        m = re.match(r"^\s*-?\s*([a-zA-Z_]\w*)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$", line)
        if m:
            name, type_part, desc = m.group(1), m.group(2), m.group(3)
            params[name] = {
                "type_str": type_part.strip() if type_part else None,
                "description": desc.strip(),
            }

    return params


def _summarize_description(comment: str) -> str:
    # Take first non-empty line that is not a signature
    for line in comment.splitlines():
        stripped = line.strip().strip('"\'')
        if not stripped:
            continue
        if stripped.startswith("def "):
            continue
        return stripped
    return "Generated function from comment"


def _json_schema_for_type(type_str: Optional[str]) -> Dict[str, Any]:
    if not type_str:
        return {"type": "string"}

    t = type_str.strip()
    t_norm = t.lower().replace(" ", "")

    # Optional[T]
    m_opt = re.match(r"^optional\[(.+)\]$", t_norm)
    if m_opt:
        return _json_schema_for_type(m_opt.group(1))

    # List[T] or list[T]
    m_list = re.match(r"^(?:list|typing\.list)\[(.+)\]$", t_norm)
    if m_list:
        item_type = _json_schema_for_type(m_list.group(1))
        return {"type": "array", "items": item_type}

    # Dict[K, V] or Mapping
    m_dict = re.match(r"^(?:dict|typing\.dict|mapping|typing\.mapping)\[.+\]$", t_norm)
    if m_dict:
        return {"type": "object"}

    # Tuple[...] -> array
    m_tuple = re.match(r"^(?:tuple|typing\.tuple)\[.+\]$", t_norm)
    if m_tuple:
        return {"type": "array"}

    # Fallback to primitives map
    return _TYPE_MAP.get(t_norm, {"type": "string"})


def _is_optional(type_str: Optional[str]) -> bool:
    if not type_str:
        return False
    t = type_str.strip().lower().replace(" ", "")
    return t.startswith("optional[")


def code_comment2function_json(comment: str) -> Dict[str, Any]:
    """
    Convert a code comment or simple function snippet to an OpenAI tool/function JSON.

    Returns a dict compatible with OpenAI tools API item:
    {
      "type": "function",
      "function": {"name": str, "description": str, "parameters": JSONSchema}
    }
    """
    comment = comment or ""

    # Extract signature and doc params
    func_name, sig_params = _parse_signature(comment)
    doc_params = _parse_doc_args(comment)
    description = _summarize_description(comment)

    if not func_name:
        # derive a name from description
        func_name = _slugify_name(description)[:64] or "generated_function"

    # Merge params by name: signature is authoritative for requiredness; types from doc override if present
    merged: Dict[str, Dict[str, Any]] = {}
    sig_names = {p["name"] for p in sig_params} if sig_params else set()

    for p in sig_params:
        name = p["name"]
        type_str = p.get("type_str")
        if name in doc_params and doc_params[name].get("type_str"):
            type_str = doc_params[name]["type_str"]
        merged[name] = {
            "type_str": type_str,
            "description": doc_params.get(name, {}).get("description"),
            "required": (not p.get("has_default", False)) and (not _is_optional(type_str)),
        }

    # Add any doc-only params not in signature (only if have signature and param is in signature)
    # This matches official LangChain behavior: strict validation that docstring params must be in function signature
    if sig_params:  # Only merge doc params if we have a function signature
        for name, meta in doc_params.items():
            if name not in merged and name in sig_names:
                merged[name] = {
                    "type_str": meta.get("type_str"),
                    "description": meta.get("description"),
                    "required": not _is_optional(meta.get("type_str")),
                }

    # If we still have no params, create a generic input
    if not merged:
        merged["input"] = {"type_str": "str", "description": "Free-form input", "required": True}

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, meta in merged.items():
        schema = _json_schema_for_type(meta.get("type_str"))
        if meta.get("description"):
            schema = {**schema, "description": meta["description"]}
        properties[name] = schema
        if meta.get("required"):
            required.append(name)

    json_schema: Dict[str, Any] = {
        "title": func_name,
        "type": "object",
        "properties": properties,
    }
    if required:
        json_schema["required"] = required

    function_json: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": json_schema,
        },
    }
    return function_json
