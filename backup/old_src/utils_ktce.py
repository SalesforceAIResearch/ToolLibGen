import re
import json
from pathlib import Path
import os
import subprocess
import traceback

# Helper functions from utils/parser.py
def extract_program(response: str) -> str:
    """
    Extracts Python code from a response string.
    Looks for code blocks enclosed in ```python ... ``` or ``` ... ```.
    """
    python_pattern = r"```python\n(.*?)\n```"
    python_matches = re.findall(python_pattern, response, re.DOTALL)
    if python_matches:
        return python_matches[0]

    code_pattern = r"```(.*?)```"
    code_matches = re.findall(code_pattern, response, re.DOTALL)
    if code_matches:
        return code_matches[0]
    return response

def extract_math_tools(response: str):
    """
    Extracts tool definitions (functions or classes) from a Python code string.
    """
    pattern = r"^(def|class)\s"
    tools = []
    current_tool = ""
    in_tool = False
    for line in response.split("\n"):
        if re.match(pattern, line):
            if in_tool:
                tools.append(current_tool.strip())
            current_tool = line + "\n"
            in_tool = True
        elif in_tool:
            current_tool += line + "\n"
    if in_tool:
        tools.append(current_tool.strip())
    return tools

def extract_function_name(code):
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
    return match.group(1) if match else None

def extract_class_name(code):
    match = re.search(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
    return match.group(1) if match else None

def extract_function_docstring(code):
    match = re.search(r'def\s.*?:\s*("""(.*?)""")?', code, re.DOTALL)
    return match.group(2).strip() if match and match.group(2) else ""

def extract_class_docstring(code):
    match = re.search(r'class\s.*?:\s*("""(.*?)""")?', code, re.DOTALL)
    return match.group(2).strip() if match and match.group(2) else ""

def remove_function_docstring(code):
    return re.sub(r'("""(.*?)""")?', "", code, 1, re.DOTALL)

def extract_function_description(code):
    docstring = extract_function_docstring(code)
    return docstring.split("\n")[0] if docstring else ""

def extract_class_description(code):
    docstring = extract_class_docstring(code)
    return docstring.split("\n")[0] if docstring else ""

# Helper functions from utils/utils.py
def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Helper functions from utils/grader.py
def normalize_answer(answer):
    if isinstance(answer, (int, float)):
        return answer
    
    if not isinstance(answer, str):
        return None
        
    answer = answer.strip()
    
    # Pattern to find a number in a string, possibly with commas
    match = re.search(r'(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\.\d+)', answer)
    if match:
        num_str = match.group(1)
        # Remove commas and convert to float
        try:
            return float(num_str.replace(',', ''))
        except ValueError:
            return None
    return None

def grade_answer(result, ground_truth):
    if result is None:
        return False
        
    normalized_result = normalize_answer(result)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_result is None or normalized_ground_truth is None:
        # Fallback to string comparison if normalization fails
        return str(result).strip() == str(ground_truth).strip()

    # If both are numbers, compare with a small tolerance for float precision
    if isinstance(normalized_result, (int, float)) and isinstance(normalized_ground_truth, (int, float)):
        return abs(normalized_result - normalized_ground_truth) < 1e-6
    
    return False

# Helper function from baselines/KTCE_agg.py
def execute_code(tool_code, generated_code, exec_dir):
    """
    Executes a given python code string and returns the result.
    """
    try:
        # Create a file to write the code to
        file_path = Path(exec_dir) / "temp_exec_code.py"
        
        # Ensure exec_dir exists
        Path(exec_dir).mkdir(parents=True, exist_ok=True)

        # The full code to be executed
        full_code = f"""
import traceback
import sys

# Redirect stdout to a file
sys.stdout = open('{Path(exec_dir) / 'exec_output.txt'}', 'w')
sys.stderr = open('{Path(exec_dir) / 'exec_error.txt'}', 'w')

try:
    # Tool definitions
{tool_code}

    # Generated code
{generated_code}

except Exception as e:
    print(traceback.format_exc())
"""
        with open(file_path, 'w') as f:
            f.write(full_code)

        # Execute the code in a separate process
        result = subprocess.run(["python", str(file_path)], timeout=30, capture_output=True, text=True)
        
        # Read output files if they exist
        output_path = Path(exec_dir) / 'exec_output.txt'
        error_path = Path(exec_dir) / 'exec_error.txt'
        
        output = output_path.read_text().strip() if output_path.exists() else ""
        error = error_path.read_text().strip() if error_path.exists() else ""
        
        # If subprocess failed or there are errors, return False
        if result.returncode != 0 or error:
            return False, error or result.stderr
            
        return True, output

    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        # Catch all other exceptions and return False
        return False, str(e) 