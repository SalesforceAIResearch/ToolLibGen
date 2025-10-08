import json
import os
import argparse
import copy
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import logging

from utils import read_json as read_json_main, call_openai_api, map_with_progress, save_json
from utils_ktce import (
    extract_program, extract_math_tools, extract_function_name,
    extract_class_name, extract_function_docstring,
    remove_function_docstring, extract_function_description,
    extract_class_description, grade_answer, execute_code,
    dump_json
)


# --- PROMPT TEMPLATES ---
# Note: Using placeholder prompts for initial code generation.
# These should be replaced with the actual prompts from the project.
# Prompts from KTCE_agg.py
get_tool_prompt_template = """As a Python programming and math expert, given a math question and some math tools, please decide tools can be used to solve this question. Do not solve this question, just judge which tools can be used to solve this question.

### Format ###
Tools are listed below:

No. 0:
Tool 0

No. 1:
Tool 1

...

No. N:
Tool N

Here are some instructions you should follow:
- Analyse what subtasks are in the problem.
- Deeply understand all the tools provided, refer to their function name and code.
- Please judge which tools (one or more) can be used to solve this problem, and give your thoughts.
- If there are tools useful, please output <answer> </answer> with the numeric number list, e.g.: <answer> [N1,N2] </answer>
- If there are not tools useful, please output <answer>[]</answer>
- Take a deep breath
- Think step by step 
- I will tip $200

### Question:
{}

Here are the math tools in the field {}, subfield {}:
{}

"""
tpot_prompt_template = '''As a Python programming and math expert, given a math question and math tools, please use the math tools to solve the math question by python program. Note that do not regenerate the tool code and function, just use the tool code and function to solve the question.

Here are some instructions you should follow:
- You need to understand the tool function in depth to ensure that the parameters called are correct. Directly call tool function without external imports, do not modify the tool function.
- You can also use python libraries, like sympy, math, scipy, numpy, etc. or other packages if necessary.
- Please pay attention to the conditions of the question, conduct accurate logical reasoning, and obtain results that meet the requirements.
---

### Math Tools:
```python
{tool_string}
```
---

Here are some examples you can refer to, you should call the tool functions directly in your code.

{examples_string}

# Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
```python
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
print(solution())
```
---
Now it's your turn!
- Take a deep breath
- Think step by step 
- I will tip $200

# Question: {question}
# Solution: {solution}
'''

delete_tool_prompt_template = """As a Python programming and math expert, you are given a set of tools and their usage statistics. Please decide if any tool should be removed, Only remove the function that you think is not needed anymore in future tasks.

### Toolset Information:
Field: {field}
Subfield: {subfield}
Toolset Size: {n}
Toolset Coverage on Given Math Dataset: {TSC}
Accuracy of Using the Toolset on Given Math Dataset in Python Inference: {TA}

### Tools and Usage Statistics:
{tools_stats}

### Instructions:
- Analyze the tools and their usage statistics.
- Consider the tool's usage frequency and the tool's generality and versatility.
- Don't remove more than 5 tools to maintain diversity in your toolset.
- If a tool should be removed, provide the tool index in the answer, e.g.: <answer> [N1,N2] </answer>

Please provide your decision in the following format:
Reasoning: [Your reasoning]
Remove (at most 3 tools):
<answer> [Tool Indexes] </answer>
"""

add_tool_prompt_template = """As a Python programming and math expert, you are given a set of tools, their usage statistics, and uncovered problems by the toolset. Please decide if any new tool should be added to improve the coverage of the toolset and solve the task in the uncovered problems. Provide your reasoning for each decision, if need to add, please output the new tool code with docstring, Note that it must a general tool for many problems and running accurately. 


### Toolset Information:
Field: {field}
Subfield: {subfield}
Toolset Size: {n}
Toolset Coverage on Given Math Dataset: {TSC}
Accuracy of Using the Toolset on Given Math Dataset in Python Inference: {TA}

### Toolset:
{tools_stats}

### Uncovered Problems:
{unsolved_problems}

### Instructions:
- Analyze and the uncoverd problems and consider what task in this subfield are not solved in current toolset.
- If the task is not solved by the provided toolset, the tool should be added to solve this task. The new tool should be different from the existing tools in the toolset.
- If a tool should be added, please generate the tool code with docstring. As an extension, you can use any tool in the previous toolset in new tool. You need to write these sub-tools as sub-functions completely inside the new tool to ensure that the new tool can run accurately.
- The added function should be general enough to be used in future tasks. For instance, if you encounter a problem that this function can solve, or one step of it, you can use the generated function directly instead of starting from scratch 
- The added new function should solve a higher-level question that encompasses the original query and extend the code's functionality to make it more versatile and widely applicable. 
- Replace specific strings or variable names with general variables to enhance the tool's applicability to various queries. All names used inside the function should be passed in as arguments.

Please provide your decision in the following format:
Reasoning: [Your reasoning]
Add New Tool (one or more):
```python
[New Tool Code] 
```
"""

modify_tool_prompt_template = """As a Python programming and math expert, you are given a math tool and its usage statistics. Please decide if this tool should be evolved to improve its accuracy, flexibility, and ease of use. If evolution is needed, please output the new tool code with a docstring. * Note that the new tool is an evolution of the original tool, and there function name and code must be similar and running accurately !!! *


### Tool Information and Usage Statistics:
# Field: {field}
# Subfield: {subfield}
# Tool Usage Frequency: {Freq}
# Tool Success Rate: {TSR}%
# Tool docstring : {docstring}
# Tool code:
```python
{tool}
```
# Wrong tool calings:
{wrong_tool_callings}
# Tool usage experience:
{experience_pool}

---

### Instructions:
- Evolution often includes expanding the tool's functionality, handling different scenarios, changing parameters and return values, and improving ease of use.
- Pay more attention to failed tasks and corresponding error information. If necessary, optimize the features used in these tasks based on conversation history.
- Function calls may fail due to incorrect input parameters (missing parameters) or incorrect function code implementation. You should focus more on the implementation of the function code and make function calls easier to succeed.
- Based on conversation history, do not modify functions that are effective and crucial for solving the problem. Modify only when necessary.
- A low success rate may be due to the difficulty of the problem, not tool usage errors. You need to judge based on the output content. If the tool is not the cause of the error, you should not modify the tool and update the experience pool instead.

* If the tool can be evolved, and provide your reasoning and the new tool code with docstring and try to update/generate the **experience pool** to prevent similar errors and guide LLM to use the tool accurately. Note what modified is the tool itself, not the wrong calling code of the tool. *

Output format 1 (evolve the tool):
Reasoning: [Your reasoning]
Evolved tool code:
```python
[new tool code] 
```
Experience pool:
<experience_list>
experience content
</experience_list>

---

* If the tool has no problem, consider to update/generate the **experience pool** to prevent similar errors and guide LLM to use the tool accurately. *

Output format 2 (not evolve the tool):
Reasoning: [Your reasoning]
Experience pool:
<experience_list>
experience content
</experience_list>
"""

validation_template_single = """Here are a math tool by python function or class. Your task is to generate  example function calling to validate corectness the tool. Please output the function calling code in block ```python ```. The last line print the result of the function calling. Note that all variables or parameters must be defined and initialized before calling. Note that you should not regenerate the function code, just function calling.

### Input ###
The tools are:
```python
{}
```

### Output ###
```python
# Example function call
```
"""

def get_useful_tools(response: str):
    """From response, extract useful tools."""
    called_idx = []
    try:
        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, response)
        for match in matches:
            # Clean the match and try to evaluate
            cleaned_match = match.strip()
            try:
                ans = eval(cleaned_match)
                # Ensure ans is a list
                if not isinstance(ans, list):
                    ans = [ans]
                for idx in ans:
                    # Only accept numeric indices
                    if isinstance(idx, (int, float)):
                        called_idx.append(int(idx))
            except (NameError, SyntaxError, ValueError):
                # If evaluation fails, try to extract numbers manually
                numbers = re.findall(r'\d+', cleaned_match)
                for num in numbers:
                    try:
                        called_idx.append(int(num))
                    except ValueError:
                        continue
    except Exception as e:
        logging.error(f"Error extracting useful tools: {e}\nResponse: {response}")
    return called_idx

def extract_experience_pool(response: str):
    """From response, extract experience pool."""
    experience_pool = ""
    try:
        pattern = r"<experience_list>(.*?)</experience_list>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            experience_pool = match.group(1).strip()
    except Exception as e:
        logging.error(f"Error extracting experience pool: {e}\nResponse: {response}")
    return experience_pool

def calculate_loss(Q_tool_values, Q_set, alpha, beta, gamma, n, k):
    """Calculates loss value."""
    Q_tool = Q_tool_values.sum()
    loss = alpha * Q_tool + beta * Q_set + gamma * max(0, n - k)
    return loss

class ToolOptimizer:
    def __init__(self, cluster_name, initial_tool_codes, questions, output_dir, model_name):
        self.cluster_name = cluster_name
        self.questions = questions
        self.output_dir = output_dir
        self.model_name = model_name
        self.max_iter_num = 5

        self.cluster_output_dir = self.output_dir / self.cluster_name
        self.cluster_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_tools = self._initialize_toolset(initial_tool_codes)
        self.q_set_history = []
        self.tool_usage_experience = {}
        self.update_actions_history = []

    def _get_tool_json(self, tool_code):
        tool_type = "function" if tool_code.startswith("def ") else "class"
        description = extract_function_description(tool_code) if tool_type == "function" else extract_class_description(tool_code)
        docstring = extract_function_docstring(tool_code) if tool_type == "function" else "" # Simplified
        tool_name = extract_function_name(tool_code) if tool_type == "function" else extract_class_name(tool_code)
        
        return {
            "tool": tool_code,
            "subfield": self.cluster_name,
            "tool_name": tool_name,
            "tool_type": tool_type,
            "description": description,
            "docstring": docstring,
            "Freq": 0,
            "TSR": 0,
        }

    def _initialize_toolset(self, initial_tool_codes):
        tool_jsons = {}
        tool_jsons[self.cluster_name] = [self._get_tool_json(code) for code in initial_tool_codes]
        dump_json(tool_jsons, self.cluster_output_dir / "subfield_tools_iter_0.json")
        return tool_jsons

    def run(self):
        logging.info(f"Starting KTCE optimization for cluster: {self.cluster_name}")
        for iter_num in range(self.max_iter_num + 1):
            logging.info(f"--- Iteration {iter_num} ---")
            
            # 1. Evaluate
            self.evaluate(iter_num)
            
            # Check for stopping condition
            if iter_num == self.max_iter_num:
                break

            # 2. Evolve
            new_tool_jsons, delete_actions = self.delete_tools()
            new_tool_jsons, modify_actions = self.modify_tools(new_tool_jsons)
            new_tool_jsons, add_actions = self.add_tools(new_tool_jsons)

            # Save state for next iteration
            self.current_tools[self.cluster_name] = new_tool_jsons
            dump_json(self.current_tools, self.cluster_output_dir / f"subfield_tools_iter_{iter_num+1}.json")
            
            actions = {"Iter": iter_num, "delete": delete_actions, "add": add_actions, "update": modify_actions}
            self.update_actions_history.append(actions)
            dump_json(self.update_actions_history, self.cluster_output_dir / "all_update_actions.json")

    def evaluate(self, iter_num):
        # Implementation from KTCE_agg.py's evaluate function
        without_tool_ids = []
        TA = 0
        
        tool_jsons = self.current_tools[self.cluster_name]
        toolset = [t['tool'] for t in tool_jsons]
        tool_descriptions = [t['description'] for t in tool_jsons]
        tool_names = [t['tool_name'] for t in tool_jsons]
        
        tool_set_string = ""
        for idx, t in enumerate(toolset):
            tool_set_string += f"No. {idx}:\n{tool_descriptions[idx]}\n```python\n{t}\n```\n"
        
        # Parallel processing of questions
        tasks = [(q, tool_set_string, tool_jsons, tool_names, toolset) for q in self.questions]
        results = map_with_progress(self._process_problem_wrapper, tasks, num_threads=10)

        for res in results:
            is_correct, actual_called_idx, problem_id = res
            if actual_called_idx:
                for idx in actual_called_idx:
                    self.current_tools[self.cluster_name][idx]["Freq"] += 1
                    if is_correct:
                        self.current_tools[self.cluster_name][idx]["TSR"] += 1
            else:
                without_tool_ids.append(problem_id)
            if is_correct:
                TA += 1

        TA = TA / len(self.questions) if self.questions else 0
        TSC = (len(self.questions) - len(without_tool_ids)) / len(self.questions) if self.questions else 0

        # Save stats
        q_entry = {"iter": iter_num, "TSC": TSC, "TA": TA, "n": len(tool_jsons), "without_tool_ids": without_tool_ids}
        self.q_set_history.append(q_entry)
        dump_json(self.q_set_history, self.cluster_output_dir / "Q_subfield_set.json")
        logging.info(f"Iter {iter_num} Eval - TSC: {TSC:.2f}, TA: {TA:.2f}, Tool Count: {len(tool_jsons)}")
    
    def _process_problem_wrapper(self, args):
        problem_data, tool_set_string, tool_jsons, tool_names, toolset = args
        problem = problem_data['problem']
        answer = problem_data['answer']
        problem_id = problem_data['id']
        
        try:
            # Step 1: Select tools
            prompt = get_tool_prompt_template.format(problem, "Science", self.cluster_name, tool_set_string)
            response = call_openai_api(prompt, self.model_name)
            called_idx = get_useful_tools(response)
            called_idx = list(set([i for i in called_idx if 0 <= i < len(tool_jsons)]))

            # Step 2: Call tools
            called_tool_string = "\n\n".join([toolset[i] for i in called_idx])
            prompt = tpot_prompt_template.format(tool_string=called_tool_string, examples_string="", question=problem, solution="")
            response = call_openai_api(prompt, self.model_name)
            code = extract_program(response)
            
            # Ensure the code has a print statement
            if "print" not in code:
                code += "\nprint(solution())"

            # Step 3: Execute and Grade
            report, result = execute_code(called_tool_string, code, self.cluster_output_dir)
            
            # If execution failed (syntax error, runtime error, etc.), treat as incorrect
            if not report:
                return False, [], problem_id
                
            is_correct = grade_answer(result, answer)
            
            actual_called_idx = []
            for idx in called_idx:
                if tool_names[idx] in code:
                    actual_called_idx.append(idx)
            
            return is_correct, actual_called_idx, problem_id
            
        except Exception as e:
            # Ignore any errors (including syntax errors) and treat as failed case
            return False, [], problem_id

    def delete_tools(self):
        # Simplified version of KTCE_agg.py's delete_tools
        tool_jsons = self.current_tools[self.cluster_name]
        q_entry = self.q_set_history[-1]
        
        tools_stats = ""
        for idx, tool in enumerate(tool_jsons):
            rate = (tool['TSR'] / tool['Freq']) * 100 if tool['Freq'] > 0 else 0
            tools_stats += f"No. {idx}: {tool['description']}\nUsage Freq: {tool['Freq']}, Success Rate: {rate:.1f}%\n"

        prompt = delete_tool_prompt_template.format(
            field="Science", subfield=self.cluster_name, n=len(tool_jsons),
            TSC=q_entry['TSC'], TA=q_entry['TA'], tools_stats=tools_stats
        )
        response = call_openai_api(prompt, self.model_name)
        delete_idx = get_useful_tools(response)
        
        actions = [tool_jsons[idx] for idx in delete_idx if 0 <= idx < len(tool_jsons)]
        new_tool_jsons = [t for i, t in enumerate(tool_jsons) if i not in delete_idx]
        return new_tool_jsons, actions

    def add_tools(self, tool_jsons):
        # Simplified version of KTCE_agg.py's add_tools
        q_entry = self.q_set_history[-1]
        
        # Get unsolved problems safely
        unsolved_problem_texts = []
        for problem_id in q_entry['without_tool_ids'][:10]:
            # Find the problem by id
            for question in self.questions:
                if question['id'] == problem_id:
                    unsolved_problem_texts.append(question['problem'])
                    break
        
        unsolved_problems = "\n---\n".join(unsolved_problem_texts)

        prompt = add_tool_prompt_template.format(
             field="Science", subfield=self.cluster_name, n=len(tool_jsons),
            TSC=q_entry['TSC'], TA=q_entry['TA'], tools_stats="", unsolved_problems=unsolved_problems
        )
        response = call_openai_api(prompt, self.model_name)
        added_tool_codes = extract_math_tools(extract_program(response))
        
        actions = []
        for tool_code in added_tool_codes:
            # Basic validation
            if "def " in tool_code or "class " in tool_code:
                 new_tool = self._get_tool_json(tool_code)
                 tool_jsons.append(new_tool)
                 actions.append(new_tool)
        return tool_jsons, actions

    def modify_tools(self, tool_jsons):
        # Simplified version of KTCE_agg.py's modify_tools
        actions = []
        for idx, tool in enumerate(tool_jsons):
            rate = (tool['TSR'] / tool['Freq']) * 100 if tool['Freq'] > 0 else 0
            if tool['Freq'] > 1 and rate < 90:
                prompt = modify_tool_prompt_template.format(
                    field="Science", subfield=self.cluster_name, Freq=tool['Freq'], TSR=rate,
                    docstring=tool['docstring'], tool=tool['tool'], wrong_tool_callings="", experience_pool=""
                )
                response = call_openai_api(prompt, self.model_name)
                modified_codes = extract_math_tools(extract_program(response))
                if modified_codes:
                    new_tool = self._get_tool_json(modified_codes[0])
                    actions.append({"original": tool, "modified": new_tool})
                    tool_jsons[idx] = new_tool # Replace in-place
        return tool_jsons, actions

class KTCEAggregator:
    def __init__(self, clusters_file_path, output_dir, model_name, debug=False):
        self.clusters_file_path = clusters_file_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.debug = debug
        self._load_clusters_data()

    def _load_clusters_data(self):
        data = read_json_main(self.clusters_file_path)
        
        clusters_list = []
        if isinstance(data, dict) and 'clusters' in data:
            clusters_list = data['clusters']
        elif isinstance(data, list):
            clusters_list = data
        else:
            raise ValueError("Unsupported clusters file format")

        self.clusters_data = {c['cluster_name']: c.get('tools', []) for c in clusters_list if 'cluster_name' in c}
        
        if self.debug:
            # Take the first cluster for debugging
            first_cluster_name = next(iter(self.clusters_data), None)
            if first_cluster_name:
                self.clusters_data = {first_cluster_name: self.clusters_data[first_cluster_name][:10]}

    def _extract_tools_for_blueprint(self, tools: list):
        tool_info_parts = []
        for i, tool in enumerate(tools):
            tool_name = tool.get('name', f'tool_{i+1}')
            description = tool.get('description', f'Function: {tool_name}')
            tool_info_parts.append(f"# Function {i+1}: {tool_name}\n# Description: {description}\n")
        return "\n".join(tool_info_parts)

    def run(self):
        for cluster_name, tools in self.clusters_data.items():
            logging.info(f"Processing cluster: {cluster_name}")
            
            # 1. Extract initial tool codes directly from the input
            initial_tool_codes = [tool['python_code'] for tool in tools if 'python_code' in tool]
            
            if not initial_tool_codes:
                logging.warning(f"No python_code found for cluster {cluster_name}, skipping.")
                continue

            # 2. Collect questions
            questions = [
                {'problem': t['original_question'], 'answer': t['original_answer'], 'id': f"{cluster_name}_{i}"}
                for i, t in enumerate(tools) if t.get('original_question')
            ]
            
            # 3. Run KTCE optimization
            optimizer = ToolOptimizer(cluster_name, initial_tool_codes, questions, self.output_dir, self.model_name)
            optimizer.run()

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='KTCE Baseline Implementation')
    parser.add_argument('--local', action='store_true', default=False, help='Enable local mode for paths')
    parser.add_argument('--file', help='Path to the clusters JSON file')
    parser.add_argument('--max-review-iterations', type=int, default=3, help='Maximum number of optimization iterations')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--model-name', type=str, default='o4-mini', help='Model name to use')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--log-folder', type=str, help='Log folder')
    args = parser.parse_args()

    if args.local:
        args.file = "/Users/murong.yue/Desktop/data/Nemotron_science_data_tools_saved_kinematics.json"
        args.output_dir = f"/Users/murong.yue/Desktop/temp_lib/phy_lib_{timestamp}_KTCE"
        args.log_folder = f"/Users/murong.yue/Desktop/log/phy_lib_{timestamp}_KTCE"
    
    # Setup logging with reduced verbosity
    log_dir = Path(args.log_folder)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Reduce HTTP request logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_dir / "ktce_run.log"), logging.StreamHandler()])

    # Run processor
    processor = KTCEAggregator(clusters_file_path=args.file,
                               output_dir=Path(args.output_dir),
                               model_name=args.model_name,
                               debug=args.debug)
    processor.run()
    logging.info("KTCE processing finished.")

if __name__ == "__main__":
    main()
