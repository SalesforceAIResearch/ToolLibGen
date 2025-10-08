#!/usr/bin/env python3
import json
import sys
import os
import datetime
from collections import defaultdict
import numpy as np
import openai
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import read_json, save_json, map_with_progress, LLMCaller, read_pkl


def read_python_code(file_path):
    with open(file_path, 'r') as file:
        return file.read()


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
    response = LLMCaller(model_name="gpt-4o-mini").call(content=prompt)
    return response.strip() == "Yes"


def eval_response(response, answer, type="multiple_choice"):
    if "Final Answer:" in response:
        pred_answer = response.split("Final Answer:")[-1].strip()
        # Extract only the first letter/character that appears to be an answer choice
        import re
        if type == "multiple_choice":
            match = re.search(r'[a-g]', pred_answer)
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




QUESTION_TEMPLATE = """
Please answer the following question: {question}
If the question is a multiple choice question, your final answer should be a letter (A, B, C, or D). Or if the final answer is a number, it should be a number.
Your last line should be your final answer and start with "Final Answer: YOUR_FINAL_ANSWER".
"""

QUESTION_TEMPLATE_FLATTEN_RETRIEVAL = """
Please answer the following question: {question}
Your last line should be your final answer and start with "Final Answer: YOUR_ALPHABETICAL_CHOICE". 
In the analysis, you can retrieve the external tools and can think about calling tools for helping your analysis.
If there is any error in tool calling, reflect the error type. If the error is because you input incorrect arguments, call it again with correct arguments.
Important note is that the tool may not always be useful in answering the question, so you need to think carefully about the situation and the reply of tools then answering the question.
"""

QUESTION_TEMPLATE_ANSWER_WITH_PYTHON_CODE = """
Reference Library Code:
{python_code}

Please answer the following question: {question}
Your last line should be your final answer and start with "Final Answer: YOUR_ALPHABETICAL_CHOICE".
In the first step, you need to think about how to answer based on the given library code. In this analysis step, be careful about the scope of the function in the library code and the scope of the question. And then you need to write python code to answer the question. When you write the code, you are very highly encouraged to use class and public function in the Reference Library Code.
You must add some print statement in the code to show the result you would like to get, make the print statement a sentence not only a number.
"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def CoT_eval(question_data):
    correct_count = 0
    log_dir = "/Users/murong.yue/Desktop/log"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"eval_log_cot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    results_lst = []
    def fn(d):
        nonlocal results_lst
        nonlocal correct_count
        prompt = QUESTION_TEMPLATE.format(question=d["question"])
        response = LLMCaller(model_name="gpt-4.1").call(content=prompt)
        answer = response.split("Final Answer: ")[-1].strip()
        if len(answer) > 1:
            answer = answer[0]
        correctness, pred_answer = eval_response(response, d["ground_truth"])
        if correctness == 1:
            correct_count += 1
        results_lst.append({
            "question": d["question"],
            "ground_truth": d["ground_truth"],
            "raw_final_messages": response,
            "correct": correctness,
            "predicted_answer": pred_answer,
        })
    map_with_progress(fn, question_data, pbar=True, num_threads=50)
    save_json(results_lst, log_file_path)
    return correct_count / len(question_data)


def Flatten_retriever_tool_eval(question_data, tool_path, tool_embedding_path):
    correct_count = 0
    
    # Create a log file with a timestamp
    log_dir = "/Users/murong.yue/Desktop/log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"eval_log_{timestamp}.json")
    all_logs = []
    
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
    
    def fn(d):
        nonlocal correct_count
        # Create local tool manager for this question
        local_gpt_client = LLMCaller(model_name="gpt-4.1")
        
        # try:
        question = d["question"]
        answer = d["ground_truth"]
        prompt = QUESTION_TEMPLATE_FLATTEN_RETRIEVAL.format(question=question)
        
        # Prepare initial messages
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Start fresh for each question
        local_tool_manager.reset_tool_pool()
        
        # Use GPT-4.1 with dynamic tool calling
        start_time = time.perf_counter()
        final_messages, total_turns = local_gpt_client.call_with_dynamic_tools(
            messages=messages,
            tool_manager=local_tool_manager,
            max_turns=6
        )
        client_runtime_seconds = time.perf_counter() - start_time
        
        # Extract final response
        response_content = ""
        # Find the last assistant message
        for msg in reversed(final_messages):
            if msg.get("role") == "assistant":
                response_content = msg.get("content", "")
                break
        
        # Evaluate correctness
        correctness, pred_answer = eval_response(response_content, answer)
        correct_count += correctness
        
        # Log the full interaction
        log_entry = {
            "question": question,
            "ground_truth": answer,
            "raw_final_messages": final_messages,
            "correct": correctness,
            "predicted_answer": pred_answer,
            "client_runtime_seconds": client_runtime_seconds,
            "tools_used": list(local_tool_manager.retrieved_tool_names) if hasattr(local_tool_manager, 'retrieved_tool_names') else []
        }
        all_logs.append(log_entry)
        
        # print(f"✅ Question processed. Correct: {correctness}, Tools used: {len(local_tool_manager.retrieved_tool_names)}")
            
        # except Exception as e:
        #     print(f"❌ Error processing question: {e}")
            # Don't increment correct_count for errors
    
    # Process all questions
    map_with_progress(fn, question_data, pbar=True, num_threads=10)  # Reduced threads for API rate limits
    
    # Save all logs to a single file
    save_json(all_logs, log_file_path)
    print(f"📝 Logs saved to {log_file_path}")
    
    return correct_count / len(question_data)

def completion_check(content):
    if "Final Answer: " in content:
        return True
    return False

# def answer_with_python_code(question_data, python_code):
#     correct_count = 0
    
#     # Create a log file with a timestamp
#     log_dir = "/Users/murong.yue/Desktop/log"
#     os.makedirs(log_dir, exist_ok=True)
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file_path = os.path.join(log_dir, f"eval_log_{timestamp}.json")
#     all_logs = []
    
#     def fn(d):
#         nonlocal correct_count
#         question = d["question"]
#         answer = d["ground_truth"]
#         prompt = QUESTION_TEMPLATE_ANSWER_WITH_PYTHON_CODE.format(question=d["question"],python_code=python_code)
#         messages = [
#             {"role": "user", "content": prompt}
#         ]

#         final_messages, total_turns = call_llm_with_python_interpreter(messages=messages, model_name="gpt-4.1", prerequisite_code=python_code,llm_type="openai",max_turns=10,completion_check=completion_check)
#         response_content = ""
#         for msg in reversed(final_messages):
#             if msg.get("role") == "assistant":
#                 response_content = msg.get("content", "")
#                 break
#         correctness, pred_answer = eval_response(response_content, answer)
#         correct_count += correctness
        
#         # Log the full interaction
#         log_entry = {
#             "question": question,
#             "ground_truth": answer,
#             "raw_final_messages": final_messages,
#             "correct": correctness,
#             "predicted_answer": pred_answer,
#         }
#         all_logs.append(log_entry)
    
#     map_with_progress(fn, question_data, pbar=True, num_threads=50)
    
#     save_json(all_logs, log_file_path)
#     print(f"📝 Logs saved to {log_file_path}")
    
#     return correct_count / len(question_data)


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--input_data_path", type=str, default="/Users/murong.yue/Desktop/data/superGPQA_test_data_physics_300.json")
   parser.add_argument("--tool_path", type=str, default="/Users/murong.yue/Desktop/data/sci_tools_to_update_144.json")
   parser.add_argument("--tool_embedding_path", type=str, default="/Users/murong.yue/Desktop/data/sci_tools_to_update_144.pkl")
   parser.add_argument("--log_dir", type=str, default="/Users/murong.yue/Desktop/log")
   args = parser.parse_args()
#    question_data = read_json("/Users/murong.yue/Desktop/temp_lib/phy_lib_20250801_163240/c_kinematics_unique_questions.json")
   question_data = read_json(args.input_data_path)
   question_data = question_data
#    print(len(question_data))
   # CoT Eval
#    print(CoT_eval(question_data))

   #Flatten Retriever Tool Eval after aggregation
   tool_path = args.tool_path
   tool_embedding_path = args.tool_embedding_path
   print(Flatten_retriever_tool_eval(question_data, tool_path, tool_embedding_path))

#    # Flatten Retriever Tool Eval before aggregation
#    tool_path = "/Users/murong.yue/Desktop/data/Nemotron_science_data_tools_saved_all_flat.json"
#    tool_embedding_path = "/Users/murong.yue/Desktop/data/Nemotron_science_data_tools_saved_all_flat.pkl"
#    print(Flatten_retriever_tool_eval(question_data, tool_path, tool_embedding_path))

   # Answer with Python Code
   # python_code = read_python_code("/Users/murong.yue/Desktop/temp_lib/phy_lib_20250801_163240/c_kinematics_v3.py")
   # print(answer_with_python_code(question_data, python_code))