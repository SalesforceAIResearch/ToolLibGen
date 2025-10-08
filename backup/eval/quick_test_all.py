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
from utils import read_json, save_json, map_with_progress, call_openai_api, read_pkl, call_llm_with_python_interpreter, LLMCaller

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

QUESTION_TEMPLATE = """
Please answer the following question: {question}
If the question is a multiple choice question, your final answer should be a letter (A, B, C, or D). Or if the final answer is a number, it should be a number.
Your last line should be your final answer and start with "Final Answer: YOUR_FINAL_ANSWER".
"""

QUESTION_TEMPLATE_FLATTEN_RETRIEVAL = """
Please answer the following question: {question}
Your last line should be your final answer and start with "Final Answer: YOUR_ALPHABETICAL_CHOICE". 
In the analysis, you are required to retrieve the required tools. You can think about calling tools for helping your analysis.
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
    eval_type = "number" if "dev_math" in args.input_data_path else "multiple_choice"
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


def Flatten_retriever_tool_eval(question_data, tool_path, tool_embedding_path, model_name):
    correct_count = 0
    
    # Create a log file with a timestamp
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"eval_log_{timestamp}.json")
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
    if "gpt" in model_name and "oss" not in model_name:
        local_gpt_client = GPT41Client(OPENAI_API_KEY, model_name)
    else:
        local_gpt_client = VLLMClient(model_name=model_name,base_url=f"http://localhost:{url}/v1")
    def fn(d):
        nonlocal correct_count
        # Create local tool manager for this question
        local_tool_manager = create_unified_tool_system(tool_data, tool_embedding)
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

def answer_with_python_code(question_data, python_code):
    correct_count = 0
    
    # Create a log file with a timestamp
    log_dir = "/Users/murong.yue/Desktop/log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"eval_log_{timestamp}.json")
    all_logs = []
    
    def fn(d):
        nonlocal correct_count
        question = d["question"]
        answer = d["ground_truth"]
        prompt = QUESTION_TEMPLATE_ANSWER_WITH_PYTHON_CODE.format(question=d["question"],python_code=python_code)
        messages = [
            {"role": "user", "content": prompt}
        ]

        final_messages, total_turns = call_llm_with_python_interpreter(messages=messages, model_name="gpt-4.1", prerequisite_code=python_code,llm_type="openai",max_turns=10,completion_check=completion_check)
        response_content = ""
        for msg in reversed(final_messages):
            if msg.get("role") == "assistant":
                response_content = msg.get("content", "")
                break
        correctness, pred_answer = eval_response(response_content, answer)
        correct_count += correctness
        
        # Log the full interaction
        log_entry = {
            "question": question,
            "ground_truth": answer,
            "raw_final_messages": final_messages,
            "correct": correctness,
            "predicted_answer": pred_answer,
        }
        all_logs.append(log_entry)
    
    map_with_progress(fn, question_data, pbar=True, num_threads=50)
    
    save_json(all_logs, log_file_path)
    print(f"📝 Logs saved to {log_file_path}")
    
    return correct_count / len(question_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, default="/export/home/data/superGPQA_test_data_medicine.json")
    parser.add_argument("--tool_path", type=str, default="/export/home/data/sci_tools_to_update_144.json")
    parser.add_argument("--tool_embedding_path", type=str, default="/export/home/data/sci_tools_to_update_144.pkl")
    parser.add_argument("--log_dir", type=str, default="/export/home/log")
    parser.add_argument("--model_name", type=str, default="gpt-4.1")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--url", type=str, default="8000")
    args = parser.parse_args()
    model_name = args.model_name
    #    question_data = read_json("/Users/murong.yue/Desktop/temp_lib/phy_lib_20250801_163240/c_kinematics_unique_questions.json")
    question_data = read_json(args.input_data_path)
    if args.debug:
        question_data = question_data[:10]
    #    print(len(question_data))
    # CoT Eval
    cot_acc = CoT_eval(question_data, model_name)
    print(f"{model_name} {args.input_data_path} CoT Accuracy: {cot_acc}")

    #Flatten Retriever Tool Eval after aggregation
    # tool_path = args.tool_path
    # tool_embedding_path = args.tool_embedding_path
    # print(Flatten_retriever_tool_eval(question_data, tool_path, tool_embedding_path, model_name))

    #    # Flatten Retriever Tool Eval before aggregation
    #    tool_path = "/Users/murong.yue/Desktop/data/Nemotron_science_data_tools_saved_all_flat.json"
    #    tool_embedding_path = "/Users/murong.yue/Desktop/data/Nemotron_science_data_tools_saved_all_flat.pkl"
    #    print(Flatten_retriever_tool_eval(question_data, tool_path, tool_embedding_path))

    # Answer with Python Code
    # python_code = read_python_code("/Users/murong.yue/Desktop/temp_lib/phy_lib_20250801_163240/c_kinematics_v3.py")
    # print(answer_with_python_code(question_data, python_code))