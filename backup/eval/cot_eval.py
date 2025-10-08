import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import read_json, save_json, map_with_progress, call_vllm_wo_tool
from report_utils import quick_report  # Import the reusable reporting function
import argparse
from datetime import datetime
from collections import defaultdict
import numpy as np
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


QUESTION_TEMPLATE = """
Please answer the following question: {question}
If the question is a multiple choice question, your final answer should be a letter (A, B, C, or D). Or if the final answer is a number, it should be a number.
Your last line should be your final answer and start with "Final Answer: YOUR_FINAL_ANSWER".
"""


def eval_response(response, answer):   
    """
    More general evaluation function that checks if the answer string appears in the final answer.
    
    Args:
        response (str): The model's full response
        answer (str): The ground truth answer
        
    Returns:
        tuple: (correctness, pred_answer) where correctness is 1 if correct, 0 if wrong
    """
    # Extract the final answer after "Final Answer:"
    if "Final Answer:" in response:
        pred_answer = response.split("Final Answer:")[-1].strip()
        # Remove any trailing punctuation or whitespace
        pred_answer = pred_answer.strip().rstrip('.')
    else:
        pred_answer = ""
    
    # Check if the ground truth answer appears in the predicted answer (case-insensitive)
    if answer.strip().lower() in pred_answer.lower():
        correctness = 1
    else:
        correctness = 0
        
    return correctness, pred_answer


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--input_data_path", type=str, default="/export/home/data/dev_physics_322.json")
   parser.add_argument("--model_nickname", type=str, default="Qwen3_8b")
   parser.add_argument("--model_name", type=str, default="/export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model")
   parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
   parser.add_argument("--save_path", type=str, default="/export/home/data/dev_CoT.json")
   parser.add_argument("--save_step", type=int, default=500)
   parser.add_argument("--passk", type=int, default=4)
   parser.add_argument("--debug", action='store_true', default=False)
   
   args = parser.parse_args()
   save_path = args.save_path.replace(".json", f"_{current_time}.json")
   data = read_json(args.input_data_path)
   if args.debug:
      data = random.sample(data, 10)
      save_path = save_path.replace(".json", "_debug.json")
   else:
      data = data
   results_lst = []
   def fn(d):
      global results_lst
      try:
         for i in range(args.passk):
            id = str(d["id"])+f"_{i}"
            question = d["question"]
            answer = d["answer"]
            prompt = QUESTION_TEMPLATE.format(question=question)
            messages = [
               {"role": "user", "content": prompt}
            ]
            
            # Simplified call: Let the function handle its own randomness.
            response = call_vllm_wo_tool(
               messages=messages, 
               model_name=args.model_name, 
               openai_api_base=args.api_base
            )
            reasoning_content = response["thinking"]
            response_content = response["answer"]
            correctness, pred_answer = eval_response(response_content, answer)
            
            # To absolutely ensure no data is overwritten, we create a brand new dictionary
            # for each result. This prevents different list items from pointing to the
            # same object in memory, which would cause the last iteration's result
            # to overwrite all previous ones for the same input `d`.
            result_item = {
                **d, # Start with all key-value pairs from the original item
                "id": id, # Overwrite the id with the new pass-k specific id
                f"{args.model_nickname}_response": f"<think>{reasoning_content}</think>{response_content}",
                f"{args.model_nickname}_correctness": correctness,
                f"{args.model_nickname}_pred_answer": pred_answer,
            }
            results_lst.append(result_item)
            
            if len(results_lst)%args.save_step == 0:
               save_json(data=results_lst, file_path=save_path)
      except Exception as e:
         print(e)
         pass
   map_with_progress(f=fn, xs=data, num_threads=50)
   save_json(data=results_lst, file_path=save_path)
   
   # Calculate and display evaluation statistics using the reusable reporting utility
   stats_dict, report_path = quick_report(
      results_lst=results_lst,
      model_nickname=args.model_nickname,
      passk=args.passk,
      original_count=len(data),
      save_path=save_path,
      current_time=current_time,
      tool_retrieval_enabled=False  # CoT evaluation doesn't use tool retrieval
   )
   
   print(f"\n🎉 CoT Evaluation complete!")
   print(f"📊 Results saved to: {save_path}")
   print(f"📈 Statistics: {save_path.replace('.json', '_stats.json')}")
   print(f"📝 Report: {report_path}")

