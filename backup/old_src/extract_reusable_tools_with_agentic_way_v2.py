from datetime import datetime
from utils import read_json, call_openai_api, map_with_progress, save_json, execute_code, call_openai_api_multi_turn
import argparse
from ToolExtractionAgent import ToolExtractionAgent
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def extract_strings_without_line_break(s: str):
   return s.replace("\n", "").replace("\r", "")

def is_code_valid(tool: str):
   return execute_code(tool) is not None

def is_code_execution_semantic_similarity(answer_1: str, answer_2: str):
   from prompt import CHECKING_CODE_EXECUTION_PROMPT
   prompt = CHECKING_CODE_EXECUTION_PROMPT.format(answer_1=answer_1, answer_2=answer_2)
   messages = [{"role": "user", "content": prompt}]
   response = call_openai_api_multi_turn(model_name="gpt-4.1", messages=messages)
   return "Yes" in response


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--file_path", type=str, default="filtered_all_science_data.json")
   parser.add_argument("--save_path", type=str, default=f"Nemotron_science_data_tools.json")
   parser.add_argument("--remote_mode", action="store_true", default=False)
   parser.add_argument("--debug", action="store_true", default=False)
   args = parser.parse_args()
   if args.remote_mode:
      data_folder = "/export/home/data/"
   else:
      data_folder = "/Users/murong.yue/Desktop/data/"
   datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
   file_path = data_folder + args.file_path
   save_path = data_folder + args.save_path
   print(f"The current read file path is {file_path} and the current save path is {save_path}")
   data = read_json(file_path)
   if args.debug:
      n = 3
      data = data[:n]
      save_path = save_path.replace(".json", f"_{datetime_str}_debug.json")
   else:
      save_path = save_path.replace(".json", f"_{datetime_str}.json")
   res_lst = []
   
   # Initialize ToolExtractionAgent with only gpt-4.1 for verification
   # agent = ToolExtractionAgent(
   #    generation_model_name="o4-mini",
   #    verification_model_name_lst=["gpt-4.1","/export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/"]
   # )
   agent = ToolExtractionAgent(
      generation_model_name="o4-mini",
      verification_model_name_lst=["gpt-4.1"],
      skip_on_validation_failure=False,
      code_validation_timeout=1
   )
   res_lst = []
   def fn(d):
      global res_lst
      try:
         question = d["question"]
         reasoning = d["reasoning"][0] if isinstance(d["reasoning"], list) else d["reasoning"]
         answer = d["answer"]
         # Use ToolExtractionAgent for agentic tool generation and validation with a 10-minute timeout
         executor = ThreadPoolExecutor(max_workers=1)
         future = executor.submit(
            agent.run_agent_system,
            question=question,
            CoT=reasoning,
            answer=answer,
            max_iterations=0,
            evaluation_repeats=1
         )
         _timed_out = False
         try:
            result = future.result(timeout=600)
         except TimeoutError:
            _timed_out = True
            print(f"[Timeout] Case exceeded 10 minutes. Skipping. Question: {extract_strings_without_line_break(str(question))[:10]}")
            executor.shutdown(wait=False, cancel_futures=True)
            return
         finally:
            if not _timed_out:
               executor.shutdown(wait=True, cancel_futures=True)
         
         # Store the complete agent result (full interaction history)
         d['agent_result'] = result
         # d['trajectory'] = result['trajectory']
         
         # Extract tools if the agent was successful
         if result['status'] == 'success' and result['final_tools']:
            d["tools"] = [tool.get('code', '') for tool in result['final_tools']]
            d["error_reason"] = ""
         else:
            d["tools"] = []
            d["error_reason"] = f"Agent failed: {result.get('message', 'Unknown error')}"
         
         if args.debug:
            print(f"Agent status: {result['status']}, Tools extracted: {len(d['tools'])}")
         res_lst.append(d)
      except Exception as e:
         if args.debug:
            print(f"Error processing item: {e}")
         d["tools"] = []
         d["error_reason"] = f"Exception: {str(e)}"
         res_lst.append(d)

      if len(res_lst) % 200 == 0:
         save_json(res_lst, save_path)


   # Use lower thread count to prevent potential issues, but keep it simple
   thread_count = 20 if args.debug else 50  # Reduced from 200
   print(f"Starting processing with {thread_count} threads...")
   
   # Use map_with_progress for parallel processing
   map_with_progress(fn, data, num_threads=thread_count)
   
   # Filter out None results and save
   save_json(res_lst, save_path)
   print(f"Processing completed. Saved {len(res_lst)} results to {save_path}")