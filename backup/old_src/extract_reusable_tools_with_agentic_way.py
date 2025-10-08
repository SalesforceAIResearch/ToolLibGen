from datetime import datetime
from utils import read_json,call_openai_api,map_with_progress,save_json,execute_code,call_openai_api_multi_turn,call_deepseek_r1_32b_api
import argparse
from prompt import TOOLKIT_EXTRACT_PROMPT_AGENTIC_V2, TOOLKIT_EXTRACT_PROMPT_AGENTIC_VERIFICATION, CHECKING_CODE_EXECUTION_PROMPT
import re


def extract_strings_without_line_break(s: str):
   return s.replace("\n", "").replace("\r", "")

def is_code_valid(tool: str):
   return execute_code(tool) is not None

def create_new_tools(messages: list):
   response = call_openai_api_multi_turn(model_name="o4-mini", messages=messages)
   failed_times = 0
   code_lst = extract_all_code_in_format(response)
   while failed_times < 3:
      failed_code_lst = []
      for code in code_lst:
         if not is_code_valid(code):
            failed_code_lst.append(code)
      if len(failed_code_lst) == 0:
         break
      messages.append({"role": "user", "content": f"The code: {failed_code_lst}\nis invalid. Please try to generate all content for the intial request again."})
      response = call_openai_api_multi_turn(model_name="o4-mini", messages=messages)
      code_lst = extract_all_code_in_format(response)
      failed_times += 1
   messages.append({"role": "assistant", "content": response})
   return messages

def is_code_correct_in_answering(response: str):
   """extract all code in the response with <code> ... </code> tags. We can have multiple code blocks with <code> ... </code> tags in the response."""
   try:
      execution_result_lst = []
      code_lst = extract_all_code_in_format(response)
      output_lst = extract_all_output_in_format(response)
      if len(code_lst) != len(output_lst):
         return f"The number of code and output is not the same"
      for code, output in zip(code_lst, output_lst):
         code_output = execute_code(code)
         execution_result_lst.append(code_output)
         if code_output is None:
            return f"code {code} cannot be executed"
         if extract_strings_without_line_break(code_output) != extract_strings_without_line_break(output):
            if not is_code_execution_semantic_similarity(code_output, output):
               return f"code output of {code_output} does not match the expected output {output}"
      return f"Correct! Code output of {execution_result_lst} matches the expected output {output_lst}"
   except:
      return "Error in code execution"

def is_code_execution_semantic_similarity(answer_1: str, answer_2: str):
   prompt = CHECKING_CODE_EXECUTION_PROMPT.format(answer_1=answer_1, answer_2=answer_2)
   messages = [{"role": "user", "content": prompt}]
   response = call_openai_api_multi_turn(model_name="gpt-4.1", messages=messages)
   return "Yes" in response

def validate_tools(question, toolset: list):
   verification_prompt = TOOLKIT_EXTRACT_PROMPT_AGENTIC_VERIFICATION.format(question=question, tools="\n".join(toolset))
   messages = [{"role": "user", "content": verification_prompt}]
   response = call_openai_api_multi_turn(model_name="o4-mini", messages=messages)
   messages.append({"role": "assistant", "content": response})
   return messages, is_code_correct_in_answering(response)

def extract_text_within_start_and_end_tags(text: str, start_tag: str, end_tag: str)->list:
   """extract all text between start_tag and end_tag in the text, return a list of text"""
   return re.findall(f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}", text, re.DOTALL)

def extract_all_tools_in_format(response: str):
   return extract_text_within_start_and_end_tags(response, "<tool>", "</tool>")

def extract_all_code_in_format(response: str):
   return extract_text_within_start_and_end_tags(response, "<code>", "</code>")

def extract_all_output_in_format(response: str):
   return extract_text_within_start_and_end_tags(response, "<output>", "</output>")


if __name__ == "__main__":
   n = 2000
   parser = argparse.ArgumentParser()
   parser.add_argument("--file_path", type=str, default="train_openr1_sci_10k.json")
   parser.add_argument("--save_path", type=str, default=f"train_openr1_sci_10k_tools_o4_mini_{n}_agentic_way_v2.json")
   parser.add_argument("--remote_mode", action="store_true", default=False)
   args = parser.parse_args()
   if args.remote_mode:
      data_folder = "/export/home/data/"
   else:
      data_folder = "/Users/murong.yue/Desktop/data/openr1_data_sci/"
   datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
   file_path = data_folder + args.file_path
   save_path = data_folder + args.save_path
   print(f"The current read file path is {file_path} and the current save path is {save_path}")
   save_path = save_path.replace(".json", f"_{datetime_str}.json")
   data = read_json(file_path)
   data = data[:n]
   res_lst = []
   index = 0
   max_index = 30
   for d in data:
      print(f"Processing {index}/{max_index}")
      if index >= max_index:
         break
      index += 1
      question = d["question"]
      reasoning = d["reasoning"][0]
      answer = d["GPT_answer"]
      messages = [{"role": "system", "content": "You are an expert chemistry/physics tutor and a good tool user/creator."}]
      messages+= [{"role": "user", "content": TOOLKIT_EXTRACT_PROMPT_AGENTIC_V2.format(question=question, answer=reasoning)}]
      messages = create_new_tools(messages)
      d['tool_extraction_conversation'] = messages
      tool_lst = extract_all_tools_in_format(messages[-1]["content"])
      validation_messages, validation_result = validate_tools(question, tool_lst)
      d[f"tool_validation_conversation"] = validation_messages
      if "Correct" in validation_result and validation_messages[-1]["content"].split("<final_choice>")[1].split("</final_choice>")[0].lower() == answer.lower():
         d["tools"] = tool_lst
      res_lst.append(d)
   save_json(res_lst, save_path)

   def fn(d):
      global res_lst
      try:
         question = d["question"]
         reasoning = d["reasoning"][0]
         answer = d["GPT_answer"]
         messages = [{"role": "system", "content": "You are an expert chemistry/physics tutor and a good tool user/creator."}]
         messages+= [{"role": "user", "content": TOOLKIT_EXTRACT_PROMPT_AGENTIC_V2.format(question=question, answer=reasoning)}]
         messages = create_new_tools(messages)
         d['tool_extraction_conversation'] = messages
         tool_lst = extract_all_tools_in_format(messages[-1]["content"])
         validation_messages, validation_result = validate_tools(question, tool_lst)
         d[f"tool_validation_conversation"] = validation_messages
         if "Correct" in validation_result:
            if answer in validation_messages[-1]["content"].split("<final_choice>")[1].split("</final_choice>")[0]:
               d["tools"] = tool_lst
               d["error_reason"] = ""
            else:
               d["tools"] = []
               print(f"Answer Error: GPT {answer} is not equal to the final choice {validation_messages[-1]['content'].split('<final_choice>')[1].split('</final_choice>')[0]}")
               d["error_reason"] = f"Answer Error: {answer} is not in the final choice"
         else:
            d["tools"] = []
            print(f"Validation Error: {validation_result}")
            d["error_reason"] = validation_result
         res_lst.append(d)
      except Exception as e:
         print(f"Error: {e}")
         return
      if len(res_lst) % 100 == 0:
         save_json(res_lst, save_path)

   map_with_progress(fn, data, num_threads=20)
   save_json(res_lst, save_path)