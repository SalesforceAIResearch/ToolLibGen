import json
from typing import Dict, List, Any
from utils import call_openai_with_temporary_tool, apply_patch, call_openai_api_multi_turn, LLMCaller, call_sfr_embedding_api
import sys
import numpy as np


class IterativeLibraryOptimizerAgent:
    """
    An AI Agent that iteratively optimizes a Python tool library through stronger LLM reasoning
    and weaker LLM testing to make it more helpful for answering specific questions.
    """

    def __init__(
            self,
            stronger_llm_model: str = "o3",
            weaker_llm_model_list: List[str] = ["gpt-4.1"],
            max_iterations: int = 12,
            tools: List[Dict] = [],
            question: str = "",
            ground_truth: str = ""
    ):
        """
        Initialize the IterativeLibraryOptimizerAgent.
        
        Args:
            stronger_llm_model: Model for the stronger LLM (o3)
            weaker_llm_model: Model for the weaker LLM (GPT-4.1)
            max_iterations: Maximum number of total iterations
        """
        self.stronger_llm_model = stronger_llm_model
        self.weaker_llm_model_list = weaker_llm_model_list
        self.weaker_llm_model = weaker_llm_model_list[0] if weaker_llm_model_list else "gpt-4.1"
        self.max_iterations = max_iterations
        self.tools = tools
        self.question = question
        self.ground_truth = ground_truth

    def optimize_library_directly(self):
        """
        Optimize the library directly without the iterative process.
        """
        initial_prompt = f"""
**Public Available Tools in tool calling format**: {len(self.tools)} tools
{json.dumps(self.tools, indent=2)}

**Optimization Task:**
Question: {self.question}
Ground Truth: {self.ground_truth}
Think about if any current tools are useful for this problem. If you believe that any current tool are very useful for helping the weaker LLM to analyze this problem, you should mark it as PASS. Otherwise, you should mark it as NEED_PATCHING.
You should only output the final report in the <final_report> tag.
<final_report>
{{
    "is_library_helpful": "PASS" or "NEED_PATCHING",
    "reason": "Brief reason for your conclusion.",
    "modification_suggestions": "Description of the modification when NEED_PATCHING. When making suggestions for patching, you must prioritize modifying the existing function. Only when you are absolutely certain that modifying the current function cannot provide a tool to address the present issue should you propose adding a new function. When adding new functions, do not introduce a general-purpose function (e.g., PhysicsToolkit), and do not reuse any legacy function names. Your modification suggestions should be precise."
}}
</final_report>
"""
        try:
            messages = [{"role": "user", "content": initial_prompt}]
            response = call_openai_api_multi_turn(
                messages=messages,
                model_name=self.stronger_llm_model,
            )
            messages.append({"role": "assistant", "content": response})
            return messages
        except Exception as e:
            return f"Error in calling optimization agent: {e}"



    def optimize_library_with_running_weaker_llm(self):
        """
        Optimize the library by running weaker LLM with tools, then having stronger LLM analyze the results.
        
        Two-step process:
        1. Weaker LLM attempts to answer the question using provided tools
        2. Stronger LLM analyzes the weaker LLM's performance and provides final report
        """
        # Step 1: Let weaker LLM attempt to answer the question with tools
        weaker_llm_prompt = f"""
Question: {self.question}

Please answer the question using the provided tools. Think step by step and use the tools as needed to solve the problem.
"""
        
        # Initialize variables for error handling
        weaker_messages = []
        weaker_final_response = ""
        weaker_llm_error = None
        
        try:
            # Create LLMCaller for weaker LLM
            weaker_caller = LLMCaller(
                model_name=self.weaker_llm_model,
                temperature=0.7,
                max_tokens=4096
            )
            
            # Extract tool_info and create function_registry from tools
            openai_tools = []
            function_registry = {}
            
            for tool in self.tools:
                if isinstance(tool, dict) and 'tool_info' in tool and 'tool_code' in tool:
                    tool_info = tool['tool_info']
                    function_name = tool_info.get('function', {}).get('name', '')
                    
                    if function_name:
                        openai_tools.append(tool_info)
                        
                        # Use the existing create_executable_function logic from eval
                        executable_func = self._create_executable_function(tool)
                        if executable_func:
                            function_registry[function_name] = executable_func
                else:
                    # If already in OpenAI format (fallback)
                    openai_tools.append(tool)
                        
            # Call weaker LLM with tools and executable functions
            weaker_messages, weaker_turns = weaker_caller.call_with_static_tools(
                content=weaker_llm_prompt,
                tools=openai_tools,
                function_registry=function_registry,  # Now tools can actually execute
                return_format="messages_turns",
                max_turns=3  # Allow more turns for thorough tool usage testing
            )
            
            # Extract weaker LLM's final response
            for msg in reversed(weaker_messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    weaker_final_response = msg["content"]
                    break
                    
        except Exception as e:
            # Capture the error but don't return - continue to analysis step
            weaker_llm_error = str(e)            
            # Create minimal error messages for analysis
            weaker_messages = [
                {"role": "user", "content": weaker_llm_prompt},
                {"role": "assistant", "content": f"Error occurred during tool usage: {weaker_llm_error}"}
            ]
            weaker_final_response = f"Error: {weaker_llm_error}"
            
        # Step 2: Stronger LLM analyzes weaker LLM's performance (including errors)
        analysis_prompt = f"""
You are analyzing whether the current tool library is helpful for a weaker LLM to solve problems.

**Original Question:**
{self.question}

**Ground Truth Answer:**
{self.ground_truth}

**Available Tools:**
{json.dumps(self.tools, indent=2)}

**Weaker LLM's Complete Conversation:**
{self._format_conversation_for_analysis(weaker_messages)}

**Weaker LLM's Final Answer:**
{weaker_final_response}

**Error Information:**
{f"ERROR OCCURRED: {weaker_llm_error}" if weaker_llm_error else "No errors occurred during execution."}

**Your Task:**
Analyze whether the weaker LLM was able to effectively use the provided tools to solve the problem. Consider:
1. Did the weaker LLM use the tools appropriately?
2. Did the tools provide sufficient functionality for the problem?
3. Was the final answer correct or close to the ground truth?
4. If errors occurred, what caused them and how can the tools be improved to prevent them?
5. What specific improvements would help the weaker LLM succeed?

**Important:** Even if errors occurred, focus on how to improve the tools to make them more robust and user-friendly for the weaker LLM.

You should only output the final report in the <final_report> tag.

<final_report>
{{
    "is_library_helpful": "PASS" or "NEED_PATCHING",
    "reason": "Detailed analysis of the weaker LLM's performance and tool usage. Include error analysis if applicable. Explain what worked well and what didn't.",
    "modification_suggestions": "Specific suggestions for improving the tools when NEED_PATCHING. If errors occurred, explain how to make tools more robust. Prioritize modifying existing functions over adding new ones. Be very detailed and precise about what changes would help the weaker LLM succeed."
}}
</final_report>
"""

        # Continue with analysis regardless of whether errors occurred in Step 1
        try:
            # Create LLMCaller for stronger LLM
            stronger_caller = LLMCaller(
                model_name=self.stronger_llm_model,
                temperature=0.3,
                max_tokens=4096
            )
            
            # Get stronger LLM's analysis
            analysis_response = stronger_caller.call(
                content=analysis_prompt,
                return_format="content"
            )
            
            # Return the stronger LLM's analysis as the main conversation
            # The weaker LLM's interaction is embedded in the analysis prompt
            analysis_messages = [
                {"role": "user", "content": analysis_prompt},
                {"role": "assistant", "content": analysis_response}
            ]
            
            return analysis_messages
            
        except Exception as e:            
            # Return basic error report
            error_report = f"""
Error occurred during analysis phase: {e}

Weaker LLM execution error: {weaker_llm_error or "None"}

<final_report>
{{
    "is_library_helpful": "NEED_PATCHING",
    "reason": "Analysis failed due to system error. Weaker LLM error: {weaker_llm_error or 'None'}. Analysis error: {e}",
    "modification_suggestions": "System encountered errors during testing. Review tool implementations and error handling. Ensure tools are robust and handle edge cases properly."
}}
</final_report>
"""
            
            return [
                {"role": "user", "content": analysis_prompt},
                {"role": "assistant", "content": error_report}
            ]
    
    def _format_conversation_for_analysis(self, messages: List[Dict]) -> str:
        """Format the conversation messages for analysis by the stronger LLM."""
        formatted_parts = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                formatted_parts.append(f"**User:** {content}")
            elif role == "assistant":
                formatted_parts.append(f"**Assistant:** {content}")
                
                # Check for tool calls
                if tool_calls := msg.get("tool_calls"):
                    for tool_call in tool_calls:
                        if fn_call := tool_call.get("function"):
                            fn_name = fn_call.get("name", "unknown")
                            fn_args = fn_call.get("arguments", "{}")
                            formatted_parts.append(f"  🔧 Called tool: {fn_name}({fn_args})")
                            
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "unknown")
                formatted_parts.append(f"  📋 Tool result: {content}")
        
        return "\n\n".join(formatted_parts)
    
    def _create_executable_function(self, tool_data: dict) -> callable:
        """
        Create an executable function from tool data containing Python code.
        Based on the implementation from eval/library_flatten_eval_v2.py
        
        Args:
            tool_data: Tool data containing tool_code with Python implementation
            
        Returns:
            Callable function or None if creation fails
        """
        try:
            tool_code = tool_data.get("tool_code", "")
            if not tool_code:
                return None
            
            # Extract Python code from markdown code blocks if present
            if "```python" in tool_code:
                code_start = tool_code.find("```python") + 9
                code_end = tool_code.find("```", code_start)
                if code_end != -1:
                    actual_code = tool_code[code_start:code_end].strip()
                else:
                    actual_code = tool_code[code_start:].strip()
            else:
                actual_code = tool_code
            
            # Create a temporary module to execute the code
            temp_module = {}
            
            # Add common imports that might be needed
            exec("""
import numpy as np
import math
import json
import os
import sys
from typing import *
import requests
try:
    from PIL import Image, ImageFilter
except ImportError:
    pass
from io import BytesIO
import re
""", temp_module)
            
            # Execute the tool code
            exec(actual_code, temp_module)
            
            # Look for 'execute' function specifically (v_2.json format)
            if 'execute' in temp_module and callable(temp_module['execute']):
                return temp_module['execute']
            
            # Fallback: look for any function that matches the tool name
            tool_name = tool_data.get('tool_info', {}).get('function', {}).get('name', '')
            if tool_name in temp_module and callable(temp_module[tool_name]):
                return temp_module[tool_name]
            
            # Last resort: find any callable function (excluding built-ins)
            for name, obj in temp_module.items():
                if callable(obj) and not name.startswith('__') and name not in ['np', 'math', 'json', 'os', 'sys', 'requests', 'Image', 'ImageFilter', 'BytesIO', 're']:
                    return obj
            
            return None
            
        except Exception as e:
            return None




#     def optimize_library_agent(
#             self,
#     ):
#         """
#         Main entry point for library optimization.
#         Args:
#             python_library: The Python library code
#             tools: List of tool definitions in OpenAI format
#             question: The question to optimize the library for
            
#         Returns:
#             Dict containing optimization results and suggestions
#         """
#         # Define tools available to the stronger LLM

#         # Create tool registry for the stronger LLM
#         tool_registry = {
#             "test_library": self._test_library_tool,
#             "revise_current_python_tool": self._revise_current_python_tool,
#             "add_new_python_tool": self._add_new_python_tool
#         }

#         # Initial prompt for the stronger LLM
#         initial_prompt = f"""
# **Current Library:**
# Python Library:
# ```python
# {self.python_library}
# ```

# **Public Available Tools in tool calling format**: {len(self.tools)} tools
# {json.dumps(self.tools, indent=2)}

# **Optimization Task:**
# Question: {self.question}
# Ground Truth: {self.ground_truth}

# **Your Task:**
# Currently, these tools are public functions within Python libraries, and their goal is to solve problems. Now, for the current problem, you need to follow these steps:

# **Instructions:**
# ### Step 1: Thinking and Analysis
# **Your Task:**
# 1.  Analyze the user's question and the available tools.
# 2.  Determine if the existing tools are sufficient for a weaker LLM to solve the problem.
# 3.  Conclude this step by explicitly stating whether you will proceed to Step 2 (Patching) or Step 3 (Testing).

# **Your Output for this Step:**
# [Your detailed analysis here. Explain your reasoning.]
# **Conclusion:** I have determined that the current tools [are/are not] sufficient.

# ---

# ### Step 2: Patching Current Tools
# If the available tools cannot sufficiently help the weaker LLM solve the problem, or if there are errors in the current implementation, reflect on the deficiencies and consider how to improve the available tools and the Python library. Use the relevant tools to modify the existing available tools accordingly. If the modification fails, use the error message to revise and try again. Possible modifications include: editing an existing Python tool and its corresponding OpenAI tool-calling description, or adding a new public Python tool implementation along with its OpenAI tool-calling description.
# **Your Task:**
# * Only perform this step if you concluded it was necessary in Step 1.
# * If not skipping, describe the necessary modifications (editing or adding tools). Provide the code and JSON descriptions for the patch.
# * Call the `revise_current_python_tool` tool to revise the current tools or `add_new_python_tool` tool to add a new tool to the library.
# * If you receive the error message from the tool creating or revising, you should carefully revise the current input parameters to the tools to avoid the error and then call these two tools again.

# ---

# ### Step 3: Testing the Weaker LLM
# When you believe there are no issues with the current tools, let the weaker LLM test whether, without access to the answer, it can correctly answer the question and invoke the appropriate tool. Note that you do not need to provide all tools to the weaker LLM, only those you believe to be the most effective in assisting analysis. In this step, call the test_library tool to test the library.
# **Your Task:**
# * Call the `test_library` tool to simulate the weaker LLM's attempt to solve the problem using the (potentially patched) tools.


# ---

# ### Step 4: Reflecting
# You must complete Step 3 before this step. Review the entire message from Step 3 and reflect on:
# Whether the weaker LLM made correct use of the provided tool(s), including whether the usage scenario was appropriate and whether parameters were correctly passed.
# Whether the weaker LLM arrived at the correct answer by leveraging the tool’s conclusions. If either criterion is not met, return to Step 2 and consider how to patch the current tools. Note that the weaker LLM is fixed and cannot be modified, whereas the tool library is modifiable; therefore, you should modify the tool library to suit the weaker LLM. Repeat Steps 2–4 until you believe the weaker LLM is well-equipped to answer the question.
# **Your Task:**
# 1.  Analyze the output from Step 3.
# 2.  Determine if the weaker LLM used the tools correctly AND if it reached the correct answer.
# 3.  If both conditions are met, you may proceed to Step 5.
# 4.  If either condition fails, you MUST state that you are returning to Step 2 to refine the tools. **In this scenario, you will not stop and not generate the final report.**

# **Your Output for this Step:**
# <reflection>
# [Your detailed reflection on the weaker LLM's performance.]
# **Conclusion:** The weaker LLM [succeeded / failed].
# **Next Action:** [Proceeding to Step 5: Final Report Generating / Returning to Step 2 for another iteration.]
# </reflection>

# ---

# ### Step 5: Final Report Generating
# Whether the current library needs modification for this problem. Only if you believe the current tool library includes tools suitable for this problem and the weaker LLM can solve it, mark it as PASS; otherwise, mark it as NEED_PATCHING.
# Modification suggestions. Based on your experiences so far, you should already know which types of modifications are effective or ineffective. Provide comprehensive recommendations, including both effective and ineffective modification attempts, to aid future LLM modifications. The final output should be in JSON format:
# **Your Task:**
# * **Only perform this step if your conclusion in Step 4 was to proceed.**
# * Generate the final JSON report based on your entire analysis.

# **Your Output for this Step:**
# <final_report>
# {{
#     "is_library_helpful": "PASS" or "NEED_PATCHING",
#     "modification_suggestions": "Description of the modification when NEED_PATCHING. Otherwise, leave it empty. Your modification suggestions should be very detailed and precise."
# }}
# </final_report>
# """

#         try:
#             # Let the stronger LLM work with the tools
#             messages, turns = call_openai_with_temporary_tool(
#                 messages=[{"role": "user", "content": initial_prompt}],
#                 tools=self.OPTIMIZATION_TOOLS,
#                 function_registry=tool_registry,
#                 model_name=self.stronger_llm_model,
#                 completion_check=self._check_completion,
#                 prerequisite_code="",
#                 max_turns=self.max_iterations
#             )
#             if "<final_report>" not in messages[-1]["content"]:
#                 messages.append({"role": "user",
#                                  "content": "You must generate the final report in the <final_report> tag. If you haven't found the most effective way to add the tools, you can share your thoughts about what is not a good way and propose some suggestions."})
#                 final_report = call_openai_api_multi_turn(
#                     messages=messages,
#                     model_name=self.stronger_llm_model
#                 )
#                 messages.append({"role": "assistant", "content": final_report})
#             return messages
#         except Exception as e:
#             return f"Error in calling optimization agent: {e}"


    def _get_weak_llm_response(self, question: str):
        """
        Get response from weaker LLM.
        """
        weak_llm_caller = LLMCaller(model_name=self.weaker_llm_model)
        prompt = f"""
Question: {question}
You are a tool operator. Your task is to solve the following question using the provided tools.
You must use the provide tools to help the analysis. Final result should start with "Final Answer:".
"""
        pass


    def optimize_library_agent(self):
        """
        Main entry point for library optimization.
        """
        weak_llm_caller = LLMCaller(model_name=self.weaker_llm_model)
        prompt = """
Question: {self.question}
"""



    def _test_library_tool(self, question: str, useful_tool_name: List[str], reason: str) -> str:
        """Tool function for testing the library."""

        try:
            # Filter tools by useful_tool_name
            selected_tools = []
            for tool in self.tools:
                tool_name = tool.get("tool_info", {}).get("function", {}).get("name", "")
                if tool_name in useful_tool_name:
                    selected_tools.append(tool)

            if not selected_tools:
                result = f"No tools found matching names: {useful_tool_name} or the useful_tool_name is empty. You must find some useful tools before starting to solve the problem. If you think the current tools are not related enough to help analyze this problem, you should first add some new tools to the library."
                return result

            # Create tool registry for testing
            tool_info, tool_code = self._create_tool_registry(
                selected_tools,
                self.python_library
            )

            if not tool_info:
                result = "No tools available for testing"
                return result

            # Test with weaker LLM
            test_prompt = f"""
You are a tool operator. Your task is to solve the following question using the provided tools.

**Question to solve:**
{question}
            
You must call tools and analyze the result. Any answer without calling tools is not allowed. Your final answer should be like this: Final Answer: <answer>.
            """

            messages, turns = call_openai_with_temporary_tool(
                messages=[{"role": "user", "content": test_prompt}],
                tools=tool_info,
                function_registry=tool_code,
                model_name=self.weaker_llm_model,
                completion_check=lambda content: "Final Answer:" in content,
                prerequisite_code=self.python_library
            )
            return json.dumps(messages, indent=2)

        except Exception as e:
            error_result = f"Test failed: {str(e)}"
            return error_result

    def _revise_current_python_tool(self, library_code_diff: str, revised_tool_calling_info: str,
                                    revised_tool_calling_implementation: str, reason: str) -> str:
        """Tool function for patching and modifying current tools."""
        try:
            # Apply the patch to the current library
            patched_library_code = apply_patch(self.python_library, library_code_diff)

            # Check if patch application was successful
            if patched_library_code is None:
                return f"Error: Failed to apply patch to library. Original library size: {len(self.python_library)}\nDiff content:\n{library_code_diff}"

            # Validate the patched code before applying it
            try:
                compile(patched_library_code, '<string>', 'exec')
            except SyntaxError as e:
                return f"Error: Patched library has syntax error at line {e.lineno}: {e.msg}\nOriginal library size: {len(self.python_library)}\nPatched library size: {len(patched_library_code)}\nDiff content:\n{library_code_diff}"

            # Only apply the patch if validation passes
            self.python_library = patched_library_code

            # Parse the revised tool calling JSON
            tool_info_json_str = revised_tool_calling_info

            # Extract tool code with better error handling
            try:
                if "<code>" in revised_tool_calling_implementation and "</code>" in revised_tool_calling_implementation:
                    tool_code_str = revised_tool_calling_implementation.split("<code>")[1].split("</code>")[0]
                else:
                    # If no code tags, assume the entire string is the code
                    tool_code_str = revised_tool_calling_implementation

                # Clean up the extracted code
                tool_code_str = tool_code_str.strip()

                # Basic syntax validation
                try:
                    compile(tool_code_str, '<string>', 'exec')
                except SyntaxError as e:
                    return f"Error: Tool code has syntax error at line {e.lineno}: {e.msg}\nProblematic code:\n{tool_code_str}"

            except Exception as e:
                return f"Error extracting tool code: {str(e)}"

            # Clean up the JSON string - handle common problematic characters
            revised_tool_data = {"tool_info": json.loads(tool_info_json_str), "tool_code": tool_code_str}

            # Find the tool with the same name and replace it
            tool_name = revised_tool_data.get("tool_info", {}).get("function", {}).get("name", "")
            if not tool_name:
                return "Error: No tool name found in revised tool calling"

            tool_found = False
            for i, tool in enumerate(self.tools):
                existing_tool_name = tool.get("tool_info", {}).get("function", {}).get("name", "")
                if existing_tool_name == tool_name:
                    self.tools[i] = revised_tool_data
                    tool_found = True
                    break

            if not tool_found:
                return f"Error: Tool '{tool_name}' not found in current tools"

            result = f"Tool '{tool_name}' revised successfully\n"
            result += f"Revised tool calling schema: {revised_tool_calling_info}\n"
            result += f"Revised tool calling implementation: {revised_tool_calling_implementation}\n"
            result += f"Patched library size: {len(self.python_library)}\n"

            return result

        except Exception as e:
            return f"Failed to revise tool: {str(e)}"

    def _add_new_python_tool(self, library_code_diff: str, new_tool_calling_info: str,
                             new_tool_calling_implementation: str, reason: str) -> str:
        """Tool function for adding a new public tool to the library."""
        try:
            # Apply the patch to the current library
            patched_library_code = apply_patch(self.python_library, library_code_diff)

            # Check if patch application was successful
            if patched_library_code is None:
                return f"Error: Failed to apply patch to library. Original library size: {len(self.python_library)}\nDiff content:\n{library_code_diff}"

            # Validate the patched code before applying it
            try:
                compile(patched_library_code, '<string>', 'exec')
            except SyntaxError as e:
                return f"Error: Patched library has syntax error at line {e.lineno}: {e.msg}\nOriginal library size: {len(self.python_library)}\nPatched library size: {len(patched_library_code)}\nDiff content:\n{library_code_diff}"

            # Only apply the patch if validation passes
            self.python_library = patched_library_code

            # Parse the new tool calling JSON
            tool_info_json_str = new_tool_calling_info

            # Extract tool code with better error handling
            try:
                if "<code>" in new_tool_calling_implementation and "</code>" in new_tool_calling_implementation:
                    tool_code_str = new_tool_calling_implementation.split("<code>")[1].split("</code>")[0]
                else:
                    # If no code tags, assume the entire string is the code
                    tool_code_str = new_tool_calling_implementation

                # Clean up the extracted code
                tool_code_str = tool_code_str.strip()

                # Basic syntax validation
                try:
                    compile(tool_code_str, '<string>', 'exec')
                except SyntaxError as e:
                    return f"Error: Tool code has syntax error at line {e.lineno}: {e.msg}\nProblematic code:\n{tool_code_str}"

            except Exception as e:
                return f"Error extracting tool code: {str(e)}"

            new_tool_data = {"tool_info": json.loads(tool_info_json_str), "tool_code": tool_code_str}
            # Append the new tool to the tools list
            self.tools.append(new_tool_data)

            tool_name = new_tool_data.get("tool_info", {}).get("function", {}).get("name", "unknown")
            result = f"New tool '{tool_name}' added successfully\n"
            result += f"Total tools: {len(self.tools)}\n"
            result += f"New tool calling schema: {new_tool_calling_info}\n"
            result += f"New tool calling implementation: {new_tool_calling_implementation}\n"
            result += f"Patched library size: {len(self.python_library)}\n"

            return result

        except Exception as e:
            return f"Failed to add new tool: {str(e)}"

    def _check_completion(self, content: str) -> bool:
        """Check if the optimization is complete."""
        return "<final_report>" in content

    def _create_tool_registry(self, selected_tools, pre_code):
        """Create tool registry for testing (enhanced with better error handling)."""
        from io import StringIO

        def create_executor(tool_code_string, pre_code):
            def executor(**kwargs):
                try:
                    exec_env = {}
                    exec(pre_code, exec_env)

                    # Clean up tool code string
                    if "```python" in tool_code_string:
                        python_code = tool_code_string.split("```python")[1].split("```")[0].strip()
                    else:
                        python_code = tool_code_string.strip()

                    # Additional cleaning for common issues
                    python_code = python_code.replace('\r\n', '\n').replace('\r', '\n')

                    # Remove any trailing incomplete lines or characters
                    lines = python_code.split('\n')
                    clean_lines = []
                    for line in lines:
                        # Skip empty lines and comments only
                        if line.strip() and not line.strip().startswith('#'):
                            clean_lines.append(line)
                        elif line.strip() == '':
                            clean_lines.append(line)

                    python_code = '\n'.join(clean_lines)

                    # Validate syntax before execution
                    try:
                        compile(python_code, '<string>', 'exec')
                    except SyntaxError as e:
                        return f"Error: Tool code has syntax error at line {e.lineno}: {e.msg}\nProblematic code:\n{python_code}"

                    # Execute the function definition
                    exec(python_code, exec_env)

                    # Check if execute function is defined
                    if 'execute' not in exec_env:
                        return f"Error: 'execute' function not found in tool code. Available functions: {list(exec_env.keys())}"

                    old_stdout = sys.stdout
                    sys.stdout = captured_output = StringIO()

                    try:
                        result = exec_env['execute'](**kwargs)
                        output = captured_output.getvalue().strip()
                        return output if output else str(result)
                    finally:
                        sys.stdout = old_stdout

                except Exception as e:
                    return f"Error executing tool: {str(e)}"

            return executor

        tool_info = []
        tool_code = {}

        for tool in selected_tools:
            if "tool_info" in tool and "tool_code" in tool:
                tool_name = tool["tool_info"]["function"]["name"]
                tool_code_string = tool["tool_code"]

                tool_info.append(tool["tool_info"])
                executor = create_executor(tool_code_string, pre_code)
                tool_code[tool_name] = executor

        return tool_info, tool_code
