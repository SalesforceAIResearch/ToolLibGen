import code
import re
import ast
import json
from typing import List, Dict, Any, Tuple, Optional
from prompt import (
    TOOL_EXTRACT_AGENT_AGNTIC_GENERATION_PROMPT,
    TOOL_EVALUATION_PROBLEM_SOLVING_PROMPT,
    TOOL_EFFECTIVENESS_EVALUATION_PROMPT,
    TOOL_SYNTAX_ERROR_REFINEMENT_PROMPT,
    TOOL_EFFECTIVENESS_REFINEMENT_PROMPT
)
from utils import LLMCaller, validate_tools_syntax, code_comment2function_json

class ToolExtractionAgent:
    """
    Comprehensive agent system for tool generation, evaluation, and refinement.
    
    The system includes three main components:
    1. Tool Generation: Generate tools using TOOL_EXTRACT_AGENT_AGNTIC_GENERATION_PROMPT
    2. Tool Evaluation: Syntax error detection + LLM evaluation with tools
    3. Tool Refinement: Syntax error fixing + effectiveness-based refinement
    
    An orchestration module controls the overall flow with configurable max iterations.
    """
    
    def __init__(self, 
                 generation_model_name: str = "o4-mini", 
                 verification_model_name_lst: list[str] = ["gpt-4.1", "Qwen/Qwen3-8B"],
                 skip_on_validation_failure: bool = False,
                 code_validation_timeout: int = 2):
        self.generation_model_name = generation_model_name
        self.verification_model_name_lst = verification_model_name_lst
        self.llm = LLMCaller(model_name=generation_model_name)
        self.skip_on_validation_failure = skip_on_validation_failure
        self.code_validation_timeout = code_validation_timeout
        
        # Initialize LLM callers for different verification models
        self.verification_llms = {
            model: LLMCaller(model_name=model) for model in verification_model_name_lst
        }
    
    def run_agent_system(self, 
                        question: str, 
                        CoT: str,
                        answer: str = None,
                        max_iterations: int = 3,
                        evaluation_repeats: int = 2) -> Dict[str, Any]:
        """
        Main orchestration function that runs the complete agent system.
        
        Args:
            question: The input question
            CoT: The CoT reasoning/answer
            answer: The expected answer (optional)
            max_iterations: Maximum number of refinement iterations
            evaluation_repeats: Number of times to repeat tool evaluation
            
        Returns:
            Dict containing tools, trajectory, and execution details
        """
        trajectory = []
        current_tools = []
        
        # Step 1: Tool Generation (always first)
        trajectory.append({"step": "tool_generation", "iteration": 0})
        generation_result = self.tool_generation(question, CoT)
        current_tools = generation_result["tools"]
        trajectory.append({
            "step": "tool_generation_result",
            "iteration": 0,
            "tools_generated": len(current_tools),
            "llm_interaction": generation_result["llm_interaction"]
        })
        
        if not current_tools:
            return {
                "final_tools": [],
                # "trajectory": trajectory,
                "status": "failed_generation",
                "message": "No tools were generated"
            }

        # Optional early validation: schema + code quick check; skip entire question on failure
        if self.skip_on_validation_failure:
            trajectory.append({"step": "pre_validation_start", "iteration": 0})
            pre_val = self._pre_evaluation_validation(current_tools)
            trajectory.append({
                "step": "pre_validation_result",
                "iteration": 0,
                "errors": pre_val.get("errors", []),
                "validated_tools": pre_val.get("validated_tools", 0),
            })
            if pre_val.get("should_skip", False):
                return {
                    "final_tools": [],
                    # "trajectory": trajectory,
                    "status": "skipped_due_to_validation",
                    "message": "Pre-evaluation validation failed (schema/code). Skipping this question."
                }
        
        # Main iteration loop
        for iteration in range(max_iterations):
            trajectory.append({"step": "evaluation_start", "iteration": iteration + 1})
            
            # Step 2: Tool Evaluation
            evaluation_result = self.tool_evaluation(question, current_tools, answer, evaluation_repeats)
            trajectory.append({
                "step": "evaluation_result",
                "iteration": iteration + 1,
                "syntax_errors": evaluation_result["syntax_errors"],
               "llm_interactions": evaluation_result["llm_interactions"],
                "effectiveness_status": evaluation_result["effectiveness_status"],
                "effectiveness_analysis": evaluation_result["effectiveness_analysis"],
            })
            
            # Decision logic
            if evaluation_result["syntax_errors"]:
                # Syntax errors found - proceed to syntax refinement
                trajectory.append({"step": "syntax_refinement_start", "iteration": iteration + 1})
                refinement_result = self.syntax_error_refinement(current_tools, evaluation_result["syntax_errors"])
                current_tools = refinement_result["refined_tools"]
                trajectory.append({
                    "step": "syntax_refinement_result",
                    "iteration": iteration + 1,
                    "tools_fixed": len(refinement_result["refined_tools"]),
                    "llm_interaction": refinement_result["llm_interaction"]
                })
                continue
            
            elif evaluation_result["effectiveness_status"] == "EFFECTIVE":
                # Tools are effective - exit and return
                return {
                    "final_tools": current_tools,
                    "trajectory": trajectory,
                    "status": "success",
                    "iterations_used": iteration + 1,
                    "message": "Tools are effective for solving the problem"
                }
            
            else:
                # Tools are ineffective - proceed to effectiveness refinement
                trajectory.append({"step": "effectiveness_refinement_start", "iteration": iteration + 1})
                refinement_result = self.effectiveness_refinement(
                    question, current_tools, evaluation_result["effectiveness_analysis"]
                )
                current_tools = refinement_result["refined_tools"]
                trajectory.append({
                    "step": "effectiveness_refinement_result",
                    "iteration": iteration + 1,
                    "tools_refined": len(refinement_result["refined_tools"]),
                    "llm_interaction": refinement_result["llm_interaction"]
                })
        
        # Max iterations reached
        return {
            "final_tools": current_tools,
            # "trajectory": trajectory,
            "status": "max_iterations_reached",
            "iterations_used": max_iterations,
            "message": f"Reached maximum iterations ({max_iterations}), returning current tools"
        }
    
    def tool_generation(self, question: str, CoT: str) -> Dict[str, Any]:
        """
        Step 1: Generate tools using TOOL_EXTRACT_AGENT_AGNTIC_GENERATION_PROMPT.
        Parse and store all generated tools.
        """
        prompt = TOOL_EXTRACT_AGENT_AGNTIC_GENERATION_PROMPT.format(question=question, answer=CoT)
        response = self.llm.call(content=prompt)
        
        # Parse tools from response
        tools = self._parse_tools_from_response(response)
        
        return {
            "tools": tools,
            "llm_interaction": {
                "prompt": prompt,
                "response": response,
                "model": self.generation_model_name
            }
        }
    
    def tool_evaluation(self, question: str, tools: List[Dict[str, Any]], answer: str, repeat_count: int = 2) -> Dict[str, Any]:
        """
        Step 2: Tool evaluation with two components:
        1. Syntax error detection for each tool
        2. LLM evaluation using verification models, then analyzed by main model
        """
        evaluation_interactions = []
        
        # Component 1: Syntax error detection using utils function
        syntax_errors = validate_tools_syntax(tools, timeout=1)  # Restore original timeout
        
        # Component 1.5: JSON Schema validation for OpenAI compatibility
        if not syntax_errors:
            schema_errors = self._validate_openai_schema(tools)
            if schema_errors:
                # Treat schema errors as syntax errors for unified handling
                syntax_errors = schema_errors
        
        if syntax_errors:
            # If syntax errors found, return immediately
            return {
                "syntax_errors": syntax_errors,
                "effectiveness_status": "CANNOT_EVALUATE",
                "effectiveness_analysis": "Cannot evaluate effectiveness due to syntax errors",
                "llm_interactions": evaluation_interactions
            }
        
        # Component 2: LLM evaluation using verification models
        all_verification_results = []
        
        # For each verification model
        for model_name, verification_llm in self.verification_llms.items():
            model_results = []
            
            # Repeat evaluation for this model
            for i in range(repeat_count):
                # Register tools for use with call_with_static_tools
                tool_definitions, function_registry = self._prepare_tools_for_llm(tools)
                
                # Create evaluation prompt
                tools_description = self._create_tools_description(tools)
                eval_prompt = TOOL_EVALUATION_PROBLEM_SOLVING_PROMPT.format(
                    question=question,
                    tools_description=tools_description
                )
                
                # Define completion check function
                def completion_check(content):
                    """Check if we have a clear final answer"""
                    if not content:
                        return False
                        
                    content_lower = content.lower()
                    # Stop if the assistant provides a clear final answer
                    return any(keyword in content_lower for keyword in [
                        "<final_answer>"
                    ])
                
                # Use call_with_static_tools for evaluation with built-in completion check
                messages, turns = verification_llm.call_with_static_tools(
                    content=eval_prompt,
                    tools=tool_definitions,
                    function_registry=function_registry,
                    return_format="messages_turns",
                    max_turns=10,  # Limit maximum turns
                    completion_check=completion_check  # Built-in completion check
                )
                
                evaluation_interactions.append({
                    "verification_model": model_name,
                    "round": i + 1,
                    "prompt": eval_prompt,
                    "messages": messages,
                    "turns": turns,
                    "model": model_name
                })
                
                # Extract final answer from the conversation
                final_response = self._extract_final_response(messages)
                model_results.append(final_response)
            
            # Store results for this verification model
            all_verification_results.extend(model_results)
        
        # Use self.llm to analyze overall effectiveness based on all verification results
        effectiveness_analysis = self._analyze_effectiveness(question, tools, all_verification_results, answer)
        
        return {
            "syntax_errors": [],
            "effectiveness_status": effectiveness_analysis["status"],
            "effectiveness_analysis": effectiveness_analysis["analysis"],
            "evaluation_results": all_verification_results,
            "llm_interactions": evaluation_interactions
        }
    
    def syntax_error_refinement(self, tools: List[Dict[str, Any]], syntax_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 3a: Fix syntax errors in tools, including both Python syntax and JSON schema errors.
        """
        refined_tools = []
        llm_interactions = []
        
        for tool in tools:
            # Check if this tool has syntax errors
            tool_errors = [err for err in syntax_errors if err.get("tool_index") == tool.get("index", -1)]
            
            if tool_errors:
                # Determine error type and use appropriate fixing strategy
                schema_errors = [err for err in tool_errors if err.get("error_type", "").startswith("schema")]
                python_errors = [err for err in tool_errors if err.get("error_type", "") != "schema" or "error_type" not in err]
                
                if schema_errors:
                    # Fix schema errors
                    refined_tool = self._fix_schema_errors(tool, schema_errors)
                    llm_interactions.append({
                        "tool_index": tool.get("index", -1),
                        "fix_type": "schema_auto_fix",
                        "errors_fixed": len(schema_errors),
                        "model": "auto_fix"
                    })
                    refined_tools.append(refined_tool)
                
                elif python_errors:
                    # Fix Python syntax errors using LLM
                    error_details = "\n".join([err["error_message"] for err in python_errors])
                    
                    fix_prompt = TOOL_SYNTAX_ERROR_REFINEMENT_PROMPT.format(
                        original_code=tool["code"],
                        error_details=error_details
                    )
                    
                    response = self.llm.call(content=fix_prompt)
                    llm_interactions.append({
                        "tool_index": tool.get("index", -1),
                        "prompt": fix_prompt,
                        "response": response,
                        "model": self.generation_model_name,
                        "fix_type": "python_llm_fix"
                    })
                    
                    # Extract fixed code
                    fixed_code = self._extract_fixed_code(response)
                    if fixed_code:
                        refined_tool = tool.copy()
                        refined_tool["code"] = fixed_code
                        refined_tools.append(refined_tool)
                    else:
                        # If fixing failed, keep original tool
                        refined_tools.append(tool)
                
                else:
                    # If fixing failed, keep original tool
                    refined_tools.append(tool)
            else:
                # No syntax errors, keep original tool
                refined_tools.append(tool)
        
        return {
            "refined_tools": refined_tools,
            "llm_interaction": {
                "interactions": llm_interactions,
                "total_tools_processed": len(tools),
                "tools_fixed": len([t for t in llm_interactions])
            }
        }
    
    def effectiveness_refinement(self, question: str, tools: List[Dict[str, Any]], effectiveness_analysis: str) -> Dict[str, Any]:
        """
        Step 3b: Refine tools based on effectiveness analysis.
        """
        current_tools_text = self._format_tools_for_refinement(tools)
        
        refinement_prompt = TOOL_EFFECTIVENESS_REFINEMENT_PROMPT.format(
            question=question,
            current_tools=current_tools_text,
            effectiveness_analysis=effectiveness_analysis
        )
        
        response = self.llm.call(content=refinement_prompt)
        
        # Parse refined tools from response
        refined_tools = self._parse_tools_from_response(response)
        
        # Fallback mechanism: if parsing fails (empty result), keep original tools
        if not refined_tools and tools:
            refined_tools = tools
        
        return {
            "refined_tools": refined_tools,
            "llm_interaction": {
                "prompt": refinement_prompt,
                "response": response,
                "model": self.generation_model_name,
                "original_tool_count": len(tools),
                "refined_tool_count": len(refined_tools),
                "parsing_success": len(refined_tools) > 0 if tools else True
            }
        }
    
    # Helper methods
    def _parse_tools_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse tools from LLM response in the new format with tool_OpenAI_json_schema."""
        tools = []
        
        # Find all tool blocks using regex
        tool_pattern = r'<tool(\d+)>(.*?)</tool\1>'
        matches = re.findall(tool_pattern, response, re.DOTALL)
        
        if not matches:
            return tools
        
        for i, (tool_num, content) in enumerate(matches):
            try:
                # Extract tag
                tag_match = re.search(r'<tag>(.*?)</tag>', content, re.DOTALL)
                if not tag_match:
                    continue
                    
                # Extract tool_OpenAI_json_schema
               #  schema_content = content.split("</tag>")[1].split("<code>")[0]
                # schema_match = re.search(r'<tool_OpenAI_json_schema>(.*?)</tool_OpenAI_json_schema>', content, re.DOTALL)
                # if not schema_match:
                #     continue
                
                # schema_content = schema_match.group(1).strip()
                
                # # Parse JSON schema - remove leading/trailing brackets if present
                # if schema_content.startswith('[') and schema_content.endswith(']'):
                #     schema_content = schema_content[1:-1].strip()
                    
                # # Handle the case where there might be trailing comma
                # if schema_content.endswith(','):
                #     schema_content = schema_content[:-1].strip()
                code_match = re.search(r'<code>(.*?)</code>', content, re.DOTALL)
                if not code_match:
                    continue
                
                tool_code = code_match.group(1)
                if "```python\n" in tool_code:
                       tool_code = tool_code.split("```python")[1].split("```")[0]  # Remove ```python\  # Remove ```

                # Parse the JSON
                tool_schema = code_comment2function_json(tool_code)
                
                # Extract tool_info and tool_code from the schema
                tool_info = tool_schema
                                
                tools.append({
                    "index": i,
                    "tool_number": tool_num,
                    "tag": tag_match.group(1).strip(),
                    "tool_info": tool_info,
                    "code": tool_code.strip()
                })
                
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                continue
        
        return tools

    def _pre_evaluation_validation(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform fast JSON schema validation and code quick validation.
        If any tool fails schema validation or code validation/timeout, signal to skip.
        """
        errors: List[Dict[str, Any]] = []

        # 1) Basic Python syntax validation (fast AST with timeout)
        syntax_errors = validate_tools_syntax(tools, timeout=1)
        if syntax_errors:
            errors.extend([{"type": "python_syntax", **e} for e in syntax_errors])

        # 2) OpenAI schema validation
        if not syntax_errors:
            schema_errors = self._validate_openai_schema(tools)
            if schema_errors:
                errors.extend([{**e, "type": "schema"} for e in schema_errors])

        # 3) Quick code execution validation (in subprocess) to ensure execute() exists and is callable
        #    Only run if previous checks passed to save time
        if not errors:
            for i, tool in enumerate(tools):
                code = tool.get("code", "")
                if not code.strip():
                    continue
                ok, msg = self._quick_check_execute_in_subprocess(code, timeout=self.code_validation_timeout)
                if not ok:
                    errors.append({
                        "type": "code_quick_validation",
                        "tool_index": i,
                        "error_message": msg
                    })

        return {
            "should_skip": len(errors) > 0,
            "errors": errors,
            "validated_tools": len(tools) - len({e.get("tool_index") for e in errors if isinstance(e.get("tool_index"), int)})
        }

    def _quick_check_execute_in_subprocess(self, code: str, timeout: int = 2) -> Tuple[bool, str]:
        """
        Use a subprocess to exec the code safely and verify it can be executed without errors.
        Returns (ok: bool, message: str). On timeout or any error, returns ok=False.
        """
        try:
            import base64
            from utils import execute_code
            
            # Extract Python code from various formats
            actual_code = code
            
            # Handle <code>...</code> format
            if "<code>" in code and "</code>" in code:
                start_idx = code.find("<code>") + len("<code>")
                end_idx = code.find("</code>", start_idx)
                if end_idx != -1:
                    actual_code = code[start_idx:end_idx].strip()
            
            # Handle ```python...``` format within the extracted code
            if "```python" in actual_code:
                start_idx = actual_code.find("```python") + len("```python")
                end_idx = actual_code.find("```", start_idx)
                if end_idx != -1:
                    actual_code = actual_code[start_idx:end_idx].strip()
                else:
                    actual_code = actual_code[start_idx:].strip()
            
            encoded = base64.b64encode(actual_code.encode("utf-8")).decode("ascii")
            script = (
                "import base64\n"
                "import sys\n"
                "import builtins\n"
                "_c = base64.b64decode('" + encoded + "').decode('utf-8')\n"
                "ns = {}\n"
                "try:\n"
                "    exec(_c, ns)\n"
                "    print('OK')\n"  # Just check if code can be executed, don't require execute function
                "except Exception as e:\n"
                "    import traceback\n"
                "    print('ERROR:' + traceback.format_exc())\n"
            )
            result = execute_code(script, timeout=timeout)
            text = (result or "").strip()
            if text.startswith("OK"):
                return True, "OK"
            if "Timeout" in text or text.startswith("Error: Timeout"):
                return False, "Timeout during quick validation"
            if text.startswith("ERROR:"):
                return False, text
            # Unknown output, consider as failure
            return False, text
        except Exception as e:
            return False, f"Validation exception: {str(e)}"
    
    def _prepare_tools_for_llm(self, tools: List[Dict[str, Any]]) -> Tuple[List[Dict], Dict]:
        """Prepare tools for use with call_with_static_tools using existing OpenAI schema."""
        tool_definitions = []
        function_registry = {}
        
        for tool in tools:
            # Since tools are now generated with complete OpenAI schema, use tool_info directly
            if "tool_info" not in tool or not tool["tool_info"]:
                continue
                
            tool_definitions.append(tool["tool_info"])
            func_name = tool["tool_info"].get("function", {}).get("name", "")
            
            if not func_name:
                continue
            
            # Execute code to create function for registry
            code = tool.get("code", "")
            if code:
                try:
                    local_namespace = {}
                    exec(code, local_namespace)
                    
                    # The new format uses 'execute' as function name in code
                    if "execute" in local_namespace:
                        function_registry[func_name] = local_namespace["execute"]
                    elif func_name in local_namespace:
                        function_registry[func_name] = local_namespace[func_name]
                except Exception as e:
                    pass
        
        return tool_definitions, function_registry
    
    def _create_tools_description(self, tools: List[Dict[str, Any]]) -> str:
        """Create a description of available tools for the evaluation prompt."""
        descriptions = []
        for tool in tools:
            if "tool_info" in tool and tool["tool_info"]:
                func_name = tool["tool_info"].get("function", {}).get("name", "unknown")
                description = tool["tool_info"].get("function", {}).get("description", "No description")
                descriptions.append(f"- {func_name}: {description}")
            else:
                # This should not happen with the new format, but keep for robustness
                descriptions.append(f"- unknown_tool: {tool.get('tag', 'No tag')}")
        return "\n".join(descriptions)
    
    def _extract_final_response(self, messages: List[Dict]) -> str:
        """Extract the final response from conversation messages."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return ""
    
    def _analyze_effectiveness(self, question: str, tools: List[Dict[str, Any]], evaluation_results: List[str], answer: str) -> Dict[str, Any]:
        """Analyze the effectiveness of tools based on evaluation results."""
        # Use LLM to analyze effectiveness
        tools_text = self._format_tools_for_refinement(tools)
        evaluation_text = "\n\n".join([f"Round {i+1}:\n{result}" for i, result in enumerate(evaluation_results)])
        
        analysis_prompt = TOOL_EFFECTIVENESS_EVALUATION_PROMPT.format(
            question=question,
            tools=tools_text,
            evaluation_results=evaluation_text,
            answer=answer
        )
        messages = [{"role": "user", "content": analysis_prompt}]
        response = self.llm.call(messages=messages)
        messages.append({"role": "assistant", "content": response})
        # Parse effectiveness status
        if "EFFECTIVENESS_STATUS: EFFECTIVE" in response:
            status = "EFFECTIVE"
        else:
            status = "INEFFECTIVE"
        
        return {
            "status": status,
            "analysis": messages
        }
    
    def _extract_fixed_code(self, response: str) -> Optional[str]:
        """Extract fixed code from refinement response."""
        match = re.search(r'<fixed_code>(.*?)</fixed_code>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _format_tools_for_refinement(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools for refinement prompts."""
        formatted = []
        for i, tool in enumerate(tools, 1):
            formatted.append(f"Tool {i}:")
            formatted.append(f"Tag: {tool.get('tag', 'No tag')}")
            if "tool_info" in tool and tool["tool_info"]:
                formatted.append(f"Tool Info: {json.dumps(tool['tool_info'], indent=2)}")
            formatted.append(f"Code:\n{tool.get('code', '')}")
            formatted.append("")
        return "\n".join(formatted)
    
    def _validate_openai_schema(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validates if the generated tools' OpenAI schema is compatible.
        Returns a list of errors if any, focusing on common OpenAI schema issues.
        """
        errors = []
        
        for i, tool in enumerate(tools):
            if "tool_info" not in tool or not tool["tool_info"]:
                errors.append({
                    "tool_index": i,
                    "error_message": f"Tool {i} missing tool_info",
                    "error_type": "schema_missing_field"
                })
                continue
            
            tool_info = tool["tool_info"]
            
            # Check basic structure
            if not isinstance(tool_info, dict):
                errors.append({
                    "tool_index": i,
                    "error_message": f"Tool {i} tool_info is not a dictionary",
                    "error_type": "schema_invalid_type"
                })
                continue
            
            # Check if it has the correct OpenAI format
            if "function" not in tool_info:
                errors.append({
                    "tool_index": i,
                    "error_message": f"Tool {i} missing 'function' key in tool_info",
                    "error_type": "schema_missing_field"
                })
                continue
            
            function_def = tool_info["function"]
            if not isinstance(function_def, dict):
                errors.append({
                    "tool_index": i,
                    "error_message": f"Tool {i} 'function' is not a dictionary",
                    "error_type": "schema_invalid_type"
                })
                continue
            
            # Check required fields in function
            required_fields = ["name", "description", "parameters"]
            for field in required_fields:
                if field not in function_def:
                    errors.append({
                        "tool_index": i,
                        "error_message": f"Tool {i} function missing required field '{field}'",
                        "error_type": "schema_missing_field"
                    })
                    continue
            
            # Check parameters field type and structure
            if "parameters" in function_def:
                parameters = function_def["parameters"]
                
                # First check if parameters is the correct type (should be dict/object)
                if not isinstance(parameters, dict):
                    errors.append({
                        "tool_index": i,
                        "error_message": f"Tool {i} 'parameters' must be an object/dictionary, got {type(parameters).__name__}: {parameters}",
                        "error_type": "schema_parameters_wrong_type"
                    })
                else:
                    # Check for invalid top-level keywords in parameters
                    invalid_keywords = ['oneOf', 'anyOf', 'allOf', 'enum', 'not']
                    found_invalid = [kw for kw in invalid_keywords if kw in parameters]
                    if found_invalid:
                        errors.append({
                            "tool_index": i,
                            "error_message": f"Tool {i} parameters cannot have {found_invalid} at the top level",
                            "error_type": "schema_parameters_invalid_keywords",
                            "invalid_keywords": found_invalid
                        })
                    
                    # Only validate schema structure if it's a dict and doesn't have invalid keywords
                    if not found_invalid:
                        param_errors = self._validate_parameters_schema(parameters, i)
                        errors.extend(param_errors)
        
        return errors
    
    def _validate_parameters_schema(self, parameters: Dict[str, Any], tool_index: int) -> List[Dict[str, Any]]:
        """
        Validates the parameters schema, specifically checking for array items issues and other OpenAI compatibility problems.
        """
        errors = []
        
        if not isinstance(parameters, dict):
            errors.append({
                "tool_index": tool_index,
                "error_message": f"Tool {tool_index} parameters is not a dictionary",
                "error_type": "schema_invalid_type"
            })
            return errors
        
        # Check if parameters has proper structure
        if "type" not in parameters or parameters["type"] != "object":
            errors.append({
                "tool_index": tool_index,
                "error_message": f"Tool {tool_index} parameters must have type 'object'",
                "error_type": "schema_invalid_type"
            })
        
        if "properties" not in parameters:
            errors.append({
                "tool_index": tool_index,
                "error_message": f"Tool {tool_index} parameters missing 'properties' field",
                "error_type": "schema_missing_field"
            })
            return errors
        
        # Recursively check properties for various issues
        properties = parameters.get("properties", {})
        if isinstance(properties, dict):
            for prop_name, prop_def in properties.items():
                if isinstance(prop_def, dict):
                    # Check for array type without items
                    if prop_def.get("type") == "array" and "items" not in prop_def:
                        errors.append({
                            "tool_index": tool_index,
                            "error_message": f"Tool {tool_index} parameter '{prop_name}' is array type but missing 'items' field",
                            "error_type": "schema_array_missing_items",
                            "property_name": prop_name
                        })
                    
                    # Check for invalid type arrays (like ['number', 'string'])
                    if "type" in prop_def:
                        prop_type = prop_def["type"]
                        if isinstance(prop_type, list):
                            errors.append({
                                "tool_index": tool_index,
                                "error_message": f"Tool {tool_index} parameter '{prop_name}' has type as array {prop_type}, should use oneOf instead",
                                "error_type": "schema_type_array",
                                "property_name": prop_name,
                                "invalid_type": prop_type
                            })
                    
                    # Check for invalid property values (like True, False, etc.)
                    for key, value in prop_def.items():
                        if key in ["default", "minimum", "maximum"] and not isinstance(value, (int, float, str, type(None))):
                            if isinstance(value, bool):
                                errors.append({
                                    "tool_index": tool_index,
                                    "error_message": f"Tool {tool_index} parameter '{prop_name}' has invalid {key} value: {value} (boolean not allowed)",
                                    "error_type": "schema_invalid_value_type",
                                    "property_name": prop_name,
                                    "field_name": key,
                                    "invalid_value": value
                                })
                    
                    # Recursively check nested objects
                    if prop_def.get("type") == "object" and "properties" in prop_def:
                        nested_errors = self._validate_parameters_schema(prop_def, tool_index)
                        errors.extend(nested_errors)
        
        return errors
    
    def _fix_schema_errors(self, tool: Dict[str, Any], schema_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Automatically fix common schema errors without using LLM.
        """
        refined_tool = tool.copy()
        
        if "tool_info" not in refined_tool or not refined_tool["tool_info"]:
            return refined_tool
        
        tool_info = refined_tool["tool_info"].copy()
        
        for error in schema_errors:
            error_type = error.get("error_type", "")
            
            if error_type == "schema_parameters_wrong_type":
                # Fix parameters that are arrays instead of objects
                if "function" in tool_info and "parameters" in tool_info["function"]:
                    current_params = tool_info["function"]["parameters"]
                    
                    # If it's an array of type definitions, convert to proper object schema
                    if isinstance(current_params, list):
                        # Create a proper object schema
                        properties = {}
                        required = []
                        
                        for i, param_def in enumerate(current_params):
                            if isinstance(param_def, dict) and "type" in param_def:
                                param_name = f"param_{i+1}"
                                # Clean up the param_def to fix any type arrays
                                cleaned_def = self._clean_property_definition(param_def)
                                properties[param_name] = cleaned_def
                                required.append(param_name)
                        
                        tool_info["function"]["parameters"] = {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    else:
                        # For other invalid types, create a minimal valid schema
                        tool_info["function"]["parameters"] = {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                        
            elif error_type == "schema_parameters_invalid_keywords":
                # Remove invalid top-level keywords from parameters
                if "function" in tool_info and "parameters" in tool_info["function"]:
                    parameters = tool_info["function"]["parameters"]
                    invalid_keywords = error.get("invalid_keywords", [])
                    
                    for keyword in invalid_keywords:
                        if keyword in parameters:
                            del parameters[keyword]
                    
                    # Ensure we have the required structure
                    if "type" not in parameters:
                        parameters["type"] = "object"
                    if "properties" not in parameters:
                        parameters["properties"] = {}
                        
            elif error_type == "schema_type_array":
                # Fix type arrays like ['number', 'string'] to use oneOf
                prop_name = error.get("property_name", "")
                invalid_type = error.get("invalid_type", [])
                
                if prop_name and "function" in tool_info and "parameters" in tool_info["function"]:
                    parameters = tool_info["function"]["parameters"]
                    if "properties" in parameters and prop_name in parameters["properties"]:
                        prop_def = parameters["properties"][prop_name]
                        
                        # Convert type array to oneOf
                        if isinstance(invalid_type, list):
                            prop_def["oneOf"] = [{"type": t} for t in invalid_type]
                            del prop_def["type"]
                            
            elif error_type == "schema_invalid_value_type":
                # Fix invalid values like boolean True/False in numeric fields
                prop_name = error.get("property_name", "")
                field_name = error.get("field_name", "")
                invalid_value = error.get("invalid_value")
                
                if prop_name and field_name and "function" in tool_info and "parameters" in tool_info["function"]:
                    parameters = tool_info["function"]["parameters"]
                    if "properties" in parameters and prop_name in parameters["properties"]:
                        prop_def = parameters["properties"][prop_name]
                        
                        # Remove or fix invalid values
                        if field_name in prop_def:
                            if isinstance(invalid_value, bool):
                                # Convert boolean to appropriate numeric value
                                if field_name in ["minimum", "maximum"]:
                                    prop_def[field_name] = 1 if invalid_value else 0
                                else:
                                    # For default values, remove if boolean
                                    del prop_def[field_name]
                        
            elif error_type == "schema_array_missing_items":
                # Fix array missing items issue
                prop_name = error.get("property_name", "")
                if prop_name and "function" in tool_info and "parameters" in tool_info["function"]:
                    parameters = tool_info["function"]["parameters"]
                    if "properties" in parameters and prop_name in parameters["properties"]:
                        prop_def = parameters["properties"][prop_name]
                        if prop_def.get("type") == "array" and "items" not in prop_def:
                            # Add default items schema
                            prop_def["items"] = {"type": "string"}  # Default to string items
                            
            elif error_type == "schema_missing_field":
                # Add missing required fields with defaults
                if "function" not in tool_info:
                    tool_info["function"] = {}
                
                function_def = tool_info["function"]
                if "name" not in function_def:
                    function_def["name"] = f"tool_function_{tool.get('index', 0)}"
                if "description" not in function_def:
                    function_def["description"] = "Auto-generated tool function"
                if "parameters" not in function_def:
                    function_def["parameters"] = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
        
        refined_tool["tool_info"] = tool_info
        return refined_tool
    
    def _clean_property_definition(self, prop_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean up a property definition to fix common schema issues.
        """
        cleaned = prop_def.copy()
        
        # Fix type arrays like ['number', 'string']
        if "type" in cleaned and isinstance(cleaned["type"], list):
            type_list = cleaned["type"]
            cleaned["oneOf"] = [{"type": t} for t in type_list]
            del cleaned["type"]
        
        # Fix invalid boolean values in numeric fields
        for field in ["default", "minimum", "maximum"]:
            if field in cleaned and isinstance(cleaned[field], bool):
                if field in ["minimum", "maximum"]:
                    cleaned[field] = 1 if cleaned[field] else 0
                else:
                    del cleaned[field]
        
        return cleaned
    