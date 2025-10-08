import os
import logging
import json
import time
import argparse
import ast
import traceback
import inspect
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from IterativeLibraryOptimizerAgent import IterativeLibraryOptimizerAgent

from utils import read_jsonl, call_openai_api, map_with_progress, save_json, read_json, call_openai_api_multi_turn
from prompt import (
    BLUEPRINT_DESIGN_PROMPT,
    CODE_IMPLEMENTATION_PROMPT,
    CONVERT_TO_OPENAI_TOOL_PROMPT,
    CODE_INSPECTOR_PROMPT,
    OPENAI_FUNCTION_CALL_INSPECTOR_PROMPT,
    CODE_REFINE_PROMPT,
    TOOL_CODE_VALIDATION_PROMPT,
    LIB_REFINEMENT_BLUEPRINT_PROMPT
)

# Import token counting functions
try:
    from extract_reasoning_template_v2 import count_tokens, calculate_cost
except ImportError:
    def count_tokens(text: str, model_name: str) -> int:
        return int(len(text.split()) * 1.3) if text else 0
    
    def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
        if model_name.lower() == "o3":
            return (input_tokens / 1_000_000) * 2.00 + (output_tokens / 1_000_000) * 8.00
        return 0.0

@dataclass
class ToolValidationResult:
    tool_name: str
    is_valid: bool
    original_code: str
    fixed_code: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    validation_details: Optional[str] = None

# Configure simple logging
def setup_logging(debug: bool = False, log_folder: str = None):
    """Setup simple logging"""
    log_level = logging.DEBUG if debug else logging.INFO
    log_dir = Path(log_folder)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"aggregate_tools_{timestamp}.log" 
    
    # File handler - captures everything
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler - only shows progress and important info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors on console
    console_formatter = logging.Formatter('%(message)s')  # Simple format for console
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )
    
    # Create a progress logger for terminal output
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False  # Don't propagate to root logger
    
    # Progress handler - only to console
    progress_handler = logging.StreamHandler()
    progress_handler.setLevel(logging.INFO)
    progress_handler.setFormatter(console_formatter)
    progress_logger.addHandler(progress_handler)
    
    print(f"📝 Log file: {log_file}")
    return log_file, progress_logger

logger = logging.getLogger(__name__)

class DesignStep(Enum):
    BLUEPRINT = "blueprint"
    IMPLEMENTATION = "implementation"

class StepResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class ClusterSummary:
    cluster_name: str
    total_tools: int = 0
    steps_completed: List[str] = None
    final_code: Optional[str] = None
    openai_tools: Optional[List[Dict]] = None
    unique_questions: int = 0
    optimization_iterations: int = 0
    final_optimization_results: List[Dict] = None
    success: bool = False
    error_message: Optional[str] = None
    code_versions_saved: int = 0
    openai_tools_versions_saved: int = 0
    
    def __post_init__(self):
        if self.steps_completed is None:
            self.steps_completed = []
        if self.final_optimization_results is None:
            self.final_optimization_results = []

class TopDownClusterProcessor:
    """Processes tool clusters using simplified approach with direct optimization."""
    
    def __init__(self, clusters_file_path: str, max_review_iterations: int = 3, debug: bool = False, 
                 model_name: str = "o4-mini", progress_logger=None):
        self.clusters_file_path = clusters_file_path
        self.max_review_iterations = max_review_iterations
        self.debug = debug
        self.progress_logger = progress_logger or logging.getLogger('progress')
        self.model_name = model_name
        
        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        # Output directory - will be set by main()
        self.output_dir = None
        
        # Store blueprint conversation history for each cluster
        self.blueprint_conversations = {}
        
        # Load data
        self._load_clusters_data()
        
    def _load_clusters_data(self):
        """Load clusters data from JSON file"""
        try:
            data = read_json(self.clusters_file_path)
            
            if 'clusters' in data:
                clusters_list = data['clusters']
                if isinstance(clusters_list, list):
                    self.clusters_data = {}
                    for cluster in clusters_list:
                        cluster_name = cluster.get('cluster_name', f'cluster_{len(self.clusters_data)}')
                        tools = cluster.get('tools', [])
                        self.clusters_data[cluster_name] = tools
                else:
                    self.clusters_data = clusters_list
            else:
                if isinstance(data, list):
                    self.clusters_data = {}
                    for i, cluster in enumerate(data):
                        if isinstance(cluster, dict) and 'cluster_name' in cluster:
                            cluster_name = cluster['cluster_name']
                            tools = cluster.get('tools', [])
                        else:
                            cluster_name = f'cluster_{i}'
                            tools = cluster if isinstance(cluster, list) else []
                        self.clusters_data[cluster_name] = tools
                else:
                    self.clusters_data = data
            
            if self.debug:
                # Debug mode: select first cluster and first 10 tools
                if self.clusters_data:
                    debug_cluster_count = 1
                    debug_tool_count = 1000
                    selected_cluster_names = list(self.clusters_data.keys())[:debug_cluster_count]
                    debug_clusters = {}
                    for cluster_name in selected_cluster_names:
                        cluster_tools = self.clusters_data[cluster_name][:debug_tool_count]
                        debug_clusters[cluster_name] = cluster_tools
                    self.clusters_data = debug_clusters
            
        except Exception as e:
            raise
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract clean Python code from LLM response"""
        import re
        
        # Extract code blocks
        python_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if python_blocks:
            return python_blocks[0].strip()
        
        code_blocks = re.findall(r'```[^\n]*\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        return response.replace('```python', '').replace('```', '').strip()

    def _call_llm_for_step(self, step: DesignStep, input_content: str, cluster_name: str, 
                          cluster_index: int = 1, total_clusters: int = 1, version: str = "v0") -> Tuple[StepResult, str, Optional[str]]:
        """Call LLM for a specific design step and save detailed logs"""
        try:
            # Select prompt
            if step == DesignStep.BLUEPRINT:
                prompt_template = BLUEPRINT_DESIGN_PROMPT
            elif step == DesignStep.IMPLEMENTATION:
                prompt_template = CODE_IMPLEMENTATION_PROMPT
            else:
                raise ValueError(f"Unknown design step: {step}")
            
            # Format prompt
            try:
                if step == DesignStep.BLUEPRINT:
                    formatted_prompt = prompt_template.format(tool_code_name_list=input_content, domain=cluster_name)
                elif step == DesignStep.IMPLEMENTATION:
                    # Extract blueprint text from conversation history
                    blueprint_text = ""
                    if cluster_name in self.blueprint_conversations:
                        # Get the assistant's response from the blueprint conversation
                        blueprint_text = self.blueprint_conversations[cluster_name][1]["content"]
                    function_number = blueprint_text.count("[Description]")
                    formatted_prompt = prompt_template.format(blueprint=blueprint_text, domain=cluster_name,function_number=function_number)
            except KeyError as e:
                raise ValueError(f"Missing format parameter in {step.value} prompt: {e}")
            
            # Call LLM with single-turn API for both steps
            input_tokens = count_tokens(formatted_prompt, self.model_name)
            response = call_openai_api(content=formatted_prompt, model_name=self.model_name)
            output_tokens = count_tokens(response, self.model_name) if response else 0
            
            # Store blueprint conversation history for this cluster (only for BLUEPRINT step)
            if step == DesignStep.BLUEPRINT and response:
                self.blueprint_conversations[cluster_name] = [
                    {"role": "user", "content": formatted_prompt},
                    {"role": "assistant", "content": response}
                ]
            
            call_cost = calculate_cost(input_tokens, output_tokens, self.model_name)
            
            # Update totals
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += call_cost
            
            if not response:
                return StepResult.FAILED, "", "Empty response from LLM"
            
            # Clean response for code steps
            if step == DesignStep.IMPLEMENTATION:
                cleaned_response = self._extract_code_from_response(response)
                
                # Check if AVAILABLE_TOOLS exists in the cleaned response, retry if not
                max_retries = 3
                retry_count = 0
                retry_info = []
                
                while "AVAILABLE_TOOLS" not in cleaned_response and retry_count < max_retries:
                    retry_count += 1
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🔄 AVAILABLE_TOOLS not found, re-generating code from blueprint (attempt {retry_count}/{max_retries})...")
                    
                    # Re-format the implementation prompt with additional instruction
                    if cluster_name in self.blueprint_conversations:
                        blueprint_text = self.blueprint_conversations[cluster_name][1]["content"]
                        function_number = blueprint_text.count("[Description]")
                        retry_formatted_prompt = CODE_IMPLEMENTATION_PROMPT.format(
                            blueprint=blueprint_text, 
                            domain=cluster_name,
                            function_number=function_number
                        ) 
                    else:
                        print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ❌ No blueprint found for retry")
                        break
                    
                    # Call LLM again with re-formatted prompt
                    retry_input_tokens = count_tokens(retry_formatted_prompt, self.model_name)
                    retry_response = call_openai_api(content=retry_formatted_prompt, model_name=self.model_name)
                    retry_output_tokens = count_tokens(retry_response, self.model_name) if retry_response else 0
                    retry_cost = calculate_cost(retry_input_tokens, retry_output_tokens, self.model_name)
                    
                    # Update totals
                    self.total_input_tokens += retry_input_tokens
                    self.total_output_tokens += retry_output_tokens
                    self.total_cost += retry_cost
                    
                    # Store retry information
                    retry_info.append({
                        "attempt": retry_count,
                        "input_tokens": retry_input_tokens,
                        "output_tokens": retry_output_tokens,
                        "cost": retry_cost,
                        "raw_response": retry_response,
                        "approach": "blueprint_regeneration"
                    })
                    
                    if retry_response:
                        response = retry_response  # Update original response for logging
                        cleaned_response = self._extract_code_from_response(retry_response)
                    else:
                        break
                
                if "AVAILABLE_TOOLS" not in cleaned_response:
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ Failed to generate code with AVAILABLE_TOOLS after {max_retries} blueprint regenerations")
                    return StepResult.FAILED, "", f"Code generation failed: AVAILABLE_TOOLS not found after {max_retries} blueprint-based retries"
                elif retry_count > 0:
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ✅ Successfully generated code with AVAILABLE_TOOLS on blueprint regeneration attempt {retry_count + 1}")
                
            else:
                cleaned_response = response.strip()
                retry_info = []
            
            # Save detailed log
            additional_info = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": call_cost,
                "model_name": self.model_name,
                "raw_response": response,
                "cleaned_response": cleaned_response,
                "conversation_type": "single_turn",
                "blueprint_provided": step == DesignStep.IMPLEMENTATION and cluster_name in self.blueprint_conversations,
                "retry_info": retry_info if step == DesignStep.IMPLEMENTATION else None,
                "available_tools_check": "AVAILABLE_TOOLS" in cleaned_response if step == DesignStep.IMPLEMENTATION else None
            }
            
            self._save_cluster_step_log(
                cluster_name=cluster_name,
                step_name=f"step_{step.value}",
                prompt=formatted_prompt,
                output=cleaned_response,
                version=version,
                cluster_index=cluster_index,
                total_clusters=total_clusters,
                additional_info=additional_info
            )
            
            return StepResult.SUCCESS, cleaned_response, None
                
        except Exception as e:
            logger.error(f"❌ Error in {step.value}: {e}")
            return StepResult.FAILED, "", str(e)
    
    def _extract_tools_for_blueprint(self, tools: List[Dict]) -> str:
        """Extract tool names and descriptions for blueprint design"""
        if not tools:
            return "No tools found in this cluster."
        
        tool_info_parts = []
        for i, tool in enumerate(tools):
            tool_name = tool.get('name', f'tool_{i+1}')
            description = tool.get('description', f'Function: {tool_name}')
            tool_info_parts.append(f"# Function {i+1}: {tool_name}")
            tool_info_parts.append(f"# Description: {description}")
            tool_info_parts.append("")
        
        return "\n".join(tool_info_parts)

    def _step1_blueprint_design(self, tools: List[Dict], cluster_name: str, cluster_index: int, total_clusters: int) -> Tuple[bool, str]:
        """Step 1: Blueprint Design"""
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]   📋 Step 1/4: Blueprint Design")
        
        tools_code = self._extract_tools_for_blueprint(tools)
        blueprint_result, blueprint_output, blueprint_error = self._call_llm_for_step(
            DesignStep.BLUEPRINT, tools_code, cluster_name, cluster_index, total_clusters, "v0"
        )
        
        if blueprint_result != StepResult.SUCCESS:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ❌ Blueprint failed: {blueprint_error}")
            return False, blueprint_error
        
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ✅ Blueprint design completed")
        return True, blueprint_output
    
    def _step2_code_implementation(self, blueprint_output: str, tools_code: str, cluster_name: str, 
                                  cluster_index: int, total_clusters: int) -> Tuple[bool, str]:
        """Step 2: Code Implementation"""
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]   💻 Step 2/4: Code Implementation")
        
        implementation_input = f"BLUEPRINT:{blueprint_output}\nTOOL_CODE_LIST:{tools_code}"
        implementation_result, implementation_output, implementation_error = self._call_llm_for_step(
            DesignStep.IMPLEMENTATION, implementation_input, cluster_name, cluster_index, total_clusters, "v0"
        )
        
        if implementation_result != StepResult.SUCCESS:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ❌ Implementation failed: {implementation_error}")
            return False, implementation_error
        
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ✅ Code implementation completed")
        return True, implementation_output

    def _step2a_code_quality_inspection(self, current_code: str, cluster_name: str, 
                                       cluster_index: int, total_clusters: int) -> str:
        """Step 2a: Code Quality Inspection"""
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]     🔍 Inspecting code quality...")
        
        try:
            prompt = CODE_INSPECTOR_PROMPT.format(code=current_code)
            
            input_tokens = count_tokens(prompt, self.model_name)
            response = call_openai_api(content=prompt, model_name=self.model_name)
            output_tokens = count_tokens(response, self.model_name) if response else 0
            call_cost = calculate_cost(input_tokens, output_tokens, self.model_name)
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += call_cost
            
            improved_code = current_code  # Default to original code
            inspection_result = "NO_NEED_TO_REFINE"
            
            if response and response.strip() != "NO_NEED_TO_REFINE":
                import re
                code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
                if code_match:
                    improved_code = code_match.group(1).strip()
                    improved_code = improved_code if improved_code else current_code
                    inspection_result = "IMPROVED"
                else:
                    extracted_code = self._extract_code_from_response(response)
                    improved_code = extracted_code if extracted_code else current_code
                    inspection_result = "IMPROVED" if extracted_code else "NO_CHANGE"
            
            # Save inspection log
            additional_info = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": call_cost,
                "model_name": self.model_name,
                "inspection_result": inspection_result,
                "code_changed": improved_code != current_code,
                "original_code_length": len(current_code),
                "improved_code_length": len(improved_code),
                "raw_response": response
            }
            
            self._save_cluster_step_log(
                cluster_name=cluster_name,
                step_name="code_quality_inspection",
                prompt=prompt,
                output=improved_code,
                version="v0",
                cluster_index=cluster_index,
                total_clusters=total_clusters,
                additional_info=additional_info
            )
            
            if improved_code != current_code:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🔧 Code improved")
            else:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ✅ Code quality passed")
            
            return improved_code
            
        except Exception as e:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ Code inspection failed: {str(e)}")
            return current_code

    def _extract_tool_code_from_response(self, tool_code_str: str) -> str:
        """Extract clean Python code from tool_code field"""
        import re
        
        # Remove markdown code blocks
        cleaned_code = tool_code_str.strip()
        
        # Extract from ```python ... ```
        python_match = re.search(r'```python\s*\n(.*?)\n```', cleaned_code, re.DOTALL)
        if python_match:
            return python_match.group(1).strip()
        
        # Extract from ``` ... ```
        code_match = re.search(r'```\s*\n(.*?)\n```', cleaned_code, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Remove any remaining markdown artifacts
        cleaned_code = cleaned_code.replace('```python', '').replace('```', '').strip()
        
        return cleaned_code

    def _validate_tool_syntax(self, tool_code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax of tool code"""
        try:
            ast.parse(tool_code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)} at line {e.lineno}"
        except Exception as e:
            return False, f"Parse Error: {str(e)}"

    def _validate_tool_execution(self, tool_code: str, pre_code: str, tool_info: Dict) -> Tuple[bool, Optional[str]]:
        """Test if tool code can execute in the environment with pre_code"""
        try:
            # Create execution environment
            exec_globals = {
                '__builtins__': __builtins__,
                'math': __import__('math'),
                'numpy': __import__('numpy'),
                'scipy': __import__('scipy'),
                'sympy': __import__('sympy'),
                'json': __import__('json'),
                'datetime': __import__('datetime'),
            }
            
            # Combine pre_code and tool_code to ensure execution in the same context
            combined_code = pre_code + "\n\n" + tool_code
            
            # Execute the combined code
            exec(combined_code, exec_globals)
            
            # Check if execute function exists
            if 'execute' not in exec_globals:
                return False, "Function 'execute' not found in tool code"
            
            execute_func = exec_globals['execute']
            
            # Validate function signature against tool_info parameters
            sig = inspect.signature(execute_func)
            required_params = tool_info.get('function', {}).get('parameters', {}).get('required', [])
            all_params = list(tool_info.get('function', {}).get('parameters', {}).get('properties', {}).keys())
            
            func_params = list(sig.parameters.keys())
            
            # Check if all required parameters are present
            missing_required = set(required_params) - set(func_params)
            if missing_required:
                return False, f"Missing required parameters: {missing_required}"
            
            # Test execution with dummy parameters
            try:
                # Create dummy arguments for testing
                test_args = {}
                for param_name, param_info in tool_info.get('function', {}).get('parameters', {}).get('properties', {}).items():
                    param_type = param_info.get('type', 'string')
                    if param_type == 'string':
                        test_args[param_name] = "test_string"
                    elif param_type == 'number':
                        test_args[param_name] = 42.0
                    elif param_type == 'integer':
                        test_args[param_name] = 42
                    elif param_type == 'boolean':
                        test_args[param_name] = True
                    elif param_type == 'array':
                        test_args[param_name] = [1, 2, 3]
                    elif param_type == 'object':
                        test_args[param_name] = {"key": "value"}
                    else:
                        test_args[param_name] = "test_value"
                
                # Filter test_args to only include function parameters
                filtered_args = {k: v for k, v in test_args.items() if k in func_params}
                
                # Try to execute with dummy parameters
                result = execute_func(**filtered_args)
                
                # Check if result is a string
                if not isinstance(result, str):
                    return False, f"Function should return a string, got {type(result)}"
                
                return True, "Validation passed"
                
            except Exception as e:
                return False, f"Runtime error during test execution: {str(e)}"
                
        except Exception as e:
            return False, f"Environment setup error: {str(e)}"

    def _validate_and_fix_tools(self, tools_json: str, pre_code: str, cluster_name: str, 
                               cluster_index: int, total_clusters: int) -> Tuple[str, List[ToolValidationResult]]:
        """Validate all tools in the JSON and fix any issues using multi-threading"""
        try:
            tools_data = json.loads(tools_json)
            if not isinstance(tools_data, list):
                tools_data = [tools_data]
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🔍 Validating {len(tools_data)} tools in parallel...")

            # Prepare arguments for parallel processing
            validation_tasks = [
                (i, tool, pre_code, cluster_name, cluster_index, total_clusters) 
                for i, tool in enumerate(tools_data)
            ]

            # Use map_with_progress for multi-threaded tool validation
            # Note: The print statements inside the worker function will be interleaved.
            results = map_with_progress(
                self._process_single_tool_validation,
                validation_tasks,
                num_threads=min(len(validation_tasks), 10),
                pbar=False  # As per user request
            )

            # Unpack results
            validation_results = [res[0] for res in results]
            fixed_tools = [res[1] for res in results]
            
            # Generate summary
            valid_count = sum(1 for r in validation_results if r.is_valid)
            fixed_count = sum(1 for r in validation_results if r.fixed_code is not None)
            failed_count = len(validation_results) - valid_count
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       📊 Validation summary: {valid_count}/{len(tools_data)} valid, {fixed_count} fixed, {failed_count} failed")
            
            # Return fixed tools JSON
            fixed_tools_json = json.dumps(fixed_tools, indent=2)
            return fixed_tools_json, validation_results
            
        except Exception as e:
            logger.error(f"Tool validation failed: {e}")
            return tools_json, []

    def _process_single_tool_validation(self, args: Tuple) -> Tuple[ToolValidationResult, Dict]:
        """Worker function to validate and fix a single tool. For use with map_with_progress."""
        i, tool, pre_code, cluster_name, cluster_index, total_clusters = args
        max_fix_attempts = 3

        tool_info = tool.get('tool_info', {})
        tool_code_raw = tool.get('tool_code', '')
        tool_name = tool_info.get('function', {}).get('name', f'tool_{i}')
        
        # This print might interleave, which is expected in parallel processing
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]         🧪 Validating {tool_name}...")
        
        current_tool_code = self._extract_tool_code_from_response(tool_code_raw)
        
        validation_result = ToolValidationResult(
            tool_name=tool_name,
            is_valid=False,
            original_code=current_tool_code
        )
        
        is_valid, error_message = self._validate_single_tool(current_tool_code, pre_code, tool_info)
        
        if is_valid:
            validation_result.is_valid = True
            validation_result.validation_details = "All validations passed"
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]           ✅ {tool_name} validation passed")
        else:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]           ❌ {tool_name} failed: {error_message}")
            
            validation_result.error_message = error_message
            validation_result.validation_details = f"Initial validation failed: {error_message}"
            
            fix_attempt = 0
            fix_history = []
            
            while fix_attempt < max_fix_attempts and not validation_result.is_valid:
                fix_attempt += 1
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]           🔧 Fix attempt {fix_attempt}/{max_fix_attempts} for {tool_name}...")
                
                fixed_code = self._fix_tool_code_with_llm(
                    tool_info, current_tool_code, pre_code, error_message, cluster_name, fix_attempt
                )
                
                if not fixed_code:
                    error_msg = f"LLM fix attempt {fix_attempt} failed: no response"
                    fix_history.append(error_msg)
                    validation_result.validation_details += f" | {error_msg}"
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]             ❌ Fix attempt {fix_attempt} failed: no LLM response")
                    continue
                
                fix_is_valid, fix_error_message = self._validate_single_tool(fixed_code, pre_code, tool_info)
                
                if fix_is_valid:
                    validation_result.is_valid = True
                    validation_result.fixed_code = fixed_code
                    current_tool_code = fixed_code
                    success_msg = f"Fixed successfully on attempt {fix_attempt}"
                    fix_history.append(success_msg)
                    validation_result.validation_details += f" | {success_msg}"
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]           ✅ {tool_name} fixed successfully on attempt {fix_attempt}")
                    break
                else:
                    error_msg = f"Fix attempt {fix_attempt} validation failed: {fix_error_message}"
                    fix_history.append(error_msg)
                    validation_result.validation_details += f" | {error_msg}"
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]             ❌ Fix attempt {fix_attempt} failed: {fix_error_message}")
                    
                    current_tool_code = fixed_code
                    error_message = fix_error_message
            
            if not validation_result.is_valid:
                final_msg = f"All {max_fix_attempts} fix attempts exhausted"
                validation_result.validation_details += f" | {final_msg}"
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]           ❌ {tool_name} could not be fixed after {max_fix_attempts} attempts")
            
            if hasattr(validation_result, 'fix_history'):
                validation_result.fix_history = fix_history
            else:
                validation_result.validation_details += f" | Fix history: {'; '.join(fix_history)}"

        final_tool_code = validation_result.fixed_code if validation_result.fixed_code else validation_result.original_code
        fixed_tool = {
            "tool_info": tool_info,
            "tool_code": f"```python\n{final_tool_code}\n```"
        }
        
        return (validation_result, fixed_tool)

    def _validate_single_tool(self, tool_code: str, pre_code: str, tool_info: Dict) -> Tuple[bool, Optional[str]]:
        """Validate a single tool code (combines syntax and execution validation)"""
        # Step 1: Syntax validation
        syntax_valid, syntax_error = self._validate_tool_syntax(tool_code)
        if not syntax_valid:
            return False, f"Syntax error: {syntax_error}"
        
        # Step 2: Execution validation
        exec_valid, exec_error = self._validate_tool_execution(tool_code, pre_code, tool_info)
        if not exec_valid:
            return False, f"Execution error: {exec_error}"
        
        return True, "Validation passed"

    def _fix_tool_code_with_llm(self, tool_info: Dict, tool_code: str, pre_code: str, 
                               error_details: str, cluster_name: str, attempt_number: int = 1) -> Optional[str]:
        """Use LLM to fix problematic tool code"""
        try:
            tool_name = tool_info.get('function', {}).get('name', 'unknown_tool')
            
            # Add specific instructions for ModuleNotFoundError
            if "No module named" in error_details and f"'{cluster_name}'" in error_details:
                error_details += (
                    "\n\n--- IMPORTANT INSTRUCTION ---\n"
                    "The error indicates the code is trying to `import` a module that is not available. "
                    f"The `tool_code` MUST NOT try to `import {cluster_name}`. "
                    "Instead, it should directly use the objects, classes, and functions defined in the `pre_code`, "
                    "as they are already available in the execution scope. For example, objects like `_facade`, `prim`, `kin1d` "
                    "are already instantiated and can be used directly."
                )
            
            # Add specific instruction for return type errors
            if "Function should return a string" in error_details:
                 error_details += (
                    "\n\n--- IMPORTANT INSTRUCTION ---\n"
                    "The error indicates the `execute` function did not return a string. "
                    "The function MUST return a final answer as a string. "
                    "Do not just `print()` the result. Instead, format the result into a user-friendly "
                    "string and use the `return` keyword."
                )

            # Enhanced fixing prompt with attempt context
            fixing_prompt = TOOL_CODE_VALIDATION_PROMPT.format(
                tool_info=json.dumps(tool_info, indent=2),
                pre_code=pre_code,
                tool_code=tool_code,
                error_details=error_details,
                attempt_number=attempt_number
            )
            
            # Call LLM to fix the code
            input_tokens = count_tokens(fixing_prompt, self.model_name)
            response = call_openai_api(content=fixing_prompt, model_name=self.model_name)
            output_tokens = count_tokens(response, self.model_name) if response else 0
            call_cost = calculate_cost(input_tokens, output_tokens, self.model_name)
            
            # Update totals
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += call_cost
            
            if response:
                # Extract fixed code
                fixed_code = self._extract_code_from_response(response)
                return fixed_code
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to fix tool code for {tool_name} (attempt {attempt_number}): {e}")
            return None
        
    def _generate_openai_tools(self, cluster_name: str, code: str, cluster_index: int = 1, total_clusters: int = 1, version: str = "v0") -> Optional[str]:
        """Generate OpenAI tool format from the library code, check for missing functions, and validate."""
        try:
            # Step 1: Initial Generation
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     🔧 Generating initial OpenAI tools...")
            tool_lst = code.split("AVAILABLE_TOOLS")[1].split("= [")[1].split("]")[0].split("\n")
            tool_lst = [tool for tool in tool_lst if len(tool)>0]
            # Store detailed logs for each tool generation
            tool_generation_logs = []
            
            def fn(tool):
                tool = tool.strip().replace(",", '')
                formatted_prompt = CONVERT_TO_OPENAI_TOOL_PROMPT.format(Python_library_source_code=code, target_tool=tool)
                
                # Count tokens and track cost
                input_tokens = count_tokens(formatted_prompt, self.model_name)
                response = call_openai_api(content=formatted_prompt, model_name=self.model_name)
                output_tokens = count_tokens(response, self.model_name) if response else 0
                call_cost = calculate_cost(input_tokens, output_tokens, self.model_name)
                
                # Update totals (using self which is thread-safe for our use case)
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_cost += call_cost
                
                # Create initial tool generation log
                tool_log = {
                    "tool_name": tool.strip(),
                    "initial_generation": {
                        "prompt": formatted_prompt,
                        "response": response,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost": call_cost,
                        "model_name": self.model_name
                    },
                    "reflection": None,
                    "final_result": None,
                    "success": False,
                    "error_message": None
                }
                
                json_content = response.split("<json>")[1].split("</json>")[0].strip()
                try:
                    json.loads(json_content)
                    res = json.loads(json_content)
                    tool_log["final_result"] = res
                    tool_log["success"] = True
                except json.JSONDecodeError as e:
                    tool_log["error_message"] = f"Initial JSON decode error: {str(e)}"
                    
                    # Reflection process
                    message = [{"role":"user","content":formatted_prompt},{"role":"assistant","content":response}]
                    message.append({"role": "user", "content": f"The previous response contained a JSON format error. Please fix the JSON format error and provide the corrected version.\nOriginal response:\n{response}\nJSON Format Error:\n{str(e)}\nPlease provide only the corrected JSON within <json> and </json> tags. Make sure it's valid JSON that can be parsed successfully."})
                    
                    reflection_input_tokens = count_tokens(str(message), self.model_name)
                    reflection_response = call_openai_api_multi_turn(model_name=self.model_name, messages=message)
                    reflection_output_tokens = count_tokens(reflection_response, self.model_name) if reflection_response else 0
                    reflection_cost = calculate_cost(reflection_input_tokens, reflection_output_tokens, self.model_name)
                    
                    # Update totals for reflection
                    self.total_input_tokens += reflection_input_tokens
                    self.total_output_tokens += reflection_output_tokens
                    self.total_cost += reflection_cost
                    
                    # Log reflection details
                    tool_log["reflection"] = {
                        "messages": message,
                        "response": reflection_response,
                        "input_tokens": reflection_input_tokens,
                        "output_tokens": reflection_output_tokens,
                        "cost": reflection_cost,
                        "model_name": self.model_name
                    }
                    
                    json_content = reflection_response.split("<json>")[1].split("</json>")[0].strip()
                    try:
                        json.loads(json_content)
                        res = json.loads(json_content)
                        tool_log["final_result"] = res
                        tool_log["success"] = True
                        tool_log["error_message"] = f"Fixed after reflection: {str(e)}"
                    except json.JSONDecodeError as e2:
                        res = {"tool_info":{},"tool_code":""}
                        tool_log["final_result"] = res
                        tool_log["success"] = False
                        tool_log["error_message"] = f"Failed after reflection: {str(e2)}"
                
                # Add this tool's log to the collection (thread-safe append)
                tool_generation_logs.append(tool_log)
                
                return res
                
            # Use map_with_progress and collect results
            toolset_openai_format = map_with_progress(fn, tool_lst, num_threads=20, pbar=False)
            initial_tools_json = json.dumps(toolset_openai_format, indent=2)
            
            # Step 2: Validation and Fixing
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🔍 Starting tool validation and fixing process...")
            validated_json, validation_results = self._validate_and_fix_tools(
                initial_tools_json, code, cluster_name, cluster_index, total_clusters
            )
            
            # Save validation logs and artifacts
            cluster_log_dir = self.output_dir / f"{cluster_name}_logs"
            cluster_log_dir.mkdir(exist_ok=True)
            
            # Save initial (pre-validation) and validated JSONs
            try:
                save_json(
                    data=json.loads(initial_tools_json),
                    file_path=cluster_log_dir / f"generation_1_initial_{version}.json"
                )
            except (json.JSONDecodeError, TypeError):
                (cluster_log_dir / f"generation_1_initial_{version}.json").write_text(initial_tools_json, encoding='utf-8')
            
            try:
                save_json(
                    data=json.loads(validated_json),
                    file_path=cluster_log_dir / f"generation_2_validated_{version}.json"
                )
            except (json.JSONDecodeError, TypeError):
                (cluster_log_dir / f"generation_2_validated_{version}.json").write_text(validated_json, encoding='utf-8')
            
            # Save detailed tool generation logs
            save_json(
                data=tool_generation_logs,
                file_path=cluster_log_dir / f"tool_generation_details_{version}.json"
            )
            
            # Calculate summary statistics from logs
            successful_generations = sum(1 for log in tool_generation_logs if log["success"])
            reflection_used = sum(1 for log in tool_generation_logs if log["reflection"] is not None)
            total_generation_tokens = sum(log["initial_generation"]["input_tokens"] + log["initial_generation"]["output_tokens"] for log in tool_generation_logs)
            total_reflection_tokens = sum((log["reflection"]["input_tokens"] + log["reflection"]["output_tokens"]) if log["reflection"] else 0 for log in tool_generation_logs)
            total_generation_cost = sum(log["initial_generation"]["cost"] for log in tool_generation_logs)
            total_reflection_cost = sum(log["reflection"]["cost"] if log["reflection"] else 0 for log in tool_generation_logs)
            
            # Prepare comprehensive prompt summary for the main log
            prompt_summary = f"Generated {len(tool_lst)} OpenAI tools using CONVERT_TO_OPENAI_TOOL_PROMPT.\n"
            prompt_summary += f"Tools generated: {[tool.strip() for tool in tool_lst]}\n"
            prompt_summary += f"Library code length: {len(code)} characters\n"
            prompt_summary += f"Each tool used the same base prompt template with different target_tool values."
            
            # Log the entire generation and validation process with detailed information
            additional_info = {
                "generation_method": "one-by-one",
                "initial_tools_count": len(toolset_openai_format),
                "tools_generated": [tool.strip() for tool in tool_lst],
                "generation_statistics": {
                    "successful_generations": successful_generations,
                    "failed_generations": len(tool_lst) - successful_generations,
                    "reflections_used": reflection_used,
                    "total_generation_tokens": total_generation_tokens,
                    "total_reflection_tokens": total_reflection_tokens,
                    "total_generation_cost": total_generation_cost,
                    "total_reflection_cost": total_reflection_cost
                },
                "validation_results": [
                    {
                        "tool_name": r.tool_name,
                        "is_valid": r.is_valid,
                        "error_type": r.error_type,
                        "error_message": r.error_message,
                        "was_fixed": r.fixed_code is not None,
                        "validation_details": r.validation_details
                    }
                    for r in validation_results
                ],
                "detailed_logs_saved_to": f"tool_generation_details_{version}.json"
            }
            
            self._save_cluster_step_log(
                cluster_name=cluster_name,
                step_name=f"openai_tools_generation_and_validation",
                prompt=prompt_summary,
                output=validated_json,
                version=version,
                cluster_index=cluster_index,
                total_clusters=total_clusters,
                additional_info=additional_info
            )
            
            return validated_json
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI tools for {cluster_name}: {e}")
            return None

    def _collect_unique_questions(self, tools: List[Dict]) -> List[Dict]:
        """Collect and deduplicate questions from tools"""
        question_map = {}
        for tool in tools:
            question = tool.get('original_question', '')
            answer = tool.get('original_answer', '')
            if question and answer:
                question_map[question] = answer
        
        return [{'question': q, 'ground_truth': a} for q, a in question_map.items()]

    def _convert_to_optimizer_tools(self, openai_tools_json: str) -> List[Dict]:
        """Convert tools from the final JSON to IterativeLibraryOptimizerAgent format."""
        optimizer_tools = []
        if not openai_tools_json:
            return optimizer_tools
        
        try:
            tools_data = json.loads(openai_tools_json)
            if not isinstance(tools_data, list):
                tools_data = [tools_data]

            for tool_entry in tools_data:
                # The structure from _validate_and_fix_tools is a list of dicts like:
                # {"tool_info": {...}, "tool_code": "..."}
                if 'tool_info' in tool_entry and 'tool_code' in tool_entry:
                    cleaned_code = self._extract_tool_code_from_response(tool_entry['tool_code'])
                    optimizer_tools.append({
                        "tool_info": tool_entry['tool_info'],
                        "tool_code": cleaned_code
                    })
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse openai_tools_json in _convert_to_optimizer_tools: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in _convert_to_optimizer_tools: {e}")

        return optimizer_tools

    def _step3_optimization_evaluation(self, unique_questions: List[Dict], cluster_name: str, 
                                     current_code: str, optimizer_tools: List[Dict], 
                                     cluster_index: int, total_clusters: int) -> List[Dict]:
        """Step 3: Multi-threaded Optimization Evaluation"""
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]     🔍 Multi-threaded Optimization Evaluation")
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]       Testing {len(unique_questions)} questions with {min(len(unique_questions), 5)} threads")
        
        optimization_items = [(q, cluster_name, current_code, optimizer_tools) for q in unique_questions]
        
        optimization_results = map_with_progress(
            self._optimize_question_wrapper,
            optimization_items,
            num_threads=min(len(unique_questions), 10),  # Reduce threads to avoid conflicts
            pbar=False
        )
        
        return optimization_results

    def _step4_iterative_optimization_loop(self, unique_questions: List[Dict], cluster_name: str, 
                                         current_code: str, optimizer_tools: List[Dict], tools: List[Dict],
                                         cluster_index: int, total_clusters: int, cluster_summary: ClusterSummary):
        """Steps 3 & 4: Iterative Optimization Loop"""
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]   🔄 Step 3-4: Iterative Optimization Loop (max {self.max_review_iterations} iterations)")
        
        iteration = 0
        while iteration < self.max_review_iterations:
            iteration += 1
            cluster_summary.optimization_iterations = iteration
            
            # Step 3: Multi-threaded Optimization Evaluation
            optimization_results = self._step3_optimization_evaluation(
                unique_questions, cluster_name, current_code, optimizer_tools, cluster_index, total_clusters
            )
            
            # Save optimization results log
            current_version = f"v{iteration-1}" if iteration > 1 else "v0"
            self._save_optimization_questions_log(
                cluster_name=cluster_name,
                questions=unique_questions,
                optimization_results=optimization_results,
                version=current_version,
                cluster_index=cluster_index,
                total_clusters=total_clusters
            )
            
            # Analyze optimization results
            analysis = self._analyze_optimization_results(optimization_results)
            cluster_summary.final_optimization_results = optimization_results
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       📊 Results: {len(analysis['passed'])} passed, {len(analysis['needs_patching'])} need patching, {len(analysis['failed'])} failed")
            
            # Check if refinement is needed
            if not analysis['needs_refinement']:
                # All questions passed, exit loop
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]     🎉 All questions passed! Optimization completed in {iteration} iterations")
                cluster_summary.steps_completed.append(f'optimization_completed_iteration_{iteration}')
                break
            
            # Step 4: Code Refinement based on suggestions
            if analysis['needs_patching']:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]     🔧 Iteration {iteration}: Code Refinement ({len(analysis['needs_patching'])} issues to address)")
                
                refinement_suggestions = self._extract_refinement_suggestions(analysis['needs_patching'])
                
                if refinement_suggestions:
                    refined_code = self._refine_code_based_on_suggestions_v2(
                        current_code, refinement_suggestions, cluster_name, iteration, cluster_index, total_clusters
                    )
                    
                    if refined_code != current_code:
                        current_code = refined_code
                        version_name = f"v{iteration}"
                        
                        # Save refined version
                        self._save_versioned_cluster_code(cluster_name, current_code, version_name, cluster_index, total_clusters)
                        cluster_summary.code_versions_saved += 1
                        
                        # Regenerate OpenAI tools for the refined code
                        openai_tools_json = self._generate_openai_tools(cluster_name, current_code, cluster_index, total_clusters, version_name)
                        if openai_tools_json:
                            optimizer_tools = self._convert_to_optimizer_tools(openai_tools_json)
                            cluster_summary.openai_tools = optimizer_tools
                            
                            # Save refined OpenAI tools
                            self._save_versioned_openai_tools(cluster_name, openai_tools_json, version_name, cluster_index, total_clusters)
                            cluster_summary.openai_tools_versions_saved += 1
                        
                        cluster_summary.steps_completed.append(f'code_refinement_iteration_{iteration}')
                        print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ✅ Code refined successfully ({version_name})")
                    else:
                        print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ⚠️ No significant changes detected, stopping iterations")
                        cluster_summary.steps_completed.append(f'refinement_no_change_iteration_{iteration}')
                        break
                else:
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ⚠️ No specific refinement suggestions found, stopping iterations")
                    cluster_summary.steps_completed.append(f'no_suggestions_iteration_{iteration}')
                    break
            else:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ⚠️ Only failed results, stopping iterations")
                cluster_summary.steps_completed.append(f'only_failures_iteration_{iteration}')
                break
        
        # Save final version if different from last saved
        if iteration >= self.max_review_iterations:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ⚠️ Maximum iterations ({self.max_review_iterations}) reached")
            final_version = f"v{iteration}_final"
            self._save_versioned_cluster_code(cluster_name, current_code, final_version, cluster_index, total_clusters)
            cluster_summary.code_versions_saved += 1
            
            # Save final OpenAI tools
            openai_tools_json = self._generate_openai_tools(cluster_name, current_code, cluster_index, total_clusters, final_version)
            if openai_tools_json:
                self._save_versioned_openai_tools(cluster_name, openai_tools_json, final_version, cluster_index, total_clusters)
                cluster_summary.openai_tools_versions_saved += 1
        
        return current_code

    def _process_single_cluster(self, cluster_name: str, tools: List[Dict], cluster_index: int = 1, total_clusters: int = 1) -> Dict[str, Any]:
        """Process a single cluster through iterative optimization loop"""
        
        print(f"[{cluster_name} {cluster_index}/{total_clusters}] 📦 Processing cluster: {cluster_name} ({len(tools)} tools)")
        
        cluster_summary = ClusterSummary(cluster_name=cluster_name, total_tools=len(tools))
        
        try:
            # Step 1: Blueprint Design
            success, blueprint_output = self._step1_blueprint_design(tools, cluster_name, cluster_index, total_clusters)
            if not success:
                cluster_summary.error_message = f"Blueprint failed: {blueprint_output}"
                return cluster_summary.__dict__
            
            cluster_summary.steps_completed.append('blueprint')
            
            # Step 2: Code Implementation
            tools_code = self._extract_tools_for_blueprint(tools)
            success, implementation_output = self._step2_code_implementation(
                blueprint_output, tools_code, cluster_name, cluster_index, total_clusters
            )
            if not success:
                cluster_summary.error_message = f"Implementation failed: {implementation_output}"
                return cluster_summary.__dict__
            
            cluster_summary.steps_completed.append('implementation')
            current_code = implementation_output
            
            # Step 2a: Code Quality Inspection
            current_code = self._step2a_code_quality_inspection(current_code, cluster_name, cluster_index, total_clusters)
            cluster_summary.steps_completed.append('code_inspection')
            
            # Generate initial OpenAI tools (v0)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     🔧 Generating and Validating OpenAI tools (v0)")
            openai_tools_json = self._generate_openai_tools(cluster_name, current_code, cluster_index, total_clusters, "v0")
            if not openai_tools_json:
                cluster_summary.error_message = "Failed to generate OpenAI tools"
                return cluster_summary.__dict__
            
            # Save v0 versions (final after inspections)
            self._save_versioned_cluster_code(cluster_name, current_code, "v0", cluster_index, total_clusters)
            self._save_versioned_openai_tools(cluster_name, openai_tools_json, "v0", cluster_index, total_clusters)
            cluster_summary.code_versions_saved += 1
            cluster_summary.openai_tools_versions_saved += 1
            
            # Convert to optimizer tools format
            optimizer_tools = self._convert_to_optimizer_tools(openai_tools_json)
            cluster_summary.openai_tools = optimizer_tools
            cluster_summary.steps_completed.append('openai_tools_generation')
            
            # Collect unique questions for optimization
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ❓ Collecting unique questions")
            unique_questions = self._collect_unique_questions(tools)
            save_json(data=unique_questions, file_path=f"{self.output_dir}/{cluster_name}_unique_questions.json")
            cluster_summary.unique_questions = len(unique_questions)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ✅ Found {len(unique_questions)} unique questions")
            
            if not unique_questions:
                cluster_summary.error_message = "No valid questions found"
                return cluster_summary.__dict__
            
            # Steps 3 & 4: Iterative Optimization Loop
            current_code = self._step4_iterative_optimization_loop(
                unique_questions, cluster_name, current_code, optimizer_tools, tools,
                cluster_index, total_clusters, cluster_summary
            )
            
            # Final results
            cluster_summary.final_code = current_code
            cluster_summary.success = True
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]   ✅ Cluster {cluster_name} completed successfully!")
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     📦 Total versions saved: {cluster_summary.code_versions_saved} code, {cluster_summary.openai_tools_versions_saved} OpenAI tools")
            
            # Save cluster summary log
            self._save_cluster_summary_log(cluster_summary.__dict__, cluster_index, total_clusters)
            
        except Exception as e:
            cluster_summary.error_message = str(e)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]   ❌ Error processing {cluster_name}: {e}")
            
            # Save error cluster summary log
            self._save_cluster_summary_log(cluster_summary.__dict__, cluster_index, total_clusters)
        
        return cluster_summary.__dict__

    def _analyze_optimization_results(self, optimization_results: List[Dict]) -> Dict[str, Any]:
        """Analyze optimization results to determine if refinement is needed"""
        needs_patching_results = []
        pass_results = []
        failed_results = []
        
        for result in optimization_results:
            if not result['success']:
                failed_results.append(result)
                continue
                
            # Extract the final report from optimization result
            if isinstance(result['result'], list) and result['result']:
                final_message = result['result'][-1].get('content', '')
                if 'NEED_PATCHING' in final_message or 'is_library_helpful": "NEED_PATCHING"' in final_message:
                    needs_patching_results.append(result)
                elif 'PASS' in final_message or 'is_library_helpful": "PASS"' in final_message:
                    pass_results.append(result)
                else:
                    failed_results.append(result)
            else:
                failed_results.append(result)
        
        return {
            'needs_patching': needs_patching_results,
            'passed': pass_results,
            'failed': failed_results,
            'total': len(optimization_results),
            'needs_refinement': len(needs_patching_results) > 0 or len(failed_results) > 0
        }
    
    def _extract_refinement_suggestions(self, needs_patching_results: List[Dict]) -> str:
        """Extract refinement suggestions from optimization results that need patching"""
        suggestions = []
        
        for index, result in enumerate(needs_patching_results):
            question = result['question']
            if isinstance(result['result'], list) and result['result']:
                final_message = result['result'][-1].get('content', '')
                
                # Try to extract modification suggestions from the final report
                import re
                suggestion_match = re.search(r'"modification_suggestions":\s*"([^"]*)"', final_message)
                if suggestion_match:
                    suggestion = suggestion_match.group(1)
                    suggestions.append(f"Suggestion {index}: {suggestion}")
                else:
                    suggestions.append(f"Suggestion {index}: Question: {question}\nLibrary needs improvement for this question")
        
        return "\n\n".join(suggestions)
    
    def _refine_code_based_on_suggestions_v2(self, current_code: str, refinement_suggestions: str, 
                                           cluster_name: str, iteration: int, cluster_index: int, total_clusters: int) -> str:
        """Refine the library code based on optimization suggestions using two-step approach"""
        try:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🛠️ Applying two-step refinement (v2)...")
            
            # Step 1: Get current blueprint from conversation history
            current_blueprint = ""
            if cluster_name in self.blueprint_conversations:
                current_blueprint = self.blueprint_conversations[cluster_name][1]["content"]
            else:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ⚠️ No existing blueprint found for {cluster_name}")
                return current_code
            
            # Step 2: Convert refinement suggestions to refinement blueprint
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         📋 Generating refinement blueprint...")
            
            blueprint_prompt = LIB_REFINEMENT_BLUEPRINT_PROMPT.format(
                blueprint=current_blueprint,
                refinement_suggestions=refinement_suggestions
            )
            
            # Call LLM to generate refinement blueprint
            blueprint_input_tokens = count_tokens(blueprint_prompt, self.model_name)
            blueprint_response = call_openai_api(content=blueprint_prompt, model_name=self.model_name)
            blueprint_output_tokens = count_tokens(blueprint_response, self.model_name) if blueprint_response else 0
            blueprint_cost = calculate_cost(blueprint_input_tokens, blueprint_output_tokens, self.model_name)
            
            # Update totals
            self.total_input_tokens += blueprint_input_tokens
            self.total_output_tokens += blueprint_output_tokens
            self.total_cost += blueprint_cost
            
            if not blueprint_response:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ❌ Blueprint generation failed")
                return current_code
            
            # Step 3: Extract revised blueprint using separator
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         🔍 Extracting revised blueprint...")
            
            revised_blueprint = ""
            if "Revised SIB Blueprint" in blueprint_response:
                revised_blueprint = blueprint_response.split("Revised SIB Blueprint")[-1].strip()
            else:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ⚠️ 'Revised SIB Blueprint:' separator not found, using full response")
                revised_blueprint = blueprint_response.strip()
            
            if not revised_blueprint:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ❌ Failed to extract revised blueprint")
                return current_code

            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         💻 Generating code from revised blueprint...")

            # Step 4: Generate new code based on the revised blueprint and original code
            function_number = revised_blueprint.count("[Description]")
            reimplementation_prompt = CODE_IMPLEMENTATION_PROMPT.format(
                blueprint=revised_blueprint,
                domain=cluster_name,
                function_number=function_number
            )

            # Call LLM directly for reimplementation with retries
            max_retries = 3
            retry_count = 0
            implementation_output = ""
            reimpl_response = ""
            reimpl_input_tokens = 0
            reimpl_output_tokens = 0
            reimpl_cost = 0
            
            while "AVAILABLE_TOOLS" not in implementation_output and retry_count < max_retries:
                retry_count += 1
                if retry_count > 1:
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]         🔄 AVAILABLE_TOOLS not found, re-generating code (attempt {retry_count}/{max_retries})...")

                current_input_tokens = count_tokens(reimplementation_prompt, self.model_name)
                current_response = call_openai_api(content=reimplementation_prompt, model_name=self.model_name)
                current_output_tokens = count_tokens(current_response, self.model_name) if current_response else 0
                current_cost = calculate_cost(current_input_tokens, current_output_tokens, self.model_name)
                
                self.total_input_tokens += current_input_tokens
                self.total_output_tokens += current_output_tokens
                self.total_cost += current_cost
                
                if current_response:
                    reimpl_response = current_response
                    implementation_output = self._extract_code_from_response(current_response)

                reimpl_input_tokens = current_input_tokens
                reimpl_output_tokens = current_output_tokens
                reimpl_cost = current_cost
            
            if "AVAILABLE_TOOLS" not in implementation_output:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ❌ Code reimplementation failed: AVAILABLE_TOOLS not found after {max_retries} attempts.")
                return current_code
            
            if retry_count > 1:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ✅ Successfully generated code on attempt {retry_count}")
            
            # Save comprehensive refinement log
            refinement_version = f"v{iteration}"
            refinement_additional_info = {
                "step1_blueprint_generation": {
                    "input_tokens": blueprint_input_tokens,
                    "output_tokens": blueprint_output_tokens,
                    "cost": blueprint_cost,
                    "raw_response": blueprint_response,
                    "prompt": blueprint_prompt,
                    "revised_blueprint": revised_blueprint,
                    "original_blueprint": current_blueprint
                },
                "step2_code_implementation": {
                    "success": True,
                    "input_tokens": reimpl_input_tokens,
                    "output_tokens": reimpl_output_tokens,
                    "cost": reimpl_cost,
                    "prompt": reimplementation_prompt,
                    "raw_response": reimpl_response,
                    "error": None
                },
                "model_name": self.model_name,
                "original_code_length": len(current_code),
                "refined_code_length": len(implementation_output),
                "refinement_suggestions": refinement_suggestions,
                "approach": "two_step_refinement_v2_direct",
            }
            
            self._save_cluster_step_log(
                cluster_name=cluster_name,
                step_name=f"step4_refinement_iteration_{iteration}",
                prompt=f"{reimplementation_prompt}",
                output=implementation_output,
                version=refinement_version,
                cluster_index=cluster_index,
                total_clusters=total_clusters,
                additional_info=refinement_additional_info
            )
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ✅ Two-step refinement completed successfully")
            return implementation_output
                
        except Exception as e:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ Two-step refinement failed: {str(e)}")
            return current_code

    def _refine_code_based_on_suggestions(self, current_code: str, refinement_suggestions: str, 
                                        cluster_name: str, iteration: int, cluster_index: int, total_clusters: int) -> str:
        """Refine the library code based on optimization suggestions"""
        try:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🛠️ Applying refinement suggestions...")
            
            # Create a refinement prompt
            refinement_prompt = CODE_REFINE_PROMPT.format(
                current_code=current_code, 
                refinement_suggestions=refinement_suggestions
            )
            
            # Call LLM to refine the code
            input_tokens = count_tokens(refinement_prompt, self.model_name)
            response = call_openai_api(content=refinement_prompt, model_name=self.model_name)
            output_tokens = count_tokens(response, self.model_name) if response else 0
            call_cost = calculate_cost(input_tokens, output_tokens, self.model_name)
            
            # Update totals
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += call_cost
            
            if response:
                # Extract code from response
                refined_code = self._extract_code_from_response(response)
                
                # Save refinement log
                refinement_version = f"v{iteration}"
                refinement_additional_info = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": call_cost,
                    "model_name": self.model_name,
                    "original_code_length": len(current_code),
                    "refined_code_length": len(refined_code),
                    "refinement_suggestions": refinement_suggestions,
                    "raw_response": response
                }
                
                self._save_cluster_step_log(
                    cluster_name=cluster_name,
                    step_name=f"step4_refinement_iteration_{iteration}",
                    prompt=refinement_prompt,
                    output=refined_code,
                    version=refinement_version,
                    cluster_index=cluster_index,
                    total_clusters=total_clusters,
                    additional_info=refinement_additional_info
                )
                
                return refined_code
            else:
                return current_code
                
        except Exception as e:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ Refinement failed: {str(e)}")
            return current_code

    def _optimize_question_wrapper(self, optimization_item: Tuple) -> Dict:
        """Wrapper for multi-threaded optimization"""
        question_data, cluster_name, current_code, optimizer_tools = optimization_item
        return self._optimize_single_question(question_data, cluster_name, current_code, optimizer_tools)
    
    def _optimize_single_question(self, question_data: Dict, cluster_name: str, current_code: str, optimizer_tools: List[Dict]) -> Dict:
        """Optimize library for a single question using optimize_library_directly"""
        question = question_data['question']
        ground_truth = question_data['ground_truth']
        
        try:
            optimizer = IterativeLibraryOptimizerAgent(
                stronger_llm_model=self.model_name,
                weaker_llm_model_list=["gpt-4.1"],
                max_iterations=1,
                python_library=current_code,
                tools=optimizer_tools,
                question=question,
                ground_truth=ground_truth
            )
            
            # # Create lib folder for saving optimizer objects
            # lib_dir = self.output_dir / "lib"
            # lib_dir.mkdir(exist_ok=True)
            
            # # Create safe filename from question (remove special characters)
            # safe_question = re.sub(r'[^\w\s-]', '', question)[:50]  # Take first 50 chars
            # safe_question = re.sub(r'[\s]+', '_', safe_question)  # Replace spaces with underscores
            
            # # Save the optimizer object to lib folder
            # timestamp = datetime.now().strftime("%H%M%S")
            # optimizer_filename = f"{cluster_name}_optimizer_{safe_question}_{timestamp}.pkl"
            # optimizer_pickle_path = lib_dir / optimizer_filename
            
            # with open(optimizer_pickle_path, 'wb') as f:
            #     pickle.dump(optimizer, f)
            # print(f"[{cluster_name}]       💾 Saved optimizer to lib/{optimizer_filename}")
            
            result = optimizer.optimize_library_directly()
            
            return {
                'question': question,
                'ground_truth': ground_truth,
                'success': True,
                'result': result,
                # 'optimizer_path': str(optimizer_pickle_path)  # Add path to result for reference
            }
        except Exception as e:
            return {
                'question': question,
                'ground_truth': ground_truth,
                'success': False,
                'error': str(e)
            }

    def _save_versioned_cluster_code(self, cluster_name: str, code: str, version: str, cluster_index: int = 1, total_clusters: int = 1):
        """Save a versioned code for a cluster to a .py file"""
        try:
            file_path = self.output_dir / f"{cluster_name}_{version}.py"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       💾 Saved {cluster_name}_{version}.py")
        except Exception as e:
            pass
    
    def _save_versioned_openai_tools(self, cluster_name: str, tools_json: str, version: str, cluster_index: int = 1, total_clusters: int = 1):
        """Save versioned OpenAI tools JSON for a cluster to a .json file"""
        try:
            file_path = self.output_dir / f"{cluster_name}_{version}_openai_tools.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(tools_json)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🔧 Saved {cluster_name}_{version}_openai_tools.json")
        except Exception as e:
            pass

    def _save_cluster_code(self, cluster_name: str, code: str):
        """Save the final code for a cluster to a .py file"""
        try:
            file_path = self.output_dir / f"{cluster_name}.py"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            pass

    def _save_cluster_step_log(self, cluster_name: str, step_name: str, prompt: str, output: str, version: str = "v0", cluster_index: int = 1, total_clusters: int = 1, additional_info: Dict = None):
        """Save detailed log for each cluster step"""
        try:
            # Create cluster-specific log directory
            cluster_log_dir = self.output_dir / f"{cluster_name}_logs"
            cluster_log_dir.mkdir(exist_ok=True)
            
            # Create log entry
            log_entry = {
                "cluster_name": cluster_name,
                "cluster_index": cluster_index,
                "total_clusters": total_clusters,
                "step_name": step_name,
                "version": version,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "prompt": prompt,
                "output": output,
                "additional_info": additional_info or {}
            }
            
            # Save to JSON file
            log_filename = f"{step_name}_{version}.json"
            log_file_path = cluster_log_dir / log_filename
            
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       📝 Saved log: {log_filename}")
            
        except Exception as e:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ⚠️ Failed to save log: {e}")

    def _save_optimization_questions_log(self, cluster_name: str, questions: List[Dict], optimization_results: List[Dict], version: str = "v0", cluster_index: int = 1, total_clusters: int = 1):
        """Save detailed log for optimization questions and results"""
        try:
            # Create cluster-specific log directory
            cluster_log_dir = self.output_dir / f"{cluster_name}_logs"
            cluster_log_dir.mkdir(exist_ok=True)
            
            # Create detailed log for each question
            questions_log = {
                "cluster_name": cluster_name,
                "cluster_index": cluster_index,
                "total_clusters": total_clusters,
                "version": version,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "total_questions": len(questions),
                "questions_and_results": []
            }
            
            # Match questions with their results
            for i, (question_data, result) in enumerate(zip(questions, optimization_results)):
                question_log = {
                    "question_index": i + 1,
                    "question": question_data.get('question', ''),
                    "ground_truth": question_data.get('ground_truth', ''),
                    "success": result.get('success', False),
                    "error": result.get('error', ''),
                    "optimization_result": result.get('result', [])
                }
                questions_log["questions_and_results"].append(question_log)
            
            # Save to JSON file
            log_filename = f"step3_optimization_{version}.json"
            log_file_path = cluster_log_dir / log_filename
            
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(questions_log, f, indent=2, ensure_ascii=False)
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       📝 Saved optimization log: {log_filename}")
            
        except Exception as e:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ⚠️ Failed to save optimization log: {e}")

    def _save_cluster_summary_log(self, cluster_summary: Dict, cluster_index: int = 1, total_clusters: int = 1):
        """Save final cluster summary log"""
        try:
            cluster_name = cluster_summary.get('cluster_name', 'unknown')
            
            # Create cluster-specific log directory
            cluster_log_dir = self.output_dir / f"{cluster_name}_logs"
            cluster_log_dir.mkdir(exist_ok=True)
            
            # Add additional metadata
            summary_with_metadata = {
                **cluster_summary,
                "cluster_index": cluster_index,
                "total_clusters": total_clusters,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "model_name": self.model_name,
                "max_review_iterations": self.max_review_iterations
            }
            
            # Save to JSON file
            log_filename = f"cluster_summary.json"
            log_file_path = cluster_log_dir / log_filename
            
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(summary_with_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       📝 Saved cluster summary log")
            
        except Exception as e:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ⚠️ Failed to save cluster summary log: {e}")
    def _process_cluster_wrapper(self, cluster_item: Tuple[int, str, List[Dict], int]) -> Dict[str, Any]:
        """Wrapper for multi-threaded cluster processing with index"""
        cluster_index, cluster_name, tools, total_clusters = cluster_item
        return self._process_single_cluster(cluster_name, tools, cluster_index, total_clusters)

    def process_all_clusters(self) -> Dict[str, Any]:
        """Process all clusters with iterative optimization approach using multi-threading"""
        
        total_clusters = len(self.clusters_data)
        print(f"\n🚀 Starting iterative optimization for {total_clusters} clusters")
        print(f"🔧 Model: {self.model_name}")
        print(f"🔄 Max iterations per cluster: {self.max_review_iterations}")
        
        # List all clusters to be processed
        print(f"\n📋 Clusters to process:")
        for i, (cluster_name, tools) in enumerate(self.clusters_data.items(), 1):
            print(f"  {i}. {cluster_name} ({len(tools)} tools)")
        
        summary = {
            'total_clusters': total_clusters,
            'successful_clusters': 0,
            'failed_clusters': 0,
            'cluster_results': [],
            'processing_time': 0,
            'token_usage': {},
            'total_unique_questions': 0,
            'total_optimization_iterations': 0,
            'total_passed_questions': 0,
            'total_needs_patching_questions': 0,
            'total_failed_questions': 0
        }
        
        start_time = time.time()
        
        try:
            # Prepare cluster items with index information
            cluster_items = []
            for i, (cluster_name, tools) in enumerate(self.clusters_data.items(), 1):
                cluster_items.append((i, cluster_name, tools, total_clusters))
            
            print(f"\n🎯 Starting parallel processing...")
            
            # Use map_with_progress for multi-threaded cluster processing
            cluster_results = map_with_progress(
                self._process_cluster_wrapper,
                cluster_items,
                num_threads=min(len(cluster_items), 10),  # Use up to 10 threads for clusters
                pbar=True
            )
            
            # Process results
            for cluster_result in cluster_results:
                summary['cluster_results'].append(cluster_result)
                
                if cluster_result['success']:
                    summary['successful_clusters'] += 1
                    summary['total_unique_questions'] += cluster_result.get('unique_questions', 0)
                    summary['total_optimization_iterations'] += cluster_result.get('optimization_iterations', 0)
                    
                    # Analyze final optimization results
                    final_results = cluster_result.get('final_optimization_results', [])
                    if final_results:
                        analysis = self._analyze_optimization_results(final_results)
                        summary['total_passed_questions'] += len(analysis['passed'])
                        summary['total_needs_patching_questions'] += len(analysis['needs_patching'])
                        summary['total_failed_questions'] += len(analysis['failed'])
                    
                    # Save cluster code (final version)
                    if cluster_result['final_code']:
                        self._save_cluster_code(cluster_result['cluster_name'], cluster_result['final_code'])
                else:
                    summary['failed_clusters'] += 1
        
        except Exception as e:
            print(f"❌ Error in cluster processing: {e}")
        
        print(f"\n🏁 Processing completed!")
        print(f"✅ {summary['successful_clusters']}/{summary['total_clusters']} clusters processed successfully")
        
        summary['processing_time'] = time.time() - start_time
        
        # Token usage summary
        summary['token_usage'] = {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_cost': self.total_cost,
            'model_name': self.model_name
        }
        
        return summary

def main():
    """Main function"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Iterative Tool Cluster Optimization System')
    parser.add_argument('--local', action='store_true', default=True, help='Enable local mode')
    parser.add_argument('--file', default='/export/home/data/adaptive_merged_tool_clusters_with_QA.json', help='Path to the clusters JSON file')
    parser.add_argument('--max-review-iterations', type=int, default=3, help='Maximum number of optimization iterations per cluster')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--model-name', type=str, default='o4-mini', help='Model name to use (default: o4-mini)')
    debug_suffix = "_debug" if '--debug' in os.sys.argv else ""
    default_output_dir = f'/export/home/temp_lib/phy_lib_{timestamp}{debug_suffix}'
    parser.add_argument('--output-dir', type=str, default=default_output_dir, help='Output directory for generated code')
    parser.add_argument('--log-folder', type=str, default=default_output_dir, help='Log folder for generated code')
    args = parser.parse_args()
    
    if args.local:
        args.file = "/Users/murong.yue/Desktop/data/Nemotron_science_data_tools_saved_kinematics.json"
        args.output_dir = f"/Users/murong.yue/Desktop/temp_lib/phy_lib_{timestamp}"
        args.log_folder = f"/Users/murong.yue/Desktop/log/phy_lib_{timestamp}"
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file, progress_logger = setup_logging(debug=args.debug, log_folder=args.log_folder)
    
    try:
        # Create processor
        processor = TopDownClusterProcessor(
            clusters_file_path=args.file,
            max_review_iterations=args.max_review_iterations,
            debug=args.debug,
            model_name=args.model_name,
            progress_logger=progress_logger
        )
        processor.output_dir = output_dir
        
        # Process all clusters
        summary = processor.process_all_clusters()
        
        # Final summary
        print("\n" + "="*60)
        print("🎯 ITERATIVE OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"✅ Successful: {summary['successful_clusters']}/{summary['total_clusters']}")
        print(f"❓ Total unique questions: {summary['total_unique_questions']}")
        print(f"🔄 Total optimization iterations: {summary['total_optimization_iterations']}")
        print(f"✅ Questions passed: {summary['total_passed_questions']}")
        print(f"🔧 Questions needing patching: {summary['total_needs_patching_questions']}")
        print(f"❌ Questions failed: {summary['total_failed_questions']}")
        print(f"⏱️ Time: {summary['processing_time']:.1f}s")
        print(f"💰 Total cost: ${summary['token_usage']['total_cost']:.4f}")
        print(f"📁 Output: {args.output_dir}")
        
        # Show version statistics
        total_code_versions = sum(r.get('code_versions_saved', 0) for r in summary['cluster_results'])
        total_openai_versions = sum(r.get('openai_tools_versions_saved', 0) for r in summary['cluster_results'])
        print(f"💾 Total versions saved: {total_code_versions} code files, {total_openai_versions} OpenAI tools")
        
        # Show failed clusters
        failed_clusters = [r for r in summary['cluster_results'] if not r['success']]
        if failed_clusters:
            print(f"\n❌ Failed clusters:")
            for cluster_result in failed_clusters:
                print(f"  • {cluster_result['cluster_name']}: {cluster_result.get('error_message', 'Unknown error')}")
        
        # Show successful clusters with their optimization details
        successful_clusters = [r for r in summary['cluster_results'] if r['success']]
        if successful_clusters:
            print(f"\n✅ Successful clusters:")
            for cluster_result in successful_clusters:
                question_count = cluster_result.get('unique_questions', 0)
                iteration_count = cluster_result.get('optimization_iterations', 0)
                code_versions = cluster_result.get('code_versions_saved', 0)
                openai_versions = cluster_result.get('openai_tools_versions_saved', 0)
                final_results = cluster_result.get('final_optimization_results', [])
                
                if final_results:
                    analysis = processor._analyze_optimization_results(final_results)
                    passed = len(analysis['passed'])
                    needs_patching = len(analysis['needs_patching'])
                    failed = len(analysis['failed'])
                    print(f"  • {cluster_result['cluster_name']}: {question_count} questions, {iteration_count} iterations")
                    print(f"    Results: {passed} passed, {needs_patching} need patching, {failed} failed")
                    print(f"    Versions: {code_versions} code, {openai_versions} OpenAI tools")
                else:
                    print(f"  • {cluster_result['cluster_name']}: {question_count} questions, {iteration_count} iterations")
                    print(f"    Versions: {code_versions} code, {openai_versions} OpenAI tools")
        
        print(f"\n📝 Log: {log_file}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()

