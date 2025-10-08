import os
import logging
import json
import time
import argparse
import traceback
import inspect
import ast
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from IterativeLibraryOptimizerAgent import IterativeLibraryOptimizerAgent

from utils import read_jsonl, call_openai_api, map_with_progress, save_json, read_json, call_openai_api_multi_turn, validate_code_syntax, execute_code, apply_patch
from prompt import (
    BLUEPRINT_DESIGN_PROMPT,
    CODE_IMPLEMENTATION_PROMPT,
    CONVERT_TO_OPENAI_TOOL_PROMPT,
    CODE_INSPECTOR_PROMPT,
    OPENAI_FUNCTION_CALL_INSPECTOR_PROMPT,
    CODE_INSPECTOR_PROMPT_REVISE,
    CODE_REFINE_PROMPT,
    TOOL_CODE_VALIDATION_PROMPT,
    LIB_REFINEMENT_BLUEPRINT_PROMPT
)

# Import token counting functions
# Token counting and cost calculation removed for cleaner output

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
                 model_name: str = "gpt-5", progress_logger=None):
        self.clusters_file_path = clusters_file_path
        self.max_review_iterations = max_review_iterations
        self.debug = debug
        self.progress_logger = progress_logger or logging.getLogger('progress')
        self.model_name = model_name
        self.batch_suggestion_size = 60
        
        # Output directory - will be set by main()
        self.output_dir = None
        
        # Store blueprint for each cluster
        self.blueprints = {}
        
        # Store comprehensive processing log for each cluster
        self.cluster_processing_logs = {}
        
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

    def _save_blueprint_file(self, cluster_name: str, blueprint_content: str, version: str = "v0", cluster_index: int = 1, total_clusters: int = 1):
        """Save blueprint content to a dedicated .md file"""
        try:
            # Save to main output directory
            blueprint_file_path = self.output_dir / f"{cluster_name}_blueprint_{version}.md"
            with open(blueprint_file_path, 'w', encoding='utf-8') as f:
                f.write(blueprint_content)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       📋 Saved {cluster_name}_blueprint_{version}.md")
            
            # Also save to cluster logs directory
            cluster_log_dir = self.output_dir / f"{cluster_name}_logs"
            cluster_log_dir.mkdir(exist_ok=True)
            log_blueprint_path = cluster_log_dir / f"blueprint_{version}.md"
            with open(log_blueprint_path, 'w', encoding='utf-8') as f:
                f.write(blueprint_content)
            
        except Exception as e:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ⚠️ Failed to save blueprint: {e}")

    def _log_llm_call(self, cluster_name: str, step_name: str, prompt: str, response: str, 
                     success: bool = True, error_msg: str = None, additional_context: Dict = None) -> None:
        """Immediately log every LLM call with raw response for debugging purposes"""
        if cluster_name not in self.cluster_processing_logs:
            self._init_cluster_log(cluster_name)
        
        # Add to a dedicated LLM calls log within the cluster log
        if "llm_calls" not in self.cluster_processing_logs[cluster_name]:
            self.cluster_processing_logs[cluster_name]["llm_calls"] = []
        
        llm_call_log = {
            "call_index": len(self.cluster_processing_logs[cluster_name]["llm_calls"]) + 1,
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "prompt": prompt,
            "raw_response": response,
            "success": success,
            "error_message": error_msg,
            "additional_context": additional_context or {},
            "prompt_length": len(prompt) if prompt else 0,
            "response_length": len(response) if response else 0
        }
        
        self.cluster_processing_logs[cluster_name]["llm_calls"].append(llm_call_log)
        
        # Immediately save the log to disk to prevent data loss
        try:
            self._save_comprehensive_cluster_log(cluster_name)
        except Exception as e:
            print(f"⚠️ Failed to immediately save LLM call log: {e}")

    def _call_blueprint_step(self, input_content: str, cluster_name: str, 
                            cluster_index: int = 1, total_clusters: int = 1, version: str = "v0") -> Tuple[StepResult, str, Optional[str]]:
        """Call LLM for blueprint design step and use unified logging"""
        formatted_prompt = None
        try:
            # Format prompt
            try:
                formatted_prompt = BLUEPRINT_DESIGN_PROMPT.format(tool_code_name_list=input_content, domain=cluster_name)
            except KeyError as e:
                error_msg = f"Missing format parameter in blueprint prompt: {e}"
                self._log_llm_call(cluster_name, "blueprint_design", "", "", False, error_msg)
                raise ValueError(error_msg)
            
            # Call LLM
            response = call_openai_api(content=formatted_prompt, model_name=self.model_name)
            
            # Immediately log the LLM call
            self._log_llm_call(
                cluster_name, "blueprint_design", formatted_prompt, response or "", 
                bool(response), "Empty response from LLM" if not response else None,
                {"input_tools_count": len(input_content.split('\n')) if input_content else 0}
            )
            
            if not response:
                self._log_blueprint_step(cluster_name, formatted_prompt, "", False)
                return StepResult.FAILED, "", "Empty response from LLM"
            
            cleaned_response = response.strip()
            
            # Store blueprint for this cluster
            self.blueprints[cluster_name] = cleaned_response
            
            # Log to unified system
            self._log_blueprint_step(cluster_name, formatted_prompt, cleaned_response, True)
            
            return StepResult.SUCCESS, cleaned_response, None
                
        except Exception as e:
            error_msg = str(e)
            if formatted_prompt:
                self._log_llm_call(cluster_name, "blueprint_design", formatted_prompt, "", False, error_msg)
            self._log_blueprint_step(cluster_name, formatted_prompt or "Error in prompt formatting", error_msg, False)
            logger.error(f"❌ Error in blueprint step: {e}")
            return StepResult.FAILED, "", error_msg

    def _call_implementation_step(self, cluster_name: str, 
                                cluster_index: int = 1, total_clusters: int = 1, version: str = "v0") -> Tuple[StepResult, str, Optional[str]]:
        """Call LLM for implementation step using SIB-based approach"""
        try:
            # Get blueprint for this cluster
            blueprint_text = self.blueprints.get(cluster_name, "")
            if not blueprint_text:
                return StepResult.FAILED, "", f"No blueprint found for cluster {cluster_name}"
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       📋 Splitting blueprint into SIBs...")
            
            # Step 1: Split blueprint into individual SIBs
            sibs = self._split_blueprint_into_sibs(blueprint_text)
            if not sibs:
                return StepResult.FAILED, "", "No SIBs found in blueprint"
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         📊 Found {len(sibs)} SIBs")
            
            # Step 2: Generate function for each SIB (parallel processing)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         🔧 Generating {len(sibs)} SIB functions in parallel...")
            
            # Prepare arguments for parallel processing
            sib_tasks = [
                (sib_data, cluster_name, cluster_index, total_clusters, version) 
                for sib_data in sibs
            ]
            
            # Use map_with_progress for parallel SIB generation
            results = map_with_progress(
                self._generate_single_sib_wrapper,
                sib_tasks,
                num_threads=min(len(sib_tasks), 50),
                pbar=False
            )
            
            # Process results
            sib_functions = []
            failed_sibs = []
            
            for success, sib_function, error in results:
                if success:
                    sib_functions.append(sib_function)
                else:
                    failed_sibs.append({
                        "error": error
                    })
            
            # Check if we have enough successful SIBs
            if not sib_functions:
                error_msg = f"All {len(sibs)} SIBs failed to generate"
                return StepResult.FAILED, "", error_msg
            
            if failed_sibs:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ⚠️ {len(failed_sibs)} SIBs failed, continuing with {len(sib_functions)} successful SIBs")
            
            # Step 3: Merge all SIB functions into a single library
            merged_code = self._merge_sib_functions(sib_functions, cluster_name, cluster_index, total_clusters)
            
            # Save detailed log
            additional_info = {
                "model_name": self.model_name,
                "total_sibs": len(sibs),
                "successful_sibs": len(sib_functions),
                "failed_sibs": len(failed_sibs),
                "failed_sib_details": failed_sibs,
                "approach": "sib_based_implementation"
            }
            
            self._save_cluster_step_log(
                cluster_name=cluster_name,
                step_name=f"step_implementation",
                prompt=f"SIB-based implementation with {len(sibs)} SIBs",
                output=merged_code,
                version=version,
                cluster_index=cluster_index,
                total_clusters=total_clusters,
                additional_info=additional_info
            )
            
            return StepResult.SUCCESS, merged_code, None
                
        except Exception as e:
            logger.error(f"❌ Error in implementation step: {e}")
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
        blueprint_result, blueprint_output, blueprint_error = self._call_blueprint_step(
            tools_code, cluster_name, cluster_index, total_clusters, "v0"
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
        
        implementation_result, implementation_output, implementation_error = self._call_implementation_step(
            cluster_name, cluster_index, total_clusters, "v0"
        )
        
        if implementation_result != StepResult.SUCCESS:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ❌ Implementation failed: {implementation_error}")
            return False, implementation_error
        
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ✅ Code implementation completed")
        return True, implementation_output

    def _step2a_code_quality_inspection(self, current_code: str, cluster_name: str, 
                                       cluster_index: int, total_clusters: int) -> str:
        """Step 2a: Safer Code Quality Inspection using unified logging"""
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]     🔍 Inspecting code quality...")
        
        try:
            # Short-circuit: If this is the final merged module (has AVAILABLE_TOOLS),
            # we will NOT send it to LLM inspection to avoid accidental removal of the registry.
            # We still do syntax and basic execution checks.
            has_final_registry = False
            try:
                module_ast = ast.parse(current_code)
                for node in getattr(module_ast, 'body', []):
                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                        # Handle Assign and AnnAssign
                        if isinstance(node, ast.Assign):
                            for t in node.targets:
                                if isinstance(t, ast.Name) and t.id == 'AVAILABLE_TOOLS':
                                    has_final_registry = True
                                    break
                        else:
                            if isinstance(node.target, ast.Name) and node.target.id == 'AVAILABLE_TOOLS':
                                has_final_registry = True
                        if has_final_registry:
                            break
            except Exception:
                has_final_registry = False

            # Step 1: Use safer syntax validation from utils
            syntax_result = validate_code_syntax(current_code, timeout=5)
            syntax_valid = syntax_result["is_valid"]
            syntax_error = syntax_result["error_message"]
            
            # New behavior: On syntax error, request unified diff and apply patch (no legacy code kept)
            if not syntax_valid:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ Syntax validation failed: {syntax_error}")
                fixing_prompt = CODE_INSPECTOR_PROMPT_REVISE.format(code=current_code, error=syntax_error)
                response = call_openai_api(content=fixing_prompt, model_name=self.model_name)
                # Log this LLM call into complete log
                self._log_llm_call(
                    cluster_name,
                    "code_inspection_diff_fix",
                    fixing_prompt,
                    response or "",
                    bool(response),
                    "Empty response from LLM" if not response else None,
                    {"phase": "step2a_syntax_fix", "syntax_error": syntax_error}
                )
                diff_text = ""
                if response:
                    if "<diff>" in response and "</diff>" in response:
                        try:
                            diff_text = response.split("<diff>")[1].split("</diff>")[0].strip()
                        except Exception:
                            diff_text = response.strip()
                    else:
                        diff_text = response.strip()
                if not diff_text:
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ Diff generation failed")
                    return current_code
                patched_code = apply_patch(current_code, diff_text)
                if patched_code is None:
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ Patch application failed")
                    return current_code
                version_name = "v_syntax_fix"
                self._save_versioned_cluster_code(cluster_name, patched_code, version_name, cluster_index, total_clusters)
                self._log_code_inspection(cluster_name, current_code, patched_code, {"syntax_error": syntax_error, "diff_applied": True, "diff_text": diff_text}, True)
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🔁 Applied syntax fix via diff; saved {cluster_name}_{version_name}.py")
                return patched_code

            # Syntax is valid: rebuild AVAILABLE_TOOLS from public functions
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ✅ Syntax validation passed")
            try:
                module_ast = ast.parse(current_code)
                public_functions: List[str] = []
                for node in getattr(module_ast, 'body', []):
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                        public_functions.append(node.name)
                # Deduplicate while preserving order
                seen: set = set()
                ordered_public = []
                for name in public_functions:
                    if name not in seen:
                        seen.add(name)
                        ordered_public.append(name)
            except Exception:
                ordered_public = []

            # Remove any existing AVAILABLE_TOOLS block
            rebuilt_code = re.sub(
                r"(?ms)^\s*AVAILABLE_TOOLS\s*(?::[^\n=]+)?=\s*\[(.*?)\]\s*$",
                "",
                current_code,
            ).rstrip()

            # Append rebuilt registry
            registry_lines = "AVAILABLE_TOOLS = [\n" + "".join(f"    {n},\n" for n in ordered_public) + "]\n"
            rebuilt_code = f"{rebuilt_code}\n\n# Rebuilt registry by inspection\n{registry_lines}"

            version_name = "v_registry_rebuilt"
            self._save_versioned_cluster_code(cluster_name, rebuilt_code, version_name, cluster_index, total_clusters)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🔧 Rebuilt AVAILABLE_TOOLS with {len(ordered_public)} public functions; saved {cluster_name}_{version_name}.py")
            return rebuilt_code
            
        except Exception as e:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ Code inspection failed: {str(e)}")
            # Log inspection failure
            self._log_code_inspection(cluster_name, current_code, current_code, {"error": str(e)}, False)
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
        """Validate Python syntax of tool code using safer utils function"""
        try:
            syntax_result = validate_code_syntax(tool_code, timeout=5)
            if syntax_result["is_valid"]:
                return True, None
            else:
                error_msg = syntax_result["error_message"]
                line_num = syntax_result.get("line_number")
                if line_num:
                    return False, f"Syntax Error: {error_msg} at line {line_num}"
                else:
                    return False, f"Syntax Error: {error_msg}"
        except Exception as e:
            return False, f"Validation Error: {str(e)}"

    def _validate_tool_execution(self, tool_code: str, pre_code: str, tool_info: Dict) -> Tuple[bool, Optional[str]]:
        """Test if tool code can execute safely using utils execute_code function"""
        try:
            # Abort early if library has syntax errors
            pre_syntax = validate_code_syntax(pre_code, timeout=5)
            if not pre_syntax["is_valid"]:
                return False, f"Library syntax invalid: {pre_syntax.get('error_message')} at line {pre_syntax.get('line_number')}"

            # Step 0: Sanitize tool_code – move/remove `from __future__` imports to top-level
            future_import_lines: List[str] = []
            sanitized_tool_lines: List[str] = []
            for line in tool_code.splitlines():
                if re.match(r"^\s*from\s+__future__\s+import\s+.+", line):
                    future_import_lines.append(line.strip())
                else:
                    sanitized_tool_lines.append(line)
            raw_tool_body = "\n".join(sanitized_tool_lines).strip()

            # Step 0.1: Keep only the single `def execute(...):` block if present
            exec_block = None
            try:
                pattern = re.compile(r"^\s*def\s+execute\s*\(.*\):", re.M)
                match = pattern.search(raw_tool_body)
                if match:
                    start = match.start()
                    remainder = raw_tool_body[start:]
                    next_def = re.search(r"^\s*def\s+\w+\s*\(.*\):", remainder, re.M)
                    if next_def and next_def.start() > 0:
                        exec_block = remainder[:next_def.start()].rstrip()
                    else:
                        exec_block = remainder.rstrip()
            except Exception:
                exec_block = None
            sanitized_tool_code = (exec_block or raw_tool_body).strip()
            future_block = "\n".join(dict.fromkeys(future_import_lines))  # dedupe, keep order

            # Step 1: Combine pre_code and sanitized tool_code with future imports at the very top
            if future_block:
                combined_code = f"{future_block}\n\n{pre_code}\n\n{sanitized_tool_code}"
            else:
                combined_code = f"{pre_code}\n\n{sanitized_tool_code}"
            
            # Step 2: Test basic execution using safe execute_code
            execution_result = execute_code(combined_code, timeout=10)
            
            # Check if execution failed
            if execution_result.startswith("Error"):
                return False, f"Execution failed: {execution_result}"
            
            # Step 3: Test if 'execute' function exists and can be called
            # Create a test script that tries to call the execute function
            test_script = f"""
{combined_code}

# Test if execute function exists and is callable
try:
    import inspect
    if 'execute' not in globals():
        print("ERROR: Function 'execute' not found in tool code")
        exit(1)
    
    execute_func = globals()['execute']
    if not callable(execute_func):
        print("ERROR: 'execute' is not a callable function")
        exit(1)
    
    # Get function signature
    sig = inspect.signature(execute_func)
    func_params = list(sig.parameters.keys())
    print(f"PARAMS: {{','.join(func_params)}}")
    
    # Test with dummy parameters
    test_args = {{}}
    for param in func_params:
        test_args[param] = "test_value"  # Simple string for all params
    
    # Try to call the function
    result = execute_func(**test_args)
    
    # Check if result is a string
    if not isinstance(result, str):
        print(f"ERROR: Function should return a string, got {{type(result)}}")
        exit(1)
    
    print("SUCCESS: Function validation passed")
    
except Exception as e:
    print(f"ERROR: Runtime error during test execution: {{str(e)}}")
    exit(1)
"""
            
            test_result = execute_code(test_script, timeout=15)
            
            if test_result.startswith("Error"):
                return False, f"Function test failed: {test_result}"
            
            if "ERROR:" in test_result:
                error_line = [line for line in test_result.split('\n') if 'ERROR:' in line][0]
                return False, error_line.replace("ERROR: ", "")
            
            if "SUCCESS:" not in test_result:
                return False, f"Unexpected test result: {test_result}"
            
            # Optional: Validate function signature against tool_info if available
            if "PARAMS:" in test_result:
                params_line = [line for line in test_result.split('\n') if 'PARAMS:' in line][0]
                actual_params = params_line.replace("PARAMS: ", "").split(',') if params_line.replace("PARAMS: ", "") else []
                
                # Check required parameters if specified in tool_info
                required_params = tool_info.get('function', {}).get('parameters', {}).get('required', [])
                if required_params:
                    missing_required = set(required_params) - set(actual_params)
                    if missing_required:
                        return False, f"Missing required parameters: {missing_required}"
            
            return True, "Validation passed"
                
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
        # print(f"[{cluster_name} {cluster_index}/{total_clusters}]         🧪 Validating {tool_name}...")
        
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
            # print(f"[{cluster_name} {cluster_index}/{total_clusters}]           ✅ {tool_name} validation passed")
        else:
            # print(f"[{cluster_name} {cluster_index}/{total_clusters}]           ❌ {tool_name} failed: {error_message}")
            
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
            response = call_openai_api(content=fixing_prompt, model_name=self.model_name)
            # Log this LLM call into complete log
            self._log_llm_call(
                cluster_name, f"tool_fix_attempt_{attempt_number}", fixing_prompt, response or "",
                bool(response), "Empty response from LLM" if not response else None,
                {"tool_name": tool_name, "attempt_number": attempt_number}
            )
            
            # Update totals
            
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
        """Generate OpenAI tool format from the library code using function names from AVAILABLE_TOOLS"""
        try:
            # Step 1: Extract function names from AVAILABLE_TOOLS in the code
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     🔧 Extracting function names from code...")
            
            # Attempt robust extraction using AST, fallback to regex (supports type annotations)
            tool_names: List[str] = []
            try:
                module_ast = ast.parse(code)
                available_list_nodes: List[ast.AST] = []
                for node in getattr(module_ast, 'body', []):
                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                        target_name = None
                        value_node = None
                        if isinstance(node, ast.Assign):
                            for t in node.targets:
                                if isinstance(t, ast.Name) and t.id == 'AVAILABLE_TOOLS':
                                    target_name = t.id
                                    value_node = node.value
                                    break
                        else:
                            if isinstance(node.target, ast.Name) and node.target.id == 'AVAILABLE_TOOLS':
                                target_name = node.target.id
                                value_node = node.value
                        if target_name and isinstance(value_node, (ast.List, ast.Tuple)):
                            available_list_nodes.append(value_node)
                if available_list_nodes:
                    selected_value = available_list_nodes[-1]  # prefer the last (merged) registry
                    for elt in getattr(selected_value, 'elts', []):
                        if isinstance(elt, ast.Name):
                            tool_names.append(elt.id)
                        elif isinstance(elt, ast.Attribute):
                            tool_names.append(elt.attr)
                        elif isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name):
                            tool_names.append(elt.func.id)
            except Exception:
                # AST parse may fail; continue to regex fallback
                tool_names = []

            if not tool_names:
                pattern = r'AVAILABLE_TOOLS\s*(?::[^\n=]+)?=\s*\[(.*?)\]'
                matches = list(re.finditer(pattern, code, re.DOTALL))
                if not matches:
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ AVAILABLE_TOOLS not found in code")
                    return None
                tools_content = matches[-1].group(1)  # choose the last occurrence (merged registry)
                candidates: List[str] = []
                for part in tools_content.split(','):
                    name = part.strip()
                    name = name.split('#')[0].strip()  # strip inline comments
                    if not name:
                        continue
                    name = name.strip().strip(',').strip()
                    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
                        candidates.append(name)
                tool_names = candidates
            
            if not tool_names:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ No function names found in AVAILABLE_TOOLS")
                return None
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       📊 Found {len(tool_names)} functions: {tool_names}")
            
            # Step 2: Generate OpenAI tools for each function (parallel processing)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🔧 Generating OpenAI tools for {len(tool_names)} functions...")
            
            # Store detailed logs for each tool generation
            tool_generation_logs = []
            
            def generate_single_tool(tool_name):
                tool_name = tool_name.strip()
                formatted_prompt = CONVERT_TO_OPENAI_TOOL_PROMPT.format(
                    Python_library_source_code=code, 
                    target_tool=tool_name
                )
                
                response = call_openai_api(content=formatted_prompt, model_name=self.model_name)
                
                # Create tool generation log
                tool_log = {
                    "tool_name": tool_name,
                    "initial_generation": {
                        "prompt": formatted_prompt,
                        "response": response,
                        "model_name": self.model_name
                    },
                    "reflection": None,
                    "final_result": None,
                    "success": False,
                    "error_message": None
                }
                
                if response and "<json>" in response and "</json>" in response:
                    json_content = response.split("<json>")[1].split("</json>")[0].strip()
                    try:
                        parsed_json = json.loads(json_content)
                        # Normalize missing function name using target tool name
                        try:
                            if isinstance(parsed_json, dict):
                                ti = parsed_json.get("tool_info", {})
                                fn = ti.get("function", {}) if isinstance(ti, dict) else {}
                                if not fn.get("name"):
                                    fn["name"] = tool_name
                                    if isinstance(ti, dict):
                                        ti["function"] = fn
                                        parsed_json["tool_info"] = ti
                        except Exception:
                            pass
                        tool_log["final_result"] = parsed_json
                        tool_log["success"] = True
                        return parsed_json, tool_log
                    except json.JSONDecodeError as e:
                        tool_log["error_message"] = f"Initial JSON decode error: {str(e)}"
                        
                        # Reflection process
                        message = [
                            {"role": "user", "content": formatted_prompt},
                            {"role": "assistant", "content": response}
                        ]
                        message.append({
                            "role": "user", 
                            "content": f"The previous response contained a JSON format error. Please fix the JSON format error and provide the corrected version.\nOriginal response:\n{response}\nJSON Format Error:\n{str(e)}\nPlease provide only the corrected JSON within <json> and </json> tags. Make sure it's valid JSON that can be parsed successfully."
                        })
                        
                        reflection_response = call_openai_api_multi_turn(model_name=self.model_name, messages=message)
                        
                        # Log reflection details
                        tool_log["reflection"] = {
                            "messages": message,
                            "response": reflection_response,
                            "model_name": self.model_name
                        }
                        
                        if reflection_response and "<json>" in reflection_response and "</json>" in reflection_response:
                            json_content = reflection_response.split("<json>")[1].split("</json>")[0].strip()
                            try:
                                parsed_json = json.loads(json_content)
                                # Normalize missing function name using target tool name
                                try:
                                    if isinstance(parsed_json, dict):
                                        ti = parsed_json.get("tool_info", {})
                                        fn = ti.get("function", {}) if isinstance(ti, dict) else {}
                                        if not fn.get("name"):
                                            fn["name"] = tool_name
                                            if isinstance(ti, dict):
                                                ti["function"] = fn
                                                parsed_json["tool_info"] = ti
                                except Exception:
                                    pass
                                tool_log["final_result"] = parsed_json
                                tool_log["success"] = True
                                tool_log["error_message"] = f"Fixed after reflection: {str(e)}"
                                return parsed_json, tool_log
                            except json.JSONDecodeError as e2:
                                tool_log["final_result"] = {"tool_info": {}, "tool_code": ""}
                                tool_log["success"] = False
                                tool_log["error_message"] = f"Failed after reflection: {str(e2)}"
                        else:
                            tool_log["final_result"] = {"tool_info": {}, "tool_code": ""}
                            tool_log["success"] = False
                            tool_log["error_message"] = "No valid JSON in reflection response"
                else:
                    tool_log["final_result"] = {"tool_info": {}, "tool_code": ""}
                    tool_log["success"] = False
                    tool_log["error_message"] = "No valid JSON tags in initial response"
                
                return tool_log["final_result"], tool_log
            
            # Use map_with_progress to generate tools in parallel
            results = map_with_progress(generate_single_tool, tool_names, num_threads=20, pbar=False)
            
            # Separate results and logs
            toolset_openai_format = []
            for result, log in results:
                toolset_openai_format.append(result)
                tool_generation_logs.append(log)
            
            initial_tools_json = json.dumps(toolset_openai_format, indent=2)
            
            # Step 3: Validation and Fixing (parallel processing)
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
            # Persist raw optimization responses for each question
            try:
                self._save_optimization_questions_log(
                    cluster_name=cluster_name,
                    questions=unique_questions,
                    optimization_results=optimization_results,
                    version=f"iter_{iteration}",
                    cluster_index=cluster_index,
                    total_clusters=total_clusters,
                )
            except Exception as _e:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ⚠️ Failed to persist optimization questions log: {_e}")
            
            # Analyze optimization results
            analysis = self._analyze_optimization_results(optimization_results)
            cluster_summary.final_optimization_results = optimization_results
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       📊 Results: {len(analysis['passed'])} passed, {len(analysis['needs_patching'])} need patching, {len(analysis['failed'])} failed")
            
            # Log this optimization iteration
            refinement_details = None
            
            # Check if refinement is needed
            if not analysis['needs_refinement']:
                # All questions passed, exit loop
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]     🎉 All questions passed! Optimization completed in {iteration} iterations")
                cluster_summary.steps_completed.append(f'optimization_completed_iteration_{iteration}')
                
                # Log final successful iteration
                self._log_optimization_iteration(cluster_name, iteration, unique_questions, optimization_results, None)
                break
            
            # Step 4: Code Refinement based on suggestions
            if analysis['needs_patching']:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]     🔧 Iteration {iteration}: Code Refinement ({len(analysis['needs_patching'])} issues to address)")
                
                refinement_suggestions = self._extract_refinement_suggestions(analysis['needs_patching'])
                all_batch_info = []  # Collect info from all batches
                refined_code = current_code  # Track the code through all batches
                
                for i in range(0, len(refinement_suggestions), self.batch_suggestion_size):
                    batch_end = min(i + self.batch_suggestion_size, len(refinement_suggestions))
                    refinement_suggestions_batch = "\n\n".join(refinement_suggestions[i:batch_end])
                    
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🔧 Processing batch {i//self.batch_suggestion_size + 1}/{(len(refinement_suggestions) + self.batch_suggestion_size - 1)//self.batch_suggestion_size}")
                    
                    batch_refined_code, batch_info = self._refine_code_based_on_suggestions_v2(
                        refined_code, refinement_suggestions_batch, i, cluster_name, iteration, cluster_index, total_clusters
                    )
                    
                    # Update refined_code for next batch
                    refined_code = batch_refined_code
                    
                    # Collect batch info
                    if batch_info:
                        all_batch_info.append(batch_info)
                
                # Create refinement details for logging
                refinement_details = {
                    "total_batches": len(all_batch_info),
                    "batch_size": self.batch_suggestion_size,
                    "total_suggestions": len(refinement_suggestions),
                    "batches": all_batch_info,
                    "code_changed": refined_code != current_code,
                    "original_code_length": len(current_code),
                    "refined_code_length": len(refined_code)
                }
                
                if refined_code != current_code:
                    current_code = refined_code
                    version_name = f"v{iteration}"
                    
                    # Save refined version - only final code
                    self._save_cluster_code(cluster_name, current_code)
                    cluster_summary.code_versions_saved += 1
                    
                    # Regenerate OpenAI tools for the refined code
                    openai_tools_json = self._generate_openai_tools(cluster_name, current_code, cluster_index, total_clusters, version_name)
                    if openai_tools_json:
                        optimizer_tools = self._convert_to_optimizer_tools(openai_tools_json)
                        cluster_summary.openai_tools = optimizer_tools
                    
                    cluster_summary.steps_completed.append(f'code_refinement_iteration_{iteration}')
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ✅ Code refined successfully ({version_name})")
                else:
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ⚠️ No significant changes detected, stopping iterations")
                    cluster_summary.steps_completed.append(f'refinement_no_change_iteration_{iteration}')
                    
                    # Log iteration with no changes
                    self._log_optimization_iteration(cluster_name, iteration, unique_questions, optimization_results, refinement_details)
                    break
            else:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ⚠️ Only failed results, stopping iterations")
                cluster_summary.steps_completed.append(f'only_failures_iteration_{iteration}')
                
                # Log iteration with only failures
                self._log_optimization_iteration(cluster_name, iteration, unique_questions, optimization_results, None)
                break
            
            # Log this iteration
            self._log_optimization_iteration(cluster_name, iteration, unique_questions, optimization_results, refinement_details)
        
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
        """Process a single cluster through iterative optimization loop with unified logging"""
        
        print(f"[{cluster_name} {cluster_index}/{total_clusters}] 📦 Processing cluster: {cluster_name} ({len(tools)} tools)")
        
        # Initialize comprehensive log
        self._init_cluster_log(cluster_name)
        
        cluster_summary = ClusterSummary(cluster_name=cluster_name, total_tools=len(tools))
        
        try:
            # Step 1: Blueprint Design
            success, blueprint_output = self._step1_blueprint_design(tools, cluster_name, cluster_index, total_clusters)
            if not success:
                cluster_summary.error_message = f"Blueprint failed: {blueprint_output}"
                self._finalize_cluster_log(cluster_name, success=False, error_message=cluster_summary.error_message)
                self._save_comprehensive_cluster_log(cluster_name)
                return cluster_summary.__dict__
            
            cluster_summary.steps_completed.append('blueprint')
            
            # Step 2: Code Implementation
            tools_code = self._extract_tools_for_blueprint(tools)
            success, implementation_output = self._step2_code_implementation(
                blueprint_output, tools_code, cluster_name, cluster_index, total_clusters
            )
            if not success:
                cluster_summary.error_message = f"Implementation failed: {implementation_output}"
                self._finalize_cluster_log(cluster_name, success=False, error_message=cluster_summary.error_message)
                self._save_comprehensive_cluster_log(cluster_name)
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
                self._finalize_cluster_log(cluster_name, final_code=current_code, success=False, error_message=cluster_summary.error_message)
                self._save_comprehensive_cluster_log(cluster_name)
                return cluster_summary.__dict__
            
            # Save v0 versions - only final code
            self._save_cluster_code(cluster_name, current_code)
            cluster_summary.code_versions_saved += 1
            
            # Convert to optimizer tools format
            optimizer_tools = self._convert_to_optimizer_tools(openai_tools_json)
            cluster_summary.openai_tools = optimizer_tools
            cluster_summary.steps_completed.append('openai_tools_generation')
            
            # Collect unique questions for optimization
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ❓ Collecting unique questions")
            unique_questions = self._collect_unique_questions(tools)
            cluster_summary.unique_questions = len(unique_questions)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]     ✅ Found {len(unique_questions)} unique questions")
            
            if not unique_questions:
                cluster_summary.error_message = "No valid questions found"
                self._finalize_cluster_log(cluster_name, final_code=current_code, openai_tools=openai_tools_json, success=False, error_message=cluster_summary.error_message)
                self._save_comprehensive_cluster_log(cluster_name)
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
            
            # Finalize and save comprehensive log
            self._finalize_cluster_log(cluster_name, final_code=current_code, openai_tools=openai_tools_json, success=True)
            self._save_comprehensive_cluster_log(cluster_name)
            
        except Exception as e:
            cluster_summary.error_message = str(e)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]   ❌ Error processing {cluster_name}: {e}")
            
            # Finalize log with error
            self._finalize_cluster_log(cluster_name, success=False, error_message=str(e))
            self._save_comprehensive_cluster_log(cluster_name)
        
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
                # Prefer regex capture; fallback to simple split
                m = re.search(r"modification_suggestions:\s*(.*)", final_message, re.DOTALL)
                if m:
                    suggestion = m.group(1).strip()
                    suggestions.append(f"Suggestion {index}: {suggestion}")
                else:
                    part = final_message.split("modification_suggestions:", 1)
                    if len(part) == 2:
                        suggestion = part[1].strip()
                        suggestions.append(f"Suggestion {index}: {suggestion}")
                    else:
                        suggestions.append(f"Suggestion {index}: Question: {question}\nLibrary needs improvement for this question")
        
        return suggestions
    
    def _refine_code_based_on_suggestions_v2(self, current_code: str, refinement_suggestions: str, suggestion_index: int,
                                           cluster_name: str, iteration: int, cluster_index: int, total_clusters: int) -> Tuple[str, Optional[Dict]]:
        """Refine the library code based on optimization suggestions using two-step approach"""
        try:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🛠️ Applying two-step refinement (v2)...")
            
            # Step 1: Get current blueprint
            current_blueprint = self.blueprints.get(cluster_name, "")
            if not current_blueprint:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ⚠️ No existing blueprint found for {cluster_name}")
                error_info = {
                    "batch_index": suggestion_index,
                    "batch_size": self.batch_suggestion_size,
                    "error": "No existing blueprint found",
                    "cluster_name": cluster_name,
                    "has_blueprint": bool(self.blueprints.get(cluster_name)),
                    "model_name": self.model_name,
                    "original_code_length": len(current_code),
                    "refinement_suggestions": refinement_suggestions,
                    "approach": "two_step_refinement_v2_no_blueprint",
                }
                return current_code, error_info
            
            # Step 2: Convert refinement suggestions to refinement blueprint
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         📋 Generating refinement blueprint...")
            
            blueprint_prompt = LIB_REFINEMENT_BLUEPRINT_PROMPT.format(
                blueprint=current_blueprint,
                refinement_suggestions=refinement_suggestions
            )
            
            # Call LLM to generate refinement blueprint
            blueprint_response = call_openai_api(content=blueprint_prompt, model_name=self.model_name)
            
            # Immediately log the LLM call
            self._log_llm_call(
                cluster_name, f"refinement_blueprint_iteration_{iteration}", blueprint_prompt, blueprint_response or "",
                bool(blueprint_response), "Empty response from LLM" if not blueprint_response else None,
                {
                    "iteration": iteration,
                    "batch_index": suggestion_index,
                    "original_code_length": len(current_code),
                    "suggestions_count": len(refinement_suggestions.split('\n\n')) if refinement_suggestions else 0
                }
            )
            
            if not blueprint_response:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ❌ Blueprint generation failed")
                error_info = {
                    "batch_index": suggestion_index,
                    "batch_size": self.batch_suggestion_size,
                    "error": "Blueprint generation failed - no response from LLM",
                    "step1_blueprint_generation": {
                        "prompt": blueprint_prompt,
                        "error": "No response from LLM",
                        "original_blueprint": current_blueprint
                    },
                    "model_name": self.model_name,
                    "original_code_length": len(current_code),
                    "refinement_suggestions": refinement_suggestions,
                    "approach": "two_step_refinement_v2_no_llm_response",
                }
                return current_code, error_info
            
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
                error_info = {
                    "batch_index": suggestion_index,
                    "batch_size": self.batch_suggestion_size,
                    "error": "Failed to extract revised blueprint",
                    "blueprint_response_length": len(blueprint_response) if blueprint_response else 0,
                    "has_separator": "Revised SIB Blueprint" in blueprint_response if blueprint_response else False,
                    "blueprint_response_preview": blueprint_response[:500] if blueprint_response else "No response",
                    "step1_blueprint_generation": {
                        "raw_response": blueprint_response,
                        "prompt": blueprint_prompt,
                        "error": "Failed to extract revised blueprint from response",
                        "original_blueprint": current_blueprint
                    },
                    "model_name": self.model_name,
                    "original_code_length": len(current_code),
                    "refinement_suggestions": refinement_suggestions,
                    "approach": "two_step_refinement_v2_failed_blueprint_extraction",
                }
                return current_code, error_info

            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         💻 Generating code from revised blueprint...")

            # Step 4: Generate new code using SIB-based approach
            # Split revised blueprint into SIBs
            revised_sibs = self._split_blueprint_into_sibs(revised_blueprint)
            if not revised_sibs:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ❌ No SIBs found in revised blueprint")
                error_info = {
                    "batch_index": suggestion_index,
                    "batch_size": self.batch_suggestion_size,
                    "error": "No SIBs found in revised blueprint",
                    "step1_blueprint_generation": {
                        "raw_response": blueprint_response,
                        "prompt": blueprint_prompt,
                        "revised_blueprint": revised_blueprint,
                        "original_blueprint": current_blueprint
                    },
                    "step2_code_implementation": {
                        "success": False,
                        "error": "No SIBs found in revised blueprint",
                    },
                    "model_name": self.model_name,
                    "original_code_length": len(current_code),
                    "refined_code_length": 0,
                    "refinement_suggestions": refinement_suggestions,
                    "approach": "two_step_refinement_v2_no_sibs",
                }
                return current_code, error_info

            # Generate functions for each revised SIB (parallel processing)
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         🔧 Generating {len(revised_sibs)} revised SIB functions in parallel...")
            
            # Prepare arguments for parallel processing
            revised_sib_tasks = [
                (sib_data, cluster_name, cluster_index, total_clusters, f"refinement_v{iteration}") 
                for sib_data in revised_sibs
            ]
            
            # Use map_with_progress for parallel revised SIB generation
            revised_results = map_with_progress(
                self._generate_single_sib_wrapper,
                revised_sib_tasks,
                num_threads=min(len(revised_sib_tasks), 10),
                pbar=False
            )
            
            # Process results
            revised_sib_functions = []
            failed_revised_sibs = []
            
            for success, sib_function, error in revised_results:
                if success:
                    revised_sib_functions.append(sib_function)
                else:
                    failed_revised_sibs.append({
                        "error": error
                    })

            # Check if we have enough successful SIBs
            if not revised_sib_functions:
                error_msg = f"All {len(revised_sibs)} revised SIBs failed to generate"
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ❌ Code reimplementation failed: {error_msg}")
                error_info = {
                    "batch_index": suggestion_index,
                    "batch_size": self.batch_suggestion_size,
                    "error": error_msg,
                    "step1_blueprint_generation": {
                        "raw_response": blueprint_response,
                        "prompt": blueprint_prompt,
                        "revised_blueprint": revised_blueprint,
                        "original_blueprint": current_blueprint
                    },
                    "step2_code_implementation": {
                        "success": False,
                        "error": error_msg,
                        "total_sibs": len(revised_sibs),
                        "failed_sibs": len(failed_revised_sibs)
                    },
                    "model_name": self.model_name,
                    "original_code_length": len(current_code),
                    "refined_code_length": 0,
                    "refinement_suggestions": refinement_suggestions,
                    "approach": "two_step_refinement_v2_all_sibs_failed",
                }
                return current_code, error_info

            if failed_revised_sibs:
                print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ⚠️ {len(failed_revised_sibs)} revised SIBs failed, continuing with {len(revised_sib_functions)} successful SIBs")

            # Merge all revised SIB functions into a single library
            implementation_output = self._merge_sib_functions(revised_sib_functions, cluster_name, cluster_index, total_clusters)
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ✅ Successfully generated refined code with {len(revised_sib_functions)} SIBs")
            
            # Save comprehensive refinement log
            refinement_version = f"v{iteration}"
            refinement_additional_info = {
                "batch_index": suggestion_index,
                "batch_size": self.batch_suggestion_size,
                "suggestions_in_batch": len(refinement_suggestions.split("\n\n")) if refinement_suggestions else 0,
                "step1_blueprint_generation": {
                    "raw_response": blueprint_response,
                    "prompt": blueprint_prompt,
                    "revised_blueprint": revised_blueprint,
                    "original_blueprint": current_blueprint
                },
                "step2_code_implementation": {
                    "success": True,
                    "total_sibs": len(revised_sibs),
                    "successful_sibs": len(revised_sib_functions),
                    "failed_sibs": len(failed_revised_sibs),
                    "approach": "sib_based_refinement"
                },
                "model_name": self.model_name,
                "original_code_length": len(current_code),
                "refined_code_length": len(implementation_output),
                "refinement_suggestions": refinement_suggestions,
                "approach": "two_step_refinement_v2_sib_based",
            }
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ✅ Two-step refinement completed successfully")
            return implementation_output, refinement_additional_info
                
        except Exception as e:
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]       ❌ Two-step refinement failed: {str(e)}")
            error_info = {
                "batch_index": suggestion_index,
                "batch_size": self.batch_suggestion_size,
                "error": f"Exception during two-step refinement: {str(e)}",
                "exception_type": type(e).__name__,
                "model_name": self.model_name,
                "original_code_length": len(current_code),
                "refinement_suggestions": refinement_suggestions,
                "approach": "two_step_refinement_v2_exception",
            }
            return current_code, error_info

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
            response = call_openai_api(content=refinement_prompt, model_name=self.model_name)
            # Log this LLM call into complete log
            self._log_llm_call(
                cluster_name, f"code_refine_iteration_{iteration}", refinement_prompt, response or "",
                bool(response), "Empty response from LLM" if not response else None,
                {"iteration": iteration}
            )
            
            # Update totals
            
            if response:
                # Extract code from response
                refined_code = self._extract_code_from_response(response)
                
                # Save refinement log
                refinement_version = f"v{iteration}"
                refinement_additional_info = {
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
                'result': result.split("<final_report>")[1].split("</final_report>")[0].strip(),
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
            print(f"📁 Saved final code: {cluster_name}.py")
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
        
        return summary

    def _split_blueprint_into_sibs(self, blueprint: str) -> List[Dict[str, str]]:
        """Split blueprint into individual SIBs based on [SIB] markers"""
        import re
        
        # Split by <SIB> markers
        sib_sections = blueprint.split("<SIB>")
        sib_sections = [d.split("</SIB>")[0] if "</SIB>" in d else "" for d in sib_sections]
        sibs = []
        for i, section in enumerate(sib_sections):
            if len(section)==0:
                continue
            sibs.append({
                    "content": section,
                    "index": i
                })
        
        return sibs

    def _generate_single_sib_wrapper(self, args: Tuple) -> Tuple[bool, str, Optional[str]]:
        """Wrapper function for parallel SIB generation"""
        sib_data, cluster_name, cluster_index, total_clusters, version = args
        return self._generate_single_sib_function(sib_data, cluster_name, cluster_index, total_clusters, version)

    def _generate_single_sib_function(self, sib_data: Dict[str, str], cluster_name: str, 
                                     cluster_index: int, total_clusters: int, sib_version: str = "v0") -> Tuple[bool, str, Optional[str]]:
        """Generate Python function for a single SIB using unified logging"""
        sib_index = sib_data.get("index", 0)
        formatted_prompt = None
        
        try:
            sib_content = sib_data["content"]
            
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]         🔧 Generating SIB {sib_index}")
            
            # Format prompt for single SIB (function_number = 1 since we're generating one function)
            formatted_prompt = CODE_IMPLEMENTATION_PROMPT.format(
                blueprint=sib_content,
                domain=cluster_name,
                function_number=1
            )
            
            # Call LLM with retries
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                retry_count += 1
                
                response = call_openai_api(content=formatted_prompt, model_name=self.model_name)
                
                # Immediately log this LLM call
                self._log_llm_call(
                    cluster_name, f"sib_{sib_index}_implementation", formatted_prompt, response or "",
                    bool(response), "Empty response from LLM" if not response else None,
                    {
                        "sib_index": sib_index,
                        "retry_attempt": retry_count,
                        "max_retries": max_retries
                    }
                )
                
                if response:
                    cleaned_response = self._extract_code_from_response(response)
                    
                    # Check if we got a valid function (contains 'def ')
                    if 'def ' in cleaned_response:
                        # Log successful SIB implementation
                        self._log_sib_implementation(
                            cluster_name, sib_index, 
                            formatted_prompt, cleaned_response, True, None
                        )
                        
                        print(f"[{cluster_name} {cluster_index}/{total_clusters}]           ✅ SIB {sib_index} generated successfully")
                        return True, cleaned_response, None
                
                if retry_count < max_retries:
                    print(f"[{cluster_name} {cluster_index}/{total_clusters}]           🔄 SIB {sib_index} generation failed, retrying ({retry_count}/{max_retries})...")
            
            # All retries failed
            error_msg = f"Failed to generate SIB {sib_index} after {max_retries} attempts"
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]           ❌ {error_msg}")
            
            # Log failed SIB implementation
            self._log_sib_implementation(
                cluster_name, sib_index, 
                formatted_prompt, "", False, error_msg
            )
            
            return False, "", error_msg
            
        except Exception as e:
            error_msg = f"Exception in SIB generation: {str(e)}"
            print(f"[{cluster_name} {cluster_index}/{total_clusters}]           ❌ {error_msg}")
            
            # Log the exception
            if formatted_prompt:
                self._log_llm_call(
                    cluster_name, f"sib_{sib_index}_implementation", formatted_prompt, "", False, error_msg,
                    {"sib_index": sib_index, "exception": True}
                )
            
            # Log exception
            self._log_sib_implementation(
                cluster_name, sib_index, 
                formatted_prompt or "", "", False, error_msg
            )
            
            return False, "", error_msg

    def _extract_function_names_from_code(self, code: str) -> List[str]:
        """Extract all function names from Python code"""
        import re
        
        # Find all function definitions
        function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        function_names = re.findall(function_pattern, code)
        
        # Filter out private functions (starting with _)
        public_functions = [name for name in function_names if not name.startswith('_')]
        
        return public_functions

    def _merge_sib_functions(self, sib_functions: List[str], cluster_name: str, 
                           cluster_index: int, total_clusters: int) -> str:
        """Merge all SIB functions into a single Python library with AVAILABLE_TOOLS"""
        print(f"[{cluster_name} {cluster_index}/{total_clusters}]       🔗 Merging {len(sib_functions)} SIB functions...")
        
        # Prepare parts and import hoisting
        header_lines: List[str] = [
            "# Generated Python Library with Static Inference Blocks (SIBs)",
        ]
        future_import_lines: List[str] = []
        normal_import_lines: List[str] = []
        body_sections: List[str] = []
        all_function_names: List[str] = []

        # Helper matchers
        future_pat = re.compile(r"^\s*from\s+__future__\s+import\s+.+")
        import_pat = re.compile(r"^\s*(?:import\s+\S+|from\s+\S+\s+import\s+.+)")

        for i, sib_function in enumerate(sib_functions):
            sib_lines = sib_function.splitlines()
            sib_future: List[str] = []
            sib_imports: List[str] = []
            sib_body: List[str] = []
            for line in sib_lines:
                if future_pat.match(line):
                    sib_future.append(line.strip())
                elif import_pat.match(line):
                    sib_imports.append(line.strip())
                else:
                    sib_body.append(line)

            future_import_lines.extend(sib_future)
            normal_import_lines.extend(sib_imports)

            cleaned_body = "\n".join(sib_body).strip()
            section = [f"# === SIB {i+1} ===", cleaned_body, ""]
            body_sections.append("\n".join(section))

            function_names = self._extract_function_names_from_code(cleaned_body)
            all_function_names.extend(function_names)

        # Deduplicate imports while preserving order
        def dedupe(seq: List[str]) -> List[str]:
            seen: set = set()
            result: List[str] = []
            for item in seq:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result

        future_import_lines = dedupe(future_import_lines)
        normal_import_lines = dedupe(normal_import_lines)

        # Compose merged code: header, future imports, normal imports, body, registry
        merged_parts: List[str] = []
        merged_parts.extend(header_lines)
        # __future__ imports must be the first statements (comments allowed before)
        if future_import_lines:
            merged_parts.extend(future_import_lines)
        if normal_import_lines:
            merged_parts.extend(normal_import_lines)
        merged_parts.append("")
        merged_parts.extend(body_sections)

        # Tool Registry
        merged_parts.append("# === Tool Registry ===")
        if all_function_names:
            merged_parts.append("AVAILABLE_TOOLS = [")
            for func_name in all_function_names:
                merged_parts.append(f"    {func_name},")
            merged_parts.append("]")
        else:
            merged_parts.append("AVAILABLE_TOOLS = []")

        merged_code = "\n".join(merged_parts)

        print(f"[{cluster_name} {cluster_index}/{total_clusters}]         ✅ Merged library with {len(all_function_names)} functions: {all_function_names}")

        return merged_code

    def _init_cluster_log(self, cluster_name: str) -> None:
        """Initialize comprehensive processing log for a cluster"""
        self.cluster_processing_logs[cluster_name] = {
            "cluster_name": cluster_name,
            "model_name": self.model_name,
            "start_time": datetime.now().isoformat(),
            "steps": {
                "blueprint": {
                    "prompt": None,
                    "output": None,
                    "timestamp": None,
                    "success": False
                },
                "sib_implementations": [],  # List of SIB implementations
                "code_inspection": {
                    "original_code": None,
                    "improved_code": None,
                    "inspection_details": None,
                    "timestamp": None,
                    "success": False
                },
                "optimization_iterations": []  # List of optimization iterations
            },
            "final_results": {
                "code": None,
                "openai_tools": None,
                "success": False,
                "error_message": None
            },
            "end_time": None,
            "processing_time_seconds": None
        }

    def _log_blueprint_step(self, cluster_name: str, prompt: str, output: str, success: bool = True) -> None:
        """Log blueprint generation step"""
        if cluster_name not in self.cluster_processing_logs:
            self._init_cluster_log(cluster_name)
        
        self.cluster_processing_logs[cluster_name]["steps"]["blueprint"] = {
            "prompt": prompt,
            "output": output,
            "timestamp": datetime.now().isoformat(),
            "success": success
        }

    def _log_sib_implementation(self, cluster_name: str, sib_index: int, 
                               prompt: str, output: str, success: bool = True, error_msg: str = None) -> None:
        """Log individual SIB implementation"""
        if cluster_name not in self.cluster_processing_logs:
            self._init_cluster_log(cluster_name)
        
        sib_log = {
            "sib_index": sib_index,
            "sib_title": "SIB",
            "prompt": prompt,
            "output": output,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "error_message": error_msg
        }
        
        self.cluster_processing_logs[cluster_name]["steps"]["sib_implementations"].append(sib_log)

    def _log_code_inspection(self, cluster_name: str, original_code: str, improved_code: str, 
                           inspection_details: Dict, success: bool = True) -> None:
        """Log code inspection step"""
        if cluster_name not in self.cluster_processing_logs:
            self._init_cluster_log(cluster_name)
        
        self.cluster_processing_logs[cluster_name]["steps"]["code_inspection"] = {
            "original_code": original_code,
            "improved_code": improved_code,
            "inspection_details": inspection_details,
            "timestamp": datetime.now().isoformat(),
            "success": success
        }

    def _log_optimization_iteration(self, cluster_name: str, iteration: int, questions: List[Dict], 
                                  optimization_results: List[Dict], refinement_details: Dict = None) -> None:
        """Log optimization iteration with questions and results"""
        if cluster_name not in self.cluster_processing_logs:
            self._init_cluster_log(cluster_name)
        
        iteration_log = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "questions": questions,
            "optimization_results": optimization_results,
            "refinement_details": refinement_details,
            "needs_refinement": len([r for r in optimization_results if not r.get('success', False) or 'NEED_PATCHING' in str(r.get('result', ''))]) > 0
        }
        
        self.cluster_processing_logs[cluster_name]["steps"]["optimization_iterations"].append(iteration_log)

    def _finalize_cluster_log(self, cluster_name: str, final_code: str = None, 
                            openai_tools: str = None, success: bool = True, error_message: str = None) -> None:
        """Finalize cluster processing log"""
        if cluster_name not in self.cluster_processing_logs:
            self._init_cluster_log(cluster_name)
        
        log = self.cluster_processing_logs[cluster_name]
        log["end_time"] = datetime.now().isoformat()
        
        # Calculate processing time
        start_time = datetime.fromisoformat(log["start_time"])
        end_time = datetime.fromisoformat(log["end_time"])
        log["processing_time_seconds"] = (end_time - start_time).total_seconds()
        
        log["final_results"] = {
            "code": final_code,
            "openai_tools": openai_tools,
            "success": success,
            "error_message": error_message
        }

    def _save_comprehensive_cluster_log(self, cluster_name: str) -> None:
        """Save the comprehensive cluster processing log to a single file"""
        try:
            if cluster_name not in self.cluster_processing_logs:
                return
            
            log_file_path = self.output_dir / f"{cluster_name}_complete_log.json"
            
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.cluster_processing_logs[cluster_name], f, indent=2, ensure_ascii=False)
            
            print(f"📝 Saved comprehensive log: {cluster_name}_complete_log.json")
            
        except Exception as e:
            print(f"⚠️ Failed to save comprehensive log for {cluster_name}: {e}")

    def _log_parallel_processing_step(self, cluster_name: str, step_name: str, 
                                     input_data: Any, output_data: Any, success: bool = True, 
                                     error_msg: str = None, processing_details: Dict = None) -> None:
        """Log a parallel processing step (like OpenAI tools generation or validation)"""
        if cluster_name not in self.cluster_processing_logs:
            self._init_cluster_log(cluster_name)
        
        # Add to a dedicated parallel processing log within the cluster log
        if "parallel_steps" not in self.cluster_processing_logs[cluster_name]:
            self.cluster_processing_logs[cluster_name]["parallel_steps"] = []
        
        parallel_step_log = {
            "step_index": len(self.cluster_processing_logs[cluster_name]["parallel_steps"]) + 1,
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "input_summary": self._summarize_data_for_log(input_data),
            "output_summary": self._summarize_data_for_log(output_data),
            "success": success,
            "error_message": error_msg,
            "processing_details": processing_details or {},
        }
        
        self.cluster_processing_logs[cluster_name]["parallel_steps"].append(parallel_step_log)
        
        # Immediately save the log to disk
        try:
            self._save_comprehensive_cluster_log(cluster_name)
        except Exception as e:
            print(f"⚠️ Failed to immediately save parallel step log: {e}")

    def _summarize_data_for_log(self, data: Any) -> Dict:
        """Create a summary of data for logging without storing full content"""
        if isinstance(data, str):
            return {
                "type": "string",
                "length": len(data),
                "preview": data[:200] + "..." if len(data) > 200 else data
            }
        elif isinstance(data, list):
            return {
                "type": "list",
                "count": len(data),
                "item_types": [type(item).__name__ for item in data[:5]]  # First 5 types
            }
        elif isinstance(data, dict):
            return {
                "type": "dict",
                "keys": list(data.keys())[:10],  # First 10 keys
                "key_count": len(data)
            }
        else:
            return {
                "type": type(data).__name__,
                "str_representation": str(data)[:100]
            }

def main():
    """Main function"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Iterative Tool Cluster Optimization System')
    parser.add_argument('--local', action='store_true', default=True, help='Enable local mode')
    parser.add_argument('--file', default='/export/home/data/adaptive_merged_tool_clusters_with_QA.json', help='Path to the clusters JSON file')
    parser.add_argument('--max-review-iterations', type=int, default=3, help='Maximum number of optimization iterations per cluster')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--model-name', type=str, default='o3', help='Model name to use (default: o4-mini)')
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

