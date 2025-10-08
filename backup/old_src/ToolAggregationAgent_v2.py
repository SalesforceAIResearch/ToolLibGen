import os
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from utils import read_json, call_openai_api, call_openai_api_multi_turn, map_with_progress, validate_code_syntax, execute_code, apply_patch, code_comment2function_json
from prompt import BLUEPRINT_DESIGN_PROMPT, CODE_IMPLEMENTATION_PROMPT, CODE_INSPECTOR_PROMPT_REVISE, LIB_REFINEMENT_BLUEPRINT_PROMPT,SIB_HELPFULNESS_CHECK_PROMPT,SIB_GENERALIZATION_PROMPT,OPENAI_TOOL_IMPLEMENTATION_PROMPT
from IterativeLibraryOptimizerAgent import IterativeLibraryOptimizerAgent

@dataclass
class ToolAggregationResult:
    cluster_name: str
    total_tools: int = 0
    steps_completed: List[str] = None
    final_code: Optional[str] = None
    openai_tools: Optional[List[Dict]] = None
    success: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.steps_completed is None:
            self.steps_completed = []

class ToolAggregationAgent:
    """Processes individual tool clusters one at a time."""
    
    def __init__(self, model_name: str = "gpt-5", debug: bool = False):
        self.model_name = model_name
        self.debug = debug
        self.output_dir = None
        self.llm_call_logs = []  # Store all LLM calls for this cluster

    def _log_llm_call(self, step_name: str, prompt: str, response: str, 
                     success: bool = True, error_msg: str = None, additional_context: Dict = None) -> None:
        """Log an LLM call with all relevant information"""
        log_entry = {
            "call_index": len(self.llm_call_logs) + 1,
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "prompt": prompt,
            "response": response,
            "success": success,
            "error_message": error_msg,
            "additional_context": additional_context or {},
            "prompt_length": len(prompt) if prompt else 0,
            "response_length": len(response) if response else 0
        }
        
        self.llm_call_logs.append(log_entry)
        
        # Log basic info to console
        status = "✅" if success else "❌"
        print(f"  {status} LLM Call {log_entry['call_index']}: {step_name}")
        if not success and error_msg:
            print(f"    Error: {error_msg}")

    def _save_llm_logs(self, cluster_name: str) -> None:
        """Save all LLM call logs to a JSON file"""
        if not self.llm_call_logs:
            print(f"⚠️ No LLM call logs to save for {cluster_name}")
            return
            
        if not self.output_dir:
            print(f"⚠️ No output directory set for {cluster_name}")
            return
            
        try:
            log_file_path = self.output_dir / f"{cluster_name}_llm_calls.json"
            
            log_data = {
                "cluster_name": cluster_name,
                "model_name": self.model_name,
                "total_calls": len(self.llm_call_logs),
                "timestamp": datetime.now().isoformat(),
                "calls": self.llm_call_logs
            }
            
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            print(f"📝 Saved {len(self.llm_call_logs)} LLM calls to {log_file_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save LLM logs to {self.output_dir}: {e}")
            print(f"   Error details: {type(e).__name__}: {str(e)}")

    def _extract_tools_for_blueprint(self, tools: List[Dict]) -> str:
        """Extract tool names, descriptions, and code implementations for blueprint design"""
        if not tools:
            return "No tools found in this cluster."
        
        tool_info_parts = []
        for i, tool in enumerate(tools):
            tool_name = tool.get('name', f'tool_{i+1}')
            description = tool.get('description', f'Function: {tool_name}')
            tool_code = tool.get('tool_code', '')
            
            tool_info_parts.append(f"# Function {i+1}: {tool_name}")
            tool_info_parts.append(f"# Description: {description}")
            
            # Add code implementation if available
            if tool_code:
                # Clean the tool code (remove markdown if present)
                clean_code = self._extract_code_from_response(tool_code)
                tool_info_parts.append(f"# Code Implementation:")
                tool_info_parts.append(clean_code)
            else:
                tool_info_parts.append(f"# Code Implementation: Not available")
            
            tool_info_parts.append("")
        
        return "\n".join(tool_info_parts)

    def _design_blueprint(self, cluster_name: str, tools: List[Dict], model_name: str = None) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        Design blueprint for the given tool cluster.
        
        Returns:
            Tuple[bool, List[Dict], Optional[str]]: 
                - success: whether the blueprint design succeeded
                - sibs: list of SIB dictionaries with 'content' and 'covered_tools' keys
                - error_message: error message if failed
        """
        try:
            if model_name is None:
                model_name = self.model_name
            print(f"  📋 Designing blueprint for {cluster_name}...")
            
            # Step 1: Extract tool information for blueprint prompt
            tools_info = self._extract_tools_for_blueprint(tools)
            
            # Step 2: Call LLM with BLUEPRINT_DESIGN_PROMPT
            blueprint_prompt = BLUEPRINT_DESIGN_PROMPT.format(
                tool_code_name_list=tools_info,
                domain=cluster_name
            )
            
            blueprint_response = call_openai_api(
                content=blueprint_prompt,
                model_name=model_name
            )
            
            # Log the LLM call
            self._log_llm_call(
                step_name="blueprint_design",
                prompt=blueprint_prompt,
                response=blueprint_response or "",
                success=bool(blueprint_response),
                error_msg="Empty response from LLM" if not blueprint_response else None,
                additional_context={
                    "cluster_name": cluster_name,
                    "tools_count": len(tools),
                    "tools_info_length": len(tools_info)
                }
            )
            
            if not blueprint_response:
                return False, [], "Empty response from LLM for blueprint design"
            
            print(f"  ✅ Blueprint design completed for {cluster_name}")
            
            # Step 3: Parse SIBs from blueprint response
            sibs = self._parse_sibs_from_blueprint(blueprint_response)
            
            if not sibs:
                return False, [], "No SIBs found in blueprint response"
            
            print(f"  📊 Found {len(sibs)} SIBs in blueprint")
            
            # Step 4: Extract tool coverage from SIB content
            self._extract_covered_tools_from_sibs(sibs)
            
            # Report coverage statistics
            total_covered = sum(len(sib['covered_tools']) for sib in sibs)
            print(f"  📈 Coverage: {total_covered} tool references found in SIBs")
            
            return True, sibs, None
            
        except Exception as e:
            error_msg = f"Error in blueprint design: {str(e)}"
            print(f"  ❌ {error_msg}")
            return False, [], error_msg

    def _parse_sibs_from_blueprint(self, blueprint_response: str) -> List[Dict]:
        """Parse SIBs from blueprint response"""
        sibs = []
        
        # Split by <SIB> markers
        sib_sections = blueprint_response.split("<SIB>")
        
        for i, section in enumerate(sib_sections):
            if i == 0:  # Skip the part before the first <SIB>
                continue
                
            # Remove </SIB> if present
            if "</SIB>" in section:
                section = section.split("</SIB>")[0]
            
            section = section.strip()
            if section:
                sibs.append({
                    "index": i,
                    "content": section,
                    "covered_tools": []  # Will be filled by tool coverage analysis
                })
        
        return sibs

    def _extract_covered_tools_from_sibs(self, sibs: List[Dict]) -> None:
        """Extract covered tool indices from the [Covered Tools] section in each SIB"""
        for sib in sibs:
            sib['covered_tools'] = []
            content = sib.get('content', '')
            
            # Look for [Covered Tools] section
            covered_tools_match = re.search(r'\[Covered Tools\]\s*\n(.*?)(?:\n\[|$)', content, re.DOTALL | re.IGNORECASE)
            
            if covered_tools_match:
                covered_tools_text = covered_tools_match.group(1).strip()
                
                # Extract numbers from the text (tool indices)
                tool_indices = []
                
                # Pattern 1: "Tool X" or "tool X"
                tool_matches = re.findall(r'tool\s*(\d+)', covered_tools_text, re.IGNORECASE)
                tool_indices.extend([int(match) for match in tool_matches])
                
                # Pattern 2: standalone numbers (comma or space separated)
                number_matches = re.findall(r'\b(\d+)\b', covered_tools_text)
                tool_indices.extend([int(match) for match in number_matches])
                
                # Remove duplicates and sort
                sib['covered_tools'] = sorted(list(set(tool_indices)))
            
            if sib['covered_tools']:
                print(f"    SIB {sib['index']}: covers tools {sib['covered_tools']}")
            else:
                print(f"    SIB {sib['index']}: no tool coverage found")

    def _implement_code(self, cluster_name: str, sibs: List[Dict], tools: List[Dict]) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        Implement code for each SIB using parallel processing.
        
        Returns:
            Tuple[bool, List[Dict], Optional[str]]:
                - success: whether code implementation succeeded
                - implemented_tools: list of enhanced tools with OpenAI format
                - error_message: error message if failed
        """
        try:
            print(f"  💻 Implementing code for {len(sibs)} SIBs in {cluster_name}...")
            
            # Prepare tasks for parallel processing
            implementation_tasks = []
            for sib in sibs:
                # Get tools covered by this SIB
                covered_tools = []
                for tool_index in sib.get('covered_tools', []):
                    if 0 <= tool_index < len(tools):
                        covered_tools.append(tools[tool_index])
                
                implementation_tasks.append((
                    sib,
                    covered_tools,
                    cluster_name
                ))
            
            print(f"  🔧 Processing {len(implementation_tasks)} SIBs in parallel...")
            
            # Use map_with_progress for parallel SIB implementation
            results = map_with_progress(
                self._implement_single_sib,
                implementation_tasks,
                num_threads=min(len(implementation_tasks), 10),
                pbar=False
            )
            
            # Process results and create enhanced tool data
            implemented_tools = []
            failed_sibs = []
            
            for i, (success, tool_data, error) in enumerate(results):
                if success and tool_data:
                    # Get the corresponding SIB and task info
                    sib, covered_tools, cluster_name = implementation_tasks[i]
                    
                    # Create enhanced tool data with SIB and original tool information
                    enhanced_tool_data = {
                        "openai_tool": tool_data,
                        "sib_info": {
                            "sib_index": sib.get('index', i),
                            "sib_content_preview": sib.get('content', '')[:200] + "..." if len(sib.get('content', '')) > 200 else sib.get('content', ''),
                            "covered_tool_indices": sib.get('covered_tools', [])
                        },
                        "original_tools": []
                    }
                    
                    # Add original tool information
                    for tool_index in sib.get('covered_tools', []):
                        if 0 <= tool_index < len(tools):
                            original_tool = tools[tool_index]
                            enhanced_tool_data["original_tools"].append({
                                "tool_index": tool_index,
                                "tool_name": original_tool.get('name', f'tool_{tool_index}'),
                                "tool_description": original_tool.get('description', 'No description'),
                                "original_question": original_tool.get('original_question', ''),
                                "original_answer": original_tool.get('original_answer', '')
                            })
                    
                    implemented_tools.append(enhanced_tool_data)
                else:
                    # Get the corresponding SIB for failed cases
                    sib, covered_tools, cluster_name = implementation_tasks[i]
                    failed_sibs.append({
                        "sib_index": sib.get('index', i),
                        "covered_tool_indices": sib.get('covered_tools', []),
                        "error": error or "Unknown error"
                    })
            
            # Check results
            if not implemented_tools:
                error_msg = f"All {len(sibs)} SIBs failed to implement"
                return False, [], error_msg
            
            if failed_sibs:
                print(f"  ⚠️ {len(failed_sibs)} SIBs failed, continuing with {len(implemented_tools)} successful implementations")
            
            print(f"  ✅ Code implementation completed: {len(implemented_tools)} tools generated")
            
            return True, implemented_tools, None
            
        except Exception as e:
            error_msg = f"Error in code implementation: {str(e)}"
            print(f"  ❌ {error_msg}")
            return False, [], error_msg

    def _implement_single_sib(self, args: Tuple) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Implement a single SIB using CODE_IMPLEMENTATION_PROMPT.
        
        Args:
            args: Tuple containing (sib, covered_tools, cluster_name)
        
        Returns:
            Tuple[bool, Optional[Dict], Optional[str]]:
                - success: whether implementation succeeded
                - tool_data: OpenAI format tool data if successful
                - error_message: error message if failed
        """
        try:
            # Unpack arguments
            sib, covered_tools, cluster_name = args
            sib_index = sib.get('index', 0)
            print(f"    🔧 Implementing SIB {sib_index}...")
            
            # Prepare tool code from covered tools
            tool_code_parts = []
            for tool in covered_tools:
                tool_name = tool.get('name', 'unknown_tool')
                tool_code = tool.get('tool_code', '')
                if tool_code:
                    # Clean the tool code (remove markdown if present)
                    clean_code = self._extract_code_from_response(tool_code)
                    tool_code_parts.append(f"# Tool: {tool_name}\n{clean_code}\n")
            
            combined_tool_code = "\n".join(tool_code_parts) if tool_code_parts else "# No tool code available"
            
            # Format the implementation prompt
            implementation_prompt = CODE_IMPLEMENTATION_PROMPT.format(
                blueprint=sib.get('content', ''),
                tool_code=combined_tool_code
            )
            
            # Call LLM for implementation
            implementation_response = call_openai_api(
                content=implementation_prompt,
                model_name="gpt-5"
            )
            
            # Log the LLM call
            self._log_llm_call(
                step_name=f"sib_{sib_index}_implementation",
                prompt=implementation_prompt,
                response=implementation_response or "",
                success=bool(implementation_response),
                error_msg="Empty response from LLM" if not implementation_response else None,
                additional_context={
                    "sib_index": sib_index,
                    "covered_tools_count": len(covered_tools),
                    "tool_code_length": len(combined_tool_code)
                }
            )
            
            if not implementation_response:
                return False, None, "Empty response from LLM for SIB implementation"
            
            # Parse the JSON response with error recovery
            tool_data = self._parse_implementation_response(implementation_response, sib_index, f"sib_{sib_index}")
            
            if not tool_data:
                return False, None, f"Failed to parse implementation response for SIB {sib_index}"
            
            print(f"      ✅ SIB {sib_index} implemented successfully")
            return True, tool_data, None
            
        except Exception as e:
            error_msg = f"Error implementing SIB {sib.get('index', 0)}: {str(e)}"
            print(f"      ❌ {error_msg}")
            return False, None, error_msg

    def _extract_code_from_response(self, response: str) -> str:
        """Extract clean Python code from response (remove markdown blocks)"""
        # Prefer explicit <code>...</code> wrapping if present
        try:
            import re as _re
            code_tag_match = _re.search(r'<code>\s*(.*?)\s*</code>', response, _re.DOTALL)
            if code_tag_match:
                return code_tag_match.group(1).strip()
        except Exception:
            pass
        if "```python" in response:
            response = response.split("```python")[1]
            if response.endswith("```"):
                response = response[:-3]
            return response.strip()
        elif "```" in response:
            response = response.split("```")[1]
            if response.endswith("```"):
                response = response[:-3]
            return response.strip()
        else:
            return response.strip()
        # Extract from ```python ... ```
        # python_match = re.search(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
        # if python_match:
        #     return python_match.group(1).strip()
        
        # # Extract from ``` ... ```
        # code_match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
        # if code_match:
        #     return code_match.group(1).strip()
        
        # # Remove markdown artifacts
        # cleaned = response.replace('```python', '').replace('```', '').strip()
        # return cleaned

    def _parse_implementation_response(self, response: str, sib_index: int = 0, tool_name: str = "unknown") -> Optional[Dict]:
        """Parse the implementation response to extract tool_info and tool_code with error recovery"""
        try:
            # Look for <json> ... </json> tags first
            json_match = re.search(r'<json>\s*(.*?)\s*</json>', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
            else:
                # Try to find JSON-like content in the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
                else:
                    print(f"      ⚠️ No JSON content found in response for SIB {sib_index}")
                    return None
            
            # Parse the JSON
            try:
                tool_data = json.loads(json_content)
                
                # Validate the structure
                if not isinstance(tool_data, dict):
                    print(f"      ⚠️ JSON is not a dictionary for SIB {sib_index}")
                    return None
                
                if 'tool_info' not in tool_data or 'tool_code' not in tool_data:
                    print(f"      ⚠️ Missing required keys (tool_info/tool_code) for SIB {sib_index}")
                    return None
                
                # Comprehensive validation with all three layers
                success, validated_tool, validation_message = self._comprehensive_tool_validation(tool_data, sib_index)
                
                print(f"      📋 Validation result: {validation_message}")
                
                if validated_tool:
                    return validated_tool
                else:
                    print(f"      ❌ Validation failed completely for SIB {sib_index}")
                    return tool_data
                
            except json.JSONDecodeError as e:
                print(f"      ⚠️ JSON parsing error for SIB {sib_index}: {e}")
                
                # Try to fix the JSON using LLM
                fixed_json = self._fix_json_with_llm(json_content, str(e), sib_index, tool_name)
                
                if fixed_json:
                    try:
                        tool_data = json.loads(fixed_json)
                        
                        # Validate the structure again
                        if isinstance(tool_data, dict) and 'tool_info' in tool_data and 'tool_code' in tool_data:
                            print(f"      ✅ JSON fixed successfully for SIB {sib_index}")
                            
                            # Comprehensive validation for fixed JSON
                            success, validated_tool, validation_message = self._comprehensive_tool_validation(tool_data, sib_index)
                            
                            print(f"      📋 Fixed JSON validation result: {validation_message}")
                            
                            if validated_tool:
                                return validated_tool
                            else:
                                print(f"      ❌ Fixed JSON validation failed completely for SIB {sib_index}")
                                return tool_data
                        else:
                            print(f"      ❌ Fixed JSON still invalid for SIB {sib_index}")
                            return None
                            
                    except json.JSONDecodeError as e2:
                        print(f"      ❌ Fixed JSON still has parsing error for SIB {sib_index}: {e2}")
                        return None
                else:
                    print(f"      ❌ Failed to fix JSON for SIB {sib_index}")
                    return None
            
        except Exception as e:
            print(f"      ⚠️ Error parsing implementation response for SIB {sib_index}: {e}")
            return None

    def _fix_json_with_llm(self, broken_json: str, error_message: str, sib_index: int, tool_name: str) -> Optional[str]:
        """Use LLM to fix broken JSON with multiple strategies"""
        try:
            print(f"        🔧 Attempting to fix JSON for SIB {sib_index}...")
            
            # First try simple fixes before calling LLM
            simple_fix = self._try_simple_json_fixes(broken_json)
            if simple_fix:
                try:
                    json.loads(simple_fix)
                    print(f"        ✅ JSON fixed with simple rules for SIB {sib_index}")
                    return simple_fix
                except json.JSONDecodeError:
                    pass  # Simple fix didn't work, continue to LLM
            
            # Use LLM to fix the JSON
            fixing_prompt = f"""The following JSON has a parsing error. Please fix it and return only the corrected JSON.

Broken JSON:
{broken_json}

Error Message:
{error_message}

Common JSON issues to check:
- Missing commas between object properties
- Trailing commas before closing braces/brackets
- Unescaped quotes in strings
- Unclosed strings or objects
- Invalid escape sequences

Requirements:
1. Fix the JSON syntax error
2. Ensure it's valid JSON that can be parsed
3. Keep the same structure and content, just fix the formatting
4. Return ONLY the corrected JSON, no explanations
5. Make sure it has both "tool_info" and "tool_code" keys

Corrected JSON:"""
            
            response = call_openai_api(
                content=fixing_prompt,
                model_name=self.model_name
            )
            
            # Log the LLM call
            self._log_llm_call(
                step_name=f"json_fix_sib_{sib_index}_{tool_name}",
                prompt=fixing_prompt,
                response=response or "",
                success=bool(response),
                error_msg="Empty response from LLM" if not response else None,
                additional_context={
                    "sib_index": sib_index,
                    "tool_name": tool_name,
                    "original_error": error_message,
                    "json_length": len(broken_json),
                    "simple_fix_attempted": simple_fix is not None
                }
            )
            
            if not response:
                return None
            
            # Extract JSON from response (in case LLM added explanations)
            cleaned_response = response.strip()
            
            # Look for JSON content in the response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                return json_match.group(0)
            else:
                # If no clear JSON structure, return the whole response
                return cleaned_response
                
        except Exception as e:
            print(f"        ❌ Error in JSON fix: {e}")
            return None

    def _try_simple_json_fixes(self, broken_json: str) -> Optional[str]:
        """Try simple rule-based fixes for common JSON errors"""
        try:
            fixed = broken_json
            
            # Fix 1: Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)
            
            # Fix 2: Add missing commas between properties (simple heuristic)
            fixed = re.sub(r'}\s*"', '},"', fixed)
            fixed = re.sub(r']\s*"', '],"', fixed)
            
            # Fix 3: Ensure proper closing of objects/arrays
            open_braces = fixed.count('{')
            close_braces = fixed.count('}')
            open_brackets = fixed.count('[')
            close_brackets = fixed.count(']')
            
            # Add missing closing braces
            while close_braces < open_braces:
                fixed += '}'
                close_braces += 1
            
            # Add missing closing brackets
            while close_brackets < open_brackets:
                fixed += ']'
                close_brackets += 1
            
            return fixed if fixed != broken_json else None
            
        except Exception:
            return None

    def _validate_tools(self, cluster_name: str, enhanced_tools: List[Dict]) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        Validate and fix all enhanced tools using parallel processing.
        
        Returns:
            Tuple[bool, List[Dict], Optional[str]]:
                - success: whether validation succeeded
                - validated_tools: list of validated enhanced tools
                - error_message: error message if failed
        """
        try:
            print(f"  🔍 Validating {len(enhanced_tools)} tools in {cluster_name}...")
            
            # Prepare tasks for parallel processing
            validation_tasks = []
            for i, enhanced_tool in enumerate(enhanced_tools):
                validation_tasks.append((
                    i,
                    enhanced_tool,
                    cluster_name
                ))
            
            print(f"  🧪 Processing {len(validation_tasks)} tools in parallel...")
            
            # Use map_with_progress for parallel tool validation with comprehensive validation
            results = map_with_progress(
                self._validate_single_tool_comprehensive,
                validation_tasks,
                num_threads=min(len(validation_tasks), 10),
                pbar=False
            )
            
            # Process results
            validated_tools = []
            failed_tools = []
            
            for success, validated_tool, error in results:
                if success and validated_tool:
                    validated_tools.append(validated_tool)
                else:
                    failed_tools.append({
                        "error": error or "Unknown error"
                    })
            
            # Check results
            if not validated_tools:
                error_msg = f"All {len(enhanced_tools)} tools failed validation"
                return False, [], error_msg
            
            if failed_tools:
                print(f"  ⚠️ {len(failed_tools)} tools failed validation, continuing with {len(validated_tools)} validated tools")
            
            print(f"  ✅ Tool validation completed: {len(validated_tools)} tools validated")
            
            return True, validated_tools, None
            
        except Exception as e:
            error_msg = f"Error in tool validation: {str(e)}"
            print(f"  ❌ {error_msg}")
            return False, [], error_msg

    def _validate_single_tool_comprehensive(self, args: Tuple) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate a single enhanced tool using comprehensive validation.
        
        Args:
            args: Tuple containing (tool_index, enhanced_tool, cluster_name)
        
        Returns:
            Tuple[bool, Optional[Dict], Optional[str]]:
                - success: whether validation succeeded
                - validated_tool: validated enhanced tool if successful
                - error_message: error message if failed
        """
        try:
            tool_index, enhanced_tool, cluster_name = args
            
            # Extract tool information
            openai_tool = enhanced_tool.get('openai_tool', {})
            tool_info = openai_tool.get('tool_info', {})
            function_info = tool_info.get('function', {})
            tool_name = function_info.get('name', f'tool_{tool_index}')
            
            print(f"    🔍 Comprehensive validation for tool {tool_index}: {tool_name}")
            
            # Prepare tool data for comprehensive validation
            tool_data = {
                'tool_info': tool_info,
                'tool_code': openai_tool.get('tool_code', '')
            }
            
            if not tool_data['tool_code']:
                return False, None, "No tool code found"
            
            # Use comprehensive validation (3-layer validation)
            success, validated_tool_data, validation_message = self._comprehensive_tool_validation(
                tool_data, 
                tool_index  # Use tool_index as sib_index for logging
            )
            
            if success and validated_tool_data:
                # Update the enhanced tool with validated data
                validated_enhanced_tool = enhanced_tool.copy()
                validated_enhanced_tool['openai_tool'] = {
                    'tool_info': validated_tool_data.get('tool_info', tool_info),
                    'tool_code': validated_tool_data.get('tool_code', tool_data['tool_code'])
                }
                validated_enhanced_tool['validation_info'] = {
                    "method": "comprehensive_validation",
                    "final_status": "valid",
                    "validation_message": validation_message
                }
                
                print(f"      ✅ Tool {tool_index} passed comprehensive validation")
                return True, validated_enhanced_tool, None
            else:
                error_msg = f"Comprehensive validation failed: {validation_message}"
                print(f"      ❌ Tool {tool_index} failed comprehensive validation: {error_msg}")
                
                # Return failed tool with validation info
                failed_tool = enhanced_tool.copy()
                failed_tool['validation_info'] = {
                    "method": "comprehensive_validation",
                    "final_status": "failed",
                    "validation_message": validation_message
                }
                
                return False, failed_tool, error_msg
            
        except Exception as e:
            error_msg = f"Error in comprehensive validation for tool {tool_index}: {str(e)}"
            print(f"      ❌ {error_msg}")
            return False, None, error_msg

    def _quick_validate_tool_execution(self, tool_code: str, timeout: int = 3) -> Dict[str, Any]:
        """Quick validation by executing tool code in a sandboxed subprocess with timeout."""
        # Build a small inspector script that loads the code, checks for execute(), and prints JSON
        try:
            import base64
            from utils import execute_code

            encoded = base64.b64encode((tool_code or "").encode("utf-8")).decode("ascii")
            inspector = (
                "import base64, json, inspect\n"
                f"_code = base64.b64decode('{encoded}').decode('utf-8')\n"
                "env = {}\n"
                "try:\n"
                "    exec(_code, env)\n"
                "    dup = _code.count('def execute(')\n"
                "    if 'execute' not in env:\n"
                "        result = {\"is_valid\": False, \"error\": \"No 'execute' function found\", \"can_call\": False, \"function_params\": [], \"duplicates\": dup}\n"
                "    else:\n"
                "        fn = env['execute']\n"
                "        if not callable(fn):\n"
                "            result = {\"is_valid\": False, \"error\": \"'execute' is not callable\", \"can_call\": False, \"function_params\": [], \"duplicates\": dup}\n"
                "        else:\n"
                "            try:\n"
                "                sig = inspect.signature(fn)\n"
                "                params = list(sig.parameters.keys())\n"
                "            except Exception:\n"
                "                params = []\n"
                "            result = {\"is_valid\": True, \"error\": None, \"can_call\": True, \"function_params\": params, \"duplicates\": dup}\n"
                "except Exception as e:\n"
                "    result = {\"is_valid\": False, \"error\": f\"Execution failed: {str(e)}\", \"can_call\": False}\n"
                "print(json.dumps(result))\n"
            )

            output = execute_code(inspector, timeout=timeout)
            if output and output.strip().startswith("{"):
                import json as _json
                try:
                    data = _json.loads(output.strip().splitlines()[-1])
                    # Normalize return keys
                    return {
                        "is_valid": bool(data.get("is_valid")),
                        "error": data.get("error"),
                        "can_call": bool(data.get("can_call")),
                        "function_params": data.get("function_params", []),
                    }
                except Exception as _:
                    pass

            # Handle timeout or non-JSON outputs
            if output and "Timeout" in output:
                return {"is_valid": False, "error": "Timeout during validation", "can_call": False}
            return {"is_valid": False, "error": (output or "Unknown error"), "can_call": False}
        except Exception as e:
            return {"is_valid": False, "error": f"Validation harness error: {str(e)}", "can_call": False}

    def _validate_execute_function(self, code: str) -> Dict[str, Any]:
        """Enhanced validation that checks execute function exists, is callable, and no duplicates"""
        try:
            # Step 1: Direct execution test with enhanced checking
            temp_module = {}
            
            # Add common imports
            exec("""
import numpy as np
import math
import json
import os
import sys
from typing import *
import re
""", temp_module)
            
            # Execute the tool code
            exec(code, temp_module)
            
            # Step 2: Check for execute function
            execute_functions = []
            all_functions = []
            
            for name, obj in temp_module.items():
                if callable(obj) and not name.startswith('_') and name not in [
                    'np', 'math', 'json', 'os', 'sys', 're'
                ]:
                    all_functions.append(name)
                    if name == 'execute':
                        execute_functions.append(name)
            
            # Step 3: Validate execute function requirements
            if len(execute_functions) == 0:
                return {
                    "is_valid": False,
                    "error_message": f"No 'execute' function found. Available functions: {all_functions}"
                }
            
            if len(execute_functions) > 1:
                return {
                    "is_valid": False,
                    "error_message": f"Multiple 'execute' functions detected. Only one 'execute' function is allowed."
                }
            
            # Step 4: Test function signature
            execute_func = temp_module['execute']
            try:
                import inspect
                sig = inspect.signature(execute_func)
                params = list(sig.parameters.keys())
                
                return {
                    "is_valid": True,
                    "error_message": None,
                    "function_params": params,
                    "all_functions": all_functions
                }
                
            except Exception as e:
                return {
                    "is_valid": False,
                    "error_message": f"Cannot inspect execute function: {str(e)}"
                }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error_message": f"Code execution failed: {str(e)}"
            }

    def _fix_code_with_llm(self, code: str, error_message: str, cluster_name: str, 
                          tool_name: str, attempt: int, error_type: str, 
                          conversation_history: List[Dict] = None) -> Tuple[Optional[str], List[Dict]]:
        """Use LLM to fix code based on error message with multi-turn conversation"""
        try:
            print(f"        🔧 Attempting LLM fix for {error_type} (attempt {attempt})")
            
            # Initialize conversation history if not provided
            if conversation_history is None:
                conversation_history = []
            
            # Create the current fixing prompt
            fixing_prompt = CODE_INSPECTOR_PROMPT_REVISE.format(
                code=code,
                error=error_message
            )
            
            # Build messages for multi-turn conversation
            if not conversation_history:
                # First attempt - start new conversation
                messages = [
                    {"role": "user", "content": fixing_prompt}
                ]
            else:
                # Subsequent attempts - continue conversation
                messages = conversation_history.copy()
                
                # Add the new error and request for fix
                follow_up_message = f"""The previous fix didn't work. Here's the new error:

Code:
{code}

Error:
{error_message}

Please provide another fix in the same unified diff format. Learn from the previous attempts and avoid repeating the same mistakes."""
                
                messages.append({"role": "user", "content": follow_up_message})
            
            # Call LLM with multi-turn conversation
            response = call_openai_api_multi_turn(
                model_name=self.model_name,
                messages=messages
            )
            
            # Update conversation history
            if response:
                messages.append({"role": "assistant", "content": response})
                conversation_history = messages
            
            # Log the LLM call
            self._log_llm_call(
                step_name=f"code_fix_{error_type}_{tool_name}_attempt_{attempt}",
                prompt=str(messages),  # Log the entire conversation
                response=response or "",
                success=bool(response),
                error_msg="Empty response from LLM" if not response else None,
                additional_context={
                    "tool_name": tool_name,
                    "error_type": error_type,
                    "attempt": attempt,
                    "original_error": error_message,
                    "conversation_turns": len(messages) // 2,
                    "is_multi_turn": len(conversation_history) > 1
                }
            )
            
            if not response:
                return None, conversation_history
            
            # Extract diff from response
            diff_text = ""
            if "<diff>" in response and "</diff>" in response:
                try:
                    diff_text = response.split("<diff>")[1].split("</diff>")[0].strip()
                except Exception:
                    diff_text = response.strip()
            else:
                diff_text = response.strip()
            
            if not diff_text:
                print(f"        ❌ No diff found in LLM response")
                return None, conversation_history
            
            # Apply the patch
            fixed_code = apply_patch(code, diff_text)
            
            if fixed_code is None:
                print(f"        ❌ Failed to apply patch")
                return None, conversation_history
            
            print(f"        ✅ Successfully applied LLM fix (turn {len(conversation_history) // 2})")
            return fixed_code, conversation_history
            
        except Exception as e:
            print(f"        ❌ Error in LLM fix: {e}")
            return None, conversation_history or []

    def _collect_unique_questions(self, tools: List[Dict]) -> List[Dict]:
        """Collect and deduplicate questions from tools"""
        question_map = {}
        for tool in tools:
            question = tool.get('original_question', '')
            answer = tool.get('original_answer', '')
            if question and answer:
                question_map[question] = answer
        
        return [{'question': q, 'ground_truth': a} for q, a in question_map.items()]

    def _optimize_sib_text_only(self, cluster_name: str, validated_tool: Dict, sib_questions: List[Dict], sib_index: int) -> Dict:
        try:
            # 1) Get original SIB text
            original_sib_text = ''
            if isinstance(validated_tool, dict):
                original_sib_text = validated_tool.get('sib_text', '') or ''

            # 2) Take up to 10 questions
            questions = sib_questions[:10] if isinstance(sib_questions, list) else []

            # 2a) Parallel suggestion collection: one LLM call per question
            def _suggest_worker(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
                idx, q = item
                try:
                    q_text = q.get('question', '') or ''
                    gt_text = q.get('ground_truth', '') or ''
                    
                    # Single question suggestion prompt
                    prompt = SIB_HELPFULNESS_CHECK_PROMPT.format(
                        original_sib_text=original_sib_text,
                        q_text=q_text,
                        gt_text=gt_text
                    )
                    messages = [{"role": "user", "content": prompt}]
                    response = call_openai_api_multi_turn(
                        messages=messages,
                        model_name=self.model_name,
                    )

                    # Log per-question call
                    self._log_llm_call(
                        step_name=f"sib_{sib_index}_text_suggestion_q{idx}",
                        prompt=prompt,
                        response=response or "",
                        success=bool(response),
                        error_msg="Empty response from LLM" if not response else None,
                        additional_context={
                            "cluster_name": cluster_name,
                            "sib_index": sib_index,
                            "question_index": idx
                        }
                    )

                    # Extract JSON from <final_report>
                    def _extract_json(text: str) -> Dict[str, Any]:
                        import json as _json
                        start, end = "<final_report>", "</final_report>"
                        if start in text and end in text:
                            try:
                                body = text.split(start)[1].split(end)[0].strip()
                                return _json.loads(body)
                            except Exception:
                                return {}
                        # fallback: try to parse first JSON object
                        try:
                            first = text.find('{')
                            last = text.rfind('}')
                            if first != -1 and last != -1 and last > first:
                                return _json.loads(text[first:last+1])
                        except Exception:
                            pass
                        return {}

                    parsed = _extract_json(response or "")
                    suggestion = parsed.get('modification_suggestions', '') if isinstance(parsed, dict) else ''
                    return {"idx": idx, "suggestion": suggestion or '', "raw": response or ''}
                except Exception:
                    return {"idx": idx, "suggestion": '', "raw": ''}

            # Run parallel suggestion collection
            indexed_questions: List[Tuple[int, Dict[str, Any]]] = list(enumerate(questions, 1))
            per_q_results: List[Dict[str, Any]] = map_with_progress(
                _suggest_worker,
                indexed_questions,
                num_threads=min(len(indexed_questions), 10),
                pbar=False
            ) if indexed_questions else []

            # 2b) Aggregate suggestions (simple join)
            suggestion_lines: List[str] = []
            for r in per_q_results:
                s = r.get('suggestion', '')
                if s:
                    suggestion_lines.append(f"- From Q{r.get('idx', '?')}: {s}")
            suggestions_text = "\n".join(suggestion_lines)

            # 3) Single rewrite call using aggregated suggestions + all QA for context
            qa_lines: List[str] = []
            for i, q in enumerate(questions, 1):
                q_text = q.get('question', '') or ''
                gt_text = q.get('ground_truth', '') or ''
                if q_text:
                    qa_lines.append(f"Q{i}: {q_text}\nGT{i}: {gt_text}")
            qa_block = "\n\n".join(qa_lines) if qa_lines else "(no questions provided)"

            rewrite_prompt = f"""
You are a senior knowledge engineer. Rewrite the SIB using the aggregated suggestions. Keep standard headings and keep [Covered Tools] indices unchanged.

QA Pairs:
{qa_block}

Aggregated Suggestions:
{suggestions_text if suggestions_text else '(no suggestions)'}


Original SIB:
----- SIB START -----
{original_sib_text}
----- SIB END -----

Output only the fully rewritten SIB inside this tag. Follow the format of the original SIB, e.g., multiple classes and only one "execute" function (the name for execute function must be execute!)
Output format:
<REWRITTEN_SIB>
<complete SIB markdown>
</REWRITTEN_SIB>
"""
            messages = [{"role": "user", "content": rewrite_prompt}]
            response_rewrite = call_openai_api_multi_turn(
                messages=messages,
                model_name=self.model_name,
            )

            # Log rewrite call
            self._log_llm_call(
                step_name=f"sib_{sib_index}_text_rewrite",
                prompt=rewrite_prompt,
                response=response_rewrite or "",
                success=bool(response_rewrite),
                error_msg="Empty response from LLM" if not response_rewrite else None,
                additional_context={
                    "cluster_name": cluster_name,
                    "sib_index": sib_index,
                    "questions_count": len(questions),
                    "suggestions_len": len(suggestions_text)
                }
            )

            # Step 2c: Optimize (rewrite) SIB text to make it more general

            # Extract rewritten SIB
            def _extract_rewritten(text: str) -> str:
                start, end = "<REWRITTEN_SIB>", "</REWRITTEN_SIB>"
                if start in text and end in text:
                    try:
                        return text.split(start)[1].split(end)[0].strip()
                    except Exception:
                        return ''
                return text.strip()

            rewritten_sib = _extract_rewritten(response_rewrite or '')
            if not rewritten_sib:
                rewritten_sib = original_sib_text

            # Run a second-pass generalization to make the SIB more universal
            try:
                generalization_prompt = SIB_GENERALIZATION_PROMPT.format(
                    original_sib_text=rewritten_sib or original_sib_text
                )
                # Thread into the same conversation: add assistant rewrite, then generalization prompt
                if response_rewrite:
                    messages.append({"role": "assistant", "content": response_rewrite})
                messages.append({"role": "user", "content": generalization_prompt})
                response_generalize = call_openai_api_multi_turn(
                    messages=messages,
                    model_name=self.model_name,
                )

                # Log generalization call
                self._log_llm_call(
                    step_name=f"sib_{sib_index}_text_generalize",
                    prompt=generalization_prompt,
                    response=response_generalize or "",
                    success=bool(response_generalize),
                    error_msg="Empty response from LLM" if not response_generalize else None,
                    additional_context={
                        "cluster_name": cluster_name,
                        "sib_index": sib_index,
                    }
                )

                generalized_sib = _extract_rewritten(response_generalize or '')
                if generalized_sib:
                    rewritten_sib = generalized_sib
            except Exception:
                # If generalization fails, keep the first rewritten version
                pass

            return {
                'sib_index': sib_index,
                'rewritten_sib': rewritten_sib,
                'suggestions': suggestions_text,
                'model_name': self.model_name
            }
        except Exception as e:
            print(f"  ❌ Error optimizing SIB {sib_index}: {e}")
            return {
                'sib_index': sib_index,
                'rewritten_sib': validated_tool.get('sib_text', '') if isinstance(validated_tool, dict) else '',
                'suggestions': '',
                'model_name': self.model_name,
                'error': str(e)
            }


    def _process_single_sib_complete(self, args: Tuple) -> Tuple[bool, Optional[Dict], Dict]:
        try:
            sib, tools, cluster_name = args
            sib_index = sib.get('index', 0)

            print(f"  🔧 Processing SIB {sib_index} (parallel)...")

            processing_result = {
                "sib_index": sib_index,
                "success": False,
                "error": None,
                "step": "unknown",
                "questions_count": 0
            }

            # Step 1: Collect questions for this SIB
            sib_questions = self._get_questions_for_sib(sib, tools)
            processing_result["questions_count"] = len(sib_questions)

            # Step 2: Optimize (rewrite) SIB text first (single-pass rewrite based on questions)
            optimized_text_info = None
            if sib_questions:
                print(f"    🔄 Optimizing SIB {sib_index} (text-only) with {len(sib_questions)} questions...")
                optimized_text_info = self._optimize_sib_text_only(
                    cluster_name,
                    { 'sib_text': sib.get('content', '') },
                    sib_questions,
                    sib_index
                )
                if isinstance(optimized_text_info, dict) and optimized_text_info.get('rewritten_sib'):
                    sib['content'] = optimized_text_info['rewritten_sib']            

            # Step 3: Generate OpenAI tool from the (possibly rewritten) SIB
            print("Generating OpenAI tool for this SIB")
            success_gen, sib_tool, error_msg_gen = self._generate_sib_tool(cluster_name, sib, tools)
            if not success_gen:
                processing_result.update({
                    "success": False,
                    "error": error_msg_gen,
                    "step": "tool_generation"
                })
                print(f"    ❌ SIB {sib_index} tool generation failed: {error_msg_gen}")
                self._save_llm_logs(f"{cluster_name}_sib_{sib_index}")
                return False, None, processing_result

            # Step 4: Comprehensive validation of the generated tool (skipped)
            # success_val, validated_tool, error_msg_val = self._validate_sib_tool_comprehensive(cluster_name, sib_tool, sib_index)
            # if not success_val:
            #     processing_result.update({
            #         "success": False,
            #         "error": error_msg_val,
            #         "step": "validation"
            #     })
            #     print(f"    ❌ SIB {sib_index} validation failed: {error_msg_val}")
            #     self._save_llm_logs(f"{cluster_name}_sib_{sib_index}")
            #     return False, None, processing_result
            validated_tool = sib_tool

            # Step 5: Save SIB (markdown) and final tools
            if hasattr(self, 'output_dir') and self.output_dir:
                self._save_sib_as_markdown(sib, validated_tool, self.output_dir, cluster_name)

            processing_result.update({
                "success": True,
                "step": "completed_with_text_optimization" if sib_questions else "completed_without_optimization"
            })

            print(f"    ✅ SIB {sib_index} completed")
            self._save_llm_logs(f"{cluster_name}_sib_{sib_index}")
            self._save_final_openai_tools(f"{cluster_name}_sib_{sib_index}", [validated_tool])
            return True, validated_tool, processing_result

        except Exception as e:
            error_msg = f"Error processing SIB {sib.get('index', 0)}: {str(e)}"
            print(f"    ❌ {error_msg}")
            processing_result = {
                "sib_index": sib.get('index', 0),
                "success": False,
                "error": error_msg,
                "step": "exception",
                "questions_count": 0
            }
            self._save_llm_logs(f"{cluster_name}_sib_{sib.get('index', 0)}")
            return False, None, processing_result



    # def _process_single_sib_complete(self, args: Tuple) -> Tuple[bool, Optional[Dict], Dict]:
    #     """
    #     Complete processing of a single SIB: generation + validation + optimization
        
    #     Args:
    #         args: Tuple containing (sib, tools, cluster_name)
        
    #     Returns:
    #         Tuple[bool, Optional[Dict], Dict]:
    #             - success: whether SIB processing succeeded
    #             - final_tool: final optimized tool if successful
    #             - processing_result: detailed processing result for logging
    #     """
    #     try:
    #         sib, tools, cluster_name = args
    #         sib_index = sib.get('index', 0)
            
    #         print(f"  🔧 Processing SIB {sib_index} (parallel)...")
            
    #         processing_result = {
    #             "sib_index": sib_index,
    #             "success": False,
    #             "error": None,
    #             "step": "unknown",
    #             "questions_count": 0
    #         }
            
    #         # Step 1: Generate OpenAI tool for this SIB
    #         print("Generating OpenAI tool for this SIB")
    #         success, sib_tool, error_msg = self._generate_sib_tool(cluster_name, sib, tools)
            
    #         if not success:
    #             processing_result.update({
    #                 "success": False,
    #                 "error": error_msg,
    #                 "step": "tool_generation"
    #             })
    #             print(f"    ❌ SIB {sib_index} tool generation failed: {error_msg}")
    #             self._save_llm_logs(f"{cluster_name}_sib_{sib_index}")
    #             return False, None, processing_result
            
    #         # Step 2: Validate the generated tool using comprehensive validation
    #         success, validated_tool, error_msg = self._validate_sib_tool_comprehensive(cluster_name, sib_tool, sib_index)
            
    #         if not success:
    #             processing_result.update({
    #                 "success": False,
    #                 "error": error_msg,
    #                 "step": "validation"
    #             })
    #             print(f"    ❌ SIB {sib_index} validation failed: {error_msg}")
    #             self._save_llm_logs(f"{cluster_name}_sib_{sib_index}")
    #             return False, None, processing_result
            
    #         # Step 3: Find questions for this SIB and optimize
    #         sib_questions = self._get_questions_for_sib(sib, tools)
            
    #         if sib_questions:
    #             print(f"    🔄 Optimizing SIB {sib_index} with {len(sib_questions)} questions...")
    #             optimized_tool = self._optimize_sib_tool(cluster_name, validated_tool, sib_questions, sib_index)
                
    #             # Save SIB as markdown file
    #             if hasattr(self, 'output_dir') and self.output_dir:
    #                 self._save_sib_as_markdown(sib, optimized_tool, self.output_dir, cluster_name)
                
    #             processing_result.update({
    #                 "success": True,
    #                 "questions_count": len(sib_questions),
    #                 "step": "completed_with_optimization"
    #             })
                
    #             print(f"    ✅ SIB {sib_index} completed with optimization")
    #             self._save_llm_logs(f"{cluster_name}_sib_{sib_index}")
    #             self._save_final_openai_tools(f"{cluster_name}_sib_{sib_index}", [optimized_tool])
    #             return True, optimized_tool, processing_result
    #         else:
    #             print(f"    ⚠️ No questions found for SIB {sib_index}, using validated tool")
                
    #             processing_result.update({
    #                 "success": True,
    #                 "questions_count": 0,
    #                 "step": "completed_without_optimization"
    #             })
                
    #             self._save_llm_logs(f"{cluster_name}_sib_{sib_index}")
    #             self._save_final_openai_tools(f"{cluster_name}_sib_{sib_index}", [validated_tool])
    #             return True, validated_tool, processing_result
            
    #     except Exception as e:
    #         error_msg = f"Error processing SIB {sib.get('index', 0)}: {str(e)}"
    #         print(f"    ❌ {error_msg}")
            
    #         processing_result.update({
    #             "success": False,
    #             "error": error_msg,
    #             "step": "exception"
    #         })
            
    #         self._save_llm_logs(f"{cluster_name}_sib_{sib_index}")
    #         return False, None, processing_result

    def _generate_sib_tool(self, cluster_name: str, sib: Dict, tools: List[Dict]) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Generate OpenAI tool for a single SIB"""
        try:
            sib_index = sib.get('index', 0)
            
            # Get tools covered by this SIB
            covered_tools = []
            for tool_index in sib.get('covered_tools', []):
                if 0 <= tool_index < len(tools):
                    covered_tools.append(tools[tool_index])
            
            # Prepare tool code from covered tools
            tool_code_parts = []
            for tool in covered_tools:
                tool_name = tool.get('name', 'unknown_tool')
                tool_code = tool.get('tool_code', '')
                if tool_code:
                    clean_code = self._extract_code_from_response(tool_code)
                    tool_code_parts.append(f"# Tool: {tool_name}\n{clean_code}\n")
            
            # combined_tool_code = "\n".join(tool_code_parts) if tool_code_parts else "# No tool code available"
            
            # Format the implementation prompt
            implementation_prompt = CODE_IMPLEMENTATION_PROMPT.format(
                blueprint=sib.get('content', ''),
            )
            
            # Create messages for implementation (save for future refinement)
            implementation_messages = [{"role": "user", "content": implementation_prompt}]
            
            # Use unified JSON generation with retry mechanism
            tool_data = self._generate_tool_json_with_retry(
                messages=implementation_messages,
                step_name=f"sib_{sib_index}_tool_generation",
                max_retries=3,
                additional_context={
                    "sib_index": sib_index,
                    "covered_tools_count": len(covered_tools),
                }
            )


            
            if not tool_data:
                return False, None, f"Failed to generate valid tool JSON for SIB {sib_index} after retries"
            
            # Update implementation_messages with the final conversation
            # The _generate_tool_json_with_retry function will have modified the messages list
            
            # Create enhanced tool data structure
            enhanced_tool_data = {
                "openai_tool": tool_data,
                "sib_info": {
                    "sib_index": sib_index,
                    "sib_content_preview": sib.get('content', '')[:200] + "..." if len(sib.get('content', '')) > 200 else sib.get('content', ''),
                    "covered_tool_indices": sib.get('covered_tools', [])
                },
                "generation_context": {
                    "blueprint": sib.get('content', ''),
                    "tool_code": None,
                    "implementation_messages": implementation_messages,
                    "timestamp": datetime.now().isoformat(),
                    "model_name": self.model_name
                },
                "original_tools": []
            }
            
            # Add original tool information
            for tool_index in sib.get('covered_tools', []):
                if 0 <= tool_index < len(tools):
                    original_tool = tools[tool_index]
                    enhanced_tool_data["original_tools"].append({
                        "tool_index": tool_index,
                        "tool_name": original_tool.get('name', f'tool_{tool_index}'),
                        "tool_description": original_tool.get('description', 'No description'),
                        "original_question": original_tool.get('original_question', ''),
                        "original_answer": original_tool.get('original_answer', '')
                    })
            
            return True, enhanced_tool_data, None
            
        except Exception as e:
            error_msg = f"Error generating tool for SIB {sib.get('index', 0)}: {str(e)}"
            return False, None, error_msg

    def _validate_sib_tool_comprehensive(self, cluster_name: str, sib_tool: Dict, sib_index: int) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Validate a single SIB tool using comprehensive validation"""
        try:
            # Extract tool information
            openai_tool = sib_tool.get('openai_tool', {})
            tool_info = openai_tool.get('tool_info', {})
            function_info = tool_info.get('function', {})
            tool_name = function_info.get('name', f'sib_{sib_index}_tool')
            
            print(f"      🔍 Comprehensive validation for SIB {sib_index}: {tool_name}")
            
            # Prepare tool data for comprehensive validation
            tool_data = {
                'tool_info': tool_info,
                'tool_code': openai_tool.get('tool_code', '')
            }
            
            if not tool_data['tool_code']:
                return False, None, "No tool code found"
            
            # Use comprehensive validation (3-layer validation)
            success, validated_tool_data, validation_message = self._comprehensive_tool_validation(
                tool_data, 
                sib_index
            )
            
            if success and validated_tool_data:
                # Update the SIB tool with validated data
                validated_sib_tool = sib_tool.copy()
                validated_sib_tool['openai_tool'] = {
                    'tool_info': validated_tool_data.get('tool_info', tool_info),
                    'tool_code': validated_tool_data.get('tool_code', tool_data['tool_code'])
                }
                validated_sib_tool['validation_info'] = {
                    "method": "comprehensive_validation",
                    "final_status": "valid",
                    "validation_message": validation_message
                }
                
                print(f"      ✅ SIB {sib_index} passed comprehensive validation")
                return True, validated_sib_tool, None
            else:
                error_msg = f"Comprehensive validation failed: {validation_message}"
                print(f"      ❌ SIB {sib_index} failed comprehensive validation: {error_msg}")
                
                # Return failed tool with validation info
                failed_tool = sib_tool.copy()
                failed_tool['validation_info'] = {
                    "method": "comprehensive_validation",
                    "final_status": "failed",
                    "validation_message": validation_message
                }
                
                return False, failed_tool, error_msg
            
        except Exception as e:
            error_msg = f"Error in comprehensive validation for SIB {sib_index}: {str(e)}"
            print(f"      ❌ {error_msg}")
            return False, None, error_msg


    def _get_questions_for_sib(self, sib: Dict, tools: List[Dict]) -> List[Dict]:
        """Get unique questions for tools covered by this SIB"""
        questions = []
        question_set = set()
        
        for tool_index in sib.get('covered_tools', []):
            if 0 <= tool_index < len(tools):
                tool = tools[tool_index]
                question = tool.get('original_question', '')
                answer = tool.get('original_answer', '')
                
                if question and answer and question not in question_set:
                    questions.append({
                        'question': question,
                        'ground_truth': answer,
                        'tool_index': tool_index
                    })
                    question_set.add(question)
        
        return questions

    def _optimize_sib_tool(self, cluster_name: str, validated_tool: Dict, sib_questions: List[Dict], sib_index: int, max_rounds: int = 2) -> Dict:
        """Optimize a single SIB tool using its associated questions with multi-round optimization"""
        try:
            print(f"      🔄 Running multi-round optimization for SIB {sib_index} (max {max_rounds} rounds)...")
            
            current_tool = validated_tool.copy()
            all_round_results = []
            
            for round_num in range(1, max_rounds + 1):
                print(f"        🎯 Round {round_num}/{max_rounds}")
                
                # Convert current tool to optimizer format
                openai_tool = current_tool.get('openai_tool', {})
                tool_info = openai_tool.get('tool_info', {})
                tool_code = openai_tool.get('tool_code', '')
                
                if not tool_info or not tool_code:
                    print(f"        ⚠️ Missing tool info or code for SIB {sib_index}")
                    break
                
                clean_code = self._extract_code_from_response(tool_code)
                
                optimizer_tools = [{
                    "tool_info": tool_info,
                    "tool_code": clean_code
                }]
                if len(sib_questions)>10:
                    sib_questions = sib_questions[:10]
                # Run optimization for each question in parallel
                print(f"        🔄 Processing {len(sib_questions)} questions in parallel...")
                # Prepare optimization tasks
                optimization_tasks = []
                for question_data in sib_questions:
                    optimization_tasks.append((
                        question_data, 
                        cluster_name, 
                        optimizer_tools, 
                        sib_index,
                        round_num  # Add round number for context
                    ))
                
                # Use map_with_progress for parallel optimization
                optimization_results = map_with_progress(
                    self._optimize_single_question_for_sib,
                    optimization_tasks,
                    num_threads=min(len(sib_questions), 10),
                    pbar=False
                )
                
                # Save optimization results for this round
                if hasattr(self, 'output_dir') and self.output_dir:
                    self._save_optimization_results(optimization_results, self.output_dir, cluster_name, sib_index, round_num)
                
                successful_opts = len([r for r in optimization_results if r.get('success', False)])
                print(f"        📊 Round {round_num} completed: {successful_opts}/{len(optimization_results)} questions successful")
                
                # Store round results
                round_result = {
                    'round': round_num,
                    'optimization_results': optimization_results,
                    'successful_count': successful_opts,
                    'total_count': len(optimization_results)
                }
                all_round_results.append(round_result)
                
                # Check if any results suggest the tool needs patching
                needs_patching = False
                all_suggestions = []
                
                for result in optimization_results:
                    if not isinstance(result, dict):
                        continue
                        
                    if result.get('success', False) and 'final_report' in result:
                        final_report = result['final_report']
                        try:
                            import json
                            if isinstance(final_report, str):
                                report_data = json.loads(final_report)
                            else:
                                report_data = final_report
                            
                            if report_data.get('is_library_helpful') == 'NEED_PATCHING':
                                needs_patching = True
                                suggestions = report_data.get('modification_suggestions', '')
                                if suggestions:
                                    all_suggestions.append(suggestions)
                        except (json.JSONDecodeError, TypeError):
                            if isinstance(final_report, str) and 'NEED_PATCHING' in final_report:
                                needs_patching = True
                
                # If all questions passed, exit early
                if successful_opts == len(optimization_results) and not needs_patching:
                    print(f"        ✅ All questions passed in round {round_num}! Early exit.")
                    break
                
                # If this is not the last round and we need patching, apply suggestions
                if needs_patching and all_suggestions and round_num < max_rounds:
                    print(f"        🔧 Applying optimization suggestions for next round...")
                    
                    updated_tool = self._apply_optimization_suggestions(
                        current_tool, all_suggestions, sib_index
                    )
                    
                    if updated_tool:
                        current_tool = updated_tool
                        print(f"        ✅ Tool updated for round {round_num + 1}")
                    else:
                        print(f"        ⚠️ Failed to apply optimization suggestions, stopping optimization")
                        break
                elif round_num == max_rounds:
                    # Last round, apply suggestions to final tool
                    if needs_patching and all_suggestions:
                        print(f"        🔧 Applying final optimization suggestions...")
                        updated_tool = self._apply_optimization_suggestions(
                            current_tool, all_suggestions, sib_index
                        )
                        if updated_tool:
                            current_tool = updated_tool
                            print(f"        ✅ Final tool updated")
            
            # Add all round results to the final tool
            current_tool['multi_round_optimization'] = {
                'total_rounds': len(all_round_results),
                'max_rounds': max_rounds,
                'round_results': all_round_results,
                'final_round': all_round_results[-1] if all_round_results else None
            }
            
            # Keep the last round's optimization_results for backward compatibility
            if all_round_results:
                current_tool['optimization_results'] = all_round_results[-1]['optimization_results']
            
            total_successful = sum(r['successful_count'] for r in all_round_results)
            total_questions = sum(r['total_count'] for r in all_round_results)
            print(f"        🎉 Multi-round optimization completed: {total_successful}/{total_questions} total successful across {len(all_round_results)} rounds")
            
            return current_tool
            
        except Exception as e:
            print(f"      ❌ Error in SIB {sib_index} multi-round optimization: {e}")
            return validated_tool

    def _optimize_single_question_for_sib(self, task_data: Tuple) -> Dict:
        """Optimize a single question for SIB tool - worker function for parallel processing"""
        # Handle both old and new task data formats for backward compatibility
        if len(task_data) == 5:
            question_data, cluster_name, optimizer_tools, sib_index, round_num = task_data
        else:
            question_data, cluster_name, optimizer_tools, sib_index = task_data
            round_num = 1
            
        question = question_data['question']
        ground_truth = question_data['ground_truth']
        
        try:
            # Create optimizer instance (no longer need python_library parameter)
            optimizer = IterativeLibraryOptimizerAgent(
                stronger_llm_model=self.model_name,
                weaker_llm_model_list=["gpt-4o-mini"],
                max_iterations=1,
                tools=optimizer_tools,
                question=question,
                ground_truth=ground_truth
            )
            
            # Run optimization
            result = optimizer.optimize_library_directly()
            
            # Handle the result which could be messages list or error string
            if isinstance(result, list) and len(result) > 0:
                # Extract the assistant's response from messages
                assistant_response = None
                for message in reversed(result):  # Start from the end to get the latest assistant response
                    if message.get('role') == 'assistant':
                        assistant_response = message.get('content', '')
                        break
                
                response_text = assistant_response or str(result)
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result) if result else ""
            
            # Log the optimization
            self._log_llm_call(
                step_name=f"sib_{sib_index}_optimization_question_round_{round_num}",
                prompt=f"Question: {question}\nGround Truth: {ground_truth}",
                response=response_text,
                success=bool(result),
                error_msg="Empty result from optimization" if not result else None,
                additional_context={
                    "sib_index": sib_index,
                    "round_num": round_num,
                    "question_preview": question[:100],
                    "ground_truth_preview": ground_truth[:100] if ground_truth else "",
                    "tool_index": question_data.get('tool_index', -1)
                }
            )
            
            if result and response_text:
                # Extract final report
                if "<final_report>" in response_text and "</final_report>" in response_text:
                    final_report = response_text.split("<final_report>")[1].split("</final_report>")[0].strip()
                else:
                    final_report = response_text
                
                return {
                    'question': question,
                    'ground_truth': ground_truth,
                    'success': True,
                    'final_report': final_report,
                    'full_messages': result if isinstance(result, list) else [],  # Save complete messages
                    'response_text': response_text  # Save extracted response text
                }
            else:
                return {
                    'question': question,
                    'ground_truth': ground_truth,
                    'success': False,
                    'error': 'Empty result from optimization',
                    'full_messages': result if isinstance(result, list) else [],  # Save even failed messages
                    'response_text': response_text or ''
                }
                
        except Exception as e:
            print(f"        ❌ Optimization failed for question: {str(e)}")
            return {
                'question': question,
                'ground_truth': ground_truth,
                'success': False,
                'error': str(e),
                'full_messages': [],  # Empty messages for exceptions
                'response_text': ''
            }

    def _apply_optimization_suggestions(self, validated_tool: Dict, suggestions: List[str], sib_index: int) -> Optional[Dict]:
        """Apply optimization suggestions using the original implementation messages + refinement suggestions"""
        try:
            print(f"        🔧 Applying optimization suggestions to SIB {sib_index}...")
            
            # Get current tool info and code
            openai_tool = validated_tool.get('openai_tool', {})
            tool_info = openai_tool.get('tool_info', {})
            tool_code = openai_tool.get('tool_code', '')
            
            # Get original generation context with messages
            generation_context = validated_tool.get('generation_context', {})
            original_messages = generation_context.get('implementation_messages', [])
            
            if not tool_info or not tool_code or not original_messages:
                print(f"        ⚠️ Missing tool info, code, or original messages for SIB {sib_index}")
                return None
            
            clean_code = self._extract_code_from_response(tool_code)
            
            # Combine all suggestions
            combined_suggestions = "\n\n".join([f"Suggestion {i+1}: {suggestion}" for i, suggestion in enumerate(suggestions)])
            
            # Create refinement message using structured JSON/CODE operations (SIB-agnostic)
            refinement_message = f"""Act as an expert software developer.
Always use best practices when coding.
Respect and use existing conventions, libraries, and style already present.

Take requests for targeted changes to the supplied code. Do NOT regenerate the whole file.
Provide ONLY structured operations that are idempotent and minimal.

**Current Tool Info:**
{json.dumps(tool_info, indent=2)}

**Current Tool Code:**
```python
{clean_code}
```

**Optimization Feedback:**
{combined_suggestions}

**Your Task:**
Provide two sections:
1) JSON field-level operations for tool_info
2) CODE operations for tool_code (structured, idempotent, diff-free)

--- JSON operations ---
Wrap EXACTLY in <JSON_OPERATIONS> ... </JSON_OPERATIONS> as a JSON list.
Each item has: field, action, and action-specific keys.
Supported JSON actions (non-exhaustive):
- replace: {{"field": "function.description", "action": "replace", "new_value": "..."}}
- add_property: {{"field": "function.parameters.properties", "action": "add_property", "property_name": "...", "property_definition": {{...}} }}
- append: {{"field": "function.parameters.required", "action": "append", "new_value": "..."}}
- remove: {{"field": "function.strict", "action": "remove"}}
- update_description: {{"field": "function.parameters.properties.<param>.description", "action": "update_description", "new_value": "..."}}

Example:
<JSON_OPERATIONS>
[
  {{"field": "function.description", "action": "replace", "new_value": "Updated description..."}}
]
</JSON_OPERATIONS>

--- CODE operations ---
Wrap EXACTLY in <CODE_OPERATIONS> ... </CODE_OPERATIONS> as a JSON list.
General schema: {{"action": "...", "target": {{...}}, ...}}
Supported actions (generic, SIB-agnostic):
- add_import: {{"action":"add_import","target":{{"module": true}},"module":"json","alias":null}}
- upsert_import_from: {{"action":"upsert_import_from","target":{{"module": true}},"module":"math","names":["sqrt","isfinite"],"alias_map":{{}}}}
- upsert_function: {{"action":"upsert_function","target":{{"module": true,"function":"helper"}},"code":"def helper(x):\n    return x\n"}}
- replace_in_function: {{"action":"replace_in_function","target":{{"function":"execute"}},"use_regex":true,"pattern":"return .*","replacement":"return summary"}}
- upsert_method: {{"action":"upsert_method","target":{{"class":"<YourSIBClass>","method":"_human_hint"}},"code":"def _human_hint(self):\n    return '...'\n"}}
- replace_in_method: {{"action":"replace_in_method","target":{{"class":"<YourSIBClass>","method":"_to_float"}},"use_regex":true,"pattern":"raise ValueError.*","replacement":"raise ValueError('invalid input')"}}
- insert_after: {{"action":"insert_after","target":{{"regex":"class\\s+<YourSIBClass>:\\s*$","scope":"module"}},"code":"\n    # note: ...\n"}}
- insert_before: same as insert_after but before anchor
- add_dict_key: {{"action":"add_dict_key","target":{{"regex":"\\\b[A-Za-z_][A-Za-z0-9_]*\\s*=\\s*\\\\\\{{","scope":"method"}},"key":"tags","value":["vertical","drag"],"create_if_missing":true}}
- update_assign: {{"action":"update_assign","target":{{"name":"VERSION","scope":"class","class":"<YourSIBClass>"}},"new_value_code":"\"1.0.1\""}}
- rename_symbol: {{"action":"rename_symbol","target":{{"scope":"class","class":"<YourSIBClass>"}},"old":"old_name","new":"new_name"}}
- upsert_class_attr: {{"action":"upsert_class_attr","target":{{"class":"<YourSIBClass>"}},"attr":"TAGS","value_code":"['vertical','drag']"}}
- add_or_update_docstring: {{"action":"add_or_update_docstring","target":{{"scope":"class","class":"<YourSIBClass>"}},"value":"Multi-line docstring..."}}

Rules:
- Idempotent: if the change already exists, do nothing.
- Minimal: only touch what is necessary.
- Keep public API stable unless explicitly requested.
- Ensure syntactically correct Python after changes.

Example CODE operations:
<CODE_OPERATIONS>
[
  {{"action":"add_import","target":{{"module": true}},"module":"json"}},
  {{"action":"upsert_function","target":{{"module": true,"function":"_example_payload"}},"code":"def _example_payload():\n    return {{\"mass_kg\": 80.0}}\n"}}
]
</CODE_OPERATIONS>

Only output these two tagged sections. No other text.
"""

            # Use the original messages and add the refinement request
            refinement_messages = original_messages.copy()
            refinement_messages.append({"role": "user", "content": refinement_message})

            # Get LLM response with patch instructions
            try:
                response = call_openai_api_multi_turn(
                    messages=refinement_messages,
                    model_name=self.model_name
                )
                
                # Log the improvement attempt
                self._log_llm_call(
                    step_name=f"sib_{sib_index}_improvement",
                    prompt=refinement_message,
                    response=response,
                    success=bool(response),
                    error_msg="Empty response from improvement" if not response else None,
                    additional_context={
                        "sib_index": sib_index,
                        "suggestions_count": len(suggestions),
                        "has_original_messages": bool(original_messages),
                        "original_messages_count": len(original_messages)
                    }
                )
                
                if not response:
                    print(f"        ❌ Empty response from LLM")
                    return None
                
                # Parse JSON operations, CODE operations, and code diff from response
                json_operations, code_operations, code_diff = self._parse_patch_response(response, sib_index)
                
                if not json_operations and not code_operations and not code_diff:
                    print(f"        ⚠️ No valid patches found in response")
                    return None
                
                # Apply patches to create updated tool
                updated_tool_info = self._apply_json_operations(tool_info, json_operations, sib_index)
                # Prefer CODE_OPERATIONS over diff when provided
                if code_operations:
                    updated_tool_code = self._apply_code_operations(clean_code, code_operations, sib_index)
                else:
                    updated_tool_code = self._apply_code_diff(tool_code, code_diff, sib_index)
                
                # Validate the updated tool with basic patching validation first
                if self._validate_patched_tool(updated_tool_info, updated_tool_code, sib_index):
                    # Create updated tool for comprehensive validation
                    temp_updated_tool = {
                        'tool_info': updated_tool_info,
                        'tool_code': updated_tool_code
                    }
                    
                    # Comprehensive validation for optimized tool
                    print(f"        🔍 Comprehensive validation for optimized SIB {sib_index}...")
                    success, validated_optimized_tool, validation_message = self._comprehensive_tool_validation(temp_updated_tool, sib_index)
                    
                    print(f"        📋 Optimized tool validation result: {validation_message}")
                    
                    if validated_optimized_tool:
                        updated_tool_info = validated_optimized_tool.get('tool_info', updated_tool_info)
                        updated_tool_code = validated_optimized_tool.get('tool_code', updated_tool_code)
                    else:
                        print(f"        ⚠️ Using original optimized tool despite validation issues")
                    
                    # Create updated tool
                    updated_tool = validated_tool.copy()
                    updated_tool['openai_tool'] = {
                        'tool_info': updated_tool_info,
                        'tool_code': updated_tool_code
                    }
                    
                    # Update generation context to include refinement info
                    updated_tool['generation_context']['refinement_applied'] = True
                    updated_tool['generation_context']['refinement_timestamp'] = datetime.now().isoformat()
                    updated_tool['generation_context']['refinement_suggestions'] = suggestions
                    updated_tool['generation_context']['applied_json_operations'] = json_operations
                    updated_tool['generation_context']['applied_code_operations'] = code_operations
                    updated_tool['generation_context']['applied_code_diff'] = code_diff
                    # Save the updated conversation history
                    updated_tool['generation_context']['implementation_messages'] = refinement_messages.copy()
                    if response:
                        updated_tool['generation_context']['implementation_messages'].append({"role": "assistant", "content": response})
                    
                    print(f"        ✅ Successfully applied patches with original context")
                    return updated_tool
                else:
                    print(f"        ❌ Patched tool failed validation")
                    return None
                
            except Exception as e:
                print(f"        ❌ Failed to get improvement response: {e}")
                return None
                
        except Exception as e:
            print(f"        ❌ Error applying optimization suggestions: {e}")
            return None
        
        return None

    def _validate_and_refine_code_with_retry(self, tool_code: str, context: Dict, max_attempts: int = 2) -> Tuple[bool, str, Dict]:
        """Validate code (syntax/callability) and refine with minimal changes if needed.
        Returns (ok, refined_code, metadata)."""
        metadata: Dict[str, Any] = {"attempts": []}
        refined = self._extract_code_from_response(tool_code or "")
        for i in range(max_attempts):
            attempt_info = {"attempt": i + 1}
            try:
                quick = self._quick_validate_tool_execution(refined, timeout=3)
                attempt_info["quick"] = quick
                if quick.get("is_valid") and quick.get("can_call"):
                    metadata["attempts"].append(attempt_info)
                    return True, refined, metadata
                # Build error msg
                err = quick.get("error") or "Invalid code"
                # Ask LLM to fix code minimally
                fixed_code, _ = self._fix_code_with_llm(
                    refined,
                    err,
                    context.get("step_name", "unknown"),
                    str(context.get("sib_index", "0")),
                    attempt=i + 1,
                    error_type="code_validation",
                    conversation_history=[]
                )
                if fixed_code and fixed_code.strip():
                    refined = self._extract_code_from_response(fixed_code)
                metadata["attempts"].append(attempt_info)
            except Exception as e:
                attempt_info["exception"] = str(e)
                metadata["attempts"].append(attempt_info)
        return False, refined, metadata

    def _validate_and_refine_schema_with_retry(self, tool_info: Dict, code: str, context: Dict, max_attempts: int = 2) -> Tuple[bool, Dict, Dict]:
        """Validate schema and refine (rewrite schema only) if needed. Returns (ok, refined_tool_info, metadata)."""
        metadata: Dict[str, Any] = {"attempts": []}
        current_tool_info = tool_info or {}
        # Prepare a tool wrapper for validators
        for i in range(max_attempts):
            attempt_info = {"attempt": i + 1}
            try:
                schema_errors = self._validate_openai_schema([{"tool_info": current_tool_info, "tool_code": code}])
                attempt_info["error_count"] = len(schema_errors or [])
                if not schema_errors:
                    metadata["attempts"].append(attempt_info)
                    return True, current_tool_info, metadata
                # Optionally try auto-fix first
                try:
                    print(f"      ⚠️ Found {len(schema_errors)} schema errors, auto-fixing...")
                    fixed_tool = self._fix_schema_errors({"tool_info": current_tool_info, "tool_code": code}, schema_errors)
                    remaining = self._validate_openai_schema([fixed_tool])
                    if not remaining:
                        current_tool_info = fixed_tool.get("tool_info", current_tool_info)
                        metadata["attempts"].append({**attempt_info, "auto_fix": True})
                        return True, current_tool_info, metadata
                except Exception:
                    pass
                # If still errors, ask LLM to rewrite schema only
                func_name = current_tool_info.get("function", {}).get("name", "")
                rewrite_prompt = (
                    "You are fixing an OpenAI function-calling schema. Only rewrite the JSON schema (do not modify code).\n"
                    f"Function schema: {current_tool_info}\n\n"
                    f"Code signature should match execute() in the provided code.\n"
                    "Errors found:\n" + json.dumps(schema_errors, ensure_ascii=False, indent=2) + "\n\n"
                    "Return ONLY the function object JSON wrapped in <json>...</json>."
                )
                rewrite_messages = [{"role": "user", "content": rewrite_prompt}]
                response = call_openai_api_multi_turn(messages=rewrite_messages, model_name=self.model_name)
                self._log_llm_call(
                    step_name=f"{context.get('step_name','schema')}_schema_refine_attempt_{i+1}",
                    prompt=rewrite_prompt,
                    response=response or "",
                    success=bool(response),
                    error_msg="Empty response from LLM" if not response else None,
                    additional_context={"phase": "schema", "attempt": i + 1}
                )
                if response:
                    if "```json" in response:
                        js = response.split("```json", 1)[1]
                        js = js.split("```", 1)[0]
                        json_text = js.strip()
                    elif "<json>" in response and "</json>" in response:
                        json_text = response.split("<json>", 1)[1].split("</json>", 1)[0].strip()
                    else:
                        json_text = response.strip()
                    try:
                        new_info = json.loads(json_text)
                        if isinstance(new_info, dict) and "function" in new_info:
                            current_tool_info = {"type": "function", "function": new_info.get("function", {})}
                    except Exception:
                        pass
                metadata["attempts"].append(attempt_info)
            except Exception as e:
                attempt_info["exception"] = str(e)
                metadata["attempts"].append(attempt_info)
        return False, current_tool_info, metadata

    def _validate_api_and_refine_schema_with_retry(self, tool_info: Dict, code: str, context: Dict, max_attempts: int = 2) -> Tuple[bool, Dict, Dict]:
        """Validate with real API and refine by rewriting schema only if it fails. Returns (ok, refined_tool_info, metadata)."""
        metadata: Dict[str, Any] = {"attempts": []}
        current_tool_info = tool_info or {}
        for i in range(max_attempts):
            attempt_info = {"attempt": i + 1}
            try:
                api_res = self._validate_api_layer({"tool_info": current_tool_info, "tool_code": code}, context.get("sib_index", 0))
                attempt_info["api_success"] = bool(api_res.get("success"))
                if api_res.get("success"):
                    metadata["attempts"].append(attempt_info)
                    return True, current_tool_info, metadata
                # On failure, rewrite schema only based on runtime error
                print(f"      ⚠️ API test failed, rewriting schema...")
                api_error = api_res.get("error", "API test failed")
                func_name = current_tool_info.get("function", {}).get("name", "")
                rewrite_prompt = (
                    "The tool failed during a real API execution test. Fix ONLY the schema to make the tool invocable.\n"
                    f"Function schema: {current_tool_info}\n\n"
                    f"Runtime error: {api_error}\n"
                    "Use code's execute() signature as the source of truth.\n"
                    "Return ONLY the function object JSON wrapped in <json>...</json>."
                )
                messages = [{"role": "user", "content": rewrite_prompt}]
                response = call_openai_api_multi_turn(messages=messages, model_name=self.model_name)
                self._log_llm_call(
                    step_name=f"{context.get('step_name','api')}_api_refine_attempt_{i+1}",
                    prompt=rewrite_prompt,
                    response=response or "",
                    success=bool(response),
                    error_msg="Empty response from LLM" if not response else None,
                    additional_context={"phase": "api", "attempt": i + 1}
                )
                if response:
                    if "```json" in response:
                        js = response.split("```json", 1)[1]
                        js = js.split("```", 1)[0]
                        json_text = js.strip()
                    elif "<json>" in response and "</json>" in response:
                        json_text = response.split("<json>", 1)[1].split("</json>", 1)[0].strip()
                    else:
                        json_text = response.strip()
                    try:
                        new_info = json.loads(json_text)
                        if isinstance(new_info, dict) and "function" in new_info:
                            current_tool_info = {"type": "function", "function": new_info.get("function", {})}
                    except Exception:
                        pass
                metadata["attempts"].append(attempt_info)
            except Exception as e:
                attempt_info["exception"] = str(e)
                metadata["attempts"].append(attempt_info)
        return False, current_tool_info, metadata

    # def _generate_tool_json_with_retry(self, messages: List[Dict], step_name: str, max_retries: int = 3, 
    #                                    additional_context: Dict = None) -> Optional[Dict]:
    #     """
    #     Two-step generation with retry:
    #     1) Generate code via CODE_IMPLEMENTATION_PROMPT (messages provided by caller)
    #     2) Validate/fix code
    #     3) Generate tool_info via OPENAI_TOOL_IMPLEMENTATION_PROMPT using the validated code
    #     4) Validate/fix schema
    #     5) Validate API and, if needed, refine schema
    #     Returns dict with keys: tool_info, tool_code
    #     """
    #     # Step 1: Generate code
    #     code_response: Optional[str] = None
    #     for attempt in range(max_retries):
    #         try:
    #             print(f"        🧱 Code generation attempt {attempt + 1}/{max_retries}...")
    #             code_response = call_openai_api_multi_turn(
    #                 messages=messages,
    #                 model_name=self.model_name
    #             )
    #             self._log_llm_call(
    #                 step_name=f"{step_name}_code_attempt_{attempt + 1}",
    #                 prompt=messages[-1]['content'] if messages else "",
    #                 response=code_response or "",
    #                 success=bool(code_response),
    #                 error_msg="Empty response from LLM" if not code_response else None,
    #                 additional_context={
    #                     **(additional_context or {}),
    #                     "phase": "code",
    #                     "attempt": attempt + 1,
    #                     "max_retries": max_retries
    #                 }
    #             )
    #             if not code_response:
    #                 print(f"        ⚠️ Empty code response on attempt {attempt + 1}/{max_retries}")
    #                 continue
    #             # Extract code from <code> or markdown blocks
    #             tool_code_clean = self._extract_code_from_response(code_response)
    #             if not tool_code_clean.strip():
    #                 raise ValueError("Empty code extracted from response")
    #             print(f"        ✅ Code generated (length={len(tool_code_clean)})")
    #             break
    #         except Exception as e:
    #             print(f"        ❌ Code generation failed on attempt {attempt + 1}/{max_retries}: {e}")
    #             if attempt == max_retries - 1:
    #                 return None
    #             # Provide brief feedback and retry
    #             retry_message = "The previous response did not include a valid <code>...</code> or ```python block with complete code. Please return only the code wrapped in <code>...</code>."
    #             messages.append({"role": "assistant", "content": code_response or ""})
    #             messages.append({"role": "user", "content": retry_message})
    #             continue

    #     # Ensure we have clean code ready
    #     tool_code_clean = self._extract_code_from_response(code_response or "")
    #     if not tool_code_clean.strip():
    #         return None

    #     # Step 2: Validate and refine code (only modify code here)
    #     try:
    #         print(f"        🧪 Layer: Code validation & minimal refine...")
    #         code_ok, refined_code, _code_meta = self._validate_and_refine_code_with_retry(
    #             tool_code_clean,
    #             context={"step_name": step_name},
    #             max_attempts=2
    #         )
    #         tool_code_clean = refined_code if refined_code else tool_code_clean
    #         if not code_ok:
    #             print(f"        ❌ Code did not pass validation after refinement attempts")
    #             return None
    #         else:
    #             print(f"        ✅ Code validation passed")
    #     except Exception as e:
    #         print(f"        ❌ Error during code validation/refinement: {e}")
    #         return None

    #     # Step 2: Generate tool_info from code
    #     import textwrap as _textwrap
    #     tool_info_prompt = OPENAI_TOOL_IMPLEMENTATION_PROMPT.format(code=tool_code_clean)
    #     tool_info_messages = [{"role": "user", "content": tool_info_prompt}]

    #     tool_info: Optional[Dict] = None
    #     for attempt in range(max_retries):
    #         try:
    #             print(f"        🧩 Tool schema generation attempt {attempt + 1}/{max_retries}...")
    #             info_response = call_openai_api_multi_turn(
    #                 messages=tool_info_messages,
    #                 model_name="gpt-5"
    #             )
    #             self._log_llm_call(
    #                 step_name=f"{step_name}_info_attempt_{attempt + 1}",
    #                 prompt=tool_info_messages[-1]['content'] if tool_info_messages else "",
    #                 response=info_response or "",
    #                 success=bool(info_response),
    #                 error_msg="Empty response from LLM" if not info_response else None,
    #                 additional_context={
    #                     **(additional_context or {}),
    #                     "phase": "tool_info",
    #                     "attempt": attempt + 1,
    #                     "max_retries": max_retries
    #                 }
    #             )
    #             if not info_response:
    #                 print(f"        ⚠️ Empty tool_info response on attempt {attempt + 1}/{max_retries}")
    #                 continue

    #             # Parse <json> ... </json> or raw JSON
    #             if "```json" in info_response:
    #                 js = info_response.split("```json", 1)[1]
    #                 js = js.split("```", 1)[0]
    #                 json_text = js.strip()
    #             elif "<json>" in info_response and "</json>" in info_response:
    #                 json_text = info_response.split("<json>", 1)[1].split("</json>", 1)[0].strip()
    #             else:
    #                 json_text = info_response.strip()

    #             import json as _json
    #             parsed = _json.loads(json_text)
    #             if not isinstance(parsed, dict) or "function" not in parsed:
    #                 raise ValueError("Parsed tool_info is not a dict or missing 'function'")
    #             tool_info = {"type": "function", "function": parsed.get("function", {})}
    #             fn = tool_info.get("function", {}).get("name", "")
    #             print(f"        ✅ Tool schema generated for function: {fn or '(unnamed)'}")
    #             break
    #         except Exception as e:
    #             print(f"        ❌ tool_info generation failed on attempt {attempt + 1}/{max_retries}: {e}")
    #             if attempt == max_retries - 1:
    #                 return None
    #             retry_msg = _textwrap.dedent(
    #                 """
    #                 The previous response was not a valid JSON for OpenAI function schema. Return only the JSON wrapped in <json>...</json> with keys: type, function{name, description, parameters}.
    #                 """
    #             ).strip()
    #             tool_info_messages.append({"role": "assistant", "content": info_response or ""})
    #             tool_info_messages.append({"role": "user", "content": retry_msg})
    #             continue

    #     # Step 3: Schema validation and refinement (only modify schema here)
    #     try:
    #         print(f"        🔍 Layer: Schema validation & refine...")
    #         schema_ok, refined_tool_info, _schema_meta = self._validate_and_refine_schema_with_retry(
    #             tool_info=tool_info or {},
    #             code=tool_code_clean,
    #             context={"step_name": step_name},
    #             max_attempts=2
    #         )
    #         tool_info = refined_tool_info if refined_tool_info else tool_info
    #         if schema_ok:
    #             print(f"        ✅ Schema validation passed")
    #         else:
    #             print(f"        ⚠️ Schema validation not fully passed, continuing with best-effort schema")
    #     except Exception as e:
    #         print(f"        ❌ Error during schema validation/refinement: {e}")
    #         # continue with best-effort schema

    #     # Step 4: API execution validation and potential schema refinement
    #     try:
    #         print(f"        🌐 Layer: Real API execution test & schema refine if needed...")
    #         api_ok, api_refined_tool_info, _api_meta = self._validate_api_and_refine_schema_with_retry(
    #             tool_info=tool_info or {},
    #             code=tool_code_clean,
    #             context={"step_name": step_name},
    #             max_attempts=4
    #         )
    #         tool_info = api_refined_tool_info if api_refined_tool_info else tool_info
    #         if api_ok:
    #             print(f"        ✅ Real API execution test passed")
    #         else:
    #             print(f"        ⚠️ Real API execution still failing after schema refine attempts (using best-effort schema)")
    #     except Exception as e:
    #         print(f"        ❌ Error during API validation/refinement: {e}")
    #         # keep current tool_info

    #     # Assemble result. Wrap code back in python fence for downstream compatibility.
    #     tool_code_wrapped = f"```python\n{tool_code_clean}\n```"
    #     return {"tool_info": tool_info, "tool_code": tool_code_wrapped}

    def _generate_tool_json_with_retry(self, messages: List[Dict], step_name: str, max_retries: int = 3, 
                                       additional_context: Dict = None) -> Optional[Dict]:
        """
        Two-step generation with retry:
        1) Generate code via CODE_IMPLEMENTATION_PROMPT (messages provided by caller)
        2) Validate/fix code
        3) Generate tool_info via OPENAI_TOOL_IMPLEMENTATION_PROMPT using the validated code
        4) Validate/fix schema
        5) Validate API and, if needed, refine schema
        Returns dict with keys: tool_info, tool_code
        """
        # Step 1: Generate code
        code_response: Optional[str] = None
        for attempt in range(max_retries):
            try:
                print(f"        🧱 Code generation attempt {attempt + 1}/{max_retries}...")
                code_response = call_openai_api_multi_turn(
                    messages=messages,
                    model_name=self.model_name
                )
                self._log_llm_call(
                    step_name=f"{step_name}_code_attempt_{attempt + 1}",
                    prompt=messages[-1]['content'] if messages else "",
                    response=code_response or "",
                    success=bool(code_response),
                    error_msg="Empty response from LLM" if not code_response else None,
                    additional_context={
                        **(additional_context or {}),
                        "phase": "code",
                        "attempt": attempt + 1,
                        "max_retries": max_retries
                    }
                )
                if not code_response:
                    print(f"        ⚠️ Empty code response on attempt {attempt + 1}/{max_retries}")
                    continue
                # Extract code from <code> or markdown blocks
                tool_code_clean = self._extract_code_from_response(code_response)
                if not tool_code_clean.strip():
                    raise ValueError("Empty code extracted from response")
                print(f"        ✅ Code generated (length={len(tool_code_clean)})")
                break
            except Exception as e:
                print(f"        ❌ Code generation failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    return None
                # Provide brief feedback and retry
                retry_message = "The previous response did not include a valid <code>...</code> or ```python block with complete code. Please return only the code wrapped in <code>...</code>."
                messages.append({"role": "assistant", "content": code_response or ""})
                messages.append({"role": "user", "content": retry_message})
                continue

        # Ensure we have clean code ready
        tool_code_clean = self._extract_code_from_response(code_response or "")
        if not tool_code_clean.strip():
            return None

        # Step 2: Validate and refine code (only modify code here)
        try:
            print(f"        🧪 Layer: Code validation & minimal refine...")
            code_ok, refined_code, _code_meta = self._validate_and_refine_code_with_retry(
                tool_code_clean,
                context={"step_name": step_name},
                max_attempts=2
            )
            tool_code_clean = refined_code if refined_code else tool_code_clean
            if not code_ok:
                print(f"        ❌ Code did not pass validation after refinement attempts")
                return None
            else:
                print(f"        ✅ Code validation passed")
        except Exception as e:
            print(f"        ❌ Error during code validation/refinement: {e}")
            return None

        # Step 2: Generate tool_info from code
        import textwrap as _textwrap
        tool_info_prompt = OPENAI_TOOL_IMPLEMENTATION_PROMPT.format(code=tool_code_clean)
        tool_info_messages = [{"role": "user", "content": tool_info_prompt}]

        tool_info: Optional[Dict] = None
        for attempt in range(max_retries):
            try:
                print(f"        🧩 Tool schema generation attempt {attempt + 1}/{max_retries}...")
                info_response = call_openai_api_multi_turn(
                    messages=tool_info_messages,
                    model_name="gpt-5"
                )
                self._log_llm_call(
                    step_name=f"{step_name}_info_attempt_{attempt + 1}",
                    prompt=tool_info_messages[-1]['content'] if tool_info_messages else "",
                    response=info_response or "",
                    success=bool(info_response),
                    error_msg="Empty response from LLM" if not info_response else None,
                    additional_context={
                        **(additional_context or {}),
                        "phase": "tool_info",
                        "attempt": attempt + 1,
                        "max_retries": max_retries
                    }
                )
                if not info_response:
                    print(f"        ⚠️ Empty tool_info response on attempt {attempt + 1}/{max_retries}")
                    continue

                # Parse <json> ... </json> or raw JSON
                if "```json" in info_response:
                    js = info_response.split("```json", 1)[1]
                    js = js.split("```", 1)[0]
                    json_text = js.strip()
                elif "<json>" in info_response and "</json>" in info_response:
                    json_text = info_response.split("<json>", 1)[1].split("</json>", 1)[0].strip()
                else:
                    json_text = info_response.strip()

                import json as _json
                parsed = _json.loads(json_text)
                if not isinstance(parsed, dict) or "function" not in parsed:
                    raise ValueError("Parsed tool_info is not a dict or missing 'function'")
                tool_info = {"type": "function", "function": parsed.get("function", {})}
                fn = tool_info.get("function", {}).get("name", "")
                print(f"        ✅ Tool schema generated for function: {fn or '(unnamed)'}")
                break
            except Exception as e:
                print(f"        ❌ tool_info generation failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    return None
                retry_msg = _textwrap.dedent(
                    """
                    The previous response was not a valid JSON for OpenAI function schema. Return only the JSON wrapped in <json>...</json> with keys: type, function{name, description, parameters}.
                    """
                ).strip()
                tool_info_messages.append({"role": "assistant", "content": info_response or ""})
                tool_info_messages.append({"role": "user", "content": retry_msg})
                continue

        # Step 3: Schema validation and refinement (only modify schema here)
        try:
            print(f"        🔍 Layer: Schema validation & refine...")
            schema_ok, refined_tool_info, _schema_meta = self._validate_and_refine_schema_with_retry(
                tool_info=tool_info or {},
                code=tool_code_clean,
                context={"step_name": step_name},
                max_attempts=2
            )
            tool_info = refined_tool_info if refined_tool_info else tool_info
            if schema_ok:
                print(f"        ✅ Schema validation passed")
            else:
                print(f"        ⚠️ Schema validation not fully passed, continuing with best-effort schema")
        except Exception as e:
            print(f"        ❌ Error during schema validation/refinement: {e}")
            # continue with best-effort schema

        # Step 4: API execution validation and potential schema refinement
        try:
            print(f"        🌐 Layer: Real API execution test & schema refine if needed...")
            api_ok, api_refined_tool_info, _api_meta = self._validate_api_and_refine_schema_with_retry(
                tool_info=tool_info or {},
                code=tool_code_clean,
                context={"step_name": step_name},
                max_attempts=4
            )
            tool_info = api_refined_tool_info if api_refined_tool_info else tool_info
            if api_ok:
                print(f"        ✅ Real API execution test passed")
            else:
                print(f"        ⚠️ Real API execution still failing after schema refine attempts (using best-effort schema)")
        except Exception as e:
            print(f"        ❌ Error during API validation/refinement: {e}")
            # keep current tool_info

        # Assemble result. Wrap code back in python fence for downstream compatibility.
        tool_code_wrapped = f"```python\n{tool_code_clean}\n```"
        return {"tool_info": tool_info, "tool_code": tool_code_wrapped}


    def _save_sib_as_markdown(self, sib: Dict, sib_tool: Dict, output_dir: Path, cluster_name: str) -> None:
        """Save SIB information as markdown file"""
        try:
            sib_index = sib.get('index', 0)
            sib_dir = output_dir / "sibs"
            sib_dir.mkdir(exist_ok=True)
            
            # Create markdown content
            md_content = f"""# SIB {sib_index} - {cluster_name}

## SIB Blueprint
{sib.get('content', 'No blueprint content')}

## Generated Tool Info
```json
{json.dumps(sib_tool.get('openai_tool', {}).get('tool_info', {}), indent=2)}
```

## Generated Tool Code
```python
{sib_tool.get('openai_tool', {}).get('tool_code', 'No code generated')}
```

## Generation Context
- **Timestamp**: {sib_tool.get('generation_context', {}).get('timestamp', 'Unknown')}
- **Model**: {sib_tool.get('generation_context', {}).get('model_name', 'Unknown')}
- **Covered Tools**: {sib.get('covered_tools', [])}

## Original Tools
"""
            
            # Add original tools information
            for tool_info in sib_tool.get('original_tools', []):
                md_content += f"""
### Tool {tool_info.get('tool_index', 'Unknown')}
- **Name**: {tool_info.get('tool_name', 'Unknown')}
- **Description**: {tool_info.get('tool_description', 'No description')}
- **Original Question**: {tool_info.get('original_question', 'No question')}
- **Original Answer**: {tool_info.get('original_answer', 'No answer')}
"""
            
            # Save to markdown file
            sib_file = sib_dir / f"sib_{sib_index}_{cluster_name}.md"
            sib_file.write_text(md_content)
            print(f"        💾 Saved SIB {sib_index} to {sib_file}")
            
        except Exception as e:
            print(f"        ⚠️ Failed to save SIB {sib.get('index', 0)} as markdown: {e}")

    def _save_optimization_results(self, optimization_results: List[Dict], output_dir: Path, 
                                  cluster_name: str, sib_index: int, round_num: int = 1) -> None:
        """Save optimization results to a file for easy viewing"""
        try:
            opt_dir = output_dir / "optimization_results"
            opt_dir.mkdir(exist_ok=True)
            
            # Create optimization results content
            results_content = f"""# Optimization Results - SIB {sib_index} - Round {round_num}

**Cluster**: {cluster_name}
**SIB Index**: {sib_index}
**Round**: {round_num}
**Timestamp**: {datetime.now().isoformat()}
**Total Questions**: {len(optimization_results)}

## Summary
- **Successful**: {len([r for r in optimization_results if r.get('success', False)])}
- **Failed**: {len([r for r in optimization_results if not r.get('success', False)])}
- **Need Patching**: {len([r for r in optimization_results if r.get('success') and 'NEED_PATCHING' in str(r.get('final_report', ''))])}

## Detailed Results

"""
            
            for i, result in enumerate(optimization_results, 1):
                results_content += f"""
### Question {i}

**Question**: {result.get('question', 'Unknown')}
**Ground Truth**: {result.get('ground_truth', 'Unknown')}
**Success**: {result.get('success', False)}

"""
                if result.get('success'):
                    results_content += f"""**Final Report**:
```
{result.get('final_report', 'No report')}
```
"""
                else:
                    results_content += f"""**Error**: {result.get('error', 'Unknown error')}
"""
                
                # Add complete conversation messages if available
                full_messages = result.get('full_messages', [])
                if full_messages:
                    results_content += f"""\n**Complete Conversation Messages**:
```json
{json.dumps(full_messages, indent=2, ensure_ascii=False)}
```\n"""
                
                # Add extracted response text if different from final report
                response_text = result.get('response_text', '')
                if response_text and response_text != result.get('final_report', ''):
                    results_content += f"""\n**Full Response Text**:
```
{response_text}
```\n"""
                
                results_content += "\n---\n"
            
            # Save to file
            results_file = opt_dir / f"sib_{sib_index}_{cluster_name}_round_{round_num}.md"
            results_file.write_text(results_content)
            print(f"        💾 Saved optimization results to {results_file}")
            
        except Exception as e:
            print(f"        ⚠️ Failed to save optimization results: {e}")

    def _parse_patch_response(self, response: str, sib_index: int) -> Tuple[List[Dict], List[Dict], str]:
        """Parse JSON operations, CODE operations, and improved unified diff from LLM response"""
        try:
            json_operations = []
            code_operations = []
            code_diff = ""
            
            # Extract JSON operations
            if "<JSON_OPERATIONS>" in response and "</JSON_OPERATIONS>" in response:
                json_start = response.find("<JSON_OPERATIONS>") + len("<JSON_OPERATIONS>")
                json_end = response.find("</JSON_OPERATIONS>")
                json_text = response[json_start:json_end].strip()
                
                try:
                    import json
                    json_operations = json.loads(json_text)
                    if not isinstance(json_operations, list):
                        json_operations = []
                    print(f"        📝 Parsed {len(json_operations)} JSON operations")
                except json.JSONDecodeError as e:
                    print(f"        ⚠️ Failed to parse JSON operations: {e}")
                    json_operations = []
            
            # Extract CODE operations
            if "<CODE_OPERATIONS>" in response and "</CODE_OPERATIONS>" in response:
                ops_start = response.find("<CODE_OPERATIONS>") + len("<CODE_OPERATIONS>")
                ops_end = response.find("</CODE_OPERATIONS>")
                ops_text = response[ops_start:ops_end].strip()
                try:
                    import json
                    code_operations = json.loads(ops_text)
                    if not isinstance(code_operations, list):
                        code_operations = []
                    print(f"        📝 Parsed {len(code_operations)} CODE operations")
                except json.JSONDecodeError as e:
                    print(f"        ⚠️ Failed to parse CODE operations: {e}")
                    code_operations = []

            # Extract unified diff from ```diff blocks or <CODE_DIFF> tags
            if "```diff" in response:
                # Extract from markdown diff block
                diff_start = response.find("```diff") + 7
                diff_end = response.find("```", diff_start)
                if diff_end != -1:
                    code_diff = response[diff_start:diff_end].strip()
                else:
                    code_diff = response[diff_start:].strip()
                
                if code_diff:
                    print(f"        🔧 Parsed unified diff from markdown ({len(code_diff.splitlines())} lines)")
                    
            elif "<CODE_DIFF>" in response and "</CODE_DIFF>" in response:
                # Fallback to old format
                diff_start = response.find("<CODE_DIFF>") + len("<CODE_DIFF>")
                diff_end = response.find("</CODE_DIFF>")
                code_diff = response[diff_start:diff_end].strip()
                
                if code_diff:
                    print(f"        🔧 Parsed code diff from tags ({len(code_diff.splitlines())} lines)")
            
            # Validate diff format
            if code_diff:
                if not self._validate_diff_format(code_diff):
                    print(f"        ⚠️ Invalid diff format, attempting to fix...")
                    code_diff = self._fix_diff_format(code_diff)
            
            return json_operations, code_operations, code_diff
            
        except Exception as e:
            print(f"        ❌ Error parsing patch response: {e}")
            return [], [], ""

    def _apply_json_operations(self, original_tool_info: Dict, operations: List[Dict], sib_index: int) -> Dict:
        """Apply JSON field-level operations to tool_info"""
        try:
            import copy
            updated_tool_info = copy.deepcopy(original_tool_info)
            applied_count = 0
            
            for operation in operations:
                try:
                    field = operation.get('field', '')
                    action = operation.get('action', '')
                    
                    if action == 'replace':
                        new_value = operation.get('new_value')
                        if self._set_nested_field(updated_tool_info, field, new_value):
                            applied_count += 1
                            print(f"        ✅ Replaced {field}")
                    
                    elif action == 'add_property':
                        property_name = operation.get('property_name')
                        property_def = operation.get('property_definition')
                        if self._add_json_property(updated_tool_info, field, property_name, property_def):
                            applied_count += 1
                            print(f"        ✅ Added property {property_name}")
                    
                    elif action == 'append':
                        new_value = operation.get('new_value')
                        if self._append_to_json_array(updated_tool_info, field, new_value):
                            applied_count += 1
                            print(f"        ✅ Appended to {field}")
                    
                    elif action == 'remove':
                        if self._remove_nested_field(updated_tool_info, field):
                            applied_count += 1
                            print(f"        ✅ Removed {field}")
                    
                    elif action == 'update_description':
                        new_desc = operation.get('new_value')
                        if self._update_parameter_description(updated_tool_info, field, new_desc):
                            applied_count += 1
                            print(f"        ✅ Updated description for {field}")
                    
                    else:
                        print(f"        ⚠️ Unknown JSON operation: {action}")
                        
                except Exception as e:
                    print(f"        ⚠️ Failed to apply JSON operation {operation}: {e}")
                    continue
            
            print(f"        📊 Applied {applied_count}/{len(operations)} JSON operations")
            return updated_tool_info
            
        except Exception as e:
            print(f"        ❌ Error applying JSON operations: {e}")
            return original_tool_info

    def _apply_code_operations(self, original_code: str, operations: List[Dict], sib_index: int) -> str:
        """Apply structured CODE operations to tool code (idempotent, diff-free)."""
        try:
            import re
            code = original_code
            def upsert_import_from(module: str, names: List[str], alias_map: Optional[Dict[str, str]] = None) -> None:
                nonlocal code
                alias_map = alias_map or {}
                # Ensure from-import line exists and contains required names
                pattern = rf"^\s*from\s+{re.escape(module)}\s+import\s+(.+)$"
                m = re.search(pattern, code, flags=re.MULTILINE)
                if not m:
                    imported = ", ".join([f"{n} as {alias_map.get(n, n)}" if alias_map.get(n) else n for n in names])
                    code = f"from {module} import {imported}\n" + code
                    return
                existing = [s.strip() for s in m.group(1).split(",")]
                need = []
                for n in names:
                    alias = alias_map.get(n)
                    target = f"{n} as {alias}" if alias else n
                    if target not in existing and n not in existing:
                        need.append(target)
                if need:
                    new_line = f"from {module} import {', '.join(existing + need)}"
                    code = code[:m.start()] + new_line + code[m.end():]

            def upsert_function(func_name: str, func_code: str) -> None:
                nonlocal code
                pattern = rf"^\s*def\s+{re.escape(func_name)}\(.*\):[\s\S]*?(?=^\s*def\s+|^\s*class\s+|\Z)"
                m = re.search(pattern, code, flags=re.MULTILINE)
                if m:
                    code = code[:m.start()] + func_code.rstrip() + "\n" + code[m.end():]
                else:
                    code = code.rstrip() + "\n\n" + func_code.rstrip() + "\n"

            def replace_in_function(func_name: str, pattern: str, replacement: str, use_regex: bool) -> None:
                nonlocal code
                block_pat = rf"(^\s*def\s+{re.escape(func_name)}\(.*\):[\s\S]*?)(?=^\s*def\s+|^\s*class\s+|\Z)"
                m = re.search(block_pat, code, flags=re.MULTILINE)
                if not m:
                    return
                block = m.group(1)
                new_block = re.sub(pattern, replacement, block) if use_regex else block.replace(pattern, replacement)
                code = code[:m.start()] + new_block + code[m.end():]

            def add_or_update_docstring(scope: str, class_name: Optional[str], value: str) -> None:
                nonlocal code
                if scope == 'class' and class_name:
                    class_pat = rf"class\s+{re.escape(class_name)}\s*:\s*\n([\s\S]*?)(?=^\s*class\s+|\Z)"
                    m = re.search(class_pat, code, flags=re.MULTILINE)
                    if not m:
                        return
                    body = m.group(1)
                    # If body starts with a docstring, replace; else insert at top
                    doc_pat = r"^\s*\"\"\"[\s\S]*?\"\"\"\s*\n"
                    if re.match(doc_pat, body):
                        new_body = re.sub(doc_pat, f"    \"\"\"{value}\"\"\"\n", body, count=1)
                    else:
                        new_body = "    \"\"\"" + value + "\"\"\"\n" + body
                    code = code.replace(body, new_body, 1)

            def ensure_import(mod: str, alias: Optional[str] = None) -> None:
                nonlocal code
                pattern = rf"^\s*import\s+{re.escape(mod)}(\s+as\s+{re.escape(alias)})?\s*$|^\s*from\s+{re.escape(mod)}\s+import\s+.*$"
                if not re.search(pattern, code, flags=re.MULTILINE):
                    code = f"import {mod}{f' as {alias}' if alias else ''}\n" + code

            def upsert_method(class_name: str, method_name: str, method_code: str) -> None:
                nonlocal code
                class_pattern = rf"class\s+{re.escape(class_name)}\s*:\s*\n([\s\S]*?)\n(?=class\s+|def\s+|$)"
                m = re.search(class_pattern, code)
                if not m:
                    return
                class_body = m.group(1)
                method_pattern = rf"\n\s*def\s+{re.escape(method_name)}\("
                if re.search(method_pattern, class_body):
                    # replace existing method
                    method_block_pattern = rf"\n\s*def\s+{re.escape(method_name)}\([\s\S]*?\n\s*(?=def\s+|class\s+|$)"
                    new_class_body = re.sub(method_block_pattern, "\n    " + method_code.rstrip() + "\n", class_body)
                else:
                    # append method at end of class
                    new_class_body = class_body.rstrip() + "\n\n    " + method_code.rstrip() + "\n"
                code = code.replace(class_body, new_class_body)

            def replace_in_method(class_name: str, method_name: str, pattern: str, replacement: str, use_regex: bool) -> None:
                nonlocal code
                class_pattern = rf"class\s+{re.escape(class_name)}\s*:\s*\n([\s\S]*?)\n(?=class\s+|def\s+|$)"
                m = re.search(class_pattern, code)
                if not m:
                    return
                class_body = m.group(1)
                method_block_pattern = rf"(\n\s*def\s+{re.escape(method_name)}\([\s\S]*?\n\s*)(?=def\s+|class\s+|$)"
                mb = re.search(method_block_pattern, class_body)
                if not mb:
                    return
                method_block = mb.group(0)
                if use_regex:
                    new_method_block = re.sub(pattern, replacement, method_block)
                else:
                    new_method_block = method_block.replace(pattern, replacement)
                code = code.replace(method_block, new_method_block)

            def insert_relative(anchor_regex: str, insert_code: str, before: bool, scope: str) -> None:
                nonlocal code
                flags = re.MULTILINE
                m = re.search(anchor_regex, code, flags)
                if not m:
                    return
                idx = m.start() if before else m.end()
                code = code[:idx] + ("\n" + insert_code.rstrip() + "\n") + code[idx:]

            def add_dict_key(anchor_regex: str, key: str, value_json: Any) -> None:
                nonlocal code
                m = re.search(anchor_regex, code)
                if not m:
                    return
                # naive insertion: find the closing '}' from anchor
                start = m.end()
                end = code.find('}', start)
                if end == -1:
                    return
                snippet = code[start:end]
                if f'"{key}":' in snippet:
                    return
                insertion = f"\n            \"{key}\": {json.dumps(value_json)} ,"
                code = code[:start] + insertion + code[start:]

            def update_assign(name: str, new_value_code: str, class_name: Optional[str]) -> None:
                nonlocal code
                if class_name:
                    class_pattern = rf"class\s+{re.escape(class_name)}\s*:\s*\n([\s\S]*?)\n(?=class\s+|def\s+|$)"
                    m = re.search(class_pattern, code)
                    if not m:
                        return
                    class_body = m.group(1)
                    assign_pattern = rf"\n\s*{re.escape(name)}\s*=.*\n"
                    if re.search(assign_pattern, class_body):
                        new_body = re.sub(assign_pattern, f"\n    {name} = {new_value_code}\n", class_body)
                        code = code.replace(class_body, new_body)
                else:
                    assign_pattern = rf"^\s*{re.escape(name)}\s*=.*$"
                    if re.search(assign_pattern, code, flags=re.MULTILINE):
                        code = re.sub(assign_pattern, f"{name} = {new_value_code}", code, flags=re.MULTILINE)

            def rename_symbol(class_name: Optional[str], old: str, new: str) -> None:
                nonlocal code
                # simple safe rename within class scope
                if class_name:
                    class_pattern = rf"class\s+{re.escape(class_name)}\s*:\s*\n([\s\S]*?)\n(?=class\s+|def\s+|$)"
                    m = re.search(class_pattern, code)
                    if not m:
                        return
                    class_body = m.group(1)
                    new_body = re.sub(rf"\b{re.escape(old)}\b", new, class_body)
                    code = code.replace(class_body, new_body)
                else:
                    code = re.sub(rf"\b{re.escape(old)}\b", new, code)

            def upsert_class_attr(class_name: str, attr: str, value_code: str) -> None:
                nonlocal code
                class_pattern = rf"class\s+{re.escape(class_name)}\s*:\s*\n([\s\S]*?)\n(?=class\s+|def\s+|$)"
                m = re.search(class_pattern, code)
                if not m:
                    return
                class_body = m.group(1)
                assign_pattern = rf"\n\s*{re.escape(attr)}\s*=.*\n"
                if re.search(assign_pattern, class_body):
                    new_body = re.sub(assign_pattern, f"\n    {attr} = {value_code}\n", class_body)
                else:
                    new_body = class_body.rstrip() + f"\n\n    {attr} = {value_code}\n"
                code = code.replace(class_body, new_body)

            for op in operations or []:
                try:
                    action = op.get('action')
                    target = op.get('target', {})
                    if action == 'add_import':
                        ensure_import(op.get('module', ''), op.get('alias'))
                    elif action == 'upsert_import_from':
                        upsert_import_from(op.get('module', ''), op.get('names', []), op.get('alias_map'))
                    elif action == 'upsert_function':
                        upsert_function(target.get('function', ''), op.get('code', ''))
                    elif action == 'replace_in_function':
                        replace_in_function(target.get('function', ''), op.get('pattern', ''), op.get('replacement', ''), bool(op.get('use_regex', False)))
                    elif action == 'upsert_method':
                        upsert_method(target.get('class', ''), target.get('method', ''), op.get('code', ''))
                    elif action == 'replace_in_method':
                        replace_in_method(target.get('class', ''), target.get('method', ''), op.get('pattern', ''), op.get('replacement', ''), bool(op.get('use_regex', False)))
                    elif action == 'insert_after':
                        insert_relative(target.get('regex', ''), op.get('code', ''), before=False, scope=target.get('scope', 'module'))
                    elif action == 'insert_before':
                        insert_relative(target.get('regex', ''), op.get('code', ''), before=True, scope=target.get('scope', 'module'))
                    elif action == 'add_dict_key':
                        add_dict_key(target.get('regex', ''), op.get('key', ''), op.get('value'))
                    elif action == 'update_assign':
                        update_assign(target.get('name', ''), op.get('new_value_code', ''), target.get('class'))
                    elif action == 'rename_symbol':
                        rename_symbol(target.get('class'), op.get('old', ''), op.get('new', ''))
                    elif action == 'upsert_class_attr':
                        upsert_class_attr(target.get('class', ''), op.get('attr', ''), op.get('value_code', ''))
                    elif action == 'add_or_update_docstring':
                        add_or_update_docstring(target.get('scope', 'class'), target.get('class'), op.get('value', ''))
                    else:
                        print(f"        ⚠️ Unknown CODE operation: {action}")
                except Exception as e:
                    print(f"        ⚠️ Failed to apply CODE operation {op}: {e}")

            # Validate syntax after operations
            syntax_result = validate_code_syntax(code, timeout=5)
            if not syntax_result["is_valid"]:
                print(f"        ❌ CODE operations produced syntax error: {syntax_result['error_message']}")
                return original_code
            print(f"        ✅ Applied CODE operations successfully")
            return code
        except Exception as e:
            print(f"        ❌ Error applying CODE operations: {e}")
            return original_code

    def _validate_diff_format(self, diff_text: str) -> bool:
        """Validate that diff text follows proper unified diff format"""
        lines = diff_text.strip().split('\n')
        
        # Check for proper header
        if len(lines) < 2:
            return False
        
        # Should start with --- and +++
        if not (lines[0].startswith('---') and lines[1].startswith('+++')):
            return False
        
        # Should have at least one hunk with @@ marker
        has_hunk = any(line.startswith('@@') for line in lines)
        return has_hunk
    
    def _fix_diff_format(self, diff_text: str) -> str:
        """Fix common diff format issues"""
        lines = diff_text.strip().split('\n')
        
        # Add proper headers if missing
        if not lines[0].startswith('---'):
            lines.insert(0, '--- tool_code')
        if len(lines) < 2 or not lines[1].startswith('+++'):
            lines.insert(1, '+++ tool_code')
        
        # Add hunk header if missing
        has_hunk = any(line.startswith('@@') for line in lines)
        if not has_hunk:
            # Find first change line
            for i, line in enumerate(lines):
                if line.startswith(('+', '-')) and i > 1:
                    lines.insert(i, '@@ ... @@')
                    break
        
        return '\n'.join(lines)

    def _apply_code_diff(self, original_code: str, code_diff: str, sib_index: int) -> str:
        """Apply code diff patch to tool code with enhanced validation"""
        try:
            if not code_diff or not code_diff.strip():
                print(f"        ℹ️ No code diff to apply")
                return original_code
            
            # Validate and fix diff format
            if not self._validate_diff_format(code_diff):
                print(f"        🔧 Fixing diff format...")
                code_diff = self._fix_diff_format(code_diff)
            
            # Use existing apply_patch function
            updated_code = apply_patch(original_code, code_diff)
            
            if updated_code is None:
                print(f"        ❌ Failed to apply patch - trying LLM fallback")
                # Fallback: use LLM to apply the patch
                from utils import apply_patch_with_llm
                updated_code = apply_patch_with_llm(original_code, code_diff, model_name="gpt-4o-mini")
                
                if updated_code and updated_code != original_code:
                    print(f"        ✅ Applied patch with LLM fallback")
                    
                    # Validate the result
                    syntax_result = validate_code_syntax(updated_code, timeout=5)
                    if syntax_result["is_valid"]:
                        return updated_code
                    else:
                        print(f"        ❌ LLM patched code has syntax errors: {syntax_result['error_message']}")
                        return original_code
                else:
                    print(f"        ❌ LLM fallback also failed")
                    return original_code
            
            elif updated_code != original_code:
                print(f"        ✅ Applied code diff successfully")
                
                # Validate the patched code
                syntax_result = validate_code_syntax(updated_code, timeout=5)
                if syntax_result["is_valid"]:
                    return updated_code
                else:
                    print(f"        ❌ Patched code has syntax errors: {syntax_result['error_message']}")
                    return original_code
            else:
                print(f"        ⚠️ Code diff application had no effect")
                return original_code
                
        except Exception as e:
            print(f"        ❌ Error applying code diff: {e}")
            return original_code

    def _validate_patched_tool(self, tool_info: Dict, tool_code: str, sib_index: int) -> bool:
        """Validate the patched tool for correctness with enhanced checking"""
        try:
            # Basic structure validation
            if not isinstance(tool_info, dict):
                print(f"        ❌ tool_info is not a dictionary")
                return False
            
            function_info = tool_info.get('function', {})
            if not function_info:
                print(f"        ❌ Missing function info")
                return False
            
            # Check required fields
            required_fields = ['name', 'description', 'parameters']
            for field in required_fields:
                if field not in function_info:
                    print(f"        ❌ Missing required field: {field}")
                    return False
            
            # Enhanced code validation using quick execution test
            clean_code = self._extract_code_from_response(tool_code)
            
            # Use quick validation (direct execution - faster and more accurate)
            quick_result = self._quick_validate_tool_execution(clean_code, timeout=3)
            
            if not quick_result["is_valid"]:
                print(f"        ❌ Quick validation failed: {quick_result['error']}")
                return False
            
            print(f"        ✅ Patched tool validation passed - execute function found with params: {quick_result.get('function_params', [])}")
            return True
            
        except Exception as e:
            print(f"        ❌ Error validating patched tool: {e}")
            return False

    def _set_nested_field(self, obj: Dict, field_path: str, value: Any) -> bool:
        """Set a nested field value using dot notation"""
        try:
            keys = field_path.split('.')
            current = obj
            
            # Navigate to the parent of the target field
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the final value
            current[keys[-1]] = value
            return True
            
        except Exception as e:
            print(f"        ⚠️ Failed to set {field_path}: {e}")
            return False

    def _add_json_property(self, obj: Dict, field_path: str, property_name: str, property_def: Dict) -> bool:
        """Add a property to a JSON object"""
        try:
            keys = field_path.split('.')
            current = obj
            
            # Navigate to the target object
            for key in keys:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Add the property
            current[property_name] = property_def
            return True
            
        except Exception as e:
            print(f"        ⚠️ Failed to add property {property_name}: {e}")
            return False

    def _append_to_json_array(self, obj: Dict, field_path: str, value: Any) -> bool:
        """Append a value to a JSON array"""
        try:
            keys = field_path.split('.')
            current = obj
            
            # Navigate to the target array
            for key in keys:
                if key not in current:
                    if key == keys[-1]:
                        current[key] = []
                    else:
                        current[key] = {}
                current = current[key]
            
            # Append to array
            if isinstance(current, list):
                current.append(value)
                return True
            else:
                print(f"        ⚠️ Field {field_path} is not an array")
                return False
                
        except Exception as e:
            print(f"        ⚠️ Failed to append to {field_path}: {e}")
            return False

    def _remove_nested_field(self, obj: Dict, field_path: str) -> bool:
        """Remove a nested field"""
        try:
            keys = field_path.split('.')
            current = obj
            
            # Navigate to the parent
            for key in keys[:-1]:
                if key not in current:
                    return False
                current = current[key]
            
            # Remove the field
            if keys[-1] in current:
                del current[keys[-1]]
                return True
            return False
            
        except Exception as e:
            print(f"        ⚠️ Failed to remove {field_path}: {e}")
            return False

    def _update_parameter_description(self, obj: Dict, param_name: str, new_desc: str) -> bool:
        """Update description of a specific parameter"""
        try:
            params = obj.get('function', {}).get('parameters', {}).get('properties', {})
            if param_name in params:
                params[param_name]['description'] = new_desc
                return True
            return False
            
        except Exception as e:
            print(f"        ⚠️ Failed to update description for {param_name}: {e}")
            return False

    def design_blueprint_only(self, cluster_name: str, tools: List[Dict], model_name: str = None) -> Tuple[bool, List[Dict], Optional[str]]:
        """Public wrapper: only design blueprint and return SIBs."""
        return self._design_blueprint(cluster_name, tools, model_name)

    def process_single_sib(self, cluster_name: str, tools: List[Dict], sib: Dict, output_dir: Path) -> Tuple[bool, Optional[Dict], Dict]:
        """Public wrapper: process one SIB end-to-end (generation + validation + optimization)."""
        self.output_dir = output_dir
        return self._process_single_sib_complete((sib, tools, cluster_name))

    def process_single_cluster(self, cluster_name: str, tools: List[Dict], output_dir: Path) -> ToolAggregationResult:
        """Process a single cluster and return results."""
        self.output_dir = output_dir
        
        # Reset LLM logs for this cluster
        self.llm_call_logs = []
        
        print(f"🔄 Processing cluster: {cluster_name} ({len(tools)} tools)")
        print(f"📁 Output directory: {output_dir}")
        
        result = ToolAggregationResult(
            cluster_name=cluster_name,
            total_tools=len(tools)
        )
        
        try:
            # Step 1: Blueprint design
            print(f"📋 Step 1: Blueprint Design")
            success, sibs, error_msg = self._design_blueprint(cluster_name, tools)
            
            if not success:
                result.success = False
                result.error_message = f"Blueprint design failed: {error_msg}"
                return result
            
            result.steps_completed.append("blueprint_design")


            
            # Step 2-4: Process each SIB individually (implementation + validation + optimization)
            print(f"💻 Step 2-4: Processing each SIB individually in parallel")
            
            # Prepare tasks for parallel SIB processing
            sib_processing_tasks = []
            for sib in sibs:
                sib_processing_tasks.append((
                    sib,
                    tools,
                    cluster_name
                ))
            
            print(f"  🔧 Processing {len(sib_processing_tasks)} SIBs in parallel...")
            
            # Use map_with_progress for parallel SIB processing
            results = map_with_progress(
                self._process_single_sib_complete,
                sib_processing_tasks,
                num_threads=min(len(sib_processing_tasks), 10),
                pbar=False
            )
            
            # Process results
            final_tools = []
            sib_processing_results = []
            
            for success, final_tool, processing_result in results:
                if success and final_tool:
                    final_tools.append(final_tool)
                    sib_processing_results.append(processing_result)
                else:
                    sib_processing_results.append(processing_result)
            
            result.steps_completed.append("sib_processing_completed")
            result.openai_tools = final_tools
            
            # Report processing results
            successful_sibs = len([r for r in sib_processing_results if r['success']])
            print(f"  📊 SIB processing summary: {successful_sibs}/{len(sibs)} SIBs processed successfully")
            
            # Create a summary of the complete processing results
            implementation_summary = f"# Complete SIB Processing Results for {cluster_name}\n\n"
            implementation_summary += f"Generated {len(sibs)} Static Inference Blocks (SIBs)\n"
            implementation_summary += f"Successfully processed {successful_sibs}/{len(sibs)} SIBs\n"
            implementation_summary += f"Final tools: {len(final_tools)}\n\n"
            
            for i, enhanced_tool in enumerate(final_tools):
                # Extract OpenAI tool information
                openai_tool = enhanced_tool.get('openai_tool', {})
                tool_info = openai_tool.get('tool_info', {})
                function_info = tool_info.get('function', {})
                tool_name = function_info.get('name', f'tool_{i+1}')
                description = function_info.get('description', 'No description')
                
                # Extract SIB information
                sib_info = enhanced_tool.get('sib_info', {})
                sib_index = sib_info.get('sib_index', i)
                covered_indices = sib_info.get('covered_tool_indices', [])
                
                # Extract validation information
                validation_info = enhanced_tool.get('validation_info', {})
                validation_status = validation_info.get('final_status', 'unknown')
                validation_attempts = validation_info.get('attempts', 0)
                fix_history = validation_info.get('fix_history', [])
                
                # Extract optimization information
                optimization_results = enhanced_tool.get('optimization_results', [])
                
                implementation_summary += f"## Final Tool {i+1}: {tool_name}\n"
                implementation_summary += f"**Description:** {description}\n"
                implementation_summary += f"**Source SIB:** SIB {sib_index}\n"
                implementation_summary += f"**Covers Original Tools:** {covered_indices}\n"
                implementation_summary += f"**Validation Status:** {validation_status.upper()}\n"
                
                if validation_attempts > 1:
                    implementation_summary += f"**Validation Attempts:** {validation_attempts}\n"
                    if fix_history:
                        implementation_summary += f"**Fix History:** {'; '.join(fix_history)}\n"
                
                if optimization_results:
                    successful_opts = len([r for r in optimization_results if r.get('success', False)])
                    implementation_summary += f"**Optimization:** {successful_opts}/{len(optimization_results)} questions successful\n"
                    
                    if successful_opts > 0:
                        implementation_summary += f"**Optimized Questions:**\n"
                        for opt_result in optimization_results:
                            if opt_result.get('success', False):
                                question = opt_result['question'][:100] + "..." if len(opt_result['question']) > 100 else opt_result['question']
                                implementation_summary += f"  - {question}\n"
                
                implementation_summary += "\n"
            
            result.success = True
            result.steps_completed.append("implementation_and_validation_completed")
            result.final_code = implementation_summary
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            print(f"❌ Error processing {cluster_name}: {e}")
        
        # Always save LLM logs at the end
        self._save_llm_logs(cluster_name)
        
        # Save final OpenAI tools and questions
        if result.openai_tools:
            self._save_final_openai_tools(cluster_name, result.openai_tools)
            self._save_all_questions(cluster_name, result.openai_tools)
            self._save_solver_performance(cluster_name, result.openai_tools)
        
        return result

    def _save_final_openai_tools(self, cluster_name: str, final_tools: List[Dict]) -> None:
        """Save final OpenAI tools in standard format"""
        if not final_tools or not self.output_dir:
            return
            
        try:
            # Extract just the OpenAI tools without additional metadata
            openai_tools_only = []
            
            for enhanced_tool in final_tools:
                openai_tool = enhanced_tool.get('openai_tool', {})
                if openai_tool:
                    openai_tools_only.append(openai_tool)
            
            # Save OpenAI tools
            openai_tools_file = self.output_dir / f"{cluster_name}_final_openai_tools.json"
            
            openai_tools_data = {
                "cluster_name": cluster_name,
                "total_tools": len(openai_tools_only),
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "tools": openai_tools_only
            }
            
            with open(openai_tools_file, 'w', encoding='utf-8') as f:
                json.dump(openai_tools_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {len(openai_tools_only)} final OpenAI tools to {openai_tools_file}")
            
            # Also save the complete enhanced tools with all metadata
            enhanced_tools_file = self.output_dir / f"{cluster_name}_enhanced_tools_complete.json"
            
            enhanced_tools_data = {
                "cluster_name": cluster_name,
                "total_tools": len(final_tools),
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "enhanced_tools": final_tools
            }
            
            with open(enhanced_tools_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_tools_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {len(final_tools)} enhanced tools with metadata to {enhanced_tools_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save final OpenAI tools: {e}")

    def _save_all_questions(self, cluster_name: str, final_tools: List[Dict]) -> None:
        """Save all questions and their optimization results"""
        if not final_tools or not self.output_dir:
            return
            
        try:
            all_questions = []
            
            for enhanced_tool in final_tools:
                sib_info = enhanced_tool.get('sib_info', {})
                sib_index = sib_info.get('sib_index', 0)
                
                # Get original questions from original_tools
                original_tools = enhanced_tool.get('original_tools', [])
                for orig_tool in original_tools:
                    question = orig_tool.get('original_question', '')
                    answer = orig_tool.get('original_answer', '')
                    
                    if question and answer:
                        question_entry = {
                            "tool_index": orig_tool.get('tool_index', -1),
                            "tool_name": orig_tool.get('tool_name', ''),
                            "sib_index": sib_index,
                            "question": question,
                            "ground_truth": answer,
                            "optimization_result": None
                        }
                        
                        # Find corresponding optimization result
                        optimization_results = enhanced_tool.get('optimization_results', [])
                        for opt_result in optimization_results:
                            if opt_result.get('question') == question:
                                question_entry["optimization_result"] = {
                                    "success": opt_result.get('success', False),
                                    "final_report": opt_result.get('final_report', ''),
                                    "error": opt_result.get('error', '')
                                }
                                break
                        
                        all_questions.append(question_entry)
            
            # Save questions data
            questions_file = self.output_dir / f"{cluster_name}_all_questions_and_optimization.json"
            
            questions_data = {
                "cluster_name": cluster_name,
                "total_questions": len(all_questions),
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "questions": all_questions,
                "statistics": {
                    "total_questions": len(all_questions),
                    "optimized_questions": len([q for q in all_questions if q.get('optimization_result', {}).get('success', False)]),
                    "failed_optimizations": len([q for q in all_questions if q.get('optimization_result') and not q.get('optimization_result', {}).get('success', False)]),
                    "unoptimized_questions": len([q for q in all_questions if not q.get('optimization_result')])
                }
            }
            
            with open(questions_file, 'w', encoding='utf-8') as f:
                json.dump(questions_data, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Saved {len(all_questions)} questions and optimization results to {questions_file}")
            
            # Print statistics
            stats = questions_data["statistics"]
            print(f"    📊 Questions statistics:")
            print(f"      Total: {stats['total_questions']}")
            print(f"      Optimized: {stats['optimized_questions']}")
            print(f"      Failed: {stats['failed_optimizations']}")
            print(f"      Unoptimized: {stats['unoptimized_questions']}")
            
        except Exception as e:
            print(f"⚠️ Failed to save questions data: {e}")

    def _save_solver_performance(self, cluster_name: str, final_tools: List[Dict]) -> None:
        """Save solver LLM performance analysis"""
        if not final_tools or not self.output_dir:
            return
            
        try:
            performance_data = {
                "cluster_name": cluster_name,
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "solver_performance": {
                    "total_sibs": len(final_tools),
                    "sib_details": [],
                    "overall_statistics": {}
                }
            }
            
            total_questions = 0
            total_successful = 0
            total_failed = 0
            total_unoptimized = 0
            
            # Analyze each SIB's performance
            for enhanced_tool in final_tools:
                sib_info = enhanced_tool.get('sib_info', {})
                sib_index = sib_info.get('sib_index', 0)
                
                openai_tool = enhanced_tool.get('openai_tool', {})
                tool_info = openai_tool.get('tool_info', {})
                function_info = tool_info.get('function', {})
                tool_name = function_info.get('name', f'sib_{sib_index}_tool')
                
                optimization_results = enhanced_tool.get('optimization_results', [])
                
                # Calculate performance metrics for this SIB
                sib_total = len(optimization_results)
                sib_successful = len([r for r in optimization_results if r.get('success', False)])
                sib_failed = len([r for r in optimization_results if r.get('success') == False])
                
                # Get original tools count for this SIB
                original_tools = enhanced_tool.get('original_tools', [])
                sib_unoptimized = len(original_tools) - sib_total
                
                sib_performance = {
                    "sib_index": sib_index,
                    "tool_name": tool_name,
                    "covered_tool_indices": sib_info.get('covered_tool_indices', []),
                    "total_original_tools": len(original_tools),
                    "questions_optimized": sib_total,
                    "questions_successful": sib_successful,
                    "questions_failed": sib_failed,
                    "questions_unoptimized": sib_unoptimized,
                    "success_rate": sib_successful / sib_total if sib_total > 0 else 0,
                    "optimization_details": []
                }
                
                # Add detailed optimization results
                for opt_result in optimization_results:
                    question = opt_result.get('question', '')
                    success = opt_result.get('success', False)
                    final_report = opt_result.get('final_report', '')
                    error = opt_result.get('error', '')
                    
                    # Analyze the final report to extract performance indicators
                    performance_indicators = self._analyze_optimization_report(final_report, success)
                    
                    optimization_detail = {
                        "question_preview": question[:150] + "..." if len(question) > 150 else question,
                        "success": success,
                        "error": error,
                        "performance_indicators": performance_indicators,
                        "report_length": len(final_report) if final_report else 0
                    }
                    
                    sib_performance["optimization_details"].append(optimization_detail)
                
                performance_data["solver_performance"]["sib_details"].append(sib_performance)
                
                # Update totals
                total_questions += sib_total
                total_successful += sib_successful
                total_failed += sib_failed
                total_unoptimized += sib_unoptimized
            
            # Calculate overall statistics
            performance_data["solver_performance"]["overall_statistics"] = {
                "total_questions": total_questions,
                "successful_optimizations": total_successful,
                "failed_optimizations": total_failed,
                "unoptimized_questions": total_unoptimized,
                "overall_success_rate": total_successful / total_questions if total_questions > 0 else 0,
                "optimization_coverage": total_questions / (total_questions + total_unoptimized) if (total_questions + total_unoptimized) > 0 else 0
            }
            
            # Save performance data
            performance_file = self.output_dir / f"{cluster_name}_solver_performance.json"
            
            with open(performance_file, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, indent=2, ensure_ascii=False)
            
            print(f"📈 Saved solver performance analysis to {performance_file}")
            
            # Print performance summary
            stats = performance_data["solver_performance"]["overall_statistics"]
            print(f"    🎯 Solver Performance Summary:")
            print(f"      Questions optimized: {stats['total_questions']}")
            print(f"      Success rate: {stats['successful_optimizations']}/{stats['total_questions']} ({stats['overall_success_rate']:.1%})")
            print(f"      Optimization coverage: {stats['optimization_coverage']:.1%}")
            
        except Exception as e:
            print(f"⚠️ Failed to save solver performance: {e}")

    def _analyze_optimization_report(self, final_report: str, success: bool) -> Dict[str, Any]:
        """Analyze optimization report to extract performance indicators"""
        indicators = {
            "success": success,
            "report_available": bool(final_report),
            "contains_pass": False,
            "contains_need_patching": False,
            "contains_error": False,
            "report_length": len(final_report) if final_report else 0
        }
        
        if final_report:
            report_lower = final_report.lower()
            
            # Check for key performance indicators
            indicators["contains_pass"] = any(keyword in report_lower for keyword in [
                "pass", "successful", "correct", "accurate", "helpful"
            ])
            
            indicators["contains_need_patching"] = any(keyword in report_lower for keyword in [
                "need_patching", "needs improvement", "not helpful", "insufficient"
            ])
            
            indicators["contains_error"] = any(keyword in report_lower for keyword in [
                "error", "failed", "exception", "incorrect", "wrong"
            ])
            
            # Extract specific patterns if present
            if "is_library_helpful" in final_report:
                if "PASS" in final_report:
                    indicators["library_helpful_status"] = "PASS"
                elif "NEED_PATCHING" in final_report:
                    indicators["library_helpful_status"] = "NEED_PATCHING"
                else:
                    indicators["library_helpful_status"] = "UNKNOWN"
            
            # Count key phrases
            indicators["phrase_counts"] = {
                "pass_mentions": report_lower.count("pass"),
                "error_mentions": report_lower.count("error"),
                "helpful_mentions": report_lower.count("helpful"),
                "correct_mentions": report_lower.count("correct")
            }
        
        return indicators
    
    def _validate_openai_schema(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validates if the generated tools' OpenAI schema is compatible.
        Returns a list of errors if any, focusing on common OpenAI schema issues.
        """
        errors = []
        
        for i, tool in enumerate(tools):
            # Handle both 'tool_info' and 'openai_tool' structures
            tool_info = None
            if "tool_info" in tool and tool["tool_info"]:
                tool_info = tool["tool_info"]
            elif "openai_tool" in tool and tool["openai_tool"] and "tool_info" in tool["openai_tool"]:
                tool_info = tool["openai_tool"]["tool_info"]
            
            if not tool_info:
                errors.append({
                    "tool_index": i,
                    "error_message": f"Tool {i} missing tool_info",
                    "error_type": "schema_missing_field"
                })
                continue
            
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
                    # Check for invalid top-level keywords in parameters (root must be object and must not use compositions)
                    invalid_keywords = ['oneOf', 'anyOf', 'allOf', 'not']
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

                        # Root-level enforcement: additionalProperties: false and required must include all fields
                        if parameters.get("type") == "object":
                            if "additionalProperties" not in parameters or parameters.get("additionalProperties") is not False:
                                errors.append({
                                    "tool_index": i,
                                    "error_message": f"Tool {i} parameters object must set additionalProperties: false",
                                    "error_type": "schema_missing_additional_properties",
                                })
                            props = parameters.get("properties", {}) if isinstance(parameters.get("properties", {}), dict) else {}
                            required_list = parameters.get("required")
                            if not isinstance(required_list, list):
                                errors.append({
                                    "tool_index": i,
                                    "error_message": f"Tool {i} parameters object must include 'required' listing all properties",
                                    "error_type": "schema_missing_required"
                                })
                            else:
                                missing = [k for k in props.keys() if k not in set(required_list)]
                                if missing:
                                    errors.append({
                                        "tool_index": i,
                                        "error_message": f"Tool {i} parameters missing required entries for: {missing}",
                                        "error_type": "schema_required_fields_missing",
                                        "missing": missing
                                    })

                        # Global limits and enum constraints
                        def _accumulate_stats(node: Any, depth: int = 1, acc: Dict[str, Any] = None, location: str = "root") -> Dict[str, Any]:
                            if acc is None:
                                acc = {"max_depth": 0, "total_props": 0, "total_string_size": 0, "enum_total_count": 0, "enum_large_props": []}
                            try:
                                acc["max_depth"] = max(acc["max_depth"], depth)
                                if isinstance(node, dict):
                                    # Count names and supported fields for string-size budget
                                    # property names and $defs/definitions names
                                    if node.get("type") == "object":
                                        props = node.get("properties", {})
                                        if isinstance(props, dict):
                                            acc["total_props"] += len(props)
                                            for pname, pdef in props.items():
                                                if isinstance(pname, str):
                                                    acc["total_string_size"] += len(pname)
                                                _accumulate_stats(pdef, depth + 1, acc, location + f".properties.{pname}")
                                        # required array contributes no string budget explicitly
                                    # arrays: traverse items
                                    if node.get("type") == "array" and "items" in node:
                                        _accumulate_stats(node.get("items"), depth + 1, acc, location + ".items")

                                    # enums and consts
                                    if "enum" in node and isinstance(node["enum"], list):
                                        enum_vals = node["enum"]
                                        acc["enum_total_count"] += len(enum_vals)
                                        if all(isinstance(v, str) for v in enum_vals) and len(enum_vals) > 250:
                                            enum_str_total = sum(len(v) for v in enum_vals)
                                            if enum_str_total > 15000:
                                                acc["enum_large_props"].append({"location": location, "total_string": enum_str_total, "count": len(enum_vals)})
                                        # enum strings contribute to total string size
                                        for v in enum_vals:
                                            if isinstance(v, str):
                                                acc["total_string_size"] += len(v)

                                    if "const" in node and isinstance(node["const"], str):
                                        acc["total_string_size"] += len(node["const"]) 

                                    # formats do not add to size budget, but traverse nested compositions/defs
                                    if "anyOf" in node and isinstance(node["anyOf"], list):
                                        for idx, opt in enumerate(node["anyOf"]):
                                            _accumulate_stats(opt, depth + 1, acc, location + f".anyOf[{idx}]")
                                    if "oneOf" in node and isinstance(node["oneOf"], list):
                                        for idx, opt in enumerate(node["oneOf"]):
                                            _accumulate_stats(opt, depth + 1, acc, location + f".oneOf[{idx}]")
                                    # defs
                                    for defs_key in ("$defs", "definitions"):
                                        if defs_key in node and isinstance(node[defs_key], dict):
                                            for dname, ddef in node[defs_key].items():
                                                if isinstance(dname, str):
                                                    acc["total_string_size"] += len(dname)
                                                _accumulate_stats(ddef, depth + 1, acc, location + f".{defs_key}.{dname}")
                                elif isinstance(node, list):
                                    for idx, item in enumerate(node):
                                        _accumulate_stats(item, depth, acc, location + f"[{idx}]")
                            except Exception:
                                pass
                            return acc

                        stats = _accumulate_stats(parameters, depth=1, acc=None)
                        # Limits: depth <= 10
                        if stats.get("max_depth", 0) > 10:
                            errors.append({
                                "tool_index": i,
                                "error_message": f"Tool {i} schema nesting depth {stats['max_depth']} exceeds 10 levels",
                                "error_type": "schema_depth_limit_exceeded",
                                "max_depth": stats.get("max_depth")
                            })
                        # Total properties <= 5000
                        if stats.get("total_props", 0) > 5000:
                            errors.append({
                                "tool_index": i,
                                "error_message": f"Tool {i} schema has {stats['total_props']} properties, exceeds 5000",
                                "error_type": "schema_property_limit_exceeded",
                                "total_props": stats.get("total_props")
                            })
                        # Total string size <= 120000
                        if stats.get("total_string_size", 0) > 120000:
                            errors.append({
                                "tool_index": i,
                                "error_message": f"Tool {i} schema string budget {stats['total_string_size']} exceeds 120000",
                                "error_type": "schema_string_budget_exceeded",
                                "total_string_size": stats.get("total_string_size")
                            })
                        # Enum total count <= 1000
                        if stats.get("enum_total_count", 0) > 1000:
                            errors.append({
                                "tool_index": i,
                                "error_message": f"Tool {i} schema has {stats['enum_total_count']} enum values total, exceeds 1000",
                                "error_type": "schema_enum_count_exceeded",
                                "enum_total_count": stats.get("enum_total_count")
                            })
                        # Per-property enum string budget
                        for item in stats.get("enum_large_props", []):
                            errors.append({
                                "tool_index": i,
                                "error_message": f"Tool {i} enum at {item['location']} exceeds 15000 total string length when >250 values",
                                "error_type": "schema_enum_string_budget_exceeded",
                                "location": item.get("location"),
                                "total_string": item.get("total_string"),
                                "count": item.get("count")
                            })
        
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
                        # Enforce additionalProperties and required for nested objects
                        if prop_def.get("type") == "object":
                            if "additionalProperties" not in prop_def or prop_def.get("additionalProperties") is not False:
                                errors.append({
                                    "tool_index": tool_index,
                                    "error_message": f"Tool {tool_index} object '{prop_name}' must set additionalProperties: false",
                                    "error_type": "schema_missing_additional_properties",
                                    "property_name": prop_name
                                })
                            nested_props = prop_def.get("properties", {}) if isinstance(prop_def.get("properties", {}), dict) else {}
                            required_list = prop_def.get("required")
                            if not isinstance(required_list, list):
                                errors.append({
                                    "tool_index": tool_index,
                                    "error_message": f"Tool {tool_index} object '{prop_name}' must include 'required' listing all properties",
                                    "error_type": "schema_missing_required",
                                    "property_name": prop_name
                                })
                            else:
                                missing = [k for k in nested_props.keys() if k not in set(required_list)]
                                if missing:
                                    errors.append({
                                        "tool_index": tool_index,
                                        "error_message": f"Tool {tool_index} object '{prop_name}' missing required entries for: {missing}",
                                        "error_type": "schema_required_fields_missing",
                                        "property_name": prop_name,
                                        "missing": missing
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
                        # Unsupported composition/keywords inside properties
                        if key in ["oneOf", "allOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else", "patternProperties"]:
                            errors.append({
                                "tool_index": tool_index,
                                "error_message": f"Tool {tool_index} property '{prop_name}' uses unsupported keyword '{key}'",
                                "error_type": "schema_unsupported_keyword",
                                "property_name": prop_name,
                                "keyword": key
                            })
                        # Validate supported string formats when present
                        if key == "format" and isinstance(value, str):
                            allowed_formats = {"date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"}
                            if value not in allowed_formats:
                                errors.append({
                                    "tool_index": tool_index,
                                    "error_message": f"Tool {tool_index} property '{prop_name}' uses unsupported string format '{value}'",
                                    "error_type": "schema_unsupported_string_format",
                                    "property_name": prop_name,
                                    "format": value
                                })
                    
                    # Recursively check nested objects
                    if prop_def.get("type") == "object" and "properties" in prop_def:
                        nested_errors = self._validate_parameters_schema(prop_def, tool_index)
                        errors.extend(nested_errors)
                    # Recursively check arrays' items
                    if prop_def.get("type") == "array" and "items" in prop_def and isinstance(prop_def["items"], dict):
                        nested_errors = self._validate_parameters_schema(prop_def["items"], tool_index)
                        errors.extend(nested_errors)
                    # anyOf branch schemas must each be valid subset schemas
                    if "anyOf" in prop_def and isinstance(prop_def["anyOf"], list):
                        for opt in prop_def["anyOf"]:
                            if isinstance(opt, dict):
                                nested_errors = self._validate_parameters_schema(opt, tool_index)
                                errors.extend(nested_errors)
                            else:
                                errors.append({
                                    "tool_index": tool_index,
                                    "error_message": f"Tool {tool_index} property '{prop_name}' has non-object anyOf option",
                                    "error_type": "schema_invalid_anyof_option",
                                    "property_name": prop_name
                                })
        
        return errors
    
    def _fix_schema_errors(self, tool: Dict[str, Any], schema_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Automatically fix common schema errors without using LLM.
        """
        refined_tool = tool.copy()
        
        # Handle both 'tool_info' and 'openai_tool' structures
        tool_info = None
        if "tool_info" in refined_tool and refined_tool["tool_info"]:
            tool_info = refined_tool["tool_info"].copy()
        elif "openai_tool" in refined_tool and refined_tool["openai_tool"] and "tool_info" in refined_tool["openai_tool"]:
            tool_info = refined_tool["openai_tool"]["tool_info"].copy()
        
        if not tool_info:
            return refined_tool
        
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
                            "required": required,
                            "additionalProperties": False
                        }
                    else:
                        # For other invalid types, create a minimal valid schema
                        tool_info["function"]["parameters"] = {
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False
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
                    if "additionalProperties" not in parameters:
                        parameters["additionalProperties"] = False
                        
            elif error_type == "schema_type_array":
                # Fix type arrays like ['number', 'string'] to use oneOf
                prop_name = error.get("property_name", "")
                invalid_type = error.get("invalid_type", [])
                
                if prop_name and "function" in tool_info and "parameters" in tool_info["function"]:
                    parameters = tool_info["function"]["parameters"]
                    if "properties" in parameters and prop_name in parameters["properties"]:
                        prop_def = parameters["properties"][prop_name]
                        
                        # Convert type array to anyOf
                        if isinstance(invalid_type, list):
                            prop_def["anyOf"] = [{"type": t} for t in invalid_type]
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
                        "required": [],
                        "additionalProperties": False
                    }

            elif error_type in ["schema_missing_additional_properties", "schema_missing_required", "schema_required_fields_missing", "schema_unsupported_keyword", "schema_unsupported_string_format"]:
                # Will be addressed by normalization pass below
                pass
        
        # Normalization pass: enforce additionalProperties: false, all fields required, remove unsupported keywords,
        # replace oneOf with anyOf, fix type arrays, and recursively normalize nested schemas ($defs, anyOf, arrays)
        def _normalize(node: Any) -> Any:
            if isinstance(node, dict):
                # Replace oneOf with anyOf
                if "oneOf" in node and isinstance(node["oneOf"], list):
                    node["anyOf"] = [ _normalize(x) for x in node.get("oneOf", []) ]
                    del node["oneOf"]
                # Remove unsupported keywords
                for kw in ["allOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else", "patternProperties"]:
                    if kw in node:
                        del node[kw]
                # Objects: enforce properties/required/additionalProperties
                if node.get("type") == "object":
                    if "properties" not in node or not isinstance(node.get("properties"), dict):
                        node["properties"] = {}
                    # Recursively normalize each property
                    for k, v in list(node["properties"].items()):
                        node["properties"][k] = _normalize(v)
                    # Enforce required includes all property keys
                    node["required"] = list(node["properties"].keys())
                    node["additionalProperties"] = False
                # Arrays: ensure items and normalize items
                if node.get("type") == "array":
                    if "items" not in node or not isinstance(node["items"], dict):
                        node["items"] = {"type": "string"}
                    else:
                        node["items"] = _normalize(node["items"])
                # Supported anyOf: normalize each option
                if "anyOf" in node and isinstance(node["anyOf"], list):
                    node["anyOf"] = [ _normalize(x) for x in node["anyOf"] ]
                # Clean type arrays -> anyOf
                if "type" in node and isinstance(node["type"], list):
                    node["anyOf"] = [{"type": t} for t in node["type"]]
                    del node["type"]
                # Clean unsupported string formats (keep only allowed)
                if node.get("type") == "string" and "format" in node and isinstance(node["format"], str):
                    allowed_formats = {"date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"}
                    if node["format"] not in allowed_formats:
                        del node["format"]
                # Traverse defs
                for defs_key in ("$defs", "definitions"):
                    if defs_key in node and isinstance(node[defs_key], dict):
                        for dname, ddef in list(node[defs_key].items()):
                            node[defs_key][dname] = _normalize(ddef)
                return node
            elif isinstance(node, list):
                return [ _normalize(x) for x in node ]
            else:
                return node

        if "function" in tool_info and "parameters" in tool_info["function"] and isinstance(tool_info["function"]["parameters"], dict):
            tool_info["function"]["parameters"] = _normalize(tool_info["function"]["parameters"])
        
        # Update the tool with fixed tool_info
        if "tool_info" in refined_tool:
            refined_tool["tool_info"] = tool_info
        elif "openai_tool" in refined_tool:
            refined_tool["openai_tool"]["tool_info"] = tool_info
        
        return refined_tool
    
    def _clean_property_definition(self, prop_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean up a property definition to fix common schema issues.
        """
        cleaned = prop_def.copy()
        
        # Fix type arrays like ['number', 'string']
        if "type" in cleaned and isinstance(cleaned["type"], list):
            type_list = cleaned["type"]
            cleaned["anyOf"] = [{"type": t} for t in type_list]
            del cleaned["type"]
        
        # Fix invalid boolean values in numeric fields
        for field in ["default", "minimum", "maximum"]:
            if field in cleaned and isinstance(cleaned[field], bool):
                if field in ["minimum", "maximum"]:
                    cleaned[field] = 1 if cleaned[field] else 0
                else:
                    del cleaned[field]
        
        return cleaned
    
    def _comprehensive_tool_validation(self, tool_data: Dict, sib_index: int, max_retries: int = 2) -> Tuple[bool, Optional[Dict], str]:
        """
        Comprehensive tool validation with automatic fixing and regeneration.
        
        Three-layer validation:
        1. JSON Schema validation + auto fix
        2. Code syntax + execution validation + LLM fix  
        3. Real OpenAI API execution test + GPT-4.1 regeneration
        
        Args:
            tool_data: Tool data containing tool_info and tool_code
            sib_index: SIB index for logging
            max_retries: Maximum GPT-4.1 regeneration attempts
            
        Returns:
            Tuple of (success: bool, validated_tool: Optional[Dict], final_message: str)
        """
        current_tool = tool_data.copy()
        messages = []
        
        # Layer 1: Schema validation + auto fix
        print(f"      🔍 Layer 1: Schema validation for SIB {sib_index}...")
        schema_errors = self._validate_openai_schema([current_tool])
        
        if schema_errors:
            print(f"      ⚠️ Found {len(schema_errors)} schema errors, auto-fixing...")
            current_tool = self._fix_schema_errors(current_tool, schema_errors)
            
            # Validate again after fixing
            remaining_errors = self._validate_openai_schema([current_tool])
            if remaining_errors:
                messages.append(f"Schema auto-fix incomplete: {len(remaining_errors)} errors remain")
            else:
                messages.append("Schema validation passed after auto-fix")
        else:
            messages.append("Schema validation passed")
        
        # Layer 2: Code validation + LLM fix
        print(f"      🧪 Layer 2: Code validation for SIB {sib_index}...")
        code_validation = self._validate_code_layer(current_tool, sib_index)
        
        if not code_validation["success"]:
            print(f"      ⚠️ Code validation failed: {code_validation['error']}")
            
            # Try LLM fix for code issues
            fixed_tool, _ = self._fix_code_with_llm(
                current_tool.get("tool_code", ""),
                code_validation["error"],
                f"cluster_sib_{sib_index}",
                str(sib_index),
                attempt=1,
                error_type="code_validation",
                conversation_history=[]
            )
            
            if fixed_tool and fixed_tool != current_tool.get("tool_code", ""):
                current_tool = current_tool.copy()
                current_tool["tool_code"] = fixed_tool
                
                # Re-validate fixed code
                code_revalidation = self._validate_code_layer(current_tool, sib_index)
                if code_revalidation["success"]:
                    messages.append("Code validation passed after LLM fix")
                else:
                    messages.append(f"Code validation failed after LLM fix: {code_revalidation['error']}")
            else:
                messages.append(f"Code validation failed and LLM fix unsuccessful")
        else:
            messages.append("Code validation passed")
        
        # Layer 3: Real API execution test
        print(f"      🌐 Layer 3: Real API execution test for SIB {sib_index}...")
        api_result = self._validate_api_layer(current_tool, sib_index)
        
        if not api_result["success"]:
            print(f"      ❌ Real API test failed: {api_result['error']}")
            messages.append(f"API execution failed: {api_result['error']}")
            
            # GPT-4.1 regeneration if retries available
            if max_retries > 0:
                print(f"      🔄 Regenerating with GPT-4.1 (retries left: {max_retries})...")
                
                regenerated_tool = self._regenerate_tool_with_gpt41(
                    current_tool,
                    api_result["error"],
                    sib_index
                )
                
                if regenerated_tool:
                    # Recursive validation with reduced retries
                    return self._comprehensive_tool_validation(
                        regenerated_tool,
                        sib_index,
                        max_retries - 1
                    )
                else:
                    messages.append("GPT-4.1 regeneration failed")
            else:
                messages.append("Max retries reached, no further regeneration")
        else:
            print(f"      ✅ All validation layers passed for SIB {sib_index}")
            messages.append("Real API execution test passed")
            
            final_message = f"Validation completed: {'; '.join(messages)}"
            return True, current_tool, final_message
        
        # If we reach here, validation failed but we have the best available tool
        final_message = f"Validation completed with issues: {'; '.join(messages)}"
        return False, current_tool, final_message
    
    def _validate_code_layer(self, tool_data: Dict, sib_index: int) -> Dict[str, Any]:
        """Validate code syntax and basic execution"""
        tool_code = tool_data.get("tool_code", "")
        
        if not tool_code.strip():
            return {
                "success": False,
                "error": "Empty tool code",
                "error_type": "missing_code"
            }
        
        # Quick syntax validation (clean markdown fences first)
        clean = self._extract_code_from_response(tool_code)
        syntax_result = self._quick_validate_tool_execution(clean, timeout=3)
        
        if not syntax_result["is_valid"]:
            return {
                "success": False,
                "error": syntax_result["error"],
                "error_type": "syntax_error"
            }
        
        return {
            "success": True,
            "message": "Code validation passed"
        }
    
    def _validate_api_layer(self, tool_data: Dict, sib_index: int) -> Dict[str, Any]:
        """Test tool with real OpenAI API execution"""
        try:
            from utils import call_openai_with_temporary_tool
            
            # 1. Prepare tool in OpenAI format
            tool_info = tool_data.get("tool_info", {})
            tool_code = tool_data.get("tool_code", "")
            tool_code = self._extract_code_from_response(tool_code)
            
            if not tool_info or not tool_code:
                return {
                    "success": False,
                    "error": "Missing tool_info or tool_code",
                    "error_type": "missing_data"
                }
            
            # 2. Create function registry
            function_registry = {}
            function_name = tool_info.get("function", {}).get("name", "")
            
            if not function_name:
                return {
                    "success": False,
                    "error": "Missing function name in tool_info",
                    "error_type": "missing_function_name"
                }
            
            # 3. Execute tool code and register function
            try:
                exec_globals = {
                    '__builtins__': __builtins__,
                    'math': __import__('math'),
                    'numpy': __import__('numpy'),
                    'scipy': __import__('scipy'),
                    'sympy': __import__('sympy'),
                    'json': __import__('json'),
                    'datetime': __import__('datetime'),
                    'os': __import__('os'),
                    'sys': __import__('sys'),
                    're': __import__('re'),
                }
                
                exec(tool_code, exec_globals)
                
                if 'execute' not in exec_globals:
                    return {
                        "success": False,
                        "error": "Function 'execute' not found in tool code",
                        "error_type": "missing_execute_function"
                    }
                
                function_registry[function_name] = exec_globals['execute']
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to execute tool code: {str(e)}",
                    "error_type": "code_execution_error"
                }
            
            # 4. Test with OpenAI API
            test_messages = [{"role": "user", "content": "Test this tool with valid parameters you can think of."}]
            
            final_messages, total_turns = call_openai_with_temporary_tool(
                messages=test_messages,
                tools=[tool_info],
                function_registry=function_registry,
                model_name="gpt-4.1",
                max_turns=2,
                temperature=0.1
            ) 
            
            # Log the API validation test
            self._log_llm_call(
                step_name=f"real_api_validation_sib_{sib_index}",
                prompt="Test this tool",
                response=str(final_messages) if final_messages else "",
                success=bool(final_messages),
                error_msg="No messages returned from API test" if not final_messages else None,
                additional_context={
                    "sib_index": sib_index,
                    "test_model": "gpt-4.1",
                    "function_name": function_name,
                    "total_turns": total_turns,
                    "messages_count": len(final_messages) if final_messages else 0
                }
            )
            
            # 5. Check results
            tool_was_called = False
            tool_execution_success = False
            tool_result = ""
            error_message = ""
            
            for msg in final_messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    tool_was_called = True
                elif msg.get("role") == "tool":
                    tool_execution_success = True
                    tool_result = msg.get("content", "")
                    
                    if "error" in tool_result.lower() or "exception" in tool_result.lower() or "failed" in tool_result.lower():
                        tool_execution_success = False
                        error_message = tool_result
                    break
            
            if tool_was_called and tool_execution_success:
                return {
                    "success": True, 
                    "message": "Tool executed successfully",
                    "tool_result": tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                }
            elif tool_was_called and not tool_execution_success:
                return {
                    "success": False,
                    "error": f"Tool execution failed: {error_message}",
                    "error_type": "execution_error"
                }
            else:
                return {
                    "success": False,
                    "error": "Tool was not called by GPT",
                    "error_type": "not_called"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def _regenerate_tool_with_gpt41(self, original_tool_data: Dict, api_error: str, sib_index: int) -> Optional[Dict]:
        """Use GPT-4.1 to regenerate tool based on real API error"""
        try:
            print(f"        🔄 Using GPT-4.1 to regenerate tool based on error...")
            
            # 1. Construct fix prompt
            fix_prompt = f"""The following OpenAI tool failed real API validation/execution with error: {api_error}

Original Tool Info:
{json.dumps(original_tool_data.get("tool_info", {}), indent=2)}

Original Tool Code:
{original_tool_data.get("tool_code", "")}

Please regenerate a corrected OpenAI tool that will work with the OpenAI Function Calling API. 
Requirements:
1. Remove any unsupported fields like 'additionalProperties', 'strict', etc.
2. Ensure the schema is fully compatible with OpenAI Function Calling API
3. Fix any code issues that might cause execution errors
4. Keep the core functionality intact
5. Ensure the execute() function exists and works properly

Return ONLY a valid JSON in this exact format:
{{
  "tool_info": {{
    "type": "function",
    "function": {{
      "name": "function_name",
      "description": "description",
      "parameters": {{
        "type": "object",
        "properties": {{ ... }},
        "required": [...]
      }}
    }}
  }},
  "tool_code": "...\\ndef execute(...):\\n    # implementation\\n    return result"
}}"""
            
            # 2. Call GPT-4.1 to regenerate
            response = call_openai_api(
                content=fix_prompt,
                model_name="gpt-4.1"
            )
            
            # Log the GPT-4.1 regeneration call
            self._log_llm_call(
                step_name=f"gpt41_regeneration_sib_{sib_index}",
                prompt=fix_prompt,
                response=response or "",
                success=bool(response),
                error_msg="Empty response from GPT-4.1" if not response else None,
                additional_context={
                    "sib_index": sib_index,
                    "api_error": api_error,
                    "regeneration_model": "gpt-4.1",
                    "prompt_length": len(fix_prompt)
                }
            )
            
            if not response:
                print(f"        ❌ Empty response from GPT-4.1")
                return None
            
            # 3. Parse the new tool
            try:
                # Try to extract JSON
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_text = response[json_start:json_end].strip()
                elif "{" in response and "}" in response:
                    # Find the first complete JSON object
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    json_text = response[start:end]
                else:
                    json_text = response.strip()
                
                new_tool_data = json.loads(json_text)
                
                # Validate new tool structure
                if "tool_info" in new_tool_data and "tool_code" in new_tool_data:
                    print(f"        ✅ Successfully regenerated tool with GPT-4.1")
                    return new_tool_data
                else:
                    print(f"        ❌ Regenerated tool missing required fields")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"        ❌ Failed to parse regenerated JSON: {e}")
                return None
                
        except Exception as e:
            print(f"        ❌ Error in tool regeneration: {e}")
            return None

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
