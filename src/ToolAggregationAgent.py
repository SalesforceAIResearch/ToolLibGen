import logging
import json
import re
import ast
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils import call_openai_api, call_openai_api_multi_turn, map_with_progress, apply_patch, code_comment2function_json
from prompt import BLUEPRINT_DESIGN_PROMPT, CODE_IMPLEMENTATION_PROMPT, CODE_INSPECTOR_PROMPT_REVISE, OPENAI_TOOL_IMPLEMENTATION_PROMPT, SIB_HELPFULNESS_CHECK_PROMPT, SIB_GENERALIZATION_PROMPT, SIB_HELPFULNESS_CHECK_WITH_TOOL_PROMPT
from IterativeLibraryOptimizerAgent import IterativeLibraryOptimizerAgent


# Logging setup (self-contained)
def setup_logging(debug: bool = False, log_folder: str = None):
    log_level = logging.DEBUG if debug else logging.INFO
    log_dir = Path(log_folder)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"aggregate_tools_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])

    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False
    progress_handler = logging.StreamHandler()
    progress_handler.setLevel(logging.INFO)
    progress_handler.setFormatter(console_formatter)
    progress_logger.addHandler(progress_handler)

    print(f"📝 Log file: {log_file}")
    return log_file, progress_logger


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
    def __init__(self, model_name: str = "gpt-5", debug: bool = False):
        self.model_name = model_name
        self.debug = debug
        self.output_dir: Optional[Path] = None
        self.llm_call_logs: List[Dict[str, Any]] = []

    def _log_llm_call(self, step_name: str, prompt: str, response: str,
                       success: bool = True, error_msg: str = None, additional_context: Dict = None) -> None:
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
            "response_length": len(response) if response else 0,
        }
        self.llm_call_logs.append(log_entry)
        status = "✅" if success else "❌"
        # print(f"  {status} LLM Call {log_entry['call_index']}: {step_name}")
        if not success and error_msg:
            print(f"    Error: {error_msg}")

    def _save_llm_logs(self, cluster_name: str) -> None:
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
                "calls": self.llm_call_logs,
            }
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"📝 Saved {len(self.llm_call_logs)} LLM calls to {log_file_path}")
        except Exception as e:
            print(f"⚠️ Failed to save LLM logs to {self.output_dir}: {e}")

    def _save_blueprint_logs(self, cluster_name: str, start_index: int = 0) -> None:
        """Save blueprint-specific LLM call logs to a separate file"""
        if start_index >= len(self.llm_call_logs):
            print(f"⚠️ No blueprint LLM call logs to save for {cluster_name}")
            return
        if not self.output_dir:
            # print(f"⚠️ No output directory set for {cluster_name}")
            return
        try:
            blueprint_logs = self.llm_call_logs[start_index:]
            log_file_path = self.output_dir / f"{cluster_name}_blueprint_llm_calls.json"
            log_data = {
                "cluster_name": cluster_name,
                "model_name": self.model_name,
                "total_calls": len(blueprint_logs),
                "timestamp": datetime.now().isoformat(),
                "phase": "blueprint_design",
                "calls": blueprint_logs,
            }
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"📝 Saved {len(blueprint_logs)} blueprint LLM calls to {log_file_path}")
        except Exception as e:
            print(f"⚠️ Failed to save blueprint LLM logs to {self.output_dir}: {e}")

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

    def _extract_tools_for_blueprint(self, tools: List[Dict]) -> str:
        if not tools:
            return "No tools found in this cluster."
        parts: List[str] = []
        for i, tool in enumerate(tools):
            tool_name = tool.get('name', f'tool_{i+1}')
            description = tool.get('description', f'Function: {tool_name}')
            parts.append(f"# Function {i+1}: {tool_name}")
            parts.append(f"# Description: {description}")
            parts.append("")
        return "\n".join(parts)

    def _extract_code_from_response(self, response: str) -> str:
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
            return (response or "").strip()

    def _parse_sibs_from_blueprint(self, blueprint_response: str) -> List[Dict]:
        sibs: List[Dict] = []
        sections = blueprint_response.split("<SIB>")
        for i, section in enumerate(sections):
            if i == 0:
                continue
            if "</SIB>" in section:
                section = section.split("</SIB>")[0]
            section = section.strip()
            if section:
                sibs.append({
                    "index": i,
                    "content": section,
                    "covered_tools": [],
                })
        return sibs

    def _extract_covered_tools_from_sibs(self, sibs: List[Dict]) -> None:
        for sib in sibs:
            sib['covered_tools'] = []
            content = sib.get('content', '')
            match = re.search(r'\[Covered Tools\]\s*\n(.*?)(?:\n\[|$)', content, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                indices: List[int] = []
                indices += [int(m) for m in re.findall(r'tool\s*(\d+)', text, re.IGNORECASE)]
                indices += [int(m) for m in re.findall(r'\b(\d+)\b', text)]
                sib['covered_tools'] = sorted(list(set(indices)))
            if sib['covered_tools']:
                print(f"    SIB {sib['index']}: covers tools {sib['covered_tools']}")
            else:
                print(f"    SIB {sib['index']}: no tool coverage found")

    def _design_blueprint(self, cluster_name: str, tools: List[Dict], model_name: str = None) -> Tuple[bool, List[Dict], Optional[str]]:
        try:
            # Track blueprint-specific LLM calls
            blueprint_start_log_index = len(self.llm_call_logs)
            
            model_to_use = model_name or self.model_name
            print(f"  📋 Designing blueprint for {cluster_name}...")
            tools_info = self._extract_tools_for_blueprint(tools)
            prompt_text = BLUEPRINT_DESIGN_PROMPT.format(tool_code_name_list=tools_info, domain=cluster_name)
            response = call_openai_api(content=prompt_text, model_name=model_to_use)
            self._log_llm_call(
                step_name="blueprint_design",
                prompt=prompt_text,
                response=response or "",
                success=bool(response),
                error_msg="Empty response from LLM" if not response else None,
                additional_context={"cluster_name": cluster_name, "tools_count": len(tools), "tools_info_length": len(tools_info)},
            )
            if not response:
                # Save blueprint logs even on failure
                self._save_blueprint_logs(cluster_name, blueprint_start_log_index)
                return False, [], "Empty response from LLM for blueprint design"
            print(f"  ✅ Blueprint design completed for {cluster_name}")
            
            # Apply SIB generalization after blueprint design
            print(f"  🔄 Applying SIB generalization for {cluster_name}...")
            try:
                generalization_prompt = SIB_GENERALIZATION_PROMPT.format(
                    original_sib_text=response
                )
                messages = [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": generalization_prompt}
                ]
                generalized_response = call_openai_api_multi_turn(
                    messages=messages,
                    model_name=model_to_use,
                )
                self._log_llm_call(
                    step_name="blueprint_generalization",
                    prompt=generalization_prompt,
                    response=generalized_response or "",
                    success=bool(generalized_response),
                    error_msg="Empty response from LLM" if not generalized_response else None,
                    additional_context={"cluster_name": cluster_name, "tools_count": len(tools)}
                )
                
                if generalized_response:
                    # Extract generalized SIBs
                    def _extract_rewritten(text: str) -> str:
                        start, end = "<REWRITTEN_SIB>", "</REWRITTEN_SIB>"
                        if start in text and end in text:
                            try:
                                return text.split(start)[1].split(end)[0].strip()
                            except Exception:
                                return ""
                        return ""
                    
                    generalized_sibs = _extract_rewritten(generalized_response)
                    if generalized_sibs:
                        response = generalized_sibs
                        print(f"  ✅ SIB generalization completed for {cluster_name}")
                    else:
                        print(f"  ⚠️ Could not extract generalized SIBs, using original blueprint")
                else:
                    print(f"  ⚠️ SIB generalization failed, using original blueprint")
            except Exception as e:
                print(f"  ⚠️ SIB generalization error: {e}, using original blueprint")
            
            sibs = self._parse_sibs_from_blueprint(response)
            if not sibs:
                # Save blueprint logs even on failure
                self._save_blueprint_logs(cluster_name, blueprint_start_log_index)
                return False, [], "No SIBs found in blueprint response"
            print(f"  📊 Found {len(sibs)} SIBs in blueprint")
            self._extract_covered_tools_from_sibs(sibs)
            total_covered = sum(len(s['covered_tools']) for s in sibs)
            print(f"  📈 Coverage: {total_covered} tool references found in SIBs")
            
            # Save blueprint-specific logs
            self._save_blueprint_logs(cluster_name, blueprint_start_log_index)
            
            return True, sibs, None
        except Exception as e:
            msg = f"Error in blueprint design: {str(e)}"
            # print(f"  ❌ {msg}")
            # Save blueprint logs even on exception
            blueprint_start_log_index = getattr(self, '_blueprint_start_log_index', 0)
            self._save_blueprint_logs(cluster_name, blueprint_start_log_index)
            return False, [], msg

    def design_blueprint_only(self, cluster_name: str, tools: List[Dict], model_name: str = None) -> Tuple[bool, List[Dict], Optional[str]]:
        return self._design_blueprint(cluster_name, tools, model_name)

    def _sanitize_tool_name(self, name: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
        if not name or not re.match(r"^[a-zA-Z]", name):
            name = f"tool_{name}" if name else "tool"
        return name[:64]

    def _extract_sig_and_doc_from_code(self, code: str) -> Tuple[str, str]:
        """
        Extract execute function signature and docstring from code.
        Returns (signature_line, docstring_text) with appropriate formatting.
        """
        try:
            # Try to parse AST
            tree = ast.parse(code)
        except SyntaxError:
            # If syntax error, fallback to simple string processing
            return self._fallback_extract_sig_and_doc(code)
        except Exception:
            return "", ""
        
        # Find execute function
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "execute":
                try:
                    # Construct signature line
                    if hasattr(ast, 'unparse'):  # Python 3.9+
                        args_str = ast.unparse(node.args)
                        returns_str = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                    else:
                        # Python 3.8 fallback - use astor or simple processing
                        args_str = self._format_args_legacy(node.args)
                        returns_str = ""
                    
                    sig_line = f"def execute({args_str}){returns_str}:"
                    
                    # Extract docstring
                    doc_text = ""
                    if (node.body and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str)):
                        doc_text = node.body[0].value.value
                    elif hasattr(ast, 'get_docstring'):
                        doc_text = ast.get_docstring(node) or ""
                    
                    if doc_text:
                        doc_text = f'"""\n{doc_text}\n"""'
                    
                    return sig_line, doc_text
                    
                except Exception:
                    # AST processing failed, fallback
                    return self._fallback_extract_sig_and_doc(code)
        
        return "", ""

    def _format_args_legacy(self, args_node) -> str:
        """Format parameter list for Python 3.8 compatibility"""
        parts = []
        
        # Handle regular parameters
        for arg in args_node.args:
            arg_str = arg.arg
            if arg.annotation:
                # Simple type annotation processing
                if hasattr(arg.annotation, 'id'):
                    arg_str += f": {arg.annotation.id}"
                elif hasattr(arg.annotation, 'attr'):
                    arg_str += f": {arg.annotation.attr}"
            parts.append(arg_str)
        
        return ", ".join(parts)

    def _fallback_extract_sig_and_doc(self, code: str) -> Tuple[str, str]:
        """Fallback method when AST parsing fails"""
        lines = code.split('\n')
        sig_lines = []
        doc_lines = []
        
        in_signature = False
        in_docstring = False
        paren_count = 0
        docstring_quote = None
        
        for line in lines:
            stripped = line.strip()
            
            # Find function definition start
            if not in_signature and stripped.startswith('def execute('):
                in_signature = True
                sig_lines.append(line)
                paren_count = line.count('(') - line.count(')')
                if ':' in line and paren_count == 0:
                    in_signature = False
                continue
            
            # Continue collecting signature lines
            if in_signature:
                sig_lines.append(line)
                paren_count += line.count('(') - line.count(')')
                if ':' in line and paren_count == 0:
                    in_signature = False
                continue
            
            # Find docstring
            if not in_docstring and sig_lines and (stripped.startswith('"""') or stripped.startswith("'''")):
                in_docstring = True
                docstring_quote = '"""' if stripped.startswith('"""') else "'''"
                doc_lines.append(line)
                if stripped.count(docstring_quote) >= 2:  # Single line docstring
                    in_docstring = False
                continue
            
            if in_docstring:
                doc_lines.append(line)
                if docstring_quote in stripped:
                    in_docstring = False
                continue
            
            # If signature collected but no docstring, stop
            if sig_lines and not stripped:
                break
        
        sig_text = '\n'.join(sig_lines) if sig_lines else ""
        doc_text = '\n'.join(doc_lines) if doc_lines else ""
        
        return sig_text, doc_text

    def _quick_validate_tool_execution(self, tool_code: str, timeout: int = 3) -> Dict[str, Any]:
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
               #  "except Exception as e:\n"
               #  "    result = {\"is_valid\": False, \"error\": f\"Execution failed: {str(e)}\", \"can_call\": False}\n"
               "except Exception as e:\n"
                "    import traceback, sys\n"
                "    exc_type, exc_value, exc_traceback = sys.exc_info()\n"
                "    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)\n"
                "    full_traceback = ''.join(tb_lines)\n"
                "    # Try to find the specific line where the error occurred\n"
                "    error_line_info = None\n"
                "    try:\n"
                "        for i, line in enumerate(_code.split('\\n')):\n"
                "            if line.strip() and any(f'line {i+1}' in tb_line for tb_line in tb_lines):\n"
                "                error_line_info = f'Line {i+1}: {line.strip()}'\n"
                "                break\n"
                "    except:\n"
                "        pass\n"
                "    # Get available functions and variables\n"
                "    available_names = [name for name in env.keys() if not name.startswith('_')]\n"
                "    result = {\n"
                "        'is_valid': False, \n"
                "        'error': f'Execution failed: {str(e)}'+error_line_info+full_traceback, \n"
                "        'can_call': False,\n"
                "        'error_type': exc_type.__name__ if exc_type else 'Unknown',\n"
                "        'error_line': error_line_info,\n"
                "        'full_traceback': full_traceback,\n"
                "        'available_names': available_names,\n"
                "        'code_length': len(_code),\n"
                "        'duplicates': _code.count('def execute(')\n"
                "    }\n"
                "print(json.dumps(result))\n"
            )
            output = execute_code(inspector, timeout=timeout)
            if output and output.strip().startswith("{"):
                import json as _json
                try:
                    data = _json.loads(output.strip().splitlines()[-1])
                    return {
                        "is_valid": bool(data.get("is_valid")),
                        "error": data.get("error"),
                        "can_call": bool(data.get("can_call")),
                        "function_params": data.get("function_params", []),
                    }
                except Exception:
                    pass
            if output and "Timeout" in output:
                return {"is_valid": False, "error": "Timeout during validation", "can_call": False}
            return {"is_valid": False, "error": (output or "Unknown error"), "can_call": False}
        except Exception as e:
            return {"is_valid": False, "error": f"Validation harness error: {str(e)}", "can_call": False}

    def _fix_code_with_llm(self, code: str, error_message: str, cluster_name: str,
                          tool_name: str, attempt: int, error_type: str,
                          conversation_history: List[Dict] = None, generation_context: str = None) -> Tuple[Optional[str], List[Dict]]:
        try:
            print(f"        🔧 Attempting LLM fix for {error_type} (attempt {attempt})")
            if conversation_history is None:
                conversation_history = []
            fixing_prompt = CODE_INSPECTOR_PROMPT_REVISE.format(
                code=code,
                error=error_message
            )
            if not conversation_history:
                messages = []
                # Add generation context as first message if available
                if generation_context:
                    messages.append({
                        "role": "user", 
                        "content": f"Context: This code was generated from the following blueprint/prompt:\n\n{generation_context}"
                    })
                messages.append({"role": "user", "content": fixing_prompt})
            else:
                messages = conversation_history.copy()
                follow_up_message = f"""The previous fix didn't work. Here's the new error:

Code:
{code}

Error:
{error_message}

Please provide another fix in the same unified diff format. Learn from the previous attempts and avoid repeating the same mistakes."""
                messages.append({"role": "user", "content": follow_up_message})
            response = call_openai_api_multi_turn(
                model_name=self.model_name,
                messages=messages
            )
            if response:
                messages.append({"role": "assistant", "content": response})
                conversation_history = messages
            self._log_llm_call(
                step_name=f"code_fix_{error_type}_{tool_name}_attempt_{attempt}",
                prompt=str(messages),
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
            diff_text = ""
            if "<diff>" in response and "</diff>" in response:
                try:
                    diff_text = response.split("<diff>")[1].split("</diff>")[0].strip()
                except Exception:
                    diff_text = response.strip()
            else:
                diff_text = response.strip()
            if not diff_text:
                # print(f"        ❌ No diff found in LLM response")
                return None, conversation_history
            fixed_code = apply_patch(code, diff_text)
            if fixed_code is None:
                print(f"        ❌ Failed to apply patch")
                return None, conversation_history
            print(f"        ✅ Successfully applied LLM fix (turn {len(conversation_history) // 2})")
            return fixed_code, conversation_history
        except Exception as e:
            print(f"        ❌ Error in LLM fix: {e}")
            return None, conversation_history or []

    def _validate_and_refine_code_with_retry(self, tool_code: str, context: Dict, max_attempts: int = 2) -> Tuple[bool, str, Dict]:
        metadata: Dict[str, Any] = {"attempts": []}
        refined = self._extract_code_from_response(tool_code or "")
        for i in range(max_attempts):
            attempt_info = {"attempt": i + 1}
            try:
                # Validate current code before any fix
                quick_before = self._quick_validate_tool_execution(refined, timeout=3)
                attempt_info["quick"] = quick_before
                attempt_info["quick_before_fix"] = quick_before

                # If already valid, return success
                if quick_before.get("is_valid") and quick_before.get("can_call"):
                    metadata["attempts"].append(attempt_info)
                    return True, refined, metadata

                # Otherwise, try to fix with LLM
                err = quick_before.get("error") or "Invalid code"
                generation_context = context.get("blueprint", "") or context.get("implementation_prompt", "")
                fixed_code, _ = self._fix_code_with_llm(
                    refined,
                    err,
                    context.get("step_name", "unknown"),
                    str(context.get("sib_index", "0")),
                    attempt=i + 1,
                    error_type="code_validation",
                    conversation_history=[],
                    generation_context=generation_context
                )

                # If we received a fix, re-validate the fixed code
                if fixed_code and fixed_code.strip():
                    refined_candidate = self._extract_code_from_response(fixed_code)
                    quick_after = self._quick_validate_tool_execution(refined_candidate, timeout=3)
                    attempt_info["quick_after_fix"] = quick_after

                    if quick_after.get("is_valid") and quick_after.get("can_call"):
                        # Fixed code is valid; update refined and return success
                        refined = refined_candidate
                        metadata["attempts"].append(attempt_info)
                        return True, refined, metadata
                    else:
                        # Fixed code still invalid; record details and continue to next attempt
                        attempt_info["fix_failed"] = True
                        attempt_info["fix_error"] = quick_after.get("error")
                        refined = refined_candidate  # keep latest candidate for transparency
                else:
                    # No usable fix produced
                    attempt_info["fix_failed"] = True
                    attempt_info["fix_error"] = "No fix produced or failed to apply patch"

                metadata["attempts"].append(attempt_info)
            except Exception as e:
                attempt_info["exception"] = str(e)
                metadata["attempts"].append(attempt_info)
        return False, refined, metadata

    def _validate_api_and_refine_schema_with_retry(self, tool_info: Dict, code: str, context: Dict, max_attempts: int = 2) -> Tuple[bool, Dict, Dict]:
        """Validate with real API and refine by rewriting schema only if it fails. Returns (ok, refined_tool_info, metadata)."""
        metadata: Dict[str, Any] = {"attempts": []}
        current_tool_info = tool_info or {}
        for i in range(max_attempts):
            attempt_info = {"attempt": i + 1}
            try:
                # Reuse v2 API layer validation approach
                try:
                    api_res = self._validate_api_layer({"tool_info": current_tool_info, "tool_code": code}, context.get("sib_index", 0))
                except AttributeError:
                    # If _validate_api_layer not present in v3, inline minimal fallback using utils.call_openai_with_temporary_tool
                    from utils import call_openai_with_temporary_tool
                    function_name = current_tool_info.get("function", {}).get("name", "")
                    if not function_name:
                        return False, current_tool_info, metadata
                    exec_globals = {"__builtins__": __builtins__}
                    exec(code, exec_globals)
                    if 'execute' not in exec_globals:
                        return False, current_tool_info, metadata
                    function_registry = {function_name: exec_globals['execute']}
                    test_messages = [{"role": "user", "content": "Test this tool with valid parameters you can think of."}]
                    final_messages, _ = call_openai_with_temporary_tool(
                        messages=test_messages,
                        tools=[current_tool_info],
                        function_registry=function_registry,
                        model_name="gpt-4.1",
                        max_turns=2,
                        temperature=0.1
                    )
                    tool_was_called = any(m.get("role") == "assistant" and m.get("tool_calls") for m in final_messages)
                    tool_execution_success = any(m.get("role") == "tool" for m in final_messages)
                    api_res = {"success": bool(tool_was_called and tool_execution_success)}

                attempt_info["api_success"] = bool(api_res.get("success"))
                if api_res.get("success"):
                    metadata["attempts"].append(attempt_info)
                    return True, current_tool_info, metadata
                # On failure, we don't call LLM in v3; just return failure metadata
               #  print(f"        ⚠️ API validation failed: {api_res.get('error')}")
                metadata["attempts"].append(attempt_info)
            except Exception as e:
                attempt_info["exception"] = str(e)
                metadata["attempts"].append(attempt_info)
        return False, current_tool_info, metadata

    def _validate_api_layer(self, tool_data: Dict, sib_index: int) -> Dict[str, Any]:
        """Test tool with real OpenAI API execution (mirrored from v2)."""
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
            function_registry: Dict[str, Any] = {}
            function_name = tool_info.get("function", {}).get("name", "")

            if not function_name:
                return {
                    "success": False,
                    "error": "Missing function name in tool_info",
                    "error_type": "missing_function_name"
                }

            # 3. Execute tool code and register function (using safe execution from _quick_validate_tool_execution)
            try:
                # Reuse the safe execution logic from _quick_validate_tool_execution
                validation_result = self._quick_validate_tool_execution(tool_code, timeout=2)
                
                if not validation_result.get("is_valid", False):
                    return {
                        "success": False,
                        "error": f"Tool code validation failed: {validation_result.get('error', 'Unknown error')}",
                        "error_type": "code_validation_error"
                    }
                
                if not validation_result.get("can_call", False):
                    return {
                        "success": False,
                        "error": "Function 'execute' not found or not callable in tool code",
                        "error_type": "missing_execute_function"
                    }
                
                # Create a safe execution environment for the actual function registry
                import base64
                from utils import execute_code
                encoded = base64.b64encode(tool_code.encode("utf-8")).decode("ascii")
                registry_script = (
                    "import base64, json\n"
                    f"_code = base64.b64decode('{encoded}').decode('utf-8')\n"
                    "env = {}\n"
                    "exec(_code, env)\n"
                    "if 'execute' in env:\n"
                    "    # Store the function for later use\n"
                    "    _func = env['execute']\n"
                    "    print('FUNCTION_READY')\n"
                    "else:\n"
                    "    print('FUNCTION_NOT_FOUND')\n"
                )
                
                result = execute_code(registry_script, timeout=3)
                if "FUNCTION_READY" not in result:
                    return {
                        "success": False,
                        "error": "Failed to prepare function for registry",
                        "error_type": "function_preparation_error"
                    }
                
                # Create a wrapper function that safely executes the tool code
                def safe_execute_wrapper(*args, **kwargs):
                    import base64
                    from utils import execute_code
                    import json
                    
                    # Prepare arguments
                    args_json = json.dumps({"args": args, "kwargs": kwargs})
                    args_b64 = base64.b64encode(args_json.encode("utf-8")).decode("ascii")
                    code_b64 = base64.b64encode(tool_code.encode("utf-8")).decode("ascii")
                    
                    wrapper_script = (
                        "import base64, json\n"
                        f"_code = base64.b64decode('{code_b64}').decode('utf-8')\n"
                        f"_args_data = json.loads(base64.b64decode('{args_b64}').decode('utf-8'))\n"
                        "env = {}\n"
                        "exec(_code, env)\n"
                        "if 'execute' in env:\n"
                        "    try:\n"
                        "        result = env['execute'](*_args_data['args'], **_args_data['kwargs'])\n"
                        "        print('RESULT:', result)\n"
                        "    except Exception as e:\n"
                        "        print('ERROR:', str(e))\n"
                        "else:\n"
                        "    print('ERROR: execute function not found')\n"
                    )
                    
                    exec_result = execute_code(wrapper_script, timeout=10)
                    if exec_result.startswith("RESULT:"):
                        return exec_result[7:].strip()  # Remove "RESULT:" prefix
                    elif exec_result.startswith("ERROR:"):
                        raise Exception(exec_result[6:].strip())  # Remove "ERROR:" prefix
                    else:
                        raise Exception(f"Unexpected execution result: {exec_result}")
                
                function_registry[function_name] = safe_execute_wrapper

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to prepare safe execution environment: {str(e)}",
                    "error_type": "safe_execution_setup_error"
                }

            # 4. Test with OpenAI API
            test_messages = [{"role": "user", "content": "Test this tool with valid parameters."}]

            final_messages, total_turns = call_openai_with_temporary_tool(
                messages=test_messages,
                tools=[tool_info],
                function_registry=function_registry,
                model_name="gpt-4.1",
                max_turns=1,
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

    def _generate_tool_json_with_retry(self, messages: List[Dict], step_name: str, max_retries: int = 3,
                                       additional_context: Dict = None) -> Optional[Dict]:
        code_response: Optional[str] = None
        for attempt in range(max_retries):
            try:
                print(f"        🧱 Code generation attempt {attempt + 1}/{max_retries}...")
                code_response = call_openai_api_multi_turn(
                    messages=messages,
                    model_name="gpt-5"
                )
#                 code_response = '''
#                '''
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
                tool_code_clean = self._extract_code_from_response(code_response)
                if not tool_code_clean.strip():
                    raise ValueError("Empty code extracted from response")
                print(f"        ✅ Code generated (length={len(tool_code_clean)})")
                break
            except Exception as e:
                # print(f"        ❌ Code generation failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    return None
                retry_message = "The previous response did not include a valid <code>...</code> or ```python block with complete code. Please return only the code wrapped in <code>...</code>."
                messages.append({"role": "assistant", "content": code_response or ""})
                messages.append({"role": "user", "content": retry_message})
                continue
        tool_code_clean = self._extract_code_from_response(code_response or "")
        
        if not tool_code_clean.strip():
            return None
        # NOTE: We now validate per combined function code after splitting class/functions
        # try:
        #     print(f"        🧪 Layer: Code validation & minimal refine...")
        #     code_ok, refined_code, _code_meta = self._validate_and_refine_code_with_retry(
        #         tool_code_clean,
        #         context={"step_name": step_name},
        #         max_attempts=2
        #     )
        #     tool_code_clean = refined_code if refined_code else tool_code_clean
        #     if not code_ok:
        #         print(f"        ❌ Code did not pass validation after refinement attempts")
        #         return None
        #     else:
        #         print(f"        ✅ Code validation passed")
        # except Exception as e:
        #     print(f"        ❌ Error during code validation/refinement: {e}")
        #     return None
        # NOTE: Replace LLM-based tool schema generation with local parsing
        # import textwrap as _textwrap
        # tool_info_prompt = OPENAI_TOOL_IMPLEMENTATION_PROMPT.format(code=tool_code_clean)
        # tool_info_messages = [{"role": "user", "content": tool_info_prompt}]
        # tool_info: Optional[Dict] = None
        # for attempt in range(max_retries):
        #     try:
        #         print(f"        🧩 Tool schema generation attempt {attempt + 1}/{max_retries}...")
        #         info_response = call_openai_api_multi_turn(
        #             messages=tool_info_messages,
        #             model_name="gpt-5"
        #         )
        #         self._log_llm_call(
        #             step_name=f"{step_name}_info_attempt_{attempt + 1}",
        #             prompt=tool_info_messages[-1]['content'] if tool_info_messages else "",
        #             response=info_response or "",
        #             success=bool(info_response),
        #             error_msg="Empty response from LLM" if not info_response else None,
        #             additional_context={
        #                 **(additional_context or {}),
        #                 "phase": "tool_info",
        #                 "attempt": attempt + 1,
        #                 "max_retries": max_retries
        #             }
        #         )
        #         if not info_response:
        #             print(f"        ⚠️ Empty tool_info response on attempt {attempt + 1}/{max_retries}")
        #             continue
        #         if "```json" in info_response:
        #             js = info_response.split("```json", 1)[1]
        #             js = js.split("```", 1)[0]
        #             json_text = js.strip()
        #         elif "<json>" in info_response and "</json>" in info_response:
        #             json_text = info_response.split("<json>", 1)[1].split("</json>", 1)[0].strip()
        #         else:
        #             json_text = info_response.strip()
        #         parsed = json.loads(json_text)
        #         if not isinstance(parsed, dict) or "function" not in parsed:
        #             raise ValueError("Parsed tool_info is not a dict or missing 'function'")
        #         tool_info = {"type": "function", "function": parsed.get("function", {})}
        #         fn = tool_info.get("function", {}).get("name", "")
        #         print(f"        ✅ Tool schema generated for function: {fn or '(unnamed)'}")
        #         break
        #     except Exception as e:
        #         print(f"        ❌ tool_info generation failed on attempt {attempt + 1}/{max_retries}: {e}")
        #         if attempt == max_retries - 1:
        #             return None
        #         retry_msg = _textwrap.dedent(
        #             """
        #             The previous response was not a valid JSON for OpenAI function schema. Return only the JSON wrapped in <json>...</json> with keys: type, function{name, description, parameters}.
        #             """
        #         ).strip()
        #         tool_info_messages.append({"role": "assistant", "content": info_response or ""})
        #         tool_info_messages.append({"role": "user", "content": retry_msg})
        #         continue
        try:
            # Parse <class> and <function_X> blocks from the refined code
            import re as _re
            code_body = tool_code_clean
            # If still wrapped with <code> ... </code>, extract inside
            m_code = _re.search(r"<code>\s*(.*?)\s*</code>", code_body, _re.DOTALL)
            if m_code:
                code_body = m_code.group(1).strip()

            class_blocks = _re.findall(r"<class>\s*(.*?)\s*</class>", code_body, _re.DOTALL)
            class_code_combined = "\n\n".join(cb.strip() for cb in class_blocks if cb and cb.strip())
            if "import" in code_body:
               class_code_combined = code_body.split("<class>")[0] + class_code_combined

            func_blocks = []
            for m in _re.finditer(r"<function_(\d+)>\s*(.*?)\s*</function_\1>", code_body, _re.DOTALL):
                idx = m.group(1)
                func_code = m.group(2).strip()
                if func_code:
                    func_blocks.append((idx, func_code))

            function_tools: List[Dict[str, Any]] = []
            if func_blocks:
                for idx, fcode in func_blocks:
                    try:
                        # Extract original function name FIRST, before any processing
                        orig_name_match = _re.search(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", fcode, _re.MULTILINE)
                        original_func_name = orig_name_match.group(1) if orig_name_match else None
                        
                        # Replace function name to 'execute' in fcode for all subsequent processing
                        exec_fcode = fcode
                        if original_func_name and original_func_name != 'execute':
                            exec_fcode = _re.sub(rf"^\s*def\s+{original_func_name}\s*\(", "def execute(", exec_fcode, count=1, flags=_re.MULTILINE)
                        
                        # Create combined code with renamed function
                        combined_code = (class_code_combined + "\n\n" + exec_fcode).strip() if class_code_combined else exec_fcode
                        
                        # Validate each combined function+class code separately
                        code_ok, refined_code, _meta = self._validate_and_refine_code_with_retry(
                            combined_code,
                            context={"step_name": f"{step_name}_func_{idx}"},
                            max_attempts=1
                        )
                        if not code_ok:
                            # print(f"        ❌ Combined code validation failed for function {idx}; skipping")
                            continue
                        final_code = refined_code if refined_code else combined_code

                        # Extract function signature and optional docstring text for schema (from exec_fcode)
                        sig_line, doc_text = self._extract_sig_and_doc_from_code(exec_fcode)
                        if sig_line and doc_text:
                            comment_input = f"{sig_line}\n\n{doc_text}"
                        else:
                            comment_input = sig_line or doc_text or ""
                        tinfo = code_comment2function_json(comment_input)
                        tinfo_str = json.dumps(tinfo)
                        if not isinstance(tinfo, dict) or not tinfo.get("function"):
                            raise ValueError("Invalid tool schema")

                        # API-level validation and possible schema refine (using final_code with execute)
                        try:
                            sib_idx = (additional_context or {}).get("sib_index", 0)
                            api_ok, api_refined_tool_info, _api_meta = self._validate_api_and_refine_schema_with_retry(
                                tool_info=tinfo,
                                code=final_code,
                                context={"step_name": f"{step_name}_func_{idx}", "sib_index": sib_idx},
                                max_attempts=1
                            )
                            if not api_ok:
                                print(f"        ⚠️ API validation/refine failed for sib {sib_idx} function {idx}: {_api_meta}")
                                continue
                            tinfo = api_refined_tool_info if api_refined_tool_info else tinfo
                        except Exception as _api_e:
                            print(f"        ⚠️ API validation/refine failed for function {idx}: {_api_e}")

                        # NOW restore original function name in tool_info and tool_code
                        if original_func_name and original_func_name != 'execute':
                            # Restore function name in tool_info
                            if tinfo.get("function", {}).get("name") == "execute":
                                tinfo["function"]["name"] = original_func_name
                            # Restore function name in final_code for tool_code
                            final_code_restored = _re.sub(r"^\s*def\s+execute\s*\(", f"def {original_func_name}(", final_code, count=1, flags=_re.MULTILINE)
                        else:
                            final_code_restored = final_code

                        function_tools.append({
                            "tool_info": tinfo,
                            "tool_code": f"```python\n{final_code_restored}\n```"
                        })
                    except Exception as _inner_e:
                        # Skip this function if schema construction or validation fails
                        print(f"        ⚠️ Skipping function {idx} due to error: {_inner_e}")

            if not function_tools:
                # Fallback: validate and construct from entire code
                # Extract original function name FIRST
                orig_name_match_whole = _re.search(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", tool_code_clean, _re.MULTILINE)
                original_func_name_whole = orig_name_match_whole.group(1) if orig_name_match_whole else None
                
                # Replace function name to 'execute' for all processing
                exec_code_clean = tool_code_clean
                if original_func_name_whole and original_func_name_whole != 'execute':
                    exec_code_clean = _re.sub(rf"^\s*def\s+{original_func_name_whole}\s*\(", "def execute(", exec_code_clean, count=1, flags=_re.MULTILINE)
                
                code_ok, refined_code, _meta = self._validate_and_refine_code_with_retry(
                    exec_code_clean,
                    context={"step_name": f"{step_name}_single"},
                    max_attempts=2
                )
                if not code_ok:
                    # print(f"        ❌ Full code validation failed; aborting tool schema construction")
                    return None
                used_code = refined_code if refined_code else exec_code_clean
                
                # For schema, extract function signature from the execute version
                sig_line_whole, doc_text_whole = self._extract_sig_and_doc_from_code(used_code)
                if sig_line_whole and doc_text_whole:
                    comment_input_whole = f"{sig_line_whole}\n\n{doc_text_whole}"
                else:
                    comment_input_whole = sig_line_whole or doc_text_whole or ""
                tinfo_single = code_comment2function_json(comment_input_whole)
                if not isinstance(tinfo_single, dict) or not tinfo_single.get("function"):
                    raise ValueError("Failed to construct function schema from code")

                # API-level validation and possible schema refine on the single tool
                try:
                    sib_idx = (additional_context or {}).get("sib_index", 0)
                    api_ok, api_refined_tool_info, _api_meta = self._validate_api_and_refine_schema_with_retry(
                        tool_info=tinfo_single,
                        code=used_code,
                        context={"step_name": f"{step_name}_single", "sib_index": sib_idx},
                        max_attempts=1
                    )
                    tinfo_single = api_refined_tool_info if api_refined_tool_info else tinfo_single
                except Exception as _api_e:
                    print(f"        ⚠️ API validation/refine failed for single tool: {_api_e}")
                # NOW restore original function name in tool_info and tool_code
                if original_func_name_whole and original_func_name_whole != 'execute':
                    # Restore function name in tool_info
                    if tinfo_single.get("function", {}).get("name") == "execute":
                        tinfo_single["function"]["name"] = original_func_name_whole
                    # Restore function name in used_code for tool_code
                    used_code_restored = _re.sub(r"^\s*def\s+execute\s*\(", f"def {original_func_name_whole}(", used_code, count=1, flags=_re.MULTILINE)
                else:
                    used_code_restored = used_code
                
                fn = tinfo_single.get("function", {}).get("name", "")
                print(f"        ✅ Tool schema constructed locally (single) for function: {fn or '(unnamed)'}")
                return {"tool_info": tinfo_single, "tool_code": f"```python\n{used_code_restored}\n```"}

            # Use the first function as primary for backward compatibility
            primary = function_tools[0]
            pname = primary["tool_info"].get("function", {}).get("name", "")
            print(f"        ✅ Constructed {len(function_tools)} tool schemas; primary: {pname or '(unnamed)'}")
            return {
                "tool_info": primary["tool_info"],
                "tool_code": primary["tool_code"],
                "function_tools": function_tools
            }
        except Exception as e:
            # print(f"        ❌ Local tool schema construction failed: {e}")
            return None

    def _generate_sib_tool(self, cluster_name: str, sib: Dict, tools: List[Dict]) -> Tuple[bool, Optional[Dict], Optional[str]]:
        try:
            sib_index = sib.get('index', 0)
            blueprint_content = sib.get('content', '')
            implementation_prompt = CODE_IMPLEMENTATION_PROMPT.format(
                blueprint=blueprint_content,
            )
            implementation_messages = [{"role": "user", "content": implementation_prompt}]
            tool_data = self._generate_tool_json_with_retry(
                messages=implementation_messages,
                step_name=f"sib_{sib_index}_tool_generation",
                max_retries=3,
                additional_context={
                    "sib_index": sib_index,
                    "blueprint": blueprint_content,
                    "implementation_prompt": implementation_prompt
                }
            )
            if not tool_data:
                return False, None, f"Failed to generate valid tool JSON for SIB {sib_index} after retries"
            # Post-generation API validation (redundant now that per-function validation is applied)
            # try:
            #     tool_info_pg = tool_data.get("tool_info", {})
            #     tool_code_pg = self._extract_code_from_response(tool_data.get("tool_code", ""))
            #     api_ok, api_refined_tool_info, _ = self._validate_api_and_refine_schema_with_retry(
            #         tool_info=tool_info_pg,
            #         code=tool_code_pg,
            #         context={"step_name": f"sib_{sib_index}_postgen", "sib_index": sib_index},
            #         max_attempts=2
            #     )
            #     if api_ok and api_refined_tool_info:
            #         tool_data["tool_info"] = api_refined_tool_info
            # except Exception:
            #     pass
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
            # Populate original tools information from covered tool indices
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

    def _get_questions_for_sib(self, sib: Dict, tools: List[Dict]) -> List[Dict]:
        questions: List[Dict[str, Any]] = []
        question_set = set()
        for tool_index in sib.get('covered_tools', []):
            if 0 <= tool_index < len(tools):
                tool = tools[tool_index]
                question = tool.get('original_question', '')
                answer = tool.get('original_answer', '')
                if question and answer and question not in question_set:
                    questions.append({'question': question, 'ground_truth': answer, 'tool_index': tool_index})
                    question_set.add(question)
        return questions

    def _optimize_sib(self, cluster_name: str, validated_tool: Dict, sib_tools: List[Dict], sib_index: int, verification_model_name: str = 'gpt-4.1') -> Dict:
        try:
            original_sib_text = ''
            if isinstance(validated_tool, dict):
                original_sib_text = validated_tool.get('sib_text', '') or ''
            tools_for_eval = sib_tools if isinstance(sib_tools, list) else []

            def _suggest_worker(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
                idx, t = item
                try:
                    # Prefer explicit python code if present
                    tool_code = t.get('python_code') or t.get('tool_code') or ''
                    if tool_code:
                        # Clean possible fenced code
                        try:
                            tool_code_clean = self._extract_code_from_response(tool_code)
                        except Exception:
                            tool_code_clean = tool_code
                    else:
                        name = t.get('name', 'unknown_tool')
                        desc = t.get('description', '')
                        tool_code_clean = f"# Tool: {name}\n# Description: {desc}\n"
                    question = t.get('original_question', '')
                    answer = t.get('original_answer', '')
                    optimizer = IterativeLibraryOptimizerAgent(
                        stronger_llm_model=self.model_name,
                        weaker_llm_model_list=[verification_model_name],
                        max_iterations=1,
                        question=question,
                        ground_truth=answer
                    )
                    response = optimizer.optimize_library_with_running_weaker_llm_using_sib(SIB=original_sib_text)
                    prompt = SIB_HELPFULNESS_CHECK_WITH_TOOL_PROMPT.format(
                        original_sib_text=original_sib_text,
                        tool_code=tool_code_clean,
                        weaker_llm_response=response
                    )
                    messages = [{"role": "user", "content": prompt}]
                    response = call_openai_api_multi_turn(
                        messages=messages,
                        model_name=self.model_name,
                    )
                    self._log_llm_call(
                        step_name=f"sib_{sib_index}_text_suggestion_tool{idx}",
                        prompt=prompt,
                        response=response or "",
                        success=bool(response),
                        error_msg="Empty response from LLM" if not response else None,
                        additional_context={"cluster_name": cluster_name, "sib_index": sib_index, "tool_index": idx}
                    )
                    def _extract_json(text: str) -> Dict[str, Any]:
                        import json as _json
                        start, end = "<final_report>", "</final_report>"
                        if start in text and end in text:
                            try:
                                body = text.split(start)[1].split(end)[0].strip()
                                return _json.loads(body)
                            except Exception:
                                return {}
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
                    status = parsed.get('is_SIB_helpful', '') if isinstance(parsed, dict) else ''
                    return {"idx": idx, "suggestion": suggestion or '', "status": status or '', "raw": response or ''}
                except Exception:
                    return {"idx": idx, "suggestion": '', "status": '', "raw": ''}

            indexed_tools: List[Tuple[int, Dict[str, Any]]] = list(enumerate(tools_for_eval, 1))
            per_tool_results: List[Dict[str, Any]] = map_with_progress(
                _suggest_worker,
                indexed_tools,
                num_threads=min(len(indexed_tools), 10),
                pbar=False
            ) if indexed_tools else []

            suggestion_lines: List[str] = []
            for r in per_tool_results:
                s = r.get('suggestion', '')
                if s:
                    suggestion_lines.append(f"- From Tool{r.get('idx', '?')}: {s}")

            rewritten_sib = original_sib_text

            return {"rewritten_sib": rewritten_sib, "suggestions": "\n".join(suggestion_lines)}
        except Exception:
            return {}

    def _process_single_sib_complete(self, args: Tuple) -> Tuple[bool, Optional[Dict], Dict]:
        try:
            sib, tools, cluster_name, verification_model_name = args
            sib_index = sib.get('index', 0)
            # print(f"  🔧 Processing SIB {sib_index} (parallel)...")
            processing_result = {"sib_index": sib_index, "success": False, "error": None, "step": "unknown", "questions_count": 0}
            # Collect covered tools for evaluation
            covered_indices = sib.get('covered_tools', []) or []
            sib_tools_for_eval: List[Dict[str, Any]] = []
            for ti in covered_indices:
                if isinstance(ti, int) and 0 <= ti < len(tools):
                    sib_tools_for_eval.append(tools[ti])
            processing_result["questions_count"] = len(sib_tools_for_eval)
            optimized_text_info = None
            if sib_tools_for_eval:
                print(f"    🔄 Optimizing SIB {sib_index} with {len(sib_tools_for_eval)} covered tools...")
                optimized_text_info = self._optimize_sib(cluster_name, { 'sib_text': sib.get('content', '') }, sib_tools_for_eval, sib_index, verification_model_name)
                if isinstance(optimized_text_info, dict) and optimized_text_info.get('rewritten_sib'):
                    sib['content'] = optimized_text_info['rewritten_sib']
            print("Generating OpenAI tool for this SIB")
            success_gen, sib_tool, error_msg_gen = self._generate_sib_tool(cluster_name, sib, tools)
            if not success_gen:
                processing_result.update({"success": False, "error": error_msg_gen, "step": "tool_generation"})
                # print(f"    ❌ SIB {sib_index} tool generation failed: {error_msg_gen}")
                self._save_llm_logs(f"{cluster_name}_sib_{sib_index}")
                return False, None, processing_result
            validated_tool = sib_tool
            if hasattr(self, 'output_dir') and self.output_dir:
                self._save_sib_as_markdown(sib, validated_tool, self.output_dir, cluster_name)
            processing_result.update({"success": True, "step": "completed_with_text_optimization" if sib_tools_for_eval else "completed_without_optimization"})
            print(f"    ✅ SIB {sib_index} completed")
            self._save_llm_logs(f"{cluster_name}_sib_{sib_index}")
            self._save_final_openai_tools(f"{cluster_name}_sib_{sib_index}", [validated_tool])
            return True, validated_tool, processing_result
        except Exception as e:
            error_msg = f"Error processing SIB {sib.get('index', 0)}: {str(e)}"
            # print(f"    ❌ {error_msg}")
            processing_result = {"sib_index": sib.get('index', 0), "success": False, "error": error_msg, "step": "exception", "questions_count": 0}
            self._save_llm_logs(f"{cluster_name}_sib_{sib.get('index', 0)}")
            return False, None, processing_result

    def process_single_sib(self, cluster_name: str, tools: List[Dict], sib: Dict, output_dir: Path, verification_model_name: str = 'gpt-4.1') -> Tuple[bool, Optional[Dict], Dict[str, Any]]:
        self.output_dir = output_dir
        return self._process_single_sib_complete((sib, tools, cluster_name, verification_model_name))

    def _process_single_sib_for_map(self, args: Tuple[Dict, List[Dict], str]) -> Tuple[bool, Optional[Dict], Dict[str, Any]]:
        sib, tools, cluster_name = args
        return self.process_single_sib(cluster_name, tools, sib, self.output_dir)

    def process_single_cluster(self, cluster_name: str, tools: List[Dict], output_dir: Path) -> ToolAggregationResult:
        self.output_dir = output_dir
        # Don't reset llm_call_logs here - keep accumulating logs across the entire process
        print(f"🔄 Processing cluster: {cluster_name} ({len(tools)} tools)")
        print(f"📁 Output directory: {output_dir}")
        result = ToolAggregationResult(cluster_name=cluster_name, total_tools=len(tools))
        try:
            print(f"📋 Step 1: Blueprint Design")
            success, sibs, error_msg = self._design_blueprint(cluster_name, tools)
            if not success:
                result.success = False
                result.error_message = f"Blueprint design failed: {error_msg}"
                self._save_llm_logs(cluster_name)
                return result
            result.steps_completed.append("blueprint_design")

            print(f"💻 Step 2-4: Processing each SIB individually in parallel")
            sib_tasks: List[Tuple[Dict, List[Dict], str]] = [(sib, tools, cluster_name) for sib in sibs]
            print(f"  🔧 Processing {len(sib_tasks)} SIBs in parallel...")
            map_results = map_with_progress(self._process_single_sib_for_map, sib_tasks, num_threads=min(len(sib_tasks), 10), pbar=False)
            final_tools: List[Dict] = []
            for ok, final_tool, proc in map_results:
                if ok and final_tool:
                    final_tools.append(final_tool)
            result.openai_tools = final_tools
            result.success = len(final_tools) > 0
            # Summary text (mirroring v2 high-level behavior)
            try:
                successful_sibs = sum(1 for ok, _, pr in map_results if ok)
                implementation_summary = f"# Complete SIB Processing Results for {cluster_name}\n\n"
                implementation_summary += f"Generated {len(sibs)} Static Inference Blocks (SIBs)\n"
                # implementation_summary += f"Successfully processed {successful_sibs}/{len(sibs)} SIBs\n"
                implementation_summary += f"Final tools: {len(final_tools)}\n\n"
                result.final_code = implementation_summary
                result.steps_completed.append("implementation_and_validation_completed")
            except Exception:
                pass
            # Save artifacts
            self._save_llm_logs(cluster_name)
            if result.openai_tools:
                self._save_final_openai_tools(cluster_name, result.openai_tools)
                self._save_all_questions(cluster_name, result.openai_tools)
                self._save_solver_performance(cluster_name, result.openai_tools)
            return result
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._save_llm_logs(cluster_name)
            return result


   