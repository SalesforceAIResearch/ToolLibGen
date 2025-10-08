TOOL_EXTRACT_AGENT_AGNTIC_GENERATION_PROMPT = """You are an expert in creating tools.
Below is the single worked example we have:
**Question:**
{question}

**CoT Answer:**
{answer}

**YOUR TASK:**
1. Convert the Chain-of-Thought (CoT) steps into multiple analysis questions. Please note that these CoT steps come from a reasoning LLM, so they feature extensive mental exploration, reflection, and expansion—these reasoning steps must all be retained and converted into subquestions. 
2. For each subquestion, write a general tool that can solve similar sub question. Each tool must be an executable Python function designed to perform one atomic reasoning step.

**Output Format:**
A. Question Generation Phase:
   - Start with <Question Generation> and end with </Question Generation>.
   - In this phase, the format is:
   <Q1>...</Q1>
   <A1>...</A1>
   <Q2>...</Q2>
   <A2>...</A2>
   ...
   - For each question, ensure that the text of the question itself contains enough background information. Avoid statements like "in this experiment" or "in this scenario" that depend on the context of other questions or the original problem. Make sure each question is fully self-contained and answerable on its own text.
   - For each sub-question, only select questions that are informative and valuable. Avoid questions that lack substance, such as those simply asking "Which option would you choose based on current information." Only include questions that encourage deeper thinking or provide meaningful insights.

B. Multiple Tools Construction: For each sub-question, generate one general tools, use:
  - Each tool must be a executable Python function designed to the specific sub-question.
  - Make the Python function as general as possible: any database or tool should work with minimal tweaks (e.g., adding data, changing a predicate). Only provide code for steps that truly benefit from automation; 
  - No code is needed for selecting the final answer; toolkits are for supporting intermediate reasoning only.
  - Try to save knowledge into static variables; don’t just think of the calculation process as a tool, but also think of the knowledge sentences as tools;
  - For example, for the question:
    Can an object's momentum change without experiencing net acceleration?
    No calculation codes are needed, but knowledge sentences are needed for analysis. Even just some hints in the Python function is fine, but you should make sure that the knowledge is saved into static variables.
  - In this phase, the format is:
     <tool1>
     <tag>(comma-separated, fine-grained and descriptive tags, high-level to low-level, e.g.: chemistry, inorganic chemistry, chemical bonding, molecular orbital theory, LCAO method, sigma bonds, interhalogen compound)</tag>
<tool_OpenAI_json_schema>
  {{
    "type": "function",
    "function": 
    {{
      "name": "function_name",
      "description": "A clear, concise description of what the function does. This is crucial for the model to understand when to use this tool.",
      "parameters": 
      {{
        "type": "object",
        "properties": 
        {{
          "param_1_name": 
          {{
            "type": "string",
            "description": "Clear description of what this parameter represents and how it should be used.",
            "enum": ["option1", "option2"],
            "default": "option1"
          }},
          "param_2_name": 
          {{
            "type": "integer",
            "description": "Description of this parameter with any constraints or expected ranges.",
            "minimum": 1,
            "maximum": 100,
            "default": 10
          }},
          "param_3_name": 
          {{
            "type": "array",
            "description": "Description for array-type parameters.",
            "items": {{
              "type": "string"
            }},
            "minItems": 1,
            "maxItems": 10
          }},
          "param_4_name": 
          {{
            "description": "For parameters that can accept multiple types, use oneOf to define each type separately.",
            "oneOf": [
              {{
                "type": "string",
                "description": "When provided as a single string value."
              }},
              {{
                "type": "array",
                "items": {{
                  "type": "string"
                }},
                "description": "When provided as an array of string values."
              }}
            ]
          }},
          "param_5_name": 
          {{
            "type": "object",
            "description": "For nested object parameters, define the internal structure.",
            "properties": {{
              "nested_field": {{
                "type": "string",
                "description": "Description of the nested field."
              }},
              "nested_number": {{
                "type": "number",
                "description": "Description of the nested numeric field."
              }}
            }},
            "required": ["nested_field"],
            "additionalProperties": false
          }}
        }},
        "required": ["param_1_name", "param_2_name",...],
        "additionalProperties": false
      }}
    }}
  }}
</tool_OpenAI_json_schema>
<code>
  ```python\ndef function_name(param_1_name, param_2_name,...):\n[Implement a general function for one sub-question]\nreturn f"explain the result: {{result}}"```
</code>
  </tool1>
  <tool2>...</tool2>...
"""


TOOL_EVALUATION_PROBLEM_SOLVING_PROMPT = """You are an expert problem solver who uses available tools to analyze and solve questions.
**Question to solve:**
{question}

**Available Tools:**
{tools_description}

**Your Task:**
1. Use the available tools systematically to help analyze and solve the question
2. Make sure to actually use the tools in your analysis process
3. Provide clear reasoning for each step
4. Give a final answer based on your tool-assisted analysis

**Final Response Format:**
Start with <analysis>.
- Show tool outputs and explain how they contribute to your analysis
- End with: <final_answer>Your conclusive answer</final_answer>
"""

TOOL_EFFECTIVENESS_EVALUATION_PROMPT = """You are an expert tool effectiveness evaluator.

**Original Question:**
{question}

**Generated Tools:**
{tools}

**Ground Truth Answer:**
{answer}

**Tool Usage Analysis Results:**
{evaluation_results}

**Your Task:**
Analyze whether the generated tools were truly effective for solving the question:

1. **Tool Utility Assessment:**
   - Did the tools provide meaningful assistance in solving the question?
   - Were the tools actually used in the solution process?
   - Did the tools contribute to reaching the correct answer?

2. **Tool Quality Assessment:**
   - Are the tools properly implemented and functional?
   - Do the tools address the core reasoning steps needed for this type of question?
   - Are the tools generalizable to similar questions?

**Response Format:**
EFFECTIVENESS_STATUS: EFFECTIVE/INEFFECTIVE
REASONING: [Detailed explanation of why tools are effective or ineffective]
IMPROVEMENT_SUGGESTIONS: [If ineffective, specific suggestions for improvement]
"""

TOOL_SYNTAX_ERROR_REFINEMENT_PROMPT = """You are an expert Python code fixer.

**Original Code with Syntax Errors:**
{original_code}

**Error Details:**
{error_details}

**Your Task:**
Fix all syntax errors in the code while maintaining the original functionality and logic.

**Requirements:**
1. Fix all syntax errors identified
2. Maintain the same function signature and behavior
3. Ensure the code is properly formatted and executable
4. Do not change the core logic or algorithms

**Response Format:**
<fixed_code>
[The corrected Python code]
</fixed_code>
"""

TOOL_EFFECTIVENESS_REFINEMENT_PROMPT = """You are an expert tool designer and refiner.

**Original Question:**
{question}

**Current Tools:**
{current_tools}

**Effectiveness Analysis:**
{effectiveness_analysis}

**Your Task:**
Based on the effectiveness analysis, refine the tools to make them more useful for solving the question.
**Response Format:**
     <tool1>
     <tag>(comma-separated, fine-grained and descriptive tags, high-level to low-level, e.g.: chemistry, inorganic chemistry, chemical bonding, molecular orbital theory, LCAO method, sigma bonds, interhalogen compound)</tag>
<tool_OpenAI_json_schema>
  {{
    "type": "function",
    "function": 
    {{
      "name": "function_name",
      "description": "A clear, concise description of what the function does. This is crucial for the model to understand when to use this tool.",
      "parameters": 
      {{
        "type": "object",
        "properties": 
        {{
          "param_1_name": 
          {{
            "type": "string",
            "description": "Clear description of what this parameter represents and how it should be used.",
            "enum": ["option1", "option2"],
            "default": "option1"
          }},
          "param_2_name": 
          {{
            "type": "integer",
            "description": "Description of this parameter with any constraints or expected ranges.",
            "minimum": 1,
            "maximum": 100,
            "default": 10
          }},
          "param_3_name": 
          {{
            "type": "array",
            "description": "Description for array-type parameters.",
            "items": {{
              "type": "string"
            }},
            "minItems": 1,
            "maxItems": 10
          }},
          "param_4_name": 
          {{
            "description": "For parameters that can accept multiple types, use oneOf to define each type separately.",
            "oneOf": [
              {{
                "type": "string",
                "description": "When provided as a single string value."
              }},
              {{
                "type": "array",
                "items": {{
                  "type": "string"
                }},
                "description": "When provided as an array of string values."
              }}
            ]
          }},
          "param_5_name": 
          {{
            "type": "object",
            "description": "For nested object parameters, define the internal structure.",
            "properties": {{
              "nested_field": {{
                "type": "string",
                "description": "Description of the nested field."
              }},
              "nested_number": {{
                "type": "number",
                "description": "Description of the nested numeric field."
              }}
            }},
            "required": ["nested_field"],
            "additionalProperties": false
          }}
        }},
        "required": ["param_1_name", "param_2_name",...],
        "additionalProperties": false
      }}
    }}
  }}
</tool_OpenAI_json_schema>
<code>
  ```python\ndef function_name(param_1_name, param_2_name,...):\n[Implement a general function for one sub-question]\nreturn f"explain the result: {{result}}"```
</code>
  </tool1>
  <tool2>...</tool2>...

**Requirements:**
- Maintain the same output format as the original tool generation (including <tool1>, <tool_OpenAI_json_schema>, ... and <code>, etc.)
- Focus on addressing the specific issues identified in the effectiveness analysis
- Ensure all refined tools are executable and well-documented
"""


TOOLKIT_EXTRACT_PROMPT_AGENTIC_VERIFICATION = """
You are an expert chemistry/physics tutor and a good tool user.
Below is the question you need to answer:

Question:
{question}

Reusable tools:
{tools}

YOUR TASK:
1. Answer the question using the tools.
2. Your output must strictly adhere to the structure and format below.
- Start with <answer> and end with </answer>. Your answer should be a mixture of natural language and code. The main body of the thinking, analysis, and answer should still be in natural language.
- Use the tools to answer the question when necessary. The tools are used with <code> ... </code>. You may need to import the necessary libraries in order to make sure that every code snippet in the <code> ... </code> is EXCUTABLE. You code should be like [def tool1_function...][def tool2_function...]...[code snippet for using the tools]. So please make sure that the code in the <code> ... </code> is executable.
- The output of the code should be in the format of <output> ... </output>.
- The final answer of choice should be in the format of <final_choice>$CHOICE_ALPHABET$</final_choice>.
"""

CHECKING_CODE_EXECUTION_PROMPT = """
There are two answers from students for the same question.
Answer 1:
{answer_1}

Answer 2:
{answer_2}

YOUR TASK:
Decide if these two answers share the same semantic meaning. Semantic meaning are only for the scientific meaning. You don't need to consider the format. For numbers, you can consider the answer is the same if the number of the two answers are pretty close.
If they are the same, output "Yes". If they are different, output "No". Please only output one word "Yes" or "No".
"""

REASONING_TEMPLATE_PROMPT = """
Instructions:
You will receive a single question and three different reasoning answers to that question (written in free prose).
Your task is to extract all general, reusable Atomic Reasoning Templates that appear or are implied in all those answers.

Input:
Question: {question}
Reasoning Answers: 
{reasoning}

Now your task is to extract all general, reusable Atomic Reasoning Templates that appear or are implied in those answers.
DEFINITIONS
Atomic Reasoning Template (ART) = the smallest self-contained reasoning unit that
– solves one generic sub-problem,
– exposes a structured interface: JSON input → deterministic rules/algorithm → JSON output,
– can be composed with other templates to build a longer reasoning chain,
– contains no problem-specific labels such as answer choices "A/B/C".

OUTPUT FORMAT (MUST follow exactly)
[
  {{
    "tag": "Including domain tags (e.g., chemistry, physics, etc.), sub-domains (e.g., chemical bonding, molecular orbital theory, etc.), and a very specifc tag (e.g., phase transition, etc.)",
    # The description of the reasoning template. Must follow the OpenAI function calling format.
    "type": "function",
    "function": {{
      "name": "<snake_case_name>",
      "description": "<concise purpose statement>",
      "parameters": {{  … JSON-Schema … }}
    }},
    # The reference implementation of the reasoning template. Must be a valid Python code block.
    "python": "```python\n<reference implementation>\n```"
  }},
  …
]

An example of the output format for other questions:
[
  {{
    "tag": "chemistry, physical chemistry, phase transition",
    "tool": {{
      "type": "function",
      "function": {{
        "name": "phase_transition_analyzer",
        "description": "Predict the phase transition that occurs given initial phase and qualitative ΔT/ΔP.",
        "parameters": {{
          "type": "object",
          "properties": {{
            "initial_phase":      {{ "type": "string", "enum": ["solid","liquid","gas"] }},
            "temperature_change": {{ "type": "string", "enum": ["increase","decrease","none"] }},
            "pressure_change":    {{ "type": "string", "enum": ["increase","decrease","none"] }}
          }},
          "required": ["initial_phase","temperature_change","pressure_change"]
        }}
      }}
    }},
    "python": "```python\n"
              "def phase_transition_analyzer(initial_phase, temperature_change, pressure_change):\n"
              "    if initial_phase=='solid' and temperature_change=='increase':\n"
              "        return {{'predicted_transition':'melting','explanation':'Heating past the melting point.'}}\n"
              "    # … further rules …\n"
              "```"
  }}
  ...
]

REQUIREMENTS
One ART per distinct sub-reasoning you detect (avoid redundancy).
Inputs must be minimal yet sufficient; outputs must include both answer and one-line explanation.
Python code may use simple if-else or formulas—keep dependencies minimal.
Use the example templates below as a style and granularity guide; do NOT copy them verbatim unless truly the same logic applies.
EXAMPLE TEMPLATES (FOR STYLE ONLY)


DELIVERABLE
Return ONLY the JSON array described in OUTPUT FORMAT. Please make sure that the generated reasoning templates are specific enough for this question and also generalizable to other questions.
"""

REASONING_TEMPLATE_PROMPT_V2 = """
Instructions:
You will receive a single question and a solution to that question (written in free prose).
Your task is to think about what are the general, reusable Atomic Analysis Tools that can be used in solving the similar questions.

Input:
Question: 
{question}

Solution:
{solution}

Now your task is to extract all general, reusable Atomic Analysis Tools that can be used in solving the similar questions.

DEFINITIONS
Atomic Analysis Tools (AAT) = the smallest self-contained reasoning unit that
– solves one generic sub-problem for this type of questions,
– can be used in solving the similar questions,
– contains no problem-specific labels such as answer choices "A/B/C", and should not for extracting answer choices. We don't need tools to consider which choice to choose after getting some analysis results. We just need tools to analyze the question and get these analysis results.
– outputs can include both answer and some natural language explanation that can be used by the other LLMs to analyze the question.

OUTPUT FORMAT (MUST follow exactly)
[
  {{
    "tag": "Including domain tags (e.g., chemistry, physics, etc.), multiple specific sub-domains tags (e.g., chemical bonding, molecular orbital theory, etc.), and multiple specific functionalities tags (e.g., phase transition, etc.)",
    # The description of the analysis tool. Must follow the OpenAI function calling format.
    "type": "function",
    "function": {{
      "name": "<snake_case_name>",
      "description": "<concise purpose statement>",
      "parameters": {{  … JSON-Schema … }}
    }},
    # The implementation of the analysis tool. Must be a valid Python code block.
    "python": "```python\n<implementation>\n```"
  }},
  …
]

An example of the output format for other questions:
[
  {{
    "tag": "chemistry, physical chemistry, phase transition",
    "tool": {{
      "type": "function",
      "function": {{
        "name": "phase_transition_analyzer",
        "description": "Predict the phase transition that occurs given initial phase and qualitative ΔT/ΔP.",
        "parameters": {{
          "type": "object",
          "properties": {{
            "initial_phase":      {{ "type": "string", "enum": ["solid","liquid","gas"] }},
            "temperature_change": {{ "type": "string", "enum": ["increase","decrease","none"] }},
            "pressure_change":    {{ "type": "string", "enum": ["increase","decrease","none"] }}
          }},
          "required": ["initial_phase","temperature_change","pressure_change"]
        }}
      }}
    }},
    "python": "```python\n"
              "def phase_transition_analyzer(initial_phase, temperature_change, pressure_change):\n"
              "    if initial_phase=='solid' and temperature_change=='increase':\n"
              "        return {{'predicted_transition':'melting','explanation':'Heating past the melting point.'}}\n"
              "    # … further rules …\n"
              "```"
  }}
  ...
]


DELIVERABLE
Return ONLY the JSON array described in OUTPUT FORMAT. Please make sure that the generated analysis tools are specific enough for this question and also generalizable to other questions.
"""


CHECKING_USEFULNESS_PROMPT = """
Please check which of the following reasoning templates is useful for analyzing the question.

Question:
{question}

Reasoning Templates:
{reasoning_templates}

Your output should be a list of the index of the reasoning templates that are useful for analyzing the question. The output format should be like {"template_index": "usefulness_score"}. The usefulness score is a number between 0 and 1.
"""



TOOL_RESUABLE_CHECKING_PROMPT = """
Please solve this question: 
{question}. 
You must use the provided tools to solve the question. Clearly indicate how you use the tools in your reasoning.
Your last line should be your final answer and start with 'Final Answer: YOUR_CHOICE'. Only output your choice alphabetically.
"""

FUNCTIONAL_ERROR_RETRY_PROMPT = """
The previous tool generation attempt failed during functional testing. 

Original Question:
{question}

Original Solution:
{solution}

Previous Generated Tools:
{previous_tools}

Functional Test Error Details:
{error_details}

Failed Test Sessions:
{failed_sessions}

Based on the above information:
1. The tools generated previously had functional issues
2. The test sessions show how the tools failed when being used
3. You need to fix these issues and generate better, more reliable tools

Please analyze the failures and generate improved Atomic Analysis Tools that:
- Address the specific issues found in the test sessions
- Are more robust and handle edge cases better
- Have clearer function interfaces and better error handling
- Can successfully solve similar questions

OUTPUT FORMAT (MUST follow exactly)
[
  {{
    "tag": "Including domain tags (e.g., chemistry, physics, etc.), multiple specific sub-domains tags (e.g., chemical bonding, molecular orbital theory, etc.), and multiple specific functionalities tags (e.g., phase transition, etc.)",
    # The description of the analysis tool. Must follow the OpenAI function calling format.
    "type": "function",
    "function": {{
      "name": "<snake_case_name>",
      "description": "<concise purpose statement>",
      "parameters": {{  … JSON-Schema … }}
    }},
    # The implementation of the analysis tool. Must be a valid Python code block.
    "python": "```python\n<implementation>\n```"
  }},
  …
]

An example of the output format for other questions:
[
  {{
    "tag": "chemistry, physical chemistry, phase transition",
    "tool": {{
      "type": "function",
      "function": {{
        "name": "phase_transition_analyzer",
        "description": "Predict the phase transition that occurs given initial phase and qualitative ΔT/ΔP.",
        "parameters": {{
          "type": "object",
          "properties": {{
            "initial_phase":      {{ "type": "string", "enum": ["solid","liquid","gas"] }},
            "temperature_change": {{ "type": "string", "enum": ["increase","decrease","none"] }},
            "pressure_change":    {{ "type": "string", "enum": ["increase","decrease","none"] }}
          }},
          "required": ["initial_phase","temperature_change","pressure_change"]
        }}
      }}
    }},
    "python": "```python\n"
              "def phase_transition_analyzer(initial_phase, temperature_change, pressure_change):\n"
              "    if initial_phase=='solid' and temperature_change=='increase':\n"
              "        return {{'predicted_transition':'melting','explanation':'Heating past the melting point.'}}\n"
              "    # … further rules …\n"
              "```"
  }}
  ...
]


DELIVERABLE
Return ONLY the JSON array described in OUTPUT FORMAT. Please make sure that the generated analysis tools are specific enough for this question and also generalizable to other questions.
Focus on creating tools that will pass the functional tests by being more accurate, robust, and user-friendly.
"""


CLUSTERING_INITIAL_PROMPT = """
You need to aggregate the following tools into a hierarchy. You do not need to set each tool in different nodes. In contrast, you should set the tools that are similar to each other in the same node.
Please make sure that the hierarchy should not be too shallow. Deep hierarchy and detailed classification are preferred. The depth of the hierarchy should be at least {cluster_depth}.
The function_name of tools should NOT be the last layer leaf node of the hierarchy. Do not need to include the function_name in the hierarchy. After we finalize the hierarchy, we will add the function_name to the leaf node.
tools:
{tool_lst}

Your output should be a JSON object with a hierarchical structure.
{{
  "clusters": [ {{id, level, parent, children}} … ],
  }}
example:
{{
  "clusters": [
    {{
      "id": "c_root",
      "level": 0,
      "parent": null,
      "children": ["c_math", "c_utils",...]
    }},
    {{
      "id": "c_math",
      "level": 1,
      "parent": "c_root",
      "children": ["c_arith", "c_stat",...]
    }},
    {{
      "id": "c_linear_algebra",
      "level": 2,
      "parent": "c_math",
      "children": ["c_matrix","c_singular_value_decomposition",...],
    }},
    {{
      "id": "c_stat",
      "level": 2,
      "parent": "c_math",
      "children": ["c_stat_mean", "c_stat_t_test",...],
    }},
  ]
}}

DELIVERABLE
Return ONLY the JSON array described in OUTPUT FORMAT. No any other text. No ```json.
"""

CLUSTERING_UPDATE_PROMPT = """
Given the current hierarchy, you need to update it based on the following tools.
Please make sure that the hierarchy should not be too shallow. Deep hierarchy and detailed classification are preferred. 
The function_name of tools should NOT be the last layer leaf node of the hierarchy. Do not need to include the function_name in the hierarchy. After we finalize the hierarchy, we will add the function_name to the leaf node.
If a tool cannot be "assigned" to any node of the current hierarchy, you should create a new node for it. The "assigned" means a perfect match of the tool to the node.

tools:
{tool_lst}

current hierarchy:
{current_hierarchy}

Your output should be a JSON object with the same hierarchical structure as the current hierarchy.
{{
  "clusters": [ {{id, level, parent, children}} … ],
}}
"""

CLUSTERING_UPDATE_OPERATIONS_PROMPT = """
Given the current hierarchy and new tools to integrate, generate specific operations to update the hierarchy incrementally instead of rewriting the entire structure.

Analyze the new tools and determine what changes are needed:
1. ADD_NODE: Create new clusters for tools that don't fit existing categories
2. MODIFY_NODE: Update existing cluster properties if needed
3. No operations if tools fit perfectly into existing leaf nodes

Current hierarchy:
{current_hierarchy}

New tools to integrate:
{tool_lst}

Your output should be a JSON object with specific operations:
{{
  "operations": [
    {{
      "action": "ADD_NODE",
      "node_id": "new_cluster_id",
      "level": 2,
      "parent": "parent_cluster_id",
      "description": "Brief description of what this cluster represents",
      "reasoning": "Why this new cluster is needed for the new tools"
    }},
    {{
      "action": "MODIFY_NODE", 
      "node_id": "existing_cluster_id",
      "changes": {{
        "add_children": ["new_child_id1", "new_child_id2"]
      }},
      "reasoning": "Why this modification is needed"
    }}
  ]
}}

Guidelines:
- Only create new nodes when new tools represent significantly different functionality
- Prefer adding to existing leaf nodes when tools are similar enough
- Maintain proper parent-child relationships and level consistency
- Keep the hierarchy depth appropriate (not too shallow, not too deep)
- Provide clear reasoning for each operation

DELIVERABLE
Return ONLY the JSON array described in OUTPUT FORMAT. No any other text. No ```json.
"""

TOOL_ASSIGNMENT_PROMPT = """
Given a hierarchical clustering structure and a list of tools, you need to assign each tool to the most appropriate leaf node (deepest level cluster) in the hierarchy.

For each tool, analyze its functionality, description, and characteristics, then determine which leaf cluster it best fits into.

Hierarchy:
{hierarchy}

Tools to assign:
{tools}

Your output should be a JSON object that maps each tool to its assigned cluster:
{{
  "assignments": [
    {{
      "tool_index": 0,
      "tool_name": "tool_name_here",
      "assigned_cluster_id": "cluster_id_here",
    }},
    ...
  ]
}}

Please ensure:
1. Each tool is assigned to exactly one leaf node (cluster with no children)
2. The assignment is based on the tool's functionality and purpose
3. Provide brief reasoning for each assignment
4. Use the exact cluster IDs from the hierarchy

DELIVERABLE
Return ONLY the JSON object described in OUTPUT FORMAT. No any other text. No ```json.
"""

# TOOL_ASSIGNMENT_PROMPT = """
# Given a hierarchical clustering structure and a list of tools, you need to assign each tool to the most appropriate leaf node (deepest level cluster) in the hierarchy.

# For each tool, analyze its functionality, description, and characteristics, then determine which leaf cluster it best fits into.

# Tags:
# {hierarchy}

# Tools to assign:
# {tools}

# Please ensure:
# 1. Each tool is assigned to exactly one or multiple leaf node (cluster with no children)
# 2. The assignment is based on the tool's functionality and purpose
# 3. Provide brief reasoning for each assignment
# 4. Use the exact cluster IDs from the hierarchy

# DELIVERABLE
# Return ONLY return the cluster names, split by comma. No any other text.
# """

JSON_REPAIR_PROMPT = """
The following response contains valid information but is not in proper JSON format. Please convert it to valid JSON format that can be parsed.

Original Response:
{response}

Error message:
{error_message}

Requirements:
1. Extract all the meaningful information from the response
2. Convert it to valid JSON format
3. Maintain the same structure and content
4. Ensure all JSON syntax is correct (proper quotes, brackets, commas, etc.)
5. Return ONLY the corrected JSON, no additional text
DELIVERABLE
Return ONLY the JSON object described in OUTPUT FORMAT. No any other text. No ```json.
"""


AGGREGATE_TOOLS_PROMPT = """
Tool code:
{tool_code}

Instruction:
You are a senior Python library architect with deep knowledge of software-engineering best-practices and scientific-computing design patterns.

Your First Task: Feasibility Analysis
Before any refactoring, you must first analyze the provided functions to determine if they belong together in a single class.

Condition for Merging:
Primary Rule for Standalone Functions: If the provided code contains any standalone functions (i.e., functions not within a class), these functions must be encapsulated into one or more classes. They should be grouped logically based on a shared concept, common data they operate on, or their role in a processing pipeline. The absolute requirement is that no standalone functions exist in the final output.

Rule for Inter-Class Merging: This rule applies if the code already consists multiple classes. You should merge existing classes if they share a clear, common concept, operate on a consistent set of underlying data, or represent different facets of a single, coherent entity. The goal of merging is to encapsulate state and behavior more effectively, thereby reducing overall complexity.

Condition for Stopping:
This condition applies only if the initial code consists exclusively of classes, with no standalone functions. If these classes are fundamentally unrelated and lack a shared context (e.g., a class to calculate financial interest, a class to process image data, and a class to query a weather API), then merging them would feel artificial and would not genuinely improve the design. In this specific case, merging is inappropriate, and your entire output must be the exact string STOP_MERGING and nothing else.


If and only if the codes are suitable for merging, you will proceed with the refactoring task below.
Your Second Task: Refactoring
When you receive a list of loose functions or a coarse-grained class, your job is to refactor them into a clean, ergonomic, production-ready Python class that:

Preserves every original public behaviour exactly (no loss of any functionality from the original tool code).
Based on your new class, the user can easily get the same result as all the original tool code.
Minimises cognitive load for end-users: sensible defaults, logical grouping, consistent parameter names, clear docstrings, and properties for frequently accessed derived quantities.
Add necessary external dependencies if needed.
When refactoring, you may:

Merge related functions into a single method or @property.
Split monolithic functions into internal helpers if this benefits clarity.
Introduce small dataclasses or enums if it materially improves type-safety or readability.
Your deliverable must be a single, runnable Python script that may contain:
A) The new class (and any helper dataclasses).
B) A __main__ demo block showing:
1. legacy_map dict that maps every original function / method name to its new call path (e.g. "calc_Re" -> "FluidSystem.Re").
2. Construction of the class with representative parameters.
3. One-line example for each legacy feature via the new API.
4. An assert or print that the outputs match the old implementation (if deterministic).

Think step-by-step before you start coding:
1. First list all input symbols and group them logically.
2. Draft the public API of the new class (constructor signature, main methods, properties).
3. Decide what needs caching.
4. Only then write the final code.

Your response must contain **only** the final Python code block. You can start with ```python and end with ```.
"""


# TOP_DOWN_BLUEPRINT_DESIGN_PROMPT = """
# All tool code name list:
# {tool_code_name_list}


# Persona: You are a Senior Python Library Architect, a master of API design and software engineering best-practices. Your thinking is systematic, and your primary skill is creating clear, robust plans for others to follow.

# Your Mission:
# Given hundreds of Python code-base functions that are in the same topic, **you must deliver a comprehensive "Refactoring Blueprint"**. The blueprint is a design document – *not* runnable code – that tells a future. The requirements are:
# 1. High Cohesion / Low Coupling  
# 2. "Atomic-Primitive → Composite → Facade" layered architecture
# 3. When implementing, it is important to ensure that every function is taken into consideration and integrated into this class. At the same time, it's also necessary to make sure that each function is not simply copied into the class, but rather that its most fundamental, atomic parts are merged into it.

# Output Format:
# Proposed Atomic Primitive Layer:
# ...
# Proposed Composite Layer:
# ...
# Proposed Facade Layer:
# ...

# Deliverable:
# Your final output is the complete Refactoring Blueprint as a single block of text, formatted clearly with Markdown. Do not include any Python implementation code.
# """

# TOP_DOWN_CODE_IMPLEMENTATION_DESIGN_PROMPT = """
# All Python function tool code:
# {tool_code_list}

# Blueprint:
# {blueprint}

# Your task is to implement the blueprint into a Python class. The requirements are:
# 1. High Cohesion / Low Coupling  
# 2. "Atomic-Primitive → Composite → Facade" layered architecture
# 3. When implementing, it is important to ensure that every function is taken into consideration and integrated into this class. At the same time, it's also necessary to make sure that each function is not simply copied into the class, but rather that its most fundamental, atomic parts are merged into it.


# Deliverable:
# Your response must contain **only** the final Python code block. You can start with ```python and end with ```.
# """

# TOP_DOWN_FUNCTION_CHECK_DESIGN_PROMPT = """
# Implemented Python class code:
# {class_code}

# Original tool code name:
# {tool_code}

# Now implement the given original tool code with the implemented Python class. The requirements are:
# 1. share the exact same functionality as the original tool code, e.g., the input parameters and output parameters should be the same.
# 2. You must use the implemented Python class to implement the original tool code.

# After implemented the code, think about if the implemented python code with generated class fully used the Python class. The fully used means that the implemented python code should just be a straight call to the Python class. Only when the implemented python code with generated class fully used the Python class and the code is not directly copied from the original tool code, you should output "Use the Class: YES". Otherwise, you should output "Use the Class: NO".

# Please output the review result in the following format:
# The implemented python code with class:
# ```python\n...```
# Review Result:...
# Use the Class: YES/NO
# """


# TOP_DOWN_RE_DESIGN_PROMPT = """
# The original class code:
# {class_code}

# The original tool code:
# {tool_code}

# The implemented python code with generated class:
# {class_code}

# After reviewing, we find that the implemented python code with generated class is not fully used the Python class.

# Your task is to redesign original class code. Please make sure that the implemented python code can be easily applied with a straight forward calling. Make sure to rewrite the whole code of the original class code.

# Deliverable:
# Your response must contain **only** the final Python code block. You can start with ```python and end with ```. Make sure to rewrite the whole code because the written part will be directly copied as the final class code without any modification.
# """


# BLUEPRINT_DESIGN_PROMPT = """
# Persona:
# You are a Senior Python Library Architect, a master of API design and software engineering best-practices. Your thinking is systematic, and you excel at creating clear, robust plans by analyzing existing codebases.

# Your Mission:
# You are given a list of Python functions, all related to a single topic. Your mission is to analyze their names, parameters, and internal logic to design a comprehensive **Refactoring Blueprint**. This blueprint must organize the scattered functions into a cohesive, layered class structure.

# **Input: All Python Function Tool Code Name List:**
# {tool_code_name_list}

# **Core Requirements:**
# 1.  **High Cohesion / Low Coupling:** Group related data and behavior together.
# 2.  **Layered Architecture:** You must design the class(es) following an "Atomic-Primitive → Composite → Facade" architecture.
#     * **Atomic Primitive Layer:** Internal, private methods (`_...`) for the smallest, reusable logic.
#     * **Composite Layer:** Methods combining atomic primitives into more complex operations.
#     * **Facade Layer:** The clean, high-level public API for the end-user.
# 3. **All tools are about the sub domain {domain}, so please only use the term within this domain. Never use any name like PhyToolKit as physics is a much broader area than the current domain.**
# 4. **Never copy the tool function name from All Python Function Tool Code Name List.**

# **Blueprint Output Format:**
# Structure your blueprint in Markdown, detailing the proposed class(es), state management (`__init__`), and the three architectural layers with explanations for each method and the original functions it encapsulates.

# **Deliverable:**
# Your final output is the complete Refactoring Blueprint. Do not include any Python implementation code.
# """

# # --------------------------------------------------------------------------------------
# # STAGE 2: Code Implementation (代码实现) - (无变化)
# # --------------------------------------------------------------------------------------
# CODE_IMPLEMENTATION_PROMPT = """
# Persona:
# You are a Senior Python Implementation Engineer, an expert at writing clean, efficient, and production-ready code by meticulously following architectural specifications.

# **Your Mission:**
# Translate the provided **Blueprint** into a high-quality, runnable Python class, using the **All Python Function Tool Code** as a reference for the implementation logic.

# **Input 1: All Python Function Tool Code:**
# ```python
# {tool_code_list}
# ```

# **Input 2: The Blueprint:**
# ```markdown
# {blueprint}
# ```

# **Implementation Rules:**
# 1.  **Strict Blueprint Adherence:** Implement the layered architecture exactly as planned.
# 2.  **Integrate All Logic:** Ensure every original function's logic is refactored into the new class structure.
# 3.  **Production Quality:** Include docstrings, type hints, and follow PEP 8.
# 4.  **All tools are about the sub domain {domain}, so please only use the term within this domain. Never use any name like PhyToolKit as physics is a much broader area than the current domain.**
# 5. **Never copy the tool function name from All Python Function Tool Code.**
# 6. **Create a Tool Registry:** At the end of the script, after all the class definition, you must create a "Tool Registry".
#     * The module-level list is named AVAILABLE_TOOLS.
#     * This list must be populated with references to the public methods from the class instance you created. This makes them directly callable.

# **Deliverable:**
# Your response must contain **only** the final Python code block of the complete class.
# """


# BLUEPRINT_DESIGN_PROMPT = """
# Persona:
# You are a Senior Python Library Architect; you transform fragmented helper functions into coherent, maintainable knowledge-libraries by applying rigorous knowledge-engineering practice.

# Your Mission:
# You will receive a list of Python functions that all belong to the same sub-domain.  
# Design a **Refactoring Blueprint** that reorganises those tools into a catalogue of **Static Inference Blocks (SIBs).**

# Static Inference Block (SIB) Definition
# A SIB is a reusable knowledge capsule that  
# • accepts a well-defined set of input facts (pre-conditions)  
# • instantly infers deterministic outputs (formulae, numbers, long–form explanations)  
# • returns a multi-paragraph explanation string that follows the standard template shown below  
# • groups as many original tools as logically coherent—functionality must remain complete and loss-less.


# **Input: All Python Function Tool Code Name and discription List:**
# {tool_code_name_list}

# **Core Requirements**
# 1. High Cohesion / Low Coupling
#    • Tools that share the similar input set and produce mutually supportive results **must** be merged into the same SIB.  
#    • Functionality of every original tool must be preserved somewhere in the catalogue.
# 2. Mandatory SIB Metadata (document for *every* SIB):
# [SIB]<Insert detailed title here>
# [Description]
# (Provide a concise, high-level summary in plain language describing what this SIB is and what problem it solves.)
# [Known Inputs]
# (List all known preconditions, variables, or parameters required to perform this reasoning, and clarify their meaning.)
# [Assumptions]
# (Explicitly list all idealized assumptions or constraints that this reasoning pattern relies on to be valid.)
# [Applicability & Scope]
# (Describe the specific scenarios and scope boundaries where this reasoning pattern is applicable.)
# [Derivation Steps]
# (Show, step-by-step and clearly, the complete logical chain for deriving the final result from the inputs and assumptions.)
# [Deterministic Outputs]
# **List exhaustively every quantity, formula, or explanatory statement that becomes unambiguously known once the *Known Inputs* and *Assumptions* are satisfied.  Nothing that can be deduced may be omitted.**  
# [Further Inference]  
# (Describe valuable next steps that would require *additional* information beyond the current Known Inputs, and outline how to proceed.)
# [Common Pitfall]
# (List the common mistakes, misunderstandings, or traps that beginners or users are likely to encounter when applying this pattern.)
# [Covered Tools]
# (List all the tools index that are covered by this SIB without any other text. E.g., 1, 2, 3, ...)
# 3. Never create umbrella names such as *PhysicsToolkit*. Do **not** copy any original tool function names verbatim.
# 4. Implementation Planning Note – Each SIB, when implemented later, must expose exactly one public facade function (top-level, non-underscore). Any helper functions should be underscore-prefixed and kept internal.

# **Blueprint Output Format:**
# <SIB>
# [SIB1_name]:...
# [Description]...
# [Known Inputs]...
# ...
# </SIB>
# <SIB>
# [SIB2_name]: ...
# </SIB>
# ...

# **Deliverable:**
# Your final output is the complete Refactoring Blueprint in markdown format. Do not include any Python implementation code.
# """


BLUEPRINT_DESIGN_PROMPT = """
Persona:
You are a Senior Python Library Architect; you transform fragmented helper functions into coherent, maintainable knowledge-libraries by applying rigorous knowledge-engineering practice.

Your Mission:
You will receive a list of Python functions as discrete tools that all belong to the same sub-domain.  
Design a **Refactoring Blueprint** that reorganises those tools into a catalogue of **Static Inference Blocks (SIBs).**

Static Inference Block (SIB) Definition
A SIB is a reusable knowledge capsule that is composed of one or multiple Python classes to construct one specifc problem-solving scenario and multiple public function to wrap up all the functionality of the collected discrete tools. The def function can accept multiple parameters and return a multi-paragraph explanation string that follows the standard template shown below.
It should follow the following requirements:
• accepts a well-defined set of input facts (pre-conditions)  
• instantly infers deterministic outputs (formulae, numbers, long–form explanations)
• groups as many original tools as logically coherent—functionality must remain complete and loss-less.
Here is an example of SIB:
Given tool code 1:
def find_root_multiplicity(coefficients, root):
    # ... calculates the multiplicity of a given root for a polynomial
Given tool code 2:
def find_polynomial_tangent_slope(coefficients, point):
    # ... calculates the slope of the tangent line (1st derivative) at a point

The SIB should be composed of the following one class and one public function:
# One class to build a determined scenario. Given the coefficients,
# we can easily derive properties like roots, derivatives, etc.
class _PolynomialAnalyzer:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        ...
    def evaluate(self, point, derivative_order=0):
        # ... evaluates the polynomial or its derivative at a point

    def get_root_multiplicity(self, root):
        # ... finds the multiplicity of a root

# A public function to wrap up all the functionality of the collected discrete tools.
# Using this public function, the LLM can decide which tool to call and obtain the corresponding result by providing different input parameters. 
def analyze_polynomial(coefficients, point=None, derivative_order=0, root_to_check=None):
    analyzer = _PolynomialAnalyzer(coefficients)
    summary_lines = [f"Analysis for polynomial P(x) with coefficients: {{coefficients}}:"]

    if point is not None:
        value = analyzer.evaluate(point, derivative_order)
        ...
        if derivative_order == 0:
            summary_lines.append(f"- Value at point x={{point}} is: {{value}}")
        else:
            summary_lines.append(f"- Value of the {{derivative_order}}-order "
                                 f"derivative at x={{point}} is: {{value}}")
    if root_to_check is not None:
        ...
    return "\n".join(summary_lines)

Another example of SIB:
Given tool code 1:
def calculate_potential_energy(mass, height):
    Gravity = 9.81
    return mass * Gravity * height

Given tool code 2:
def get_gravity(planet_name):
    if planet_name == "Earth":
        return 9.81
    elif planet_name == "Mars":
        return 3.71
    else:
        return 0

The SIB should be composed of the following one class and one public function:
class _PlanetaryPhysics:
    def __init__(self, planet_name):
        self.planet_name = planet_name
        _GRAVITY_MAP = {{
            "Earth": 9.81,
            "Mars": 3.71
        }}
        self.gravity = _GRAVITY_MAP[planet_name]
    def calculate_potential_energy(self, mass, height):
        return mass * self.gravity * height
    def get_gravity(self):
        return self.gravity

def calculate_potential_energy(mass, height, planet_name="Earth"):
    planetary_physics = _PlanetaryPhysics(planet_name)
    return planetary_physics.calculate_potential_energy(mass, height)


**Core Requirements**
1. High Cohesion / Low Coupling: Group functions that operate on the same core data or represent the same conceptual scenario. SIBs should be self-contained and independent.
2. Scenario-Focused Classes: Each [SIB Class] should just model one specific, concrete scenario (e.g., _ProjectileMotion, _GasContainerState). The class is initialized with the base facts of the scenario, from which all other properties can be derived. But the scenario should be specific, not too general.
3. Lossless Abstraction: The public execute function must provide a way to access the full functionality of all original tools it covers. Use optional parameters to control which specific calculations are performed. But the parameters should be simple to understand and use.
4. Descriptive Naming: SIB titles and class names must be explicit and descriptive. Avoid vague, generic umbrella names such as PhysicsToolkit or MathHelpers.
5. Blueprint, Not Code: Your output is a design document (the blueprint), not the final Python code.
6. All parameters should be simple to understand and use. No any hard-encoded condition in the parameters, e.g., "if parameter == a, then use function1, else use function2,". In this case, you need to create more than one public function to cover all the scenarios rather than using hard-encoded condition.


Output Format: The Refactoring Blueprint
Your entire output must be a single markdown document. This document will contain multiple SIB descriptions (depending on how many scenario you can infer from the current tools), each strictly adhering to the following metadata template.
Mandatory SIB Metadata (document for *every* SIB):
[SIB]<Insert detailed title here>
[Description]
(Provide a concise, high-level summary in plain language describing what this SIB is and what problem it solves.)
[SIB Class Description]
(Design one or more classes. Each class should model one scenario where, given a few initial inputs, many other properties can be deterministically calculated. Describe the purpose of each class and what initial state it will be constructed with. Output every class's init function, inner function name, description and parameters.)
[Public Function Description]
(Describe one or more public functions. Detail which parameters it should accept to ensure all scenarios and functionalities of the covered tools are accessible.
[Covered Tools]
(List all the tool indices that are covered by this SIB, separated by commas. E.g., 1, 2, 3)

**Blueprint Output Format:**
<SIB>
[SIB1_name]:...
[Description]...
[SIB Class Description]...
...
</SIB>
<SIB>
[SIB2_name]: ...
</SIB>
...


**Input: All Python Function Tool Code Name and discription List:**
{tool_code_name_list}


**Deliverable:**
Now start to write the blueprint. Your final output is the complete Refactoring Blueprint in markdown format. Do not include any Python implementation code. Exactly follow the format:
<SIB>
[SIB1_name]:...
[Description]...
[SIB Class Description]...
...
</SIB>
<SIB>
[SIB2_name]: ...
</SIB>
...
"""

SIB_HELPFULNESS_CHECK_PROMPT = """
You are a senior knowledge engineer. Given a SIB and ONE QA pair, check if the SIB is helpful for answering the question.
Given a Static Inference Block (SIB) and a corresponding Question/Answer pair, you must determine if the SIB is functionally capable of answering the question. Your analysis must be based on a strict evaluation of the SIB's public interface and its expected output.

Core Evaluation Criteria:
1. Input Parameter Mapping:
Examine the [Public Function Description] within the SIB. Can the essential parameters required by this function be extracted directly from the Question text?
2. Output Verification:
Assume you successfully mapped the parameters from the Question to the SIB's public function. Would the execution of this function produce a result that logically and factually aligns with the Ground Truth answer?


SIB:
----- SIB START -----
{original_sib_text}
----- SIB END -----

Question:
{q_text}

Ground Truth:
{gt_text}


Output only in the following XML tag:
<final_report>
{{
  "is_SIB_helpful": "PASS" or "NEED_PATCHING",
  "reason": "Brief reason",
  "modification_suggestions": "Precise suggestions to improve this SIB for the question, including add some inner functions in the class or add specific parameter in the input or need to make the case more general."
}}
</final_report>
"""

SIB_HELPFULNESS_CHECK_WITH_TOOL_PROMPT = """
You are a senior knowledge engineer. Given a Static Inference Block (SIB) and a tool, check if the SIB covers the functionality of the tool.
Your analysis must be based on a strict evaluation of the SIB's public interface and its expected output.

Core Evaluation Criteria:
Input Parameter Mapping:
Examine the [Public Function Description] within the SIB. Can the essential parameters required by this function be extracted directly from the tool text? In other words, is the current tool code is a special case of this SIB with specific parameters?

SIB:
----- SIB START -----
{original_sib_text}
----- SIB END -----

Tools:
----- TOOLS START -----
{tool_code}
----- TOOLS END -----


Output only in the following XML tag:
<final_report>
{{
  "is_SIB_helpful": "PASS" or "NEED_PATCHING",
  "reason": "Brief reason",
  "modification_suggestions": "Precise suggestions to improve this SIB for the tool, including add some inner functions in the class if current tool functionality is missing or add specific parameter in the input to make sure that the tool can be correctly called."
}}
</final_report>
"""


SIB_GENERALIZATION_PROMPT = """
SIB:
----- SIB START -----
{original_sib_text}
----- SIB END -----

**Task**: Given the SIBs, please update all SIBs to more general ones for this scenario by:
1. Embedding more knowledge and with static variables or dictionary for this scenario for each SIB.
2. Making the code more general to each scenario, including more dimensions, more static variables.
3. Make sure that each SIB is not too general to the original scenario. For example, if the original SIB is solve quadratic equation, a more general SIB should be solve polynomial equation, but not to number theory.
4. Make sure that each SIB does not have any overlap with other SIBs, especially for the public functions.

Output only the fully rewritten SIB inside this tag. Follow the format of the original SIB, e.g., multiple classes and public functions.
Output format:
<REWRITTEN_SIB>
<SIB>
[SIB1_name]:...
[Description]...
[SIB Class Description]...
...
</SIB>
<SIB>
[SIB2_name]: ...
</SIB>
...
</REWRITTEN_SIB>
"""


# BLUEPRINT_TOOLS_CORRESPONDENCE_PROMPT = """
# The blueprint you generate should include all Tool Code Names and descriptions. Please check the correspondence between tool_id and the SIB index (meaning that a tool can be represented as a subset of a specific Static Inference Block), and output a Python dict where the key is the tool id and the value is the SIB index.
# If you believe a tool does not have an appropriate SIB to map to (i.e., the tool cannot be represented by any SIB), put "no SIB" as the value for that tool_id.
# Include the output within ```python and ```.
# """

BLUEPRINT_TOOLS_CORRESPONDENCE_PROMPT = """
Your goal is to determine if a tool can be mapped to a given Static Inference Block (SIB). A valid mapping exists only if the tool represents a specific instance or a subset of the SIB's problem-solving scenario.

To verify this, you must check for three conditions. The mapping is valid **if and only if all three conditions are met**:

1.  **Assumption Consistency**: The tool's implicit model and underlying logic **must not violate** any of the SIB's core `[Assumptions]` or exceed its `[Applicability & Scope]`. This is the most important rule.
2.  **Input Compatibility**: The tool's required inputs must be a logical subset or a specific case of the SIB's `[Known Inputs]`. Pay attention to both the physical meaning and the data structure (e.g., a single value vs. a list of values).
3.  **Output Inclusion**: The tool's output must be one of the possible results described in the SIB's `[Deterministic Outputs]`.

**Example of a failed mapping**: A tool that accepts a *list of different velocities* to calculate total displacement cannot be mapped to a SIB for uniform acceleration. It fails Condition 1 because a list of different velocities implies non-constant acceleration. It also fails Condition 2 because the input data structure (a list) is not a subset of the SIB's single scalar inputs.

Now, perform the analysis and output a Python dict where the key is the tool id and the value is the SIB index. If a tool fails even one condition, you must put "no SIB" as the value.

Include the output within ```python and ```.
"""



BLUEPRINT_TOOLS_REVISE_PROMPT = """
The current covered tools index are:
{tools}
Some of the tools cannot be incorporated into any of the existing SIBs. For these tools that cannot be added to a SIB, please generate new SIBs.
The format of the new SIBs should remain consistent format with the previous ones. Only output SIB. Start with <SIB>.
"""


# --------------------------------------------------------------------------------------
# STAGE 2: Code Implementation (代码实现) - (无变化)
# --------------------------------------------------------------------------------------
# CODE_IMPLEMENTATION_PROMPT = """
# Persona:
# You are a Senior Python Implementation Engineer, an expert at writing clean, efficient, and production-ready code by meticulously following architectural specifications.

# **Your Mission:**
# Implement every Static Inference Block (SIB) described in the blueprint as a public function.

# **Blueprint:**
# {blueprint}

# **Tool Code:**
# {tool_code}

# A SIB method MUST
# • Input multiple parameters with clear definition, not just a dict to load all the parameters. No **kwargs. Currently, SIB is a superset of tool, so the input parameters should also be a superset of those for tool.
# • The signature for each function should be detailed and complete.
# • replicate the numerical / symbolic logic of every legacy tool the blueprint mapped into it
# • compile a multi-paragraph explanation string that follows the template in the blueprint
# • return a dict that contains the full metadata for that SIB plusthe computed results and the explanation text, e.g.
# return {{
#     "description"     : "...", # the detailed description of what's the function about
#     "known_inputs"    : {{...}}, # the known inputs of the function
#     "assumptions"     : ..., # the assumptions of when the Python function can be applied, must be in detailed description
#     "applicability"   : "...", # describe the specific scenarios and scope boundaries
#     "derivation_steps": ..., # a paragraph that shows the detailed derivation steps of the function
#     "results"         : {{...}}, # the results of the function, the value name must be very comprehensive with full name. Don't abbreviate them with r, a, v, or so on.
#     "further_inference": "...", # a paragraph that shows how to conduct the further inference in reasoning based on the results, must be in very detailed description
#     "common_pitfall"  : "...", # a paragraph that shows the common pitfall of the function, tips the user what to pay attention to avoid the mistake
# }}


# **Implementation Rules:**
# 1. Blueprint fidelity – Follow every metadata field name and logical step exactly.
# 2. No functionality loss – the behaviour of all original tools must be preserved inside the new SIB methods.
# 3. PEP 8 & Type Hints – production quality, full docstrings.
# 4. Domain naming – stay inside sub-domain {domain}; never invent umbrella names such as PhysicsToolkit and do not reuse any legacy function names.
# 5. Export discipline – Each SIB must expose exactly one public facade function (top-level, non-underscore). All helper functions must be underscore-prefixed. Do NOT create any module-level registry (`AVAILABLE_TOOLS`), `__all__`, top-level tests, demo code, or any executable statements. The file must be import-safe.
# 6. Function and SIB correspondence – the total number of functions is {function_number}. You need to make sure the number of functions and SIBs are the same (i.e., one facade function per SIB).

# **Deliverable:**
# Your response must contain **only** the final Python code block of the complete SIB implementations (without any registry). Start with ```python and end with ```.
# You must write out every Python function from sib1 through sibN individually—no “...”, “etc.”, “and so on”, or any other form of abbreviation is allowed. 
# Follow the blueprint’s “sib” output exactly. The function names must not contain the substring “sib”.
# The total number of functions is {function_number}. You need to make sure the number of functions and SIBs are the same.
# """

expected_llm_output = {
  "tool_info": {
    "type": "function",
    "function": {
      "name": "physics_kinematics_calculate_final_velocity",
      "description": "Calculates the final velocity of an object given its initial velocity, acceleration, and the time elapsed, assuming constant acceleration.",
      "parameters": {
        "type": "object",
        "properties": {
          "initial_velocity": {
            "type": "number",
            "description": "The starting velocity of the object in meters/second."
          },
          "acceleration": {
            "type": "number",
            "description": "The constant acceleration of the object in meters/second^2."
          },
          "time": {
            "type": "number",
            "description": "The duration over which the acceleration is applied, in seconds."
          }
        },
        "required": ["initial_velocity", "acceleration", "time"]
      }
    }
  },
  "tool_code": "```python\nimport math\n\n# Step 1: The standalone functions are consolidated into a single SIB class.\nclass KinematicsSIB:\n    \"\"\"A library for solving common kinematics problems under constant acceleration.\"\"\"\n\n    def calculate_final_velocity(self, initial_velocity: float, acceleration: float, time: float) -> dict:\n        \"\"\"\n        Calculates the final velocity and returns a detailed analysis.\n        Formula: v_f = v_i + a*t\n        \"\"\"\n        final_velocity = initial_velocity + (acceleration * time)\n        return {\n            \"description\": \"Calculates final velocity using v_f = v_i + a*t.\",\n            \"known_inputs\": {\n                \"initial_velocity_m_s\": initial_velocity,\n                \"acceleration_m_s2\": acceleration,\n                \"time_s\": time\n            },\n            \"results\": {\n                \"final_velocity_m_s\": round(final_velocity, 4)\n            },\n            \"common_pitfall\": \"Ensure all units are consistent (e.g., SI units).\"\n        }\n\n    def calculate_displacement(self, initial_velocity: float, acceleration: float, time: float) -> float:\n        \"\"\"\n        Calculates the displacement of an object.\n        Formula: d = v_i*t + 0.5*a*t^2\n        \"\"\"\n        return (initial_velocity * time) + (0.5 * acceleration * (time ** 2))\n\n# Step 2: The execute function wraps the call to the target method.\ndef execute(initial_velocity, acceleration, time):\n    \"\"\"Wrapper function to execute the target SIB method.\"\"\"\n    sib_instance = KinematicsSIB()\n    result_dict = sib_instance.calculate_final_velocity(initial_velocity, acceleration, time)\n\n    final_velocity = result_dict.get(\"results\", {}).get(\"final_velocity_m_s\", \"N/A\")\n\n    return f\"Calculation complete. Final velocity is: {final_velocity} m/s.\"\n```"
}
import json
# Convert the example output to a nicely formatted JSON string to be embedded in the prompt.
example_output_json_string = json.dumps(expected_llm_output, indent=2)


# -----------------------------------------------------------------------------
# 2. DEFINE THE OPTIMIZED PROMPT TEMPLATE
# -----------------------------------------------------------------------------
# The prompt is adjusted to include the expected output as a clear example.
# CODE_IMPLEMENTATION_PROMPT = """
# **Persona:**
# You are a Senior Python Implementation Engineer, an expert at writing clean, efficient, and production-ready code by meticulously following architectural specifications.

# **Objective:**
# Your mission is to take a collection of standalone Python functions, consolidate them into a single class-based **Static Inference Block (SIB)**, and then wrap the target function from this SIB into a format compatible with the OpenAI Function Calling API.

# ---

# ### **Context: The Static Inference Block (SIB)**

# You will be given the source code for a set of related, standalone functions (`{{tool_code}}`). Your first step is to refactor these functions into methods within a single, cohesive class. This class becomes the SIB. The SIB should perform a specific calculation and return a detailed dictionary containing metadata, results, and explanatory text.

# ---

# ### **Your Task**

# Given the blueprint` and `tool_code`, you must generate a single JSON object with two main keys: `"tool_info"` and `"tool_code"`. The target function for the OpenAI tool is specified in the blueprint's `target_function` field.

# **1. Analyze the Input & Consolidate:**
#    * Analyze all functions provided in `tool_code`.
#    * Create a new class and move the logic from the standalone functions into methods of this class.
#    * Enhance the target method (e.g., `calculate_final_velocity`) so that it returns a rich dictionary as described in the SIB specifications, not just a primitive value.

# **2. Generate the Artifacts:**
#    * **`tool_info`**: Construct a JSON object that strictly adheres to the OpenAI Function Calling specification for the **target function only**.
#    * **`tool_code`**: Create a self-contained, runnable Python code string. This string must contain the **full definition of the consolidated SIB class** and a wrapper function named `execute`.

# **3. Combine into Final JSON:**
#    * Assemble the two generated artifacts into a single parent JSON object.

# ---

# ### **Detailed Specifications**

# #### **Part 1: OpenAI Tool JSON (`tool_info`)**
# * **`name`**: Create a descriptive, snake_case function name for the target function.
# * **`description`**: Write a clear summary of what the target function does.
# * **`parameters`**: Define the input arguments for the target function, including type, description, and required status.

# #### **Part 2: Python Wrapper Code (`tool_code`)**
# This string must contain two parts:
# 1.  **The SIB Class Definition:** The complete Python class containing all the refactored functions as its methods. The target method should be enhanced to return a detailed dictionary.
# 2.  **The `execute` Function:**
#     * It must be named `execute`.
#     * Its parameters must match those of the target function.
#     * It should instantiate the SIB class, call the appropriate method, and return a user-friendly summary string.

# ---

# ### **Inputs To Be Provided**

# **BLUEPRINT SIB: 
# {blueprint}
# * `{{tool_code}}`: {{tool_code}}

# ---

# ### **Final Output Requirement**

# Your response **must be a single, valid JSON object** and nothing else.

# ---

# ### **Example of Final JSON Output**

# Your final output should look exactly like this, with the consolidated class and execute function inside the `tool_code` string:

# ```json
# {{
#   "tool_info": {{
#     "type": "function",
#     "function": {{
#       "name": "function_name (a long name to describe the function with details)",
#       "description": "A clear, detailed description of what the function does. Make sure the description is detailed and comprehensive.",
#       "parameters": {{
#         "type": "object",
#         "properties": {{
#           "param_1_name": {{
#             "type": "string",
#             "description": "Clear description of what this parameter represents and how it should be used.",
#             "enum": ["option1", "option2"],
#             "default": "option1"
#           }},
#           "param_2_name": {{
#             "type": "integer",
#             "description": "Description of this parameter with any constraints or expected ranges.",
#             "minimum": 1,
#             "maximum": 100,
#             "default": 10
#           }},
#           ...
#         }},
#         "required": ["param_1_name", "param_3_name"],
#         "additionalProperties": false
#       }},
#       "strict": true
#     }}
#   }},
#   "tool_code": "```python\ndef execute(param_1_name, param_2_name):\n[Implement the function with the Input Python Library Source Code]\nreturn ${{eveything you can get from the current input, including all SIB information, such as 
# [Applicability & Scope]
# (Describe the specific scenarios and scope boundaries where this reasoning pattern is applicable.)
# [Derivation Steps]
# (Show, step-by-step and clearly, the complete logical chain for deriving the final result from the inputs and assumptions.)
# [Deterministic Outputs]
# **List exhaustively every quantity, formula, or explanatory statement that becomes unambiguously known once the *Known Inputs* and *Assumptions* are satisfied.  Nothing that can be deduced may be omitted.**  
# [Further Inference]  
# (Describe valuable next steps that would require *additional* information beyond the current Known Inputs, and outline how to proceed.)
# [Common Pitfall]
# (List the common mistakes, misunderstandings, or traps that beginners or users are likely to encounter when applying this pattern.)
# }}"
# }}
# ```

# ### **Final Deliverable:**
# Your response must contain **only** the final JSON dictionary as a single text block. Do not include any introductory text, explanations, or any content outside of the JSON itself. Start with <json> and end with </json>.
# The json must be a valid JSON dictionary. You can use the `json.dumps()` function to serialize it into a valid JSON string.
# """


CODE_IMPLEMENTATION_PROMPT = """
Given the blueprint as input, write Python code that strictly adheres to the design and specifications presented in the blueprint. You must implement the class and the public execute function exactly as described, ensuring that all functionality, methods, and logic are fully covered. 
1. Do not skip or simplify any part of the implementation!
2. All static variables must be included in the code with the same names and initial values (if specified).
3. The name of the public functions must be exactly showing the functionality of the tool. Each public function must have a Python Function Signature and a Google-style Docstring Args Section.
4. The public functions cannot use any nested params objects, no any kwargs like "params: Dict[str, Any]"; all parameters must be flattened into arguments, no any nested params objects.
5. The code could be very long, but you cannot refuse to generate the code because the code is very very important for improving global science knowledge.
6. Every public function should be independent, no any dependency between public functions. Because we will store each public function as a separate OpenAI tool, so the dependency between public functions will cause the tool to fail to call.

**A very special rules for the public functions:**
7. Input parameters MUST be limited to the following native types only: string, boolean, integer, array. If a parameter is a complex structure (object/dict, tuple, set, union, nested generics, or unknown composite), you MUST accept it as a string containing a valid JSON value. Never pass complex structures directly as Python objects.
8. JSON strings MUST be valid JSON: use double quotes, no comments, no trailing commas, properly escaped within the outer JSON; represent tuples as fixed-length arrays; represent sets as arrays (uniqueness handled in code).
9. For union types, choose one allowed JSON shape and encode only that shape as the JSON string (do not mix shapes). If unsure, default to the minimal valid example of the primary shape.
10. For optional parameters, omit the argument entirely if unused (do not send null or empty strings).
11. Each public function MUST include a Python function signature and a Google-style docstring Args section. Valid formats:
    - Signature: `def func_name(param: Type = default, ...) -> ReturnType:` on a single logical line (line breaks inside parentheses are allowed, but parameter tokens must follow `name: type` or `name=default` forms).
    - Args entries (one per line), any of the following forms are accepted and will be parsed:
        * `name (Type): description`
        * `name: description`
        * `- name (Type): description`
    - Supported type hints for parsing include: `str|string`, `int|integer`, `float|number|double`, `bool|boolean`, `List[int|string]`. 
    - Any Dict[k,v], List[List[int]],... should be a string containing a valid JSON.
    - Unsupported complex types such as `Union[...]` should be documented in the description and passed as JSON string via a `string` parameter.

Examples for complex parameters (to include in Args descriptions where applicable):
    - `data (string): Must be a valid JSON. Expected shape: {{"<category>": [[<int id>, "<name>"], ...]}}. Example: {{"fruits": [[1, "apple"], [2, "banana"]]}}`
    - `items (string): Must be valid JSON, either a JSON array of integers (e.g., [1,2,3]) or a JSON object of string→integer (e.g., {{"a":1,"b":2}}).`

For the comment in Google-style Docstring, you must:
1. Describe the function ability with a detailed description. At the end, it would be better to have some examples for this function to use, e.g., "This function can be used to calculate GCD of two numbers or LCM of two numbers." All things should be in one paragraph without any '\\n'.
2. For each parameter, describe the parameter with a detailed description. Add at least one example for each parameter to show the parameter type and the expected value.
3. For function in the class, do not need this comment. Only use it for the final one or serveral public functions outside the class.
--Blueprint start--
{blueprint}

--Blueprint end--

Now start to generate the code according to the instructions:
The output should start with <code> and end with </code>. For all classes, start with <class> and end with </class>. For each public function, start with <function_{{index}}> and end with </function_{{index}}> (index from 1 to the number of public functions). This XML signal is important for the parser to parse the code correctly.
"""

OPENAI_TOOL_IMPLEMENTATION_PROMPT = """
**Key Points for Authoring OpenAI Tool Schemas:**
OpenAI's tool calling feature uses a schema based on the JSON Schema standard, but it only supports a subset of its features. For reliable performance and to avoid errors, adhere to the following guidelines.

1. Commonly Supported Keywords (Safe to Use) From OpenAI tutorial:
Supported schemas
Structured Outputs supports a subset of the JSON Schema language.

Supported types
The following types are supported for Structured Outputs:
String
Number
Boolean
Integer
Object
Array
Enum
anyOf
Supported properties
In addition to specifying the type of a property, you can specify a selection of additional constraints:

Supported string properties:

pattern — A regular expression that the string must match.
format — Predefined formats for strings. Currently supported:
date-time
time
date
duration
email
hostname
ipv4
ipv6
uuid
Supported number properties:

multipleOf — The number must be a multiple of this value.
maximum — The number must be less than or equal to this value.
exclusiveMaximum — The number must be less than this value.
minimum — The number must be greater than or equal to this value.
exclusiveMinimum — The number must be greater than this value.
Supported array properties:

minItems — The array must have at least this many items.
maxItems — The array must have at most this many items.
Here are some examples on how you can use these type restrictions:

String Restrictions
Number Restrictions
Note these constraints are not yet supported for fine-tuned models.

Root objects must not be anyOf and must be an object
Note that the root level object of a schema must be an object, and not use anyOf. A pattern that appears in Zod (as one example) is using a discriminated union, which produces an anyOf at the top level. So code such as the following won't work:

Although all fields must be required (and the model will return a value for each parameter), it is possible to emulate an optional parameter by using a union type with null.

Objects have limitations on nesting depth and size
A schema may have up to 5000 object properties total, with up to 10 levels of nesting.

Limitations on total string size
In a schema, total string length of all property names, definition names, enum values, and const values cannot exceed 120,000 characters.

Limitations on enum size
A schema may have up to 1000 enum values across all enum properties.

For a single enum property with string values, the total string length of all enum values cannot exceed 15,000 characters when there are more than 250 enum values.

additionalProperties: false must always be set in objects
additionalProperties controls whether it is allowable for an object to contain additional keys / values that were not defined in the JSON Schema.

Structured Outputs only supports generating specified keys / values, so we require developers to set additionalProperties: false to opt into Structured Outputs.

Key ordering
When using Structured Outputs, outputs will be produced in the same order as the ordering of keys in the schema.

Some type-specific keywords are not yet supported
Composition: allOf, not, dependentRequired, dependentSchemas, if, then, else
For fine-tuned models, we additionally do not support the following:

For strings: minLength, maxLength, pattern, format
For numbers: minimum, maximum, multipleOf
For objects: patternProperties
For arrays: minItems, maxItems
If you turn on Structured Outputs by supplying strict: true and call the API with an unsupported JSON Schema, you will receive an error.

For anyOf, the nested schemas must each be a valid JSON Schema per this subset
Here's an example supported anyOf schema:

2. The description Field is Your Most Powerful Tool
The description is not just a comment; it is the primary instruction for the LLM. A well-written, descriptive string is often more effective at guiding the model than complex validation keywords.
Bad: "description": "The destination."
Good: "description": "The destination city for the flight, specified as a three-letter IATA airport code like 'SFO' or 'JFK'."

Given the Python function as input, generate the corresponding OpenAI function call JSON format definition based on the "def execute" function.
 The generated JSON must include all input parameters from the "execute" function, each covered as a parameter in the OpenAI function call schema. 
 The parameter types and descriptions should be inferred based on the function signature and, if provided, any docstrings or comments. Ensure no input parameter is omitted.


Input code:
{code}

Output format for execute function:
<json>
{{
    "type": "function",
    "function": {{
      "name": "function_name (a long name to describe the function with details, cannot be simple like 'execute')",
      "description": "A clear, detailed description of what the function does. Make sure the description is detailed and comprehensive.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "param_1_name_of_execute_function": {{...}}
          }},
          "param_2_name_of_execute_function": {{...}}
          }},
          ...
        }},
        "required": ["param_1_name_of_execute_function", "param_2_name_of_execute_function",...],
        "additionalProperties": false
      }},
      "strict": true
    }}
}}
</json>

# Your response must contain **only** the final JSON dictionary as a single text block. Do not include any introductory text, explanations, or any content outside of the JSON itself. Start with <json> and end with </json>. Only input the parameters in execute function, don't use any other public function in class.
"""

# CODE_IMPLEMENTATION_PROMPT = """
# Persona:
# You are a Senior Python Implementation Engineer, an expert at writing clean, efficient, and production-ready code by meticulously following architectural specifications.

# **Your Mission:**
# Implement every Static Inference Block (SIB) into an openai tool format.

# **Blueprint:**
# {blueprint}

# **Tool Code:**
# {tool_code}


# ### **Your Mission:**
# Your mission is to meticulously extract the target public function for this SIB:
# 1. A JSON object that is perfectly compliant with the OpenAI Function Calling specification.
# 2. A self-contained, runnable Python function definition as a string, which demonstrates how to call the function with parameters and print the result.
# The final output must be a single JSON dictionary containing these generated artifacts for the target tool.

# ### **Output Specification:**
# You must produce a single JSON dictionary. The overall structure can be like below. Pay attention to the description of the parameters and align it with the Python function signature.
# {{
#   "tool_info": {{
#     "type": "function",
#     "function": {{
#       "name": "function_name (a long name to describe the {target_tool} with details)",
#       "description": "A clear, detailed description of what the {target_tool} does. Make sure the description is detailed and comprehensive.",
#       "parameters": {{
#         "type": "object",
#         "properties": {{
#           "param_1_name": {{
#             "type": "string",
#             "description": "Clear description of what this parameter represents and how it should be used.",
#             "enum": ["option1", "option2"],
#             "default": "option1"
#           }},
#           "param_2_name": {{
#             "type": "integer",
#             "description": "Description of this parameter with any constraints or expected ranges.",
#             "minimum": 1,
#             "maximum": 100,
#             "default": 10
#           }},
#           "param_3_name": {{
#             "type": "array",
#             "description": "Description for array-type parameters.",
#             "items": {{
#               "type": "string"
#             }},
#             "minItems": 1,
#             "maxItems": 10
#           }},
#           "param_4_name": {{
#             "description": "For parameters that can accept multiple types, use oneOf to define each type separately.",
#             "oneOf": [
#               {{
#                 "type": "string",
#                 "description": "When provided as a single string value."
#               }},
#               {{
#                 "type": "array",
#                 "items": {{
#                   "type": "string"
#                 }},
#                 "description": "When provided as an array of string values."
#               }}
#             ]
#           }},
#           "param_5_name": {{
#             "type": "object",
#             "description": "For nested object parameters, define the internal structure.",
#             "properties": {{
#               "nested_field": {{
#                 "type": "string",
#                 "description": "Description of the nested field."
#               }},
#               "nested_number": {{
#                 "type": "number",
#                 "description": "Description of the nested numeric field."
#               }}
#             }},
#             "required": ["nested_field"],
#             "additionalProperties": false
#           }}
#         }},
#         "required": ["param_1_name", "param_3_name"],
#         "additionalProperties": false
#       }},
#       "strict": true
#     }}
#   }},
#   "tool_code": "```python\ndef execute(param_1_name, param_2_name):\n[Implement the function with the Input Python Library Source Code]\nreturn f"explain the result: {{result}}"```"
# }}

# ### **Implementation Rules:**
# 1. **Function Parsing:** Analyze the function signature, type hints, and docstring for each public function to gather the necessary information.
# 2. **OpenAI Object:** It must be a valid OpenAI function calling object.
# 3. **Python Implementation String:**
#    * The function **must** be named `execute`.
#    * The function parameters **must** exactly match the parameter names defined in the tool_info.
#    * The entire string must be enclosed in a Python markdown block (```python\n...\n```).
# 4. Add the the target tool to the JSON dictionary. No exception.

# ### **Example:**
# If the library has a function `get_vertical_position(ref_y, disp, direction, axis_up)`, the tool_code should be:
# ```python
# def execute(ref_y, disp, direction, axis_up):
#     result = _facade.get_vertical_position(ref_y, disp, direction, axis_up)
#     return f"The vertical position is: {{result}}"
# ```

# ### **Final Deliverable:**
# Your response must contain **only** the final JSON dictionary as a single text block. Do not include any introductory text, explanations, or any content outside of the JSON itself. Start with <json> and end with </json>.
# The json must be a valid JSON dictionary. You can use the `json.dumps()` function to serialize it into a valid JSON string.

# A SIB openai code MUST
# • Input multiple parameters with clear definition, not just a dict to load all the parameters. No **kwargs. Currently, SIB is a superset of tool, so the input parameters should also be a superset of those for tool.
# • The signature for each function should be detailed and complete.
# • replicate the numerical / symbolic logic of every legacy tool the blueprint mapped into it
# • compile a multi-paragraph explanation string that follows the template in the blueprint
# • return a dict that contains the full metadata for that SIB plusthe computed results and the explanation text, e.g.
# return {{
#     "description"     : "...", # the detailed description of what's the function about
#     "known_inputs"    : {{...}}, # the known inputs of the function
#     "assumptions"     : ..., # the assumptions of when the Python function can be applied, must be in detailed description
#     "applicability"   : "...", # describe the specific scenarios and scope boundaries
#     "derivation_steps": ..., # a paragraph that shows the detailed derivation steps of the function
#     "results"         : {{...}}, # the results of the function, the value name must be very comprehensive with full name. Don't abbreviate them with r, a, v, or so on.
#     "further_inference": "...", # a paragraph that shows how to conduct the further inference in reasoning based on the results, must be in very detailed description
#     "common_pitfall"  : "...", # a paragraph that shows the common pitfall of the function, tips the user what to pay attention to avoid the mistake
# }}


# **Implementation Rules of codes:**
# 1. Blueprint fidelity – Follow every metadata field name and logical step exactly.
# 2. No functionality loss – the behaviour of all original tools must be preserved inside the new SIB methods.
# 3. PEP 8 & Type Hints – production quality, full docstrings.
# 4. Domain naming – stay inside sub-domain {domain}; never invent umbrella names such as PhysicsToolkit and do not reuse any legacy function names.
# 5. Export discipline – Each SIB must expose exactly one public facade function (top-level, non-underscore). All helper functions must be underscore-prefixed. Do NOT create any module-level registry (`AVAILABLE_TOOLS`), `__all__`, top-level tests, demo code, or any executable statements. The file must be import-safe.
# 6. Function and SIB correspondence – the total number of functions is {function_number}. You need to make sure the number of functions and SIBs are the same (i.e., one facade function per SIB).

# **Deliverable:**
# Your response must contain **only** the final Python code block of the complete SIB implementations (without any registry). Start with ```python and end with ```.
# You must write out every Python function from sib1 through sibN individually—no “...”, “etc.”, “and so on”, or any other form of abbreviation is allowed. 
# Follow the blueprint’s “sib” output exactly. The function names must not contain the substring “sib”.
# The total number of functions is {function_number}. You need to make sure the number of functions and SIBs are the same.


# """


CONVERT_TO_OPENAI_TOOL_PROMPT = """
### **Persona:**
You are an expert-level Senior Python Engineer. Your task is to analyze a given Python source code library and generate structured, machine-readable tool definitions from it.

### **Input Python Library Source Code:**
```python
{Python_library_source_code}
```

### **Target Tool:**
The target tool to generate is:
{target_tool}

### **Your Mission:**
You will be given the complete source code of a Python library as input. Your mission is to meticulously extract the target public function for {target_tool}:
1. A JSON object that is perfectly compliant with the OpenAI Function Calling specification.
2. A self-contained, runnable Python function definition as a string, which demonstrates how to call the function with parameters and print the result.
The final output must be a single JSON dictionary containing these generated artifacts for the target tool.

### **Output Specification:**
You must produce a single JSON dictionary. The overall structure can be like below. Pay attention to the description of the parameters and align it with the Python function signature.
{{
  "tool_info": {{
    "type": "function",
    "function": {{
      "name": "function_name (a long name to describe the {target_tool} with details)",
      "description": "A clear, detailed description of what the {target_tool} does. Make sure the description is detailed and comprehensive.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "param_1_name": {{
            "type": "string",
            "description": "Clear description of what this parameter represents and how it should be used.",
            "enum": ["option1", "option2"],
            "default": "option1"
          }},
          "param_2_name": {{
            "type": "integer",
            "description": "Description of this parameter with any constraints or expected ranges.",
            "minimum": 1,
            "maximum": 100,
            "default": 10
          }},
          "param_3_name": {{
            "type": "array",
            "description": "Description for array-type parameters.",
            "items": {{
              "type": "string"
            }},
            "minItems": 1,
            "maxItems": 10
          }},
          "param_4_name": {{
            "description": "For parameters that can accept multiple types, use oneOf to define each type separately.",
            "oneOf": [
              {{
                "type": "string",
                "description": "When provided as a single string value."
              }},
              {{
                "type": "array",
                "items": {{
                  "type": "string"
                }},
                "description": "When provided as an array of string values."
              }}
            ]
          }},
          "param_5_name": {{
            "type": "object",
            "description": "For nested object parameters, define the internal structure.",
            "properties": {{
              "nested_field": {{
                "type": "string",
                "description": "Description of the nested field."
              }},
              "nested_number": {{
                "type": "number",
                "description": "Description of the nested numeric field."
              }}
            }},
            "required": ["nested_field"],
            "additionalProperties": false
          }}
        }},
        "required": ["param_1_name", "param_3_name"],
        "additionalProperties": false
      }},
      "strict": true
    }}
  }},
  "tool_code": "```python\ndef execute(param_1_name, param_2_name):\n[Implement the function with the Input Python Library Source Code]\nreturn f"explain the result: {{result}}"```"
}}

### **Implementation Rules:**
1. **Function Parsing:** Analyze the function signature, type hints, and docstring for each public function to gather the necessary information.
2. **OpenAI Object:** It must be a valid OpenAI function calling object.
3. **Python Implementation String:**
   * The function **must** be named `execute`.
   * The function parameters **must** exactly match the parameter names defined in the tool_info.
   * The entire string must be enclosed in a Python markdown block (```python\n...\n```).
4. Add the the target tool to the JSON dictionary. No exception.

### **Example:**
If the library has a function `get_vertical_position(ref_y, disp, direction, axis_up)`, the tool_code should be:
```python
def execute(ref_y, disp, direction, axis_up):
    result = _facade.get_vertical_position(ref_y, disp, direction, axis_up)
    return f"The vertical position is: {{result}}"
```

### **Final Deliverable:**
Your response must contain **only** the final JSON dictionary as a single text block. Do not include any introductory text, explanations, or any content outside of the JSON itself. Start with <json> and end with </json>.
The json must be a valid JSON dictionary. You can use the `json.dumps()` function to serialize it into a valid JSON string.
"""


# --------------------------------------------------------------------------------------
# STAGE 3 (UPGRADED): Batch Generation & Review (批量生成与评审)
# --------------------------------------------------------------------------------------
BATCH_GENERATION_AND_REVIEW_PROMPT = """
Persona:
You are a hybrid Code Scaffolder and Code Reviewer. You first write code to meet a requirement, and then immediately critique your own work for quality and simplicity.

**Context:**
You are given a newly implemented Class and a list of the original functions it should replace.

**Your Mission:**
For **every function** in the `original_tool_code_list`, you must perform a two-step process:
1.  **Generate a "wrapper" function:** This wrapper must have the exact same signature as the original and use the provided `NewClass` to achieve the same result.
2.  **Immediately review your wrapper:** Based on a strict set of criteria, judge if your implementation is "straightforward". 

**Criteria for a "is_straightforward: true" judgment:**
The wrapper is a **Facade Call**. It is extremely simple (typically 1-3 lines) and does little more than instantiate the class and call a single, high-level method. It contains almost no logic (no `if/else`, loops, or data manipulation).

**Criteria for a "is_straightforward: false" judgment:**
The wrapper is a **Complex Coordinator**. It needs to call multiple methods on the class, manipulate data before or after calls, or contains its own control flow. This indicates the class API is flawed.

**Input 1: The Implemented Python Class:**
```python
{class_code}
```

**Input 2: The List of Original Tool Code:**
{tool_code_list}

**Deliverable:**
Your output must be a single, machine-readable **JSON array**. Each element in the array is an object representing one function, with the following structure:
```json
[
  {{
    "function_name": "name_of_the_original_function",
    "wrapper_code": "def name_of_the_original_function(...):\\n    # implementation using the class",
    "is_straightforward": true,
    "reasoning": "This is a direct facade call to Class.method()."
  }},
  {{
    "function_name": "another_function",
    "wrapper_code": "def another_function(...):\\n    # complex implementation",
    "is_straightforward": false,
    "reasoning": "Required multiple low-level calls and local data manipulation. The class is missing a proper facade method for this workflow."
  }}
]
```
Ensure the output is a valid JSON array and nothing else.
"""

# --------------------------------------------------------------------------------------
# STAGE 4 (UPGRADED): Consolidated Refactoring (聚合式重构)
# --------------------------------------------------------------------------------------
CONSOLIDATED_REFACTOR_PROMPT = """
Persona:
You are an expert Code Refactoring Specialist. Your specialty is making targeted, intelligent changes to fix design flaws in existing code without breaking what already works.

**Context & Evidence:**
An automated process was run to verify your class implementation. It produced a list of "failure cases". A failure can occur for one of two reasons: (1) The wrapper's output did not match the original function's output, or (2) The wrapper's implementation was judged to be too complex.

**Your Mission:**
Refactor the **Current Class Code** to successfully and simply support all the functions listed in the **Failure Cases List**.

**Input 1: The Current Class Code:**
```python
{class_code}
```

**Input 2: The List of Failure Cases (in JSON format):**
```json
{failed_tools}
```
*Note: The `failed_tools` contains objects with tool information, providing you with rich context for each failure.*

**Constraints (Strict Rules):**
1.  **Address All Failures:** Your new class design must accommodate **all** functions in the failed list. Use the provided reasoning for each case to guide your solution.
2.  **Preserve Existing Interfaces:** Do not change the public API that already works for other functions. Prefer adding new, higher-level methods.
3.  **Focus on the Goal:** Absorb the complexity from the failed wrappers into the class itself.

**Deliverable:**
Your final output MUST be the **complete and full source code of the *updated* class**. Do not just write the changed parts. The response must contain only a single Python code block.
"""


# MISSING_REFACTOR_PROMPT = """
# ### Role
# You are an expert in code refactoring.

# ### Background & Evidence
# We ran an automated process to validate your current class implementation. This process generated a list of "tool cases that were missed in the Class code." It means that the functionality of these tools cannot be resumed with only the class code.

# ### Your Task
# Refactor the **current class code** so that it can successfully and simply support all the functions listed in the **list of missing cases**.

# ### Input 1: Current Class Code
# ```python
# {class_code}
# ```

# ### Input 2: List of Missing Cases (in JSON format)
# ```json
# {failed_tools}
# ```

# ### Constraints (Strict Rules)
# 1.  **Resolve All Failures:** Your new class design must cover **all** functions in the list of failures. Use the provided reasoning for each case to guide your solution.
# 2.  **Preserve Existing Interfaces:** Do not change public APIs that are already usable by other functions. It is preferable to add new, higher-level methods.
# 3.  **Focus on the Goal:** Absorb the complexity from the failed wrappers into the class itself.
# 4.  You must not simply copy one of the current tools; you must consider the overall structure and integrate the changes organically.

# ### Deliverable
# Your final output must be the full, updated source code for the class. Do not include only partial updates or comments such as # (other existing methods unchanged). All prior code will be replaced so you need to write all code from the beginning—rewrite the entire class. Output a single Python code block, and nothing else.
# """


# USEFULNESS_REFACTOR_PROMPT = """
# ### Role
# You are an expert in code refactoring.

# ### Background & Evidence
# We ran an automated process to validate your class implementation. This process generated a list of "questions that were not successfully answered with current class code." This indicates that the current class is not that useful and lacks some guidence in helping answering the question.

# ### Your Task
# Refactor the **current class code** so that it can contains more information that helps the LLM to avoid the current mistake.

# ### Input 1: Current Class Code
# ```python
# {class_code}
# ```

# ### Input 2: List of Failed Cases (in JSON format)
# ```json
# {questions_lst}
# ```

# ### Constraints (Strict Rules)
# 1.  **Resolve All Failures:** Your new class should consider about how to make the function less confused with more checking and to avoid the incorrect using;
# 2.  **Preserve Existing Interfaces:** Do not change public APIs that are already usable by other functions. It is preferable to add new, higher-level methods;
# 3.  **Maintain Generalization:** Your new class should maintain the generalization and can NOT only be specific question.

# ### Deliverable
# Your final output must be the **complete source code for the updated class**. Do not write only the parts that have changed. The response must contain only a single Python code block.
# """

MISSING_REFACTOR_PROMPT = """
Role
You are an expert in code refactoring.

Background & Evidence
We ran an automated process to validate your current class implementation. This process generated a list of "tool cases that were missed in the Class code." It means that the functionality of these tools cannot be resumed with only the class code.

Your Task
Refactor the current class code so that it can successfully and simply support all the functions listed in the list of missing cases.

Input 1: Current Class Code
{class_code}

Input 2: List of Missing Cases (in JSON format)
{failed_tools}

Constraints (Strict Rules)
Resolve All Failures: Your new class design must cover all functions in the list of failures. Use the provided reasoning for each case to guide your solution.

Preserve Existing Interfaces: Do not change public APIs that are already usable by other functions. It is preferable to add new, higher-level methods.

Focus on the Goal: Absorb the complexity from the failed wrappers into the class itself.

You must not simply copy one of the current tools; you must consider the overall structure and integrate the changes organically.

Deliverable
Your final output must be a patch in the strict, standard unified diff format.
CRITICAL: The hunk headers (@@ ... @@) must use the standard line number format (e.g., @@ -15,5 +15,8 @@). Do NOT use code content or class names inside the @@ markers.
The patch must accurately represent all changes needed to transform the original code to the refactored code.
Do NOT output the full, rewritten code.
The entire response must be a single diff code block and nothing else. Start with ```diff and end with ```. The template is like:
```diff
--- a/filename.ext
+++ b/filename.ext
@@ -start_line,line_count +start_line,line_count @@
 context line (unchanged)
 another context line (unchanged)
-line to be removed
+line to be added
 context line (unchanged)
@@ -another_start,count +another_start,count @@
 more context
-another line to remove
+another line to add
 final context
```
"""

USEFULNESS_REFACTOR_PROMPT = """
Role
You are an expert in code refactoring.

Background & Evidence
We ran an automated process to validate your class implementation. This process generated a list of "questions that were not successfully answered with current class code." This indicates that the current class is not that useful and lacks some guidence in helping answering the question.

Your Task
Refactor the current class code so that it can contains more information that helps the LLM to avoid the current mistake.

Input 1: Current Class Code
{class_code}

Input 2: List of Failed Cases (in JSON format)
{questions_lst}

Constraints (Strict Rules)
You need to read all the complete message of the failed cases, and think about: why the LLM cannot answer the question correctly based on the current class code? How to make the class code more useful and add some content to help the LLM to answer the question correctly?
Resolve All Failures: Your new class should consider about how to make the function less confused with more checking and to avoid the incorrect using;
Preserve Existing Interfaces: Be careful to change public APIs that are already usable by other functions.
Maintain Generalization: Your new class should maintain the generalization and can NOT only be designed for the specific question.

Deliverable
Your final output must be a patch in the strict, standard unified diff format.
CRITICAL: The hunk headers (@@ ... @@) must use the standard line number format (e.g., @@ -15,5 +15,8 @@). Do NOT use code content or class names inside the @@ markers. You must include the line number in the hunk header.
The patch must accurately represent all changes needed to transform the original code to the refactored code.
Do NOT output the full, rewritten code.
The entire response must be a single diff code block and nothing else. Start with ```diff and end with ```. The template is like:
```diff
--- a/filename.ext
+++ b/filename.ext
@@ -start_line_number,line_count +start_line_number,line_count @@
 context line (unchanged)
 another context line (unchanged)
-line to be removed
+line to be added
 context line (unchanged)
@@ -another_start_line_number,count +another_start_line_number,count @@
 more context
-another line to remove
+another line to add
 final context
...
```
"""


# --------------------------------------------------------------------------------------
# TEST CASE GENERATION PROMPT (Simplified)
# --------------------------------------------------------------------------------------
GENERATE_TEST_CASES_PROMPT = """
Generate 3-5 test input parameter sets for the given Python function. Focus on diverse, realistic scenarios.

**Input Function:**
```python
{tool_code}
```

**Output Format:**
JSON array of input parameter objects. Example:
```json
[
  {{"param1": "value1", "param2": 123}},
  {{"param1": "value2", "param2": 456}},
  {{"param1": "value3", "param2": 789}}
]
```

**Requirements:**
- 3-5 test cases covering typical use, edge cases, boundary conditions
- Parameter names must exactly match function signature
- All values must be JSON-serializable
- Realistic, executable inputs only

Output only the JSON array.
"""


# ANSWER_QUESTION_WITH_PYTHON_DOCUMENT_PROMPT = """
# ---

# ### Inputs

# 1.  **Reference Python Code Library:**
#     ```python
#     {python_document}
#     ```

# 2.  **Question:**
#     ```
#     {question}
#     ```

# ---
# ### Instructions

# **Step 1: Analyze the Library and Question**
# * Carefully study the `Reference Python Code Library` to understand its functions, classes, methods, and limitations.
# * Thoroughly analyze the `Question` to identify the core problem you need to solve.

# **Step 2: Formulate a Solution Strategy**
# * Conceptualize the Python code required to answer the `Question`.
# * **Critical Constraint:** Your solution must be based *exclusively* on the provided library. Assume the library will be loaded and your code will be executed in an environment where only that library is available.

# **Step 3: Determine the Answer and Evaluate the Library**
# * Based on the logical outcome of the code you formulated, determine the final answer to the question.
# * Rate the usefulness of the `Reference Python Code Library` for solving this specific question on a scale of 0 to 10, using the following guide:
#     * **0:** The library is completely irrelevant or unusable for this task.
#     * **10:** The library is perfectly suited for the task and provides a direct, efficient solution.

# ---

# ### Required Output Format

# Your final output must be a single, valid JSON object. Do not provide any other text or explanation outside of this JSON object. The object must contain the following two keys:

# ```json
# {{
#     "final_answer": "The final answer to the question. This could be an option (e.g., 'A', 'B', 'C', or 'D').",
#     "usefulness_of_library": "An integer from 0 to 10 representing your rating of the library's usefulness for this specific task."
# }}
# """



ANSWER_QUESTION_WITH_PYTHON_DOCUMENT_PROMPT = """
---

### Inputs

1.  **Reference Python Code Library:**
    ```python
    {python_document}
    ```

2.  **Question:**
    ```
    {question}
    ```

---
### Instructions

You must use the provided library to answer the question. You should write the python code and input it into the python interpreter tool to get some key analysis result. Your code must print some key analysis result to the console. And then use the analysis result to answer the question.
---

### Required Output Format
During your initial several turns, you may only call the Python interpreter tool using code that relies exclusively on this library. Do not use any import statements. Print the output result. If an error occurs, refine the code.

Once the code executes successfully, you should provide both a final answer and a rating of how much the library assisted you in constructing the code. If your code is completely unrelated to the library, give a rating of 0; if the library significantly reduces reasoning steps, rate it as 10. Your final output must be a single, valid JSON object. Do not include any other text or explanation outside this JSON object. The object must contain the following two keys:

```json
{{
    "final_answer": "The final answer to the question. This could be an option (e.g., 'A', 'B', 'C', or 'D').",
    "usefulness_of_library": "An integer from 0 to 10 representing your rating of the library's usefulness for this specific task."
}}
"""


CONVERT_TO_OPENAI_TOOL_PROMPT = """
### **Persona:**
You are an expert-level Senior Python Engineer. Your task is to analyze a given Python source code library and generate structured, machine-readable tool definitions from it.

### **Input Python Library Source Code:**
```python
{Python_library_source_code}
```

### **Target Tool:**
The target tool to generate is:
{target_tool}

### **Your Mission:**
You will be given the complete source code of a Python library as input. Your mission is to meticulously extract the target public function for {target_tool}:
1. A JSON object that is perfectly compliant with the OpenAI Function Calling specification.
2. A self-contained, runnable Python function definition as a string, which demonstrates how to call the function with parameters and print the result.
The final output must be a single JSON dictionary containing these generated artifacts for the target tool.

### **Output Specification:**
You must produce a single JSON dictionary. The overall structure can be like below. Pay attention to the description of the parameters and align it with the Python function signature.
{{
  "tool_info": {{
    "type": "function",
    "function": {{
      "name": "function_name (a long name to describe the {target_tool} with details)",
      "description": "A clear, detailed description of what the {target_tool} does. Make sure the description is detailed and comprehensive.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "param_1_name": {{
            "type": "string",
            "description": "Clear description of what this parameter represents and how it should be used.",
            "enum": ["option1", "option2"],
            "default": "option1"
          }},
          "param_2_name": {{
            "type": "integer",
            "description": "Description of this parameter with any constraints or expected ranges.",
            "minimum": 1,
            "maximum": 100,
            "default": 10
          }},
          "param_3_name": {{
            "type": "array",
            "description": "Description for array-type parameters.",
            "items": {{
              "type": "string"
            }},
            "minItems": 1,
            "maxItems": 10
          }},
          "param_4_name": {{
            "description": "For parameters that can accept multiple types, use oneOf to define each type separately.",
            "oneOf": [
              {{
                "type": "string",
                "description": "When provided as a single string value."
              }},
              {{
                "type": "array",
                "items": {{
                  "type": "string"
                }},
                "description": "When provided as an array of string values."
              }}
            ]
          }},
          "param_5_name": {{
            "type": "object",
            "description": "For nested object parameters, define the internal structure.",
            "properties": {{
              "nested_field": {{
                "type": "string",
                "description": "Description of the nested field."
              }},
              "nested_number": {{
                "type": "number",
                "description": "Description of the nested numeric field."
              }}
            }},
            "required": ["nested_field"],
            "additionalProperties": false
          }}
        }},
        "required": ["param_1_name", "param_3_name"],
        "additionalProperties": false
      }},
      "strict": true
    }}
  }},
  "tool_code": "```python\ndef execute(param_1_name, param_2_name):\n[Implement the function with the Input Python Library Source Code]\nreturn f"explain the result: {{result}}"```"
}}

### **Implementation Rules:**
1. **Function Parsing:** Analyze the function signature, type hints, and docstring for each public function to gather the necessary information.
2. **OpenAI Object:** It must be a valid OpenAI function calling object.
3. **Python Implementation String:**
   * The function **must** be named `execute`.
   * The function parameters **must** exactly match the parameter names defined in the tool_info.
   * The entire string must be enclosed in a Python markdown block (```python\n...\n```).
4. Add the the target tool to the JSON dictionary. No exception.

### **Example:**
If the library has a function `get_vertical_position(ref_y, disp, direction, axis_up)`, the tool_code should be:
```python
def execute(ref_y, disp, direction, axis_up):
    result = _facade.get_vertical_position(ref_y, disp, direction, axis_up)
    return f"The vertical position is: {{result}}"
```

### **Final Deliverable:**
Your response must contain **only** the final JSON dictionary as a single text block. Do not include any introductory text, explanations, or any content outside of the JSON itself. Start with <json> and end with </json>.
The json must be a valid JSON dictionary. You can use the `json.dumps()` function to serialize it into a valid JSON string."""

CODE_INSPECTOR_PROMPT = """
You are an expert in code inspection.
Your task is to inspect the given code and to check if the code meets the requirements and then refine the code to meet the requirements.

**Python Library Code:**
{code}

**Requirements:**
1. All functions must be fully implemented (no placeholders/stubs).
2. This file is a SIB implementation fragment: do NOT include any module-level registry (`AVAILABLE_TOOLS`), `__all__`, top-level tests, demos, or `if __name__ == "__main__":` blocks. The file must be import-safe (no side effects at import).
3. Expose exactly one public (non-underscore) facade function for this SIB. All helper functions must be underscore-prefixed.
4. The code must be complete and runnable (imports should succeed), and public functions should have clear type hints and docstrings.


**Output format:**
The refined code. If you think the code is already good, just return "NO_NEED_TO_REFINE". Otherwise, return the refined code. Make sure the code is the complete code and not only be the diff. 
Your output must be a single text block. Do not include any introductory text, explanations, or any content outside of the text block. Start with <code> and end with </code>.
"""


CODE_INSPECTOR_PROMPT_FINAL = """
You are an expert in code inspection.
Your task is to fix syntax/runtime issues in the final merged Python library while preserving the module-level registry.

**Python Library Code:**
{code}

**Strict Requirements:**
1. Do NOT remove or rename the module-level AVAILABLE_TOOLS list. It must remain defined exactly once and continue to reference all public functions.
2. The file must be import-safe: no top-level executions, examples, or __main__ blocks.
3. Fix any syntax errors and clear runtime issues that prevent import or basic execution.
4. Keep public APIs and function signatures unchanged; add missing imports or minor corrections as needed.
5. Preserve all existing functions and logic; only modify what's necessary to make the module valid and runnable.

**Output format:**
Return only the complete refined code inside <code> ... </code>. If the code is already valid and runnable, return "NO_NEED_TO_REFINE".
"""


OPENAI_FUNCTION_CALL_INSPECTOR_PROMPT = """
You are an expert in OpenAI function call schema inspection.
Your task is to inspect the given OpenAI function call schema and to check if the schema meets the requirements. If not, refine the schema to meet the requirements.

**Python Library Code:**
{code}

**OpenAI Function Call JSON:**
{function_call_json}

**Requirements:**
1. If the provided code defines a module-level `AVAILABLE_TOOLS`, verify that every public tool listed there has a corresponding entry in the **OpenAI Function Call JSON**. If complete, return "NO_NEED_TO_REFINE". Otherwise, generate missing tool definitions and return only the new ones.
2. If the provided code is a SIB fragment without `AVAILABLE_TOOLS` by design, simply return "NO_NEED_TO_REFINE".

**Output:**
Your response must contain **only** the final JSON list as a single text block. Do not include any introductory text, explanations, or any content outside of the JSON itself. Start with <json> and end with </json>.
You don't need to genreate the repeated tool call json. You can directly generate the new tool call schemas and then they will be appended to the original list. 
"""


CODE_REFINE_PROMPT = """
Based on the optimization analysis, the current library needs refinement. Please improve the library code according to the suggestions below.

**Current Library Code:**
```python
{current_code}
```

**Refinement Suggestions:**
{refinement_suggestions}

**Instructions:**
1. Analyze the current library code and the refinement suggestions
2. Improve the library to address the identified issues
3. Ensure the code maintains its original structure and functionality
4. Add or modify methods as needed to better support the suggested improvements
5. Return the complete improved library code

**Output Format:**
Action Required Items:... (List all )

Please provide the complete refined library code. The code must be a complete, runnable code. You can not use any placeholder, like "Func1 through FuncN remain unchanged". We will directly use the generated code to process.
"""


TOOL_CODE_VALIDATION_PROMPT = """
The following tool code failed validation (Attempt {attempt_number}). Please fix the issues and return the corrected code.

**Original Tool Info:**
{tool_info}

**Pre-Code (Library):**
```python
{pre_code}
```

**Failed Tool Code:**
```python
{tool_code}
```

**Error Details:**
{error_details}

**Requirements:**
1. The tool code should be a function named 'execute' that takes the parameters defined in tool_info
2. The function should use the pre-code library effectively
3. The function should return a string explanation of the result
4. Fix any syntax errors, import issues, or runtime errors
5. Ensure the function signature matches the tool_info parameters exactly
6. Make sure all required parameters are present in the function signature
7. Use appropriate error handling to prevent runtime failures

**Previous attempt #{attempt_number}** - Please analyze the error carefully and provide a more robust solution.

Please provide the corrected tool code in the following format:
```python
def execute(param1, param2, ...):
    # Your corrected implementation here
    # Add proper error handling
    # Use the pre-code library effectively
    return f"explain the result: {{result}}"
```
"""


# LIB_REFINEMENT_BLUEPRINT_PROMPT = """
# Persona:
# You are a Senior Python Library Architect; you transform fragmented helper modification advice into coherent, maintainable knowledge-libraries by applying rigorous knowledge-engineering practice.

# Your Mission:
# You will receive a Python Library SIB blueprint and advice about this library.  
# Design a new SIB **Blueprint** that (1) merge the advice to modify the current SIB module and (2) create new SIB module if necessary. Make sure to prioritize modifying the existing SIB and only when you are absolutely certain that modifying the current SIB cannot provide a tool to address the present issue should you propose adding a new one.

# Static Inference Block (SIB) Definition
# A SIB is a reusable knowledge capsule that  
# • accepts a well-defined set of input facts (pre-conditions)  
# • instantly infers deterministic outputs (formulae, numbers, long–form explanations)  
# • returns a multi-paragraph explanation string that follows the standard template shown below  
# • groups as many original tools as logically coherent—functionality must remain complete and loss-less.


# **Input: Input SIB blueprint:**
# {blueprint}

# **Advice**
# {refinement_suggestions}

# **Core Requirements**
# 1. Mandatory SIB Metadata (document for *every* SIB):
# [SIB]<Insert detailed title here>
# [Description]
# (Provide a concise, high-level summary in plain language describing what this SIB is and what problem it solves.)
# [Known Inputs]
# (List all known preconditions, variables, or parameters required to perform this reasoning, and clarify their meaning.)
# [Assumptions]
# (Explicitly list all idealized assumptions or constraints that this reasoning pattern relies on to be valid.)
# [Applicability & Scope]
# (Describe the specific scenarios and scope boundaries where this reasoning pattern is applicable.)
# [Derivation Steps]
# (Show, step-by-step and clearly, the complete logical chain for deriving the final result from the inputs and assumptions.)
# [Deterministic Outputs]
# **List exhaustively every quantity, formula, or explanatory statement that becomes unambiguously known once the *Known Inputs* and *Assumptions* are satisfied.  Nothing that can be deduced may be omitted.**  
# [Further Inference]  
# (Describe valuable next steps that would require *additional* information beyond the current Known Inputs, and outline how to proceed.)
# [Common Pitfall]
# (List the common mistakes, misunderstandings, or traps that beginners or users are likely to encounter when applying this pattern.)
# 2. Never create umbrella names such as *PhysicsToolkit*. Do **not** copy any original tool function names verbatim.
# 3. You must thoroughly take all advice into account. Please note that the number of advice items is large and they can be easily overlooked. Therefore, we require you to track the modifications for every single piece of advice. Specifically, at the beginning, you should provide a brief sentence for each piece of advice explaining how you plan to modify or add content. After that, present all of the SIB modules.
# 4. Do not omit any of the previous SIB modules, and do not add any marks indicating which modules are old and which are new. You must generate the complete set of SIB modules in full, as the previous version has been discarded.
# 5. Make sure to prioritize modifying the existing SIB and only when you are absolutely certain that modifying the current SIB cannot provide a tool to address the present issue should you propose adding a new one.

# **Blueprint Output Format:**
# Sugguestion Tracking: Sugguestion 1: (a short sentence to say how to merge Sugguestion 1 in blueprint);...

# <SIB>
# [SIB1_name]:...
# [Description]...
# [Known Inputs]...
# ...
# </SIB>
# <SIB>
# [SIB2_name]: ...
# </SIB>
# ...

# **Deliverable:**
# Your final output is the complete Sugguestion Tracking and Refactoring Blueprint in markdown format. Do not include any Python implementation code. Follow the core requirements.
# """


# LIB_REFINEMENT_BLUEPRINT_PROMPT = """
# Persona:
# You are a Senior Python Library Architect; you transform fragmented helper modification advice into coherent, maintainable knowledge-libraries by applying rigorous knowledge-engineering practice.

# Your Mission:
# You will receive a Python Library SIB blueprint and advice about this library.  
# Design a new SIB **Blueprint** that (1) merge the advice to modify the current SIB module and (2) create new SIB module if necessary. Make sure to prioritize modifying the existing SIB and only when you are absolutely certain that modifying the current SIB cannot provide a tool to address the present issue should you propose adding a new one.

# Static Inference Block (SIB) Definition
# A SIB is a reusable knowledge capsule that  
# • accepts a well-defined set of input facts (pre-conditions)  
# • instantly infers deterministic outputs (formulae, numbers, long–form explanations)  
# • returns a multi-paragraph explanation string that follows the standard template shown below  
# • groups as many original tools as logically coherent—functionality must remain complete and loss-less.


# **Input: Input SIB blueprint:**
# {blueprint}

# **Advice**
# {refinement_suggestions}

# **Core Requirements**
# 1. You must thoroughly take all advice into account. Please note that the number of advice items is large and they can be easily overlooked. Therefore, we require you to track the modifications for every single piece of advice. Specifically, at the beginning, you should provide a brief sentence for each piece of advice explaining how you plan to modify or add content. After that, present all of the SIB modules.
# 2. Do not omit any of the previous SIB modules, and do not add any marks indicating which modules are old and which are new. You must generate the complete set of SIB modules in full, as the previous version has been discarded.
# 3. Make sure to prioritize modifying the existing SIB and only when you are absolutely certain that modifying the current SIB cannot provide a tool to address the present issue should you propose adding a new one.

# **Blueprint Output Format:**
# Sugguestion Tracking: Sugguestion 1: (a short sentence to say how to merge Sugguestion 1 in blueprint);...


# **Deliverable:**
# Your final output is the complete Sugguestion Tracking and Refactoring Blueprint in markdown format. Do not include any Python implementation code. Follow the core requirements.
# """


LIB_REFINEMENT_BLUEPRINT_PROMPT = """
Persona:
You are a Senior Python Library Architect; you transform fragmented helper modification advice into coherent, maintainable knowledge-libraries by applying rigorous knowledge-engineering practice.

Your Mission:
You will receive a Python Library SIB blueprint and advice about this library.  
Design a new SIB **Blueprint** that (1) merge the advice to modify the current SIB module and (2) create new SIB module if necessary. Make sure to prioritize modifying the existing SIB and only when you are absolutely certain that modifying the current SIB cannot provide a tool to address the present issue should you propose adding a new one.

Static Inference Block (SIB) Definition
A SIB is a reusable knowledge capsule that  
• accepts a well-defined set of input facts (pre-conditions)  
• instantly infers deterministic outputs (formulae, numbers, long–form explanations)  
• returns a multi-paragraph explanation string that follows the standard template shown below  
• groups as many original tools as logically coherent—functionality must remain complete and loss-less.


**Input: Input SIB blueprint:**
{blueprint}

**Advice**
{refinement_suggestions}

**Core Requirements**
1. You must thoroughly take all advice into account. Please note that the number of advice items is large and they can be easily overlooked. Therefore, we require you to track the modifications for every single piece of advice. Specifically, at the beginning, you should provide a brief sentence for each piece of advice explaining how you plan to modify or add content. After that, present all of the SIB modules.
2. Do not omit any of the previous SIB modules, and do not add any marks indicating which modules are old and which are new. You must generate the complete set of SIB modules in full, as the previous version has been discarded.
3. Make sure to prioritize modifying the existing SIB and only when you are absolutely certain that modifying the current SIB cannot provide a tool to address the present issue should you propose adding a new one.

**Blueprint Output Format:**
Sugguestion Tracking: Sugguestion 1: (a short sentence to say how to merge Sugguestion 1 in blueprint);...


**Deliverable:**
Your final output is the complete Sugguestion Tracking and Refactoring Blueprint in markdown format. Do not include any Python implementation code. Follow the core requirements.
"""



CODE_INSPECTOR_PROMPT_REVISE = """
Code:
{code}

Error:
{error}

Requirements:
Your response must be only the code fix presented in the unified diff format.
Unified Diff Format Requirements:
Start with a header like --- a/original_code.py and +++ b/fixed_code.py.
Include one or more "hunks" that start with @@ -old_line,old_count +new_line,new_count @@.
Prefix deleted lines with -.
Prefix added lines with +.
Prefix unchanged context lines with a single space.
Include at least 1-2 lines of context around the change.
Do NOT include any explanations, comments, apologies, or any text whatsoever outside of the diff itself. Your entire response must be the raw diff.

**Output Format:**
<diff>
...
</diff>
"""

