# ToolLibGen

A comprehensive framework for extracting, clustering, and aggregating reusable tools from Chain-of-Thought (CoT) reasoning data using Large Language Models.

## Setup

```bash
pip install -r requirements.txt
touch .env
echo "OPENAI_API_KEY=<YOUR_OPENAI_KEY>" >> .env
```

## Pipeline Overview

The framework consists of three main stages:

1. **Tool Creation**: Extract reusable tools from CoT reasoning data
2. **Clustering**: Group similar tools into hierarchical clusters  
3. **Aggregation**: Merge and optimize tools within clusters to create final tool libraries

## 1. Tool Creation & Clustering

Extract reusable tools from question-answer pairs with CoT reasoning.

### Basic Usage

```bash
cd src
export PYTHONPATH=$PYTHONPATH:$(pwd)
python create_specific_tool.py --file_path $INPUT_DATA --save_folder $SAVE_FOLDER --generation_model_name $MODEL_NAME --verification_model_name_lst $LLM_SOLVER_FOR_VERIFICATION
```

### Arguments

- `--file_path`: Path to input JSON file containing question-answer pairs
- `--save_folder`: Path to save extracted tools
- `--generation_model_name`: the model for tool creation and clustering
- `--verification_model_name_lst`: the model for verificaiton
- `--debug`: Enable debug mode (process only a samll samples)

An example of input file is in src/data/example_for_aggregation.json



### Output Files
- `*_extracted_tools.json`: Flattened Extracted Tools
- `*clustered_hierarchy*.json`: Hierarchical Cluster Structure
- `*clustered_assigned_tools*.json`: Tool Assignment
- `*merged_tools.json*.json`: All created tools and their assignments


## 2. Tool Aggregation

Merge and optimize tools within clusters to create final consolidated tool libraries.

### Basic Usage

```bash
python aggregate_tools.py --file $merged_tools_json --model_name $MODEL_NAME --verification_model_name $VERIFICATION_MODEL
```

### Arguments

- `--file`: Path to clustered tools JSON file
- `--model-name`: LLM model to use
- `--verification_model_name`: LLM solver for verification


### Aggregation Process

1. **Blueprint Design**: Create high-level design for each cluster
2. **SIB Processing**: Process blocks in parallel
   - Implementation: Generate optimized code
   - Validation: Test tool functionality
   - Optimization: Iterative refinement
3. **Library Generation**: Create final tool libraries with OpenAI schemas

## Output Structure

The final output includes:
- **Tool Libraries**: Optimized Python functions with OpenAI schemas

## 3. Evaluation

```bash
python eval.py --input_data_path $test_file_path --tool_path $library_path --model_name $model_to_test
```

An example of test file is in src/data/example_test.json
