#!/usr/bin/env python3
"""
Example of how to use the tool retrieval functionality from flatten_eval.py
This demonstrates the reusability of the core classes and functions.
"""

import sys
import os

# Add the eval directory to the path to import flatten_eval
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))

from flatten_eval import (
    ToolRetriever, 
    DynamicToolManager, 
    create_tool_retrieval_system,
    TOOL_RETRIEVE_FUNCTION
)
from utils import read_json, read_pkl


def example_basic_usage():
    """Basic example of using the tool retrieval functionality."""
    print("=== Basic Tool Retrieval Example ===")
    
    # Load your tool data and embeddings
    tool_path = "/export/home/data/all_tagged_tools_20250619_054608.json"
    tool_embedding_path = "/export/home/data/all_tagged_tools_20250619_054608_embedding.pkl"
    
    # Check if files exist (adjust paths as needed)
    if not os.path.exists(tool_path):
        print(f"Tool data file not found: {tool_path}")
        print("Please adjust the paths in this example to match your setup.")
        return
    
    tool_data = read_json(tool_path)
    tool_embedding = read_pkl(tool_embedding_path)
    
    # Create the tool retrieval system with LLM query generation
    model_name = "/export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model"
    api_base = "http://localhost:8000/v1"
    tool_manager = create_tool_retrieval_system(
        tool_data, tool_embedding, model_name, api_base
    )
    
    # Example queries - now with LLM query generation
    queries = [
        "What's the molecular weight of caffeine and how do I calculate it?",
        "I need to analyze chemical reaction kinetics for my experiment",
        "How can I determine thermodynamic properties of a compound?",
        "What tools are available for process safety analysis?",
        "I want to look up material properties for my research"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        # Try with LLM query generation
        result = tool_manager.retrieve_relevant_tools(query, k=3, use_llm_query_generation=True)
        
        if result['success']:
            print(f"  Found {len(result['tools'])} relevant tools:")
            if result.get('used_llm_generation', False):
                print(f"  🤖 Generated optimized query: '{result['search_query']}'")
            else:
                print(f"  📝 Used direct query: '{result['search_query']}'")
            
            for tool in result['tools']:
                print(f"    - {tool['name']}: {tool['description'][:80]}...")
                print(f"      Relevance: {tool['relevance_score']:.3f}")
        else:
            print(f"  Error: {result['message']}")
    
    print(f"\nTotal available tools: {len(tool_manager.get_available_tools())}")
    print(f"Retrieved tools: {list(tool_manager.retrieved_tool_names)}")


def example_openai_integration():
    """Example of how to integrate with OpenAI-style function calling."""
    print("\n=== OpenAI Integration Example ===")
    
    # This is how you would use it in a chatbot or agent system
    tool_path = "/export/home/data/all_tagged_tools_20250619_054608.json"
    tool_embedding_path = "/export/home/data/all_tagged_tools_20250619_054608_embedding.pkl"
    
    if not os.path.exists(tool_path):
        print("Tool data files not found. Please adjust paths.")
        return
    
    tool_data = read_json(tool_path)
    tool_embedding = read_pkl(tool_embedding_path)
    
    # Initialize the system
    tool_manager = create_tool_retrieval_system(tool_data, tool_embedding)
    
    # Simulate a conversation where tools are dynamically retrieved
    user_messages = [
        "I need to calculate the molecular weight of caffeine",
        "What are the combustion properties of methane?",
        "Can you help me with thermodynamic calculations?"
    ]
    
    for i, message in enumerate(user_messages):
        print(f"\nTurn {i+1}: {message}")
        
        # Get currently available tools (including the retrieval function)
        available_tools = tool_manager.get_available_tools()
        print(f"Available tools: {len(available_tools)}")
        
        # If we only have the retrieval function, auto-retrieve relevant tools
        if len(available_tools) == 1:
            retrieval_result = tool_manager.retrieve_relevant_tools(message, k=3)
            if retrieval_result['success']:
                print(f"Auto-retrieved: {[tool['name'] for tool in retrieval_result['tools']]}")
                available_tools = tool_manager.get_available_tools()
        
        # Now available_tools contains both the retrieval function and retrieved tools
        print(f"Tools ready for LLM: {len(available_tools)}")
        for tool in available_tools:
            print(f"  - {tool['function']['name']}")


def example_standalone_retriever():
    """Example of using just the ToolRetriever class without the full manager."""
    print("\n=== Standalone Retriever Example ===")
    
    tool_path = "/export/home/data/all_tagged_tools_20250619_054608.json"
    tool_embedding_path = "/export/home/data/all_tagged_tools_20250619_054608_embedding.pkl"
    
    if not os.path.exists(tool_path):
        print("Tool data files not found. Please adjust paths.")
        return
    
    tool_data = read_json(tool_path)
    tool_embedding = read_pkl(tool_embedding_path)
    
    # Create just the retriever (lower-level interface)
    retriever = ToolRetriever(tool_data, tool_embedding)
    
    # Direct retrieval
    query = "chemical synthesis reaction"
    results = retriever.retrieve_top_k(query, k=5)
    
    print(f"Direct retrieval for '{query}':")
    for result in results:
        tool_name = result['description']['function']['name']
        description = result['description']['function']['description']
        score = result['relevance_score']
        print(f"  {result['rank']}. {tool_name} (score: {score:.3f})")
        print(f"     {description[:100]}...")
        
        # You can also get the OpenAI format
        openai_format = retriever.convert_tool_to_openai_format(result)
        print(f"     OpenAI name: {openai_format['function']['name']}")


def example_llm_query_generation_comparison():
    """Example comparing direct query vs LLM-generated query."""
    print("\n=== LLM Query Generation Comparison ===")
    
    tool_path = "/export/home/data/all_tagged_tools_20250619_054608.json"
    tool_embedding_path = "/export/home/data/all_tagged_tools_20250619_054608_embedding.pkl"
    
    if not os.path.exists(tool_path):
        print("Tool data files not found. Please adjust paths.")
        return
    
    tool_data = read_json(tool_path)
    tool_embedding = read_pkl(tool_embedding_path)
    
    # Initialize with LLM capabilities
    model_name = "/export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model"
    api_base = "http://localhost:8000/v1"
    tool_manager = create_tool_retrieval_system(
        tool_data, tool_embedding, model_name, api_base
    )
    
    test_question = "I'm trying to figure out if my chemical process is safe and what the explosion risks are"
    
    print(f"Test question: '{test_question}'\n")
    
    # Compare direct vs LLM-generated queries
    print("1. Using direct query:")
    result1 = tool_manager.retrieve_relevant_tools(test_question, k=3, use_llm_query_generation=False)
    if result1['success']:
        print(f"   Found {len(result1['tools'])} tools:")
        for tool in result1['tools']:
            print(f"     - {tool['name']} (score: {tool['relevance_score']:.3f})")
    
    # Reset the tool manager for fair comparison
    tool_manager.reset_tool_pool()
    
    print("\n2. Using LLM-generated query:")
    result2 = tool_manager.retrieve_relevant_tools(test_question, k=3, use_llm_query_generation=True)
    if result2['success']:
        if result2.get('used_llm_generation', False):
            print(f"   Generated query: '{result2['search_query']}'")
        print(f"   Found {len(result2['tools'])} tools:")
        for tool in result2['tools']:
            print(f"     - {tool['name']} (score: {tool['relevance_score']:.3f})")


if __name__ == "__main__":
    print("Tool Retrieval System Examples")
    print("=" * 50)
    
    # Make sure to adjust file paths before running
    try:
        example_basic_usage()
        example_openai_integration() 
        example_standalone_retriever()
        example_llm_query_generation_comparison()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print("\nTo use this in your own code:")
        print("1. Import the classes: from flatten_eval import ToolRetriever, DynamicToolManager")
        print("2. Load your tool data and embeddings")
        print("3. Create the system: tool_manager = create_tool_retrieval_system(tool_data, embeddings)")
        print("4. Use: tool_manager.retrieve_relevant_tools(query)")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure to adjust the file paths to match your setup.") 