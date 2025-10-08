#!/usr/bin/env python3
"""
Test script for Unified Dynamic Tool System
测试统一动态工具系统的功能
"""

import sys
import os
import json
import numpy as np
from unittest.mock import Mock, patch

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from unified_dynamic_tools import (
        ToolFormatDetector,
        UnifiedToolRetriever,
        UnifiedDynamicToolManager,
        GPT41Client,
        get_unified_tool_embedding,
        create_unified_tool_system
    )
    print("✅ Successfully imported unified_dynamic_tools")
except ImportError as e:
    print(f"❌ Failed to import unified_dynamic_tools: {e}")
    sys.exit(1)


def create_sample_v2_tool():
    """Create a sample v_2.json format tool"""
    return {
        "tool_info": {
            "function": {
                "name": "calculate_energy",
                "description": "Calculate molecular energy using quantum mechanics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "molecule": {
                            "type": "string",
                            "description": "Molecular formula"
                        }
                    },
                    "required": ["molecule"]
                }
            }
        },
        "tool_code": """
def execute(molecule):
    # Simple energy calculation
    energy = len(molecule) * 10.5  # Simplified calculation
    return f"Energy of {molecule}: {energy} kJ/mol"
"""
    }


def create_sample_valid_science_tool():
    """Create a sample valid_science_toolset format tool"""
    return {
        "description": {
            "function": {
                "name": "calculate_mass",
                "description": "Calculate molecular mass",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "formula": {
                            "type": "string",
                            "description": "Chemical formula"
                        }
                    },
                    "required": ["formula"]
                }
            }
        },
        "python": """
def calculate_mass(formula):
    # Simple mass calculation
    mass = len(formula) * 2.5  # Simplified calculation
    return f"Mass of {formula}: {mass} g/mol"
"""
    }


def test_format_detection():
    """Test format detection functionality"""
    print("\n🔍 Testing Format Detection...")
    
    # Test v_2.json format
    v2_tools = [create_sample_v2_tool()]
    format_v2 = ToolFormatDetector.detect_format(v2_tools)
    print(f"  v_2.json format detected: {format_v2}")
    assert format_v2 == "v2", f"Expected 'v2', got '{format_v2}'"
    
    # Test valid_science_toolset format
    valid_science_tools = [create_sample_valid_science_tool()]
    format_valid = ToolFormatDetector.detect_format(valid_science_tools)
    print(f"  valid_science format detected: {format_valid}")
    assert format_valid == "valid_science", f"Expected 'valid_science', got '{format_valid}'"
    
    print("✅ Format detection tests passed!")


def test_tool_retriever():
    """Test unified tool retriever functionality"""
    print("\n🔍 Testing Tool Retriever...")
    
    # Create sample tools
    v2_tools = [create_sample_v2_tool()]
    valid_science_tools = [create_sample_valid_science_tool()]
    
    # Create sample embeddings
    sample_embeddings = np.random.rand(2, 1024).astype('float32')
    
    # Test v_2.json retriever
    retriever_v2 = UnifiedToolRetriever(v2_tools, sample_embeddings[:1])
    print(f"  v_2 retriever format: {retriever_v2.format}")
    assert retriever_v2.format == "v2"
    
    # Test valid_science retriever
    retriever_valid = UnifiedToolRetriever(valid_science_tools, sample_embeddings[:1])
    print(f"  valid_science retriever format: {retriever_valid.format}")
    assert retriever_valid.format == "valid_science"
    
    print("✅ Tool retriever tests passed!")


def test_tool_manager():
    """Test unified dynamic tool manager"""
    print("\n🔍 Testing Tool Manager...")
    
    # Create sample tools and embeddings
    v2_tools = [create_sample_v2_tool()]
    sample_embeddings = np.random.rand(1, 1024).astype('float32')
    
    # Create retriever and manager
    retriever = UnifiedToolRetriever(v2_tools, sample_embeddings)
    manager = UnifiedDynamicToolManager(retriever)
    
    # Test initial state
    initial_tools = manager.get_available_tools()
    print(f"  Initial tools count: {len(initial_tools)}")
    assert len(initial_tools) == 1, "Should start with 1 tool (retrieve_relevant_tools)"
    
    # Test function registry
    registry = manager.get_function_registry()
    print(f"  Initial function registry size: {len(registry)}")
    assert "retrieve_relevant_tools" in registry
    
    print("✅ Tool manager tests passed!")


def test_executable_function_creation():
    """Test executable function creation for different formats"""
    print("\n🔍 Testing Executable Function Creation...")
    
    # Test v_2.json format
    v2_tool = create_sample_v2_tool()
    v2_tool['format'] = 'v2'  # Add format info
    
    # Create sample embeddings
    sample_embeddings = np.random.rand(1, 1024).astype('float32')
    
    # Create retriever and manager
    retriever = UnifiedToolRetriever([v2_tool], sample_embeddings)
    manager = UnifiedDynamicToolManager(retriever)
    
    # Test function creation
    func = manager.create_executable_function(v2_tool)
    print(f"  v_2.json function created: {func is not None}")
    assert func is not None, "Should create executable function"
    
    # Test function execution
    try:
        result = func(molecule="H2O")
        print(f"  v_2.json function result: {result}")
        assert "Energy of H2O" in result
    except Exception as e:
        print(f"  v_2.json function execution failed: {e}")
        assert False, f"Function execution should not fail: {e}"
    
    # Test valid_science format
    valid_science_tool = create_sample_valid_science_tool()
    
    # Create a separate retriever for valid_science format to ensure proper format detection
    valid_science_retriever = UnifiedToolRetriever([valid_science_tool], sample_embeddings)
    valid_science_manager = UnifiedDynamicToolManager(valid_science_retriever)
    
    func_valid = valid_science_manager.create_executable_function(valid_science_tool)
    print(f"  valid_science function created: {func_valid is not None}")
    assert func_valid is not None, "Should create executable function"
    
    # Test function execution
    try:
        result_valid = func_valid(formula="CO2")
        print(f"  valid_science function result: {result_valid}")
        assert "Mass of CO2" in result_valid
    except Exception as e:
        print(f"  valid_science function execution failed: {e}")
        assert False, f"Function execution should not fail: {e}"
    
    print("✅ Executable function creation tests passed!")


def test_gpt41_client():
    """Test GPT-4.1 client initialization"""
    print("\n🔍 Testing GPT-4.1 Client...")
    
    # Test client initialization
    test_api_key = "sk-test-api-key"
    client = GPT41Client(test_api_key)
    
    print(f"  Client API key set: {client.client.api_key == test_api_key}")
    print(f"  Client model name: {client.model_name}")
    
    assert client.client.api_key == test_api_key
    assert client.model_name == "gpt-4-1106-preview"
    
    print("✅ GPT-4.1 client tests passed!")


def test_mock_integration():
    """Test integration with mocked API calls"""
    print("\n🔍 Testing Mock Integration...")
    
    # Create sample tools
    v2_tools = [create_sample_v2_tool()]
    sample_embeddings = np.random.rand(1, 1024).astype('float32')
    
    # Create unified system
    tool_manager = create_unified_tool_system(v2_tools, sample_embeddings)
    
    # Test with mock embedding API
    with patch('unified_dynamic_tools.call_sfr_embedding_api_lst') as mock_embedding:
        mock_embedding.return_value = [np.random.rand(1024).tolist()]
        
        # Test tool retrieval
        result = tool_manager.retrieve_relevant_tools("calculate energy", k=1)
        print(f"  Mock retrieval result: {result['success']}")
        assert result['success'] == True
    
    print("✅ Mock integration tests passed!")


def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Unified Dynamic Tool System Tests...")
    
    try:
        test_format_detection()
        test_tool_retriever()
        test_tool_manager()
        test_executable_function_creation()
        test_gpt41_client()
        test_mock_integration()
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 