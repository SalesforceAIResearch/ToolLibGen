#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from utils import call_vllm_wo_tool

def test_randomness_with_seeds():
    """测试使用不同种子是否能产生不同的结果"""
    
    # 测试消息
    test_messages = [
        {"role": "user", "content": "Tell me a short, random fact about space."}
    ]
    
    # 用不同的种子调用4次
    results = []
    seeds = [1234, 5678, 9012, 3456]  # 不同的种子
    
    print("🔬 Testing randomness with different seeds...")
    print("=" * 50)
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Call #{i+1} with seed={seed} ---")
        
        result = call_vllm_wo_tool(
            messages=test_messages,
            model_name="/export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model",
            openai_api_base="http://localhost:8000/v1",
            seed=seed,
            max_tokens=100  # 限制token数量加快测试
        )
        
        answer = result.get("answer", "No answer returned.")
        results.append(answer)
        print(f"Answer: {answer}")
    
    # 验证结果
    print("\n" + "=" * 50)
    print("🔍 Verification Results:")
    
    unique_results = set(results)
    if len(unique_results) > 1:
        print("✅ SUCCESS: Different seeds produced different answers!")
        print(f"   Got {len(unique_results)} unique answers out of {len(results)} calls")
    else:
        print("❌ FAILURE: All seeds produced identical answers")
        print("   This suggests the seed parameter is not working properly")
    
    # 测试相同种子是否产生相同结果
    print("\n🔬 Testing determinism with same seed...")
    same_seed = 1111
    same_seed_results = []
    
    for i in range(2):
        print(f"\n--- Call #{i+1} with same seed={same_seed} ---")
        result = call_vllm_wo_tool(
            messages=test_messages,
            model_name="/export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model",
            openai_api_base="http://localhost:8000/v1",
            seed=same_seed,
            max_tokens=100
        )
        answer = result.get("answer", "No answer returned.")
        same_seed_results.append(answer)
        print(f"Answer: {answer}")
    
    print("\n" + "=" * 50)
    print("🔍 Determinism Results:")
    
    if same_seed_results[0] == same_seed_results[1]:
        print("✅ SUCCESS: Same seed produced identical answers!")
        print("   This confirms the seed parameter is working correctly")
    else:
        print("❌ FAILURE: Same seed produced different answers")
        print("   This suggests there might be other sources of randomness")

if __name__ == "__main__":
    test_randomness_with_seeds() 