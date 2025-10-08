#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from utils import call_vllm_wo_tool

def simulate_cot_eval_passk():
    """模拟cot_eval.py中的pass-k逻辑"""
    
    # 模拟数据
    test_data_item = {
        "id": 123,
        "question": "What is the primary reason for the occurrence of tides on Earth?",
        "answer": "c"
    }
    
    passk = 4  # 模拟args.passk
    
    print("🧪 Simulating CoT evaluation with pass-k...")
    print(f"📋 Question: {test_data_item['question']}")
    print(f"🎯 Expected answer: {test_data_item['answer']}")
    print(f"🔄 Pass-k: {passk}")
    print("=" * 80)
    
    results = []
    
    # 模拟cot_eval.py中的循环逻辑
    for i in range(passk):
        id = str(test_data_item["id"]) + f"_{i}"
        question = test_data_item["question"]
        
        # 模拟QUESTION_TEMPLATE
        prompt = f"""
Please answer the following question: {question}
Your last line should be your final answer and start with "Final Answer: YOUR_ALPHABETICAL_CHOICE".
"""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 生成唯一的种子：结合数据项ID和pass-k迭代次数（与cot_eval.py中相同的逻辑）
        unique_seed = (int(test_data_item["id"]) * 1000 + i) % 1_000_000
        
        print(f"\n--- Pass #{i+1} (ID: {id}, Seed: {unique_seed}) ---")
        
        response = call_vllm_wo_tool(
            messages=messages,
            model_name="/export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model",
            openai_api_base="http://localhost:8000/v1",
            seed=unique_seed,
            max_tokens=200  # 稍大一些以获得完整答案
        )
        
        reasoning_content = response["thinking"]
        response_content = response["answer"]
        
        results.append({
            "id": id,
            "seed": unique_seed,
            "thinking": reasoning_content,
            "answer": response_content
        })
        
        print(f"💭 Thinking: {reasoning_content[:100]}..." if len(reasoning_content) > 100 else f"💭 Thinking: {reasoning_content}")
        print(f"💬 Answer: {response_content}")
    
    # 验证结果多样性
    print("\n" + "=" * 80)
    print("🔍 Results Analysis:")
    
    unique_answers = set(result["answer"] for result in results)
    unique_thinking = set(result["thinking"] for result in results)
    
    print(f"📊 Total responses: {len(results)}")
    print(f"🔢 Unique answers: {len(unique_answers)}")
    print(f"🧠 Unique thinking: {len(unique_thinking)}")
    
    if len(unique_answers) > 1:
        print("✅ SUCCESS: Different seeds produced different answers!")
        print("🎉 The fix is working correctly!")
    else:
        print("❌ FAILURE: All seeds produced identical answers")
        print("😞 The fix may not be working as expected")
    
    if len(unique_thinking) > 1:
        print("✅ SUCCESS: Different seeds produced different reasoning!")
    else:
        print("❌ NOTICE: All seeds produced identical reasoning")
    
    # 显示所有答案用于比较
    print("\n📝 All responses for comparison:")
    for i, result in enumerate(results):
        print(f"\nPass #{i+1} (Seed: {result['seed']}):")
        print(f"  Answer: {result['answer'][:200]}..." if len(result['answer']) > 200 else f"  Answer: {result['answer']}")

if __name__ == "__main__":
    simulate_cot_eval_passk() 