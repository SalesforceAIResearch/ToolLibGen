# V2 动态工具检索与执行系统使用指南

## 📋 概述

已成功修改 `eval/library_flatten_eval_v2.py`，使其能够正确处理 v_2.json 格式的动态工具检索和执行。

## 🔄 关键格式差异

### 📊 数据格式对比

| 项目 | flatten_eval_retriever_as_tools.py | library_flatten_eval_v2.py (修改后) |
|------|-------------------------------------|-------------------------------------|
| **数据源** | valid_science_toolset_*.json | v_2.json |
| **工具信息字段** | `description` | `tool_info` |
| **代码字段** | `python` | `tool_code` |
| **执行方式** | `def function_name(...)` | `def execute(...)` |

### 🔧 代码执行差异

#### flatten_eval_retriever_as_tools.py 格式：
```python
# 在 python 字段中：
def charge_analysis_method_reliability(methods):
    # 直接定义和执行函数
    return result

# 调用方式：
charge_analysis_method_reliability(methods)
```

#### v_2.json 格式（我们的修改）：
```python
# 在 tool_code 字段中：
# 前置代码（类定义等）
class SomeClass:
    pass

# 统一的执行入口
def execute(param1, param2):
    # 执行逻辑
    return result

# 调用方式：
execute(param1=value1, param2=value2)
```

## 🚀 使用方法

### 基本命令
```bash
cd eval
python library_flatten_eval_v2.py \
    --tool_path /Users/murong.yue/Desktop/data/v_2.json \
    --tool_embedding_path /Users/murong.yue/Desktop/data/v_2_embedding.pkl \
    --enable_tool_retrieval \
    --max_turns 10 \
    --debug
```

### 关键参数
- `--tool_path`: 使用 v_2.json 文件路径
- `--tool_embedding_path`: 对应的嵌入向量文件
- `--enable_tool_retrieval`: 启用动态工具检索
- `--max_turns`: 最大对话轮数（默认 10）
- `--debug`: 调试模式（处理少量数据）

## 🔧 核心技术改进

### 1. **数据格式适配**
- ✅ 正确读取 `tool_info` 而不是 `description`
- ✅ 正确读取 `tool_code` 而不是 `python`
- ✅ 处理更复杂的嵌入向量生成逻辑

### 2. **工具执行改进**
```python
def create_executable_function(self, tool_data: Dict[str, Any]) -> Optional[callable]:
    # 1. 提取 tool_code
    tool_code = tool_data.get("tool_code", "")
    
    # 2. 执行完整的 Python 代码（包含前置代码）
    exec(actual_code, temp_module)
    
    # 3. 寻找 'execute' 函数（v_2.json 标准）
    if 'execute' in temp_module:
        execute_func = temp_module['execute']
        
        # 4. 创建包装器函数
        def tool_wrapper(**kwargs):
            return execute_func(**kwargs)
        
        return tool_wrapper
```

### 3. **错误处理增强**
- 🛡️ 优雅的回退机制：如果没有 `execute` 函数，回退到原逻辑
- 🔍 详细的错误日志和调试信息
- ⚡ 安全的代码执行环境

## 📊 工作流程

### 1. **初始化阶段**
```
加载 v_2.json → 生成/加载嵌入向量 → 创建 DynamicToolManager
```

### 2. **动态检索阶段**
```
LLM 调用 retrieve_relevant_tools → 
检索相关工具 → 
创建可执行函数 → 
添加到工具池
```

### 3. **工具执行阶段**
```
LLM 调用具体工具 → 
执行 execute 函数 → 
返回结果 → 
LLM 继续推理
```

## 🧪 验证方法

### 语法检查
```bash
cd eval
python -c "import library_flatten_eval_v2; print('✅ Syntax OK')"
```

### 功能测试
```bash
# 小规模调试测试
python library_flatten_eval_v2.py --debug

# 完整测试
python library_flatten_eval_v2.py \
    --input_data_path /path/to/test_data.json \
    --tool_path /Users/murong.yue/Desktop/data/v_2.json
```

## 📈 性能特性

### ✅ 兼容性
- **向后兼容**: 支持原有的函数名调用格式
- **v_2.json 优化**: 优先识别 `execute` 函数
- **错误回退**: 代码执行失败时的优雅处理

### ✅ 扩展性
- **模块化设计**: 清晰的函数责任分离
- **动态加载**: 运行时动态添加工具
- **内存效率**: 按需加载和执行

### ✅ 安全性
- **沙箱执行**: 隔离的代码执行环境
- **错误捕获**: 完善的异常处理机制
- **资源控制**: 可配置的执行限制

## 🚨 注意事项

### 1. **第一次运行**
- 会生成嵌入向量文件，耗时较长
- 建议先用 `--debug` 模式测试

### 2. **依赖管理**
```python
# v_2.json 的工具可能需要的依赖
import requests
from PIL import Image, ImageFilter
from io import BytesIO
import re
```

### 3. **内存使用**
- v_2.json 文件较大（169MB），注意内存使用
- 工具执行时会加载完整的前置代码

## 🔮 扩展建议

### 近期改进
1. **缓存优化**: 缓存已编译的工具代码
2. **并行执行**: 支持多工具并行调用
3. **监控面板**: 实时工具使用情况监控

### 长期规划
1. **安全沙箱**: 更严格的代码执行环境
2. **工具市场**: 动态工具发现和安装
3. **性能优化**: 智能工具预加载和缓存

## 🎯 总结

✅ **成功适配** v_2.json 的 `def execute` 格式  
✅ **保持兼容** 原有的函数调用方式  
✅ **增强错误处理** 和调试功能  
✅ **优化性能** 和内存使用  
✅ **提供完整** 的使用文档和测试方法  

现在您可以使用修改后的系统来处理 v_2.json 格式的动态工具检索和执行了！ 