# CallGraph Model Level Forward/Backward Statistics 工具文档

## 概述

`call_graph_model_level_fwd_bwd_statistics.py` 是一个基于 HTA (Holistic Trace Analysis) 库的深度学习训练性能分析工具。该工具专门用于分析 Megatron-LM 等大规模模型训练框架的 GPU 跟踪数据，计算模型层级的前向传播（forward）和后向传播（backward）的详细统计信息。

## 主要功能

1. **跟踪数据加载与分析**：加载 GPU 跟踪数据并构建调用图（CallGraph）
2. **模型层级分析**：按照预定义的模型模板分析各个层级的性能
3. **前向/后向传播统计**：计算每个模型层的前向和后向传播时间统计
4. **百分比分析**：计算每个层在前向/后向传播中的时间占比
5. **详细报告生成**：输出结构化的统计报告，包含均值、分位数、最大值、最小值、方差等

## 依赖项

### 核心依赖
- `hta` (Holistic Trace Analysis) 库
- `pandas` 和 `numpy` 用于数据处理
- `call_graph_template.py` 中的模板定义


### 文件结构依赖
```
musa_examples/
├── call_graph_model_level_fwd_bwd_statistics.py  # 主脚本
├── call_graph_template.py                        # 模板定义
├── musa_fwdbwd_util.py                          # 前向/后向计算工具
├── musa_basic_kernel_info.py                    # 基础内核信息计算
└── 其他相关文件
```

## 使用方法

### 1. 基本运行

```bash
# 设置环境变量（禁用纳秒舍入）
export HTA_DISABLE_NS_ROUNDING=1

# 运行脚本
python call_graph_model_level_fwd_bwd_statistics.py
```

### 2. 配置修改

脚本中的主要配置参数：

```python
# 跟踪数据目录（第43行）
trace_dir = str(Path(base_dir).joinpath("epoverlap-16-drop"))

# 分析的rank（第50行）
if rank != 16:  # 默认只分析rank 16
    continue

# 输出文件（第102行）
with open(f"./epoverlap-{rank}-main-stack-iter20-drop.txt", "w") as f:
```

### 3. 自定义模板

脚本使用 `output_template_to_file_kimi_epoverlap` 模板定义模型结构。要修改分析层级，需要更新 `call_graph_template.py` 中的模板定义。

## 输出说明

### 1. 统计报告文件
脚本生成 `epoverlap-{rank}-main-stack-iter20-drop.txt` 文件，包含：
- 模型层级结构（缩进表示调用关系）
- 每个层的前向传播统计
- 每个层的后向传播统计

### 2. 统计指标
对于每个层，报告包含以下指标：
- **mean_percent**: 在前向/后向传播中的时间占比（%）
- **mean**: 平均持续时间（ms）
- **q_25, q_50, q_75**: 25%、50%、75%分位数
- **max/min**: 最大/最小持续时间
- **count**: 调用次数
- **var**: 方差

### 3. 示例输出格式
```
nn.Module: RMSNorm_0
    fwd: mean_percent: 0.85, mean: 0.12, q_25: 0.10, q_50: 0.12, q_75: 0.14, max: 0.18, min: 0.08, count: 20.00
    bwd: mean_percent: 0.92, mean: 0.15, q_25: 0.13, q_50: 0.15, q_75: 0.17, max: 0.21, min: 0.11, count: 20.00
```

## 核心函数说明

### 1. `calculate_statistics()`
计算数据框的统计信息：
- 输入：包含 `kernel_span` 列的数据框
- 输出：包含均值、分位数、极值、方差和计数的数据框

### 2. `cal_dur_percent()`
计算持续时间百分比：
- 根据前向/后向传播总时间计算每个层的占比

### 3. `get_forward_duration_uniq()` 和 `get_forward_duration_dup()`
获取前向传播持续时间：
- `uniq`: 处理唯一函数名
- `dup`: 处理有重复标记的函数

### 4. `get_backward_duration()`
获取后向传播持续时间：
- 通过调用图查找前向节点的后向子节点

## 模板系统

### 模板标记
- `@dup@`: 标记有重复调用的函数
- `@shape@`: 标记需要形状信息的函数

### 模板示例
```python
output_template_to_file_kimi_epoverlap = r"""
pretrain_kimi.py(\d+): <module>
    musa_patch/training.py(\d+): train_step
        megatron/core/pipeline_parallel/combined_1f1b.py(\d+): combined_forward_backward_step
            # ... 更多层级
"""
```

## 使用场景

### 1. 性能瓶颈分析
识别模型训练中的性能瓶颈层，优化计算效率。

### 2. 分布式训练优化
分析不同rank的性能差异，优化通信和计算重叠。

### 3. 模型架构评估
比较不同模型架构或优化策略的性能影响。

### 4. 硬件性能评估
评估GPU等硬件在不同模型层上的性能表现。

## 注意事项

1. **数据准备**：需要正确的GPU跟踪数据文件（JSON格式）
2. **内存需求**：大规模跟踪数据可能需要较大内存
3. **模板匹配**：确保模板定义与实际代码调用栈匹配
4. **rank选择**：默认只分析rank 16，可根据需要修改

## 扩展与定制

### 1. 添加新的分析指标
修改 `calculate_statistics()` 函数添加新的统计指标。

### 2. 支持新的模型架构
在 `call_graph_template.py` 中添加新的模板定义。

### 3. 输出格式定制
修改输出部分的代码，支持CSV、JSON等格式。

### 4. 批量分析
修改循环部分，支持多个rank的批量分析。

## 故障排除

### 常见问题
1. **导入错误**：确保所有依赖包已安装
2. **模板不匹配**：检查模板定义与实际调用栈
3. **内存不足**：减少分析的数据量或增加内存
4. **文件不存在**：确认跟踪数据路径正确

### 调试建议
1. 启用详细日志输出
2. 检查中间数据文件（如CSV输出）
3. 验证模板提取结果
4. 检查调用图构建是否正确

## 相关工具

- `call_graph_kernel_level_fwd_bwd_statistics.py`: 内核级别的统计工具
- `temporal_breakdown.py`: 时间分解分析工具
- `trace_etl.py`: 跟踪数据ETL工具

## 性能优化建议

1. **缓存中间结果**：使用pickle缓存分析结果
2. **并行处理**：对多个rank进行并行分析
3. **增量分析**：只分析变化的迭代
4. **数据采样**：对大规模数据进行分析采样

---

*此文档基于 `musa_examples/call_graph_model_level_fwd_bwd_statistics.py` 脚本分析生成，适用于深度学习训练性能分析和优化场景。*