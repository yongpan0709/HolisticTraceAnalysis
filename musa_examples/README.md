# musa_examples：DHTA 分布式 Trace 分析工具

## 概述

`musa_examples/` 目录的核心是 DHTA（Distributed Holistic Trace Analysis）分布式分析工具。DHTA 基于 HTA（Holistic Trace Analysis）的 trace 解析与基础分析能力，面向 Megatron-LM 大规模分布式训练场景，将超大规模 GPU/Rank tracs 按 Pipeline Parallel Group 分发到多节点并行处理，并在节点间聚合分析结果。

DHTA 的目标不仅是简单地跑单机脚本，而是解决集群规模下 trace 数据量大、单节点处理慢、跨 rank 性能差异难定位的问题。它重点支持 Megatron Pipeline Parallel 场景下的集群级性能瓶颈分析、异常算子检测、Pipeline bubble 分析和负载不均衡分析。

除 DHTA 主流程外，本目录还包含若干基于调用图模板的模型层级、内核层级分析脚本，可用于对单个 rank 或局部 trace 进行更细粒度的 forward/backward 统计。

## 目录定位

`musa_examples/` 主要包含2类工具：

1. **DHTA 分布式分析主流程**
   - 按 Pipeline Parallel Group 对 trace 分组
   - 多节点、多进程并行执行分析
   - 汇总生成集群级结果
   - 支持异常耗时、Pipeline bubble、负载不均衡等分析。

2. **模型 / 内核层级统计脚本**
   - 基于 `CallGraph` 和模板匹配模型结构。
   - 统计模型组件的 forward/backward 时间。
   - 对指定 kernel 进一步计算 shape、TFLOPS、带宽等指标。

## DHTA 分布式分析

### 设计目标

DHTA 用于扩展 HTA 的分析能力，使其适用于基于Pytorch + Megatron-LM 的大规模分布式训练。它以 Pipeline Parallel Group 为基本任务单元，将不同 PP group 分发到多台机器处理，每台机器内部再并行处理 group 内多个 rank，最后通过 MPI 聚合集群级结果。

### 主要能力(规划)

1. **按 Pipeline Parallel Group 分发分析任务**
   - 每个 PP group 对应一组 rank。
   - 每组 trace 可分配到不同节点处理。
   - 降低单节点 trace 加载、解析和分析压力。

2. **节点内多 rank 并行处理**
   - 单节点内可通过多进程处理同一 PP group 内的多个 rank。
   - 提升单组 trace 的解析和报告生成效率。

3. **集群级结果聚合**
   - 各节点完成本地分析后，通过 MPI 汇总结果。
   - 支持从局部 rank 分析提升到全局集群视角。

4. **Megatron Pipeline Parallel 专项分析**
   - 分析 Pipeline bubble 占比。
   - 分析不同 pipeline stage 的负载不均衡。
   - 可视化或导出不同 stage 之间的数据流和等待关系。

5. **异常算子检测**
   - 比较同一 rank 上不同 layer 之间的耗时差异，定位时间不稳定的算子。
   - 比较同一时间不同 rank 之间的耗时差异，定位跨机器或跨卡的空间不稳定性。

### 典型工作流

```bash
# 1. 获取代码
git clone https://sh-code.mthreads.com/ai/HolisticTraceAnalysis
cd HolisticTraceAnalysis

# 如需切换到特定分支，请按实际开发分支执行 git checkout

# 安装单机
pip install -r requirements.txt
pip install -e .

# 构建wheel
pip wheel . --wheel-dir=dist/ --no-deps --use-pep517 --no-build-isolation

# 或直接安装whl
pip install traceinsight-*-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装多机(当1000卡甚至更大规模的trace需要分析时，可以支持多机并行分析)
cd musa_examples/
bash install_hta.sh <HolisticTraceAnalysis_Path>  # 需要hostfile

```

> 具体脚本名称和路径可能随实验分支调整，请以当前目录下实际文件为准。

### 运行前需要关注的配置

通常需要根据当前训练任务和集群环境修改以下内容：

- **HTA 安装路径**：用于确保各节点能够导入本地 HTA 代码。
- **Trace 根目录**：原始 Megatron trace 所在路径。
- **TP / DP / PP / EP size**：张量并行、数据并行、流水线并行规模。
- **NUM_PP_GROUP**：通常为 `world_size / pp_size`，也可理解为 `tp_size * dp_size`。
- **Workspace 路径**：DHTA 中间文件、日志和输出结果目录。
- **节点与 rank 映射**：决定不同 PP group 分发到哪些机器处理。

### Workspace 与输出结构

DHTA 通常会在 `workspace/<project_name>/` 下生成以下目录：

```text
workspace/
└── <project_name>/
    ├── log/      # 各节点、各 rank 的分析日志
    ├── output/   # 本地 PP group 结果和集群级聚合结果
    └── trace/    # 每个 PP group 内各 rank 的 trace.json 软链接
        └── report-pp0.csv
```

本地分析结果通常包括：

1. Transformer Layer 关键组件的时间占比，保存为 txt 文件。
2. 关键组件对应的 trace 片段和多 rank 合并结果，保存为 json 文件。
3. Pipeline bubble 与负载不均衡开销分析，保存为 csv 文件。

### `parse_megatron.py` 命令行用法

`parse_megatron.py` 是 `DistributedMegatronTraceAnalysis` 的轻量入口脚本，用于按给定的 Megatron 并行配置直接启动分布式 trace 分析。当前版本支持通过命令行传入 trace 路径、TP/PP/DP/EP 并行规模、pipeline 调度方式、micro batch 数量、virtual pipeline parallel size，以及可选的 PP group 范围过滤。

#### `parse_megatron.py` 运行常用参数

```bash
cd <HTA Path>
python -m musa_examples.parse_megatron \
  --trace-dir /path/to/trace-dir \
  --tp 1 \
  --pp 4 \
  --dp 2 \
  --ep 8 \
  --num-bs 16 \
  --pp-schedule 1f1b
```

#### 只分析其中部分 PP Group
如果只想分析部分 pipeline parallel group，可以增加 `--pp-group-id-range START END`，其中 `START` 和 `END` 都是闭区间端点：

```bash
cd <HTA Path>
python -m musa_examples.parse_megatron \
  --trace-dir /path/to/trace-dir \
  --tp 1 \
  --pp 4 \
  --dp 2 \
  --ep 8 \
  --num-bs 16 \
  --pp-schedule 1f1b \
  --pp-group-id-range 0 3
```

#### VPP 分析
当前支持的调度方式为 `1f1b`、`1f1b-interleaved` 和 `1f1b-interleaved-epoverlap`。对于 interleaved / EP overlap 场景，可额外指定 `--vpp`：

```bash
cd <HTA Path>
python -m musa_examples.parse_megatron \
  --trace-dir /path/to/trace-dir \
  --tp 1 \
  --pp 2 \
  --dp 1 \
  --ep 8 \
  --num-bs 16 \
  --vpp 2 \
  --pp-schedule 1f1b-interleaved
```

#### MPI 多机并行分析加速
如果需要多机运行，可使用 MPI 启动该脚本，例如文件末尾保留的示例命令所示：
* -np = hostfile ip数量 * <cnt> (对应ppr:<cnt>:node)
* ppr:<cnt>:node 代表每个机器起<cnt>个进程，下面过滤的流程不会爆 cpu 内存，所以建议ppr::node建议设成2~4

```bash
# 2机，每机启动一个进程
# hostfile中，需要准备两台服务器对应IP, 并且都安装了HTA，trace放在共享目录/存储中
mpirun -allow-run-as-root -np 2 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:1:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m musa_examples.parse_megatron --trace-dir /path/to/trace-dir

# 2机，每机启动一个进程
# hostfile中，需要准备两台服务器对应IP, 并且都安装了HTA，trace放在共享目录/存储中
mpirun -allow-run-as-root -np 16 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:8:node --wdir /path/to/HolisticTraceAnalysis \
  python -m musa_examples.parse_megatron --trace-dir /path/to/trace-dir \
  --tp 1 --pp 31 --dp 3 --num-bs 128
```

### `parse_megatron.py` 命令行参数

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--trace-dir` | 是 | 无 | trace 目录。 |
| `--tp` | 否 | `1` | Tensor Parallel size。 |
| `--pp` | 否 | `2` | Pipeline Parallel size。 |
| `--dp` | 否 | `1` | Data Parallel size。 |
| `--ep` | 否 | `8` | Expert Parallel size。 |
| `--pp-schedule` | 否 | `1f1b` | pipeline 调度方式，可选：`1f1b`、`1f1b-interleaved`、`1f1b-interleaved-epoverlap`。 |
| `--num-bs` | 否 | `16` | micro batch 数量，传给 `micro_bs`。 |
| `--vpp` | 否 | `2` | Virtual Pipeline Parallel size。 |
| `--pp-group-id-range` | 否 | `None` | 仅分析指定的 PP group 闭区间，格式为 `START END`，例如 `0 3`。 |

集群级聚合结果通常包括：

1. **时间维度异常算子**：同一 rank 上不同 layer 之间的异常耗时。
2. **空间维度异常算子**：同一时间不同 rank 之间的异常耗时。
3. **Pipeline stage 对比结果**：用于分析 stage 间等待、bubble 和负载差异。

## 关键文件说明

### DHTA 主流程相关

- `parse_megatron.py`
  - `DistributedMegatronTraceAnalysis` 的当前命令行入口。
  - 用于按给定的 TP / PP / DP / EP 配置、pipeline 调度方式和可选 PP group 范围直接启动分布式分析。

- `megatron_pipeline_group/distribute_trace_analysis.py`
  - DHTA 的核心编排脚本。
  - 负责按 Pipeline Parallel Group 切分 trace、调度本地分析、生成报告，并执行 MPI 聚合与异常检测。

- `trace_etl.py`
  - Trace ETL 入口脚本。
  - 用于对原始 trace 做过滤与修复，并生成 `*-etl` 目录供后续分布式分析使用。

### 模型 / 内核层级统计相关

- `call_graph_model_level_fwd_bwd_statistics.py`
  - 基于调用图模板的模型层级 forward/backward 统计脚本。
  - 适合分析单个 rank 中模型组件的耗时占比。

- `call_graph_kernel_level_fwd_bwd_statistics.py`
  - 当前主要提供一组 kernel 层级 shape / TFLOPS / 带宽提取函数与模板。
  - 脚本入口仍偏实验性质，默认通过内嵌路径、固定 rank 和输出文件名运行，适合二次改造或交互式分析，不像模型层级脚本那样已经完成通用 CLI 化。

- `call_graph_template.py`
  - 调用图模板定义文件。
  - 通过模板描述 Megatron 训练过程中的关键调用栈结构。

- `musa_fwdbwd_util.py`
  - forward/backward 匹配辅助逻辑。
  - 处理函数名重复、祖先上下文匹配、反向节点定位等问题。

- `musa_basic_kernel_info.py`
  - 基础 kernel 信息计算工具。

## Trace ETL 与过滤预处理

`trace_etl.py` 用于在正式执行 DHTA 之前，对原始 trace 做一次面向 Megatron 场景的清洗与重定向。当前脚本仍然是一个轻量 CLI 入口，但它已经固定了核心处理流程：修复部分 JSON 问题、过滤噪声函数、保留必要事件，并将结果输出到新的 `*-etl` 目录。

> 注意：`trace_etl.py` 现在同时兼容 `--trace-dir` 和历史参数 `--trace-dir`；为了和其他脚本保持一致，下面示例统一使用 `--trace-dir`。

### 当前处理逻辑

`trace_etl.py` 的主要步骤如下：

1. 通过 `--trace-dir` 指定原始 trace 目录。
2. 通过 `--tp`、`--pp`、`--dp`、`--ep` 提供当前 Megatron 并行配置。
3. 为每个 trace 文件先调用 `trace_json_repair.fix_json_value_missing()` 修复缺失值问题。
4. 过滤一批对分析帮助不大的 Python / runtime 噪声函数，例如 `__init__`、`threading.py(...)`、`socket.py(...)`、`multiprocessing/...` 等。
5. 对 `mooncake_p2p_recv_from` / `mooncake_p2p_send_to` 事件保留记录，但统一改写为 `user_annotation` 类别，便于后续分析。
6. 将过滤后的结果写入 `<trace-dir>-etl/`，再交给 `DistributedMegatronTraceAnalysis.pp_etl()` 按 PP group 组织。

### 基本运行

```bash
python -m musa_examples.trace_etl \
  --trace-dir /path/to/trace-dir \
  --tp 1 \
  --pp 4 \
  --dp 2 \
  --ep 8
```

如果需要多机并行执行 ETL，可按脚本中的示例通过 MPI 启动：

```bash
mpirun -allow-run-as-root -np 2 --bind-to none \
  --hostfile ./hostfile \
  --map-by ppr:1:node \
  --wdir /path/to/HolisticTraceAnalysis \
  python -m musa_examples.trace_etl \
    --trace-dir /path/to/trace-dir \
    --tp 1 --pp 4 --dp 2 --ep 8
```

### 命令行参数

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--trace-dir` | 是 | 无 | 原始 trace 目录。 |
| `--tp` | 是 | 无 | Tensor Parallel size。 |
| `--pp` | 是 | 无 | Pipeline Parallel size。 |
| `--dp` | 是 | 无 | Data Parallel size。 |
| `--ep` | 是 | 无 | Expert Parallel size。 |

### 输出结果

脚本会在原目录旁生成一个新的 ETL 目录：

```text
/path/to/trace-dir-etl/
```

该目录中的 trace 已完成基础过滤与修复，适合继续作为 `parse_megatron.py --trace-dir ...` 的输入。

## 模板驱动的模型层级 forward/backward 统计


虽然 DHTA 是 `musa_examples/` 的重点，但在定位单个 rank 或局部模型结构的性能问题时，`call_graph_model_level_fwd_bwd_statistics.py` 是最常用的细查脚本。当前版本已经改为命令行参数驱动，不再需要在脚本内手动修改 trace 路径、rank 和输出文件名。

### 当前处理逻辑

`call_graph_model_level_fwd_bwd_statistics.py` 的分析流程如下：

1. 通过 `--trace-dir` 指定 trace 目录，并使用 HTA 的 `get_trace_files()` 自动发现目录中的 rank trace 文件。
2. 如果指定 `--rank`，只分析该 rank；如果不指定 `--rank`，默认按 rank 顺序分析 trace 目录中的全部 rank。
3. 根据 `--template` 选择 `call_graph_template.py` 中的调用图模板，默认使用 `kimi_epoverlap`。
4. 对每个 rank：
   - 加载 trace 并 decode symbol，保留完整函数名。
   - 构建 `CallGraph`，获取该 rank 的 main stack。
   - 从模板中提取需要统计的函数节点及其祖先关系。
   - 对普通函数使用唯一名称匹配；对带 `@dup@` 标记的重复函数，结合祖先上下文匹配。
   - 对每个 forward 节点查找对应 backward 节点，并分别基于 `kernel_span` 计算统计值。
5. 每个 rank 输出一个文本报告，文件名为 `<template>-<rank>-main-stack.txt`。

### 基本运行

```bash
# 建议禁用纳秒舍入，保留更精确的 trace 时间
export HTA_DISABLE_NS_ROUNDING=1

# 分析单个 rank
python -m musa_examples.call_graph_model_level_fwd_bwd_statistics \
  --trace-dir /path/to/trace-dir \
  --rank 16 \
  --template kimi_epoverlap \
  --output-dir model_main_stack
```

输出示例：

```text
model_main_stack/
└── kimi_epoverlap-16-main-stack.txt
```

### 分析全部 rank

如果不传 `--rank`，脚本会分析 `--trace-dir` 中发现的全部 rank：

```bash
python -m musa_examples.call_graph_model_level_fwd_bwd_statistics \
  --trace-dir /path/to/trace-dir \
  --template kimi_epoverlap \
  --output-dir model_main_stack
```

输出示例：

```text
model_main_stack/
├── kimi_epoverlap-0-main-stack.txt
├── kimi_epoverlap-1-main-stack.txt
├── kimi_epoverlap-2-main-stack.txt
└── ...
```

### 命令行参数

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--trace-dir` | 是 | 无 | trace 目录，脚本会从该目录自动发现 rank trace 文件。 |
| `--rank` | 否 | `None` | 指定要分析的 rank；不指定时分析全部 rank。 |
| `--template` | 否 | `kimi_epoverlap` | 调用图模板名称。可选：`default`、`debug`、`kimi`、`kimi_epoverlap`。 |
| `--output-dir` / `--output` | 否 | `model_main_stack` | 输出目录。每个 rank 输出一个 `<template>-<rank>-main-stack.txt` 文件。 |

### 支持的模板

当前脚本内置的模板映射如下：

| 模板名 | 对应模板变量 | 适用场景 |
| --- | --- | --- |
| `default` | `output_template_to_file` | DeepSeek / MoE 通用模板。 |
| `debug` | `output_template_to_file_debug` | 小范围调用栈调试。 |
| `kimi` | `output_template_to_file_kimi` | Kimi 常规 Pipeline 调度模板。 |
| `kimi_epoverlap` | `output_template_to_file_kimi_epoverlap` | Kimi fine-grained / EP overlap 场景，当前默认模板。 |

模板定义位于 `call_graph_template.py`。如果模型代码、Megatron 版本或调度方式发生变化，应优先检查模板中的函数名和调用层级是否仍能匹配实际 trace。

### 输出指标

模型层级报告会按模板层级缩进输出每个函数节点的 forward 和 backward 统计信息。统计列基于 `kernel_span` 计算，并转换为毫秒级展示：

- `mean_percent`：该节点总耗时相对根节点 forward/backward 总耗时的比例值，计算方式为 `mean * count / total_mean`。
- `mean`：平均耗时。
- `q_25 / q_50 / q_75`：25%、50%、75% 分位数。
- `max / min`：最大值和最小值。
- `count`：参与统计的调用次数。

示例：

```text
# Rank: 16
pretrain_kimi.py(\d+): <module>
     fwd: mean_percent: 1.00, mean: 1234.56, q_25: 1200.00, q_50: 1230.00, q_75: 1260.00, max: 1300.00, min: 1180.00, count: 1.00
     bwd: mean_percent: 1.00, mean: 1100.00, q_25: 1080.00, q_50: 1100.00, q_75: 1120.00, max: 1150.00, min: 1050.00, count: 1.00
    megatron/core/pipeline_parallel/combined_1f1b.py(\d+): combined_forward_backward_step
         fwd: mean_percent: 0.95, mean: 1170.00, q_25: 1150.00, q_50: 1170.00, q_75: 1190.00, max: 1210.00, min: 1130.00, count: 1.00
         bwd: mean_percent: 0.92, mean: 1012.00, q_25: 990.00, q_50: 1010.00, q_75: 1035.00, max: 1060.00, min: 970.00, count: 1.00
```

### 模板标记

`call_graph_template.py` 中的模板支持特殊标记：

- `@dup@`：标记存在重复调用的函数。脚本会结合祖先上下文和已匹配节点索引区分不同位置的同名函数。
- `@shape@`：标记需要提取 shape 信息的函数或 kernel。该标记主要被 kernel 层级统计脚本使用，模型层级脚本会识别但不输出 shape 指标。

模板示例：

```python
output_template_to_file_kimi_epoverlap = r"""
pretrain_kimi.py(\d+): <module>
    musa_patch/training.py(\d+): train_step
        megatron/core/pipeline_parallel/combined_1f1b.py(\d+): combined_forward_backward_step
            megatron/core/models/common/model_chunk_schedule_plan.py(\d+): run
                megatron/core/models/gpt/fine_grained_callables.py(\d+): submodule_attn_forward
                    megatron/core/transformer/transformer_layer.py(\d+): _forward_attention
                        megatron/core/tensor_parallel/random.py(\d+): checkpoint @dup@
                            nn.Module: RMSNorm_0 @shape@
"""
```

### 常见使用建议

1. 先用 `--rank` 分析一个代表性 rank，确认模板能匹配到数据。
2. 如果输出为空或报 `No statistics were generated`，优先检查 `--template` 是否适配当前 trace 调用栈。
3. 如果 trace 很大，避免一开始不传 `--rank` 直接分析全部 rank。
4. 若某个函数名在模板中重复出现，需要添加 `@dup@` 并保证其祖先链路足够区分不同位置。
5. 若要分析 kernel shape、TFLOPS 或带宽，使用 `call_graph_kernel_level_fwd_bwd_statistics.py`，不要依赖模型层级脚本输出这些指标。

### Kernel 层级 shape / TFLOPS / 带宽分析

`call_graph_kernel_level_fwd_bwd_statistics.py` 当前更像一个实验性分析脚本，而不是已经封装完成的通用 CLI。README 这里更适合描述它“能做什么”和“当前怎么使用”，而不是把它写成一个已经参数化完成的命令行工具。

### 当前处理逻辑

该脚本当前提供的核心能力包括：

1. 基于调用图模板定位带 `@shape@` 标记的 forward kernel。
2. 从当前节点或其父节点提取 `input_dims` / `input_type`。
3. 对以下类型 kernel 计算 shape 相关指标：
   - `general_gemm`
   - `general_grouped_gemm`
   - `quantize`
   - `aten::_scaled_dot_product_attention_flash_musa`
4. 同时尝试关联 backward 路径中的对应 kernel，分别输出 `bwd-0`、`bwd-1` 等位置的统计。
5. 将结果写入文本文件，输出每个 kernel 的 shape、平均时间、TFLOPS 或带宽分位数。

### 当前脚本状态

与 `call_graph_model_level_fwd_bwd_statistics.py` 不同，这个脚本目前仍存在以下固定项：

- 入口使用 `if __name__ == "__main__":` 内嵌流程，而不是 `argparse`。
- 默认 `base_dir = "../"`，并将 `trace-dir` 固定指向 `../good_perf`。
- 默认只分析 `rank == 32`。
- 默认输出文件名形如 `20260211-<rank>-repo6.txt`。
- 启动前会显式把 `ParserConfig.ARGS_INPUT_SHAPE` 加入默认解析配置，以确保 trace 中的 shape 信息被提取出来。

### 当前使用方式

如果要直接使用该脚本，通常需要先按当前任务手动修改以下内容：

- `base_dir`
- `trace-dir`
- 目标 `rank`
- 输出文件名

也就是说，它更适合在研究某类 kernel 指标时作为分析模板或二次开发入口，而不是像 `parse_megatron.py`、`call_graph_model_level_fwd_bwd_statistics.py` 那样直接复用现成 CLI。

### 输出指标

脚本当前会围绕匹配到的 kernel 输出以下信息：

- `shape`
- `mean_time(us)`
- `TFLOPS` 或 `GB/s` 的平均值
- `q_25 / q_50 / q_75`
- `count`

### 使用建议

1. 先用模型层级脚本确认模板能稳定匹配到目标函数，再做 kernel 级 shape / TFLOPS 分析。
2. 运行前确认解析配置已经启用 `ParserConfig.ARGS_INPUT_SHAPE`，否则 shape 相关字段可能不存在。
3. 如果模型代码或 kernel 名称变化，优先更新 `call_graph_template.py` 中带 `@shape@` 的模板项，以及脚本内的 `SHAPE_POSITION_FWD_BWD` / `SHAPE_POSITION_FWD_BWD_OF_FLASH_ATTENTION` 映射。

## 适用场景

### DHTA 优先适用的场景

1. **大规模 Megatron Pipeline Parallel 训练分析**
   - trace 数据量较大，单机分析成本高。
   - 需要按 PP group 分发处理。

2. **集群级性能瓶颈定位**
   - 需要比较不同节点、不同 rank、不同 pipeline stage 的性能差异。

3. **Pipeline bubble 和负载不均衡分析**
   - 需要定位 stage 等待、通信阻塞或计算分布不均造成的开销。

4. **异常算子检测**
   - 需要判断某个算子是在单 rank 内不稳定，还是跨 rank / 跨机器存在系统性异常。

### 单 rank 统计脚本适用的场景

1. **模型局部性能拆解**
   - 查看某个 rank 上 Transformer Layer 内部各组件耗时。

2. **模板调试**
   - 验证调用图模板是否能正确匹配实际 trace。

3. **Kernel 级指标分析**
   - 针对特定算子计算 TFLOPS、带宽或 shape 相关指标。

## 注意事项

1. **Trace 数据准备**
   - 确保每个 rank 的 trace 文件完整且命名、目录结构符合脚本预期。

2. **并行配置一致性**
   - TP / DP / PP size、world size、rank 映射必须与训练任务一致。

3. **模板匹配准确性**
   - 模板中的函数名、调用层级和实际 trace 调用栈需要保持一致。

4. **内存与磁盘空间**
   - 大规模 trace 分析会占用较多内存和磁盘空间。
   - DHTA 可降低单节点压力，但 workspace 中仍会生成中间文件和聚合结果。

5. **实验脚本特性**
   - `musa_examples/` 中部分脚本包含实验相关路径和假设。
   - 迁移到新任务时，优先检查路径、rank 选择、并行配置和输出文件名。

## 故障排查

常见问题：

1. **导入错误**
   - 检查 HTA 是否已安装，或 `PYTHONPATH` 是否包含仓库根目录。

2. **trace 文件找不到**
   - 检查原始 trace 路径、workspace 软链接和 rank 命名规则。

3. **模板不匹配**
   - 检查 `call_graph_template.py` 中的模板是否与当前 Megatron 版本的调用栈一致。

4. **MPI 聚合失败**
   - 检查节点间环境、hostfile、rank 数和启动命令是否一致。

5. **分析结果为空或明显异常**
   - 检查 trace 是否包含目标 iteration、目标 rank 和 GPU kernel 事件。

## 扩展建议

1. **新增 DHTA 分析项**
   - 优先在分布式主流程中明确本地分析结果格式，再设计 MPI 聚合逻辑。

2. **支持新的 Megatron 调度方式**
   - 在调度分析器中补充对应 schedule 的事件识别和 stage 关系解析。

3. **新增模型结构模板**
   - 在 `call_graph_template.py` 中新增模板，并用单 rank 脚本验证匹配结果。

4. **新增 kernel 指标**
   - 在 kernel 层级统计脚本中扩展 shape 到 FLOPs / bandwidth 的映射规则。

---

本文档面向 `musa_examples/` 下的 DHTA 分布式分析流程，同时保留模型层级和内核层级统计脚本的使用说明。实际使用时建议先确认 DHTA 的 workspace、并行配置和 trace 组织方式，再根据需要使用单 rank 脚本做局部细查。
