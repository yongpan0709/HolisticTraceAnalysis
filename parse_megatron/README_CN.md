# Distribute Holistic Trace Analysis
Holistic Trace Analysis (HTA)是一种性能分析工具，用于识别分布式训练工作负载中的性能瓶颈. HTA通过分析[PyTorch
Profiler](https://github.com/pytorch/kineto)（也称为Kineto）收集的trace数据来实现这一点。然而，目前HTA的功能仅限于本地单节点分析。随着集群规模的扩大，需要处理的trace信息可能会超出单节点的处理能力。Distribute Holistic Trace Analysis (DHTA)拓展了HTA的能力，使其能够在多节点集群上并行分析trace数据，并汇总所有节点的分析结果进行集群层面的综合分析。

## 特性

DHTA在HTA的基础上提供下面的特性（修改的代码行数在2000行左右）：

1. __机间并行分析__：利用多台机器并行分析，以Pipeline Parallel Group为单位把任务平均分配到所有机器上
2. __机内并行分析__：每个Pipeline Parallel Group中包含多个rank，在单机上利用多进程对多个rank的trace进行并行分析
3. __集群综合分析__：在分析完所有Pipeline Parallel Group后，通过MPI把所有分析结果进行汇总，并进行集群层面的综合分析，例如异常算子监测
4. __Megatron Pipeline Parallel分析__：针对Megatron Pipeline Paralle的特性进行定制化分析，包括bubble占比、负载不均衡开销分析、Pipeline Parallel stage之间的数据流动可视化等等


## 安装及使用
```
git clone https://sh-code.mthreads.com/ai/HolisticTraceAnalysis
cd HolisticTraceAnalysis
git checkout megatron_dev
cd parse_megatron
bash dist_install_env.sh
bash dist_run.sh
# bash dist_stop.sh # to stop analysis processes on all ranks
```
运行之前，需要修改一些参数：

- HTA安装路径：`install_env.sh`
- trace文件夹路径: `parse_megatron.py`
- TP/DP/PP size: `parse_megatron.py`
- NUM_PP_GROUP (word_size/pp_size or tp_size * dp_size): `run.sh`

DHTA的分析是以Pipeline Parallel Group为基本单位来进行的，一个Pipeline Parallel Group作为一个任务，平均分配给多个节点进行分析。
在workspace/project_name目录下，会生成3个目录，log、output、trace。其中trace目录存储了每个Pipeline Parallel Group对应的所有rank的trace.json文件（以软链接的形式链接到原文件），log文件夹存储每个节点上的分析日志，output文件夹存储所有的分析结果。分析结果包括每个Pipeline Pipeline Group的本地分析结果和整个集群的分析结果。

本地分析结果包括：

1. Transformer Layer关键部分时间占比，以txt文件格式保存

2. 过滤出Transformer Layer关键部分的trace信息，以及把多rank上trace信息整合在一起的综合结果，以json文件格式保存

3. 针对Pipeline Parallel的bubble和负载不均衡造成的overhead的分析结果，以csv文件格式保存

整个集群的分析结果主要包括检测出的异常算子：

1. 同一个rank上不同layer之间的异常算子，反应出同一机器在时间上的不稳定性

2. 同一时间不同rank之间的异常算子，反应出不同机器之间在空间上的不稳定


## 文档
了解更多关于HTA功能和API的信息，请参阅[HTA官方文档](https://hta.readthedocs.io/en/latest/index.html)。