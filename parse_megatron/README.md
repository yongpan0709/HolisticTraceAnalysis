# Distribute Holistic Trace Analysis
Holistic Trace Analysis (HTA) is a performance analysis tool designed to identify performance bottlenecks in distributed training workloads. HTA achieves this by analyzing traces collected through the PyTorch Profiler, also known as Kineto. However, the current capabilities of HTA are limited to local single-node analysis. As the cluster size grows, the volume of trace information can exceed the processing capability of a single node. Distribute Holistic Trace Analysis (DHTA) extends HTA's capabilities by enabling parallel trace analysis across multiple nodes and aggregating the results from all nodes for a comprehensive cluster-level analysis.

## Features
DHTA builds upon HTA with the following features (approximately 2000 lines of code modified):

1. Inter-node Parallel Analysis: Utilizes multiple machines to perform parallel analysis, distributing tasks across all machines based on Pipeline Parallel Groups.
2. Intra-node Parallel Analysis: Each Pipeline Parallel Group includes multiple ranks, and parallel analysis is conducted on these ranks using multiple processes within a single machine.
3. Cluster-wide Aggregated Analysis: After analyzing all Pipeline Parallel Groups, MPI is used to aggregate the results from all nodes and conduct cluster-level comprehensive analysis, such as anomaly detection for operators.
4. Megatron Pipeline Parallel Analysis: Custom analysis tailored to the characteristics of Megatron Pipeline Parallel, including bubble proportion, load imbalance overhead analysis, and visualization of data flow between Pipeline Parallel stages.

## Installation & Usage
```
git clone https://sh-code.mthreads.com/ai/HolisticTraceAnalysis
cd HolisticTraceAnalysis
git checkout megatron_dev
cd parse_megatron
bash dist_install_env.sh
bash dist_run.sh
# bash dist_stop.sh # to stop analysis processes on all ranks
```

Before running, some parameters need to be modified:

- HTA installation path: `install_env.sh`
- Trace folder path: `parse_megatron.py`
- TP/DP/PP size: `parse_megatron.py`
- NUM_PP_GROUP (word_size/pp_size or tp_size * dp_size): `run.sh`

DHTA performs analysis based on Pipeline Parallel Groups, treating each Pipeline Parallel Group as a task and distributing it across multiple nodes for analysis. In the workspace/project_name directory, three directories will be generated: log, output, and trace. The trace directory contains trace.json files for each rank within a Pipeline Parallel Group (linked as symbolic links to the original files). The log folder stores analysis logs for each node, while the output folder contains all analysis results. The analysis results include both local results for each Pipeline Parallel Group and cluster-wide results.

Local analysis results include:

1. Time proportion of key Transformer Layer components, saved in a txt file.
2. Trace information for key Transformer Layer components, along with consolidated results from multiple ranks, saved in a json file.
3. Analysis results of overhead caused by Pipeline Parallel bubbles and load imbalance, saved in a csv file.

Cluster-wide analysis results primarily include detected anomalies in operators:

1. Anomalous operators between different layers on the same rank, indicating temporal instability within the same machine.
2. Anomalous operators between different ranks at the same time, reflecting spatial instability between different machines.

## Documentation
Learn more about the features and the API from HTA documents [documentation](https://hta.readthedocs.io/en/latest/index.html).
