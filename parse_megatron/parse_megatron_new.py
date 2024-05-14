from hta.trace_analysis import TraceAnalysis, PipelineParallelGroupTraceAnalysis
from hta.utils.parallel_state import get_3d_parallel_groups, is_first_stage, is_last_stage, get_next_pipeline_rank
from hta.utils.utils import partition_files_across_directories, LogToFile, prepare_directory
from hta.common.trace_call_graph import CallGraph 
from hta.configs.config import logger
from hta.common.trace_df import calculate_flops_for_trace_df, calculate_comm_volume_for_trace_df


from mpi4py import MPI
import os
import pickle
import sys
import logging
import time
import pandas as pd
import multiprocessing as mp
import numpy as np

pd.set_option('display.max_columns', None)

trace_dir = '/home/dist/yiyuan/llama7b_trace_gpu_complete'
trace_dir_for_pp_group = os.path.join(trace_dir, 'pp_group')

TP_SIZE = 2
PP_SIZE = 4
DP_SIZE = 2
NUM_MICROBATCHES = 8

def load_trace_analyer(trace_dir):
    cache_path = os.path.join(trace_dir, 'analyzer_cache.pkl')
    # if os.path.exists(cache_path):
    if False:
        with open(cache_path, 'rb') as f:
            analyzer = pickle.load(f)
        print(f'analyzer has been loaded from {cache_path}')
    else:
        analyzer = PipelineParallelGroupTraceAnalysis(trace_dir=trace_dir, data_parallel_size=DP_SIZE, tensor_parallel_size=TP_SIZE, pipeline_parallel_size=PP_SIZE, num_microbatch=NUM_MICROBATCHES)
        with open(cache_path, 'wb') as f:
            pickle.dump(analyzer, f)
        print(f'analyzer has been saved to {cache_path}')
    return analyzer

def display_traces_info(traces):
    first_trace_df = next(iter(traces.values()))
    print(f'total {len(traces)} traces, and each trace has {len(first_trace_df)} items')
    print(first_trace_df['s_cat'].value_counts())

def process_single_pp_group(trace_dir):
    output_dir = os.path.join(trace_dir, 'output')
    prepare_directory(output_dir, force_clear=False)
    
    analyzer = load_trace_analyer(trace_dir)
    analyzer.analyze_pipeline_parallel()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    processor_name = MPI.Get_processor_name()

    # 为每个进程创建一个独立的日志文件
    log_filename = f"log_process_{rank}.txt"
    with LogToFile(filepath=log_filename):
        print(f"Process {rank} out of {size} on {processor_name}")
        logging.basicConfig(level=logging.DEBUG, filename=log_filename, filemode='w')
        all_data_parallel_group_ranks, all_tensor_parallel_group_ranks, all_pipeline_parallel_group_ranks = get_3d_parallel_groups(TP_SIZE, PP_SIZE, DP_SIZE)
        all_pp_group_sub_dirs = partition_files_across_directories(trace_dir, trace_dir_for_pp_group, all_pipeline_parallel_group_ranks, skip=(not rank == 0))
        time.sleep(3)

        num_folders = len(all_pp_group_sub_dirs)
        folders_per_process = num_folders // size
        remainder = num_folders % size

        # 为前 'remainder' 个进程分配额外的一个文件夹
        if rank < remainder:
            start_index = rank * (folders_per_process + 1)
            end_index = start_index + folders_per_process + 1
        else:
            start_index = remainder * (folders_per_process + 1) + (rank - remainder) * folders_per_process
            end_index = start_index + folders_per_process

        assigned_folders = all_pp_group_sub_dirs[start_index:end_index]
        print(f'Process {rank} on {processor_name}: assigned_folders={assigned_folders}')

        for folder in assigned_folders:
            process_single_pp_group(folder)

if __name__ == '__main__':
    # main()
    logging.basicConfig(level=logging.DEBUG)
    process_single_pp_group('/home/dist/yiyuan/llama7b_trace_gpu_complete/pp_group_0')