from pathlib import Path 
from hta.common.trace import Trace
from hta.common.trace_file import get_trace_files
from hta.configs.config import logger
from hta.configs.parser_config import  ParserConfig, AVAILABLE_ARGS
from hta.common.trace_call_graph import CallGraph, CallStackIdentity
from collections import defaultdict
from typing import Dict, List, Set
import numpy as np
import pandas as pd
from collections import deque
from call_graph_template import extract_func_name_from_template, extract_dup_or_shape_func_name_from_template, output_template_to_file, SHAPE_POSITION, set_pandas_display_options, output_template_to_file_kimi, output_template_to_file_debug
import pickle
import re
from musa_fwdbwd_util import get_forward_duration_dup, get_forward_duration_uniq, get_backward_duration


def calculate_statistics(df: pd.DataFrame, func_name: str, calculate_col_name: str = 'kernel_span'): #, fwd_bwd: str = 'fwd'):
    df[calculate_col_name] = df[calculate_col_name]/1000.0
    df = df[df[calculate_col_name] > 0]
    df_dur = pd.DataFrame({
                        'mean': df[calculate_col_name].mean(),
                        'q_25': df[calculate_col_name].quantile(.25),
                        'q_50': df[calculate_col_name].quantile(.5),
                        'q_75': df[calculate_col_name].quantile(.75),
                        'max':  df[calculate_col_name].max(),
                        'min':  df[calculate_col_name].min(),
                        'var':  df[calculate_col_name].var(),
                        'count': df[calculate_col_name].count()}, index=[func_name], columns=['mean', 'q_25', 'q_50', 'q_75', 'max', 'min', 'var', 'count'])
    return df_dur


def cal_dur_percent(row, fwd_total_dur_mean, bwd_total_dur_mean):
    if row.name.endswith('-bwd'):
        return row['mean']*row['count']/bwd_total_dur_mean
    else:
        return row['mean']*row['count']/fwd_total_dur_mean

# HTA_DISABLE_NS_ROUNDING=1 python call_graph_kernel_dur_sum.py
if __name__ == "__main__":
    import time
    base_dir = "../"
    trace_dir = str(Path(base_dir).joinpath("fp8-0125"))
    cfg = ParserConfig.get_default_cfg()
    # config for extracting shape info
    cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
    ParserConfig.set_default_cfg(cfg)
    trace_files = get_trace_files(trace_dir)
    for rank, trace_file in trace_files.items():
        if rank != 8:
            continue
        # HTA starts from here
        t = Trace(trace_files={rank: trace_file}, trace_dir="")
        t.load_traces()
        # transform name and cat columns to s_name and s_cat
        # name and cat are kernel id
        t.decode_symbol_ids(use_shorten_name=False)
        set_pandas_display_options()
        t0 = time.perf_counter()
        cg = CallGraph(t)
        t1 = time.perf_counter()
        print(f"Rank {rank}, CallGraph took {t1 - t0:.2f} seconds")
        #df = cg.call_stacks[-1].full_df
        _, main_stack = cg.get_main_stack_on_rank(rank)
        df = main_stack.full_df
        #df.to_csv(f"./full_df_{rank}.csv", index=False)
        dup_func_name, need_shape_func_name = extract_dup_or_shape_func_name_from_template(output_template_to_file_kimi)
        func_name = extract_func_name_from_template(output_template_to_file_kimi)
        stat_info_funcs_grouped = pd.DataFrame()
        func_mapping_node_index: Dict[str, List[np.float64]] = defaultdict(list)

        for forward_func_name, func_ancestors in func_name:
            print(f"Processing function: {forward_func_name}")
            if forward_func_name.split('@')[0] not in dup_func_name:
                fwd_df = get_forward_duration_uniq(df, forward_func_name.split('@')[0])
            else:
                fwd_df = get_forward_duration_dup(df, forward_func_name.split('@')[0], func_ancestors, cg, rank, func_mapping_node_index)
            if len(fwd_df) == 0:
                print(f"forward_func_name no data: {forward_func_name}")
                continue
            fwd_dur = calculate_statistics(fwd_df, forward_func_name)
            func_mapping_node_index[forward_func_name] = fwd_df.index
            bwd_df = get_backward_duration(df, cg, rank, fwd_df.index)
            bwd_dur = calculate_statistics(bwd_df, forward_func_name+'-bwd')
            stat_info_funcs_grouped = pd.concat([stat_info_funcs_grouped, fwd_dur, bwd_dur], axis=0)

        #with open(f"./full_tflops_mapping_index_{rank}.pkl", 'wb') as f:
        #    pickle.dump(tflop_bw_mapping_index, f)
        #cache_path ='./full_tflops_analyzer_cache.pkl'
        #with open(cache_path, 'wb') as f:
        #    pickle.dump(df, f)
        #df.to_csv(f"./full_df_{rank}_tflops.csv", index=False)
        (root_func_name, _) = func_name[0]
        fwd_total_dur_mean = stat_info_funcs_grouped[stat_info_funcs_grouped.index == root_func_name]['mean'].values[0]
        bwd_total_dur_mean = stat_info_funcs_grouped[stat_info_funcs_grouped.index == root_func_name+'-bwd']['mean'].values[0]
        stat_info_funcs_grouped['mean_percent'] = 0.0
        stat_info_funcs_grouped['mean_percent'] = stat_info_funcs_grouped.apply(lambda row: cal_dur_percent(row, fwd_total_dur_mean, bwd_total_dur_mean), axis=1)

        with open(f"./fp8-0125-fa-kimi-{rank}-main-stack.txt", "w") as f:
            for forward_func_name, func_ancestors in func_name:
                print(f'{"    " * len(func_ancestors)}{forward_func_name}')
                f.write(f'{"    " * len(func_ancestors)}{forward_func_name}\n')
                if forward_func_name in stat_info_funcs_grouped.index:
                    backward_func_name = forward_func_name + '-bwd'
                    f.write(f'{"    " * (len(func_ancestors)+1)} fwd: mean_percent: {stat_info_funcs_grouped.loc[forward_func_name,"mean_percent"]:.2f}, mean: {stat_info_funcs_grouped.loc[forward_func_name,"mean"]:.2f}, q_25: {stat_info_funcs_grouped.loc[forward_func_name,"q_25"]:.2f}, q_50: {stat_info_funcs_grouped.loc[forward_func_name,"q_50"]:.2f}, q_75: {stat_info_funcs_grouped.loc[forward_func_name,"q_75"]:.2f}, max: {stat_info_funcs_grouped.loc[forward_func_name,"max"]:.2f}, min: {stat_info_funcs_grouped.loc[forward_func_name,"min"]:.2f}, count: {stat_info_funcs_grouped.loc[forward_func_name,"count"]:.2f}\n')
                    f.write(f'{"    " * (len(func_ancestors)+1)} bwd: mean_percent: {stat_info_funcs_grouped.loc[backward_func_name,"mean_percent"]:.2f}, mean: {stat_info_funcs_grouped.loc[backward_func_name,"mean"]:.2f}, q_25: {stat_info_funcs_grouped.loc[backward_func_name,"q_25"]:.2f}, q_50: {stat_info_funcs_grouped.loc[backward_func_name,"q_50"]:.2f}, q_75: {stat_info_funcs_grouped.loc[backward_func_name,"q_75"]:.2f}, max: {stat_info_funcs_grouped.loc[backward_func_name,"max"]:.2f}, min: {stat_info_funcs_grouped.loc[backward_func_name,"min"]:.2f}, count:  {stat_info_funcs_grouped.loc[backward_func_name,"count"]:.2f}\n')