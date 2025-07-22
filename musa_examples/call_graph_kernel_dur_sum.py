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
from call_graph_template import extract_func_name_from_template, extract_dup_or_shape_func_name_from_template, output_template_to_file, SHAPE_POSITION

def set_pandas_display_options():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.2f}".format)


def get_backward_duration(df, cg, rank, forward_index): 
    # forward_index  = df[df['name'] == forward_sym_id].index
    forward_children_with_bwd_id: Dict[np.int64, List[np.int64]] = defaultdict(list)
    """
    Todo: refact to void checking children and grand-children funcs
    """
    for forward_as_parent in forward_index:
        parents = deque()
        parents.append(forward_as_parent)
        while len(parents) > 0:
            cur_node = parents.popleft()
            cur_node_fwdbwd_index = df.loc[cur_node, 'fwdbwd_index']
            if cur_node_fwdbwd_index > 0:
                cur_node_fwdbwd_num_kernels = df.loc[cur_node_fwdbwd_index, 'num_kernels']
                if cur_node_fwdbwd_num_kernels > 0:
                    forward_children_with_bwd_id[forward_as_parent].append(cur_node_fwdbwd_index)
            else:
                cur_callStackNode = cg.rank_to_nodes[rank].get(cur_node)
                cur_node_children = cur_callStackNode.children
                if len(cur_node_children) > 0:
                    for child in cur_node_children:
                        child_type = df.loc[child, 's_cat']
                        if child_type not in ['kernel']:
                            parents.append(child)
    
    backward_info: Dict[np.int64, np.float64] = defaultdict(np.float64)
    for forward_as_parent, bwd_children_indices in forward_children_with_bwd_id.items():
        bwd_children = df[df['index'].isin(bwd_children_indices)]
        first_kernel_start = bwd_children['first_kernel_start'].min()
        last_kernel_end = bwd_children['last_kernel_end'].max()
        backward_info[forward_as_parent] = (last_kernel_end - first_kernel_start)
    backward_stat_info = pd.DataFrame.from_dict(backward_info, orient='index', columns=['kernel_span'])
    return backward_stat_info

def get_forward_duration_uniq(df, forward_func_name):
    node_index = df[df['s_name'].str.match(pat=r"^"+forward_func_name+r"$")].index
    return df[df['index'].isin(node_index)]

def get_forward_duration_dup(df, forward_func_name, func_ancestors, cg, rank, func_mapping_node_index):
    node_index: List[np.int64] = []
    nearest_ancestor_index = func_mapping_node_index[func_ancestors[-1]]
    for index, row in df[df['s_name'].str.match(pat=r"^"+forward_func_name+r"$")].iterrows():
        pid, tid = row['pid'], row['tid']
        path_to_root = cg.rank_to_stacks[rank][CallStackIdentity(rank, pid, tid)].get_path_to_root(index)
        if len(set(path_to_root).intersection(set(nearest_ancestor_index))) > 0:
            node_index.append(index)
    # print("dup node index: ", node_index)
    return df[df['index'].isin(node_index)]

def calculate_statistics(df: pd.DataFrame, func_name: str, calculate_col_name: str = 'kernel_span'): #, fwd_bwd: str = 'fwd'):
    df[calculate_col_name] = df[calculate_col_name]/1000.0
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

def extract_shape(func_name, df, func_mapping_node_index, need_shape_func_name):
    shape_info: Dict[str, Dict[str, pd.Series]] = defaultdict(lambda: defaultdict(pd.Series))
    for forward_func_name, _ in func_name:
        if forward_func_name.split('@')[0] in need_shape_func_name:
            func_index = func_mapping_node_index[forward_func_name]
            shape_array = []
            for index, row in df[df.index.isin(func_index)].iterrows(): 
                shape_array.append(row['input_dims'][0][0])
            shape_info[forward_func_name]['data'] = pd.Series(shape_array)
            shape_info[forward_func_name]['example'] = df.loc[func_index[0], 'input_dims']
    return shape_info

def cal_dur_percent(row, fwd_total_dur_mean, bwd_total_dur_mean):
    if row.name.endswith('-bwd'):
        return row['mean']/bwd_total_dur_mean
    else:
        return row['mean']/fwd_total_dur_mean
    
if __name__ == "__main__":
    import time
    base_dir = "../"
    trace_dir = str(Path(base_dir).joinpath("ds-0627"))
    cfg = ParserConfig.get_default_cfg()
    # config for extracting shape info
    cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
    ParserConfig.set_default_cfg(cfg)
    trace_files = get_trace_files(trace_dir)
    for rank, trace_file in trace_files.items():
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
        df = cg.call_stacks[rank].full_df
        dup_func_name, need_shape_func_name = extract_dup_or_shape_func_name_from_template(output_template_to_file)
        func_name = extract_func_name_from_template(output_template_to_file)
        stat_info_funcs_grouped = pd.DataFrame()
        func_mapping_node_index: Dict[str, List[np.float64]] = defaultdict(list)

        for forward_func_name, func_ancestors in func_name:
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

        shape_info = extract_shape(func_name, df, func_mapping_node_index, need_shape_func_name)
        (root_func_name, _) = func_name[0]
        fwd_total_dur_mean = stat_info_funcs_grouped[stat_info_funcs_grouped.index == root_func_name]['mean'].values[0]
        bwd_total_dur_mean = stat_info_funcs_grouped[stat_info_funcs_grouped.index == root_func_name+'-bwd']['mean'].values[0]
        stat_info_funcs_grouped['mean_percent'] = 0.0
        stat_info_funcs_grouped['mean_percent'] = stat_info_funcs_grouped.apply(lambda row: cal_dur_percent(row, fwd_total_dur_mean, bwd_total_dur_mean), axis=1)

        shape_info = extract_shape(func_name, df, func_mapping_node_index, need_shape_func_name)
        with open(f"./call_graph_duration_{rank}.txt", "w") as f:
            for forward_func_name, func_ancestors in func_name:
                if forward_func_name.split('@')[0] in need_shape_func_name:
                    assert len(shape_info[forward_func_name]['example']) >= SHAPE_POSITION[forward_func_name.split('@')[0]], "No shape info of func name:" + forward_func_name 
                    shape_dim = shape_info[forward_func_name]['example'][:SHAPE_POSITION[forward_func_name.split('@')[0]]]
                    shape_dim[0][0] = int(shape_info[forward_func_name]['data'].mean())
                    if SHAPE_POSITION[forward_func_name.split('@')[0]] == 2:
                        dims = set({shape_dim[0][0], shape_dim[0][1], shape_dim[1][0], shape_dim[1][1]})
                        m,n,k = dims
                        mfu = 2*m*n*k/stat_info_funcs_grouped.loc[forward_func_name,"mean"]*1000/(1e12)/458
                        print(f'{"    " * len(func_ancestors)}{forward_func_name}   mean: {shape_dim}, mfu: {mfu:.3f}')
                        f.write(f'{"    " * len(func_ancestors)}{forward_func_name}   mean: {shape_dim}, mfu: {mfu:.3f}\n')
                    else:
                        m,n = shape_dim[0]
                        bw_usage = 2*m*n/stat_info_funcs_grouped.loc[forward_func_name,"mean"]*1000/(1024**3)
                        print(f'{"    " * len(func_ancestors)}{forward_func_name}   mean: {shape_dim}, bw_usage: {bw_usage:.3f}')
                        f.write(f'{"    " * len(func_ancestors)}{forward_func_name}   mean: {shape_dim}, bw_usage: {bw_usage:.3f}\n')
                else:
                    print(f'{"    " * len(func_ancestors)}{forward_func_name}')
                    f.write(f'{"    " * len(func_ancestors)}{forward_func_name}\n')

                if forward_func_name in stat_info_funcs_grouped.index:
                    print(f'{"    " * (len(func_ancestors)+1)} fwd: mean_percent: {stat_info_funcs_grouped.loc[forward_func_name,"mean_percent"]:.2f}, mean: {stat_info_funcs_grouped.loc[forward_func_name,"mean"]:.2f}, q_25: {stat_info_funcs_grouped.loc[forward_func_name,"q_25"]:.2f}, q_50: {stat_info_funcs_grouped.loc[forward_func_name,"q_50"]:.2f}, q_75: {stat_info_funcs_grouped.loc[forward_func_name,"q_75"]:.2f}, max: {stat_info_funcs_grouped.loc[forward_func_name,"max"]:.2f}, min: {stat_info_funcs_grouped.loc[forward_func_name,"min"]:.2f}')
                    backward_func_name = forward_func_name + '-bwd'
                    print(f'{"    " * (len(func_ancestors)+1)} bwd: mean_percent: {stat_info_funcs_grouped.loc[backward_func_name,"mean_percent"]:.2f}, mean: {stat_info_funcs_grouped.loc[backward_func_name,"mean"]:.2f}, q_25: {stat_info_funcs_grouped.loc[backward_func_name,"q_25"]:.2f}, q_50: {stat_info_funcs_grouped.loc[backward_func_name,"q_50"]:.2f}, q_75: {stat_info_funcs_grouped.loc[backward_func_name,"q_75"]:.2f}, max: {stat_info_funcs_grouped.loc[backward_func_name,"max"]:.2f}, min: {stat_info_funcs_grouped.loc[backward_func_name,"min"]:.2f}')
                    f.write(f'{"    " * (len(func_ancestors)+1)} fwd: mean_percent: {stat_info_funcs_grouped.loc[forward_func_name,"mean_percent"]:.2f}, mean: {stat_info_funcs_grouped.loc[forward_func_name,"mean"]:.2f}, q_25: {stat_info_funcs_grouped.loc[forward_func_name,"q_25"]:.2f}, q_50: {stat_info_funcs_grouped.loc[forward_func_name,"q_50"]:.2f}, q_75: {stat_info_funcs_grouped.loc[forward_func_name,"q_75"]:.2f}, max: {stat_info_funcs_grouped.loc[forward_func_name,"max"]:.2f}, min: {stat_info_funcs_grouped.loc[forward_func_name,"min"]:.2f}\n')
                    f.write(f'{"    " * (len(func_ancestors)+1)} bwd: mean_percent: {stat_info_funcs_grouped.loc[backward_func_name,"mean_percent"]:.2f}, mean: {stat_info_funcs_grouped.loc[backward_func_name,"mean"]:.2f}, q_25: {stat_info_funcs_grouped.loc[backward_func_name,"q_25"]:.2f}, q_50: {stat_info_funcs_grouped.loc[backward_func_name,"q_50"]:.2f}, q_75: {stat_info_funcs_grouped.loc[backward_func_name,"q_75"]:.2f}, max: {stat_info_funcs_grouped.loc[backward_func_name,"max"]:.2f}, min: {stat_info_funcs_grouped.loc[backward_func_name,"min"]:.2f}\n')
