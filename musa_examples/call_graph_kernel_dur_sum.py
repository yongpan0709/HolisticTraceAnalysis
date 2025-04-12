from pathlib import Path 
from hta.common.trace import Trace
from hta.configs.config import logger
from hta.configs.parser_config import  ParserConfig, AVAILABLE_ARGS
from hta.common.trace_call_graph import CallGraph
from hta.common.call_stack import CallStackIdentity
from collections import defaultdict
from typing import Dict, List, Set
import numpy as np
import pandas as pd
from collections import deque
from call_graph_template import extract_func_name_from_template, extract_dup_or_shape_func_name_from_template, output_template_to_file

def set_pandas_display_options():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.2f}".format)


def get_backward_duration(df, forward_index): 
    # forward_index  = df[df['name'] == forward_sym_id].index
    forward_children_with_bwd_id: Dict[np.int64, List[np.int64]] = defaultdict(list)
    """
    Todo: refact to void checking children and grand-children funcs
    """
    for forward_as_parent in forward_index:
        parents = deque()
        parents.append(forward_as_parent)
        while len(parents) > 0:
            parent = parents.popleft()
            # print(f"parent: {parent}")
            for child in df[df['parent'] == parent].index:
                child_fwdbwd_index = df.loc[child, 'fwdbwd_index']
                if child_fwdbwd_index > 0:
                    child_fwdbwd_num_kernels = df.loc[child_fwdbwd_index, 'num_kernels']
                    if child_fwdbwd_num_kernels > 0:
                        forward_children_with_bwd_id[forward_as_parent].append(child_fwdbwd_index)
                child_type = df.loc[child, 's_cat']
                if child_type not in ['kernel']:
                    parents.append(child)
    
    backward_info: Dict[np.int64, np.float64] = defaultdict(np.float64)
    for forward_as_parent, bwd_children_indices in forward_children_with_bwd_id.items():
        bwd_children = df[df['index'].isin(bwd_children_indices)]
        first_kernel_start = bwd_children['first_kernel_start'].min()
        last_kernel_end = bwd_children['last_kernel_end'].max()
        backward_info[forward_as_parent] = (last_kernel_end - first_kernel_start)/1000.0
    backward_stat_info = pd.DataFrame.from_dict(backward_info, orient='index', columns=['kernel_dur_sum'])
    return backward_stat_info

def get_forward_duration_uniq(df, forward_func_name):
    node_index = df[df['s_name'].str.match(pat=r"^"+forward_func_name+r"$")].index
    return df[df['index'].isin(node_index)]

def get_forward_duration_dup(df, forward_func_name, func_ancestors, cg, func_mapping_node_index):
    node_index: List[np.int64] = []
    nearest_ancestor_index = func_mapping_node_index[func_ancestors[-1]]
    for index, row in df[df['s_name'].str.match(pat=forward_func_name)].iterrows():
        pid, tid = row['pid'], row['tid']
        path_to_root = cg.rank_to_stacks[0][CallStackIdentity(0, pid, tid)].get_path_to_root(index)
        if len(set(path_to_root).intersection(set(nearest_ancestor_index))) > 0:
            node_index.append(index)
    # print("dup node index: ", node_index)
    return df[df['index'].isin(node_index)]

def calculate_statistics(df: pd.DataFrame, func_name: str, calculate_col_name: str, fwd_bwd: str = 'fwd'):
    if fwd_bwd == 'fwd':
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

# def get_forward_duration(df, forward_func_name, func_ancestors, dup_func_name, cg, func_mapping_node_index):
#     for 
#def print_call_stack_statistic_info(df, call_stack_template):

# print(f"All kernels duration sum: {MLASelfAttention['kernel_dur_sum'].values}")
# print(f"The total number of kernels executed: {MLASelfAttention['num_kernels'].values}")
# print(f"The start time of first kernel executed: {MLASelfAttention['first_kernel_start'].values}")
# print(f"The end time of last kernel executed: {MLASelfAttention['last_kernel_end'].values}")

# expect_kernel_name = "musa_asm_bf16bf16bf16bf16gemm_nt_tce_768_256x384B128_squad_level_epilogue"
# print(f"\n\nGet the shape info of kernel name[{expect_kernel_name}]")
# df = cg.trace_data.traces[0]

# index_kernel_in_df = df[df['s_name'] == expect_kernel_name].index
# first_index = index_kernel_in_df.values[0]
# # print(first_index)
# check_shape_of_index  = first_index
# while True:
#     shape = df.loc[check_shape_of_index,'input_dims']
#     if shape == '-1':
#         check_shape_of_index = df.loc[check_shape_of_index,'parent']
#         print(f'parent index: {check_shape_of_index}')
#         # print()
#     else:
#         print(f'The shape info: {shape}')
#         break

if __name__ == "__main__":
    base_dir = "../"
    trace_dir = str(Path(base_dir).joinpath("ds-0405-0301"))
    cfg = ParserConfig.get_default_cfg()
    # config for extracting shape info
    cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
    ParserConfig.set_default_cfg(cfg)
    t = Trace(trace_dir=trace_dir)
    t.parse_traces()
    # transform name and cat columns to s_name and s_cat
    # name and cat are kernel id
    t.decode_symbol_ids(use_shorten_name=False)
    set_pandas_display_options()
    cg = CallGraph(t, ranks=[0])
    df = cg.trace_data.traces[0]
    dup_func_name, need_shape_func_name = extract_dup_or_shape_func_name_from_template(output_template_to_file)
    func_name = extract_func_name_from_template(output_template_to_file)
    stat_info_funcs_grouped = pd.DataFrame()
    func_mapping_node_index: Dict[str, List[np.float64]] = defaultdict(list)
    for forward_func_name, func_ancestors in func_name:
        if forward_func_name.split('@')[0] in dup_func_name:
            continue
        fwd_df = get_forward_duration_uniq(df, forward_func_name.split('@')[0])
        fwd_dur = calculate_statistics(fwd_df, forward_func_name, 'kernel_span')
        func_mapping_node_index[forward_func_name] = fwd_df.index
        bwd_df = get_backward_duration(df, fwd_df.index)
        bwd_dur = calculate_statistics(bwd_df, forward_func_name+'-bwd', 'kernel_dur_sum', 'bwd')
        stat_info_funcs_grouped = pd.concat([stat_info_funcs_grouped, fwd_dur, bwd_dur], axis=0)
    for forward_func_name, func_ancestors in func_name:
        # fwd_df = get_forward_duration(df, func_to_filters[calculated_idx:], func_mapping_node_index[func_to_filters[calculated_idx]])
        if forward_func_name.split('@')[0] not in dup_func_name:
            continue
        # print(f'forward_func_name: {forward_func_name}, func_ancestors: {func_ancestors}')
        fwd_df = get_forward_duration_dup(df, forward_func_name.split('@')[0], func_ancestors, cg, func_mapping_node_index)
        # print(f"forward_func_name: {forward_func_name}, df:\n {fwd_df[['s_name', 'kernel_span', 'first_kernel_start', 'last_kernel_end']]}")
        # print(f'fwd_df: {fwd_df}')
        fwd_dur = calculate_statistics(fwd_df, forward_func_name, 'kernel_span')
        # print(fwd_dur)
        func_mapping_node_index[forward_func_name] = fwd_df.index
        bwd_df = get_backward_duration(df, fwd_df.index)
        bwd_dur = calculate_statistics(bwd_df, forward_func_name+'-bwd', 'kernel_dur_sum', 'bwd')
        # print(bwd_dur)
        stat_info_funcs_grouped = pd.concat([stat_info_funcs_grouped, fwd_dur, bwd_dur], axis=0)

    (root_func_name, _) = func_name[0]
    fwd_total_dur_mean = stat_info_funcs_grouped[stat_info_funcs_grouped.index == root_func_name]['mean'].values[0]
    bwd_total_dur_mean = stat_info_funcs_grouped[stat_info_funcs_grouped.index == root_func_name+'-bwd']['mean'].values[0]
    stat_info_funcs_grouped['mean_percent'] = 0.0

    def cal_dur_percent(row, fwd_total_dur_mean, bwd_total_dur_mean):
        if row.name.endswith('-bwd'):
            return row['mean']/bwd_total_dur_mean
        else:
            return row['mean']/fwd_total_dur_mean
    stat_info_funcs_grouped['mean_percent'] = stat_info_funcs_grouped.apply(lambda row: cal_dur_percent(row, fwd_total_dur_mean, bwd_total_dur_mean), axis=1)

    stat_info_funcs_grouped.to_csv('profile-0320-update.csv')
    # print(f'stat_info_funcs_grouped: \n{stat_info_funcs_grouped}')
    for forward_func_name, func_ancestors in func_name:
        print(f'{"    " * len(func_ancestors)}{forward_func_name}')
        forward_func_name
        print(f'{"    " * (len(func_ancestors)+1)} fwd: mean_percent: {stat_info_funcs_grouped.loc[forward_func_name,"mean_percent"]:.2f}, mean: {stat_info_funcs_grouped.loc[forward_func_name,"mean"]:.2f}, q_25: {stat_info_funcs_grouped.loc[forward_func_name,"q_25"]:.2f}, q_50: {stat_info_funcs_grouped.loc[forward_func_name,"q_50"]:.2f}, q_75: {stat_info_funcs_grouped.loc[forward_func_name,"q_75"]:.2f}, max: {stat_info_funcs_grouped.loc[forward_func_name,"max"]:.2f}, min: {stat_info_funcs_grouped.loc[forward_func_name,"min"]:.2f}')
        backward_func_name = forward_func_name + '-bwd'
        print(f'{"    " * (len(func_ancestors)+1)} bwd: mean_percent: {stat_info_funcs_grouped.loc[backward_func_name,"mean_percent"]:.2f}, mean: {stat_info_funcs_grouped.loc[backward_func_name,"mean"]:.2f}, q_25: {stat_info_funcs_grouped.loc[backward_func_name,"q_25"]:.2f}, q_50: {stat_info_funcs_grouped.loc[backward_func_name,"q_50"]:.2f}, q_75: {stat_info_funcs_grouped.loc[backward_func_name,"q_75"]:.2f}, max: {stat_info_funcs_grouped.loc[backward_func_name,"max"]:.2f}, min: {stat_info_funcs_grouped.loc[backward_func_name,"min"]:.2f}')
