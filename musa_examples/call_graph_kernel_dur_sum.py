from pathlib import Path 
from hta.common.trace import Trace
from hta.configs.config import logger
from hta.configs.parser_config import  ParserConfig, AVAILABLE_ARGS
from hta.common.trace_call_graph import CallGraph
from collections import defaultdict
from typing import Dict, List, Set
import numpy as np
import pandas as pd
from collections import deque
from call_graph_template import extract_func_name_from_template, output_template_1

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

def find_child_index_in_ancestor(df: pd.DataFrame, child_func_name: str, ancestors_index):
    parents = deque()
    target_index = []
    for idx in ancestors_index:
        if (df.loc[idx, 's_cat'] not in ['cpu_op', 'gpu_memcpy', 'gpu_memset', 'musa_driver', 'musa_runtime', 'user_annotation']):
            parents.append(idx)

    while len(parents)>0:
        ancestor_as_parent_index = parents.popleft()
        child_df = df[df['parent'] == ancestor_as_parent_index]
        # print(child_df[['parent', 's_cat', 's_name']])
        if (child_df['s_name'] == child_func_name).any():
            # print("matched: ", child_df[child_df['s_name'] == child_func_name].index)
            for child_indx in child_df[child_df['s_name'] == child_func_name].index:
                target_index.append(child_indx)
        else:
            for child_index in child_df.index:
                if (df.loc[child_index, 's_cat'] not in ['cpu_op', 'gpu_memcpy', 'gpu_memset', 'musa_driver', 'musa_runtime', 'user_annotation']):
                    parents.append(child_index)
    return target_index

def get_forward_duration(df, func_to_filters, ancestors_index):
    node_index: List[np.int64] = ancestors_index
    if len(func_to_filters) > 1:
        # ancestor_s_name = func_to_filters[0].split('@')[0]
        node_index = ancestors_index
        for i in range(1, len(func_to_filters)):
            # print(f'i {i}, func_to_filters[i].split([0]: {func_to_filters[i].split("@")[0]}')
            # print(f'child name: {func_to_filters[i].split("@")[0]}, ancestors_index: {ancestors_index}')
            node_index = find_child_index_in_ancestor(df, func_to_filters[i].split('@')[0], node_index)
            # print(f'forward_func_name -- node_index: \n{node_index}')
            # return node_index
    else:
        node_index = df[df['s_name'] == func_to_filters[0].split('@')[0]].index
        # print(f'in else ancestors_index:{node_index}')
        # return node_index    
    # print('in get foward duration', df[df['index'].isin(node_index)])
    return df[df['index'].isin(node_index)]

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
    trace_dir = str(Path(base_dir).joinpath("ds-0321"))
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
    func_name = extract_func_name_from_template(output_template_1)
    stat_info_funcs_grouped = pd.DataFrame()
    func_mapping_node_index: Dict[str, List[np.float64]] = defaultdict(list)
    for forward_func_name, func_ancestors in func_name:
        func_to_filters = [*func_ancestors, forward_func_name]
        calculated_idx = 0
        for i in range(len(func_to_filters)-1, -1, -1):
            if func_to_filters[i] not in func_mapping_node_index:
                continue
            else:
                calculated_idx = i
        
        # print(f'forward_func_name: {forward_func_name}, calculated_idx: {calculated_idx}')
        # print(f'func_ancestors: {func_to_filters[calculated_idx:]}, mapping reused: {func_mapping_node_index[func_to_filters[calculated_idx]]}')
        fwd_df = get_forward_duration(df, func_to_filters[calculated_idx:], func_mapping_node_index[func_to_filters[calculated_idx]])
        # print(f"forward_func_name: {forward_func_name}, df:\n {fwd_df[['s_name', 'kernel_span', 'first_kernel_start', 'last_kernel_end']]}")
        # print(f'fwd_df: {fwd_df}')
        fwd_df['kernel_span'] = fwd_df['kernel_span']/1000.0
        fwd_dur = pd.DataFrame({
                            'mean': fwd_df['kernel_span'].mean(),
                            'q_25': fwd_df['kernel_span'].quantile(.25),
                            'q_50': fwd_df['kernel_span'].quantile(.5),
                            'q_75': fwd_df['kernel_span'].quantile(.75),
                            'max': fwd_df['kernel_span'].max(),
                            'min': fwd_df['kernel_span'].min(),
                            'var': fwd_df['kernel_span'].var(),
                            'count': fwd_df['kernel_span'].count()}, index=[forward_func_name], columns=['mean', 'q_25', 'q_50', 'q_75', 'max', 'min', 'var', 'count'])
        # print(fwd_dur)
        func_mapping_node_index[forward_func_name] = fwd_df.index
        bwd_ids = get_backward_duration(df, fwd_df.index)
        bwd_dur = pd.DataFrame({
                        'mean': bwd_ids['kernel_dur_sum'].mean(),
                        'q_25': bwd_ids['kernel_dur_sum'].quantile(.25),
                        'q_50': bwd_ids['kernel_dur_sum'].quantile(.5),
                        'q_75': bwd_ids['kernel_dur_sum'].quantile(.75),
                        'max': bwd_ids['kernel_dur_sum'].max(),
                        'min': bwd_ids['kernel_dur_sum'].min(),
                        'var': bwd_ids['kernel_dur_sum'].var(),
                        'count': 0}, index=[forward_func_name+'-bwd'],columns=['mean', 'q_25', 'q_50', 'q_75', 'max', 'min', 'var', 'count'])
        stat_info_funcs_grouped = pd.concat([stat_info_funcs_grouped, fwd_dur, bwd_dur], axis=0)
    
    (root_func_name, _) = func_name[0]
    # print(f'root_func_name: {root_func_name}')
    # print(stat_info_funcs_grouped[stat_info_funcs_grouped.index == root_func_name]['mean'])
    fwd_total_dur_mean = stat_info_funcs_grouped[stat_info_funcs_grouped.index == root_func_name]['mean'].values[0]
    bwd_total_dur_mean = stat_info_funcs_grouped[stat_info_funcs_grouped.index == root_func_name+'-bwd']['mean'].values[0]
    stat_info_funcs_grouped['mean_percent'] = 0.0

    def cal_dur_percent(row, fwd_total_dur_mean, bwd_total_dur_mean):
        if row.name.endswith('-bwd'):
            return row['mean']/bwd_total_dur_mean
        else:
            return row['mean']/fwd_total_dur_mean
    stat_info_funcs_grouped['mean_percent'] = stat_info_funcs_grouped.apply(lambda row: cal_dur_percent(row, fwd_total_dur_mean, bwd_total_dur_mean), axis=1)

    # print_call_stack_statistic_info(stat_info_funcs_grouped, output_template_to_file)
    stat_info_funcs_grouped.to_csv('profile-0320-update.csv')
    print(f'stat_info_funcs_grouped: \n{stat_info_funcs_grouped}')
    # for forward_step_name in expect_func_names:
    # bwd_ids = get_backward_duration(cg.trace_data.traces[0], forward_step_sym_id)
   
#     print(f'bwd_ids: {bwd_ids}')
#     stat_info_funcs_grouped = pd.concat([stat_info_funcs_grouped, bwd_df], axis=0)
