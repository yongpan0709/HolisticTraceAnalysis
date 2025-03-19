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
from call_graph_template import extract_func_name_from_template, output_template

def set_pandas_display_options():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.2f}".format)


def get_backward_duration(df, forward_sym_id): 
    forward_index  = df[df['name'] == forward_sym_id].index
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
                if child_type != 'kernel':
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
    for idx in ancestors_index:
        if (df.loc[idx, 's_cat'] != 'musa_runtime'):
            parents.append(idx)

    while len(parents)>0:
        ancestor_as_parent_index = parents.popleft()
        child_df = df[df['parent'] == ancestor_as_parent_index]
        print(child_df[['parent', 's_cat', 's_name']])
        if (child_df['s_name'] == child_func_name).any():
            print("matched: ", child_df[child_df['s_name'] == child_func_name].index)
            return child_df[child_df['s_name'] == child_func_name].index
        else:
            for child_index in child_df.index:
                if (df.loc[child_index, 's_cat'] != 'musa_runtime'):
                    parents.append(child_index)
    return []

def get_forward_duration(df, forward_func_name, func_ancestors):
    func_to_filters = [*func_ancestors, forward_func_name]
    node_index: List[np.int64] = []
    if len(func_to_filters) > 1:
        ancestor_s_name = func_to_filters[0].split('@')[0]
        node_index = df[df['s_name'] == ancestor_s_name].index
        for i in range(1, len(func_to_filters)):
            # print(f'i {i}, func_to_filters[i].split([0]: {func_to_filters[i].split("@")[0]}')
            # print(f'child name: {func_to_filters[i].split("@")[0]}, ancestors_index: {ancestors_index}')
            node_index = find_child_index_in_ancestor(df, func_to_filters[i].split('@')[0], node_index)
            # print(f'forward_func_name -- ancestors_index:{node_index}')
            # return node_index
    else:
        node_index = df[df['s_name'] == func_to_filters[0].split('@')[0]].index
        # print(f'in else ancestors_index:{node_index}')
        # return node_index    
    # print('in get foward duration', df[df['index'].isin(node_index)])
    return df[df['index'].isin(node_index)]
    # forward_info: Dict[np.int64, np.float64] = defaultdict(np.float64)
    # fwd_nodes = df[df['index'].isin(node_index)]
    # forward_info['kernel_dur_sum'] = fwd_nodes['kernel_span']/1000.0
    # backward_stat_info = pd.DataFrame.from_dict(backward_info, orient='index', columns=['kernel_dur_sum'])
    # return backward_stat_info


# # forward step calculations
# expect_func_names = ["pretrain_deepseekv2.py(139): get_batch",
#                     "nn.Module: LanguageModelEmbedding_0",
#                     "nn.Module: TransformerLayer_0",
#                     "nn.Module: TransformerLayer_1",
#                     "nn.Module: RMSNorm_1",
#                     "nn.Module: MLASelfAttention_1",
#                     "megatron/core/fusions/fused_bias_dropout.py(42): _bias_dropout_add",
#                     "nn.Module: RMSNorm_2",
#                     "nn.Module: MoELayer_0",
#                     "nn.Module: TopKRouter_0",
#                     "megatron/core/transformer/moe/token_dispatcher.py(473): token_permutation",
#                     "nn.Module: TEGroupedMLP_0",
#                     "megatron/core/transformer/moe/token_dispatcher.py(565): token_unpermutation",
#                     "nn.Module: SharedExpertMLP_0",
#                     "megatron/core/fusions/fused_bias_dropout.py(42): _bias_dropout_add",
#                     "nn.Module: ColumnParallelLinear_0",
#                     "megatron/core/models/common/language_module/language_module.py(66): compute_language_model_loss",
#                     "megatron/core/distributed/finalize_model_grads.py(250): finalize_model_grads",
#                     "megatron/core/optimizer/optimizer.py(1040): step",
#                     "megatron/core/transformer/moe/token_dispatcher.py(341): preprocess",
#                     "megatron/core/transformer/moe/moe_utils.py(221): permute",
#                     "megatron/core/tensor_parallel/mappings.py(524): all_to_all",
#                     "megatron/core/transformer/moe/moe_utils.py(353): sort_chunks_by_idxs",
#                     "megatron/core/transformer/moe/moe_utils.py(282): unpermute"
#                     ]
# expect_funcs_info = cg.trace_data.traces[0][cg.trace_data.traces[0]['s_name'].isin(expect_func_names)]
# # expect_funcs_info['expect_func_dur'] = (expect_funcs_info['last_kernel_end'] - expect_funcs_info['first_kernel_start'])/1000
# expect_funcs_info['expect_func_dur'] = expect_funcs_info['kernel_span']/1000
# funcs_grouped = expect_funcs_info.groupby(['s_name'])
# # expect_funcs_grouped = expect_funcs_info.groupby(['s_name'])
# stat_info_funcs_grouped = pd.DataFrame()
# stat_info_funcs_grouped['mean'] = funcs_grouped['expect_func_dur'].mean()
# stat_info_funcs_grouped['q_25'] = funcs_grouped['expect_func_dur'].quantile(.25)
# stat_info_funcs_grouped['q_50'] = funcs_grouped['expect_func_dur'].quantile(.5)
# stat_info_funcs_grouped['q_75'] = funcs_grouped['expect_func_dur'].quantile(.75)
# stat_info_funcs_grouped['max'] = expect_funcs_info.groupby(['s_name'])['expect_func_dur'].max()
# stat_info_funcs_grouped['min'] = expect_funcs_info.groupby(['s_name'])['expect_func_dur'].min()
# stat_info_funcs_grouped['var'] = expect_funcs_info.groupby(['s_name'])['expect_func_dur'].var()
# stat_info_funcs_grouped['count'] = expect_funcs_info.groupby(['s_name'])['expect_func_dur'].count()
# print(stat_info_funcs_grouped)

# forward_step_name = "megatron/core/pipeline_parallel/schedules.py(173): forward_step"
# nn.Module: GPTModel_0  backward
# forward_step_sym_id = t.symbol_table.get_sym_id_map().get(forward_step_name)
# bwd_ids = get_backward_duration(cg.trace_data.traces[0], forward_step_sym_id)
# print(f'bwd_ids: {bwd_ids}')

# for forward_step_name in expect_func_names:
#     forward_step_sym_id = t.symbol_table.get_sym_id_map().get(forward_step_name)
#     bwd_ids = get_backward_duration(cg.trace_data.traces[0], forward_step_sym_id)
#     bwd_df = pd.DataFrame({'mean': bwd_ids['kernel_dur_sum'].mean(),
#                         'q_25': bwd_ids['kernel_dur_sum'].quantile(.25),
#                         'q_50': bwd_ids['kernel_dur_sum'].quantile(.5),
#                         'q_75': bwd_ids['kernel_dur_sum'].quantile(.75),
#                         'max': bwd_ids['kernel_dur_sum'].max(),
#                         'min': bwd_ids['kernel_dur_sum'].min(),
#                         'var': bwd_ids['kernel_dur_sum'].var(),
#                         'count': 0}, index=[forward_step_name + '-bwd'], columns=['mean', 'q_25', 'q_50', 'q_75', 'max', 'min', 'var', 'count'])
# #     print(f'bwd_ids: {bwd_ids}')
# #     stat_info_funcs_grouped = pd.concat([stat_info_funcs_grouped, bwd_df], axis=0)

# print(stat_info_funcs_grouped)
# exit(0)

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
    trace_dir = str(Path(base_dir).joinpath("ds-count-32"))
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
    # ('megatron/core/tensor_parallel/mappings.py(524): all_to_all@0', ['megatron/core/pipeline_parallel/schedules.py(173): forward_step@0', 'nn.Module: TransformerLayer_1@0', 'nn.Module: MoELayer_0@0', 'megatron/core/transformer/moe/token_dispatcher.py(473): token_permutation@0']), 
    forward_func_name, func_ancestors = ('megatron/core/tensor_parallel/mappings.py(528): all_to_all@0', ['megatron/core/pipeline_parallel/schedules.py(173): forward_step@0', 'nn.Module: TransformerLayer_1@0', 'nn.Module: MoELayer_0@0', 'megatron/core/transformer/moe/token_dispatcher.py(473): token_permutation@0'])
    # forward_func_name, func_ancestors = ('nn.Module: TransformerLayer_1@0', ["megatron/core/transformer/transformer_block.py(314): custom_forward@0"])
    df = cg.trace_data.traces[0]
    func_name = extract_func_name_from_template(output_template)
    for forward_func_name, func_ancestors in func_name:
        func_to_filters = [func_ancestors[:], forward_func_name]
        fwd_df = get_forward_duration(df, forward_func_name, func_ancestors)
        print(f'forward_func_name: {forward_func_name}, df: {fwd_df}')
    # forward_func_name, func_ancestors = ('megatron/core/pipeline_parallel/schedules.py(173): forward_step@0', [])
    # for forward_step_name in expect_func_names:
    # bwd_ids = get_backward_duration(cg.trace_data.traces[0], forward_step_sym_id)
    # bwd_df = pd.DataFrame({'mean': bwd_ids['kernel_dur_sum'].mean(),
    #                     'q_25': bwd_ids['kernel_dur_sum'].quantile(.25),
    #                     'q_50': bwd_ids['kernel_dur_sum'].quantile(.5),
    #                     'q_75': bwd_ids['kernel_dur_sum'].quantile(.75),
    #                     'max': bwd_ids['kernel_dur_sum'].max(),
    #                     'min': bwd_ids['kernel_dur_sum'].min(),
    #                     'var': bwd_ids['kernel_dur_sum'].var(),
    #                     'count': 0}, index=[forward_step_name + '-bwd'], columns=['mean', 'q_25', 'q_50', 'q_75', 'max', 'min', 'var', 'count'])
#     print(f'bwd_ids: {bwd_ids}')
#     stat_info_funcs_grouped = pd.concat([stat_info_funcs_grouped, bwd_df], axis=0)
