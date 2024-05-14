from hta.trace_analysis import TraceAnalysis
from hta.utils.parrallel_state import get_3d_parallel_groups, is_first_stage, is_last_stage, get_next_pipeline_rank
from hta.utils.utils import partition_files_across_directories, LogToFile, prepare_directory
from hta.common.trace_df import keep_target_category, keep_target_category, keep_target_category, keep_rows_starting_with_names, calculate_flops_for_trace_df, calculate_comm_volume_for_trace_df, build_call_tree, save_trace_df_to_file, mark_send_recv_direction
from hta.common.call_stack import CallGraph 
from hta.configs.config import logger

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

trace_dir = '/home/dist/yiyuan/trace_dir_7B'
trace_dir_for_pp_group = os.path.join(trace_dir, 'pp_group')

TP_SIZE = 2
PP_SIZE = 2
DP_SIZE = 4
NUM_MICROBATCHES = 16

def load_trace_analyer(trace_dir):
    cache_path = os.path.join(trace_dir, 'analyzer_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            analyzer = pickle.load(f)
        print(f'analyzer has been loaded from {cache_path}')
    else:
        analyzer = TraceAnalysis(trace_dir=trace_dir)
        with open(cache_path, 'wb') as f:
            pickle.dump(analyzer, f)
        print(f'analyzer has been saved to {cache_path}')
    return analyzer

def display_traces_info(traces):
    first_trace_df = next(iter(traces.values()))
    print(f'total {len(traces)} traces, and each trace has {len(first_trace_df)} items')
    print(first_trace_df['s_cat'].value_counts())

def convert_to_int(trace_df):
    int_fields = ["pid", "tid", "ts", "dur", 'end', "index", "name", "cat", 'external_id', 'iteration']
    for field in int_fields:
        if field in trace_df.columns:
            trace_df[field] = trace_df[field].astype(int)
    return trace_df

def filter_single_trace_df(trace_df):
    trace_df['end'] = trace_df['ts'] + trace_df['dur']

    trace_df = convert_to_int(trace_df)
    trace_df_user_annotation = keep_target_category(trace_df, 'user_annotation')
    trace_df_python_function = keep_target_category(trace_df, 'python_function')
    trace_df_cpu_op = keep_target_category(trace_df, 'cpu_op')

    trace_df_user_annotation = keep_rows_starting_with_names(trace_df_user_annotation, ['forward_step', 'backward_step', 'forward_backward_pipelining_without_interleaving', 'get_batch', 'warmup_state', 'steady_state', 'cooldown_state', 'mccl:', 'ProfilerStep', 'recv_forward', 'recv_backward', 'send_forward', 'send_backward', 'send_forward_recv_backward', 'send_backward_recv_forward'])
    trace_df_python_function = keep_rows_starting_with_names(trace_df_python_function, ['Embedding', 'RotaryEmbedding', 'apply_rotary_pos_emb', 'ParallelAttention', 'ParallelMLP', 'get_batch', 'RmsNormBackward', 'LinearWithGradAccumulationAndAsyncCommunicationBackward', 'ColumnParallelLinear', 'RowParallelLinear', 'post_language_model_processing', 'parallel_lm_logits', 'vocab_parallel_cross_entropy', 'average_losses_across_data_parallel_group', 'ParallelTransformerLayer', 'gather_model_params', 'step'])
    trace_df_cpu_op = keep_rows_starting_with_names(trace_df_cpu_op, ['aten::matmul', 'aten::rms_norm_forward', 'aten::scaled_dot_product_attention', 'aten::rms_norm_backward', 'aten::_scaled_dot_product_attention_flash_musa_backward', 'aten::embedding_backward'])

    trace_df = pd.concat([trace_df_user_annotation, trace_df_python_function, trace_df_cpu_op])
    
    # trace_df['stream'] = 0
        
    return trace_df 

def set_micro_batch_id(df):
    df.sort_values(by=['ts', 'dur'], ascending=[True, False], inplace=True)

    # 初始化micro_batch_id列
    df['micro_batch_id_forward'] = -1
    df['micro_batch_id_backward'] = -1

    # 标记包含'recv_forward'和'recv_backward'的行
    df['recv_forward'] = df['s_name'].str.contains('recv_forward').astype(int).cumsum() - 1
    df['recv_backward'] = df['s_name'].str.contains('recv_backward').astype(int).cumsum() - 1

    # 更新'micro_batch_id_forward'和'micro_batch_id_backward'
    df.loc[df['s_name'].str.contains('forward'), 'micro_batch_id_forward'] = df['recv_forward']
    df.loc[df['s_name'].str.contains('backward'), 'micro_batch_id_backward'] = df['recv_backward']

    # 移除辅助列
    df.drop(['recv_forward', 'recv_backward'], axis=1, inplace=True)

def set_p2p_id(df):
    df['send_prev'] = -1
    df['send_next'] = -1
    df['recv_prev'] = -1
    df['recv_next'] = -1

    df.loc[df['s_name'].str.contains('mccl:recv(forward)', regex=False), 'recv_prev'] = df['micro_batch_id_forward']
    df.loc[df['s_name'].str.contains('mccl:recv(backward)', regex=False), 'recv_next'] = df['micro_batch_id_backward']
    df.loc[df['s_name'].str.contains('mccl:send(forward)', regex=False), 'send_next'] = df['micro_batch_id_forward']
    df.loc[df['s_name'].str.contains('mccl:send(backward)', regex=False), 'send_prev'] = df['micro_batch_id_backward']

def process_pipeline_start_end(rank, df):
    if is_first_stage(rank, TP_SIZE, PP_SIZE, DP_SIZE):
        df.loc[df['s_name'].str.contains('recv_forward'), 'recv_prev'] = -1
        df.loc[df['s_name'].str.contains('send_backward'), 'send_prev'] = -1
    if is_last_stage(rank, TP_SIZE, PP_SIZE, DP_SIZE):
        df.loc[df['s_name'].str.contains('recv_backward'), 'recv_next'] = -1
        df.loc[df['s_name'].str.contains('send_forward'), 'send_next'] = -1

def set_p2p_micro_batch_id(rank_trace_tuple):
    rank, trace_df = rank_trace_tuple
    set_micro_batch_id(trace_df)
    set_p2p_id(trace_df)
    process_pipeline_start_end(rank, trace_df)

    # trace_df = remove_rows_starting_with_names(trace_df, ['forward_step', 'backward_step'])
    # trace_df = trace_df.head(5)
    
    return rank, trace_df

def apply_function_for_parallel(traces, function, use_multiprocessing: bool = True, need_rank: bool = False):
    if not use_multiprocessing:
        total_results = {}
        for rank, trace_df in traces:
            logger.debug(f"applying func {function.__name__} for traces on rank {rank}")
            if need_rank:
                result = function((rank, trace_df))
            else:
                result = function(trace_df)
            total_results[rank] = result
        logger.debug(f"finished applying func {function.__name__} for traces")
        return total_results
    else:
        num_procs = min(mp.cpu_count(), len(traces))
        logger.debug(f"parallel applying func {function.__name__} for traces using {num_procs} processes.")
        with mp.get_context("fork").Pool(num_procs) as pool:
            if need_rank:
                results = pool.map(function, traces.items())
            else:
                results = pool.map(function, traces.values())
            pool.close()
            pool.join()
        logger.debug(f"finished parallel applying func {function.__name__} for traces using {num_procs} processes.")
        
        if need_rank:
            return {rank: processed_df for rank, processed_df in results}
        return {rank: processed_df for rank, processed_df in zip(traces.keys(), results)}

# def get_p2p_trace_for_one_pair(rank_df_tuple):
#     rank_prev = rank_df_tuple[0]
#     trace_df_prev_orig = rank_df_tuple[1][0]
#     trace_df_next_orig = rank_df_tuple[1][1]

#     p2p_forward_pd = pd.merge(trace_df_prev_orig, trace_df_next_orig, left_on='send_next', right_on='recv_prev', how='inner', suffixes=('_on_prev', '_on_next'))
#     p2p_forward_pd = p2p_forward_pd[p2p_forward_pd['send_next_on_prev'] >= 0]
#     p2p_forward_pd['p2p_forward'] = True
#     p2p_forward_pd['comm_time'] = np.minimum(p2p_forward_pd['dur_on_prev'], p2p_forward_pd['dur_on_next'])
#     print('p2p_forward_pd')
#     print(p2p_forward_pd[['s_name_on_prev', 's_name_on_next', 'send_next_on_prev', 'recv_prev_on_next', 'dur_on_prev', 'dur_on_next', 'comm_time']].head(20))
#     print('trace_df_prev')
#     print(trace_df_prev_orig[['s_name', 'send_next', 'recv_next', 'send_prev', 'recv_prev', 'dur']].head(20))
#     trace_df_prev_new = pd.merge(
#         trace_df_prev_orig,
#         p2p_forward_pd[['send_next_on_prev', 'comm_time']],
#         left_on='send_next',
#         right_on='send_next_on_prev',
#         how='left').drop(columns='send_next_on_prev')
#     # print('after merge trace_df_prev')
#     # print(trace_df_prev_new[['s_name', 'send_next', 'recv_next', 'send_prev', 'recv_prev', 'send_next_on_prev', 'dur', 'comm_time']].head(20))
#     # trace_df_prev = trace_df_prev.drop(columns='send_next_on_prev')
    
#     trace_df_next_new = pd.merge(
#         trace_df_next_orig,
#         p2p_forward_pd[['recv_prev_on_next', 'comm_time']],
#         left_on='recv_prev',
#         right_on='recv_prev_on_next',
#         how='left').drop(columns='recv_prev_on_next')

#     p2p_backward_pd = pd.merge(trace_df_prev_orig, trace_df_next_orig, left_on='recv_next', right_on='send_prev', how='inner', suffixes=('_on_prev', '_on_next'))
#     p2p_backward_pd = p2p_backward_pd[p2p_backward_pd['recv_next_on_prev'] >= 0]
#     p2p_backward_pd['p2p_backward'] = True
#     p2p_backward_pd['comm_time'] = np.minimum(p2p_backward_pd['dur_on_prev'], p2p_backward_pd['dur_on_next'])
#     print('p2p_backward_pd')
#     print(p2p_backward_pd[['s_name_on_prev', 's_name_on_next', 'send_next_on_prev', 'recv_prev_on_next', 'dur_on_prev', 'dur_on_next', 'comm_time']].head(20))
#     print('trace_df_prev')
#     print(trace_df_prev_new[['s_name', 'send_next', 'recv_next', 'send_prev', 'recv_prev', 'dur', 'comm_time']].head(20))
#     trace_df_prev_new = pd.merge(
#         trace_df_prev_new,
#         p2p_backward_pd[['recv_next_on_prev', 'comm_time']],
#         left_on='recv_next',
#         right_on='recv_next_on_prev',
#         how='left',
#         suffixes=('', '_new'))
#     print('after merge trace_df_prev')
#     print(trace_df_prev_new[['s_name', 'send_next', 'recv_next', 'send_prev', 'recv_prev', 'recv_next_on_prev', 'dur', 'comm_time', 'comm_time_new']].head(20))
#     trace_df_prev_new = trace_df_prev_new.drop(columns='recv_next_on_prev')
#     trace_df_prev_new['comm_time'] = trace_df_prev_new['comm_time'].fillna(0) + trace_df_prev_new['comm_time_new'].fillna(0)
#     trace_df_prev_new = trace_df_prev_new.drop(columns='comm_time_new')

#     trace_df_next_new = pd.merge(
#         trace_df_next_new,
#         p2p_backward_pd[['send_prev_on_next', 'comm_time']],
#         left_on='send_prev',
#         right_on='send_prev_on_next',
#         how='left',
#         suffixes=('', '_new')).drop(columns='send_prev_on_next')

#     trace_df_next_new['comm_time'] = trace_df_next_new['comm_time'].fillna(0) + trace_df_next_new['comm_time_new'].fillna(0)
#     trace_df_next_new = trace_df_next_new.drop(columns='comm_time_new')

#     trace_df_prev_orig['comm_time'] = trace_df_prev_new['comm_time']
#     trace_df_prev_orig['wait_time'] = trace_df_prev_orig['dur'] - trace_df_prev_orig['comm_time']
#     trace_df_next_orig['comm_time'] = trace_df_next_new['comm_time']
#     trace_df_next_orig['wait_time'] = trace_df_next_orig['dur'] - trace_df_next_orig['comm_time']
#     print(trace_df_prev_orig[['dur', 'comm_time', 'wait_time']].head(20))
#     print(trace_df_next_orig[['dur', 'comm_time', 'wait_time']].head(20))

#     all_p2p_pd = pd.concat([p2p_forward_pd, p2p_backward_pd])
    
#     return rank_prev, all_p2p_pd 

# def get_p2p_devices_pairs(ranks, traces):
#     if len(ranks) < 2: return {}
#     ranks_sorted = sorted(ranks)
#     p2p_devices_pairs = {}
#     for i in range(len(ranks_sorted) - 1):
#         if(ranks[i+1] == get_next_pipeline_rank(ranks[i], TP_SIZE, PP_SIZE, DP_SIZE)):
#             p2p_devices_pairs[ranks[i]] = [traces[ranks[i]], traces[ranks[i+1]]]
#     return p2p_devices_pairs

def get_p2p_trace_for_one_pair(trace_df_prev, trace_df_next):
    trace_df_prev_orig = trace_df_prev
    trace_df_next_orig = trace_df_next

    p2p_forward_pd = pd.merge(trace_df_prev_orig, trace_df_next_orig, left_on='send_next', right_on='recv_prev', how='inner', suffixes=('_on_prev', '_on_next'))
    p2p_forward_pd = p2p_forward_pd[p2p_forward_pd['send_next_on_prev'] >= 0]
    p2p_forward_pd['p2p_forward'] = True
    p2p_forward_pd['comm_time'] = np.minimum(p2p_forward_pd['dur_on_prev'], p2p_forward_pd['dur_on_next'])
    # print('p2p_forward_pd')
    # print(p2p_forward_pd[['s_name_on_prev', 's_name_on_next', 'send_next_on_prev', 'recv_prev_on_next', 'dur_on_prev', 'dur_on_next', 'comm_time']].head(20))
    # print('trace_df_prev')
    # print(trace_df_prev_orig[['s_name', 'send_next', 'recv_next', 'send_prev', 'recv_prev', 'dur']].head(20))
    trace_df_prev_new = pd.merge(
        trace_df_prev_orig,
        p2p_forward_pd[['send_next_on_prev', 'comm_time']],
        left_on='send_next',
        right_on='send_next_on_prev',
        how='left').drop(columns='send_next_on_prev')
    # print('after merge trace_df_prev')
    # print(trace_df_prev_new[['s_name', 'send_next', 'recv_next', 'send_prev', 'recv_prev', 'send_next_on_prev', 'dur', 'comm_time']].head(20))
    # trace_df_prev = trace_df_prev.drop(columns='send_next_on_prev')
    
    trace_df_next_new = pd.merge(
        trace_df_next_orig,
        p2p_forward_pd[['recv_prev_on_next', 'comm_time']],
        left_on='recv_prev',
        right_on='recv_prev_on_next',
        how='left').drop(columns='recv_prev_on_next')

    p2p_backward_pd = pd.merge(trace_df_prev_orig, trace_df_next_orig, left_on='recv_next', right_on='send_prev', how='inner', suffixes=('_on_prev', '_on_next'))
    p2p_backward_pd = p2p_backward_pd[p2p_backward_pd['recv_next_on_prev'] >= 0]
    p2p_backward_pd['p2p_backward'] = True
    p2p_backward_pd['comm_time'] = np.minimum(p2p_backward_pd['dur_on_prev'], p2p_backward_pd['dur_on_next'])
    # print('p2p_backward_pd')
    # print(p2p_backward_pd[['s_name_on_prev', 's_name_on_next', 'send_next_on_prev', 'recv_prev_on_next', 'dur_on_prev', 'dur_on_next', 'comm_time']].head(20))
    # print('trace_df_prev')
    # print(trace_df_prev_new[['s_name', 'send_next', 'recv_next', 'send_prev', 'recv_prev', 'dur', 'comm_time']].head(20))
    trace_df_prev_new = pd.merge(
        trace_df_prev_new,
        p2p_backward_pd[['recv_next_on_prev', 'comm_time']],
        left_on='recv_next',
        right_on='recv_next_on_prev',
        how='left',
        suffixes=('', '_new'))
    # print('after merge trace_df_prev')
    # print(trace_df_prev_new[['s_name', 'send_next', 'recv_next', 'send_prev', 'recv_prev', 'recv_next_on_prev', 'dur', 'comm_time', 'comm_time_new']].head(20))
    trace_df_prev_new = trace_df_prev_new.drop(columns='recv_next_on_prev')
    trace_df_prev_new['comm_time'] = trace_df_prev_new['comm_time'].fillna(0) + trace_df_prev_new['comm_time_new'].fillna(0)
    trace_df_prev_new = trace_df_prev_new.drop(columns='comm_time_new')

    trace_df_next_new = pd.merge(
        trace_df_next_new,
        p2p_backward_pd[['send_prev_on_next', 'comm_time']],
        left_on='send_prev',
        right_on='send_prev_on_next',
        how='left',
        suffixes=('', '_new')).drop(columns='send_prev_on_next')

    trace_df_next_new['comm_time'] = trace_df_next_new['comm_time'].fillna(0) + trace_df_next_new['comm_time_new'].fillna(0)
    trace_df_next_new = trace_df_next_new.drop(columns='comm_time_new')

    trace_df_prev_orig['comm_time'] = trace_df_prev_new['comm_time']
    trace_df_prev_orig['wait_time'] = trace_df_prev_orig['dur'] - trace_df_prev_orig['comm_time']
    trace_df_next_orig['comm_time'] = trace_df_next_new['comm_time']
    trace_df_next_orig['wait_time'] = trace_df_next_orig['dur'] - trace_df_next_orig['comm_time']
    # print(trace_df_prev_orig[['dur', 'comm_time', 'wait_time']].head(20))
    # print(trace_df_next_orig[['dur', 'comm_time', 'wait_time']].head(20))

    all_p2p_pd = pd.concat([p2p_forward_pd, p2p_backward_pd])
    
    return all_p2p_pd, trace_df_prev_orig, trace_df_next_orig

def get_p2p_ranks_pairs(ranks):
    if len(ranks) < 2: return []
    ranks_sorted = sorted(ranks)
    p2p_devices_pairs = []
    for i in range(len(ranks_sorted) - 1):
        if(ranks[i+1] == get_next_pipeline_rank(ranks[i], TP_SIZE, PP_SIZE, DP_SIZE)):
            p2p_devices_pairs.append([ranks[i], ranks[i+1]])
            # p2p_devices_pairs[ranks[i]] = [traces[ranks[i]], traces[ranks[i+1]]]
    return p2p_devices_pairs 

def combine_into_one_trace(traces_dict: dict):
    all_trace_dfs = []
    for rank, trace_df in traces_dict.items():
        all_trace_dfs.append(trace_df)
    trace_df = pd.concat(all_trace_dfs, ignore_index=True)
    return trace_df

def process_p2p_relation(trace):
    rank_pairs = get_p2p_ranks_pairs(trace.get_ranks())
    all_p2p_pd = {}
    for rank_prev, rank_next in rank_pairs:
        p2p_pd, trace_df_prev_new, trace_df_next_new = get_p2p_trace_for_one_pair(trace.traces[rank_prev], trace.traces[rank_next])
        all_p2p_pd[rank_prev] = p2p_pd
        trace.traces[rank_prev] = trace_df_prev_new
        trace.traces[rank_next] = trace_df_next_new
    return all_p2p_pd

def avg_time(trace_df, name, col='s_name'):
    return trace_df[trace_df[col] == name]['dur'].mean()

def get_report_for_pp_group(trace):
    output_df = pd.DataFrame()
    for rank, trace_df in trace.traces.items():
        sorted_trace_df = trace_df.sort_values(by=['ts', 'dur'], ascending=[True, False])
        time_per_batch = keep_rows_starting_with_names(sorted_trace_df, ['ProfilerStep'])['dur'].mean()

        all_forward_steps_df = keep_rows_starting_with_names(sorted_trace_df, ['forward_step'])
        steady_forward_steps_df = all_forward_steps_df.iloc[PP_SIZE+1:-1]
        forward_step_time = steady_forward_steps_df['dur'].mean()

        all_backward_steps_df = keep_rows_starting_with_names(sorted_trace_df, ['backward_step'])
        steady_backward_steps_df = all_backward_steps_df.iloc[1:-PP_SIZE-1]
        backward_step_time = steady_backward_steps_df['dur'].mean()

        all_p2p_dfs = keep_rows_starting_with_names(sorted_trace_df, ['mccl:send', 'mccl:recv'])
        total_comm_time = all_p2p_dfs['comm_time'].sum()
        total_wait_time = all_p2p_dfs['wait_time'].sum()
        total_bubble_time = (PP_SIZE - 1) * (forward_step_time + backward_step_time) 

        total_compute_time = NUM_MICROBATCHES * (forward_step_time + backward_step_time)

        info_per_rank = {
            'rank': rank,
            'time_per_batch': f'{time_per_batch/1000}ms',
            'microbatch_num': NUM_MICROBATCHES,
            'forward_step_time': f'{forward_step_time/1000}ms',
            'backward_step_time': f'{backward_step_time/1000}ms',
            'time_per_microbatch': f'{(forward_step_time+backward_step_time)/1000}ms',
            'total_comm_time': f'{total_comm_time/1000}ms',
            'total_wait_time': f'{total_wait_time/1000}ms',
            'total_bubble_time': f'{total_bubble_time/1000}ms',
            'total_compute_time': f'{total_compute_time/1000}ms',
            'comp_time_ratio': total_compute_time / time_per_batch,
            'bubble_time_ratio': total_bubble_time / time_per_batch,
            'wait_time_ratio': total_wait_time / time_per_batch
        }

        output_df[len(output_df)] = info_per_rank
    
    return output_df

def keep_only_comm_event(trace_df):
    trace_df_comm = keep_rows_starting_with_names(trace_df, ['forward_step', 'backward_step', 'recv_forward', 'recv_backward', 'send_forward', 'send_backward', 'send_forward_recv_backward', 'send_backward_recv_forward', 'mccl:send', 'mccl:recv', 'ProfilerStep'])
    trace_df_comm = mark_send_recv_direction(trace_df_comm)
    return trace_df_comm

def process_single_pp_group(trace_dir):
    output_dir = os.path.join(trace_dir, 'output')
    prepare_directory(output_dir, force_clear=True)
    
    analyzer = load_trace_analyer(trace_dir)
    analyzer.t.decode_symbol_ids()
    display_traces_info(analyzer.t.traces)
    analyzer.t.traces = analyzer.t.parallel_apply(filter_single_trace_df)
    analyzer.t.save_traces('after_filter.json', ranks=[0])
    analyzer.t.traces = analyzer.t.parallel_apply(calculate_comm_volume_for_trace_df)
    analyzer.t.traces = analyzer.t.parallel_apply(calculate_flops_for_trace_df)
    all_call_tree = analyzer.t.parallel_apply(build_call_tree)
    for rank, call_tree in all_call_tree.items():
        call_tree.print_tree(f'{output_dir}/result_rank{rank}.txt')
    analyzer.t.traces = analyzer.t.parallel_apply(keep_only_comm_event) 
    analyzer.t.traces = analyzer.t.parallel_apply(set_p2p_micro_batch_id, need_rank=True)

    # all_p2p_devices_pairs = get_p2p_devices_pairs(analyzer.t.get_ranks(), analyzer.t.traces)
    # p2p_comm_traces = apply_function_for_parallel(all_p2p_devices_pairs, get_p2p_trace_for_one_pair, need_rank=True)
    p2p_comm_traces = process_p2p_relation(analyzer.t)
    all_call_tree_only_comm = analyzer.t.parallel_apply(build_call_tree)
    for rank, call_tree in all_call_tree_only_comm.items():
        call_tree.print_tree(f'{output_dir}/result_only_comm_rank{rank}.txt')

    analyzer.t.p2p_flow_events_df = combine_into_one_trace(p2p_comm_traces)
    analyzer.t.trace_df_only_comm_for_all_ranks = combine_into_one_trace(analyzer.t.traces)

    save_trace_df_to_file(analyzer.t.trace_df_only_comm_for_all_ranks, f'{output_dir}/trace_only_comm_all_ranks_with_flow.json', p2p_comm_flow_df=analyzer.t.p2p_flow_events_df)
    output_df = get_report_for_pp_group(analyzer.t)
    print(output_df)
    output_df.to_csv(f'{output_dir}/output.csv', header=True)
    # analyzer.t.call_graph = CallGraph(analyzer.t)
    # print(analyzer.t.call_graph)

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
    main()
