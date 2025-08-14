from typing import List, Tuple

import pandas as pd
import json
from functools import reduce
import operator
import numpy as np

from hta.common.trace_symbol_table import TraceSymbolTable
from hta.configs.config import logger

def get_iterations(df: pd.DataFrame) -> List[int]:
    """Extract the iteration numbers from a trace DataFrame.

    Args:
        df (pd.DataFrame): an input DataFrame.

    Returns:
        iterations (List[int]): a list iteration numbers.
    """
    if "iteration" not in df.columns:
        raise TypeError("The input DataFrame doesn't contain the `iteration` column.")

    if df.dtypes["iteration"].kind != "i":
        raise TypeError(
            "The data type of `iteration` column in the input DataFrame is not integer."
        )

    iterations = sorted(df["iteration"].unique())

    if len(iterations) != [-1]:
        return iterations if iterations[0] != -1 else iterations[1:]
    else:
        return []


def find_op_occurrence(
    df: pd.DataFrame, op_name: str, position: int, name_column: str = "s_name"
) -> Tuple[bool, pd.Series]:
    """Find a specific occurrence of trace event matching the specified op name and position.
    Args:
        df: a DataFrame with trace data.
        op_name: name of the operator. e.g., "split_embedding_codegen_forward_unweighted_kernel".
        position: the occurrence position of the operator. Use zero or positive values for forward
            counting and negative values for backward counting. For example, position=0 means
            the first occurrence of the operator and position=-1 means the last (latest) occurrence.
        name_column: Optional; name of the data frame column containing the operator name.
            Default: "s_name".
    Returns:
        A boolean value and a Series.
        The boolean value is True if there is a match, otherwise False.
        When there is a match, the Series is the matching event.
    """
    if any([df.empty, "ts" not in df.columns, name_column not in df.columns]):
        return False, pd.Series()

    ops = df.loc[df[name_column].eq(op_name)].sort_values("ts")
    pos = position if position >= 0 else len(ops) + position

    if len(ops) > 0 and 0 <= pos < len(ops):
        return True, ops.iloc[pos]
    else:
        return False, pd.Series()


def find_events_by_name_patterns_using_symbol_table(
    df: pd.DataFrame, list_name_patterns: List[str], symbol_table: TraceSymbolTable
) -> pd.Series:
    """Searches for events in the provided DataFrame using a list of name patterns, leveraging a symbol table for memory saving and fast match.
    Args:
        df: The DataFrame to search. It should contain a column 'name' with event names and 'index' with event indices.
        list_name_patterns: A list of regular expression pattern strings to search for in the event names.
        symbol_table: The TraceSymbolTable which maps the name and cat symbols to corresponding integer IDs.
    Returns:
        A Series object containing the indices of rows in the original DataFrame that match any of
        the name patterns in the list.
    """
    sym_index = pd.Series(symbol_table.get_sym_id_map())
    matched_ids = set()
    for module in list_name_patterns:
        matched_ids.update(sym_index.loc[sym_index.index.str.match(module)].values)
    indices: pd.Series = df.loc[df["name"].isin(matched_ids)]["index"]
    return indices


def find_events_by_name_patterns_using_decoded_names(
    df: pd.DataFrame, list_name_patterns: List[str], name_column: str = "s_name"
) -> pd.Series:
    """Searches for events in the provided DataFrame using a list of name patterns.
    Args:
        df (pd.DataFrame): The DataFrame to search. It should contain a column for event names specified by `name_column`.
        list_name_patterns (List[str]):  A list of regular expression pattern strings to search for in the event names.
        name_column (str, optional): The name of the column in `df` that contains the event names. Defaults to 's_name'.

    Returns:
        A Series object containing the indices of rows in the original DataFrame that match any of
            the name patterns in the list.
    """
    indices = set()
    for module in list_name_patterns:
        indices.update(df.loc[df[name_column].str.match(module)]["index"])
    return pd.Series(df.loc[list(indices)]["index"])

def convert_to_flow_events(trace_df_p2p_comm_flow: pd.DataFrame):
    if trace_df_p2p_comm_flow is None: return []
    
    df_p2p_forward = trace_df_p2p_comm_flow[trace_df_p2p_comm_flow['p2p_forward'] == True]
    df_p2p_backward = trace_df_p2p_comm_flow[trace_df_p2p_comm_flow['p2p_backward'] == True]

    send_forward_pd = pd.DataFrame({
        'cat': 'p2p_forward',
        'name': 'p2p_forward',
        'ph': 's',
        'pid': df_p2p_forward['pid_on_prev'],
        'tid': df_p2p_forward['tid_on_prev'],
        'ts': np.maximum(df_p2p_forward['ts_on_prev'], df_p2p_forward['ts_on_next']),
        'id': df_p2p_forward.index,
        'args': [
            {'micro_batch_id': micro_batch_id}
            for micro_batch_id in df_p2p_forward['micro_batch_id_forward_on_prev']
        ]
    })
    
    send_backward_pd = pd.DataFrame({
        'cat': 'p2p_backward',
        'name': 'p2p_backward',
        'ph': 's',
        'pid': df_p2p_backward['pid_on_next'],
        'tid': df_p2p_backward['tid_on_next'],
        'ts': np.maximum(df_p2p_backward['ts_on_prev'], df_p2p_backward['ts_on_next']),
        'id': df_p2p_backward.index,
        'args': [
            {'micro_batch_id': micro_batch_id}
            for micro_batch_id in df_p2p_backward['micro_batch_id_backward_on_prev']
        ]
    })

    recv_forward_pd = pd.DataFrame({
        'cat': 'p2p_forward',
        'name': 'p2p_forward',
        'ph': 'f',
        'pid': df_p2p_forward['pid_on_next'],
        'tid': df_p2p_forward['tid_on_next'],
        'ts': df_p2p_forward['ts_on_next'] + df_p2p_forward['dur_on_next'],
        'id': df_p2p_forward.index,
        'bp': 'e',
        'args': [
            {'micro_batch_id': micro_batch_id}
            for micro_batch_id in df_p2p_forward['micro_batch_id_forward_on_prev']
        ]
    })

    recv_backward_pd = pd.DataFrame({
        'cat': 'p2p_backward',
        'name': 'p2p_backward',
        'ph': 'f',
        'pid': df_p2p_backward['pid_on_prev'],
        'tid': df_p2p_backward['tid_on_prev'],
        'ts': df_p2p_backward['ts_on_prev'] + df_p2p_backward['dur_on_prev'],
        'id': df_p2p_backward.index,
        'bp': 'e',
        'args': [
            {'micro_batch_id': micro_batch_id}
            for micro_batch_id in df_p2p_backward['micro_batch_id_backward_on_prev']
        ]
    })

    total_dicts = [
        item
        for pd in [send_forward_pd, send_backward_pd, recv_forward_pd, recv_backward_pd]
        for item in pd.to_dict('records')
    ]

    return total_dicts

def generate_metadata_events(rank_pid_pairs):
    metadata_events = []
    rank_pid_pairs = sorted(rank_pid_pairs)
    for i, (rank, pid) in enumerate(rank_pid_pairs):
        metadata_events.append(
            {
                'name': 'process_sort_index',
                'ph': 'M',
                'pid': pid,
                'args':{
                    'sort_index': i
                }
            }
        )
        metadata_events.append(
            {
                'name': 'process_name',
                'ph': 'M',
                'pid': pid,
                'args':{
                    'name': f'rank {rank}'
                }
            }
        )
    return metadata_events

def save_trace_df_to_file(df: pd.DataFrame, output_file: str, trace_df_p2p_comm_flow: pd.DataFrame=None, meta_data: dict=None):
    columns_to_keep = ['name', 'cat', 'pid', 'tid', 'ts', 'dur', 'rank']
    columns_to_drop = ['s_name', 's_cat']
    
    new_df = df[columns_to_keep].copy()
    new_df['name'] = df['s_name']
    new_df['cat'] = df['s_cat']
    new_df['ph'] = 'X'
    new_df['args'] = df.apply(lambda row: {col: row[col] for col in row.index if col not in columns_to_keep + columns_to_drop}, axis=1)

    trace_data = meta_data.copy() if meta_data is not None else {}
    trace_events = new_df.to_dict('records')
    flow_events = convert_to_flow_events(trace_df_p2p_comm_flow)
    metadata_events = generate_metadata_events([tuple(x) for x in new_df[['rank', 'pid']].drop_duplicates().to_records(index=False)])
    trace_data["traceEvents"] = trace_events + flow_events + metadata_events
    
    with open(output_file, 'w') as f:
        json.dump(trace_data, f, indent=4)

def _prod(shape):
    return reduce(operator.mul, shape, 1)

def get_calculate_flops_function(op_name):
    def calculate_attention_flops(input_dims):
        q_shape = input_dims[0]
        k_shape = input_dims[1]
        v_shape = input_dims[2]
        macs = _prod(q_shape) * k_shape[-2]
        macs += _prod(q_shape[:-1]) * k_shape[-2] * v_shape[-1]
        return 2 * macs
    
    def calculate_matmul_flops(input_dims):
        return _prod(input_dims[0]) * input_dims[1][-1] * 2
    
    def calculate_rms_norm_flops(input_dims):
        weight_shape = input_dims[2] 
        has_affine = len(weight_shape) > 0

        return _prod(input_dims[0]) * (5 if has_affine else 4)
    
    def calculate_noop_flops(*args):
        return -1

    if op_name == 'aten::matmul':
        return calculate_matmul_flops
    elif op_name == 'aten::scaled_dot_product_attention':
        return calculate_attention_flops
    elif op_name == 'aten::rms_norm_forward':
        return calculate_rms_norm_flops
    else:
        return calculate_noop_flops

def get_calculate_comm_volumn_function(op_name):
    TP_SIZE = 2
    DP_SIZE = 2
    PP_SIZE = 4
    def get_num_of_bytes(type):
        bytes_dict = {
            'c10::Half': 2,
            'c10::Float': 4,
            'c10::Int': 1,
            'c10::BFloat16': 2,
            'long int': 8,
            'float': 4,
            'unsigned char': 1
        }
        if type not in bytes_dict: 
            print(f'type {type} not in bytes_dict, is set to 1B')
        return bytes_dict.get(type, 1)

    def calculate_all_gather_comm_volumn(input_dims, input_type):
        n_rank = TP_SIZE
        return _prod(input_dims[0]) * get_num_of_bytes(input_type[0]) * (n_rank - 1) 
    
    def calculate_reduce_scatter_comm_volumn(input_dims, input_type):
        n_rank = TP_SIZE
        return _prod(input_dims[0]) * get_num_of_bytes(input_type[0]) * (n_rank - 1) / n_rank
    
    def calculate_p2p_comm_volumn(input_dims, input_type):
        return _prod(input_dims[0]) * get_num_of_bytes(input_type[0])
    
    def calculate_all_reduce_comm_volumn(input_dims, input_type):
        n_rank = DP_SIZE
        return _prod(input_dims[0]) * get_num_of_bytes(input_type[0]) * 2 * (n_rank - 1) / n_rank
    
    def calculate_broadcast_comm_volumn(input_dims, input_type):
        n_rank = DP_SIZE
        return _prod(input_dims[0]) * get_num_of_bytes(input_type[0]) * n_rank

    def calculate_noop_comm_volumn(*args):
        return -1

    if op_name == 'mccl:_all_gather_base':
        return calculate_all_gather_comm_volumn
    elif op_name == 'mccl:_reduce_scatter_base':
        return calculate_reduce_scatter_comm_volumn
    elif op_name.startswith(('mccl:send', 'mccl:recv')):
        return calculate_p2p_comm_volumn
    elif op_name in ['mccl:broadcast']:
        return calculate_broadcast_comm_volumn
    elif op_name in ['mccl:all_reduce']:
        return calculate_all_reduce_comm_volumn
    else:
        return calculate_noop_comm_volumn

def calculate_flops_for_trace_df(trace_df):
    trace_df['flops'] = trace_df.apply(lambda row: get_calculate_flops_function(row['s_name'])(row['input_dims']), axis=1)
    trace_df['TFLOPS'] = trace_df.apply(lambda row: row['flops'] / row['kernel_span'] * 1e-6 if row['flops'] > 0 and row['kernel_span'] > 0 else -1, axis=1)
    return trace_df

def calculate_comm_volume_for_trace_df(trace_df):
    trace_df['comm_volume'] = trace_df.apply(lambda row: get_calculate_comm_volumn_function(row['s_name'])(row['input_dims'], row['input_type']), axis=1)
    trace_df['bandwidth'] = trace_df.apply(
        lambda row: (
            row['comm_volume'] / row.get('comm_time', row['kernel_span']) * 1e-3 
            if row['s_name'].startswith(('send_forward', 'recv_forward')) and row.get('comm_time', row['kernel_span']) > 0
            else (row['comm_volume'] / row['kernel_span'] * 1e-3 if row['comm_volume'] > 0 and row['dur'] > 0 else -1)
        ), 
        axis=1
    )

    return trace_df
