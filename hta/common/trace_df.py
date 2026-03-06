from typing import List, Tuple

import pandas as pd
import json
import numpy as np

from hta.common.trace_symbol_table import TraceSymbolTable


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
    new_df['ts'] = df['first_kernel_start']
    new_df['dur'] = df['kernel_span']
    new_df['name'] = df['s_name']
    new_df['cat'] = df['s_cat']
    new_df['ph'] = 'X'
    # Todo: in interleaved PP, send_fwd_recv_fwd and send_bwd_recv_bwd execute asyn and in parallel with fwd_step or bwd_step
    # so for displaying in perfetto, it muse set them with different tids.
    #new_df.loc[new_df['name'].str.match(pat=r"^(send_forward_recv_forward|send_backward_recv_backward)$"), 'tid'] = 1
    #new_df.loc[new_df['name'].str.match(pat=r"^mccl:recv$"), 'tid'] = 2
    #new_df.loc[new_df['name'].str.match(pat=r"^mccl:send$"), 'tid'] = 3
    #new_df['args'] = df.apply(lambda row: {col: row[col] for col in row.index if col not in columns_to_keep + columns_to_drop}, axis=1)

    trace_data = meta_data.copy() if meta_data is not None else {}
    trace_events = new_df.to_dict('records')
    #flow_events = convert_to_flow_events(trace_df_p2p_comm_flow)
    metadata_events = generate_metadata_events([tuple(x) for x in new_df[['rank', 'pid']].drop_duplicates().to_records(index=False)])
    trace_data["traceEvents"] = trace_events + metadata_events
    
    with open(output_file, 'w') as f:
        json.dump(trace_data, f, indent=4)
