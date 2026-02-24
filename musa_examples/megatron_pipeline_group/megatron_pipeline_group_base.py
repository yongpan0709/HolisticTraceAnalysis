# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
from abc import ABC, abstractmethod
import os
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import pandas as pd
import re

from hta.common.trace import Trace
from hta.common.trace_df import save_trace_df_to_file
from hta.common.trace_filter import create_regex_for_prefix_match
from hta.configs.config import logger
from hta.configs.default_values import DEFAULT_TRACE_DIR
from hta.configs.parser_config import ParserConfig
from hta.utils.utils import get_mp_pool_size, normalize_path
from hta.common.trace_filter import NameFilter
from hta.utils.utils import add_rank_to_filename, apply_function_for_parallel
from hta.common.trace_call_graph import CallGraph
from hta.utils.parallel_state import RankGenerator
from hta.common.trace_file import get_trace_files


def parallel_callgraph_create(rank_id, trace_file):
    logger.debug(f'rank id: {rank_id}, trace_file: {trace_file}')
    t = Trace(trace_files={rank_id: trace_file}, trace_dir="")
    t.load_traces()
    t.decode_symbol_ids(use_shorten_name=False)
    cg = CallGraph(t)
    _, main_stack = cg.get_main_stack_on_rank(rank_id)
    full_df = main_stack.full_df.copy()
    del cg
    del t
    #Todo: ProfilerStep col kernel span has a wrong value 
    full_df.loc[full_df['s_name'].str.contains(r'^ProfilerStep#.*'), 'kernel_span'] = full_df[full_df['s_name'].str.match(pat=r"^megatron/training/training.py\(\d+\): pretrain$")]['kernel_span'].values
    return rank_id, full_df[full_df['s_cat'] == 'user_annotation' ]

# Todo: Does the pp group trace class need to know information about other tp, ep, dp groups?
class MegatronPipelineParallelGroupTraceBase(ABC):
    """
    MegatronPipelineParallelGroupTraceBase class for Megatron-LM with Pipeline Parallel group.
    Models are getting larger; load one pp group at a time to avoid OOM (out of memory) and slow parsing errors.
    """
    def __init__(
        self,
        trace_files: Optional[Dict[int, str]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
        dp = -1,
        tp = -1,
        pp = -1,
        ep = -1,
        cp: int =1, order: str ="tp-cp-ep-dp-pp",
        # pp_schedule: str = "1f1b",
        # vpp_size = -1,
    ) -> None:
        if trace_files is None:
            assert os.path.exists(trace_dir), f"Trace directory {trace_dir} does not exist!"
        self.trace_files = get_trace_files(trace_dir)
        assert self.trace_files is not None and len(self.trace_files) > 0, f"No trace files found in directory {trace_dir}!"
        self.trace_dir = trace_dir
        self.tensor_parallel_size = tp
        self.data_parallel_size = dp
        self.pipeline_parallel_size = pp
        self.expert_model_parallel_size = ep
        self.context_parallel_size = cp
        # self.pp_schedule = pp_schedule
        # self.vpp_size = vpp_size
        self.expert_decoder_rank_generator = RankGenerator(
            tp=tp,
            ep=ep,
            dp=dp,
            pp=pp,
            cp=cp,
            order=order,
            rank_offset=0,
        )
        self.all_data_parallel_group_ranks = self.expert_decoder_rank_generator.get_ranks('dp')
        self.all_tensor_parallel_group_ranks = self.expert_decoder_rank_generator.get_ranks('tp')
        self.all_pipeline_parallel_group_ranks = self.expert_decoder_rank_generator.get_ranks('pp')
        self.with_gpu_kernel = False
        self.traces_comm_only: Dict[int, pd.DataFrame] = {}
        #self.pp_group_trace: Dict[int, Trace] = {}
        self.full_dfs: Dict[int, pd.DataFrame] = {}
        self.is_parsed_per_pp_group: Dict[int, bool] = {}

    # def get_ranks(self, pp_group_id: int = 0) -> List[int]:
    #     """返回指定 PP 组内的 rank 列表。"""
    #     return self.all_pipeline_parallel_group_ranks[pp_group_id]

    def parse_traces_per_pp_group(self, pp_group_id=0) -> None:
        if self.is_parsed_per_pp_group.get(pp_group_id, False):
            logger.warning("Traces are already parsed and loaded!")
            return
        num_procs = min(mp.cpu_count(), len(self.all_pipeline_parallel_group_ranks[pp_group_id]))
        with mp.get_context("fork").Pool(num_procs) as pool:
            tasks = [(rank_i, self.trace_files[rank_i]) for rank_i in self.all_pipeline_parallel_group_ranks[pp_group_id]]
            results = pool.starmap(parallel_callgraph_create, tasks)
            pool.close()
            pool.join()
        for rank_id, main_stack_df in results:
            logger.debug(f"rank id: {rank_id}")
            self.full_dfs[rank_id] = main_stack_df
        self.is_parsed_per_pp_group[pp_group_id] = True

    def etl_traces_per_pp_group(self, redirect_trace_dir, filter_out_funcs, pp_group_id=0) -> None:
        if self.is_parsed_per_pp_group.get(pp_group_id, False):
            logger.warning("Traces are already parsed and loaded!")
            return
        tasks = []
        t0 = time.perf_counter()
        for rank in self.all_pipeline_parallel_group_ranks[pp_group_id]:
            trace_file = self.trace_files[rank]
            logger.debug(f'rank {rank} trace file:{trace_file}')
            filename = os.path.basename(trace_file)
            redirect_new_trace_path = os.path.join(redirect_trace_dir, filename)
            tasks.append((trace_file, redirect_new_trace_path))
        num_procs = min(mp.cpu_count(), len(self.all_pipeline_parallel_group_ranks[pp_group_id]))
        with mp.get_context("fork").Pool(num_procs) as pool:
            results = pool.starmap(filter_out_funcs, tasks)
            pool.close()
            pool.join()
        t1 = time.perf_counter()
        logger.debug(f"calculating critical path took {t1 - t0:2f} seconds")
        self.is_parsed_per_pp_group[pp_group_id] = True

    # @abstractmethod
    def set_self_microbatch_id(self, trace_df: pd.DataFrame) -> None:
        """根据当前 PP 算法为 trace 打上 micro-batch id(forward/backward 等）。子类必须实现。"""
        pass

    # @abstractmethod
    def set_recv_send_microbatch_id(self, trace_df: pd.DataFrame) -> None:
        """根据 send/recv 与 micro-batch 关系设置 recv_prev/recv_next/send_prev/send_next。子类必须实现。"""
        pass

    # Use: func annotations
    # @abstractmethod
    def process_pipeline_start(self, df):
        #if is_first_stage(rank, self.tensor_parallel_size, self.pipeline_parallel_size, self.data_parallel_size):
        pass

    # @abstractmethod
    def process_pipeline_end(self, df):
        #if is_last_stage(rank, self.tensor_parallel_size, self.pipeline_parallel_size, self.data_parallel_size):
        pass

    # @abstractmethod
    def set_micro_batch_id(self, pp_group_id: int = 0) -> None:
        """为指定 PP 组内各 rank 的 trace 设置 micro-batch id，由子类的 set_self_microbatch_id/set_recv_send_microbatch_id 实现具体算法。"""
        pass
    
    # @abstractmethod
    def filter_comm_only_traces(self, pp_group_id=0):
        pass
    
    # @abstractmethod
    def get_p2p_ranks_pairs(self, ranks):
        # different pp schedule algorithms, the pairs are different
        pass

    def get_useful_trace_df(self, trace_df):
        return trace_df[~(trace_df[['send_prev', 'send_next', 'recv_prev', 'recv_next']] < 0).all(axis=1)]
    
    def get_p2p_trace_for_one_pair(self, rank_prev, rank_next):
        """
        Compute the point-to-point (P2P) communication times between two ranks.
        Args:
            rank_prev (int): The previous rank.
            rank_next (int): The next rank.

        Returns:
            pd.DataFrame: A DataFrame containing the bidirectional P2P communication times.
        """
        # Get the DataFrames for the given ranks
        trace_df_prev = self.full_dfs[rank_prev]
        trace_df_next = self.full_dfs[rank_next]
        useful_trace_df_prev = self.get_useful_trace_df(trace_df_prev)
        useful_trace_df_next = self.get_useful_trace_df(trace_df_next)
        # Compute forward P2P communication times
        df_p2p_forward = self._compute_p2p_forward(useful_trace_df_prev, useful_trace_df_next)
        # Link the dataframes of two adjacent pp i and pp i+1 by micro-bc id to get a forward-direction pair of send and recv
        # After linking, take min(send, recv) and save it to the comm_time column
        # Then copy the comm_time data to the new comm_time column of pp i
        # Similarly, copy the comm_time data to the new comm_time column of pp i+1
        self._update_comm_time(trace_df_prev, df_p2p_forward, 'index', 'index_on_prev')
        self._update_comm_time(trace_df_next, df_p2p_forward, 'index', 'index_on_next')
        # Compute backward P2P communication times
        df_p2p_backward = self._compute_p2p_backward(useful_trace_df_prev, useful_trace_df_next)
        self._update_comm_time(trace_df_prev, df_p2p_backward, 'index', 'index_on_prev')
        self._update_comm_time(trace_df_next, df_p2p_backward, 'index', 'index_on_next')
        # same as above. Calculate the pairwise send and recv funcs in the backward direction
        # and compute min(send, recv) into comm_time.
        # After completing the calculation of the actual comm_time for both forward and backward directions, 
        # wait_time = send/recv - comm_time 
        # Update the original DataFrames with the wait times
        self._update_wait_time(trace_df_prev)
        self._update_wait_time(trace_df_next)
        # Concatenate the forward and backward P2P DataFrames
        df_p2p_bidirection = pd.concat([df_p2p_forward, df_p2p_backward])

        return df_p2p_bidirection

    def _compute_p2p_forward(self, df_prev, df_next):
        """
        Compute the forward P2P communication times between two DataFrames.

        Args:
            df_prev (pd.DataFrame): The previous rank DataFrame.
            df_next (pd.DataFrame): The next rank DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the forward P2P communication times.
        """
        
        df_p2p_forward = pd.merge(df_prev, df_next, left_on='send_next', right_on='recv_prev', how='inner', suffixes=('_on_prev', '_on_next'))
        df_p2p_forward = df_p2p_forward[df_p2p_forward['send_next_on_prev'] >= 0]
        df_p2p_forward['p2p_forward'] = True
        df_p2p_forward['comm_time'] = np.minimum(df_p2p_forward['kernel_span_on_prev'], df_p2p_forward['kernel_span_on_next'])
        return df_p2p_forward

    def _compute_p2p_backward(self, df_prev, df_next):
        """
        Compute the backward P2P communication times between two DataFrames.

        Args:
            df_prev (pd.DataFrame): The previous rank DataFrame.
            df_next (pd.DataFrame): The next rank DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the backward P2P communication times.
        """
        df_p2p_backward = pd.merge(df_prev, df_next, left_on='recv_next', right_on='send_prev', how='inner', suffixes=('_on_prev', '_on_next'))
        df_p2p_backward = df_p2p_backward[df_p2p_backward['recv_next_on_prev'] >= 0]
        df_p2p_backward['p2p_backward'] = True
        df_p2p_backward['comm_time'] = np.minimum(df_p2p_backward['kernel_span_on_prev'], df_p2p_backward['kernel_span_on_next'])
        return df_p2p_backward

    def _update_comm_time(self, original_df, p2p_df, merge_col_original, merge_col_p2p):
        """
        Update the communication times in the original DataFrame based on the P2P DataFrame.

        Args:
            original_df (pd.DataFrame): The original DataFrame to update.
            p2p_df (pd.DataFrame): The P2P DataFrame containing the communication times.
            merge_col_original (str): The column name in the original DataFrame to merge on.
            merge_col_p2p (str): The column name in the P2P DataFrame to merge on.
        """
        # Ensure 'comm_time' column exists in the original DataFrame
        if 'comm_time' not in original_df.columns:
            original_df['comm_time'] = 0.0

        original_df.loc[original_df[merge_col_original].isin(p2p_df[merge_col_p2p]), 'comm_time'] = p2p_df['comm_time'].values

    def _update_wait_time(self, df):
        """
        Update the wait times in the DataFrame based on the communication times.

        Args:
            df (pd.DataFrame): The DataFrame to update.
        """
        df['wait_time'] = df['kernel_span'] - df['comm_time']

    
    def establish_p2p_link_on_adjacent_ranks(self, pp_group_id=0):
        ranks = self.all_pipeline_parallel_group_ranks[pp_group_id]
        rank_pairs = self.get_p2p_ranks_pairs(ranks)
        self.traces_p2p_comm = {}
        for rank_prev, rank_next in rank_pairs:
            df_p2p_bidirection = self.get_p2p_trace_for_one_pair(rank_prev, rank_next)
            #df_p2p_bidirection.to_csv(f'p2p_trace_rank_{rank_prev}_{rank_next}.csv', index=False)
            self.traces_p2p_comm[rank_prev] = df_p2p_bidirection
        
        self.trace_df_p2p_flow_events = self.combine_into_one_trace(self.traces_p2p_comm)
    
    def save_traces_with_p2p_comm(self, save_path, traces=None, trace_df_p2p_flow_events=None, meta_data=None):
        if traces is None:
            traces = self.traces
        if trace_df_p2p_flow_events is None:
            trace_df_p2p_flow_events = self.trace_df_p2p_flow_events
        #if meta_data is None:
        #    meta_data = self.meta_data
        trace_df_all_ranks = self.combine_into_one_trace(traces)
        save_trace_df_to_file(trace_df_all_ranks, save_path, trace_df_p2p_flow_events)
    
    # def set_rank_info(self):
    #     for rank in self.get_ranks():
    #         self.traces[rank]['rank'] = rank    


    # def save_traces(self, file_path, ranks=None):
    #     if ranks is None:
    #         effective_ranks = self.get_ranks()
    #     else:
    #         effective_ranks = set(ranks).intersection(set(self.get_ranks()))
    #     inputs = []
    #     for rank in effective_ranks:
    #         file_path_with_rank = add_rank_to_filename(file_path, rank)
    #         inputs.append([self.traces[rank], file_path_with_rank, None, self.meta_data[rank]])
    #     apply_function_for_parallel(save_trace_df_to_file, inputs)

    @staticmethod
    def combine_into_one_trace(traces_dict: dict):
        all_trace_dfs = []
        for rank, trace_df in traces_dict.items():
            trace_df['rank'] = rank
            all_trace_dfs.append(trace_df)
        trace_df = pd.concat(all_trace_dfs, ignore_index=True)
        return trace_df

    @staticmethod
    def display_traces_info(traces):
        first_trace_df = next(iter(traces.values()))
        logger.info(f'total {len(traces)} traces, and each trace has {len(first_trace_df)} items')
        logger.info(first_trace_df['s_cat'].value_counts())

