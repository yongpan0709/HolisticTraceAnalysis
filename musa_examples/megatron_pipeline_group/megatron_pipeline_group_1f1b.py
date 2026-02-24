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
from .megatron_pipeline_group_base import MegatronPipelineParallelGroupTraceBase


class MegatronPipelineParallel1F1BGroupTrace(MegatronPipelineParallelGroupTraceBase):
    """1F1B (One-Forward-One-Backward) 调度下的 PP group trace 分析。"""
    def __init__(self,
        trace_files: Optional[Dict[int, str]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
        dp = -1,
        tp = -1,
        pp = -1,
        ep = -1,
        cp: int =1, 
        order: str ="tp-cp-ep-dp-pp",
        ) -> None:
        super().__init__(trace_files, trace_dir, dp, tp, pp, ep, cp, order)

    # Todo: func list
    @staticmethod
    def keep_comm_span_only(trace_df):
        comm_names_list = [
            'forward_step', 
            'backward_step', 
            'recv_forward', 
            'recv_backward', 
            'send_forward', 
            'send_backward', 
            'send_forward_recv_backward', 
            'send_backward_recv_forward', 
            # 'send_forward_recv_forward',
            # 'send_backward_recv_backward',
            # useing batch_isend_irecv
            # 'mccl:send', 
            # 'mccl:recv', 
            'finalize_model_grads',
            'step',
            'logical_and_across_model_parallel_group',
            'reduce_max_stat_across_model_parallel_group',
            'should_run_forward_backward',
            #'mccl:all_reduce'
        ]
        filter_comm = NameFilter(create_regex_for_prefix_match(comm_names_list))
        return filter_comm(trace_df)

    def filter_comm_only_traces(self, pp_group_id=0):
        for rank in self.all_pipeline_parallel_group_ranks[pp_group_id]:
            self.traces_comm_only[rank] = self.keep_comm_span_only(self.full_dfs[rank])

    def set_self_microbatch_id(self, trace_df: pd.DataFrame) -> None:
        trace_df.sort_values(by=['ts', 'dur'], ascending=[True, False], inplace=True)
        trace_df['micro_batch_id_forward'] = -1
        trace_df['micro_batch_id_backward'] = -1
        trace_df['recv_forward'] = trace_df['s_name'].str.contains('recv_forward').astype(int).cumsum() - 1
        trace_df['recv_backward'] = trace_df['s_name'].str.contains('recv_backward').astype(int).cumsum() - 1
        trace_df.loc[trace_df['s_name'].str.contains('forward'), 'micro_batch_id_forward'] = trace_df['recv_forward']
        trace_df.loc[trace_df['s_name'].str.contains('backward'), 'micro_batch_id_backward'] = trace_df['recv_backward']
        trace_df.drop(['recv_forward', 'recv_backward'], axis=1, inplace=True)

    def set_recv_send_microbatch_id(self, trace_df: pd.DataFrame) -> None:
        trace_df['send_prev'] = -1
        trace_df['send_next'] = -1
        trace_df['recv_prev'] = -1
        trace_df['recv_next'] = -1
        trace_df.loc[trace_df['s_name'].str.contains('recv_forward', regex=False), 'recv_prev'] = trace_df['micro_batch_id_forward']
        trace_df.loc[trace_df['s_name'].str.contains('recv_backward', regex=False), 'recv_next'] = trace_df['micro_batch_id_backward']
        trace_df.loc[trace_df['s_name'].str.contains('send_forward', regex=False), 'send_next'] = trace_df['micro_batch_id_forward']
        trace_df.loc[trace_df['s_name'].str.contains('send_backward', regex=False), 'send_prev'] = trace_df['micro_batch_id_backward']

    # Use: func annotations
    def process_pipeline_start(self, df):
        #if is_first_stage(rank, self.tensor_parallel_size, self.pipeline_parallel_size, self.data_parallel_size):
        df.loc[df['s_name'].str.contains('recv_forward'), 'recv_prev'] = -1
        df.loc[df['s_name'].str.contains('send_backward'), 'send_prev'] = -1

    def process_pipeline_end(self, df):
        #if is_last_stage(rank, self.tensor_parallel_size, self.pipeline_parallel_size, self.data_parallel_size):
        df.loc[df['s_name'].str.contains('recv_backward'), 'recv_next'] = -1
        df.loc[df['s_name'].str.contains('send_forward'), 'send_next'] = -1

    def set_micro_batch_id(self, pp_group_id: int = 0) -> None:
        """为指定 PP 组内各 rank 的 trace 设置 micro-batch id，由子类的 set_self_microbatch_id/set_recv_send_microbatch_id 实现具体算法。"""
        ranks = self.all_pipeline_parallel_group_ranks[pp_group_id]
        logger.info(f'In set micro batch id: ranks: {ranks}')
        for rank in ranks:
            self.set_self_microbatch_id(self.full_dfs[rank])
            self.set_recv_send_microbatch_id(self.full_dfs[rank])
        self.process_pipeline_start(self.full_dfs[ranks[0]])
        self.process_pipeline_end(self.full_dfs[ranks[-1]])

    def get_p2p_ranks_pairs(self, ranks):
        if len(ranks) < 2: return []
        # Todo: double check
        ranks_sorted = sorted(ranks)
        p2p_devices_pairs = []
        for i in range(len(ranks) - 1):
            p2p_devices_pairs.append([ranks[i], ranks[i+1]])
        return p2p_devices_pairs 

    def preprocess_trace_df(self, rank):
        trace_df = self.full_dfs[rank]
        sorted_trace_df = trace_df.sort_values(by=['ts', 'kernel_span'], ascending=[True, False])
        # Use regex to find the first occurrence of any 'recv_forward*' event
        recv_forward_index = sorted_trace_df[sorted_trace_df['s_name'].str.contains(r'^recv_forward.*')].index
        first_recv_forward_index = recv_forward_index[0]
        # Keep only the rows from the first 'recv_forward*' event onwards
        sorted_trace_df = sorted_trace_df.loc[first_recv_forward_index:]

        # sorted_trace_df['end'] = sorted_trace_df['ts'] + sorted_trace_df['dur']
        # sorted_trace_df.to_csv('sorted_trace_df-after-recv.csv')
        return sorted_trace_df

    def calculate_time_per_iteration(self, rank):
        trace_df = self.full_dfs[rank]
        return trace_df[trace_df['s_name'].str.contains(r'^ProfilerStep#.*')]['kernel_span'].values[0]

    def get_all_comm_df(self, sorted_trace_df):
        return NameFilter(create_regex_for_prefix_match(['send_forward',  'recv_forward', 'send_backward',  'recv_backward']))(sorted_trace_df)

    def calculate_comm_time_total(self, all_comm_time_df):
        return all_comm_time_df['kernel_span'].sum()
    
    def calculate_theoretical_bubble_time_warmup(self, all_comm_time_df, stage_id):
        if stage_id == 0:
            return 0.0
        else:
            theoretical_bubble_head_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^recv_forward(?:_\d+)?$')].index[0]
            return all_comm_time_df.loc[theoretical_bubble_head_index, 'wait_time']

    def calculate_bubble_time_warmup(self, all_comm_time_df):
        recv_index_in_head = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^recv_forward(?:_\d+)?$')].index
        send_index_in_head = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_forward(?:_\d+)?$')].index
        return all_comm_time_df.loc[recv_index_in_head, 'wait_time'].sum() + all_comm_time_df.loc[send_index_in_head, 'wait_time'].sum()

    def calculate_theoretical_bubble_time_steady(self, all_comm_time_df, stage_id):
        if stage_id == self.pipeline_parallel_size-1:
            return 0.0
        else:
            send_forward_recv_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_forward_recv_backward.*')].index
            if len(send_forward_recv_backward_index) > 0:
                return all_comm_time_df.loc[send_forward_recv_backward_index[0], 'wait_time']
            else:
                return 0.0

    def calculate_bubble_time_steady(self, all_comm_time_df):
        send_forward_recv_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_forward_recv_backward.*')].index
        send_backward_recv_forward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_backward_recv_forward.*')].index
        return all_comm_time_df.loc[send_forward_recv_backward_index, 'wait_time'].sum() + all_comm_time_df.loc[send_backward_recv_forward_index, 'wait_time'].sum()
    
    def calculate_theoretical_bubble_time_cooldown(self, all_comm_time_df, stage_id):
        if stage_id == self.pipeline_parallel_size-1:
            return 0.0
        else:
            recv_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^recv_backward(?:_\d+)?$')].index
            return all_comm_time_df.loc[recv_backward_index, 'wait_time'].sum()

    def calculate_bubble_time_cooldown(self, all_comm_time_df):
        send_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_backward(?:_\d+)?$')].index
        recv_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^recv_backward(?:_\d+)?$')].index
        bubble_time_cooldown = all_comm_time_df.loc[send_backward_index, 'wait_time'].sum() + all_comm_time_df.loc[recv_backward_index, 'wait_time'].sum()
        return bubble_time_cooldown

    def calculate_finalize_model_grads_step_time(self, sorted_trace_df):
        pattern = r'^finalize_model_grads$'
        finalize_model_grads_step_time = sorted_trace_df[sorted_trace_df['s_name'].str.contains(pattern)]['kernel_span'].values[0]
        return finalize_model_grads_step_time
    
    # Todo: func name
    #       sorted_trace_df[sorted_trace_df['full_name'].str.contains(pattern, regex=True)]['dur'].values[0]   why use the first value
    def calculate_optimizer_step_time_and_bubble(self, sorted_trace_df):
        # Use regex to match 'full_name' with the pattern '?ProfilerStep?/step?'
        # Todo: func name
        # pattern = r'.*ProfilerStep.*/step.*'
        pattern = r'^step$'
        optimizer_step_time = sorted_trace_df[sorted_trace_df['s_name'].str.contains(pattern)]['kernel_span'].values[0]
        #if first_stage_optimizer_step_time is None:
        #    first_stage_optimizer_step_time = optimizer_step_time
        #bubble_time_final = optimizer_step_time - first_stage_optimizer_step_time
        #return bubble_time_final, first_stage_optimizer_step_time, optimizer_step_time
        return optimizer_step_time
    
    def calculate_logical_and_across_model_parallel_group_time(self, sorted_trace_df):
        pattern = r'^logical_and_across_model_parallel_group$'
        logical_and_across_model_parallel_group_time = sorted_trace_df[sorted_trace_df['s_name'].str.contains(pattern)]['kernel_span'].values[0]
        return logical_and_across_model_parallel_group_time

    def calculate_true_comm_and_overhead_wait_time(self, all_comm_time_df):
        comm_time_true = all_comm_time_df['comm_time'].sum()
        overhead_wait_time_total = all_comm_time_df['wait_time'].sum()
        return comm_time_true, overhead_wait_time_total

    # Todo: 'reduce_model_grads', 'step_', 'gather_model_params' 
    def calculate_optimizer_time(self, sorted_trace_df, bubble_time_final, stage_id, rank):
        optimizer_df = NameFilter(create_regex_for_prefix_match(['finalize_model_grads', 
                                                                 'step', 
                                                                 'logical_and_across_model_parallel_group',
                                                                 'reduce_max_stat_across_model_parallel_group']))(sorted_trace_df)
        #optimizer_df.to_csv(f'optimizer_df-stageid{stage_id}-rank{rank}.csv')
        optimizer_time = optimizer_df['kernel_span'].sum()
        return optimizer_time - bubble_time_final
    
