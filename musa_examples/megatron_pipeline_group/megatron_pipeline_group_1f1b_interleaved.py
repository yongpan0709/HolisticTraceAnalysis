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


class MegatronPipelineParallel1F1BInterleavedGroupTrace(MegatronPipelineParallelGroupTraceBase):
    """1F1B interleaved (One-Forward-One-Backward) 调度下的 PP group trace 分析。"""
    def __init__(self,
        trace_files: Optional[Dict[int, str]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
        dp = -1,
        tp = -1,
        pp = -1,
        ep = -1,
        cp: int =1, 
        order: str ="tp-cp-ep-dp-pp",
        pp_schedule: str = "1f1b",
        vpp_size = -1,
        ) -> None:
        super().__init__(trace_files, trace_dir, dp, tp, pp, ep, cp, order)
        self.pp_schedule = pp_schedule
        self.vpp_size = vpp_size


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
            'send_forward_recv_forward',
            'send_backward_recv_backward',
            # useing batch_isend_irecv
            'mccl:send', 
            'mccl:recv', 
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