# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from enum import auto, Flag
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import copy

import pandas as pd

from hta.configs.config import logger
from hta.configs.default_values import DEFAULT_TRACE_DIR
from hta.configs.parser_config import ParserConfig
from hta.common.trace_call_graph import CallGraph
from hta.common.trace_filter import NameFilter, create_regex_for_prefix_match
from hta.utils.utils import prepare_directory
from hta.trace_analysis import TraceAnalysis
from .megatron_pipeline_group_1f1b_interleaved_epoverlap import MegatronPipelineParallel1F1BInterleavedEPOverlapGroupTrace
from .megatron_pipeline_group_1f1b import MegatronPipelineParallel1F1BGroupTrace
from .megatron_pipeline_group_1f1b_interleaved import MegatronPipelineParallel1F1BInterleavedGroupTrace


class MegatronPipelineParallelGroupTraceAnalysis(TraceAnalysis):
    def __init__(
        self,
        trace_files: Optional[Dict[int, str]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
        include_last_profiler_step: Optional[bool] = False,
        data_parallel_size = -1,
        tensor_parallel_size = -1,
        pipeline_parallel_size = -1,
        expert_model_parallel_size = -1,
        context_parallel_size = -1,
        pp_schedule: str= '1f1b',
        vpp_size = -1,
        micro_bs = 0,
    ):
        cfg = ParserConfig.get_default_cfg()
        #cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
        ParserConfig.set_default_cfg(cfg)
        self.pp_schedule = pp_schedule
        self.vpp_size = vpp_size
        self.micro_bs = micro_bs
        if self.pp_schedule == '1f1b':
            self.t = MegatronPipelineParallel1F1BGroupTrace(trace_files, trace_dir, dp=data_parallel_size, tp=tensor_parallel_size, pp=pipeline_parallel_size, ep=expert_model_parallel_size, cp=context_parallel_size, micro_bs = self.micro_bs)
        elif self.pp_schedule == '1f1b-interleaved':
            self.t = MegatronPipelineParallel1F1BInterleavedGroupTrace(trace_files, trace_dir, dp=data_parallel_size, tp=tensor_parallel_size, pp=pipeline_parallel_size, ep=expert_model_parallel_size, cp=context_parallel_size, vpp_size = self.vpp_size, micro_bs = self.micro_bs)
        elif self.pp_schedule == '1f1b-interleaved-epoverlap':
            self.t = MegatronPipelineParallel1F1BInterleavedEPOverlapGroupTrace(trace_files, trace_dir, dp=data_parallel_size, tp=tensor_parallel_size, pp=pipeline_parallel_size, ep=expert_model_parallel_size, cp=context_parallel_size, vpp_size = self.vpp_size, micro_bs = self.micro_bs)

        self.output_dir = os.path.join(trace_dir, 'output')
        # Todo: set force clear to true
        prepare_directory(self.output_dir, force_clear=False)
        # self.t.save_traces(f'{self.output_dir}/init.json')
    
    def analyze_pipeline_parallel_per_group(self, pp_group_id):
        # Since setting PROFILER_WITH_STACK=0 when profiling,
        # only annotated funcs, aten kernels and GPU kernels kept in trace.json
        #self.t.display_traces_info(self.t.traces)
        logger.info('construct CallGraph for traces')
        self.t.parse_traces_per_pp_group(pp_group_id=pp_group_id)
        
        #logger.info('rename_children_with_duplicate_names')
        #call_graph.rename_children_with_duplicate_names()
        # call_graph.print_call_graph(f'{self.output_dir}/rename_children_with_duplicate_names.txt')
        
        logger.info('keep comm spans only')
        self.t.filter_comm_only_traces(pp_group_id=pp_group_id)
        logger.info('set_micro_batch_id')
        self.t.set_micro_batch_id(pp_group_id=pp_group_id) 
        # self.t.save_traces(f'{self.output_dir}/set_micro_batch_id.json')
        
        logger.info('establish_p2p_link_on_adjacent_ranks')
        self.t.establish_p2p_link_on_adjacent_ranks(pp_group_id=pp_group_id) 
        
        # logger.info('construct call graphs after calculating bandwidth and flops')
        # call_graph = CallGraph(self.t)
        # logger.info('print final call graph')
        # call_graph.print_call_graph(f'{self.output_dir}/full_names.txt')
        # for rank, df in self.t.traces.items():
        #     df.to_csv(f'{self.output_dir}/after-cal-{rank}_full-df.csv')

        logger.info('save_traces_with_p2p_comm')
        self.t.save_traces_with_p2p_comm(f'{self.output_dir}/../../pp{pp_group_id}-trace.json', traces=self.t.traces_comm_only)
        logger.info('generate_report')
        self.t.generate_report(pp_group_id, f'{self.output_dir}/../../report-pp{pp_group_id}.csv')
    