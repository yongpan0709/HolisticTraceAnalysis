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
    ):
        cfg = ParserConfig.get_default_cfg()
        #cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
        ParserConfig.set_default_cfg(cfg)
        self.pp_schedule = pp_schedule
        self.vpp_size = vpp_size
        if self.pp_schedule == '1f1b':
            self.t = MegatronPipelineParallel1F1BGroupTrace(trace_files, trace_dir, dp=data_parallel_size, tp=tensor_parallel_size, pp=pipeline_parallel_size, ep=expert_model_parallel_size, cp=context_parallel_size)
        elif self.pp_schedule == '1f1b-interleaved':
            self.t = MegatronPipelineParallel1F1BInterleavedGroupTrace(trace_files, trace_dir, dp=data_parallel_size, tp=tensor_parallel_size, pp=pipeline_parallel_size, ep=expert_model_parallel_size, cp=context_parallel_size, pp_schedule = self.pp_schedule, vpp_size = self.vpp_size)

        self.output_dir = os.path.join(trace_dir, 'output')
        # Todo: set force clear to true
        prepare_directory(self.output_dir, force_clear=False)
        # self.t.save_traces(f'{self.output_dir}/init.json')
    
    def analyze_pipeline_parallel_per_group(self, pp_group_id, pp_schedule='1f1b'):
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
        self.generate_report(f'{self.output_dir}/../../report-pp{pp_group_id}.csv')
    
    def generate_report(self, save_path):
        output_df = None
        #first_stage_optimizer_step = None

        for stage_id, rank in enumerate(sorted(self.t.traces_comm_only.keys())):
            sorted_trace_df = self.t.preprocess_trace_df(rank)
            time_per_iteration = self.t.calculate_time_per_iteration(rank)
            all_forward_steps_df = NameFilter(create_regex_for_prefix_match(['forward_step']))(sorted_trace_df)
            all_backward_steps_df = NameFilter(create_regex_for_prefix_match(['backward_step']))(sorted_trace_df)
            #all_forward_steps_df.to_csv(f'all_forward_steps_df-{rank}-stageid-{stage_id}.csv')
            #all_backward_steps_df.to_csv(f'all_backward_steps_df-{rank}-stageid-{stage_id}.csv')
            forward_step_avg_time, backward_step_avg_time, compute_time_total, fwd_std, bwd_std = self.t.calculate_step_times(all_forward_steps_df, all_backward_steps_df)

            # 'send_forward_recv_backward',  'send_backward_recv_forward',
            all_comm_time_df = self.t.get_all_comm_df(sorted_trace_df, rank)
            #all_comm_time_df.to_csv(f'all_comm_time_df-{rank}-stageid-{stage_id}.csv')
            comm_time_total = self.t.calculate_comm_time_total(all_comm_time_df)

            theoretical_bubble_time_warmup = self.t.calculate_theoretical_bubble_time_warmup(all_comm_time_df, stage_id)
            bubble_time_warmup = self.t.calculate_bubble_time_warmup(all_comm_time_df, stage_id) - theoretical_bubble_time_warmup
            theoretical_bubble_time_steady = self.t.calculate_theoretical_bubble_time_steady(all_comm_time_df, stage_id)
            bubble_time_steady = self.t.calculate_bubble_time_steady(all_comm_time_df, stage_id) - theoretical_bubble_time_steady
            theoretical_bubble_time_cooldown = self.t.calculate_theoretical_bubble_time_cooldown(all_comm_time_df, stage_id)
            bubble_time_cooldown = self.t.calculate_bubble_time_cooldown(all_comm_time_df, stage_id) - theoretical_bubble_time_cooldown
            finalize_model_grads_step_time = self.t.calculate_finalize_model_grads_step_time(sorted_trace_df)
            optimizer_time = self.t.calculate_optimizer_step_time_and_bubble(sorted_trace_df)
            logical_and_across_model_parallel_group_time = self.t.calculate_logical_and_across_model_parallel_group_time(sorted_trace_df)
            #bubble_time_total = bubble_time_warmup + bubble_time_steady + bubble_time_cooldown # + bubble_time_final
            #comm_time_total += bubble_time_final
            comm_time_true, overhead_wait_time_total = self.t.calculate_true_comm_and_overhead_wait_time(all_comm_time_df)
            
            #optimizer_time = self.t.calculate_optimizer_time(sorted_trace_df, bubble_time_final, stage_id, rank)
            
            num_microbatch = self.t.get_num_microbatches(sorted_trace_df) 

            info_per_rank = {
                'rank': rank,
                'time_per_iteration': time_per_iteration,
                'num_microbatch': num_microbatch,
                'forward_step_avg_time': forward_step_avg_time,
                'fwd_step_std_time': fwd_std,
                'backward_step_avg_time': backward_step_avg_time,
                'bwd_step_std_time': bwd_std,
                'compute_time_total': compute_time_total,
                'comm_time_total': comm_time_total,
                'comm_time_true': comm_time_true,
                'overhead_wait_time_total': overhead_wait_time_total,
                #'bubble_time_total': bubble_time_total,
                'bubble_time_warmup': bubble_time_warmup,
                'bubble_time_steady': bubble_time_steady,
                'bubble_time_cooldown': bubble_time_cooldown,
                'theoretical_bubble_time_warmup': theoretical_bubble_time_warmup,
                'theoretical_bubble_time_steady': theoretical_bubble_time_steady,
                'theoretical_bubble_time_cooldown': theoretical_bubble_time_cooldown,
                #'bubble_time_final': bubble_time_final,
                #'bubble_time_detail': [args['bubble_time_warmup'] / 1000, args['bubble_time_steady'] / 1000, args['bubble_time_cooldown'] / 1000],
                'overhead_wait_time_ratio': overhead_wait_time_total / time_per_iteration,
                'bubble_time_ratio': overhead_wait_time_total/ (compute_time_total + comm_time_total),
                'bubble_time_ratio_theoretical': self.t.get_bubble_time_ratio_theoretical(num_microbatch),
                'pipeline_parallel_size': self.t.pipeline_parallel_size,
                'comm_time_true_ratio': comm_time_true / time_per_iteration,
                'comp_time_ratio': compute_time_total / time_per_iteration,
                'comm_time_ratio': comm_time_total / time_per_iteration,
                'finalize_model_grads_step_time': finalize_model_grads_step_time,
                'logical_and_across_model_parallel_group_time': logical_and_across_model_parallel_group_time,
                'optimizer_time_total': optimizer_time,
            }
            #info_per_rank = self._generate_info_per_rank(args)

            if output_df is None:
                output_df = pd.DataFrame([info_per_rank])
            else:
                output_df.loc[len(output_df)] = info_per_rank

        if save_path is not None:
            output_df.to_csv(save_path, header=True, index=False, float_format='%.3f')
        #print(output_df)
        return output_df
