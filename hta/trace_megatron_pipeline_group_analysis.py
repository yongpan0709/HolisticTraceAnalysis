# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from enum import auto, Flag
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import copy

import pandas as pd

from hta.megatron_pp_group_trace import MegatronPipelineParrallelGroupTrace
from hta.configs.config import logger
from hta.configs.default_values import DEFAULT_TRACE_DIR
from hta.configs.parser_config import ParserConfig
from hta.common.trace_call_graph import CallGraph
from hta.common.trace_filter import NameFilter, create_regex_for_prefix_match
from hta.utils.utils import prepare_directory
from hta.trace_analysis import TraceAnalysis


class MegatronPipelineParallelGroupTraceAnalysis(TraceAnalysis):
    def __init__(
        self,
        trace_files: Optional[Dict[int, str]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
        include_last_profiler_step: Optional[bool] = False,
        data_parallel_size = -1,
        tensor_parallel_size = -1,
        pipeline_parallel_size = -1,
    ):
        cfg = ParserConfig.get_default_cfg()
        #cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
        ParserConfig.set_default_cfg(cfg)
        self.t = MegatronPipelineParrallelGroupTrace(trace_files, trace_dir, data_parallel_size, tensor_parallel_size, pipeline_parallel_size)
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
        
        # logger.info('calculate_comm_volume_for_trace_df')
        # self.t.traces = self.t.parallel_apply(calculate_comm_volume_for_trace_df)
        # logger.info('calculate_flops_for_trace_df')
        # self.t.traces = self.t.parallel_apply(calculate_flops_for_trace_df)
        
        # logger.info('construct call graphs after calculating bandwidth and flops')
        # call_graph = CallGraph(self.t)
        # logger.info('print final call graph')
        # call_graph.print_call_graph(f'{self.output_dir}/full_names.txt')
        # for rank, df in self.t.traces.items():
        #     df.to_csv(f'{self.output_dir}/after-cal-{rank}_full-df.csv')

        logger.info('save_traces_with_p2p_comm')
        self.t.save_traces_with_p2p_comm(f'{self.output_dir}/trace_only_comm_all_ranks_with_flow.json', traces=self.t.traces_comm_only)
        logger.info('generate_report')
        self.generate_report(f'{self.output_dir}/report.csv')
    
    def generate_report(self, save_path):
        output_df = None
        #first_stage_optimizer_step = None

        for stage_id, rank in enumerate(sorted(self.t.traces_comm_only.keys())):
            trace_df = self.t.full_dfs[rank]
            sorted_trace_df = self._preprocess_trace_df(trace_df)

            time_per_iteration = self._calculate_time_per_iteration(trace_df)
            all_forward_steps_df = NameFilter(create_regex_for_prefix_match(['forward_step']))(sorted_trace_df)
            all_backward_steps_df = NameFilter(create_regex_for_prefix_match(['backward_step']))(sorted_trace_df)
            forward_step_avg_time, backward_step_avg_time, compute_time_total = self._calculate_step_times(all_forward_steps_df, all_backward_steps_df)

            all_comm_time_df = NameFilter(create_regex_for_prefix_match(['send_forward',  'recv_forward',  'send_forward_recv_backward',  'send_backward_recv_forward',  'send_backward',  'recv_backward']))(sorted_trace_df)
            #all_comm_time_df.to_csv(f'all_comm_time_df-{rank}-stageid-{stage_id}.csv')
            comm_time_total = self._calculate_comm_time_total(all_comm_time_df)

            bubble_time_head = self._calculate_bubble_time_head(all_comm_time_df)
            bubble_time_middle = self._calculate_bubble_time_middle(all_comm_time_df)
            bubble_time_tail = self._calculate_bubble_time_tail(all_comm_time_df)
            finalize_model_grads_step_time = self._calculate_finalize_model_grads_step_time(sorted_trace_df)
            optimizer_time = self._calculate_optimizer_step_time_and_bubble(sorted_trace_df)
            logical_and_across_model_parallel_group_time = self._calculate_logical_and_across_model_parallel_group_time(sorted_trace_df)
            #bubble_time_total = bubble_time_head + bubble_time_middle + bubble_time_tail # + bubble_time_final
            #comm_time_total += bubble_time_final
            comm_time_true, overhead_wait_time_total = self._calculate_true_comm_and_overhead_wait_time(all_comm_time_df)
            
            #optimizer_time = self._calculate_optimizer_time(sorted_trace_df, bubble_time_final, stage_id, rank)
            
            num_microbatch = len(all_forward_steps_df)  # Assuming the number of send steps represents the number of microbatches
            assert(len(all_backward_steps_df) == num_microbatch)

            args = {
                'rank': rank,
                'time_per_iteration': time_per_iteration,
                'num_microbatch': num_microbatch,
                'forward_step_avg_time': forward_step_avg_time,
                'backward_step_avg_time': backward_step_avg_time,
                'compute_time_total': compute_time_total,
                'comm_time_total': comm_time_total,
                'comm_time_true': comm_time_true,
                'overhead_wait_time_total': overhead_wait_time_total,
                #'bubble_time_total': bubble_time_total,
                'bubble_time_head': bubble_time_head,
                'bubble_time_middle': bubble_time_middle,
                'bubble_time_tail': bubble_time_tail,
                #'bubble_time_final': bubble_time_final,
                'pipeline_parallel_size': self.t.pipeline_parallel_size,
                'finalize_model_grads_step_time': finalize_model_grads_step_time,
                'logical_and_across_model_parallel_group_time': logical_and_across_model_parallel_group_time,
                'optimizer_time_total': optimizer_time
            }

            info_per_rank = self._generate_info_per_rank(args)

            if output_df is None:
                output_df = pd.DataFrame([info_per_rank])
            else:
                output_df.loc[len(output_df)] = info_per_rank

        if save_path is not None:
            output_df.to_csv(save_path, header=True, index=False, float_format='%.3f')
        #print(output_df)
        return output_df

    # Todo: 'reduce_model_grads', 'step_', 'gather_model_params' 
    def _calculate_optimizer_time(self, sorted_trace_df, bubble_time_final, stage_id, rank):
        optimizer_df = NameFilter(create_regex_for_prefix_match(['finalize_model_grads', 
                                                                 'step', 
                                                                 'logical_and_across_model_parallel_group',
                                                                 'reduce_max_stat_across_model_parallel_group']))(sorted_trace_df)
        #optimizer_df.to_csv(f'optimizer_df-stageid{stage_id}-rank{rank}.csv')
        optimizer_time = optimizer_df['kernel_span'].sum()
        return optimizer_time - bubble_time_final
    
    def _preprocess_trace_df(self, trace_df):
        sorted_trace_df = trace_df.sort_values(by=['ts', 'kernel_span'], ascending=[True, False])
        # Use regex to find the first occurrence of any 'recv_forward*' event
        recv_forward_index = sorted_trace_df[sorted_trace_df['s_name'].str.contains(r'^recv_forward.*')].index
        first_recv_forward_index = recv_forward_index[0]
        # Keep only the rows from the first 'recv_forward*' event onwards
        sorted_trace_df = sorted_trace_df.loc[first_recv_forward_index:]

        # sorted_trace_df['end'] = sorted_trace_df['ts'] + sorted_trace_df['dur']
        # sorted_trace_df.to_csv('sorted_trace_df-after-recv.csv')
        return sorted_trace_df
    
    def _calculate_time_per_iteration(self, trace_df):
        return trace_df[trace_df['s_name'].str.contains(r'^ProfilerStep#.*')]['kernel_span'].values[0]

    def _calculate_step_times(self, all_forward_steps_df, all_backward_steps_df):
        forward_step_avg_time = all_forward_steps_df['kernel_span'].mean()
        backward_step_avg_time = all_backward_steps_df['kernel_span'].mean()
        compute_time_total = all_forward_steps_df['kernel_span'].sum() + all_backward_steps_df['kernel_span'].sum()
        return forward_step_avg_time, backward_step_avg_time, compute_time_total

    def _calculate_comm_time_total(self, all_comm_time_df):
        return all_comm_time_df['kernel_span'].sum()

    def _calculate_bubble_time_head(self, all_comm_time_df):
        recv_index_in_head = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^recv_forward(?:_\d+)?$')].index
        send_index_in_head = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_forward(?:_\d+)?$')].index
        return all_comm_time_df.loc[recv_index_in_head, 'wait_time'].sum() + all_comm_time_df.loc[send_index_in_head, 'wait_time'].sum()

    def _calculate_bubble_time_middle(self, all_comm_time_df):
        send_forward_recv_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_forward_recv_backward.*')].index
        send_backward_recv_forward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_backward_recv_forward.*')].index
        return all_comm_time_df.loc[send_forward_recv_backward_index, 'wait_time'].sum() + all_comm_time_df.loc[send_backward_recv_forward_index, 'wait_time'].sum()

    def _calculate_bubble_time_tail(self, all_comm_time_df):
        send_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^send_backward(?:_\d+)?$')].index
        recv_backward_index = all_comm_time_df[all_comm_time_df['s_name'].str.contains(r'^recv_backward(?:_\d+)?$')].index
        bubble_time_tail = all_comm_time_df.loc[send_backward_index, 'wait_time'].sum() + all_comm_time_df.loc[recv_backward_index, 'wait_time'].sum()
        return bubble_time_tail

    def _calculate_finalize_model_grads_step_time(self, sorted_trace_df):
        pattern = r'^finalize_model_grads$'
        finalize_model_grads_step_time = sorted_trace_df[sorted_trace_df['s_name'].str.contains(pattern)]['kernel_span'].values[0]
        return finalize_model_grads_step_time
    
    def _calculate_logical_and_across_model_parallel_group_time(self, sorted_trace_df):
        pattern = r'^logical_and_across_model_parallel_group$'
        logical_and_across_model_parallel_group_time = sorted_trace_df[sorted_trace_df['s_name'].str.contains(pattern)]['kernel_span'].values[0]
        return logical_and_across_model_parallel_group_time
    
    # Todo: func name
    #       sorted_trace_df[sorted_trace_df['full_name'].str.contains(pattern, regex=True)]['dur'].values[0]   why use the first value
    def _calculate_optimizer_step_time_and_bubble(self, sorted_trace_df):
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

    def _calculate_true_comm_and_overhead_wait_time(self, all_comm_time_df):
        comm_time_true = all_comm_time_df['comm_time'].sum()
        overhead_wait_time_total = all_comm_time_df['wait_time'].sum()
        return comm_time_true, overhead_wait_time_total

    def _generate_info_per_rank(self, args):
        return {
            'rank': args['rank'],
            'time_per_iteration': args['time_per_iteration'] / 1000,
            'microbatch_num': args['num_microbatch'],
            'forward_step_time': args['forward_step_avg_time'] / 1000,
            'backward_step_time': args['backward_step_avg_time'] / 1000,
            'time_per_microbatch': (args['forward_step_avg_time'] + args['backward_step_avg_time']) / 1000,
            'compute_time_total': args['compute_time_total'] / 1000,
            'comm_time_total': args['comm_time_total'] / 1000,
            'optimizer_time_total': args['optimizer_time_total'] / 1000,
            'comm_time_true': args['comm_time_true'] / 1000,
            'overhead_wait_time_total': args['overhead_wait_time_total'] / 1000,
            #'bubble_time_total': args['bubble_time_total'] / 1000,
            'bubble_time_detail': [args['bubble_time_head'] / 1000, args['bubble_time_middle'] / 1000, args['bubble_time_tail'] / 1000],
            'finalize_model_grads_step_time': args['finalize_model_grads_step_time'] / 1000,
            'logical_and_across_model_parallel_group_time': args['logical_and_across_model_parallel_group_time'] / 1000,
            'comp_time_ratio': args['compute_time_total'] / args['time_per_iteration'],
            'comm_time_ratio': args['comm_time_total'] / args['time_per_iteration'],
            'comm_time_true_ratio': args['comm_time_true'] / args['time_per_iteration'],
            'overhead_wait_time_ratio': args['overhead_wait_time_total'] / args['time_per_iteration'],
            #'bubble_time_ratio': args['bubble_time_total'] / args['time_per_iteration'],
            'bubble_time_ratio_theoretical': (args['pipeline_parallel_size'] - 1) / (args['pipeline_parallel_size'] - 1 + args['num_microbatch'])
        }
    