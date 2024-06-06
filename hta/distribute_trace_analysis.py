from hta.trace_analysis import MegatronPipelineParallelGroupTraceAnalysis
from hta.utils.parallel_state import get_3d_parallel_groups
from hta.utils.utils import partition_files_across_directories, prepare_directory
from hta.configs.config import logger

from mpi4py import MPI
import os
import pickle
import logging
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import time

class DistributedMegatronTraceAnalysis:
    def __init__(self, trace_dir, tensor_parallel_size, data_parallel_size, pipeline_parallel_size):
        self.trace_dir = trace_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        self.processor_name = MPI.Get_processor_name()
        
        self.setup_dirs()
        self.setup_logging()
        self.init_pp_group_sub_dirs()
        self.assign_analysis_tasks()
    
    def post_process(self):
        if self.rank == 0:
            self.move_output_dir()
    
    def move_output_dir(self):
        src_dir = self.trace_dir_pp_group
        dest_dir = self.output_dir

        # List to store the full paths of found 'output' directories
        output_dirs = []

        # Recursively search for all 'output' directories
        for root, dirs, files in os.walk(src_dir):
            for dir_name in dirs:
                if dir_name == 'output':
                    # Construct the full path to the 'output' directory
                    output_dir_path = os.path.join(root, dir_name)
                    # Add the 'output' directory path to the list
                    output_dirs.append(output_dir_path)

        
        # Iterate over the found 'output' directories and perform the move operation
        for output_dir in output_dirs:
            # Get the name of the parent directory of 'output', which should be like p1, p2, p3, etc.
            parent_dir_name = os.path.basename(os.path.dirname(output_dir))
            
            # Construct the destination path based on the parent directory name
            dest_subpath = os.path.join(dest_dir, parent_dir_name)
            
            # Ensure the destination subpath exists, create it if it doesn't
            os.makedirs(dest_subpath, exist_ok=True)
            
            # Move the 'output' directory to the destination path
            shutil.move(output_dir, dest_subpath)
            logger.info(f"Moved: {output_dir} -> {dest_subpath}")

        logger.info("All 'output' directories have been moved.")
    
    def setup_dirs(self):
        self.trace_dir_pp_group = 'trace'
        self.output_dir = 'output'
        self.stragglers_dir = os.path.join(self.output_dir, 'stragglers')
        self.log_dir = 'log'
        if self.rank == 0:
            prepare_directory(self.trace_dir_pp_group, force_clear=False)
            prepare_directory(self.log_dir, force_clear=True)
            prepare_directory(self.output_dir, force_clear=True)
            prepare_directory(self.stragglers_dir, force_clear=True)
        self.comm.Barrier()
        time.sleep(3)
    
    def setup_logging(self):
        log_filename = os.path.join(self.log_dir, f'log_rank_{self.rank}.log')
        logging.basicConfig(
            filename=log_filename,
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logger
        self.logger.info(f'Process {self.rank} on {self.processor_name} started logging.')

    def init_pp_group_sub_dirs(self):
        self.logger.info('Initializing pipeline parallel group subdirectories.')
        self.all_data_parallel_group_ranks, \
        self.all_tensor_parallel_group_ranks, \
        self.all_pipeline_parallel_group_ranks = get_3d_parallel_groups(
            self.tensor_parallel_size, 
            self.pipeline_parallel_size, 
            self.data_parallel_size
        )
        
        trace_dir_for_pp_group = os.path.join(self.trace_dir_pp_group, 'pp_group')
        self.all_pp_group_sub_dirs = [f'{trace_dir_for_pp_group}_{i}' for i in range(len(self.all_pipeline_parallel_group_ranks))]
        
        if self.rank == 0:
            partition_files_across_directories(
                self.trace_dir, 
                trace_dir_for_pp_group, 
                self.all_pipeline_parallel_group_ranks, 
            )

        self.comm.Barrier()
        time.sleep(6)
        self.logger.info('Initialization of pipeline parallel group subdirectories completed.')
    
    def assign_analysis_tasks(self):
        self.logger.info('Assigning analysis tasks.')
        num_folders = len(self.all_pp_group_sub_dirs)
        num_folders_per_process = num_folders // self.world_size
        remainder = num_folders % self.world_size
        
        if self.rank < remainder:
            start_index = self.rank * (num_folders_per_process + 1)
            end_index = start_index + num_folders_per_process + 1
        else:
            start_index = remainder * (num_folders_per_process + 1) + (self.rank - remainder) * num_folders_per_process
            end_index = start_index + num_folders_per_process
        
        self.assigned_folders = self.all_pp_group_sub_dirs[start_index:end_index]
        self.logger.info(f'Assigned folders: {self.assigned_folders}')

    def load_trace_analyzer(self, trace_dir):
        cache_path = os.path.join(trace_dir, 'analyzer_cache.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                analyzer = pickle.load(f)
            self.logger.info(f'Analyzer loaded from {cache_path}')
        else:
            analyzer = MegatronPipelineParallelGroupTraceAnalysis(trace_dir=trace_dir, data_parallel_size=self.data_parallel_size, tensor_parallel_size=self.tensor_parallel_size, pipeline_parallel_size=self.pipeline_parallel_size)
            with open(cache_path, 'wb') as f:
                pickle.dump(analyzer, f)
            self.logger.info(f'Analyzer saved to {cache_path}')
        return analyzer
    
    def process_single_pp_group(self, trace_dir):
        output_dir = os.path.join(trace_dir, 'output')
        prepare_directory(output_dir, force_clear=True)
        
        analyzer_single_pp_group = self.load_trace_analyzer(trace_dir)
        analyzer_single_pp_group.analyze_pipeline_parallel()
        
        return analyzer_single_pp_group
    
    def analyze(self):
        self.analysis_list = []
        for folder in self.assigned_folders:
            analyzer_single_pp_group = self.process_single_pp_group(folder)
            self.analysis_list.append(analyzer_single_pp_group)
        self.gather_infos_from_all_ranks()
        self.analyze_anomalies()
        self.post_process()
    
    def gather_infos_from_all_ranks(self):
        self.logger.info('Gathering information from all ranks.')
        
        # 序列化发送的数据
        send_data = pickle.dumps(self.analysis_list)
        send_data_size = len(send_data)

        # 收集每个进程发送的数据大小
        send_counts = self.comm.gather(send_data_size, root=0)
        
        # 根进程准备接收缓冲区和偏移量
        if self.rank == 0:
            recv_data = bytearray(sum(send_counts))
            recv_displs = [sum(send_counts[:i]) for i in range(self.world_size)]
        else:
            recv_data = None
            recv_displs = None

        # 使用 MPI_Gatherv 收集所有进程的数据
        self.comm.Gatherv(send_data, [recv_data, send_counts, recv_displs, MPI.BYTE], root=0)

        if self.rank == 0:
            # 根进程反序列化并合并数据
            self.total_analysis_lists = []
            for i in range(self.world_size):
                start_index = recv_displs[i]
                end_index = start_index + send_counts[i]
                data_chunk = recv_data[start_index:end_index]
                analysis_list = pickle.loads(data_chunk)
                self.total_analysis_lists.extend(analysis_list)

            self.logger.info(f'Total analysis lists gathered: {len(self.total_analysis_lists)}')
            self.total_traces_list = [analysis.t.traces for analysis in self.total_analysis_lists]
            self.total_trace_df = self.combine_traces()
        else:
            self.total_analysis_lists = None
            self.total_traces_list = None
            self.total_trace_df = None
    
    def combine_traces(self):
        for trace in self.total_traces_list:
            for pp_stage_id, rank in enumerate(sorted(trace.keys())):
                trace_df = trace[rank]
                trace_df['rank'] = rank
                trace_df['pp_stage'] = pp_stage_id
        
        trace_dfs_list = [trace_df for trace in self.total_traces_list for trace_df in list(trace.values())]
        return pd.concat(trace_dfs_list)
    
    def analyze_anomalies(self):
        if self.total_trace_df is None:
            return
        
        self.detect_anomalies()
        self.plot_anomalies()
        
    def detect_anomalies(self, threshold=2):
        # 计算每个full_name组的均值和标准差
        mean_std_df = self.total_trace_df.groupby('full_name')['dur'].agg(['mean', 'std']).reset_index()
        # 计算异常值的阈值
        mean_std_df['lower_bound'] = mean_std_df['mean'] - threshold * mean_std_df['std']
        mean_std_df['upper_bound'] = mean_std_df['mean'] + threshold * mean_std_df['std']
        
        # 标记异常值
        self.total_trace_df = self.total_trace_df.merge(mean_std_df, on='full_name', suffixes=('', '_group'))
        self.total_trace_df['is_anomaly'] = ((self.total_trace_df['dur'] < self.total_trace_df['lower_bound']) |
                                            (self.total_trace_df['dur'] > self.total_trace_df['upper_bound']))
        self.anomaly_status = self.total_trace_df['is_anomaly']
    
    def plot_anomalies(self):
        if self.anomaly_status is None:
            self.logger.info("Anomalies have not been detected. Please run detect_anomalies_std first.")
            return
        
        spans_with_anomalies = self.total_trace_df[self.anomaly_status].groupby('full_name').any().reset_index()
        num_spans_with_anomalies = spans_with_anomalies['is_anomaly'].sum()
        
        if num_spans_with_anomalies == 0:
            self.logger.info("No anomalies found.")
            return
        
        for _, row in spans_with_anomalies.iterrows():
            if row['is_anomaly']:
                subset = self.total_trace_df[self.total_trace_df['full_name'] == row['full_name']]
                
                # 根据'rank'列对subset进行排序
                subset = subset.sort_values(by='rank')
                
                # 创建一个新的图和轴对象
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 根据是否为异常值分配颜色
                colors = ['red' if x else 'blue' for x in subset['is_anomaly']]
                
                # 绘制条形图，使用'rank'作为x轴，'dur'作为y轴
                sns.barplot(x='rank', y='dur', data=subset, palette=colors, ax=ax)
                
                # 设置图表标题
                ax.set_title(f'{row["full_name"]}')
                
                # 设置x轴的标签为排序后的'rank'列的值
                ax.set_xticklabels(subset['rank'].astype(str), rotation=45)  # rotation参数可以调整标签的旋转角度
                
                # 设置y轴标签
                ax.set_ylabel('Duration')
                
                # 调整布局以适应标签
                plt.tight_layout()
                
                # 保存图形到文件，文件名以'full_name'为依据
                filename = f"{row['full_name'].replace('/', '.')}.png"  # 替换'/'以避免文件名问题
                filename = os.path.join(self.stragglers_dir, filename)
                plt.savefig(filename)
                
                # 关闭图形以释放内存
                plt.close(fig)