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
import numpy as np
from sklearn.ensemble import IsolationForest

def gatherv_p2p(comm, sendbuf, recvbuf, root=0):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == root:
        recv_data, recvcounts, displs, recvtype = recvbuf
        recvcounts = np.asarray(recvcounts, dtype=np.int64)
        displs = np.asarray(displs, dtype=np.int64)
        
        for i in range(size):
            if i == root:
                recv_data[displs[i]:displs[i] + recvcounts[i]] = sendbuf
            else:
                temp_recvbuf = np.empty(recvcounts[i], dtype='b')
                comm.Recv([temp_recvbuf, recvtype], source=i)
                recv_data[displs[i]:displs[i] + recvcounts[i]] = temp_recvbuf
    else:
        comm.Send([sendbuf, MPI.BYTE], dest=root)

def gather_data(comm, send_data, use_p2p=True):
    rank = comm.Get_rank()
    size = comm.Get_size()
    send_data_size = len(send_data)
    
    # Collect the size of data sent by each process
    send_counts = comm.gather(send_data_size, root=0)
    
    if rank == 0:
        recv_displs = np.zeros(size, dtype=int)
        for i in range(1, size):
            recv_displs[i] = recv_displs[i - 1] + send_counts[i - 1]
        total_recv_size = sum(send_counts)
        recv_data = np.empty(total_recv_size, dtype='b')
    else:
        recv_displs = None
        recv_data = None
    
    if use_p2p:
        if rank == 0:
            recvbuf_p2p = (recv_data, send_counts, recv_displs, MPI.BYTE)
        else:
            recvbuf_p2p = None
        gatherv_p2p(comm, np.frombuffer(send_data, dtype='b'), recvbuf_p2p, root=0)
    else:
        comm.Gatherv(np.frombuffer(send_data, dtype='b'), [recv_data, send_counts, recv_displs, MPI.BYTE], root=0)
    
    return recv_data, send_counts, recv_displs
    
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
        self.comm.Barrier()
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
        logger.info(f'Process {self.rank} on {self.processor_name} started logging.')

    def init_pp_group_sub_dirs(self):
        logger.info('Initializing pipeline parallel group subdirectories.')
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
        logger.info('Initialization of pipeline parallel group subdirectories completed.')
    
    def assign_analysis_tasks(self):
        logger.info('Assigning analysis tasks.')
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
        logger.info(f'Assigned folders: {self.assigned_folders}')

    def load_trace_analyzer(self, trace_dir):
        cache_path = os.path.join(trace_dir, 'analyzer_cache.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                analyzer = pickle.load(f)
            logger.info(f'Analyzer loaded from {cache_path}')
        else:
            analyzer = MegatronPipelineParallelGroupTraceAnalysis(trace_dir=trace_dir, data_parallel_size=self.data_parallel_size, tensor_parallel_size=self.tensor_parallel_size, pipeline_parallel_size=self.pipeline_parallel_size)
            with open(cache_path, 'wb') as f:
                pickle.dump(analyzer, f)
            logger.info(f'Analyzer saved to {cache_path}')
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
        logger.info('Gathering information from all ranks.')

        # Serialize the data to be sent
        send_data = pickle.dumps(self.analysis_list)

        # Gather data using the helper function
        recv_data, send_counts, recv_displs = gather_data(self.comm, send_data)

        if self.rank == 0:
            # The root process deserializes and merges data
            self.total_analysis_lists = []
            for i in range(self.world_size):
                start_index = recv_displs[i]
                end_index = start_index + send_counts[i]
                data_chunk = recv_data[start_index:end_index]
                analysis_list = pickle.loads(data_chunk)
                self.total_analysis_lists.extend(analysis_list)

            logger.info(f'Total analysis lists gathered: {len(self.total_analysis_lists)}')
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
    
    def detect_anomalies(self):
        # return self.detect_anomalies_zscore()
        return self.detect_anomalies_ml()
    

    def detect_anomalies_zscore(self, threshold=1, min_std=0.1):
        # Calculate the mean and standard deviation for each full_name and pp_stage group
        mean_std_df = self.total_trace_df.groupby(['full_name', 'pp_stage'])['dur'].agg(['mean', 'std']).reset_index()
        
        # Set a minimum standard deviation to avoid extremely small values
        mean_std_df['std'] = mean_std_df['std'].apply(lambda x: max(x, min_std))
        
        # Calculate Z-Score
        self.total_trace_df = self.total_trace_df.merge(mean_std_df, on=['full_name', 'pp_stage'], suffixes=('', '_group'))
        self.total_trace_df['z_score'] = (self.total_trace_df['dur'] - self.total_trace_df['mean']) / self.total_trace_df['std']
        
        # Mark anomalies
        self.total_trace_df['is_anomaly'] = self.total_trace_df['z_score'].abs() > threshold
        self.anomaly_status = self.total_trace_df['is_anomaly']
    
    def detect_anomalies_ml(self):
        # Apply Isolation Forest
        iso_forest = IsolationForest()  # Set contamination to the expected proportion of anomalies
        self.total_trace_df['is_anomaly'] = iso_forest.fit_predict(self.total_trace_df[['dur']])
        self.total_trace_df['is_anomaly'] = self.total_trace_df['is_anomaly'] == -1
        self.anomaly_status = self.total_trace_df['is_anomaly']

    
    def plot_anomalies(self):
        if self.anomaly_status is None:
            logger.info("Anomalies have not been detected. Please run detect_anomalies first.")
            return
        
        spans_with_anomalies = self.total_trace_df[self.anomaly_status].groupby(['full_name', 'pp_stage']).any().reset_index()
        num_spans_with_anomalies = spans_with_anomalies['is_anomaly'].sum()
        
        if num_spans_with_anomalies == 0:
            logger.info("No anomalies found.")
            return
        
        for _, row in spans_with_anomalies.iterrows():
            if row['is_anomaly']:
                subset = self.total_trace_df[(self.total_trace_df['full_name'] == row['full_name']) &
                                            (self.total_trace_df['pp_stage'] == row['pp_stage'])]
                
                # Sort subset by 'rank' column
                subset = subset.sort_values(by='rank')
                
                # Create a new figure and axis object
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Assign colors based on whether the value is an anomaly
                colors = ['red' if x else 'blue' for x in subset['is_anomaly']]
                
                # Plot a bar chart, using 'rank' as x-axis and 'dur' as y-axis
                sns.barplot(x='rank', y='dur', data=subset, palette=colors, ax=ax)
                
                # Set the title of the chart
                ax.set_title(f'{row["full_name"]} - pp stage {row["pp_stage"]}')
                
                # Set x-axis labels to the values in the sorted 'rank' column
                ax.set_xticklabels(subset['rank'].astype(str), rotation=45)  # The rotation parameter can adjust the angle of the labels
                
                # Set y-axis label
                ax.set_ylabel('Duration')
                
                # Adjust layout to fit labels
                plt.tight_layout()
                
                # Save the plot to a file, with the filename based on 'full_name' and 'pp_stage'
                filename = f"{row['full_name'].replace('/', '.')}_stage{row['pp_stage']}.png"  # Replace '/' to avoid filename issues
                filename = os.path.join(self.stragglers_dir, filename)
                plt.savefig(filename)
                
                # Close the plot to release memory
                plt.close(fig)