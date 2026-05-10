import json
import argparse

from copy import deepcopy  # 导入 deepcopy
import re
import os
from musa_examples.trace_json_repair import fix_json_value_missing
from musa_examples.megatron_pipeline_group.distribute_trace_analysis import DistributedMegatronTraceAnalysis
  

# filepath: /home/mccxadmin/Documents/py-projects/HolisticTraceAnalysis/filter-user-annotate.py
# 加载 JSON 文件
FILTER_OUT_FUNCS = [
    ".*__init__.*",
    ".*__enter__.*",
    ".*__exit__.*",
    "torch/.*__call__.*",
    "transformer_engine/.*__call__.*",
    "triton/.*__call__.*",
    "torch/utils/data/_utils/pin_memory.py\(\d+\):.*",
    "<built-in .*>",
    "musaEventQuery",
    "threading.py\(\d+\): .*",
    "multiprocessing/.*\(\d+\): .*",
    "torch/multiprocessing/.*\(\d+\): .*",
    "torch/storage.py\(\d+\): .*",
    "selectors.py\(\d+\): .*",
    "socket.py\(\d+\): .*",
    "hmac.py\(\d+\): .*",
    "abc.py\(\d+\): .*",
]
COMBINED_PATTERN = '|'.join(FILTER_OUT_FUNCS)

MOONCAKE_P2P_FUNCS = [
    'mooncake_p2p_recv_from',
    'mooncake_p2p_send_to',
]
MOONCAKE_P2P_PATTERN = '|'.join(MOONCAKE_P2P_FUNCS)

FILTER_OUT_CAT_FUNCS = [
    "user_annotation",
    "gpu_user_annotation",
]
def filter_out_funcs(file_path, redirect_new_trace_path):
    print(f'redirect_new_trace_path: {redirect_new_trace_path}')
    fix_json_value_missing(file_path)
    with open(file_path, 'r') as file:  # 替换 'data.json' 为你的 JSON 文件路径
        data = json.load(file)
    dup_data = {}
    for key, value in data.items():
        if key == 'traceEvents':
            if 'traceEvents' not in dup_data:
                dup_data['traceEvents'] = []
            for item in value:             
                if 'name' in item:
                    if re.match(COMBINED_PATTERN, item["name"]):
                        continue
                    elif re.match(MOONCAKE_P2P_PATTERN, item["name"]):
                        item['cat'] = 'user_annotation'
                        dup_data['traceEvents'].append(deepcopy(item))
                    else:
                        dup_data['traceEvents'].append(deepcopy(item))
                else:
                    dup_data['traceEvents'].append(deepcopy(item))
        else:
            dup_data[key] = deepcopy(value)

    # 将结果写入新的 JSON 文件
    with open(redirect_new_trace_path, 'w') as file:
        json.dump(dup_data, file, indent='\t')

def create_directory_if_not_exists(path):
    """创建文件夹，如果已存在则忽略（推荐方法）"""
    os.makedirs(path, exist_ok=True)
    return path

if __name__ == "__main__":
    # 示例用法
    #ETL
    #mpirun -allow-run-as-root -np 2 --bind-to none --hostfile ./hostfile --map-by ppr:2:node --wdir /Users/huayongpan/Documents/hta/HolisticTraceAnalysis-gitlab/musa_examples python trace_etl.py --tp 1 --pp 4 --dp 2 --ep 8 --trace_dir /Users/huayongpan/Documents/hta/HolisticTraceAnalysis-gitlab/mooncake_ep_overlap_delay_wgrad_trace_p2p/iteration_64
    parser = argparse.ArgumentParser(
        description="Generate version_info.mk file which will include ddk, musa_toolkit, mudnn, mccl download url.",
        usage=f"mpirun -allow-run-as-root -np 2 --bind-to none --hostfile ./hostfile --map-by ppr:2:node --wdir /Users/huayongpan/Documents/hta/HolisticTraceAnalysis-gitlab/musa_examples python trace_etl.py --trace_dir <trace directory> --tp <tp size> --pp <pp size> --dp <dp size> --ep <ep size>")
    parser.add_argument("--trace-dir", required=True, help="trace directory")
    parser.add_argument('--tp', type=int, required=True, help='tp size')
    parser.add_argument('--pp', type=int, required=True, help='pp size')
    parser.add_argument('--dp', type=int, required=True, help='dp size')
    parser.add_argument('--ep', type=int, required=True, help='ep size')
    args = parser.parse_args()
    trace_dir = args.trace_dir
    trace_dir = trace_dir.rstrip('/')
    redirect_path = create_directory_if_not_exists(trace_dir+'-etl')
    print(f'Origin Trace_dir: {trace_dir}, After filtering Dir: {redirect_path}')
    dist_megatron_analysis = DistributedMegatronTraceAnalysis(trace_dir, args.tp, args.ep, args.dp, args.pp)
    dist_megatron_analysis.pp_etl(redirect_path, filter_out_funcs)
    # pp_groups = [
    #     [0, 16, 32, 48]
    #     #, [1, 17, 33, 49]
    # ]
    # #ranks = [0, 8, 16]
    # trace_files = get_trace_files(trace_dir)
    # t0 = time.perf_counter()
    # print(f'start ts: {t0}')
    # for ranks in pp_groups:
    #     tasks = []
    #     for rank in ranks:
    #         #print(f'rank {rank} in ranks:{ranks}')
    #         trace_file = trace_files[rank]
    #         filename = os.path.basename(trace_file)
    #         redirect_new_trace_path = os.path.join(redirect_path, filename)
    #         tasks.append((trace_file, redirect_new_trace_path))
    #     num_procs = min(mp.cpu_count(), len(ranks))
    #     print(f'tasks: {tasks}')
    #     with mp.get_context("fork").Pool(num_procs) as pool:
    #         results = pool.starmap(filter_out_funcs, tasks)
    #         pool.close()
    #         pool.join()
    # t1 = time.perf_counter()
    # print(f"calculating critical path took {t1 - t0:2f} seconds")