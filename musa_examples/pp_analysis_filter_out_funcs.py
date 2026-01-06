import json
from copy import deepcopy  # 导入 deepcopy
from pathlib import Path 
from hta.common.trace_file import get_trace_files
import re
import os
import time
import multiprocessing as mp
from trace_json_repair import fix_json_value_missing

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
                if 'name' in item and re.match(COMBINED_PATTERN, item["name"]):
                    continue
                else:
                    dup_data['traceEvents'].append(deepcopy(item))
        else:
            dup_data[key] = deepcopy(value)

    # 将结果写入新的 JSON 文件
    with open(redirect_new_trace_path, 'w') as file:
        json.dump(dup_data, file, indent='\t')

if __name__ == "__main__":
    # 示例用法
    base_dir = "../"
    trace_dir = str(Path(base_dir).joinpath("20260106-iter8"))
    redirect_path = '20260106-3h-iter8-filtered'
    pp_groups = [
        [0, 8, 16],
        [1, 9, 17],
        [2, 10, 18],
        [3, 11, 19],
        [4, 12, 20],
        [5, 13, 21],
        [6, 14, 22],
        [7, 15, 23],
    ]
    #ranks = [0, 8, 16]
    trace_files = get_trace_files(trace_dir)
    t0 = time.perf_counter()
    print(f'start ts: {t0}')
    for ranks in pp_groups:
        tasks = []
        for rank in ranks:
            #print(f'rank {rank} in ranks:{ranks}')
            trace_file = trace_files[rank]
            filename = os.path.basename(trace_file)
            redirect_new_trace_path = os.path.join(base_dir, redirect_path, filename)
            tasks.append((trace_file, redirect_new_trace_path))
        num_procs = min(mp.cpu_count(), len(ranks))
        print(f'tasks: {tasks}')
        with mp.get_context("fork").Pool(num_procs) as pool:
            results = pool.starmap(filter_out_funcs, tasks)
            pool.close()
            pool.join()
    t1 = time.perf_counter()
    print(f"calculating critical path took {t1 - t0:2f} seconds")