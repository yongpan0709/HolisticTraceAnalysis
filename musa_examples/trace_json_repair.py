from pathlib import Path
from hta.common.trace_file import get_trace_files
import json
import re


# Todo: value missing in trace, like: '"Process Group Description": ,'
# https://jira.mthreads.com/browse/MTAI-2111
def fix_json_value_missing(file_path):
    try:
        with open(file_path,"r",encoding="utf-8",errors="ignore") as fr:
            json.load(fr)
    except json.JSONDecodeError as e:
        print(f'✗ JSON Format Error: {e}, line: {e.lineno}, column: {e.colno}')
        s = ''
        with open(file_path,"r",encoding="utf-8",errors="ignore") as fr:
            s = fr.read()
        if len(s) > 0:
            s = re.sub(r':\s*(?=[,}])', r': ""', s)
            with open(file_path,"w",encoding="utf-8") as fw:
                fw.write(s)
    except Exception as e:
        print(f'✗ Error: {e}')

# HTA_DISABLE_NS_ROUNDING=1 python call_graph_kernel_dur_sum.py
if __name__ == "__main__":
    base_dir = "../"
    trace_dir = str(Path(base_dir).joinpath("2025-12-21_002019_fp8_balancing/iteration_8"))
    #cfg = ParserConfig.get_default_cfg()
    # config for extracting shape info
    #cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
    #ParserConfig.set_default_cfg(cfg)
    trace_files = get_trace_files(trace_dir)
    for rank, trace_file in trace_files.items():
        #if rank != 0:
        #   continue
        fix_json_value_missing(trace_file)