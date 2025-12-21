from pathlib import Path 
from hta.common.trace import Trace
from hta.common.trace_file import get_trace_files
from hta.configs.config import logger
from hta.configs.parser_config import  ParserConfig, AVAILABLE_ARGS
from hta.common.trace_call_graph import CallGraph, CallStackIdentity
from collections import defaultdict
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import json
from collections import deque
from call_graph_template import extract_func_name_from_template, extract_dup_or_shape_func_name_from_template, output_template_to_file, SHAPE_POSITION, set_pandas_display_options, output_template_to_file_dsv3, output_template_to_file_debug
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
    import time
    base_dir = "../"
    trace_dir = str(Path(base_dir).joinpath("1215-mr-iter2k"))
    cfg = ParserConfig.get_default_cfg()
    # config for extracting shape info
    cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
    ParserConfig.set_default_cfg(cfg)
    trace_files = get_trace_files(trace_dir)
    for rank, trace_file in trace_files.items():
        #if rank != 0:
        #   continue
        fix_json_value_missing(trace_file)