from pathlib import Path 
from hta.common.trace import Trace
from hta.configs.config import logger
from hta.configs.parser_config import  ParserConfig, AVAILABLE_ARGS
from hta.common.trace_call_graph import CallGraph

base_dir = "../"
trace_dir = str(Path(base_dir).joinpath("ds-trace-files"))
cfg = ParserConfig.get_default_cfg()
# config for extracting shape info
cfg.add_args(ParserConfig.ARGS_INPUT_SHAPE)
ParserConfig.set_default_cfg(cfg)
t = Trace(trace_dir=trace_dir)
t.parse_traces()
# transform name and cat columns to s_name and s_cat
# name and cat are kernel id
t.decode_symbol_ids(use_shorten_name=False)
cg = CallGraph(t, ranks=[0])

print("Get the info of Func nn.Module: MLASelfAttention_0")
MLASelfAttention = cg.trace_data.traces[0][cg.trace_data.traces[0]['s_name']=="nn.Module: MLASelfAttention_0"]
print(f"All kernels duration sum: {MLASelfAttention['kernel_dur_sum'].values}")
print(f"The total number of kernels executed: {MLASelfAttention['num_kernels'].values}")
print(f"The start time of first kernel executed: {MLASelfAttention['first_kernel_start'].values}")
print(f"The end time of last kernel executed: {MLASelfAttention['last_kernel_end'].values}")

expect_kernel_name = "musa_asm_bf16bf16bf16bf16gemm_nt_tce_768_256x384B128_squad_level_epilogue"
print(f"\n\nGet the shape info of kernel name[{expect_kernel_name}]")
df = cg.trace_data.traces[0]

index_kernel_in_df = df[df['s_name'] == expect_kernel_name].index
first_index = index_kernel_in_df.values[0]
# print(first_index)
check_shape_of_index  = first_index
while True:
    shape = df.loc[check_shape_of_index,'input_dims']
    if shape == '-1':
        check_shape_of_index = df.loc[check_shape_of_index,'parent']
        print(f'parent index: {check_shape_of_index}')
        # print()
    else:
        print(f'The shape info: {shape}')
        break