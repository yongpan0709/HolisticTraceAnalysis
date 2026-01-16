from functools import reduce
import operator

BYTES_DICT = {
    'c10::Half': 2,
    'c10::Float': 4,
    'c10::Int': 1,
    'c10::BFloat16': 2,
    'long int': 8,
    'float': 4,
    'unsigned char': 1
}


def _prod(shape):
    return reduce(operator.mul, shape, 1)

def calculate_groupedlinear_flops(input_dims, shape_from_func):
    if len(input_dims) < 14:
        if "TERowParallelGroupedLinear_0" in shape_from_func: # corresponding to grouped gemm 1
            input_dims = [[32768, 2048], [], [], [], [], [], [], [], [], [], [], [], [], [7168, 2048]]
        elif "TEColumnParallelGroupedLinear_0" in shape_from_func: # corresponding to grouped gemm 0 
            input_dims = [[32768, 7168], [], [], [], [], [], [], [], [], [], [], [], [], [4096, 7168]]
        else:
            return 0
    shape = input_dims[0]
    n = input_dims[16][0]
    macs = _prod(shape) * n * 2
    return macs/ 1e12

def calculate_linear_flops(input_dims, shape_from_func):
    #print(f"input_dims {input_dims}, shape_from_func: {shape_from_func}")
    macs = 0
    if shape_from_func.startswith("_LayerNormLinear"):
        shape = input_dims[0][:3]
        m = input_dims[3][0]
        macs = _prod(shape) * m * 2
    elif shape_from_func.startswith("_Linear") or shape_from_func.startswith("RouterGatingLinearFunction") or shape_from_func.startswith("LinearWithGradAccumulationAndAsyncCommunication"):
        shape = input_dims[0]
        m = input_dims[1][0]
        macs = _prod(shape) * m * 2
 
    return macs/ 1e12

def calculate_scaled_dot_product_attention_flash_musa_flops(input_dims, shape_from_func):
    #shape = input_dims[0]
    #m = input_dims[1][0]
    #macs = _prod(shape) * m
    return 2

def get_num_of_bytes(type):
    if type not in BYTES_DICT: 
        print(f'type {type} not in BYTES_DICT, is set to 1B')
    return BYTES_DICT.get(type, 1)

def calculate_CheckpointWithoutOutputFunction(input_dims, input_type):
    volume = _prod(input_dims[0]) * get_num_of_bytes(input_type[0]) * 2 # read and write
    return volume / (1024 ** 4)  # TB
