from collections import defaultdict
from typing import Dict, List,Set
import copy

DUP_LABEL = "@dup@"
SHAPEUP_LABEL = "@shape@"
SHAPEDOWN_LABEL = "shape@down"
SHAPE_LABEL = [SHAPEUP_LABEL, SHAPEDOWN_LABEL]
ANNOTATE_LABEL = [DUP_LABEL, SHAPEUP_LABEL, SHAPEDOWN_LABEL]

SHAPE_POSITION = {
    '_AllToAll': 1,
    '_GroupedLinear': 2
}

output_template_to_file = r"""
pretrain_deepseekv2.py\(\d+\): forward_step
    pretrain_deepseekv2.py\(\d+\): get_batch
    nn.Module: LanguageModelEmbedding_0

    dense: 
    nn.Module: TransformerLayer_0

    moe:
    nn.Module: TransformerLayer_1
        nn.Module: RMSNorm_1
        nn.Module: MLASelfAttention_1
            aten::scaled_dot_product_attention
        megatron/core/fusions/fused_bias_dropout.py\(\d+\): _bias_dropout_add
        nn.Module: RMSNorm_2
        nn.Module: MoELayer_0
            nn.Module: TopKRouter_0
                megatron/core/transformer/moe/router.py\(\d+\): gating
                megatron/core/transformer/moe/router.py\(\d+\): routing
                    musa_patch/moe_utils.py\(\d+\): topk_softmax_with_capacity
                        megatron/core/transformer/moe/moe_utils.py\(\d+\): group_limited_topk
                    <built-in method softmax of type object at 0x\w+>
                    megatron/core/transformer/moe/router.py\(\d+\): apply_load_balancing_loss
            megatron/core/transformer/moe/token_dispatcher.py\(\d+\): token_permutation
                megatron/core/transformer/moe/token_dispatcher.py\(\d+\): preprocess
                megatron/core/transformer/moe/moe_utils.py\(\d+\): permute
                megatron/core/tensor_parallel/mappings.py\(\d+\): all_to_all @dup@
                    _AllToAll @dup@ @shape@
                megatron/core/transformer/moe/moe_utils.py\(\d+\): sort_chunks_by_idxs @dup@
            nn.Module: TEGroupedMLP_0
                nn.Module: TEColumnParallelGroupedLinear_0
                    _GroupedLinear @dup@ @shape@
                        <built-in method fused_multi_quantize of PyCapsule object at 0x\w+> @dup@
                        transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm @dup@
                megatron/core/fusions/fused_bias_swiglu.py\(\d+\): bias_swiglu_impl
                nn.Module: TERowParallelGroupedLinear_0
                    _GroupedLinear @dup@ @shape@
                        <built-in method fused_multi_quantize of PyCapsule object at 0x\w+> @dup@
                        transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm @dup@
            megatron/core/transformer/moe/token_dispatcher.py\(\d+\): token_unpermutation
                megatron/core/transformer/moe/moe_utils.py\(\d+\): sort_chunks_by_idxs @dup@
                megatron/core/tensor_parallel/mappings.py\(\d+\): all_to_all @dup@
                    _AllToAll @dup@ @shape@
                megatron/core/transformer/moe/moe_utils.py\(\d+\): unpermute
            nn.Module: SharedExpertMLP_0
        megatron/core/fusions/fused_bias_dropout.py\(\d+\): _bias_dropout_add

    nn.Module: RMSNorm_3

    loss:
    nn.Module: ColumnParallelLinear_0
    megatron/core/models/common/language_module/language_module.py\(\d+\): compute_language_model_loss
"""
# Function to count leading spaces or tabs
def count_leading_whitespace(line):
    count = 0
    for char in line:
        if char == ' ' or char == '\t':
            count += 1
        else:
            break
    return count

# transform call stack to list
def call_stack_with_ancestors_to_list(stack, call_stack_list, level=0):
    for entry in stack:
        call_stack_list.append((entry['name'], entry['ancestors']))
        call_stack_with_ancestors_to_list(entry['children'], call_stack_list, level + 1)

def extract_dup_or_shape_func_name_from_template(output_template: str) -> (Set[str], Dict[str, str]):
    # Split the output_template by newlines and strip leading/trailing whitespace
    lines = [line for line in output_template.splitlines() if line.strip() and line.strip() not in ['dense:', 'moe:', 'loss:'] ]

    dup_func_name = set()
    need_shape_func_name: Dict[str, str] = defaultdict(str)
    for line in lines:
        Has_DUP_LABEL = False
        if DUP_LABEL in line:
            line = line.replace(DUP_LABEL, '')
            Has_DUP_LABEL = True

        Has_SHAPE_LABEL = False
        shape_direction = ''
        for shape_annotate in SHAPE_LABEL:
            if shape_annotate in line:
                line = line.replace(shape_annotate, '')
                Has_SHAPE_LABEL = True
                shape_direction = shape_annotate

        func_name = line.strip()
        if Has_DUP_LABEL:
            dup_func_name.add(func_name)
        if Has_SHAPE_LABEL:
            need_shape_func_name[func_name] = shape_direction
    return dup_func_name, need_shape_func_name

def extract_func_name_from_template(output_template: str) -> List[str]:
    # Split the output_template by newlines and strip leading/trailing whitespace
    lines = [line for line in output_template.splitlines() if line.strip() and line.strip() not in ['dense:', 'moe:', 'loss:'] ]

    dup_func_name: Dict[str, int] = defaultdict(int)
    # Parse the call stack
    call_stack = []
    stack = []

    for line in lines:
        indent_level = count_leading_whitespace(line)
        for annotate in ANNOTATE_LABEL:
            if annotate in line:
                line = line.replace(annotate, '')
        func_name = line.strip()+'@'+ str(dup_func_name[line.strip()])
        dup_func_name[line.strip()] += 1
        entry = {'name': func_name, 'children': [], 'ancestors': []}

        while stack and stack[-1]['indent'] >= indent_level:
            stack.pop()

        if stack:
            stack[-1]['node']['children'].append(entry)
            entry['ancestors'] = stack[-1]['node']['ancestors'] + [stack[-1]['node']['name']]
        else:
            call_stack.append(entry)

        stack.append({'node': entry, 'indent': indent_level})
    # Print the parsed call stack with ancestors
    call_stack_list = []
    call_stack_with_ancestors_to_list(call_stack, call_stack_list)
    return call_stack_list

# Function to print the call stack with ancestors
def print_call_stack_with_ancestors(stack, level=0):
    for forward_func_name, func_ancestors in stack:
        print('    ' * level + forward_func_name)
        print('    ' * (level + 1) + 'Ancestors: ' + ' -> '.join(func_ancestors))
        # print_call_stack_with_ancestors(entry['children'], level + 1)

if __name__ == '__main__':
    print(extract_func_name_from_template(output_template_to_file))
    dup_func_name, need_shape_func_name = extract_dup_or_shape_func_name_from_template(output_template_to_file)
    print(dup_func_name)
    print(need_shape_func_name)