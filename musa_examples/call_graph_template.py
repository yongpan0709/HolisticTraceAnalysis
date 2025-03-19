from collections import defaultdict
from typing import Dict, List
import copy

output_template = """
megatron/core/pipeline_parallel/schedules.py(173): forward_step
    pretrain_deepseekv2.py(139): get_batch
    nn.Module: LanguageModelEmbedding_0

    dense: 
    nn.Module: TransformerLayer_0

    moe:
    nn.Module: TransformerLayer_1
        nn.Module: RMSNorm_1
        nn.Module: MLASelfAttention_1
        megatron/core/fusions/fused_bias_dropout.py(42): _bias_dropout_add
        nn.Module: RMSNorm_2
        nn.Module: MoELayer_0
            nn.Module: TopKRouter_0
            megatron/core/transformer/moe/token_dispatcher.py(473): token_permutation
                megatron/core/transformer/moe/token_dispatcher.py(341): preprocess
                megatron/core/transformer/moe/moe_utils.py(221): permute
                megatron/core/tensor_parallel/mappings.py(524): all_to_all
                megatron/core/transformer/moe/moe_utils.py(353): sort_chunks_by_idxs
            nn.Module: TEGroupedMLP_0
                nn.Module: TEColumnParallelGroupedLinear_0
                    <built-in method fused_multi_quantize of PyCapsule object at 0x7f2f84f5a8b0>
                    transformer_engine/pytorch/cpp_extensions/gemm.py(147): general_grouped_gemm
                megatron/core/fusions/fused_bias_swiglu.py(76): bias_swiglu_impl
                nn.Module: TERowParallelGroupedLinear_0
                    <built-in method fused_multi_quantize of PyCapsule object at 0x7f2f84f5a8b0>
                    transformer_engine/pytorch/cpp_extensions/gemm.py(147): general_grouped_gemm
            megatron/core/transformer/moe/token_dispatcher.py(565): token_unpermutation
                megatron/core/transformer/moe/moe_utils.py(353): sort_chunks_by_idxs
                megatron/core/tensor_parallel/mappings.py(524): all_to_all
                megatron/core/transformer/moe/moe_utils.py(282): unpermute
            nn.Module: SharedExpertMLP_0
        megatron/core/fusions/fused_bias_dropout.py(42): _bias_dropout_add

    nn.Module: RMSNorm_3

    loss:
    nn.Module: ColumnParallelLinear_0
    megatron/core/models/common/language_module/language_module.py(66): compute_language_model_loss
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


def extract_func_name_from_template(output_template: str) -> List[str]:
    # Split the output_template by newlines and strip leading/trailing whitespace
    lines = [line for line in output_template.splitlines() if line.strip() and line.strip() not in ['dense:', 'moe:', 'loss:'] ]

    dup_func_name: Dict[str, int] = defaultdict(int)
    # Parse the call stack
    call_stack = []
    stack = []

    for line in lines:
        indent_level = count_leading_whitespace(line)

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
    for entry in stack:
        print('    ' * level + entry['name'])
        print('    ' * (level + 1) + 'Ancestors: ' + ' -> '.join(entry['ancestors']))
        print_call_stack_with_ancestors(entry['children'], level + 1)

if __name__ == '__main__':
    print(extract_func_name_from_template(output_template))