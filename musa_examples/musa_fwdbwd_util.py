from hta.common.trace_call_graph import CallStackIdentity
from collections import defaultdict
from typing import Dict, List
from collections import deque
import numpy as np
import pandas as pd

def get_backward_duration(df, cg, rank, forward_index): 
    # forward_index  = df[df['name'] == forward_sym_id].index
    forward_children_with_bwd_id: Dict[np.int64, List[np.int64]] = defaultdict(list)
    """
    Todo: refact to void checking children and grand-children funcs
    """
    for forward_as_parent in forward_index:
        parents = deque()
        parents.append(forward_as_parent)
        while len(parents) > 0:
            cur_node = parents.popleft()
            cur_node_fwdbwd_index = df.loc[cur_node, 'fwdbwd_index']
            if cur_node_fwdbwd_index > 0:
                cur_node_fwdbwd_num_kernels = df.loc[cur_node_fwdbwd_index, 'num_kernels']
                if cur_node_fwdbwd_num_kernels > 0:
                    forward_children_with_bwd_id[forward_as_parent].append(cur_node_fwdbwd_index)
            else:
                cur_callStackNode = cg.rank_to_nodes[rank].get(cur_node)
                cur_node_children = cur_callStackNode.children
                if len(cur_node_children) > 0:
                    for child in cur_node_children:
                        child_type = df.loc[child, 's_cat']
                        if child_type not in ['kernel']:
                            parents.append(child)
    
    backward_info: Dict[np.int64, np.float64] = defaultdict(np.float64)
    for forward_as_parent, bwd_children_indices in forward_children_with_bwd_id.items():
        bwd_children = df[df['index'].isin(bwd_children_indices)]
        first_kernel_start = bwd_children['first_kernel_start'].min()
        last_kernel_end = bwd_children['last_kernel_end'].max()
        backward_info[forward_as_parent] = (last_kernel_end - first_kernel_start)
    backward_stat_info = pd.DataFrame.from_dict(backward_info, orient='index', columns=['kernel_span'])
    return backward_stat_info

def get_forward_duration_uniq(df, forward_func_name):
    node_index = df[df['s_name'].str.match(pat=r"^"+forward_func_name+r"$")].index
    return df[df['index'].isin(node_index)]

def get_forward_duration_dup(df, forward_func_name, func_ancestors, cg, rank, func_mapping_node_index):
    node_index: List[np.int64] = []
    nearest_ancestor_index = func_mapping_node_index[func_ancestors[-1]]
    for index, row in df[df['s_name'].str.match(pat=r"^"+forward_func_name+r"$")].iterrows():
        pid, tid = row['pid'], row['tid']
        path_to_root = cg.rank_to_stacks[rank][CallStackIdentity(rank, pid, tid)].get_path_to_root(index)
        if len(set(path_to_root).intersection(set(nearest_ancestor_index))) > 0:
            node_index.append(index)
    # print("dup node index: ", node_index)
    return df[df['index'].isin(node_index)]