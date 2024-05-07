def get_3d_parallel_groups(tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size):
    world_size = tensor_model_parallel_size * pipeline_model_parallel_size * data_parallel_size

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    num_data_parallel_groups: int = world_size // data_parallel_size 

    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))

    all_tensor_parallel_group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        all_tensor_parallel_group_ranks.append(list(ranks))

    all_pipeline_parallel_group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        all_pipeline_parallel_group_ranks.append(list(ranks))

    return all_data_parallel_group_ranks, all_tensor_parallel_group_ranks, all_pipeline_parallel_group_ranks

def get_data_parallel_group_id(rank, tensor_model_parallel_size, pipeline_model_parallel_size):
    return rank % (tensor_model_parallel_size * pipeline_model_parallel_size)

def get_tensor_parallel_group_id(rank, tensor_model_parallel_size):
    return rank % tensor_model_parallel_size

def get_pipeline_parallel_group_id(rank, tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size):
    global_rank = rank % (tensor_model_parallel_size * pipeline_model_parallel_size)
    return global_rank // tensor_model_parallel_size

def get_pipeline_parallel_rank(rank, tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size):
    return (rank // tensor_model_parallel_size) % pipeline_model_parallel_size

def get_previous_pipeline_rank(rank, tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size):
    prev_rank = rank - tensor_model_parallel_size * data_parallel_size
    return prev_rank if prev_rank >= 0 else None

def get_next_pipeline_rank(rank, tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size):
    world_size = tensor_model_parallel_size * pipeline_model_parallel_size * data_parallel_size
    next_rank = rank + tensor_model_parallel_size * data_parallel_size
    return next_rank if next_rank < world_size else None

def is_first_stage(rank, tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size):
    prev_pp_rank = get_previous_pipeline_rank(rank, tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size)
    return prev_pp_rank is None

def is_last_stage(rank, tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size):
    next_pp_rank = get_next_pipeline_rank(rank, tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size)
    return next_pp_rank is None
