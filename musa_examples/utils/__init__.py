# musa_examples utils module
from .musa_basic_kernel_info import (
    BYTES_DICT,
    calculate_CheckpointWithoutOutputFunction,
    calculate_groupedlinear_tflops_or_bw,
    calculate_linear_tflops_or_bw,
    calculate_scaled_dot_product_attention_flash_musa_flops,
    drop_empty_arrays,
    get_num_of_bytes,
)
from .musa_fwdbwd_util import (
    get_backward_duration,
    get_forward_duration_dup,
    get_forward_duration_uniq,
)
from .parallel_state import (
    RankGenerator,
    generate_masked_orthogonal_rank_groups,
    get_data_parallel_group_id,
    get_next_pipeline_rank,
    get_pipeline_parallel_group_id,
    get_pipeline_parallel_rank,
    get_previous_pipeline_rank,
    get_tensor_parallel_group_id,
    is_first_stage,
    is_last_stage,
)
from .timing import (
    TimingRecord,
    TimingTracker,
    get_timer,
    reset_timer,
    time_it,
)

__all__ = [
    "TimingRecord",
    "TimingTracker",
    "get_timer",
    "reset_timer",
    "time_it",
    "RankGenerator",
    "generate_masked_orthogonal_rank_groups",
    "get_data_parallel_group_id",
    "get_next_pipeline_rank",
    "get_pipeline_parallel_group_id",
    "get_pipeline_parallel_rank",
    "get_previous_pipeline_rank",
    "get_tensor_parallel_group_id",
    "is_first_stage",
    "is_last_stage",
    "BYTES_DICT",
    "calculate_CheckpointWithoutOutputFunction",
    "calculate_groupedlinear_tflops_or_bw",
    "calculate_linear_tflops_or_bw",
    "calculate_scaled_dot_product_attention_flash_musa_flops",
    "drop_empty_arrays",
    "get_num_of_bytes",
    "get_backward_duration",
    "get_forward_duration_dup",
    "get_forward_duration_uniq",
]
