from megatron_pipeline_group.distribute_trace_analysis import DistributedMegatronTraceAnalysis
        

def main():
    # TP_SIZE = 2
    # PP_SIZE = 4
    # DP_SIZE = 1
    # trace_dir = '/home/dist/HolisticTraceAnalysis/llama3-20250728-tp2pp4/iteration_4'
    # DS MoE
    TP_SIZE = 1
    PP_SIZE = 4
    DP_SIZE = 2
    EP_SIZE = 8
    PP_SCHEDULE = '1f1b'
    MIRCRO_BATCHSIZE = 32
    #trace_dir = '/home/huayongpan/hta/HolisticTraceAnalysis-gitlab/mccl-1f1b'
    #dist_megatron_analysis = DistributedMegatronTraceAnalysis(trace_dir, TP_SIZE, EP_SIZE, DP_SIZE, PP_SIZE, pp_schedule=PP_SCHEDULE, micro_bs = MIRCRO_BATCHSIZE)
    #dist_megatron_analysis.analyze()

    VPP_SIZE = 2
    MIRCRO_BATCHSIZE = 32
    PP_SCHEDULE = '1f1b-interleaved'
    #trace_dir = '/home/huayongpan/hta/HolisticTraceAnalysis-gitlab/mooncake-vpp2'
    #dist_megatron_analysis = DistributedMegatronTraceAnalysis(trace_dir, TP_SIZE, EP_SIZE, DP_SIZE, PP_SIZE, pp_schedule=PP_SCHEDULE, vpp_size=VPP_SIZE, micro_bs = MIRCRO_BATCHSIZE)
    #dist_megatron_analysis.analyze()

    PP_SIZE = 2
    DP_SIZE = 1
    EP_SIZE = 8
    PP_SCHEDULE = '1f1b-interleaved-epoverlap'
    MIRCRO_BATCHSIZE = 16
    trace_dir = '/home/huayongpan/hta/HolisticTraceAnalysis-gitlab/ep-overlap-etl'
    dist_megatron_analysis = DistributedMegatronTraceAnalysis(trace_dir, TP_SIZE, EP_SIZE, DP_SIZE, PP_SIZE, pp_schedule=PP_SCHEDULE, vpp_size=VPP_SIZE, micro_bs = MIRCRO_BATCHSIZE)
    dist_megatron_analysis.analyze()

if __name__ == '__main__':
    main()
# mpirun -allow-run-as-root -np 1 --bind-to none --hostfile ./hostfile --map-by ppr:1:node --wdir /home/huayongpan/hta/HolisticTraceAnalysis-gitlab/musa_examples python parse_megatron.py