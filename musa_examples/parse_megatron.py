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
    VPP_SIZE = 2
    # trace_dir = '/Users/huayongpan/Documents/hta/HolisticTraceAnalysis-gitlab/20260204-vpp2-etl'
    trace_dir = '/Users/huayongpan/Documents/hta/HolisticTraceAnalysis-gitlab/20260206-nooverlap-1f1b-etl'
    # trace_dir = '/Users/huayongpan/Documents/hta/HolisticTraceAnalysis-gitlab/20260206-1f1b-etl'

    # Llama3
    # TP_SIZE = 2
    # PP_SIZE = 4
    # DP_SIZE = 1
    # trace_dir = '/Users/huayongpan/Documents/hta/HolisticTraceAnalysis-gitlab/tracedata/llama3-20250814-tp2_pp4_numbs8/llama-iter8'
    dist_megatron_analysis = DistributedMegatronTraceAnalysis(trace_dir, TP_SIZE, EP_SIZE, DP_SIZE, PP_SIZE, pp_schedule=PP_SCHEDULE, vpp_size=VPP_SIZE)
    dist_megatron_analysis.analyze()

if __name__ == '__main__':
    main()
# mpirun -allow-run-as-root -np 1 --bind-to none --hostfile ./hostfile --map-by ppr:1:node --wdir /Users/huayongpan/Documents/hta/HolisticTraceAnalysis-gitlab/musa_examples python parse_megatron.py