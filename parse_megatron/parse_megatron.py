from hta.distribute_trace_analysis import DistributedMegatronTraceAnalysis
        
# MUST: setting PROFILER_WITH_STACK=0
def main():
    # TP_SIZE = 2
    # PP_SIZE = 4
    # DP_SIZE = 1
    # trace_dir = '/home/dist/HolisticTraceAnalysis/llama3-20250728-tp2pp4/iteration_4'
    # DS MoE
    TP_SIZE = 1
    PP_SIZE = 2
    DP_SIZE = 4
    trace_dir = '/Users/huayongpan/Documents/hta/HolisticTraceAnalysis-gitlab/tracedata/ds-20250814-tp1_pp2_dp4_mbs1_numbs32/ds-iteration_4'

    # Llama3
    # TP_SIZE = 2
    # PP_SIZE = 4
    # DP_SIZE = 1
    # trace_dir = '/Users/huayongpan/Documents/hta/HolisticTraceAnalysis-gitlab/tracedata/llama3-20250814-tp2_pp4_numbs8/llama-iter8'
    dist_megatron_analysis = DistributedMegatronTraceAnalysis(trace_dir, TP_SIZE, DP_SIZE, PP_SIZE)
    dist_megatron_analysis.analyze()

if __name__ == '__main__':
    main()