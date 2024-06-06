from hta.distribute_trace_analysis import DistributedMegatronTraceAnalysis
        

def main():
    TP_SIZE = 2
    PP_SIZE = 4
    DP_SIZE = 2
    trace_dir = '/home/dist/yiyuan/trace_dir_llama7b_blocking'
    dist_megatron_analysis = DistributedMegatronTraceAnalysis(trace_dir, TP_SIZE, DP_SIZE, PP_SIZE)
    dist_megatron_analysis.analyze()

if __name__ == '__main__':
    main()