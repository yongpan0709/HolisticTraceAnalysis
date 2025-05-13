from hta.distribute_trace_analysis import DistributedMegatronTraceAnalysis
        

def main():
    TP_SIZE = 1
    PP_SIZE = 2
    DP_SIZE = 4
    trace_dir = '/home/dist/HolisticTraceAnalysis/ds-count2-tp1-pp2-dp4'
    dist_megatron_analysis = DistributedMegatronTraceAnalysis(trace_dir, TP_SIZE, DP_SIZE, PP_SIZE)
    dist_megatron_analysis.analyze()

if __name__ == '__main__':
    main()