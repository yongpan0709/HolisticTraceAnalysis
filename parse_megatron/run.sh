#!/bin/bash
cd /home/dist/yiyuan/HolisticTraceAnalysis/parse_megatron
# Define variables
NUM_PP_GROUP=4
# If out of memory, we can increase the number of pipeline parallel groups per process.
# This action increases the latency of each process as more data needs to be analyzed.
# However, it can help manage memory constraints by reducing the total number of processes.
NUM_PP_GROUP_PER_PROCESS=1
HOSTFILE="hostfile"

# Calculate NUM_PROCESS using a mathematical formula
# This will round up if there is a remainder, ensuring NUM_PROCESS is at least 1
NUM_PROCESS=$(( (NUM_PP_GROUP + NUM_PP_GROUP_PER_PROCESS - 1) / NUM_PP_GROUP_PER_PROCESS ))

# Determine the number of processes per node
NUM_NODES=$(wc -l < $HOSTFILE)
PROCESSES_PER_NODE=$(( (NUM_PROCESS + NUM_NODES - 1) / NUM_NODES ))

# Run mpirun command with even distribution
mpirun -allow-run-as-root -np $NUM_PROCESS --hostfile $HOSTFILE --map-by ppr:$PROCESSES_PER_NODE:node python parse_megatron.py

# mpirun -allow-run-as-root -np 4 --hostfile ./hostfile --map-by ppr:4:node python parse_megatron.py