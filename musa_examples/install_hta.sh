#!/bin/bash
HOSTFILE=./hostfile
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)

for host in ${hostlist[@]}; do
  echo $host
  ssh -f -n $host "cd /mnt/seed17/001688/huayp/1000b-train/HolisticTraceAnalysis; pip install -r requirements.txt;pip install -e ." 
  # ssh -f -n $host "apt update; apt install -y tmux"  
  echo $cmd
  ((COUNT++))
done
