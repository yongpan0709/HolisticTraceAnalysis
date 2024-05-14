#!/bin/bash

# Usage function to display help
usage() {
  echo "Usage: $0 <hostfile> <script_path>"
  echo "Example: $0 hosts.txt /path/to/your/script.sh"
  exit 1
}

# Check if minimum number of arguments is provided
if [ $# -ne 2 ]; then
  usage
fi

HOSTFILE=$1
SCRIPT_PATH=$2

# Check if the host file exists
if [ ! -f "$HOSTFILE" ]; then
  echo "Error: Hostfile '$HOSTFILE' does not exist."
  exit 1
fi

# Check if the script file exists
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Error: Script file '$SCRIPT_PATH' does not exist."
  exit 1
fi

# Calculate the number of active nodes
NUM_NODES=$(grep -vE '^#|^$' "$HOSTFILE" | wc -l)
echo "NUM_NODES: $NUM_NODES"

# Read hosts into an array
readarray -t HOSTLIST < <(grep -vE '^#|^$' "$HOSTFILE" | awk '{print $1}')

# Transfer and execute the script on each host in parallel
for host in "${HOSTLIST[@]}"; do
  echo "Transferring script to $host..."
  scp "$SCRIPT_PATH" "${host}:/tmp/" &
done
wait # Wait for all scp transfers to complete

for host in "${HOSTLIST[@]}"; do
  echo "Executing script on $host..."
  ssh "$host" "bash /tmp/$(basename "$SCRIPT_PATH")" &
done
wait # Wait for all commands to complete

echo "All scripts executed on all hosts."
