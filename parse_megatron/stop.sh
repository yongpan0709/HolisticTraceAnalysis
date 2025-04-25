MPIRUN_PID=$(pgrep -f "mpirun .* python parse_megatron.py")
if [ -n "$MPIRUN_PID" ]; then
    echo "Found MPIRUN process, PID: $MPIRUN_PID"
    kill $MPIRUN_PID
    echo "MPIRUN process killed."
else
    echo "MPIRUN process not found."
fi