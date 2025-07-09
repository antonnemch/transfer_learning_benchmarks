#!/bin/bash
cd "$HOME/scratch/cnn_project"
# Usage: ./scripts/watch_log.sh <jobid>

if [ -z "$1" ]; then
    echo "Usage: $0 <jobid>"
    exit 1
fi

JOBID=$1
LOGFILE="logs/cnn_gridsearch-${JOBID}.out"

echo "Waiting for $LOGFILE to be created..."
while [ ! -f "$LOGFILE" ]; do
    sleep 1
done

echo "Log file found. Tailing now..."
tail -f "$LOGFILE"
