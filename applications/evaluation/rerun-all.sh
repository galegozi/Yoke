#!/bin/bash

# Path to the script you want to run
SCRIPT_TO_RUN="makeallgraphs.sh"

# Infinite loop
while true; do
    echo "starting at $(date)"
    # Run the target script
    bash "$SCRIPT_TO_RUN"
    echo "ending at $(date)"

    # Wait for 5 minutes (300 seconds)
    sleep 300
done