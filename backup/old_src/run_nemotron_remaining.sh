LOG_FILE="/export/home/log/aggregate_$(date +%F_%H%M%S).log"
python -u aggregate_tools_final.py --file /export/home/data/ReasonMed_tools_saved_all.json > "$LOG_FILE" 2>&1 &
echo $! > /export/home/log/aggregate.pid
echo "Running. Log: $LOG_FILE"