LOG_FILE="/export/home/log/aggregate_$(date +%F_%H%M%S).log"
python -u aggregate_tools_v3.py --file /export/home/data/Nemotron_science_data_tools_saved_all.json > "$LOG_FILE" 2>&1 &
echo $! > /export/home/log/aggregate.pid
echo "Running. Log: $LOG_FILE"
