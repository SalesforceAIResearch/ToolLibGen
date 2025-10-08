# LOG_FILE="/export/home/log/create_tools_$(date +%F_%H%M%S).log"
# python -u extract_reusable_tools_with_agentic_way_v2.py --file_path math_103k_processed.json --save_path math_103k_tools.json --remote_mode > /export/home/log/create_tools_$(date +%F_%H%M%S).log 2>&1 &
# echo $! > /export/home/log/create_tools.pid
# echo "Running. Log: $LOG_FILE"



# python -u clustering.py --input_file math_103k_tools_20250912_070833.json --remote_mode > /export/home/log/clustering_$(date +%F_%H%M%S).log 2>&1 &

LOG_FILE="/export/home/log/aggregate_$(date +%F_%H%M%S).log"
python -u aggregate_tools_v4.py --file /export/home/data/math_103k_tools_saved_all.json > "$LOG_FILE" 2>&1 &
echo $! > /export/home/log/aggregate.pid
echo "Running. Log: $LOG_FILE"


# LOG_FILE="/export/home/log/aggregate_$(date +%F_%H%M%S).log"
# python -u resume_from_blueprints.py --root-output-dir /export/home/temp_lib/lib_20250917_080639 --max-threads 50 > "$LOG_FILE" 2>&1 &
# echo $! > /export/home/log/aggregate.pid
# echo "Running. Log: $LOG_FILE"

