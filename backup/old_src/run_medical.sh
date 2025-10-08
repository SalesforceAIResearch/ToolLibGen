LOG_FILE="/export/home/log/create_tools_$(date +%F_%H%M%S).log"
python -u extract_reusable_tools_with_agentic_way_v2.py --file_path ReasonMed_processed_20250915_051520.json --save_path ReasonMed_tools_100k.json --remote_mode > /export/home/log/create_tools_$(date +%F_%H%M%S).log 2>&1 &
echo $! > /export/home/log/create_tools.pid
echo "Running. Log: $LOG_FILE"



# python -u clustering.py --input_file math_103k_tools_20250912_070833.json --remote_mode > /export/home/log/clustering_$(date +%F_%H%M%S).log 2>&1 &

# python -u aggregate_tools_v3.py --file /export/home/data/math_103k_tools_20250912_070833_assigned_tools_20250915_003847.json > /export/home/log/aggregate_$(date +%F_%H%M%S).log 2>&1 &
# echo $! > /export/home/log/aggregate.pid
# echo "Running. Log: $LOG_FILE"

