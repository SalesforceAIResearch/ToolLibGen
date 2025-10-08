# LOG_FILE=/export/home/log/test_cot_$(date +%F_%H%M%S).log
 
# python -u quick_test_all.py --debug --model_name /export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/ --url 8000 --input_data_path /export/home/data/superGPQA_test_data_medicine.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --debug --model_name /export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/ --url 8000 --input_data_path /export/home/data/superGPQA_test_data_science.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --debug --model_name /export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/ --url 8000 --input_data_path /export/home/data/superGPQA_test_data_math.json >> "$LOG_FILE" 2>&1 &

# python -u quick_test_all.py --debug --model_name /export/home/model/hub/models--openai--gpt-oss-20b/snapshots/model/ --url 8001 --input_data_path /export/home/data/superGPQA_test_data_medicine.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --debug --model_name /export/home/model/hub/models--openai--gpt-oss-20b/snapshots/model/ --url 8001 --input_data_path /export/home/data/superGPQA_test_data_science.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --debug --model_name /export/home/model/hub/models--openai--gpt-oss-20b/snapshots/model/ --url 8001 --input_data_path /export/home/data/superGPQA_test_data_math.json >> "$LOG_FILE" 2>&1 &

# python -u quick_test_all.py --debug --model_name gpt-4.1 --url 8001 --input_data_path /export/home/data/superGPQA_test_data_medicine.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --debug --model_name gpt-4.1 --url 8001 --input_data_path /export/home/data/superGPQA_test_data_science.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --debug --model_name gpt-4.1 --url 8001 --input_data_path /export/home/data/superGPQA_test_data_math.json >> "$LOG_FILE" 2>&1 &

# echo "Running. Log: $LOG_FILE"

# LOG_FILE=/export/home/log/test_cot_$(date +%F_%H%M%S).log
 
# python -u quick_test_all.py --model_name /export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/ --url 8000 --input_data_path /export/home/data/superGPQA_test_data_medicine.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --model_name /export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/ --url 8000 --input_data_path /export/home/data/superGPQA_test_data_science.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --model_name /export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/ --url 8000 --input_data_path /export/home/data/superGPQA_test_data_math.json >> "$LOG_FILE" 2>&1 &

# python -u quick_test_all.py --model_name /export/home/model/hub/models--openai--gpt-oss-20b/snapshots/model/ --url 8001 --input_data_path /export/home/data/superGPQA_test_data_medicine.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --model_name /export/home/model/hub/models--openai--gpt-oss-20b/snapshots/model/ --url 8001 --input_data_path /export/home/data/superGPQA_test_data_science.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --model_name /export/home/model/hub/models--openai--gpt-oss-20b/snapshots/model/ --url 8001 --input_data_path /export/home/data/superGPQA_test_data_math.json >> "$LOG_FILE" 2>&1 &

# python -u quick_test_all.py --model_name gpt-4.1 --url 8001 --input_data_path /export/home/data/superGPQA_test_data_medicine.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --model_name gpt-4.1 --url 8001 --input_data_path /export/home/data/superGPQA_test_data_science.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --model_name gpt-4.1 --url 8001 --input_data_path /export/home/data/superGPQA_test_data_math.json >> "$LOG_FILE" 2>&1 &

# echo "Running. Log: $LOG_FILE"

LOG_FILE=/export/home/log/test_cot_$(date +%F_%H%M%S).log
 
python -u quick_test_all.py --model_name /export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/ --url 8000 --input_data_path /export/home/data/dev_phy_data.json >> "$LOG_FILE" 2>&1 &
python -u quick_test_all.py --model_name /export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/ --url 8000 --input_data_path /export/home/data/dev_med_data.json >> "$LOG_FILE" 2>&1 &
python -u quick_test_all.py --model_name /export/home/model/hub/models--Qwen--Qwen3-8B/snapshots/model/ --url 8000 --input_data_path /export/home/data/dev_math_data.json >> "$LOG_FILE" 2>&1 &

python -u quick_test_all.py --model_name /export/home/model/hub/models--openai--gpt-oss-20b/snapshots/model/ --url 8001 --input_data_path /export/home/data/dev_phy_data.json >> "$LOG_FILE" 2>&1 &
python -u quick_test_all.py --model_name /export/home/model/hub/models--openai--gpt-oss-20b/snapshots/model/ --url 8001 --input_data_path /export/home/data/dev_med_data.json >> "$LOG_FILE" 2>&1 &
python -u quick_test_all.py --model_name /export/home/model/hub/models--openai--gpt-oss-20b/snapshots/model/ --url 8001 --input_data_path /export/home/data/dev_math_data.json >> "$LOG_FILE" 2>&1 &

# python -u quick_test_all.py --model_name gpt-4.1 --url 8001 --input_data_path /export/home/data/dev_phy_data.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --model_name gpt-4.1 --url 8001 --input_data_path /export/home/data/dev_med_data.json >> "$LOG_FILE" 2>&1 &
# python -u quick_test_all.py --model_name gpt-4.1 --url 8001 --input_data_path /export/home/data/dev_math_data.json >> "$LOG_FILE" 2>&1 &

echo "Running. Log: $LOG_FILE"