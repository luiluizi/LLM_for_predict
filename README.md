# LLM_for_predict
python run_model.py --task traffic_state_pred --model PDFormer --dataset NYCTaxi --config_file NYCTaxi --evaluator TrafficStateGridEvaluator
python run_model.py --task traffic_state_pred --model PDFormer --dataset CHIBike --config_file CHIBike --evaluator TrafficStateGridEvaluator
python run_model.py --task traffic_state_pred --model PDFormer --dataset T-Drive --config_file T-Drive --evaluator TrafficStateGridEvaluator