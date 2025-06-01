FINE_TUNE_PATH="."
python utils/LoRA.py --prune_model $FINE_TUNE_PATH/first_half/merge.pt --data_path yahma/alpaca-cleaned --output_dir $FINE_TUNE_PATH/second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
echo "Lora fine-tuning for second half completed."
python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora $FINE_TUNE_PATH/first_half  --step 4
echo "SVDLLM step 4 for first half completed."
python SVDLLM.py --model_path $FINE_TUNE_PATH/first_half/merge.pt --lora $FINE_TUNE_PATH/second_half --step 4
echo "SVDLLM step 4 for second half completed."