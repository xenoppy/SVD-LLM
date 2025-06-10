#!/bin/bash

# example of compressing LLaMA-7B with SVDLLM
FINE_TUNE_PATH="."

echo "[$(date)] Step 1: Start data whitening with 20% compression ratio"
# run data whitening with 20% compression ratio
python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path .
echo "[$(date)] Step 1: Finished data whitening"

## you can also run the following command for low-resource gpu (ex. llama 7b will only need 15G gpu memory to compress) or to compress large-scale llm (ex. llama 65b)
# echo "[$(date)] Step 1 (low resource): Start"
# python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path ./ --run_low_resource
# echo "[$(date)] Step 1 (low resource): Finished"

echo "[$(date)] Step 2: Start SVD compression (step 4)"
python SVDLLM.py --step 4 --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt
echo "[$(date)] Step 2: Finished SVD compression (step 4)"

echo "[$(date)] Step 3: Start LoRA finetune (first half)"
# finetune the compressed model with lora
python utils/LoRA.py --prune_model jeffwan_llama_7b_hf_whitening_only_0.8.pt --data_path yahma/alpaca-cleaned --output_dir $FINE_TUNE_PATH/first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
echo "[$(date)] Step 3: Finished LoRA finetune (first half)"

echo "[$(date)] Step 4: Start merge LoRA (first half)"
python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora $FINE_TUNE_PATH/first_half  --step 4
echo "[$(date)] Step 4: Finished merge LoRA (first half)"

echo "[$(date)] Step 5: Start LoRA finetune (second half)"
python utils/LoRA.py --prune_model $FINE_TUNE_PATH/first_half/merge.pt --data_path yahma/alpaca-cleaned --output_dir $FINE_TUNE_PATH/second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
echo "[$(date)] Step 5: Finished LoRA finetune (second half)"

echo "[$(date)] Step 6: Start merge LoRA (second half, first merge)"
python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora $FINE_TUNE_PATH/first_half --step 4
echo "[$(date)] Step 6: Finished merge LoRA (second half, first merge)"

echo "[$(date)] Step 7: Start merge LoRA (second half, final merge)"
python SVDLLM.py --model_path $FINE_TUNE_PATH/first_half/merge.pt --lora $FINE_TUNE_PATH/second_half --step 4
echo "[$(date)] Step 7: Finished merge LoRA (second half, final merge)"