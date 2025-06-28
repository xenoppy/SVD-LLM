
## ✨Roadmap
This is Weichu.
I am working on the following tasks, please stay tuned!

- [X] Support SVD_LLM for Llama3.1-1B.
- [ ] Support LoRA for SVD_LLM compressed Llama3.1-1B.


## Quick Start
### Installation
There are some difference 
```
conda create -n compress_llama3 python=3.10 -y
conda activate compress_llama3
```
Clone and navigate to the repository
```
git clone https://github.com/xenoppy/SVD-LLM.git
```
Install requirements.txt
```
cd SVD-LLM
pip install -r requirements.txt
```


### Quick Example
download a SVD_LLM compressed Llama3.2-1B
```
wget https://huggingface.co/wyang338/Llama-3.2-1B-SVDed/resolve/main/meta_llama_Llama_3.2_1B_whitening_only_1.0.pt
```
Then run those commands to do inference things
```
python SVDLLM.py --step 7 --model_path meta_llama_Llama_3.2_1B_whitening_only_1.0.pt
```

    
## Step-by-Step Instructions of SVD-LLM
    
### 1. Truncation-Aware Data Whitening + SVD Compression
Under the low compression ratio (recommended ratio <= 0.3), we first run the data whitening of the LLM and saved the weight along with the whitening information.
```
python SVDLLM.py \
--step 1  \
--ratio COMPRESSION_RATIO \
--model HUGGINGFACE_MODEL_REPO \
--whitening_nsamples WHITENING_SAMPLE_NUMBER \
--dataset WHITENING_DATASET \
--seed SAMPLING_SEED \
--model_seq_len MODEL_SEQ_LEN \
--save_path WHITENING_INFO_SAVING_PATH
```

<!-- To compress LLM with larger size, or to run the compression under the resource-constraint platform, we can add `--run_low_resource` to the command. -->


### 2. Parameter Update with Sequential Low-rank Approximation
We first update the compressed weight matrix U and then V with LoRA fine-tuning.
```
python LoRA.py \
--prune_model COMPRESSED_MODEL_PATH \
--data_path yahma/alpaca-cleaned \
--output_dir LORA_OUTPUT_PATH  \
--lora_r 8 \
--num_epochs 2 \
--learning_rate 1e-4 \
--batch_size 64
```

### 3. SVD-LLM + GPTQ
SVD-LLM can also be integrated with quantization methods to achieve a better compression. Here is the example of how to integrate SVD-LLM (20% compression ratio) with GPTQ-4bit to compress LLaMA-7B
```
bash svdllm_gptq.sh
```

### 4. Evaluation
- Perplexity Evaluation:
```
python SVDLLM.py \
--step 4 \
--model_path COMPRESSD_MODEL_SAVING_PATH  \
```
We use the same c4 dataset as in [SparseGPT](https://github.com/IST-DASLab/sparsegpt). Since the original dowload link is invalid, please directly download it from this [link](https://drive.google.com/drive/folders/123Id1MkZVsKySGy_sMO4RgiJKrtPcvUp?usp=sharing) and add the two json files under the `utils/.` folder.
- Efficiency Evaluation:
```
python SVDLLM.py \
--step 5 \
--model_path COMPRESSD_MODEL_SAVING_PATH  \
```
## Citation
If you find this work useful, please cite
```
@inproceedings{wang2025svdllm,
  title={{SVD}-{LLM}: Truncation-aware Singular Value Decomposition for Large Language Model Compression},
  author={Xin Wang and Yu Zheng and Zhongwei Wan and Mi Zhang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://openreview.net/forum?id=LNYIUouhdt}
}
```
