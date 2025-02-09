# LoRAM (Train Large, Infer Small)

<u>*A Simple and Effective Memory-Efficient LoRA Training Scheme Using LLaMA-3.1-70B as an Example*</u>

## 0: Pruned Full-Rank Weight Generation

```bash
python 0_pruned_full_rank_weight_generation.py \
--base_model /path/to/base/model \
--tokenizer_path /path/to/tokenizer \
--pruned_path_mlp /path/to/pruned/mlp/weights \
--pruned_path_attn /path/to/pruned/attn/weights \ 
--output_dir /path/to/output/directory
```

## 1: Pruned Full-Rank Weight Alignment

```bash
python 1_pruned_full_rank_weight_alignment.py \
--base_model /path/to/base/model \
--peft_model /path/to/peft/model \
--data_path yahma/alpaca-cleaned \
--output_dir /path/to/output/directory \
--fineweb_data_path /path/to/fineweb/data \
--mathweb0_data_path /path/to/mathweb0/data \
--mathweb1_data_path /path/to/mathweb1/data \
--wandb_project LoRAM \
--wandb_name "None"
```

## 2: Pruned Low-Rank Matrix Training (LoRAM & QLoRAM)

```bash
python 2_pruned_low_rank_matrix_training.py \
--base_model /path/to/base/model \
--train_data_path /path/to/train/data \
--eval_data_path /path/to/eval/data \
--output_dir /path/to/output/directory \
--wandb_project LoRAM \
--wandb_name "None" \
--resume_from_checkpoint /path/to/checkpoint
```

## 3: Recovered Low-Rank Matrix Generation

```bash
python -u 3_recovered_low_rank_matrix_generation.py \
--ckpt_interval 50 400 50 \
--pruned_layer_interval 4 78 --recovered_lora_attn \
--pruned_ratio_attn '0.85' --recovered_lora_mlp \
--pruned_ratio_mlp '0.85' \
--pruned_path_attn "/LoRAM(Anonymous)/llama3-70b_prune_param_first_0.85" \
--pruned_path_mlp "/LoRAM(Anonymous)/llama3-70b_prune_param_first_0.85" \
--peft_model_path "./LoRAM_Model" \
>> ./Recover_LLaMA3.1-70B.log 2>&1 &
```

## 4: Recovered Low-Rank Matrix Inference (PPL & Downstream Tasks)

```bash
python 4_0_recover_low_rank_matrix_inference_ppl.py \
--base_model /path/to/base/model \
--train_data_path /path/to/train/data \
--eval_data_path /path/to/eval/data \
--output_dir /path/to/output/directory \
--wandb_project LoRAM \
--wandb_name "None" \
--resume_from_checkpoint /path/to/checkpoint
```

```bash
bash 4_1_recovered_low_rank_matrix_inference_downstream_tasks.sh
```
