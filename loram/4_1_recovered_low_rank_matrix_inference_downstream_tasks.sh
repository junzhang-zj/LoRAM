#!/bin/bash

base_path="./LoRAM_Model/gsm8k"
cp_path_template="checkpoint-%d"

start_checkpoint=50
end_checkpoint=550
step=50

sub_path="./LoRAM_Model"

paths=()
for ((checkpoint_num=start_checkpoint; checkpoint_num<=end_checkpoint; checkpoint_num+=step)); do
    cp_path=$(printf "$cp_path_template" "$checkpoint_num")
    path="$base_path/$sub_path/$cp_path"
    paths+=("$path")
done

for path in "${paths[@]}"; do
    echo "$path"
done


for path in "${paths[@]}"; do
    if [[ $path == *"8B"* ]]; then
        pretrained_path="./model/meta-llama/Llama-3.1-8B"
        parallelize=4
        CUDA_DEVICES="4,5,6,7"
    elif [[ $path == *"70B"* ]]; then
        pretrained_path="./model/meta-llama/Llama-3.1-70B"
        parallelize=4
        CUDA_DEVICES="4,5,6,7"
    fi
    # 设置 CUDA 可见设备
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

    if [[ $path == *"-random"* ]] || [[ $path == *"-structured"* ]]; then
        path="$path/recover_lora_mlp_attn_zeroA"
    fi

    if [[ "$path" == *"NoFT"* ]]; then
        merge_path="$pretrained_path"
    else
        merge_path="$path/merge_lora"
        if [ ! -d "$merge_path" ] || [ -z "$(ls -A $merge_path)" ]; then
            echo "Output path is empty or does not exist. Running apply_lora.py..."
            apply_lora_script="./LoRAM(Anonymous)/apply_lora.py"
            python $apply_lora_script --model_name_or_path $pretrained_path --output_path $merge_path --lora_path $path
        else
            echo "Output path is not empty. Skipping apply_lora.py."
        fi
    fi

    folders=("GSM8K" "HumanEval" "MATHQA" "CSR")

    for folder in "${folders[@]}"; do
        if [ ! -d "$path/$folder" ]; then
            echo "Folder $folder does not exist in $path. Creating it now."
            mkdir -p "$path/$folder"
        else
            echo "Folder $folder already exists in $path."
        fi
    done

    # GSM8K-COT Evaluation
    lm_eval --model vllm \
    --model_args pretrained=$merge_path,tensor_parallel_size=$parallelize,dtype=bfloat16,gpu_memory_utilization=0.80,data_parallel_size=1 \
    --tasks gsm8k_cot \
    --batch_size auto --num_fewshot 8 --log_samples \
    --output_path $path/GSM8K \
    --use_cache $path/GSM8K > $path/GSM8K/eval_cot_8shot.log 2>&1

    # CSR Evaluation
    lm_eval --model vllm \
    --model_args pretrained=$merge_path,tensor_parallel_size=$parallelize,dtype=bfloat16,gpu_memory_utilization=0.60,data_parallel_size=1 \
    --tasks arc_challenge,piqa,hellaswag,arc_easy,winogrande,openbookqa \
    --batch_size auto --num_fewshot 1 --log_samples \
    --output_path $path/CSR \
    --use_cache $path/CSR > $path/CSR/eval_1shot.log 2>&1
    
    compute_csr_script="./LoRAM(Anonymous)/compute_csr.py"
    python $compute_csr_script $path/CSR/eval_1shot.log

    # MATHQA Evaluation
    lm_eval --model vllm \
    --model_args pretrained=$merge_path,tensor_parallel_size=$parallelize,dtype=bfloat16,gpu_memory_utilization=0.60,data_parallel_size=1 \
    --tasks mathqa \
    --batch_size auto --num_fewshot 1 --log_samples \
    --output_path $path/MATHQA \
    --use_cache $path/MATHQA > $path/MATHQA/eval_1shot.log 2>&1

    
    # HumanEval Evaluation
    code_eval_script="./code-eval/eval_llama-2-13b_noft.py"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -u $code_eval_script \
    --num_samples_per_task 10 \
    --base_model_id $merge_path \
    --peft_model_path None \
    --output_path $path/HumanEval/eval_10shots.jsonl \
    > $path/HumanEval/eval_10shots.log 2>&1

    evaluate_functional_correctness $path/HumanEval/eval_10shots.jsonl \
    > $path/HumanEval/score_10shots.log 2>&1

done
