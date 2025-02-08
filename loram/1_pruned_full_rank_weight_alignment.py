import os
import argparse
import torch
import transformers
from transformers import AutoTokenizer, set_seed
from datasets import interleave_datasets
from modeling_llama3_70b_structured_pruning_initialize import LlamaForCausalLM

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for param_name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def create_device_map(num_gpus=8, num_layers=80):
    device_map = {}

    for layer in range(4):
        device_map[f'model.layers.{layer}'] = layer
        device_map[f'model.layers.{layer}.self_attn.rotary_emb'] = layer

    middle_layers = list(range(4, 78))
    for i, layer in enumerate(middle_layers):
        gpu_id = i % num_gpus  
        device_map[f'model.layers.{layer}'] = gpu_id
        device_map[f'model.layers.{layer}.self_attn.rotary_emb'] = gpu_id

    for layer in range(78, 80):
        device_map[f'model.layers.{layer}'] = 5
        device_map[f'model.layers.{layer}.self_attn.rotary_emb'] = 6

    device_map['model.embed_tokens'] = 0
    device_map['model.norm'] = 7
    device_map['lm_head'] = 7
    device_map['model.rotary_emb'] = 7
    return device_map

def main(args):
    # Set Seed & WanDB
    print(f"begin to set seed {args.seed}")
    set_seed(args.seed)
    output_dir = f"{args.output_dir}/{args.wandb_name}"
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = args.wandb_name
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.environ["WANDB_DIR"] = output_dir
    os.environ["WANDB_MODE"] = "online"

    device_map = create_device_map()
    print(device_map)

    # Load Full Model
    torch.nn.Linear.reset_parameters = lambda x: None
    model = LlamaForCausalLM.from_pretrained(args.base_model,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             device_map=device_map)
    model.train()
    print_trainable_parameters(model)

    print('freeze lm head before')
    for name, param in model.named_parameters():
        if 'lm_head' in name:
            print(f'Freezing parameter: {name}')
            param.requires_grad = False

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    preprocess_data_fineweb = torch.load(args.fineweb_data_path, weights_only=False)
    train_data_fineweb, val_data_fineweb = preprocess_data_fineweb['train'], preprocess_data_fineweb['val']

    preprocess_data_mathweb0 = torch.load(args.mathweb0_data_path, weights_only=False)
    train_data_mathweb0, val_data_mathweb0 = preprocess_data_mathweb0['train'], preprocess_data_mathweb0['val']

    preprocess_data_mathweb1 = torch.load(args.mathweb1_data_path, weights_only=False)
    train_data_mathweb1, val_data_mathweb1 = preprocess_data_mathweb1['train'], preprocess_data_mathweb1['val']

    stopping_strategy = "all_exhausted"
    all_datasets = [train_data_fineweb, train_data_mathweb0, train_data_mathweb1]
    interleave_probs = [0.50, 0.25, 0.25]
    train_data = interleave_datasets(all_datasets, interleave_probs, stopping_strategy=stopping_strategy)
    val_data = {"fineweb": val_data_fineweb, "mathweb": val_data_mathweb0}

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            optim="adamw_torch",
            lr_scheduler_type='cosine',
            bf16=True,
            logging_steps=1,
            logging_first_step=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=10,
            save_steps=200,
            output_dir=output_dir,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')
    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="/home/notebook/data/group/model_hub/llama2/Llama-2-70b-hf", help='base model name')
    parser.add_argument('--peft_model', type=str, default=None, help='peft model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory')
    
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=512, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # LoRA Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj,lm_head", help='lora target modules')
    
    # pruning hyperparameters
    parser.add_argument('--pruned_model_mlp', default=False, action="store_true")
    parser.add_argument('--pruned_model_attn', default=False, action="store_true")
    parser.add_argument('--pruned_ratio_mlp', type=str, default='0.75', help='pruned ratio')
    parser.add_argument('--pruned_ratio_attn', type=str, default='0.75', help='pruned ratio')
    parser.add_argument('--pruned_path_mlp', type=str, default='None', help='pruned path mlp')
    parser.add_argument('--pruned_path_attn', type=str, default='None', help='pruned path attn')
   
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
    
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="LoRAM")
    parser.add_argument('--wandb_name', type=str, default="None")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")
    
    # Data paths
    parser.add_argument('--fineweb_data_path', type=str, default="./data/fineweb/cache/preprocess_clm_000_llama3.1.bin", help='path to fineweb data')
    parser.add_argument('--mathweb0_data_path', type=str, default="./data/openmathweb/cache/preprocess_clm_000_llama3.1.bin", help='path to mathweb0 data')
    parser.add_argument('--mathweb1_data_path', type=str, default="./data/openmathweb/cache/preprocess_clm_001_llama3.1.bin", help='path to mathweb1 data')

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    main(args)