import os
import argparse
import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
from modeling_llama3_70b_structured_pruning_initialize import LlamaForCausalLM
from peft import LoraConfig, get_peft_model

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
    print(f"begin to set wandb project {args.wandb_project} \n name {args.wandb_name} \n output {output_dir}")

    # Load Full Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print('base path', args.base_model)
    model = LlamaForCausalLM.from_pretrained(args.base_model,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             device_map="auto",
                                             quantization_config=bnb_config)
    print('after online prune model', model)
    print_trainable_parameters(model)

    # Prepare For LoRA
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print('online lora with trained pruned offline model', model)
    print_trainable_parameters(model)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Load Dataset
    ### out-of-domain test set
    preprocess_data_alpaca = torch.load(args.eval_data_path)
    _, val_data_alpaca = preprocess_data_alpaca['train'], preprocess_data_alpaca['val']

    if 'openhermes' in args.train_data_path:
        print(f"openhermes in data path {args.train_data_path}")
        preprocess_data_openhermes = torch.load(args.train_data_path)
        train_data_openhermes, val_data_openhermes = preprocess_data_openhermes['train'], preprocess_data_openhermes['val']
        train_data = train_data_openhermes
        val_data = {"alpaca": val_data_alpaca, "openhermes": val_data_openhermes}
    elif 'openorca' in args.train_data_path:
        print(f"openorca in data path {args.train_data_path}")
        preprocess_data_openorca = torch.load(args.train_data_path)
        train_data_openorca, val_data_openorca = preprocess_data_openorca['train'], preprocess_data_openorca['val']
        train_data = train_data_openorca.train_test_split(train_size=0.50, seed=42)['train']
        val_data = {"alpaca": val_data_alpaca, "openorca": val_data_openorca}
    elif 'gsm8k' in args.train_data_path:
        print(f"gsm8k in data path {args.train_data_path}")
        preprocess_data_gsm8k = torch.load(args.train_data_path)
        train_data_gsm8k, val_data_gsm8k = preprocess_data_gsm8k['train'], preprocess_data_gsm8k['val']
        train_data = train_data_gsm8k
        val_data = {"alpaca": val_data_alpaca, "gsm8k": val_data_gsm8k}
    else:
        print("train data error!")

    # Training Parameters
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
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
            eval_steps=50,
            save_steps=50,
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
    parser.add_argument('--base_model', type=str, default="./", help='base model name')
    parser.add_argument('--train_data_path', type=str, default="./", help='train data path')
    parser.add_argument('--eval_data_path', type=str, default="./", help='eval data path')
    parser.add_argument('--output_dir', type=str, default="./", help='output directory')
    
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=512, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj,lm_head", help='lora target modules')
    
    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
    
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="LoRAM")
    parser.add_argument('--wandb_name', type=str, default="None")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    main(args)