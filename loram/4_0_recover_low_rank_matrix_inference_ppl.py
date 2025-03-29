import os
import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model
import wandb

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
    
    args.wandb_name = os.path.basename(args.wandb_name)  

    output_dir = args.output_dir 


    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,          
        dir=output_dir,                
        config=args,
        mode="online",
        settings=wandb.Settings(start_method="fork") 
    )

    os.environ["WANDB__SERVICE_WAIT"] = "600"

    print(f"""
    WandB initialize finish：
    Project: {args.wandb_project}
    Name: {wandb.run.name}  
    Output Dir: {output_dir}
    """)

    # Load Full Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print('recoverd loram trained base model path', args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             device_map="auto",
                                            #  quantization_config=bnb_config
                                             )
    # Prepare For LoRA
    if args.peft_model != None:
        print('loading lora path',args.peft_model)
        model.load_adapter(args.peft_model)
        print('pruned lora with pruned model',model)
    else:
        print('None load lora')
    model.eval()
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
    print(trainer.evaluate(val_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')
    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="./", help='base model name')
    parser.add_argument('--peft_model', type=str, default=None, help='peft model name') 
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
