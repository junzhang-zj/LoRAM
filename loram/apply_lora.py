import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def apply_lora(args):
    print("args",args)
    print(f"Loading the base model from {args.model_name_or_path}")
    if args.device_map == None:
        device_map = 'auto'
    else:
        device_map = args.device_map
    print(device_map)
    
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,  torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_8bit=args.quantization, device_map=device_map
    )
    print(base)
    
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if  args.lora_path == 'None':
        print('save int8 model')
    else:
        print(f"Loading the LoRA adapter from {args.lora_path}")
        torch.cuda.empty_cache()
        lora_model = PeftModel.from_pretrained(
            base,
            args.lora_path,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        print(lora_model)
        print("Applying the LoRA")
        model = lora_model.merge_and_unload()

    print(f"Saving the target model to {args.output_path}")
    model.save_pretrained(args.output_path)
    base_tokenizer.save_pretrained(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply LoRA to a base model and save the result.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the base model or model name")
    parser.add_argument("--quantization",  default=False, action="store_true",help="quantize")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the modified model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--device_map", type=str, default=None, help="device_map")

    args = parser.parse_args()

    apply_lora(args)
