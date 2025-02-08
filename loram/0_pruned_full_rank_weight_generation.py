import argparse
import torch
from transformers import AutoTokenizer
from modeling_llama3_70b_structured_pruning import LlamaForCausalLM

def main(args):
    base_model = args.base_model
    model = LlamaForCausalLM.from_pretrained(base_model,
                                             torch_dtype=torch.bfloat16,
                                             device_map="cpu")
    tokenizer_path = args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    pruned_path_mlp = args.pruned_path_mlp
    pruned_path_attn = args.pruned_path_attn
    pruned_start = args.pruned_start
    pruned_end = args.pruned_end

    print(f"begin to prune mlp, prune info in {pruned_path_mlp}")
    for pruned_layer_idx in range(pruned_start, pruned_end):
        pruned_layer_path = f"{pruned_path_mlp}/model.layers.{pruned_layer_idx}.mlp.gate_proj.pt"
        pruned_neuro_idx = torch.load(pruned_layer_path, map_location='cpu')['pruned_idx']
        print(f"MLP prune layer idx {pruned_layer_idx}, prune neuro idx shape {pruned_neuro_idx.shape}, prune neuro idx {pruned_neuro_idx}")
        model.model.layers[pruned_layer_idx].mlp.prune(pruned_neuro_idx)

    print(f"begin to prune attn, prune info in {pruned_path_attn}")
    for pruned_layer_idx in range(pruned_start, pruned_end):
        pruned_layer_path = f"{pruned_path_attn}/model.layers.{pruned_layer_idx}.self_attn.k_proj.pt"
        pruned_neuro_idx = torch.load(pruned_layer_path, map_location='cpu')['pruned_idx']
        print(f"ATTN prune layer idx {pruned_layer_idx}, prune neuro idx shape {pruned_neuro_idx.shape}, prune neuro idx {pruned_neuro_idx}")
        model.model.layers[pruned_layer_idx].self_attn.prune(pruned_neuro_idx)

    output_dir = args.output_dir
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune and save Llama model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--pruned_path_mlp", type=str, required=True, help="Path to the pruned MLP weights")
    parser.add_argument("--pruned_path_attn", type=str, required=True, help="Path to the pruned ATTN weights")
    parser.add_argument("--pruned_start", type=int, default=4, help="Start index for pruning")
    parser.add_argument("--pruned_end", type=int, default=78, help="End index for pruning")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the pruned model")

    args = parser.parse_args()
    main(args)