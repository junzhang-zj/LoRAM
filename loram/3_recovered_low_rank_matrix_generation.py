import torch
from safetensors.torch import load_file, save_file
import copy
import json
import argparse
import os

def main(args):
    print("Beigin to Recover Pruned Online-LoRA!")
    print(f"recover pruned online lora path {args.peft_model_path}")
    print(f"ckpt info {args.ckpt_interval}")
    ckpt_start = args.ckpt_interval[0]
    ckpt_end = args.ckpt_interval[1]
    ckpt_range = args.ckpt_interval[2]
    ckpt_list = [i for i in range(ckpt_start,ckpt_end,ckpt_range)]
    for ckpt in ckpt_list:
        peft_model_id=f"{args.peft_model_path}/checkpoint-{ckpt}"
        pruned_adapter=load_file(f"{peft_model_id}/adapter_model.safetensors",device='cpu')
        recover_adapter = copy.deepcopy(pruned_adapter)

        ###recovered_lora_mlp
        if args.recovered_lora_mlp:
            pruned_start = args.pruned_layer_interval[0]
            pruned_end = args.pruned_layer_interval[1]
            print(f"begin to recover mlp of trained pruned online-lora, recover info in {args.pruned_path_mlp}")
            for pruned_layer_idx in range(pruned_start,pruned_end):
                pruned_layer_path = f"{args.pruned_path_mlp}/model.layers.{pruned_layer_idx}.mlp.gate_proj.pt"
                pruned_neuro_idx = torch.load(pruned_layer_path,map_location='cpu')['pruned_idx']
                print(f"MLP recover layer idx {pruned_layer_idx}, recover neuro idx shape {pruned_neuro_idx.shape}, recover neuro idx {pruned_neuro_idx}")
                recover_moudles = [
                    f"base_model.model.model.layers.{pruned_layer_idx}.mlp.down_proj.lora_A.weight", #torch.Size([8, 10368])
                    f"base_model.model.model.layers.{pruned_layer_idx}.mlp.gate_proj.lora_B.weight", #torch.Size([10368, 8])
                    f"base_model.model.model.layers.{pruned_layer_idx}.mlp.up_proj.lora_B.weight", #torch.Size([10368, 8])
                    ]
                for recover_moudle_idx,recover_moudle in enumerate(recover_moudles):
                    pruned_weight = pruned_adapter[recover_moudle]
                    if recover_moudle_idx==0:
                        recover_shape_idx = 1
                        recover_len = pruned_neuro_idx.shape[0]+pruned_weight.shape[recover_shape_idx]
                        recover_weight = torch.zeros((pruned_weight.shape[0], recover_len),dtype=torch.float)
                        if args.lora_recover_type == 'normA':
                            r = 8
                            torch.nn.init.normal_(recover_weight, std=1/r)
                        saved_columns = [i for i in range(recover_len) if i not in pruned_neuro_idx]
                        recover_weight[:,saved_columns] = pruned_weight
                    else:
                        recover_shape_idx = 0
                        recover_len = pruned_neuro_idx.shape[0]+pruned_weight.shape[recover_shape_idx]
                        recover_weight = torch.zeros((recover_len,pruned_weight.shape[1]),dtype=torch.float)
                        saved_columns = [i for i in range(recover_len) if i not in pruned_neuro_idx]
                        recover_weight[saved_columns,:] = pruned_weight
                    recover_adapter[recover_moudle] = recover_weight
        
        ###recovered_lora_attn
        if args.recovered_lora_attn:
            pruned_start = args.pruned_layer_interval[0]
            pruned_end = args.pruned_layer_interval[1]
            print(f"begin to recover attn of trained pruned online-lora, recover info in {args.pruned_path_attn}")
            for pruned_layer_idx in range(pruned_start,pruned_end):
                head_dim = 128
                num_heads = 64
                num_key_value_heads = 8

                pruned_layer_path = f"{args.pruned_path_attn}/model.layers.{pruned_layer_idx}.self_attn.k_proj.pt"
                pruned_neuro_idx = torch.load(pruned_layer_path,map_location='cpu')['pruned_idx']
                pruned_neuro_idx_kv=pruned_neuro_idx
                print('pruned_kv_len',len(pruned_neuro_idx_kv))
                pruned_neuro_idx_kv_start = pruned_neuro_idx_kv[::head_dim]

                start_values = pruned_neuro_idx_kv_start * (num_heads//num_key_value_heads)
                end_values = start_values + (head_dim * (num_heads//num_key_value_heads))
                pruned_neuro_idx_qo = torch.cat([torch.arange(start, end) for start, end in zip(start_values, end_values)])
                print('pruned_qo_len',len(pruned_neuro_idx_qo))
                print(f"ATTN recover layer idx {pruned_layer_idx}, recover neuro idx shape {pruned_neuro_idx.shape}, recover neuro idx {pruned_neuro_idx}")
                recover_moudles = [
                    f"base_model.model.model.layers.{pruned_layer_idx}.self_attn.o_proj.lora_A.weight", #torch.Size([8, 5120])
                    f"base_model.model.model.layers.{pruned_layer_idx}.self_attn.q_proj.lora_B.weight", #torch.Size([5120, 8])
                    f"base_model.model.model.layers.{pruned_layer_idx}.self_attn.k_proj.lora_B.weight", #torch.Size([5120, 8])
                    f"base_model.model.model.layers.{pruned_layer_idx}.self_attn.v_proj.lora_B.weight", #torch.Size([5120, 8])
                    ]
                for recover_moudle_idx,recover_moudle in enumerate(recover_moudles):
                    pruned_weight = pruned_adapter[recover_moudle]
                    if recover_moudle_idx==0:
                        recover_shape_idx = 1
                        recover_len = pruned_neuro_idx_qo.shape[0]+pruned_weight.shape[recover_shape_idx]
                        recover_weight = torch.zeros((pruned_weight.shape[0], recover_len),dtype=torch.float) #,dtype=torch.float
                        if args.lora_recover_type == 'normA':
                            r = 8
                            torch.nn.init.normal_(recover_weight, std=1/r)
                        saved_columns = [i for i in range(recover_len) if i not in pruned_neuro_idx_qo]
                        recover_weight[:,saved_columns] = pruned_weight
                    else:
                        recover_shape_idx = 0
                        if 'q_proj' in recover_moudle:
                            recover_len = pruned_neuro_idx_qo.shape[0]+pruned_weight.shape[recover_shape_idx]
                            recover_weight = torch.zeros((recover_len,pruned_weight.shape[1]),dtype=torch.float)
                            saved_columns = [i for i in range(recover_len) if i not in pruned_neuro_idx_qo]
                        else:
                            recover_len = pruned_neuro_idx_kv.shape[0]+pruned_weight.shape[recover_shape_idx]
                            recover_weight = torch.zeros((recover_len,pruned_weight.shape[1]),dtype=torch.float)
                            saved_columns = [i for i in range(recover_len) if i not in pruned_neuro_idx_kv]
                        recover_weight[saved_columns,:] = pruned_weight
                    recover_adapter[recover_moudle] = recover_weight
        if args.recovered_lora_mlp and args.recovered_lora_attn:
            recover_path=f"{peft_model_id}/recover_lora_mlp_attn_{args.lora_recover_type}/"
        elif args.recovered_lora_mlp:
            recover_path=f"{peft_model_id}/recover_lora_mlp_{args.lora_recover_type}/"
        elif args.recovered_lora_attn:
            recover_path=f"{peft_model_id}/recover_lora_attn_{args.lora_recover_type}/"
        else:
            print('save error!')
        if not os.path.exists(recover_path):
            os.mkdir(recover_path)
        file_path = f"{peft_model_id}/adapter_config.json"
        # Load the JSON data from the file
        with open(file_path, 'r') as file:
            adapter_config = json.load(file)
        adapter_config_path = f"{recover_path}/adapter_config.json"
        with open(adapter_config_path, 'w') as file:
            json.dump(adapter_config, file, indent=4)
        save_file(recover_adapter,f"{recover_path}/adapter_model.safetensors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recover Pruned LoRA')
    parser.add_argument('--peft_model_path', type=str, help='prune model name')
    parser.add_argument('--lora_recover_type', type=str, default='zeroA', help='zeroA,normA')
    parser.add_argument('--ckpt_interval', nargs='+', type=int, default=[10,410,10], help="ckpt interval") 
    parser.add_argument('--pruned_layer_interval', nargs='+', type=int, help="pruned layer interval")
    parser.add_argument('--pruned_ratio_mlp', type=str, default='0.75', help='pruned mlp ratio')
    parser.add_argument('--pruned_ratio_attn', type=str, default='0.75', help='pruned attn ratio')
    parser.add_argument('--pruned_path_mlp', type=str, default='None', help='pruned path mlp')
    parser.add_argument('--pruned_path_attn', type=str, default='None', help='pruned path attn')    
    parser.add_argument('--recovered_lora_attn', default=False, action="store_true")
    parser.add_argument('--recovered_lora_mlp', default=False, action="store_true")

   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    main(args)

