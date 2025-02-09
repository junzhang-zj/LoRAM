<div align="center">
<h1><img src="assets/loram.png" height="40px" align="top"/> Train Small, Infer Large: <br> Memory-Efficient LoRA Training for LLMs</h1>

<h2 style="background: linear-gradient(to right, #ff416c, #ff4b2b); -webkit-background-clip: text; color: transparent; font-weight: bold;">
8～12× Parameter Reduction ↓
</h2>
LoRAM is a memory-efficient LoRA training method for cost-effective performance gains by <br> training low-rank matrices on a pruned model and merging them for inference on the original model.

<b><a href="https://github.com/junzhang-zj">Jun Zhang</a></b><sup>1</sup>,
<b><a href="https://github.com/LorrinWWW">Jue Wang</a></b><sup>1</sup>,
<b><a href="https://github.com/longaspire">Huan Li</a></b><sup>1</sup>,
<b><a href="">Lidan Shou</a></b><sup>1</sup>,
<b><a href="">Ke Chen</a></b><sup>1</sup>,
<b><a href=""><br>Yang You</a></b><sup>2</sup>,
<b><a href="">Guiming Xie</a></b><sup>3</sup>,
<b><a href="">Xuejian Gong</a></b><sup>3</sup>,
<b><a href="">Kunlong Zhou</a></b><sup>3</sup>

<sup>1</sup> Zhejiang University,   <sup>2</sup> National University of Singapore, <sup>3</sup> OPPO AI Center  
</div>  

---

## 🔥 Features  
✅ Train LoRA on a **pruned model** to reduce memory footprint  
✅ Recover LoRA for high-qulity **full model** inference  

---

## 🛠 Installation  

Clone the repository and install dependencies:  
```bash
git clone https://github.com/your-repo/LoRAM.git
cd LoRAM
pip install -r requirements.txt
```

---

## 🙌 Acknowledgments  
LoRAM leverages tools from [LLM-Pruner](https://github.com/horseee/LLM-Pruner) and [SparseGPT](https://github.com/IST-DASLab/sparsegpt).
<br>We appreciate the contributions to the research community.

---

## 📖 Citation
If you find the resources in this repository useful, please cite our paper:

```
@inproceedings{
zhang2025train,
title={Train Small, Infer Large: Memory-Efficient Lo{RA} Training for Large Language Models},
author={Jun Zhang and Jue WANG and Huan Li and Lidan Shou and Ke Chen and Yang You and Guiming Xie and Xuejian Gong and Kunlong Zhou},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=s7DkcgpRxL}
}
```
---
