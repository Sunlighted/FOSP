
<br>
<p align="center">
<h1 align="center"><img align="center" width="6.5%"><strong>FOSP: Fine-tuning Offline Safe Policy through World Models
</strong></h1>
  <p align="center">
    <a href='https://scholar.google.com/citations?user=uSpiLrMAAAAJ&hl=en' target='_blank'>Chenyang Cao</a>&emsp;
    <a href='' target='_blank'>Yucheng Xin</a>&emsp;
    <a href='' target='_blank'>Silang Wu</a>&emsp;
    <a href='' target='_blank'>Longxiang He</a>&emsp;
    <a href='' target='_blank'>Zichen Yan</a>&emsp;
    <a href='https://scholar.google.com/citations?user=kV-h3B8AAAAJ&hl=en&oi=ao' target='_blank'>Junbo Tan</a>&emsp;
    <a href='https://scholar.google.com/citations?user=h9dN_ykAAAAJ&hl=en&oi=ao' target='_blank'>Xueqian Wang</a>&emsp;
    <br>
    Tsinghua University
  </p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2407.04942" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2307.07176-blue?">
  </a>
  <a href="https://sunlighted.github.io/fosp_web/" target='_blank'>
    <img src="https://img.shields.io/badge/Website-&#x1F680-green">
  </a>
</p>



## About
Offline Safe Reinforcement Learning (RL) seeks to address safety constraints by learning from static datasets and restricting exploration. However, these approaches heavily rely on the dataset and struggle to generalize to unseen scenarios safely. In this paper, we aim to improve safety during the deployment of vision-based robotic tasks through online fine-tuning an offline pretrained policy. To facilitate effective fine-tuning, we introduce model-based RL, which is known for its data efficiency. Specifically, our method employs in-sample optimization to improve offline training efficiency while incorporating reachability guidance to ensure safety. After obtaining an offline safe policy, a safe policy expansion approach is leveraged for online fine-tuning. The performance of our method is validated on simulation benchmarks with five vision-only tasks and through real-world robot deployment using limited data. It demonstrates that our approach significantly improves the generalization of offline policies to unseen safety-constrained scenarios. To the best of our knowledge, this is the first work to explore offline-to-online RL for safe generalization tasks
<!-- ![Teaser](assets/teaser.jpg) -->
<div style="text-align: center;">
    <img src="assets/architecture-min.png" alt="Dialogue_Teaser" width=100% >
</div>

We have also open-sourced over **80+** [model checkpoints](https://huggingface.co/Weidong-Huang/SafeDreamer) for 20 tasks. Our codebase supports vector and vision observations. We hope this repository will become a valuable community resource for future research on model-based safe reinforcement learning.

## Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{
fosp,
title={FOSP: Fine-tuning Offline Safe Policy through World Models},
author={Weidong Huang and Jiaming Ji and Borong Zhang and Chunhe Xia and Yaodong Yang},
booktitle={The 13th International Conference on Learning Representations},
year={2025},
url={https://openreview.net/pdf?id=dbuFJg7eaw}
}
```

## Instructions

### Step0: Git clone
```sh
git clone https://github.com/Sunlight/FOSP.git
cd FOSP
```

### Step1: Check version of CUDA and CUDNN (if use GPU)
Due to the strong dependency of JAX on CUDA and cuDNN, it is essential to ensure that the versions are compatible to run the code successfully. Before installing JAX, it is recommended to carefully check the CUDA and cuDNN versions installed on your machine. Here are some methods we provide for checking the versions:

1. Checking CUDA version:
- Use the command `nvcc --version` in the terminal to check the installed CUDA version.

2. Checking cuDNN version:
- Check the version by examining the file names or metadata in the cuDNN installation directory 'cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2'.
- Or you can also use torch to check the CUDNN version 'python3 -c 'import torch;cudnn_version = torch.backends.cudnn.version();print(f"CUDNN Version: {cudnn_version}");print(torch.version.cuda)'

It is crucial to ensure that the installed CUDA and cuDNN versions are compatible with the specific version of JAX you intend to install.
### Step2: Install jax
Here is some subjections for install jax, the new manipulation should be found in [jax](https://github.com/google/jax) documentation. we tested our code in the 0.4.26 version of jax.

### 
```sh
conda create -n fosp python=3.9
conda activate fosp
pip install --upgrade pip
pip install jax==0.4.26
pip install jax-jumpy==1.0.0
# for gpu
pip install jaxlib==0.3.25+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# for cpu
pip install jaxlib==0.4.26
```

### Step3: Install Other Dependencies
```sh
pip install -r requirements.txt
```

### Step4: Install Safetygymnasium
```sh
git clone https://github.com/PKU-Alignment/safety-gymnasium.git
cd safety-gymnasium
pip install -e .
cd ..
```

### Step5: Data
Find the dataset in '.../FOSP/data/'
Download the dataset on google driven (which use be released after camera-ready)
Before training, please check './FOSP/embodied/replay/saver.py' and replace the self.load_dir with the path of the data.

### Step6: Offline Training

```sh
# FOSP-offline:
python FOSP/train.py --configs fosp --method fosp --run.script train_eval_offline --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0

# SafeDreamer-offline:
python FOSP/train_sd.py --configs safedreamer --method safedreamer --run.script train_eval_offline --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0

```

### Step6: Online Fine-tuning

```sh
# FOSP-online fine-tuning:
python FOSP/train.py --configs fosp --method fosp --run.script train_eval_online --run.from_checkpoint /xxx/checkpoint.ckpt  --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.steps 1000000

# SafeDreamer-online fine-tuning:
python  FOSP/train_sd.py --configs safedreamer --method safedreamer --run.script train_eval_online_direct --run.from_checkpoint /xxx/checkpoint.ckpt --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.steps 1000000

```

where checkpoint_path is '/xxx/checkpoint.ckpt'. Before fine-tuning, you should move the offline trained checkpoint to right path.

## Tips

- All configuration options are documented in `configs.yaml`, and you have the ability to override them through the command line.
- If you encounter CUDA errors, it is recommended to scroll up through the error messages, as the root cause is often an issue that occurred earlier, such as running out of memory or having incompatible versions of JAX and CUDA.
- To customize the GPU memory requirement, you can modify the `os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']` variable in the `jaxagent.py`. This allows you to adjust the memory allocation according to your specific needs.


## License
FOSP is released under Apache License 2.0.



## Acknowledgements
- [SafeDreamer](https://github.com/PKU-Alignment/SafeDreamer): Our codebase is built upon SafeDreamer.
- [DreamerV3](https://github.com/danijar/dreamerv3)
