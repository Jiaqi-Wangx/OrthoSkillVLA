# OrthoSkillVLA: Continual Skill Learning via Gradient-Informed Skill Subspace Adaptation (PRCV 2026)


Pretrained Vision-Language-Action models provide a strong foundation for robot learning, but sequentially adapting them to diverse skills can perturb the representations and velocity mappings used by previous skills, leading to catastrophic forgetting.
Architecture-based approaches improve retention by isolating skills but lead to increased inference footprint.
Recent subspace-constrained methods restrict parameter updates in an orthogonal subspace to minimize interference but impose a unified constraint on the entire model.
We analyze the distinct roles of internal VLA components and identify two VLA-specific challenges.
First, the VLM maintains broad semantic representations, making it vulnerable to capacity exhaustion, whereas the ActionHead refines semantics into localized velocity patterns that are highly sensitive to perturbations.
Second, the final velocity decoder serves as a readout layer.
Freezing it forms an output-stage expressivity bottleneck, while updating it risks overwriting previous velocity mappings.
To this end, we propose **OrthoSkillVLA**, a parameter-efficient framework for continual skill learning in pretrained VLA models without demonstration replay.
Given the representation heterogeneity, we impose separate subspace constraints on the VLM and ActionHead, preserving reusable semantic capacity while protecting localized velocity patterns.
For the output layer, we introduce a lightweight feature-aware MoE decoder, where each skill is allocated a compact expert and a training-free router selects the expert according to feature-space affinity.
Extensive simulated and real-world evaluations, together with ablations, demonstrate that OrthoSkillVLA better preserves prior skills while acquiring new ones.


## Installation

**Requirements:** Python ≥ 3.10, RTX 4090 GPU (training and simulation eval).

```bash
git clone https://github.com/Jiaqi-Wangx/OrthoSkillVLA.git
cd OrthoSkillVLA
uv sync --all-groups
```

## Reproduction

### 1. Pretrained Model and Dataset

[Pretrained X-VLA model weights](https://huggingface.co/Jiaqi-Wangx/pretrained_xvla) 

[Libero dataset in LeRobot v3.0 format](https://huggingface.co/datasets/Jiaqi-Wangx/libero_90_xvla)


### 2. About Skill Splits

The skill splits can be found at [`sim_eval/libero/libero_skills.json`](sim_eval/libero/libero_skills.json) 

| Skill | LIBERO `task_ids` | LeRobot `lerobot_task_ids` |
|-------|-------------------|----------------------------|
| `open_close` | 22, 6, 28, 35 | 11, 8, 69, 57 |
| `pick_place` | 12, 46, 74, 85 | 45, 0, 42, 10 |
| `turn` | 20, 39 | 65, 63 |

### 3. Continual Skill Learning

**Note:** The released training script is intended for single-process, single-GPU reproduction. It launches `accelerate` with `--num_processes 1`; multi-process / multi-GPU training is not supported in this release.

Edit [`scripts/train_orthoskillvla.sh`](scripts/train_orthoskillvla.sh):

| Variable | Description |
|----------|-------------|
| `load_from` | Path to an X-VLA pretrained model |
| `base_output_dir` | Root directory for experiment outputs |
| `repo_root` | Local path to the converted LeRobot dataset |
| `repo_id` | LeRobot dataset repo id |

Then run the following command to start training:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_orthoskillvla.sh \
  <seed> <skill1> <skill2> <skill3> <order_tag>
```

```bash
# ordering: open_close -> turn -> pick_place
CUDA_VISIBLE_DEVICES=0 bash scripts/train_orthoskillvla.sh 1 open_close turn pick_place otp

# ordering: pick_place -> open_close -> turn
CUDA_VISIBLE_DEVICES=0 bash scripts/train_orthoskillvla.sh 1 pick_place open_close turn pot

# ordering: open_close -> pick_place -> turn
CUDA_VISIBLE_DEVICES=0 bash scripts/train_orthoskillvla.sh 1 open_close pick_place turn opt
```

### 4. Outputs

After training, each skill is saved under:

```
{base_output_dir}/{run_name}/{skill_name}/
  model/                      # merged X-VLA (Hugging Face layout)
  prin_subspace/subspace.pt   # principal subspace for the subsquent skill learning
```


### 5. Simulation Evaluation

Set up Libero Environment following the [official instructions](https://github.com/Lifelong-Robot-Learning/LIBERO).

Run policy server [`scripts/deploy.py`](scripts/deploy.py) in **Terminal 1**:
```bash
source .venv/bin/activate 
CUDA_VISIBLE_DEVICES=0 python scripts/deploy.py \
    --model_path /path/to/base_output_dir/run_name/last_skill/model \
    --port 10096
```
Run Libero evaluation [`sim_eval/libero/libero_client-skills.py`](sim_eval/libero/libero_client-skills.py) in **Terminal 2**:
```bash
conda activate libero
CUDA_VISIBLE_DEVICES=0 python sim_eval/libero/libero_client-skills.py \
    --server_ip 127.0.0.1 \
    --server_port 10096 \
    --eval_time 50 \
    --output_dir orthoskillvla_logs \
    --skill_ids 0 1 2
```

`skill_ids` mapping in the eval client: `0` = `open_close`, `1` = `pick_place`, `2` = `turn`. Results are written to `{output_dir}/results.json`.

## Acknowledgement
We gratefully acknowledge the following projects for their excellent open-source contributions: 
- [X-VLA](https://github.com/2toinf/X-VLA)
- [LeRobot](https://github.com/huggingface/lerobot)
- [KeepLoRA](https://arxiv.org/abs/2601.19659)

## Citation
If you find this work or code useful in your research, please consider citing:
```
@inproceedings{jiaqi2026orthoskillvla,
  title     = {Continual Skill Learning via Gradient-Informed Skill Subspace Adaptation},
  author    = {Jiaqi Wang, Zhou Fang, Qiongfeng Shi and Yi Zhou},
  booktitle = {Proceedings of the Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  year      = {2026},
}
```