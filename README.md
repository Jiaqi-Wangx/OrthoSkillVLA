# OrthoSkillVLA: Continual Skill Learning via Gradient-Informed Skill Subspace Adaptation

The implementation of OrthoSkillVLA, which enables a pretrained VLA model to continually adapt to multiple manipulation skills while preserving the acquired ones.


## Installation

**Requirements:** Python ≥ 3.10, CUDA GPU (training and simulation eval).

```bash
git clone https://github.com/JiaqiWng/OrthoSkillVLA.git
cd OrthoSkillVLA
uv sync --all-groups
```

## Reproduction

### 1. Pretrained Model and Dataset

Pretrained X-VLA model weights can be found [here](https://anonymous-hf.up.railway.app/a/cjqqbgkyd9ae/) and Libero dataset in [LeRobot](https://github.com/huggingface/lerobot) v3.0 format [here](https://anonymous-hf.up.railway.app/a/kruojpuf79eo/)

### 2. About Skill Splits

The skill splits can be found at [`sim_eval/libero/libero_skills.json`](sim_eval/libero/libero_skills.json) 

| Skill | LIBERO `task_ids` | LeRobot `lerobot_task_ids` |
|-------|-------------------|----------------------------|
| `open_close` | 22, 6, 28, 35 | 11, 8, 69, 57 |
| `pick_place` | 12, 46, 74, 85 | 45, 0, 42, 10 |
| `turn` | 20, 39 | 65, 63 |

### 3. Continual Skill Learning

Edit [`scripts/batch/train_orthoskillvla.sh`](scripts/batch/train_orthoskillvla.sh):

| Variable | Description |
|----------|-------------|
| `load_from` | Path to an X-VLA pretrained model |
| `base_output_dir` | Root directory for experiment outputs |
| `repo_root` | Local path to the converted LeRobot dataset |
| `repo_id` | LeRobot dataset repo id |

Then run the following command to start training:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/batch/train_orthoskillvla.sh \
  <seed> <skill1> <skill2> <skill3> <order_tag>
```

Example (open_close → turn → pick_place):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/batch/train_orthoskillvla.sh 1 open_close turn pick_place otp
```

Preset scripts on 3 orderings:

| Script | Skill order |
|--------|-------------|
| [`scripts/batch/otp.sh`](scripts/batch/otp.sh) | open_close → turn → pick_place |
| [`scripts/batch/pot.sh`](scripts/batch/pot.sh) | pick_place → open_close → turn |
| [`scripts/batch/opt.sh`](scripts/batch/opt.sh) | open_close → pick_place → turn |

### 4. Outputs

After training, each skill is saved under:

```
{base_output_dir}/{run_name}/{skill_name}/
  model/                      # merged X-VLA (Hugging Face layout)
  prin_subspace/subspace.pt   # principal subspace for the subsquent skill learning
```


### 5. Simulation Evaluation

1. Run policy server [`scripts/deploy.py`](scripts/deploy.py) in **Terminal 1**:
    ```bash
    source .venv/bin/activate 
    CUDA_VISIBLE_DEVICES=0 python scripts/deploy.py \
        --model_path /path/to/base_output_dir/run_name/last_skill/model \
        --port 10096
    ```
2. Set up Libero Environment following the [official instructions](https://github.com/Lifelong-Robot-Learning/LIBERO).
3. Run Libero evaluation [`sim_eval/libero/libero_client-skills.py`](sim_eval/libero/libero_client-skills.py) in **Terminal 2**:
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
