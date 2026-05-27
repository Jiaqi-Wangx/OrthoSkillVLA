import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from ortho_skill.model.modeling_xvla import XVLA

logger = logging.getLogger(__name__)


SUBSPACE_FILENAME = "subspace.pt"


def _to_device_dtype(batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    converted = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                converted[key] = value.to(device=device, dtype=dtype, non_blocking=True)
            else:
                converted[key] = value.to(device=device, non_blocking=True)
        else:
            converted[key] = value
    return converted


def get_all_attn_linear_param_names(model: XVLA) -> list[str]:
    attn_suffixes = ("q_proj", "k_proj", "v_proj", "out_proj", "qkv", "proj")
    # fmt: off
    extra_linear_modules = {
        "transformer.vlm_proj", "transformer.aux_visual_proj", 
        # "transformer.action_decoder"
    }
    # fmt: on

    param_names = []
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module_name.startswith(("vlm.vision_tower.blocks.0", "vlm.vision_tower.blocks.1")):
            logger.info(f"skipping attn in {module_name} for low-rank adaptation")
            continue
        if ("attn" in module_name) and module_name.endswith(attn_suffixes):
            param_names.append(module_name + ".weight")
            continue
        if module_name in extra_linear_modules:
            param_names.append(module_name + ".weight")
    return param_names


def load_skill_info(skill_file: str, skill_name: str) -> dict:
    skill_path = Path(skill_file)

    with open(skill_path, "r", encoding="utf-8") as f:
        skill_infos = json.load(f)

    for skill_info in skill_infos:
        if skill_info.get("name") == skill_name:
            assert "lerobot_task_ids" in skill_info, f"Skill info missing lerobot_task_ids: {skill_name}"
            return skill_info

    raise ValueError(f"Skill name not found in {skill_file}: {skill_name}")


def load_principal_subspace(subspace_path: str | Path | None) -> dict | None:
    """load principal subspace to CPU in float32"""
    if subspace_path is None or str(subspace_path).strip() == "":
        return None

    path = Path(subspace_path)
    if path.is_dir():
        path = path / "prin_subspace" / SUBSPACE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Principal subspace file not found: {path}")

    data = torch.load(path, map_location="cpu")
    required_keys = {
        "subspaces",
        "rank_current",
        "rank_added",
        "energy_threshold",
        "source_skill",
        "merged_model_path",
    }
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Invalid subspace file, missing keys {sorted(missing)}: {path}")

    return data


def extract_gradients_t(
    model: XVLA,
    dataloader,
    target_param_names: list[str],
    max_steps: int,
    desc: str,
    device: torch.device,
    compute_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Extract running-average gradients on device, output as transposed matrices."""

    model = model.to(device=device, dtype=compute_dtype)
    for param in model.parameters():
        param.requires_grad = False

    gradients = {}
    for name, param in model.named_parameters():
        if name in target_param_names and "weight" in name:
            gradients[name] = torch.zeros_like(param, dtype=torch.float32)
            param.requires_grad = True

    steps = 0
    model.train()
    model.zero_grad(set_to_none=True)

    progress = tqdm(dataloader, desc=desc, dynamic_ncols=True)
    for batch in progress:
        batch = _to_device_dtype(batch, device=device, dtype=compute_dtype)
        outputs = model(**batch)
        action_loss = outputs["action_loss"]
        loss = action_loss["position_loss"] + action_loss["rotate6D_loss"] + action_loss["gripper_loss"]
        loss.backward()

        steps += 1
        for name, param in model.named_parameters():
            if name in gradients and param.grad is not None:
                gradients[name] = gradients[name] * ((steps - 1) / steps) + param.grad.data * (1.0 / steps)

        model.zero_grad(set_to_none=True)
        if max_steps > 0 and steps >= max_steps:
            break

    return {name: grad.t().contiguous().to(dtype=output_dtype) for name, grad in gradients.items()}


def decompose_gradients_t(
    gradients_t: dict[str, torch.Tensor],
    mode: str,
    previous_subspaces: dict[str, torch.Tensor] | None = None,
    fixed_rank: int | None = None,
    energy_threshold: float | None = None,
    svd_device: torch.device = torch.device("cpu"),
    svd_dtype: torch.dtype = torch.float32,
    energy_pattern: dict[str, float] = {"transformer": 0.9999},
) -> tuple[dict[str, torch.Tensor], dict[str, int], dict[str, int]]:
    """Decompose gradients using SVD on explicit device and dtype."""
    assert mode in {"fixed_rank", "energy"}, f"Unsupported decomposition mode: {mode}"
    if mode == "fixed_rank":
        assert fixed_rank is not None and fixed_rank > 0, f"fixed_rank must be positive, got {fixed_rank}"
    if mode == "energy":
        assert energy_threshold is not None and (0.0 < energy_threshold <= 1.0)

    decomposed: dict[str, torch.Tensor] = {}
    rank_current = {}
    rank_added = {}

    for name, grad_t in gradients_t.items():
        grad_t = grad_t.to(device=svd_device, dtype=svd_dtype)  # (in_features, out_features)
        if mode == "fixed_rank":
            if previous_subspaces is not None and name in previous_subspaces:
                prev_basis_rows = previous_subspaces[name].to(
                    device=svd_device, dtype=svd_dtype
                )  # (prev_rank, in_features)
                grad_t_proj = grad_t - prev_basis_rows.t() @ prev_basis_rows @ grad_t  # (in_features, out_features)
                U, S, V = torch.svd_lowrank(grad_t_proj, q=fixed_rank)  # (in_features, q), (q,), (out_features, q)
                logger.info(f"with prev_prin excluded, {fixed_rank=}, grad {name} decomposed shape: {U.t().shape}")
            else:
                U, S, V = torch.svd_lowrank(grad_t, q=fixed_rank)  # (in_features, q), (q,), (out_features, q)
                S_all = torch.linalg.svdvals(grad_t)
                energy_ratio = (S**2).sum() / (S_all**2).sum()
                logger.info(
                    f"no prev_prin, {fixed_rank=}, grad {name} energy_ratio: {energy_ratio:.4f}, decomposed shape: {U.t().shape}"
                )

            actual_rank = fixed_rank
            svd_result = U.t().contiguous()  # (actual_rank, in_features)

        elif mode == "energy":
            final_energy_threshold = energy_threshold
            for pattern, pattern_threshold in energy_pattern.items():
                if name.startswith(pattern):
                    final_energy_threshold = pattern_threshold
                    break
            activation = grad_t  # (in_features, out_features)
            U1, S1, Vh1 = torch.linalg.svd(
                activation, full_matrices=False
            )  # (in_features, k), (k=min(in_features, out_features),), (k, out_features)
            sval_total = (S1**2).sum()

            if previous_subspaces is not None and name in previous_subspaces:
                feature_tensor = previous_subspaces[name].to(
                    device=svd_device, dtype=svd_dtype
                )  # (prev_rank, in_features)
                act_hat = activation - feature_tensor.t() @ feature_tensor @ activation
                U, S, Vh = torch.linalg.svd(
                    act_hat, full_matrices=False
                )  # (in_features, k), (k=min(in_features, out_features),), (k, out_features)

                sval_hat = (S**2).sum()
                sval_ratio = (S**2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < final_energy_threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break

                if r == 0:
                    svd_result = feature_tensor.contiguous()  # (prev_rank, in_features)
                    actual_rank = feature_tensor.shape[0]
                    new_rank = 0
                    logger.info(
                        f"energy{final_energy_threshold} mode with prev_prin: grad {name}, previous subspaces are sufficient to cover the energy, decomposed shape: {svd_result.shape}, new_rank={new_rank}, energy_ratio: {accumulated_sval:.4f}"
                    )
                else:
                    new_feature = U[:, :r].t().contiguous()  # (r, in_features)
                    Ui = torch.cat([feature_tensor, new_feature], dim=0)  # (prev_rank + r, in_features)
                    if Ui.shape[0] > Ui.shape[1]:
                        Ui = Ui[: Ui.shape[1], :]
                    svd_result = Ui.contiguous()  # (total_rank, in_features)
                    actual_rank = Ui.shape[0]
                    new_rank = actual_rank - feature_tensor.shape[0]
                    logger.info(
                        f"energy{final_energy_threshold} mode with prev_prin: grad {name}, decomposed shape: {svd_result.shape}, new_rank={new_rank}, energy_ratio: {accumulated_sval:.4f}"
                    )

            else:
                energy = S1**2
                total_energy = energy.sum()
                cumulative_energy = torch.cumsum(energy, dim=0)
                energy_ratio = cumulative_energy / total_energy

                actual_rank = (energy_ratio >= final_energy_threshold).nonzero(as_tuple=True)[0][0].item() + 1
                svd_result = U1[:, :actual_rank].t().contiguous()  # (actual_rank, in_features)

                logger.info(
                    f"energy{final_energy_threshold} mode without prev_prin: grad {name}, decomposed shape: {svd_result.shape}, energy_ratio: {energy_ratio[actual_rank - 1]:.4f}"
                )

        decomposed[name] = svd_result
        rank_current[name] = actual_rank
        if previous_subspaces is not None and name in previous_subspaces:
            rank_added[name] = actual_rank - previous_subspaces[name].shape[0]
        else:
            rank_added[name] = actual_rank
    logger.info(f"num gradients: {len(gradients_t)}, num decomposed: {len(decomposed)}")
    return decomposed, rank_current, rank_added
