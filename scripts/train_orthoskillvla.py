import sys
import math
import gc
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated

import tyro
import peft
import torch
import torchvision.transforms.v2 as T
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import get_scheduler
from torch.utils.data import DataLoader, Subset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from ortho_skill.dataset import VLADataset, split_episode_ids
from ortho_skill.model.configuration_xvla import XVLAConfig
from ortho_skill.model.modeling_xvla import XVLA
from ortho_skill.model.processing_xvla import XVLAProcessor

sys.path.append(str(Path(__file__).parent.parent))  # for importing from the root package
from scripts.orthoskill_utils import (
    SUBSPACE_FILENAME,
    decompose_gradients_t,
    extract_gradients_t,
    get_all_attn_linear_param_names,
    load_principal_subspace,
    load_skill_info,
)


logger = logging.getLogger(__name__)
attn_implementation = "sdpa"

TRAIN_DTYPE = torch.bfloat16
GRAD_EXTRACT_DTYPE = torch.float32
GRAD_OUTPUT_DTYPE = torch.float32
SVD_DTYPE = torch.float32
SVD_DEVICE = torch.device("cpu")


@dataclass
class DataArguments:
    lerobot_repo_id: str
    lerobot_repo_root: str | None = None
    task_ids: list[int] | None = None

    observation_keys: tuple[str, ...] = ("observation.images.image", "observation.images.wrist_image")
    action_repr: tuple[str, ...] = ("action.absolute_rot6d",)
    proprio_repr: tuple[str, ...] = ("action.absolute_rot6d",)


@dataclass
class TrainingArguments:
    debug: bool = False
    load_from: str = ""
    base_output_dir: str = ""
    run_name: str = "unnamed_exp"
    run_description: str = ""

    learning_rate: float = 1e-5
    lr_scheduler_min_lr_rate: float = 0.01
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 3.0
    max_train_steps: int = -1
    warmup_steps: int = 0
    logging_steps: int = 10
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True

    dataloader_num_workers: int = 4
    dataloader_drop_last: bool = True

    seed: int = 42
    gpu_ids: str = ""

    @property
    def output_dir(self) -> Path:
        return Path(self.base_output_dir) / self.run_name

    def __post_init__(self):
        base_dir = Path(self.base_output_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"Base output directory does not exist: {base_dir}")
        if not self.load_from:
            raise ValueError("train.load_from must be set")


@dataclass
class OrthoSkillArguments:
    skill_file: str = ""
    skill_name: str = ""
    # prev_model_path: str | None = None
    prev_subspace_path: str | None = None

    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0

    decoder_lora_r: int = -1
    decoder_lora_alpha: int = -1
    use_action_decoder_moe: bool = False
    decoder_routing_basis_dim: int = -1

    gradient_batch_size: int = 8
    gradient_max_steps: int = -1
    gradient_data_ratio: float = 1.0
    gradient_num_workers: int = 4
    subspace_energy_threshold: float = 0.99
    action_head_energy_threshold: float | None = 0.9999

    def __post_init__(self):
        if not self.skill_file:
            raise ValueError("orthoskillvla.skill_file must be set")
        if not self.skill_name:
            raise ValueError("orthoskillvla.skill_name must be set")
        if self.lora_r <= 0:
            raise ValueError(f"orthoskillvla.lora_r must be > 0, got {self.lora_r}")
        if self.use_action_decoder_moe and (self.decoder_lora_r != -1 or self.decoder_lora_alpha != -1):
            raise ValueError("When use_action_decoder_moe is enabled, decoder_lora_r/alpha must both be -1")
        if self.use_action_decoder_moe and self.decoder_routing_basis_dim <= 0:
            raise ValueError("When use_action_decoder_moe is enabled, decoder_routing_basis_dim must be > 0")
        if not (0.0 < self.subspace_energy_threshold <= 1.0):
            raise ValueError(
                f"orthoskillvla.subspace_energy_threshold must be in (0, 1], got {self.subspace_energy_threshold}"
            )
        if not (0.0 < self.gradient_data_ratio <= 1.0):
            raise ValueError(f"orthoskillvla.gradient_data_ratio must be in (0, 1], got {self.gradient_data_ratio}")
        if not (0.0 < self.action_head_energy_threshold <= 1.0):
            raise ValueError(
                f"orthoskillvla.action_head_energy_threshold must be in (0, 1], got {self.action_head_energy_threshold}"
            )


@dataclass
class RunArguments:
    data: DataArguments = field(default_factory=DataArguments)
    train: TrainingArguments = field(default_factory=TrainingArguments)
    orthoskillvla: OrthoSkillArguments = field(default_factory=OrthoSkillArguments)
    model_config: Annotated[XVLAConfig, tyro.conf.Suppress] = field(default_factory=XVLAConfig)


def setup_logger(args: RunArguments) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d >> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    if not args.train.debug:
        log_dir = Path("./train_log")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{args.train.run_name}.log"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d >> %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logging.getLogger().addHandler(file_handler)


def _decoder_expert_weight_name(skill_name: str) -> str:
    # transformer.action_decoder.expert_decoders.open_close.weight
    return f"transformer.action_decoder.expert_decoders.{skill_name}.weight"


def pre_skill(
    args: RunArguments,
    previous_subspaces: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    model = XVLA.from_pretrained(
        args.train.load_from, dtype=GRAD_EXTRACT_DTYPE, attn_implementation=attn_implementation
    ).to(device)
    processor = XVLAProcessor.from_pretrained(args.train.load_from)
    if args.orthoskillvla.use_action_decoder_moe:
        logger.info(
            "Using action decoder MoE!! decoder LoRA must remain disabled and routing basis dim is "
            f"{args.orthoskillvla.decoder_routing_basis_dim}."
        )

    if args.train.gradient_checkpointing:
        model.vlm.vision_tower.enable_checkpoint = True
        model.vlm.language_model.config.use_cache = False
        model.vlm.language_model.gradient_checkpointing_enable()
        model.vlm.language_model.enable_input_require_grads()
        model.transformer.gradient_checkpointing_enable()

    target_param_names = get_all_attn_linear_param_names(model)
    if args.orthoskillvla.decoder_lora_r > 0 and args.orthoskillvla.decoder_lora_alpha > 0:
        assert "transformer.action_decoder.weight" in target_param_names, (
            "Expected action decoder weight in target parameters for applying decoder LoRA"
        )

    named_params = dict(model.named_parameters())
    named_modules = dict(model.named_modules())
    for param_name in target_param_names:
        assert param_name in named_params, f"Target weight not found in model parameters: {param_name}"
        assert param_name.endswith(".weight"), f"Target parameter must be a linear weight: {param_name}"
        module_name = param_name[: -len(".weight")]
        assert isinstance(named_modules.get(module_name), torch.nn.Linear), (
            f"Target module is not nn.Linear: {module_name}"
        )

    metadata = LeRobotDatasetMetadata(args.data.lerobot_repo_id, args.data.lerobot_repo_root)
    train_episode_ids, _ = split_episode_ids(
        metadata,
        task_ids=args.data.task_ids,
        eval_episode_num=0,
        seed=args.train.seed,
    )
    image_transforms = T.Compose(
        [
            T.Resize(224, T.InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
        ]
    )
    train_dataset = VLADataset(
        lerobot_repo_id=args.data.lerobot_repo_id,
        lerobot_root=args.data.lerobot_repo_root,
        episode_ids=train_episode_ids,
        pred_horizon=args.model_config.num_actions,
        observation_keys=args.data.observation_keys,
        action_repr=args.data.action_repr,
        proprio_repr=args.data.proprio_repr,
        processor=processor,
        image_transforms=image_transforms,
    )
    if args.orthoskillvla.gradient_data_ratio < 1.0:
        keep = max(1, int(math.ceil(len(train_dataset) * args.orthoskillvla.gradient_data_ratio)))
        indices = torch.linspace(0, len(train_dataset) - 1, keep).long().tolist()
        gradient_dataset = Subset(train_dataset, indices)
    else:
        gradient_dataset = train_dataset
    gradient_loader = DataLoader(
        gradient_dataset,
        batch_size=args.orthoskillvla.gradient_batch_size,
        num_workers=args.orthoskillvla.gradient_num_workers,
        shuffle=False,
        drop_last=False,
    )

    logger.info(f"Start pre-train extract_gradients_t for {args.orthoskillvla.skill_name}...")
    gradients_t = extract_gradients_t(
        model=model,
        dataloader=gradient_loader,
        target_param_names=target_param_names,
        max_steps=args.orthoskillvla.gradient_max_steps,
        desc=f"extract_grad-{args.orthoskillvla.skill_name}",
        device=device,
        compute_dtype=GRAD_EXTRACT_DTYPE,
        output_dtype=GRAD_OUTPUT_DTYPE,
    )

    logger.info(
        f"Start pre-train decompose_gradients_t for {args.orthoskillvla.skill_name}...(previous_subspaces is None: {previous_subspaces is None})"
    )
    lora_a_init, _, _ = decompose_gradients_t(
        gradients_t,
        mode="fixed_rank",
        previous_subspaces=previous_subspaces,
        fixed_rank=args.orthoskillvla.lora_r,
        svd_device=SVD_DEVICE,
        svd_dtype=SVD_DTYPE,
    )
    assert len(gradients_t) == len(lora_a_init), (
        f"Expected the same number of gradients and LoRA A inits, but {len(gradients_t)=} {len(lora_a_init)=}"
    )
    return lora_a_init, target_param_names


def train_one_skill(
    args: RunArguments,
    lora_a_init: dict[str, torch.Tensor],
    target_param_names: list[str],
) -> Path:
    accelerator = Accelerator(gradient_accumulation_steps=args.train.gradient_accumulation_steps)

    model = XVLA.from_pretrained(args.train.load_from, dtype=TRAIN_DTYPE, attn_implementation=attn_implementation)
    model = model.to(accelerator.device)

    processor = XVLAProcessor.from_pretrained(args.train.load_from)
    if args.orthoskillvla.use_action_decoder_moe:
        logger.info(f"Adding new skill [{args.orthoskillvla.skill_name}] expert ...")
        model.replace_action_decoder_with_moe(
            expert_names=[args.orthoskillvla.skill_name],
            routing_basis_dim=args.orthoskillvla.decoder_routing_basis_dim,
        )
        model.set_action_decoder_active_expert(args.orthoskillvla.skill_name)
        model._sync_action_decoder_config()
    if args.train.gradient_checkpointing:
        model.vlm.vision_tower.enable_checkpoint = True
        model.vlm.language_model.config.use_cache = False
        model.vlm.language_model.gradient_checkpointing_enable()
        model.vlm.language_model.enable_input_require_grads()
        model.transformer.gradient_checkpointing_enable()

    target_modules = [name[: -len(".weight")] for name in target_param_names]

    decoder_module_name = "transformer.action_decoder"
    rank_pattern = {}
    alpha_pattern = {}
    if args.orthoskillvla.decoder_lora_r > 0 and args.orthoskillvla.decoder_lora_alpha > 0:
        # "transformer.action_decoder" in target_modules is already asserted in pre_skill
        rank_pattern = {decoder_module_name: args.orthoskillvla.decoder_lora_r}
        alpha_pattern = {decoder_module_name: args.orthoskillvla.decoder_lora_alpha}
        logger.info(
            f"Added action decoder LoRA with r={args.orthoskillvla.decoder_lora_r} and alpha={args.orthoskillvla.decoder_lora_alpha}"
        )

    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=args.orthoskillvla.lora_r,
            lora_alpha=args.orthoskillvla.lora_alpha,
            lora_dropout=args.orthoskillvla.lora_dropout,
            target_modules=target_modules,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            bias="none",
        ),
    )

    num_init_params = 0
    peft_named_params = dict(model.named_parameters())
    for target_param_name, a_init in lora_a_init.items():
        # model.base_model.module_name.lora_A.default.weight
        # target_param_name: module_name.weight
        module_name = target_param_name[: -len(".weight")]
        suffix = f"{module_name}.lora_A.default.weight"
        matched = [name for name in peft_named_params if name.endswith(suffix)]
        if len(matched) != 1:
            raise ValueError(f"Cannot uniquely map LoRA A parameter for {target_param_name}: {matched}")
        name = matched[0]
        param = peft_named_params[name]
        assert param.shape == a_init.shape, f"Shape mismatch for {name}: expected {param.shape}, got {a_init.shape}"

        with torch.no_grad():
            param.copy_(a_init.to(device=param.device, dtype=param.dtype))
        param.requires_grad_(False)
        num_init_params += 1
        logger.info(f"Initialized LoRA A for {name} with shape {param.shape}")
    logger.info(f"Initialized {num_init_params} LoRA A parameters from pre-training gradients.")

    trainable_params = 0
    expert_prefix = _decoder_expert_weight_name(args.orthoskillvla.skill_name)[: -len(".weight")]
    for name, param in model.named_parameters():
        if ".lora_B." in name or (args.orthoskillvla.use_action_decoder_moe and expert_prefix in name):
            param.requires_grad_(True)
            trainable_params += param.numel()
        else:
            param.requires_grad_(False)
    if trainable_params <= 0:
        raise RuntimeError("No trainable LoRA B params after freezing LoRA A.")

    logger.info(f"Trainable LoRA B parameters: {trainable_params / 1e6:.3f}M")

    metadata = LeRobotDatasetMetadata(args.data.lerobot_repo_id, args.data.lerobot_repo_root)
    train_episode_ids, _ = split_episode_ids(
        metadata,
        task_ids=args.data.task_ids,
        eval_episode_num=0,
        seed=args.train.seed,
    )
    image_transforms = T.Compose(
        [
            T.Resize(224, T.InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
        ]
    )
    train_dataset = VLADataset(
        lerobot_repo_id=args.data.lerobot_repo_id,
        lerobot_root=args.data.lerobot_repo_root,
        episode_ids=train_episode_ids,
        pred_horizon=args.model_config.num_actions,
        observation_keys=args.data.observation_keys,
        action_repr=args.data.action_repr,
        proprio_repr=args.data.proprio_repr,
        processor=processor,
        image_transforms=image_transforms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train.per_device_batch_size,
        num_workers=args.train.dataloader_num_workers,
        shuffle=True,
        drop_last=args.train.dataloader_drop_last,
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.train.learning_rate,
        weight_decay=0.0,
        betas=(args.train.adam_beta1, args.train.adam_beta2),
    )

    num_training_steps = None
    if args.train.max_train_steps is not None and args.train.max_train_steps > 0:
        num_training_steps = args.train.max_train_steps * accelerator.num_processes
    else:
        updates_per_epoch = math.ceil(len(train_loader) / args.train.gradient_accumulation_steps)
        num_training_steps = math.ceil(updates_per_epoch * args.train.num_train_epochs)

    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=args.train.warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs={
            "min_lr_rate": args.train.lr_scheduler_min_lr_rate,
        },
    )

    model, optimizer, scheduler, train_loader = accelerator.prepare(model, optimizer, scheduler, train_loader)

    updates_per_epoch = math.ceil(len(train_loader) / args.train.gradient_accumulation_steps)
    if args.train.max_train_steps is not None and args.train.max_train_steps > 0:
        total_steps = args.train.max_train_steps
        num_epochs = math.ceil(args.train.max_train_steps / updates_per_epoch)
    else:
        num_epochs = math.ceil(args.train.num_train_epochs)
        total_steps = math.ceil(args.train.num_train_epochs * updates_per_epoch)

    progress = tqdm(
        total=total_steps,
        desc=f"{args.train.run_name}-{args.orthoskillvla.skill_name}",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                    action_loss = outputs["action_loss"]
                    loss = action_loss["position_loss"] + action_loss["rotate6D_loss"] + action_loss["gripper_loss"]

                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if global_step % args.train.logging_steps == 0 or global_step == 1:
                    log_info = {
                        "global_step": global_step,
                        "train/epoch": global_step / updates_per_epoch,
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm.item(),
                    }
                    progress.write(
                        str({k[len("train/") :]: f"{v:.6f}" for k, v in log_info.items() if k.startswith("train/")})
                    )

                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break
    progress.close()

    accelerator.wait_for_everyone()
    peft_model = accelerator.unwrap_model(model)
    skill_output_dir = args.train.output_dir / args.orthoskillvla.skill_name

    # adapter_output_dir = skill_output_dir / "adapter"
    # adapter_output_dir.mkdir(parents=True, exist_ok=True)
    # assert isinstance(peft_model, peft.PeftModel), "Expected the model to be a PeftModel after training"
    # peft_model.save_pretrained(adapter_output_dir, safe_serialization=True)
    # processor.save_pretrained(adapter_output_dir)
    # logger.info(f"Saved LoRA adapter for skill {args.orthoskillvla.skill_name} to {adapter_output_dir}")

    model_output_dir = skill_output_dir / "model"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    merged_model = peft_model.merge_and_unload()
    assert isinstance(merged_model, XVLA), "Expected the merged model to be an instance of XVLA"
    merged_model.save_pretrained(model_output_dir, safe_serialization=True)
    processor.save_pretrained(model_output_dir)
    logger.info(f"Saved merged model for skill {args.orthoskillvla.skill_name} to {model_output_dir}")

    accelerator.print(f"✅ Finished training skill {args.orthoskillvla.skill_name}.")
    accelerator.end_training()

    del merged_model
    del peft_model
    del model
    del optimizer
    del scheduler
    del train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_output_dir


def post_train(
    args: RunArguments,
    merged_model_path: Path,
    previous_subspaces: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> Path:

    model = XVLA.from_pretrained(
        str(merged_model_path), dtype=GRAD_EXTRACT_DTYPE, attn_implementation=attn_implementation
    ).to(device)
    processor = XVLAProcessor.from_pretrained(str(merged_model_path), use_fast=True)
    if args.orthoskillvla.use_action_decoder_moe:
        action_decoder = model.transformer.action_decoder
        assert hasattr(action_decoder, "expert_names"), "Expected action decoder to be MoE-enabled"
        assert args.orthoskillvla.skill_name in action_decoder.expert_names, (
            f"Expected decoder expert to already exist: {args.orthoskillvla.skill_name}"
        )
        logger.info("post-train checking skill expert existance: Done.")
        model.set_action_decoder_active_expert(args.orthoskillvla.skill_name)
        logger.info(f"post-train set expert {args.orthoskillvla.skill_name} active for gradient extraction.")

    if args.train.gradient_checkpointing:
        model.vlm.vision_tower.enable_checkpoint = True
        model.vlm.language_model.config.use_cache = False
        model.vlm.language_model.gradient_checkpointing_enable()
        model.vlm.language_model.enable_input_require_grads()
        model.transformer.gradient_checkpointing_enable()

    target_param_names = get_all_attn_linear_param_names(model)
    named_params = dict(model.named_parameters())
    named_modules = dict(model.named_modules())
    for param_name in target_param_names:
        assert param_name in named_params, f"Target weight not found in model parameters: {param_name}"
        assert param_name.endswith(".weight"), f"Target parameter must be a linear weight: {param_name}"
        module_name = param_name[: -len(".weight")]
        assert isinstance(named_modules.get(module_name), torch.nn.Linear), (
            f"Target module is not nn.Linear: {module_name}"
        )
    if args.orthoskillvla.use_action_decoder_moe:
        decoder_weight_name = _decoder_expert_weight_name(args.orthoskillvla.skill_name)
        assert decoder_weight_name in named_params, f"Expected decoder expert weight to exist: {decoder_weight_name}"
        logger.info(f"Besides attention linear weights, also extracting gradient: {decoder_weight_name}")
        target_param_names.append(decoder_weight_name)

    metadata = LeRobotDatasetMetadata(args.data.lerobot_repo_id, args.data.lerobot_repo_root)
    train_episode_ids, _ = split_episode_ids(
        metadata,
        task_ids=args.data.task_ids,
        eval_episode_num=0,
        seed=args.train.seed,
    )
    image_transforms = T.Compose(
        [
            T.Resize(224, T.InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
        ]
    )
    train_dataset = VLADataset(
        lerobot_repo_id=args.data.lerobot_repo_id,
        lerobot_root=args.data.lerobot_repo_root,
        episode_ids=train_episode_ids,
        pred_horizon=args.model_config.num_actions,
        observation_keys=args.data.observation_keys,
        action_repr=args.data.action_repr,
        proprio_repr=args.data.proprio_repr,
        processor=processor,
        image_transforms=image_transforms,
    )
    if args.orthoskillvla.gradient_data_ratio < 1.0:
        keep = max(1, int(math.ceil(len(train_dataset) * args.orthoskillvla.gradient_data_ratio)))
        indices = torch.linspace(0, len(train_dataset) - 1, keep).long().tolist()
        gradient_dataset = Subset(train_dataset, indices)
    else:
        gradient_dataset = train_dataset
    gradient_loader = DataLoader(
        gradient_dataset,
        batch_size=args.orthoskillvla.gradient_batch_size,
        num_workers=args.orthoskillvla.gradient_num_workers,
        shuffle=False,
        drop_last=False,
    )

    logger.info(f"Start post-training extract_gradients_t for {args.orthoskillvla.skill_name}...")
    gradients_t = extract_gradients_t(
        model=model,
        dataloader=gradient_loader,
        target_param_names=target_param_names,
        max_steps=args.orthoskillvla.gradient_max_steps,
        desc=f"post_grad-{args.orthoskillvla.skill_name}",
        device=device,
        compute_dtype=GRAD_EXTRACT_DTYPE,
        output_dtype=GRAD_OUTPUT_DTYPE,
    )

    logger.info(
        f"Start post-training decompose_gradients_t for {args.orthoskillvla.skill_name}...(previous_subspaces is None: {previous_subspaces is None})"
    )
    if args.orthoskillvla.action_head_energy_threshold is not None:
        energy_pattern = {"transformer": args.orthoskillvla.action_head_energy_threshold}
    else:
        energy_pattern = None
    subspaces, rank_current, rank_added = decompose_gradients_t(
        gradients_t,
        mode="energy",
        previous_subspaces=previous_subspaces,
        energy_threshold=args.orthoskillvla.subspace_energy_threshold,
        svd_device=SVD_DEVICE,
        svd_dtype=SVD_DTYPE,
        energy_pattern=energy_pattern,
    )

    if args.orthoskillvla.use_action_decoder_moe:
        decoder_weight_name = _decoder_expert_weight_name(args.orthoskillvla.skill_name)
        decoder_grad_t = gradients_t[decoder_weight_name].to(
            device=SVD_DEVICE, dtype=SVD_DTYPE
        )  # (in_features, dim_action)
        decoder_u, _, _ = torch.linalg.svd(decoder_grad_t, full_matrices=False)
        print(f"After svd, decoder_u shape: {decoder_u.shape}, decoder_grad_t shape: {decoder_grad_t.shape}")
        decoder_basis_dim = args.orthoskillvla.decoder_routing_basis_dim
        if decoder_basis_dim <= 0:
            decoder_basis_dim = decoder_grad_t.shape[1]
        if decoder_basis_dim > decoder_u.shape[1]:
            raise ValueError(
                f"decoder_routing_basis_dim={decoder_basis_dim} exceeds available SVD columns={decoder_u.shape[1]}"
            )
        decoder_basis = decoder_u[:, :decoder_basis_dim].contiguous()
        model_dtype = next(model.parameters()).dtype
        model.update_action_decoder_routing_basis(
            args.orthoskillvla.skill_name, decoder_basis.to(device=device, dtype=model_dtype)
        )
        model._sync_action_decoder_config()
        model.transformer.set_action_decoder_active_expert(None)
        logger.info(
            f"Train Skill {args.orthoskillvla.skill_name} Done: 1) set active expert to None for inference 2) updated routing basis for decoder expert"
        )

        subspaces.pop(decoder_weight_name, None)
        rank_current.pop(decoder_weight_name, None)
        rank_added.pop(decoder_weight_name, None)

    model.save_pretrained(str(merged_model_path), safe_serialization=True)
    processor.save_pretrained(str(merged_model_path))

    skill_output_dir = args.train.output_dir / args.orthoskillvla.skill_name
    subspace_dir = skill_output_dir / "prin_subspace"
    subspace_dir.mkdir(parents=True, exist_ok=True)
    subspace_file = subspace_dir / SUBSPACE_FILENAME
    torch.save(
        {
            "subspaces": subspaces,
            "rank_current": rank_current,
            "rank_added": rank_added,
            "energy_threshold": args.orthoskillvla.subspace_energy_threshold,
            "source_skill": args.orthoskillvla.skill_name,
            "merged_model_path": str(merged_model_path),
        },
        subspace_file,
    )
    logger.info(f"Post-training: Saved principal subspace for skill {args.orthoskillvla.skill_name} to {subspace_file}")
    return subspace_file


def main(args: RunArguments):
    setup_logger(args)
    torch.manual_seed(args.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train.seed)

    skill_info = load_skill_info(args.orthoskillvla.skill_file, args.orthoskillvla.skill_name)
    args.data.task_ids = skill_info["lerobot_task_ids"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.model_config = XVLAConfig.from_pretrained(args.train.load_from)
    if args.orthoskillvla.use_action_decoder_moe and args.orthoskillvla.decoder_routing_basis_dim > 0:
        args.model_config.decoder_routing_basis_dim = args.orthoskillvla.decoder_routing_basis_dim
    prev_subspace_data = load_principal_subspace(args.orthoskillvla.prev_subspace_path)
    previous_subspaces = (
        None
        if prev_subspace_data is None
        else {k: v.to(device=SVD_DEVICE, dtype=SVD_DTYPE) for k, v in prev_subspace_data["subspaces"].items()}
    )

    logger.info(f"skill_name={args.orthoskillvla.skill_name}")
    logger.info(f"lerobot_task_ids={args.data.task_ids}")
    logger.info(f"init_model_path={args.train.load_from}")
    logger.info(f"prev_subspace_path={args.orthoskillvla.prev_subspace_path}")
    logger.info(f"skill_info={skill_info}")

    lora_a_init, target_param_names = pre_skill(args, previous_subspaces, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_model_path = train_one_skill(
        args,
        lora_a_init,
        target_param_names,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    subspace_file = post_train(args, merged_model_path, previous_subspaces, device)

    logger.info(f"Finished pre- and post-training skill {args.orthoskillvla.skill_name}")
    logger.info(f"merged_model_path={merged_model_path}")
    logger.info(f"subspace_file={subspace_file}")
    logger.info(f"args.data={asdict(args.data)}")
    logger.info(f"args.train={asdict(args.train)}")
    logger.info(f"args.orthoskillvla={asdict(args.orthoskillvla)}")


if __name__ == "__main__":
    cli_args = tyro.cli(RunArguments)
    main(cli_args)
