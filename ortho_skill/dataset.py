import random
import logging
import hashlib
import jsonlines
from pathlib import Path
from collections.abc import Callable


import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

from ortho_skill.model.processing_xvla import XVLAProcessor


logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _get_episodes_by_task(meta: LeRobotDatasetMetadata, task_ids: list[int] | None) -> dict[int, list[int]]:
    if task_ids is not None:
        target_task_ids = sorted(list(set(task_ids)))
        all_task_ids = set(meta.tasks["task_index"])
        assert all_task_ids.issuperset(target_task_ids), f"Invalid task IDs: {set(task_ids) - all_task_ids}"
    else:
        target_task_ids = sorted(list(meta.tasks["task_index"]))

    episode_by_task = {}

    for t_id in target_task_ids:
        task_descs = meta.tasks[meta.tasks.values == t_id].index.tolist()
        task_desc = task_descs[0]
        logger.info(f"train task(lerobot_task_id={t_id}): {task_desc}")

        ep_filtered = meta.episodes.select_columns(["episode_index", "tasks"]).filter(lambda e: task_desc in e["tasks"])

        ep_ids = sorted(list(ep_filtered["episode_index"]))
        episode_by_task[t_id] = ep_ids

    return episode_by_task


def split_episode_ids(
    meta: LeRobotDatasetMetadata,
    task_ids: list[int] | None = None,
    eval_episode_num: int = 0,
    seed: int = 42,  # 新增：强制指定用于划分的种子
) -> tuple[list[int], list[int] | None]:
    """
    使用局部随机数生成器确保多卡训练时划分完全一致。
    """
    # 1. 获取有序的分组数据 (确定性操作)
    episode_by_task = _get_episodes_by_task(meta, task_ids)

    # 2. 展平所有 ID 并排序 (用于计算总数)
    all_ep_ids = []
    task_keys = sorted(episode_by_task.keys())
    for t_id in task_keys:
        all_ep_ids.extend(episode_by_task[t_id])
    all_ep_ids = sorted(all_ep_ids)

    # 3. 处理无验证集情况
    if eval_episode_num <= 0:
        return all_ep_ids, None

    assert eval_episode_num < len(all_ep_ids), f"Eval num {eval_episode_num} >= Total {len(all_ep_ids)}"

    # 4. 确定每个任务的验证集数量
    num_tasks = len(task_keys)
    base_count = eval_episode_num // num_tasks
    remainder = eval_episode_num % num_tasks

    # 5. 随机分配 - 使用局部 Random 实例 (确定性)
    rng = random.Random(seed)

    # 决定哪些任务多承担一个验证集 (先shuffle任务，取前remainder个)
    shuffled_tasks = list(task_keys)
    rng.shuffle(shuffled_tasks)

    task_eval_counts = {t: base_count for t in task_keys}
    for i in range(remainder):
        task_eval_counts[shuffled_tasks[i]] += 1

    val_episode_ids = []
    for t_id in task_keys:
        count = task_eval_counts[t_id]
        episodes = episode_by_task[t_id]

        # 如果任务样本不足，取全部 (极少情况，但需处理)
        if len(episodes) < count:
            logger.warning(
                f"Task {t_id} has {len(episodes)} episodes, but needs {count} val episodes. Using all for val."
            )
            selected = episodes
        else:
            selected = rng.sample(episodes, count)

        val_episode_ids.extend(selected)

    # 6. 计算训练集
    val_episode_ids = sorted(val_episode_ids)
    train_episode_ids = sorted(list(set(all_ep_ids) - set(val_episode_ids)))

    logger.info(
        f"Deterministic Balanced Split (Seed={seed}) -> Train: {len(train_episode_ids)}, Val: {len(val_episode_ids)}"
    )
    return train_episode_ids, val_episode_ids


class VLADataset(Dataset):
    def __init__(
        self,
        lerobot_repo_id: str,
        episode_ids: list[int],
        pred_horizon: int,
        observation_keys: tuple[str, ...],
        action_repr: tuple[str, ...],
        proprio_repr: tuple[str, ...],
        processor: XVLAProcessor,
        lerobot_root: str | None = None,
        image_transforms: dict[str, Callable] | Callable | None = None,
    ):
        super().__init__()
        self.metadata = LeRobotDatasetMetadata(repo_id=lerobot_repo_id, root=lerobot_root)
        self.episode_ids = self._valid_episode_ids(episode_ids)

        self.pred_horizon = pred_horizon
        self.action_repr = action_repr
        self.proprio_repr = proprio_repr
        self.processor = processor

        if image_transforms is not None:
            if isinstance(image_transforms, Callable):
                image_transforms = {key: image_transforms for key in observation_keys}
            else:
                assert isinstance(image_transforms, dict), f"{type(image_transforms)=} must be dict if provided."
                assert set(image_transforms.keys()) == set(observation_keys), (
                    f"{list(image_transforms.keys())=} not match {observation_keys=}"
                )

        self.observation_keys = observation_keys
        self.image_transforms = image_transforms
        self.delta_timestamps = self._init_lerobot_delta_timestamps()
        self.dataset = LeRobotDataset(
            repo_id=lerobot_repo_id,
            root=lerobot_root,
            episodes=self.episode_ids,
            delta_timestamps=self.delta_timestamps,
        )

    def _init_lerobot_delta_timestamps(self) -> dict[str, list[float]]:
        delta_timestamps = {}
        for obs_key in self.observation_keys:
            assert self.metadata.features[obs_key]["dtype"] in ["image", "video"], (
                f"Observation key {obs_key} must be image or video."
            )
            index_iter = range(0, 1)
            delta_timestamps[obs_key] = [t / self.metadata.fps for t in index_iter]

        for repr in set(self.action_repr) | set(self.proprio_repr):
            assert self.metadata.features[repr]["dtype"] not in ["image", "video"], (
                f"Action key {repr} must not be image or video."
            )
            index_iter = range(0, self.pred_horizon + 1)
            delta_timestamps[repr] = [t / self.metadata.fps for t in index_iter]

        return delta_timestamps

    def _valid_episode_ids(self, episode_ids: list[int] | None) -> list[int]:
        if episode_ids is None:
            return None
        all_episode_ids = set(self.metadata.episodes["episode_index"])
        episode_ids_set = set(episode_ids)
        if invalid_ids := episode_ids_set - all_episode_ids:
            raise ValueError(f"Invalid episode IDs: {invalid_ids}")
        return sorted(list(episode_ids_set))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        # sample = self._exclude_keys(sample)
        # vision
        images = []
        for image_key in self.observation_keys:
            if self.image_transforms is not None:
                images.append(self.image_transforms[image_key](sample[image_key]))
            else:
                images.append(sample[image_key])

        # language
        instruction = sample["task"]

        # action
        action = []
        for repr in self.action_repr:
            action.append(sample[repr][1:, :])
        action = torch.cat(action, dim=-1)
        action[:, -1] = (action[:, -1] + 1) / 2 # NOTE: scale gripper to [0, 1]
        action = torch.cat([action, torch.zeros_like(action)], dim=-1)  # pad to match xvla output dim

        # proprio
        proprio = []
        for repr in self.proprio_repr:
            proprio.append(sample[repr][0])
        proprio = torch.cat(proprio, dim=-1) # (D,)
        proprio[-1] = (proprio[-1] + 1) / 2 # NOTE: scale gripper to [0, 1]
        proprio = torch.cat([proprio, torch.zeros_like(proprio)], dim=-1)  # pad to match xvla input dim

        lang_input = self.processor.encode_language(instruction)

        num_images = len(images)
        while len(images) < self.processor.num_views:
            images.append(torch.zeros_like(images[0]))
        image_mask = torch.zeros(self.processor.num_views, dtype=torch.bool)
        image_mask[: num_images] = True
        image_input = torch.stack(images, dim=0)  # num_views, C, H, W

        model_inputs = {
            "input_ids": lang_input["input_ids"].squeeze(0),  # (L,)
            "image_input": image_input,
            "image_mask": image_mask,
            "domain_id": torch.tensor(30, dtype=torch.long),
            "proprio": proprio,  # (1, D*2)
            "action": action,  # (H, D*2)
        }

        return model_inputs



