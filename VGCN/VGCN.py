#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三步：GCN模型训练 - 矢量地图零水印鲁棒特征提取."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import json
import logging
import math
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from tqdm import tqdm

def set_global_seed(seed: int) -> None:
    """Set reproducible random seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def default_memory_refresh_marks(total_epochs: int) -> Tuple[int, ...]:
    """计算Memory Bank刷新标记点（模仿VGAT-IMPROVED策略）"""
    if total_epochs <= 4:
        return (max(1, total_epochs - 1),)
    marks = []
    marks.append(max(1, int(total_epochs * 0.4)))  # 40%处刷新
    marks.append(max(marks[-1] + 1, int(total_epochs * 0.55)))  # 55%处刷新
    marks = [min(total_epochs - 1, m) for m in marks]
    unique_marks = sorted(set(marks))
    return tuple(mark for mark in unique_marks if mark >= 1)


def setup_logging() -> logging.Logger:
    """Configure both persistent and latest log files under VGCN/logs."""

    base_dir = os.path.dirname(__file__)
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    log_file = os.path.join(log_dir, f"step3_training_{timestamp}_{pid}.log")
    latest_file = os.path.join(log_dir, "step3_training_latest.log")

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    file_handler_unique = logging.FileHandler(log_file, encoding="utf-8")
    file_handler_latest = logging.FileHandler(latest_file, mode="w", encoding="utf-8")
    console_handler = logging.StreamHandler()

    for handler in (file_handler_unique, file_handler_latest, console_handler):
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        if isinstance(handler, logging.FileHandler):
            handler.stream.reconfigure(line_buffering=True)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler_unique)
    root_logger.addHandler(file_handler_latest)
    root_logger.addHandler(console_handler)

    globals()["CURRENT_LOG_FILE"] = log_file
    globals()["CURRENT_LATEST_LOG"] = latest_file
    os.environ["VGCN_CURRENT_LOG"] = log_file
    os.environ["VGCN_CURRENT_LOG_LATEST"] = latest_file

    logger = logging.getLogger(__name__)
    logger.info("日志文件: %s", log_file)
    logger.info("最新日志(覆盖): %s", latest_file)
    return logger


logger = logging.getLogger(__name__)

VALID_POOLING_MODES = {"dual", "mean", "max"}
LOSS_ABLATION_PRESETS = {
    "none": set(),
    "full": set(),
    "no_contrastive": {"contrastive"},
    "no_uniqueness_sep": {"uniqueness", "proto_sep"},
}
LOSS_NAME_ALIASES = {
    "contrastive": "contrastive",
    "supcon": "contrastive",
    "similarity": "similarity",
    "diversity": "diversity",
    "margin": "margin",
    "center": "center",
    "proto_sep": "proto_sep",
    "separation": "proto_sep",
    "proto_corr": "proto_corr",
    "binary": "binary",
    "uniqueness": "uniqueness",
    "unique": "uniqueness",
    "proto_anchor": "proto_anchor",
    "anchor": "proto_anchor",
    "ortho": "ortho",
    "hard_adversarial": "hard_adversarial",
    "hard_adv": "hard_adversarial",
    "balance": "balance",
    "decor": "decor",
}


def normalize_pooling_mode(pooling_mode: Optional[str]) -> str:
    mode = (pooling_mode or "dual").strip().lower()
    if mode not in VALID_POOLING_MODES:
        logger.warning("Unknown pooling mode '%s', fallback to 'dual'", pooling_mode)
        return "dual"
    return mode


def parse_disabled_losses(preset: Optional[str], raw_losses: Optional[str]) -> Set[str]:
    disabled: Set[str] = set()
    preset_key = (preset or "none").strip().lower()
    if preset_key and preset_key not in LOSS_ABLATION_PRESETS:
        logger.warning("Unknown loss ablation preset '%s', ignoring it", preset)
    disabled.update(LOSS_ABLATION_PRESETS.get(preset_key, set()))

    for token in (raw_losses or "").split(","):
        normalized = token.strip().lower()
        if not normalized:
            continue
        canonical = LOSS_NAME_ALIASES.get(normalized)
        if canonical is None:
            logger.warning("Unknown disabled loss alias '%s', ignoring it", token.strip())
            continue
        disabled.add(canonical)

    return disabled

class GCNModel(nn.Module):
    """GCN backbone for extracting robust graph-level descriptors."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1024,
        dropout: float = 0.2,
        pooling_mode: str = "dual",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.pooling_mode = normalize_pooling_mode(pooling_mode)
        self.gcn1 = GCNConv(input_dim, hidden_dim, improved=True, add_self_loops=True)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim, improved=True, add_self_loops=True)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim, improved=True, add_self_loops=True)
        pooled_dim = hidden_dim * 2 if self.pooling_mode == "dual" else hidden_dim

        self.fusion = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            if debug:
                logger.error("输入特征包含NaN/Inf，范围=[%.4f, %.4f]", x.min(), x.max())
            return torch.full((1, 1024), float("nan"), device=x.device)

        x1 = self.dropout(F.relu(self.gcn1(x, edge_index)))
        x2 = self.dropout(F.relu(self.gcn2(x1, edge_index)))
        x3 = F.relu(self.gcn3(x2, edge_index))

        if batch is None:
            batch = torch.zeros(x3.size(0), dtype=torch.long, device=x3.device)

        mean_pool = global_mean_pool(x3, batch)
        max_pool = global_max_pool(x3, batch)
        if self.pooling_mode == "mean":
            graph_features = mean_pool
        elif self.pooling_mode == "max":
            graph_features = max_pool
        else:
            graph_features = torch.cat([mean_pool, max_pool], dim=1)
        output = self.fusion(graph_features)

        if (torch.isnan(output).any() or torch.isinf(output).any()) and debug:
            logger.error("Fusion输出包含NaN/Inf，范围=[%.4f, %.4f]", output.min(), output.max())

        return output


class AdaptiveTemperature:
    def __init__(self, init_temp: float = 1.0, final_temp: float = 0.05, total_epochs: int = 30) -> None:
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.total_epochs = total_epochs

    def get_temperature(self, epoch: int) -> float:
        if epoch >= self.total_epochs:
            return self.final_temp
        progress = epoch / max(1.0, float(self.total_epochs))
        return self.init_temp * (self.final_temp / self.init_temp) ** progress


class ContrastiveTrainer:
    """Trainer combining supervised contrastive learning with task-specific regularizers."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        temperature: float = 0.06,
        use_amp: bool = False,
        batch_size: int = 8,
        disabled_losses: Optional[Set[str]] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.initial_batch_size = batch_size
        self.min_batch_size = 1

        # 使用略低的主干学习率提升收敛稳定性和鲁棒性
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=8e-5, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.robustness_start_epoch = 12
        self.warmup_epochs = 4
        self.warmup_active = False
        self.warmup_temp_boost = 1.12
        self.base_temperature = float(temperature)
        self.memory_refresh_interval = 3
        self.last_memory_refresh_epoch = -1
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        self.disabled_losses = set(disabled_losses or set())
        self.model_basename = os.environ.get("VGAT_MODEL_BASENAME", "gcn_model")

        self.w_contrastive = 1.2
        # 初始相似性权重稍高，以增强同一原图多种攻击版本的特征一致性
        self.w_similarity = 0.9
        self.w_diversity = 0.25
        self.w_margin = 0.6
        self.w_center = 0.3
        self.w_proto_sep = 0.8
        self.w_proto_corr = 0.0
        self.w_balance_max = 0.20
        self.w_decor_max = 0.10
        self.w_binary_max = 0.25
        self.w_binary = 0.0
        self.w_uniqueness_max = 1.0
        self.w_uniqueness = 0.0

        self.num_classes = 46
        self.feat_dim = 1024
        self.prototype_bank = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, device=device))
        # 原型使用略高学习率以更快逼近各类稳定中心
        self.optimizer.add_param_group({"params": [self.prototype_bank], "lr": 4e-4, "weight_decay": 0.0})

        # 提高原型锚定最大权重，增加类内鲁棒聚集
        self.w_proto_anchor_max = 1.0
        self.w_proto_anchor = 0.0
        self.w_ortho_max = 1.5
        self.w_ortho = 0.0
        self.w_hard_adversarial_max = 1.2
        self.w_hard_adversarial = 0.0
        self.hard_pair_threshold = 0.75

        self.margin_base = 0.55
        self.margin_target = 0.45
        self.hard_base = 0.65
        self.hard_target = 0.70
        self.ramp_epochs = 20

        self.memory_size = 512  # 进一步优化：减少到512个样本，提升速度
        self.bank_features: Optional[torch.Tensor] = None
        self.bank_labels: Optional[torch.Tensor] = None
        self.bank_chunk_size = 1024
        self.topk_max = 4096

        self.proto_momentum = 0.9
        self.prototypes: Dict[int, torch.Tensor] = {}

        self.adaptive_temp = AdaptiveTemperature(init_temp=1.0, final_temp=0.04, total_epochs=12)  # 调整为12个epoch
        self.memory_refresh_marks = default_memory_refresh_marks(12)  # 使用12个epoch的刷新标记
        self.ema_median: Optional[torch.Tensor] = None
        self.ema_momentum = 0.99

        self.training_history = {
            "epoch_losses": [],
            "contrastive_losses": [],
            "similarity_losses": [],
            "diversity_losses": [],
            "margin_losses": [],
            "center_losses": [],
            "balance_losses": [],
            "decor_losses": [],
            "proto_sep_losses": [],
            "proto_corr_losses": [],
            "binary_losses": [],
            "uniqueness_losses": [],
            "proto_anchor_losses": [],
            "ortho_losses": [],
            "hard_adversarial_losses": [],
            "gradient_norms": [],
            "learning_rates": [],
            "feature_stats": [],
        }

    def get_ablation_config(self) -> Dict[str, Any]:
        return {
            "pooling_mode": getattr(self.model, "pooling_mode", "dual"),
            "disabled_losses": sorted(self.disabled_losses),
        }

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "input_dim": getattr(self.model, "input_dim", 13),
            "hidden_dim": getattr(self.model, "hidden_dim", 128),
            "output_dim": getattr(self.model, "output_dim", 1024),
            "dropout": getattr(self.model, "dropout_rate", 0.2),
            "pooling_mode": getattr(self.model, "pooling_mode", "dual"),
        }

    def _masked_weight(self, loss_name: str, weight: float) -> float:
        return 0.0 if loss_name in self.disabled_losses else weight

        self._init_prototype_bank()
        self.current_epoch = 0
        self._sched_snapshot: Dict[str, float] = {}

    def _enter_robustness_warmup(self, current_lr: float) -> None:
        if self.warmup_active:
            return
        self.warmup_active = True
        self._warmup_step = 0
        boost_start = 1.25
        boost_end = 1.05
        if self.warmup_epochs <= 1:
            self._warmup_lr_schedule = [current_lr * boost_start]
        else:
            self._warmup_lr_schedule = [
                current_lr * (boost_start - (boost_start - boost_end) * (i / (self.warmup_epochs - 1)))
                for i in range(self.warmup_epochs)
            ]

    def _apply_warmup_schedule(self, epoch: int, current_lr: float) -> float:
        if not self.warmup_active:
            return current_lr
        idx = min(self._warmup_step, len(self._warmup_lr_schedule) - 1)
        new_lr = float(self._warmup_lr_schedule[idx])
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        self.temperature = min(self.base_temperature, self.temperature * self.warmup_temp_boost)
        self._warmup_step += 1
        if self._warmup_step >= self.warmup_epochs:
            self.warmup_active = False
            self._refresh_memory_bank(force=True)
            self.last_memory_refresh_epoch = epoch
        return new_lr

    def _refresh_memory_bank(self, force: bool = False, keep_ratio: float = 0.6) -> None:
        """智能Memory Bank刷新：保留难样本（模仿VGAT-IMPROVED策略）"""
        if self.bank_features is None or self.bank_labels is None:
            return
        total = self.bank_features.size(0)
        if total == 0:
            return
        if not force and total <= self.memory_size:
            return

        keep = int(self.memory_size * keep_ratio)  # 保留60%的难样本
        if keep >= total:
            return

        # 计算每个样本的难度分数（基于与批次内其他样本的相似度）
        with torch.no_grad():
            # 计算所有样本间的相似度矩阵
            feats_norm = F.normalize(self.bank_features, p=2, dim=1, eps=1e-8)
            sim_matrix = torch.matmul(feats_norm, feats_norm.T)  # [total, total]

            # 为每个样本计算难度分数（与其他不同类样本的最大相似度）
            difficulty_scores = torch.zeros(total, device=self.bank_features.device)
            labels_expanded = self.bank_labels.unsqueeze(1)
            cross_mask = (labels_expanded != labels_expanded.T)

            for i in range(total):
                cross_sims = sim_matrix[i][cross_mask[i]]
                if len(cross_sims) > 0:
                    # 难度 = 与其他类的最大相似度（相似度越高越难区分）
                    difficulty_scores[i] = cross_sims.max()
                else:
                    difficulty_scores[i] = 0.0

            # 选择最难的样本保留
            _, top_indices = torch.topk(difficulty_scores, k=keep, largest=True)

            self.bank_features = self.bank_features[top_indices].clone()
            self.bank_labels = self.bank_labels[top_indices].clone()

        logger.info(f"🔄 Memory Bank刷新: 保留 {keep}/{total} 个难样本 (keep_ratio={keep_ratio:.1f})")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init_prototype_bank(self) -> None:
        with torch.no_grad():
            temp = torch.empty(self.feat_dim, self.num_classes, device=self.device)
            nn.init.orthogonal_(temp)
            self.prototype_bank.copy_(temp.T)
    
    # ------------------------------------------------------------------
    # Epoch scheduling and data preparation
    # ------------------------------------------------------------------
    def get_dynamic_loss_weights(self, epoch: int, max_epoch: int) -> Dict[str, float]:
        """三阶段动态损失权重策略（模仿VGAT-IMPROVED）：前期唯一性→中期平衡→后期鲁棒性，总权重恒定为8.0"""
        # 调整阶段划分：前期更长（50%）以强化唯一性
        early_end = max(1, int(max_epoch * 0.5))  # 前期50%
        mid_end = max(early_end + 1, int(max_epoch * 0.83))  # 中期到83%

        # 计算阶段内进度（0.0-1.0）
        if epoch < early_end:
            stage_progress = epoch / max(1, early_end)
            # 前期：强化唯一性，抑制鲁棒性相关损失
            supcon = 1.7 - 0.2 * stage_progress  # 1.7 → 1.5
            proto = 1.0
            binary = 0.5 + 0.3 * stage_progress  # 0.5 → 0.8
            diversity = 1.3 - 0.2 * stage_progress  # 1.3 → 1.1
            uniqueness = 2.5 - 0.2 * stage_progress  # 2.5 → 2.3（前期强化唯一性）
            similarity = 0.8 + 0.1 * stage_progress  # 0.8 → 0.9
            margin = 0.5 + 0.1 * stage_progress  # 0.5 → 0.6
            center = 0.25 + 0.05 * stage_progress  # 0.25 → 0.3
        elif epoch < mid_end:
            stage_progress = (epoch - early_end) / max(1, mid_end - early_end)
            # 中期：平衡优化，各损失权重适度调整
            supcon = 1.5 - 0.1 * stage_progress  # 1.5 → 1.4
            proto = 1.0 - 0.1 * stage_progress  # 1.0 → 0.9
            binary = 0.8 + 0.4 * stage_progress  # 0.8 → 1.2
            diversity = 1.1 - 0.2 * stage_progress  # 1.1 → 0.9
            uniqueness = 2.3 - 0.5 * stage_progress  # 2.3 → 1.8（逐渐降低唯一性权重）
            similarity = 0.9 + 0.1 * stage_progress  # 0.9 → 1.0
            margin = 0.6 + 0.2 * stage_progress  # 0.6 → 0.8
            center = 0.3 + 0.1 * stage_progress  # 0.3 → 0.4
        else:
            stage_progress = (epoch - mid_end) / max(1, max_epoch - mid_end)
            # 后期：强化鲁棒性，进一步降低唯一性权重
            supcon = 1.4 - 0.1 * stage_progress  # 1.4 → 1.3
            proto = 0.9 - 0.1 * stage_progress  # 0.9 → 0.8
            binary = 1.2 + 0.3 * stage_progress  # 1.2 → 1.5
            diversity = 0.9 - 0.1 * stage_progress  # 0.9 → 0.8
            uniqueness = 1.8 - 0.3 * stage_progress  # 1.8 → 1.5（后期降低唯一性，强化鲁棒性）
            similarity = 1.0 + 0.1 * stage_progress  # 1.0 → 1.1
            margin = 0.8 + 0.2 * stage_progress  # 0.8 → 1.0
            center = 0.4 + 0.1 * stage_progress  # 0.4 → 0.5

        # 确保总权重恒定为8.0
        total_weight = supcon + proto + binary + diversity + uniqueness + similarity + margin + center
        if abs(total_weight - 8.0) > 0.01:
            scale_factor = 8.0 / total_weight
            supcon *= scale_factor
            proto *= scale_factor
            binary *= scale_factor
            diversity *= scale_factor
            uniqueness *= scale_factor
            similarity *= scale_factor
            margin *= scale_factor
            center *= scale_factor

        return {
            'supcon': supcon,
            'proto': proto,
            'binary': binary,
            'diversity': diversity,
            'uniqueness': uniqueness,
            'similarity': similarity,
            'margin': margin,
            'center': center
        }

    def _apply_epoch_schedule(self, epoch: int) -> Dict[str, float]:
        # 使用新的三阶段动态权重策略
        dynamic_weights = self.get_dynamic_loss_weights(epoch, 12)  # 假设12个epoch

        # 更新实例变量
        self.w_contrastive = dynamic_weights['supcon']
        self.w_similarity = dynamic_weights['similarity']
        self.w_diversity = dynamic_weights['diversity']
        self.w_margin = dynamic_weights['margin']
        self.w_center = dynamic_weights['center']
        self.w_uniqueness = dynamic_weights['uniqueness']
        self.w_binary = dynamic_weights['binary']

        # 其他权重保持原有逻辑，但使用更简单的调度
        extended_ramp_epochs = max(8, int(12 * 0.6))  # 基于12个epoch调整
        ramp_progress = min(1.0, max(0.05, (epoch + 1) / max(1, extended_ramp_epochs)))
        ramp = 0.4 + 0.6 * math.sqrt(ramp_progress)

        current_margin = self.margin_base + (self.margin_target - self.margin_base) * ramp
        current_hard = self.hard_base + (self.hard_target - self.hard_base) * ramp
        # 温度从略高值缓慢下降，后期对比约束更"尖锐"
        self.temperature = max(0.03, 0.08 - (0.08 - 0.03) * ramp)

        # 其他权重保持较简单
        late_start = 8  # 基于12个epoch调整
        late_phase = max(0.0, min(1.0, (epoch + 1 - late_start) / max(1, 12 - late_start)))
        uniq_scale = 1.0 - 0.25 * late_phase

        w_balance = self.w_balance_max * ramp * uniq_scale
        w_decor = self.w_decor_max * ramp * uniq_scale

        self.w_proto_anchor = self.w_proto_anchor_max * ramp * (1.0 + 0.1 * late_phase)
        self.w_ortho = self.w_ortho_max * ramp * uniq_scale
        self.w_hard_adversarial = self.w_hard_adversarial_max * ramp * uniq_scale
        self.w_proto_sep = 0.80 * uniq_scale
        self.w_proto_corr = 0.0

        schedule = {
            "margin": float(current_margin),
            "hard_threshold": float(current_hard),
            "w_balance": float(w_balance),
            "w_decor": float(w_decor),
            "ramp": float(ramp),
            "temperature": float(self.temperature),
        }

        self._sched_snapshot = {
            "ramp": schedule["ramp"],
            "current_margin": schedule["margin"],
            "current_hard": schedule["hard_threshold"],
            "temperature": schedule["temperature"],
            "w_contrastive": self.w_contrastive,
            "w_similarity": self.w_similarity,
            "w_diversity": self.w_diversity,
            "w_margin": self.w_margin,
            "w_center": self.w_center,
            "w_proto_sep": self.w_proto_sep,
            "w_proto_corr": self.w_proto_corr,
            "w_binary": self.w_binary,
            "w_uniqueness": self.w_uniqueness,
        }

        return schedule

    def _collect_pairs(
        self,
        original_graphs: Dict[str, Any],
        attacked_graphs: Dict[str, Sequence[Any]],
    ) -> Tuple[List[Tuple[Any, Any]], List[int]]:
        label_to_pairs: Dict[int, List[Tuple[Any, Any]]] = defaultdict(list)
        for label, (graph_name, original_graph) in enumerate(original_graphs.items()):
            if graph_name not in attacked_graphs:
                continue
            for attacked_graph in attacked_graphs[graph_name]:
                label_to_pairs[label].append((original_graph, attacked_graph))

        balanced_pairs: List[Tuple[Any, Any]] = []
        balanced_labels: List[int] = []
        available_labels = [lbl for lbl, pairs in label_to_pairs.items() if pairs]

        while available_labels:
            batch_pairs: List[Tuple[Any, Any]] = []
            batch_labels: List[int] = []
            multi_labels = [lbl for lbl in available_labels if len(label_to_pairs[lbl]) >= 2]

            if multi_labels:
                pivot = random.choice(multi_labels)
                for _ in range(2):
                    if label_to_pairs[pivot]:
                        batch_pairs.append(label_to_pairs[pivot].pop())
                        batch_labels.append(pivot)
            else:
                pivot = random.choice(available_labels)
                batch_pairs.append(label_to_pairs[pivot].pop())
                batch_labels.append(pivot)

            remaining_slots = self.batch_size - len(batch_pairs)
            candidates = [lbl for lbl in available_labels if label_to_pairs[lbl] and lbl not in batch_labels]
            random.shuffle(candidates)
            for lbl in candidates[:remaining_slots]:
                batch_pairs.append(label_to_pairs[lbl].pop())
                batch_labels.append(lbl)

            balanced_pairs.extend(batch_pairs)
            balanced_labels.extend(batch_labels)
            available_labels = [lbl for lbl in available_labels if label_to_pairs[lbl]]

        return balanced_pairs, balanced_labels

    def _iter_batches(
        self,
        pairs: Sequence[Tuple[Any, Any]],
        labels: Sequence[int],
    ) -> Iterable[Tuple[List[Tuple[Any, Any]], List[int]]]:
        for start in range(0, len(pairs), self.batch_size):
            yield (
                list(pairs[start : start + self.batch_size]),
                list(labels[start : start + self.batch_size]),
            )

    def _extract_batch_features(
        self,
        batch_pairs: Sequence[Tuple[Any, Any]],
        batch_idx: int,
        debug_batches: int = 5,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        originals: List[torch.Tensor] = []
        attacks: List[torch.Tensor] = []
        debug_mode = batch_idx <= debug_batches

        for original_graph, attacked_graph in batch_pairs:
            # 移动到设备
            original_gpu = original_graph.to(self.device)
            attacked_gpu = attacked_graph.to(self.device)

            # 提取特征（AMP）
            with amp.autocast(enabled=self.use_amp):
                feat_orig = self.model(original_gpu.x, original_gpu.edge_index, debug=debug_mode)
                feat_atk = self.model(attacked_gpu.x, attacked_gpu.edge_index, debug=debug_mode)

            # 检查特征是否有NaN（前5个batch开启调试）
            if debug_mode and (torch.isnan(feat_orig).any() or torch.isnan(feat_atk).any()):
                logger.error(f"\n⚠️⚠️⚠️ Batch {batch_idx} 检测到NaN！开始详细调试...")
                logger.error(f"原始图节点数: {original_gpu.x.shape[0]}, 边数: {original_gpu.edge_index.shape[1]}")
                logger.error(f"攻击图节点数: {attacked_gpu.x.shape[0]}, 边数: {attacked_gpu.edge_index.shape[1]}")
                logger.error(f"原始图特征范围: [{original_gpu.x.min():.4f}, {original_gpu.x.max():.4f}]")
                logger.error(f"攻击图特征范围: [{attacked_gpu.x.min():.4f}, {attacked_gpu.x.max():.4f}]")

                # 重新forward开启debug
                with torch.no_grad():
                    _ = self.model(original_gpu.x, original_gpu.edge_index, debug=True)
                    _ = self.model(attacked_gpu.x, attacked_gpu.edge_index, debug=True)

                logger.error(f"跳过此batch继续训练...\n")
                return None, None

            # 保存特征（不要detach，需要保留梯度！）
            originals.append(feat_orig)
            attacks.append(feat_atk)

            # 删除GPU上的图数据（计算图已建立，可以安全删除）
            del original_gpu, attacked_gpu

        # 清理GPU缓存（在特征提取循环后）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 堆叠特征（每个features是[1, 1024]，stack后是[batch_size, 1, 1024]，需要squeeze）
        batch_original = torch.cat(originals, dim=0)  # [batch_size, 1024]
        batch_attacked = torch.cat(attacks, dim=0)  # [batch_size, 1024]

        # 强制清理中间变量
        del originals, attacks
        del batch_pairs  # 清理batch变量的引用

        return batch_original, batch_attacked

    def _compute_loss_components(
        self,
        batch_original: torch.Tensor,
        batch_attacked: torch.Tensor,
        batch_labels: torch.Tensor,
        schedule: Dict[str, float],
        epoch: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        combined = torch.cat([batch_original, batch_attacked], dim=0)
        combined_labels = torch.cat([batch_labels, batch_labels], dim=0)
        proto_target_sim = max(0.0, (1.0 - schedule["margin"]) - 0.05)

        losses = {
            "contrastive": self.contrastive_loss(batch_original, batch_attacked, batch_labels),
            "similarity": self.similarity_loss(batch_original, batch_attacked),
            "diversity": self.diversity_loss(combined),
            "margin": self.inter_class_margin_loss(combined, combined_labels, margin=schedule["margin"], hard_threshold=schedule["hard_threshold"]),
            "center": self.center_loss(combined, combined_labels),
            "balance": self.bit_balance_loss(combined),
            "decor": self.decorrelation_loss(combined),
            "proto_sep": self.prototype_separation_loss(combined, combined_labels, target_similarity=proto_target_sim, hard_threshold=schedule["hard_threshold"]),
            "proto_corr": self.prototype_correlation_loss(target_similarity=0.3, hard_threshold=0.6),
            "binary": self.binary_consistency_loss(batch_original, batch_attacked, epoch=epoch),
            "uniqueness": self.label_aware_uniqueness_loss(combined, combined_labels),
            "proto_anchor": self.prototype_anchor_loss(combined, combined_labels),
            "ortho": self.orthogonality_loss(),
            "hard_adversarial": self.hard_adversarial_loss(combined, combined_labels),
        }

        return losses, combined, combined_labels

    def _compose_total_loss(self, losses: Dict[str, torch.Tensor], schedule: Dict[str, float]) -> torch.Tensor:
        return (
            self._masked_weight("contrastive", self.w_contrastive) * losses["contrastive"]
            + self._masked_weight("similarity", self.w_similarity) * losses["similarity"]
            + self._masked_weight("diversity", self.w_diversity) * losses["diversity"]
            + self._masked_weight("margin", self.w_margin) * losses["margin"]
            + self._masked_weight("center", self.w_center) * losses["center"]
            + self._masked_weight("proto_sep", self.w_proto_sep) * losses["proto_sep"]
            + self._masked_weight("proto_corr", self.w_proto_corr) * losses["proto_corr"]
            + self._masked_weight("uniqueness", self.w_uniqueness) * losses["uniqueness"]
            + self._masked_weight("proto_anchor", self.w_proto_anchor) * losses["proto_anchor"]
            + self._masked_weight("ortho", self.w_ortho) * losses["ortho"]
            + self._masked_weight("hard_adversarial", self.w_hard_adversarial) * losses["hard_adversarial"]
            + self._masked_weight("balance", schedule["w_balance"]) * losses["balance"]
            + self._masked_weight("decor", schedule["w_decor"]) * losses["decor"]
            + self._masked_weight("binary", self.w_binary) * losses["binary"]
        )

    def _apply_gradients(self, total_loss: torch.Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        return float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    def _update_memory_bank(self, combined: torch.Tensor, combined_labels: torch.Tensor) -> None:
        with torch.no_grad():
            feats_to_store = F.normalize(combined, p=2, dim=1, eps=1e-8).detach().cpu()
            labs_to_store = combined_labels.detach().cpu()

            if self.bank_features is None:
                self.bank_features = feats_to_store
                self.bank_labels = labs_to_store
                return

            self.bank_features = torch.cat([self.bank_features, feats_to_store], dim=0)
            self.bank_labels = torch.cat([self.bank_labels, labs_to_store], dim=0)

            if self.bank_features.size(0) <= self.memory_size:
                return

            batch_feats_norm = feats_to_store.to(self.device)
            bank_feats_gpu = self.bank_features.to(self.device)
            sim_matrix = torch.matmul(batch_feats_norm, bank_feats_gpu.T)
            max_sim_per_bank = sim_matrix.max(dim=0)[0]

            batch_labels_expanded = labs_to_store.unsqueeze(1)
            bank_labels_expanded = self.bank_labels.unsqueeze(0)
            is_negative = (batch_labels_expanded != bank_labels_expanded).all(dim=0)

            difficulty_score = torch.zeros(self.bank_features.size(0), device=self.device)
            difficulty_score[is_negative] = max_sim_per_bank[is_negative]
            difficulty_score[~is_negative] = max_sim_per_bank[~is_negative] * 0.3

            _, top_indices = torch.topk(difficulty_score, k=self.memory_size, largest=True)
            top_indices_cpu = top_indices.cpu()
            self.bank_features = self.bank_features[top_indices_cpu]
            self.bank_labels = self.bank_labels[top_indices_cpu]

            del batch_feats_norm, bank_feats_gpu, sim_matrix, max_sim_per_bank

    # ------------------------------------------------------------------
    # Core loss components
    # ------------------------------------------------------------------
    def contrastive_loss(self, features_original, features_attacked, labels):
        """基于Memory Bank的监督对比损失."""
        if torch.isnan(features_original).any() or torch.isinf(features_original).any():
            logger.error("features_original包含NaN或Inf")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        if torch.isnan(features_attacked).any() or torch.isinf(features_attacked).any():
            logger.error("features_attacked包含NaN或Inf")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        combined_features = torch.cat([features_original, features_attacked], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)

        return self.supervised_contrastive_loss_with_memory(
            combined_features,
            combined_labels,
            temperature=self.temperature,
        )

    def supervised_contrastive_loss(self, features, labels, temperature=None):
        """监督对比损失（Supervised Contrastive Loss）"""
        if temperature is None:
            temperature = float(self.temperature)

        # 数值稳定性检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.error("🔴 supervised_contrastive_loss输入异常！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 归一化特征
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # 相似度矩阵 [N, N]
        similarity_matrix = torch.matmul(features, features.T) / temperature

        # 构建同label mask，排除对角线
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(self.device)
        mask_anchor = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        mask_positive = mask_positive * (~mask_anchor).float()

        losses = []
        for i in range(batch_size):
            pos_mask = mask_positive[i]
            num_positives = pos_mask.sum()
            if num_positives < 1:
                continue

            logits = similarity_matrix[i]
            logits_max, _ = torch.max(logits, dim=0, keepdim=True)
            logits = logits - logits_max.detach()

            exp_logits = torch.exp(logits)

            # 排除自己
            indices = torch.arange(batch_size, device=exp_logits.device)
            mask_exclude_self = (indices != i).float()
            exp_logits_masked = exp_logits * mask_exclude_self

            log_denominator = torch.log(exp_logits_masked.sum() + 1e-8)
            log_numerator = torch.log((exp_logits_masked * pos_mask).sum() + 1e-8)

            loss = log_denominator - log_numerator
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            losses.append(loss)

        if len(losses) == 0:
            logger.warning("⚠️ batch中所有样本都没有正样本对，SupCon损失无法计算")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = torch.mean(torch.stack(losses))
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"🔴 supervised_contrastive_loss异常: {total_loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return total_loss

    def supervised_contrastive_loss_with_memory(self, features, labels, temperature=None):
        """基于Memory Bank的增强版监督对比损失"""
        if temperature is None:
            temperature = float(self.temperature)

        # 数值稳定性检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.error("🔴 supervised_contrastive_loss_with_memory输入异常！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 归一化特征
        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)

        # ===== 部分1：与Memory Bank中的样本对比 =====
        memory_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        if (self.bank_features is not None) and (self.bank_labels is not None) and (self.bank_features.size(0) > 0):
            try:
                bank_feats_cpu = self.bank_features
                bank_labs_cpu = self.bank_labels

                device = features_norm.device
                dtype = features_norm.dtype

                bank_feats = bank_feats_cpu.to(device=device, dtype=dtype, non_blocking=True)
                bank_labs = bank_labs_cpu.to(device=labels.device, non_blocking=True)

                memory_features_norm = F.normalize(bank_feats, p=2, dim=1, eps=1e-8)

                # [B, M]
                sim_to_memory = torch.matmul(features_norm, memory_features_norm.T) / temperature

                labels_expanded = labels.unsqueeze(1)           # [B, 1]
                memory_labels_expanded = bank_labs.unsqueeze(0) # [1, M]
                mask_positive_memory = (labels_expanded == memory_labels_expanded).float()

                losses_memory = []
                for i in range(batch_size):
                    pos_mask = mask_positive_memory[i]
                    num_positives = pos_mask.sum()
                    if num_positives < 1:
                        continue

                    logits = sim_to_memory[i]
                    logits_max, _ = torch.max(logits, dim=0, keepdim=True)
                    logits = logits - logits_max.detach()

                    exp_logits = torch.exp(logits)
                    log_denominator = torch.log(exp_logits.sum() + 1e-8)
                    log_numerator = torch.log((exp_logits * pos_mask).sum() + 1e-8)

                    loss = log_denominator - log_numerator
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    losses_memory.append(loss)

                if len(losses_memory) > 0:
                    memory_loss = torch.mean(torch.stack(losses_memory))
            except RuntimeError as e:
                logger.warning(f"⚠️ supervised_contrastive_loss_with_memory 使用Memory Bank时出错: {e}")
                memory_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # ===== 部分2：当前batch内的监督对比损失 =====
        batch_loss = self.supervised_contrastive_loss(features, labels, temperature)

        # 适度融合Memory Bank信号（0.5系数避免过强干扰）
        total_loss = batch_loss + 0.5 * memory_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"🔴 supervised_contrastive_loss_with_memory异常: {total_loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return total_loss

    def similarity_loss(self, features_original, features_attacked):
        """相似性损失：确保同一原图的攻击版本特征相似"""
        # 计算余弦相似度（添加eps防止数值问题）
        similarity = F.cosine_similarity(features_original, features_attacked, dim=1, eps=1e-8)
        # 最大化相似度（最小化1-相似度）
        loss = torch.mean(1 - similarity)
        
        # 检查损失
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"⚠️ similarity_loss is NaN/Inf: {loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss
    
    def diversity_loss(self, features):
        """多样性损失：防止特征坍塌，确保不同图有不同特征"""
        # 计算特征矩阵的方差
        feature_var = torch.var(features, dim=0, unbiased=False)
        # 鼓励每个维度都有足够的方差
        diversity_loss = torch.mean(torch.relu(0.1 - feature_var))
        
        # 检查损失
        if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
            logger.error(f"⚠️ diversity_loss is NaN/Inf: {diversity_loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return diversity_loss
    
    def inter_class_margin_loss(self, features, labels, margin=0.40, hard_threshold=0.7):
        """增强版类间边界损失：批内 + 内存队列跨类约束，并对高相似度施加平方惩罚"""
        # 归一化特征
        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        target_similarity = 1.0 - margin
        
        # ===== 批内负样本 =====
        similarity_matrix = torch.matmul(features_norm, features_norm.T)  # [B, B]
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        negative_mask = ~labels_matrix  # 不同数据集为True
        triu_mask = torch.triu(torch.ones_like(similarity_matrix, dtype=torch.bool), diagonal=1)
        final_mask = negative_mask & triu_mask
        negative_similarities = similarity_matrix[final_mask]
        batch_basic = torch.tensor(0.0, device=features.device)
        batch_hard = torch.tensor(0.0, device=features.device)
        if len(negative_similarities) > 0:
            batch_basic = torch.mean(torch.relu(negative_similarities - target_similarity))
            high_sim_mask = negative_similarities > hard_threshold
            if high_sim_mask.any():
                high_similarities = negative_similarities[high_sim_mask]
                batch_hard = torch.mean((high_similarities - hard_threshold) ** 2)
        margin_loss = batch_basic + 2.0 * batch_hard
        
        # ===== 跨batch内存队列负样本（逐样本Top-K） =====
        if (self.bank_features is not None) and (self.bank_labels is not None) and (self.bank_features.size(0) > 0):
            try:
                bank_feats_cpu = self.bank_features
                bank_labs_cpu = self.bank_labels
                device = features_norm.device
                dtype = features_norm.dtype
                chunk_size = max(1, getattr(self, 'bank_chunk_size', 512))
                topk_cap = max(1, getattr(self, 'topk_max', 2048))
                B = features_norm.size(0)
                # 将总体top-K配额分配到逐样本
                topk_per_sample = max(8, min(256, topk_cap // max(1, B)))
                per_sample_vals = [[] for _ in range(B)]
                fill_val = -2.0  # 小于余弦相似度下界，避免被选中
                for start in range(0, bank_feats_cpu.size(0), chunk_size):
                    end = min(start + chunk_size, bank_feats_cpu.size(0))
                    bank_chunk = bank_feats_cpu[start:end].to(device=device, dtype=dtype, non_blocking=True)
                    bank_labs_chunk = bank_labs_cpu[start:end].to(device=labels.device, non_blocking=True)
                    with amp.autocast(enabled=self.use_amp):
                        sim_chunk = torch.matmul(features_norm, bank_chunk.T)  # [B, C]
                    neg_mask_chunk = labels.unsqueeze(1) != bank_labs_chunk.unsqueeze(0)  # [B, C]
                    # 屏蔽非负样本位置
                    masked = sim_chunk.masked_fill(~neg_mask_chunk, fill_val)
                    # 逐样本Top-K
                    k_row = min(topk_per_sample, masked.size(1))
                    row_topk, _ = torch.topk(masked, k=k_row, dim=1)
                    # 逐行收集有效值
                    for r in range(B):
                        rv = row_topk[r]
                        rv = rv[rv > (fill_val + 1e-6)]  # 去除填充值
                        if rv.numel() > 0:
                            per_sample_vals[r].append(rv)
                    del bank_chunk, bank_labs_chunk, sim_chunk, neg_mask_chunk, masked, row_topk
                # 整合各样本跨chunk的Top-K
                final_topk = []
                for r in range(B):
                    if len(per_sample_vals[r]) == 0:
                        continue
                    row_all = torch.cat(per_sample_vals[r])
                    K_r = min(topk_per_sample, row_all.numel())
                    vals_r, _ = torch.topk(row_all, k=K_r)
                    final_topk.append(vals_r)
                if len(final_topk) > 0:
                    topk_vals = torch.cat(final_topk)
                    bank_basic = torch.mean(torch.relu(topk_vals - target_similarity))
                    bank_high_mask = topk_vals > hard_threshold
                    if bank_high_mask.any():
                        bank_high = topk_vals[bank_high_mask]
                        bank_hard = torch.mean((bank_high - hard_threshold) ** 2)
                    else:
                        bank_hard = torch.tensor(0.0, device=features.device)
                    margin_loss = margin_loss + bank_basic + 2.0 * bank_hard
                del per_sample_vals
            except RuntimeError as e:
                err = str(e).lower()
                if ('out of memory' in err) or ('cuda' in err):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(self, 'bank_chunk_size'):
                        self.bank_chunk_size = max(64, int(self.bank_chunk_size // 2))
                    if hasattr(self, 'topk_max'):
                        self.topk_max = max(512, int(self.topk_max // 2))
        
        if torch.isnan(margin_loss) or torch.isinf(margin_loss):
            logger.error(f"⚠️ margin_loss is NaN/Inf: {margin_loss.item()}")
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        return margin_loss

    def center_loss(self, features, labels):
        if features.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            feats_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
            for lbl in labels.unique().tolist():
                mask = (labels == lbl)
                if not mask.any():
                    continue
                batch_mean = feats_norm[mask].mean(dim=0)
                if lbl not in self.prototypes:
                    self.prototypes[lbl] = batch_mean.detach()
                else:
                    self.prototypes[lbl] = self.proto_momentum * self.prototypes[lbl] + (1 - self.proto_momentum) * batch_mean.detach()
        feats_norm2 = F.normalize(features, p=2, dim=1, eps=1e-8)
        losses = []
        for i in range(features.size(0)):
            lbl = int(labels[i].item())
            if lbl in self.prototypes:
                center = self.prototypes[lbl].to(self.device).detach()
                losses.append(torch.mean((feats_norm2[i] - center) ** 2))
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def prototype_separation_loss(self, features, labels, target_similarity=0.4, hard_threshold=0.7):
        """将每个样本与其他类的原型拉开，产生对features的梯度"""
        if features.size(0) == 0 or not isinstance(self.prototypes, dict) or len(self.prototypes) < 2:
            return torch.tensor(0.0, device=self.device)
        feats_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        basic_terms = []
        hard_terms = []
        for i in range(feats_norm.size(0)):
            li = int(labels[i].item())
            fi = feats_norm[i]
            for pl, proto in self.prototypes.items():
                if pl == li or proto is None:
                    continue
                p = F.normalize(proto.to(self.device), p=2, dim=0, eps=1e-8)
                s = torch.dot(fi, p)
                b = torch.relu(s - target_similarity)
                h = torch.relu(s - hard_threshold)
                basic_terms.append(b)
                if h.item() > 0:
                    hard_terms.append(h * h)
        if len(basic_terms) == 0:
            return torch.tensor(0.0, device=self.device)
        loss_basic = torch.stack(basic_terms).mean()
        loss_hard = torch.stack(hard_terms).mean() if len(hard_terms) > 0 else torch.tensor(0.0, device=self.device)
        loss = loss_basic + 2.0 * loss_hard
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.device)
        return loss

    def binary_consistency_loss(self, features_original, features_attacked, epoch=0):
        """二值化一致性损失：对攻击前后特征的软二值结果进行对齐，并拉开与阈值的距离。"""
        if features_original.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        if torch.isnan(features_original).any() or torch.isinf(features_original).any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        if torch.isnan(features_attacked).any() or torch.isinf(features_attacked).any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        median_orig = torch.median(features_original, dim=1, keepdim=True)[0]
        median_attack = torch.median(features_attacked, dim=1, keepdim=True)[0]

        temp = max(self.adaptive_temp.get_temperature(epoch), 0.01)

        logits_orig = (features_original - median_orig) / temp
        logits_attack = (features_attacked - median_attack) / temp
        logits_orig = torch.clamp(logits_orig, min=-20.0, max=20.0)
        logits_attack = torch.clamp(logits_attack, min=-20.0, max=20.0)

        combined = torch.cat([features_original, features_attacked], dim=0)
        inst_median_dim = torch.median(combined, dim=0, keepdim=False)[0]
        if self.ema_median is None:
            self.ema_median = inst_median_dim.detach().to(features_original.device, dtype=features_original.dtype)
        else:
            if self.ema_median.shape != inst_median_dim.shape or self.ema_median.device != features_original.device:
                self.ema_median = inst_median_dim.detach().to(features_original.device, dtype=features_original.dtype)
            else:
                self.ema_median = self.ema_momentum * self.ema_median + (1.0 - self.ema_momentum) * inst_median_dim.detach()
        ema_median = self.ema_median.view(1, -1)
        logits_orig_shared = (features_original - ema_median) / temp
        logits_attack_shared = (features_attacked - ema_median) / temp
        logits_orig_shared = torch.clamp(logits_orig_shared, min=-20.0, max=20.0)
        logits_attack_shared = torch.clamp(logits_attack_shared, min=-20.0, max=20.0)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits_orig,
            torch.sigmoid(logits_attack.detach())
        ) + F.binary_cross_entropy_with_logits(
            logits_attack,
            torch.sigmoid(logits_orig.detach())
        )
        bce_loss = bce_loss / 2.0
        if torch.isnan(bce_loss) or torch.isinf(bce_loss):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        margin_orig = torch.abs(features_original - median_orig)
        margin_attack = torch.abs(features_attacked - median_attack)
        margin_loss = torch.mean(torch.relu(0.5 - margin_orig)) + torch.mean(torch.relu(0.5 - margin_attack))
        margin_loss = margin_loss / 2.0
        if torch.isnan(margin_loss) or torch.isinf(margin_loss):
            margin_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        logit_mse = F.mse_loss(logits_orig_shared, logits_attack_shared)

        total_loss = bce_loss + 0.2 * margin_loss + 0.1 * logit_mse
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return total_loss

    def bit_balance_loss(self, features):
        if features.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        feats = F.normalize(features, p=2, dim=1, eps=1e-8)
        col_mean = feats.mean(dim=0)
        return torch.mean(col_mean ** 2)

    def decorrelation_loss(self, features):
        B = features.size(0)
        if B < 4:
            return torch.tensor(0.0, device=self.device)
        x = F.normalize(features, p=2, dim=1, eps=1e-8)
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (B - 1 + 1e-6)
        I = torch.eye(cov.size(0), device=cov.device)
        off_diag = cov - I
        return (off_diag ** 2).sum() / (cov.size(0) ** 2)
    
    def label_aware_uniqueness_loss(self, features, labels):
        if features.size(0) < 2:
            return torch.tensor(0.0, device=self.device)
        if torch.isnan(features).any() or torch.isinf(features).any():
            return torch.tensor(0.0, device=self.device)
        feats_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        sim = torch.matmul(feats_norm, feats_norm.T)
        sim = torch.clamp(sim, -1.0, 1.0)
        labels = labels.view(-1, 1)
        cross_mask = labels != labels.T
        diag = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        cross_mask = cross_mask & (~diag)
        if not cross_mask.any():
            return torch.tensor(0.0, device=self.device)
        cross_sim = sim[cross_mask]
        # 基础惩罚：所有跨类相似度的均值
        base = torch.mean(cross_sim)
        # 分层惩罚：对高相似度施加更重惩罚
        high = torch.mean(torch.relu(cross_sim - 0.5))
        extreme = torch.mean(torch.relu(cross_sim - 0.7))
        # 灾难级惩罚：激进配置，对NC>0.80的情况施加平方惩罚（降低阈值）
        disaster_mask = cross_sim > 0.80
        if disaster_mask.any():
            disaster_vals = cross_sim[disaster_mask]
            disaster = torch.mean((disaster_vals - 0.80) ** 2)  # 平方惩罚
        else:
            disaster = torch.tensor(0.0, device=self.device)
        # 超灾难级：激进配置，对NC>0.90的情况施加三次方惩罚（降低阈值）
        ultra_disaster_mask = cross_sim > 0.90
        if ultra_disaster_mask.any():
            ultra_vals = cross_sim[ultra_disaster_mask]
            ultra_disaster = torch.mean((ultra_vals - 0.90) ** 3)  # 三次方惩罚
        else:
            ultra_disaster = torch.tensor(0.0, device=self.device)
        loss = base + 3.0 * high + 5.0 * extreme + 10.0 * disaster + 20.0 * ultra_disaster
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.device)
        return loss
    
    def prototype_anchor_loss(self, features, labels):
        """
        方案1：原型锚定损失
        鲁棒性约束：特征必须靠近自己类别的原型
        每个图的所有版本（原始+攻击）都应该围绕同一个原型聚集
        """
        if features.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 归一化特征和原型
        feats_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        protos_norm = F.normalize(self.prototype_bank, p=2, dim=1, eps=1e-8)
        
        # 为每个样本选择对应的原型
        proto_features = protos_norm[labels]  # [batch_size, feat_dim]
        
        # 计算特征与原型的余弦相似度
        similarity = F.cosine_similarity(feats_norm, proto_features, dim=1)  # [batch_size]
        
        # 损失：1 - 相似度（越相似损失越小）
        loss = 1.0 - similarity.mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.device)
        
        return loss
    
    def orthogonality_loss(self):
        """
        方案5：正交性约束损失
        唯一性约束：不同类别的原型必须正交（理论NC≈0）
        """
        # 归一化原型向量
        protos_norm = F.normalize(self.prototype_bank, p=2, dim=1, eps=1e-8)  # [num_classes, feat_dim]
        
        # 计算Gram矩阵（原型之间的内积）
        gram = torch.matmul(protos_norm, protos_norm.T)  # [num_classes, num_classes]
        
        # 理想情况：单位矩阵（对角线=1，非对角线=0）
        identity = torch.eye(self.num_classes, device=self.device)
        
        # 最小化Gram矩阵与单位矩阵的Frobenius范数
        ortho_loss = torch.norm(gram - identity, p='fro') ** 2
        
        # 额外惩罚：对角线外元素的绝对值（确保非对角线趋向0）
        off_diagonal = gram * (1 - identity)
        off_diag_penalty = (off_diagonal ** 2).sum()
        
        loss = ortho_loss + 2.0 * off_diag_penalty
        
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.device)
        
        return loss
    
    def hard_adversarial_loss(self, features, labels):
        """
        方案4：对抗性难样本增强
        专门针对高NC的难样本对施加指数级惩罚
        """
        if features.size(0) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # 归一化特征
        feats_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(feats_norm, feats_norm.T)  # [batch, batch]
        sim_matrix = torch.clamp(sim_matrix, -1.0, 1.0)
        
        # 构建跨类mask（不同标签的样本对）
        labels_expanded = labels.view(-1, 1)
        cross_mask = (labels_expanded != labels_expanded.T)
        
        # 排除对角线
        diag = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=self.device)
        cross_mask = cross_mask & (~diag)
        
        if not cross_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        # 提取跨类相似度
        cross_sim = sim_matrix[cross_mask]
        
        # 找出难样本（相似度高于阈值）
        hard_mask = cross_sim > self.hard_pair_threshold
        
        if not hard_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        hard_sims = cross_sim[hard_mask]
        
        # 指数级惩罚：相似度越高，惩罚指数增长
        # loss = exp(k * (sim - threshold))，k=10使得惩罚快速增长
        penalties = torch.exp(10.0 * (hard_sims - self.hard_pair_threshold))
        
        # 平均难样本惩罚
        loss = penalties.mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.device)
        
        return loss

    def prototype_correlation_loss(self, target_similarity=0.3, hard_threshold=0.6):
        if not isinstance(self.prototypes, dict) or len(self.prototypes) < 2:
            return torch.tensor(0.0, device=self.device)
        protos = []
        for _, proto in self.prototypes.items():
            if proto is None:
                continue
            protos.append(proto.to(self.device))
        if len(protos) < 2:
            return torch.tensor(0.0, device=self.device)
        P = torch.stack(protos, dim=0)
        P = F.normalize(P, p=2, dim=1, eps=1e-8)
        S = torch.matmul(P, P.T)
        C = S.size(0)
        if C <= 1:
            return torch.tensor(0.0, device=self.device)
        mask = torch.ones_like(S, dtype=torch.bool)
        diag = torch.eye(C, dtype=torch.bool, device=S.device)
        mask = mask & (~diag)
        off_vals = S[mask]
        if off_vals.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        basic = torch.mean(torch.relu(off_vals - target_similarity))
        high_mask = off_vals > hard_threshold
        if high_mask.any():
            high_vals = off_vals[high_mask]
            hard = torch.mean((high_vals - hard_threshold) ** 2)
        else:
            hard = torch.tensor(0.0, device=self.device)
        loss = basic + 2.0 * hard
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"⚠️ prototype_correlation_loss is NaN/Inf: {loss.item()}")
            return torch.tensor(0.0, device=self.device)
        return loss
    
    def train_epoch(self, original_graphs, attacked_graphs, epoch):
        """训练一个epoch"""
        self.model.train()

        # Epoch开始时清理CUDA缓存和同步
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 确保之前的操作完成

        schedule = self._apply_epoch_schedule(epoch)
        all_pairs, all_labels = self._collect_pairs(original_graphs, attacked_graphs)

        if not all_pairs:
            logger.warning("没有找到训练数据对")
            return (0.0,) * 16

        label_counts: Dict[int, int] = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        total_pairs = len(all_pairs)
        total_positive_pairs = sum(count * (count - 1) // 2 for count in label_counts.values())
        total_negative_pairs = total_pairs * (total_pairs - 1) // 2 - total_positive_pairs

        logger.info("训练数据统计:")
        logger.info("  总样本对数: %d", total_pairs)
        logger.info("  原图类型数: %d", len(label_counts))
        logger.info("  各原图样本数: %s", label_counts)
        logger.info("  正样本对数: %d", total_positive_pairs)
        logger.info("  负样本对数: %d", total_negative_pairs)
        logger.info("  📊 采样策略: 平衡采样（每个batch保证至少1个正样本对）")

        total_batches = max(1, math.ceil(total_pairs / self.batch_size))
        log_interval = max(1, total_batches // 20)
        logger.info("  总batch数: %d, 日志间隔: 每%d个batch", total_batches, log_interval)

        sum_keys = [
            "loss",
            "contrastive",
            "similarity",
            "diversity",
            "margin",
            "center",
            "balance",
            "decor",
            "proto_sep",
            "proto_corr",
            "binary",
            "uniqueness",
            "proto_anchor",
            "ortho",
            "hard_adversarial",
            "grad_norm",
        ]
        sums = {k: 0.0 for k in sum_keys}
        num_batches = 0

        for batch_idx, (batch_pairs, batch_labels_list) in enumerate(
            self._iter_batches(all_pairs, all_labels), start=1
        ):
            batch_features = self._extract_batch_features(batch_pairs, batch_idx)
            if batch_features[0] is None:
                logger.warning("Batch %d: 特征提取失败，跳过", batch_idx)
                continue

            batch_original, batch_attacked = batch_features
            batch_labels = torch.tensor(batch_labels_list, device=self.device)

            losses, combined, combined_labels = self._compute_loss_components(
                batch_original, batch_attacked, batch_labels, schedule, epoch
            )
            total_batch_loss = self._compose_total_loss(losses, schedule)

            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                logger.warning("Batch %d: total_loss异常，跳过", batch_idx)
                continue

            grad_norm = self._apply_gradients(total_batch_loss)

            # 更新Memory Bank
            self._update_memory_bank(combined.detach(), combined_labels.detach())

            # 累积损失（使用.item()立即转为Python标量，释放tensor）
            sums["loss"] += float(total_batch_loss.detach().item())
            for key in (
                "contrastive",
                "similarity",
                "diversity",
                "margin",
                "center",
                "balance",
                "decor",
                "proto_sep",
                "proto_corr",
                "binary",
                "uniqueness",
                "proto_anchor",
                "ortho",
                "hard_adversarial",
            ):
                sums[key] += float(losses[key].detach().item())
            sums["grad_norm"] += grad_norm
            num_batches += 1

            if (
                batch_idx % log_interval == 0
                or batch_idx == 1
                or batch_idx == total_batches
            ):
                batch_label_counts: Dict[int, int] = {}
                for lbl in batch_labels_list:
                    batch_label_counts[lbl] = batch_label_counts.get(lbl, 0) + 1
                positive = sum(count * (count - 1) // 2 for count in batch_label_counts.values())
                batch_size = len(batch_labels_list)
                negative = batch_size * (batch_size - 1) // 2 - positive
                logger.info(
                    "  Batch %d/%d: 标签分布=%s, 正样本对=%d, 负样本对=%d",
                    batch_idx,
                    total_batches,
                    batch_label_counts,
                    positive,
                    negative,
                )

            # 强制清理中间变量和GPU内存（只在成功处理完batch后清理）
            del losses, combined, combined_labels
            del batch_original, batch_attacked, total_batch_loss
            # 清理batch变量的引用
            del batch_pairs, batch_labels, batch_labels_list

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                # 每50个batch强制同步和深度清理
                if batch_idx % 50 == 0:
                    torch.cuda.synchronize()  # 确保所有操作完成
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"  💾 GPU显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")

        if num_batches == 0:
            logger.warning("所有batch均被跳过，返回0损失")
            return (0.0,) * 16

        self.scheduler.step()

        avg = {k: sums[k] / num_batches for k in sum_keys}

        self.training_history["epoch_losses"].append(avg["loss"])
        self.training_history["contrastive_losses"].append(avg["contrastive"])
        self.training_history["similarity_losses"].append(avg["similarity"])
        self.training_history["diversity_losses"].append(avg["diversity"])
        self.training_history["margin_losses"].append(avg["margin"])
        self.training_history["center_losses"].append(avg["center"])
        self.training_history["balance_losses"].append(avg["balance"])
        self.training_history["decor_losses"].append(avg["decor"])
        self.training_history["proto_sep_losses"].append(avg["proto_sep"])
        self.training_history["proto_corr_losses"].append(avg["proto_corr"])
        self.training_history["binary_losses"].append(avg["binary"])
        self.training_history["uniqueness_losses"].append(avg["uniqueness"])
        self.training_history["proto_anchor_losses"].append(avg["proto_anchor"])
        self.training_history["ortho_losses"].append(avg["ortho"])
        self.training_history["hard_adversarial_losses"].append(avg["hard_adversarial"])
        self.training_history["gradient_norms"].append(avg["grad_norm"])
        self.training_history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        return (
            avg["loss"],
            avg["contrastive"],
            avg["similarity"],
            avg["diversity"],
            avg["margin"],
            avg["center"],
            avg["balance"],
            avg["decor"],
            avg["proto_sep"],
            avg["proto_corr"],
            avg["binary"],
            avg["uniqueness"],
            avg["proto_anchor"],
            avg["ortho"],
            avg["hard_adversarial"],
            avg["grad_norm"],
        )
    
    def _train_epoch_with_adaptive_batch_size(self, original_graphs, attacked_graphs, epoch):
        """带自适应batch_size的训练（OOM时自动降低batch_size并重试）"""
        max_retries = 3  # 最多重试3次
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # 尝试使用当前batch_size训练
                return self.train_epoch(original_graphs, attacked_graphs, epoch)
            
            except RuntimeError as e:
                error_str = str(e)
                # 检查是否是OOM错误
                if "out of memory" in error_str.lower() or "cuda" in error_str.lower():
                    retry_count += 1
                    
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # 计算新的batch_size（减半，但不低于min_batch_size）
                    old_batch_size = self.batch_size
                    new_batch_size = max(self.min_batch_size, self.batch_size // 2)
                    
                    if new_batch_size == old_batch_size:
                        # 已经是最小batch_size，无法再降低
                        logger.error(f"❌ 显存不足且batch_size已是最小值({self.min_batch_size})，无法继续训练！")
                        logger.error(f"   错误信息: {error_str}")
                        logger.error(f"   建议: 关闭其他GPU程序或升级显卡")
                        raise e
                    
                    self.batch_size = new_batch_size
                    logger.warning("")
                    logger.warning("⚠️" * 40)
                    logger.warning(f"⚠️ 检测到显存不足！")
                    logger.warning(f"⚠️ 自动降低 batch_size: {old_batch_size} → {new_batch_size}")
                    logger.warning(f"⚠️ 重试第 {retry_count}/{max_retries} 次...")
                    logger.warning("⚠️" * 40)
                    logger.warning("")
                    
                    # 等待一段时间让GPU内存完全释放
                    import time
                    time.sleep(2)
                    
                    # 重新计算batch数量并记录
                    logger.info(f"🔄 使用新的batch_size={self.batch_size}重新训练epoch {epoch+1}")
                    
                else:
                    # 不是OOM错误，直接抛出
                    raise e
        
        # 超过最大重试次数
        logger.error(f"❌ 已重试{max_retries}次，仍然失败！")
        raise RuntimeError("自适应batch_size机制失败，训练终止")
    
    def train(self, original_graphs, attacked_graphs, num_epochs=20, resume_from_checkpoint=True):
        """训练模型（支持自适应batch_size和checkpoint恢复）"""
        # 更新memory_refresh_marks以适应实际的epoch数量
        self.memory_refresh_marks = default_memory_refresh_marks(num_epochs)

        logger.info(f"开始训练GCN模型（{num_epochs}个epoch）...")
        logger.info("训练目标：提取矢量地图的鲁棒特征，抵抗RST攻击")
        logger.info(f"自适应batch_size策略: 初始={self.initial_batch_size}, 最小={self.min_batch_size}")
        logger.info(f"Memory Bank刷新标记点: {self.memory_refresh_marks}")
        
        # 尝试从checkpoint恢复
        start_epoch = 0
        best_loss = float('inf')
        # 唯一性主导阶段相关参数（当前不启用基于唯一性的早停，仅记录变量以便未来扩展）
        unique_phase_start_epoch = 0
        best_uni_score = float('inf')
        patience = 10  # 早停耐心值：如果连续10个epoch损失未提升则停止（围绕最优点附近）
        patience_counter = 0
        min_epochs_for_early_stop = max(8, int(num_epochs * 0.8))  # 至少训练80%的epoch但不低于8个epoch
        
        checkpoint_name = os.environ.get("VGAT_CHECKPOINT_NAME", f"{self.model_basename}_checkpoint.pth")
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'models', checkpoint_name)
        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            logger.info("")
            logger.info("🔄 检测到checkpoint文件，尝试恢复训练...")
            checkpoint = self.load_checkpoint(checkpoint_path)
            if checkpoint is not None:
                start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
                best_loss = checkpoint['best_loss']
                patience_counter = checkpoint['patience_counter']
                logger.info(f"✅ 将从 Epoch {start_epoch + 1} 继续训练")
                logger.info("")
            else:
                logger.warning("⚠️ Checkpoint加载失败，从头开始训练")
                logger.info("")
        else:
            if resume_from_checkpoint:
                logger.info("🆕 未找到checkpoint文件，从头开始训练")
                logger.info("")
        
        # 创建CSV文件记录损失（如果是恢复训练，追加模式）
        loss_csv_path = os.path.join(os.path.dirname(__file__), 'logs', 'training_loss.csv')
        os.makedirs(os.path.dirname(loss_csv_path), exist_ok=True)
        if start_epoch == 0:
            # 从头开始，创建新文件
            with open(loss_csv_path, 'w', encoding='utf-8') as f:
                f.write('epoch,total_loss,contrastive_loss,similarity_loss,diversity_loss,margin_loss,uniqueness_loss,binary_loss,grad_norm,learning_rate\n')  # 🔥 新增margin_loss列
        logger.info(f"损失记录文件: {loss_csv_path}")
        
        import time
        for epoch in tqdm(range(start_epoch, num_epochs), desc="训练进度", initial=start_epoch, total=num_epochs):
            # Epoch开始标记
            epoch_start_time = time.time()
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"📊 Epoch {epoch+1}/{num_epochs} 开始")
            logger.info("=" * 80)
            
            # 训练（带自适应batch_size机制）
            train_loss, contrastive_loss, similarity_loss, diversity_loss, margin_loss, center_loss, balance_loss, decor_loss, proto_sep_loss, proto_corr_loss, binary_loss, uniqueness_loss, proto_anchor_loss, ortho_loss, hard_adversarial_loss, grad_norm = self._train_epoch_with_adaptive_batch_size(original_graphs, attacked_graphs, epoch)
            
            # Epoch结束，计算耗时
            epoch_time = time.time() - epoch_start_time
            
            # Epoch结束时强制清理GPU内存（三步清理法）
            if torch.cuda.is_available():
                # 第一步：清空Python端的引用
                import gc
                gc.collect()
                
                # 第二步：清空CUDA缓存
                torch.cuda.empty_cache()
                
                # 第三步：强制同步并再次清理
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                allocated = torch.cuda.memory_allocated() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"💾 Epoch结束显存: 当前={allocated:.2f}GB, 峰值={max_allocated:.2f}GB")
                torch.cuda.reset_peak_memory_stats()
                
                # 如果显存仍然很高（>10GB），警告用户
                if allocated > 10.0:
                    logger.warning(f"⚠️ Epoch结束后显存仍然较高({allocated:.2f}GB)，可能存在显存泄漏")
            
            # Warmup block for robustness phase（Epoch>=robustness_start后短暂提高LR/温度）
            if epoch + 1 >= self.robustness_start_epoch and not getattr(self, 'robust_phase_active', False):
                self.robust_phase_active = True
            if epoch + 1 == self.robustness_start_epoch:
                self._enter_robustness_warmup(current_lr)
            if self.warmup_active:
                current_lr = self._apply_warmup_schedule(epoch, current_lr)

            # 智能Memory Bank刷新：基于epoch标记点刷新（模仿VGAT-IMPROVED）
            if epoch + 1 in self.memory_refresh_marks:
                if self.last_memory_refresh_epoch != epoch:
                    self._refresh_memory_bank(force=True, keep_ratio=0.6)
                    self.last_memory_refresh_epoch = epoch

            # 记录损失到CSV
            current_lr = self.optimizer.param_groups[0]['lr']
            with open(loss_csv_path, 'a', encoding='utf-8') as f:
                f.write(f'{epoch+1},{train_loss:.6f},{contrastive_loss:.6f},{similarity_loss:.6f},{diversity_loss:.6f},{margin_loss:.6f},{uniqueness_loss:.6f},{binary_loss:.6f},{grad_norm:.6f},{current_lr:.8f}\n')  # 🔥 新增margin_loss
            
            # 早停机制和模型保存：当前仅根据总训练损失选择最佳模型
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                model_best_path = os.path.join(os.path.dirname(__file__), 'models', f'{self.model_basename}_best.pth')
                self.save_model(model_best_path)
                logger.info("💾 保存最佳模型")
            else:
                patience_counter += 1
            
            # 每个epoch都保存final模型（覆盖式）
            final_model_path = os.path.join(os.path.dirname(__file__), 'models', f'{self.model_basename}_final.pth')
            self.save_model(final_model_path)
            logger.info("💾 保存当前epoch的final模型")
            
            # 保存checkpoint（用于恢复训练）
            self.save_checkpoint(checkpoint_path, epoch, best_loss, patience_counter)
            logger.info("💾 保存checkpoint（可恢复训练）")
            
            # 每个epoch结束时都打印损失汇总
            logger.info("")
            logger.info("─" * 80)
            logger.info(f"Epoch {epoch+1}/{num_epochs} 完成 | 耗时: {epoch_time/60:.2f}分钟")
            logger.info(f"总损失: {train_loss:.6f}")
            logger.info(f"   ├─ 对比损失: {contrastive_loss:.6f} (权重{self.w_contrastive})")
            logger.info(f"   ├─ 相似损失: {similarity_loss:.6f} (权重{self.w_similarity})")
            logger.info(f"   ├─ 多样性损失: {diversity_loss:.6f} (权重{self.w_diversity})")
            logger.info(f"   ├─ 类间边界损失: {margin_loss:.6f} (权重{self.w_margin:.2f}, margin调度={self.margin_base}->{self.margin_target}, 当前margin={getattr(self, '_sched_snapshot', {}).get('current_margin', None)}) ")
            logger.info(f"   ├─ 原型分离损失: {proto_sep_loss:.6f} (权重{self.w_proto_sep:.2f})")
            logger.info(f"   ├─ 原型相关性损失: {proto_corr_loss:.6f} (权重{self.w_proto_corr:.2f})")
            logger.info(f"   ├─ 中心损失: {center_loss:.6f} (权重{self.w_center:.2f}), 位平衡: {balance_loss:.6f} (权重≤{self.w_balance_max:.2f}), 去相关: {decor_loss:.6f} (权重≤{self.w_decor_max:.2f})")
            logger.info(f"   ├─ 唯一性损失: {uniqueness_loss:.6f} (权重{self.w_uniqueness:.2f})")
            logger.info(f"   ├─ 二值一致性损失: {binary_loss:.6f} (权重{self.w_binary:.2f})")
            logger.info("")
            logger.info("   🚀 新方案损失（原型锚定 + 正交约束 + 难样本挖掘）:")
            logger.info(f"   ├─ 原型锚定损失: {proto_anchor_loss:.6f} (权重{self.w_proto_anchor:.2f}) [鲁棒性]")
            logger.info(f"   ├─ 正交约束损失: {ortho_loss:.6f} (权重{self.w_ortho:.2f}) [唯一性]")
            logger.info(f"   ├─ 难样本对抗损失: {hard_adversarial_loss:.6f} (权重{self.w_hard_adversarial:.2f}) [强化唯一性]")
            logger.info("")
            logger.info(f"   ├─ 当前权重: contrastive={self.w_contrastive:.2f}, similarity={self.w_similarity:.2f}, diversity={self.w_diversity:.2f}, margin={self.w_margin:.2f}, center={self.w_center:.2f}, proto_sep={self.w_proto_sep:.2f}, proto_corr={self.w_proto_corr:.2f}, balance≤{self.w_balance_max:.2f}, decor≤{self.w_decor_max:.2f}")
            logger.info(f"   ├─ 当前温度T: {getattr(self, '_sched_snapshot', {}).get('temperature', None)}; 当前hard阈值: {getattr(self, '_sched_snapshot', {}).get('current_hard', None)}")
            logger.info(f"   ├─ 梯度范数: {grad_norm:.6f}")
            logger.info(f"   └─ 学习率: {current_lr:.8f}")
            logger.info(f"Batch大小 : {self.batch_size} (初始: {self.initial_batch_size})")
            logger.info(f"  耐心计数  : {patience_counter}/{patience}")
            # 记忆库统计
            mb_size = int(self.bank_features.size(0)) if getattr(self, 'bank_features', None) is not None else 0
            logger.info(f"MemoryBank: size={mb_size}, chunk={getattr(self, 'bank_chunk_size', None)}, topk_max={getattr(self, 'topk_max', None)}, stored_device=cpu")
            
            # 每3个epoch打印额外的特征统计
            if (epoch + 1) % 3 == 0:
                # 记录详细的特征统计信息
                if hasattr(self, 'model') and self.model is not None:
                    with torch.no_grad():
                        # 随机选择一个batch计算特征统计
                        sample_features = []
                        for graph_name, original_graph in list(original_graphs.items())[:2]:
                            original_graph = original_graph.to(self.device)
                            features = self.model(original_graph.x, original_graph.edge_index)
                            sample_features.append(features.cpu().numpy())
                        
                        if sample_features:
                            sample_features = np.concatenate(sample_features, axis=0)
                            feature_mean = np.mean(sample_features)
                            feature_std = np.std(sample_features)
                            feature_min = np.min(sample_features)
                            feature_max = np.max(sample_features)
                            
                            logger.info(f"🔍 特征统计: 均值={feature_mean:.4f}, 标准差={feature_std:.4f}, 范围=[{feature_min:.4f}, {feature_max:.4f}]")
                            
                            # 记录到训练历史
                            self.training_history['feature_stats'].append({
                                'mean': feature_mean,
                                'std': feature_std,
                                'min': feature_min,
                                'max': feature_max
                            })
            
            # 早停
            if patience_counter >= patience and (epoch + 1) >= min_epochs_for_early_stop:
                logger.warning(f"连续{patience}个epoch没有改善，提前停止训练")
                break
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves(loss_csv_path)
        
        logger.info(f"训练完成！最佳损失值: {best_loss:.6f}")
        return best_loss
    
    def save_model(self, model_path):
        """保存模型（仅模型权重）"""
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.get_model_config(),
            'ablation_config': self.get_ablation_config(),
        }, model_path)
        logger.info(f"模型已保存到: {model_path}")
    
    def save_checkpoint(self, checkpoint_path, epoch, best_loss, patience_counter):
        """保存完整的checkpoint（用于恢复训练）"""
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'best_loss': best_loss,
            'patience_counter': patience_counter,
            'batch_size': self.batch_size,
            'initial_batch_size': self.initial_batch_size,
            'training_history': self.training_history,
            'model_config': self.get_model_config(),
            'ablation_config': self.get_ablation_config(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint已保存到: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint（恢复训练）"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint文件不存在: {checkpoint_path}")
            return None
        
        try:
            # 修复PyTorch 2.6的weights_only问题：显式设置为False
            # 因为checkpoint中包含numpy对象和训练历史数据
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.use_amp and checkpoint.get('scaler_state_dict') is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.batch_size = checkpoint.get('batch_size', self.batch_size)
            self.initial_batch_size = checkpoint.get('initial_batch_size', self.initial_batch_size)
            self.training_history = checkpoint.get('training_history', self.training_history)
            
            logger.info(f"✅ 成功加载checkpoint: {checkpoint_path}")
            logger.info(f"   恢复到Epoch {checkpoint['epoch']}, 最佳损失: {checkpoint['best_loss']:.6f}")
            logger.info(f"   Batch大小: {self.batch_size}")
            
            return checkpoint
        
        except Exception as e:
            logger.error(f"❌ 加载checkpoint失败: {e}")
            return None
    
    def save_training_history(self):
        """保存训练历史"""
        # 将训练历史保存到VGCN文件夹
        history_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(history_dir, f"training_history_{timestamp}.json")
        
        # 转换numpy数组为列表以便JSON序列化
        history_data = {}
        for key, value in self.training_history.items():
            if key == 'feature_stats':
                # 转换feature_stats中的numpy类型为Python原生类型
                converted_stats = []
                for stat_dict in value:
                    converted_dict = {}
                    for stat_key, stat_value in stat_dict.items():
                        # 将numpy类型转换为float
                        if hasattr(stat_value, 'item'):
                            converted_dict[stat_key] = float(stat_value.item())
                        else:
                            converted_dict[stat_key] = float(stat_value)
                    converted_stats.append(converted_dict)
                history_data[key] = converted_stats
            else:
                # 确保所有数值都转换为Python原生类型
                history_data[key] = [float(v) if hasattr(v, 'item') else v for v in value]
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练历史已保存到: {history_file}")
    
    def plot_training_curves(self, csv_path):
        """绘制SCI风格的训练曲线"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # 读取CSV数据
            df = pd.read_csv(csv_path)
            
            # 设置SCI风格
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.linewidth'] = 1.2
            plt.rcParams['grid.alpha'] = 0.3
            
            # 创建2x3子图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. 总损失曲线
            axes[0, 0].plot(df['epoch'], df['total_loss'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#2E86AB')
            axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[0, 0].set_ylabel('Total Loss', fontsize=11, fontweight='bold')
            axes[0, 0].set_title('(a) Total Loss', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3, linestyle='--')
            axes[0, 0].tick_params(labelsize=9)
            
            # 2. 对比损失曲线
            axes[0, 1].plot(df['epoch'], df['contrastive_loss'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#A23B72')
            axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[0, 1].set_ylabel('Contrastive Loss', fontsize=11, fontweight='bold')
            axes[0, 1].set_title('(b) Contrastive Loss', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3, linestyle='--')
            axes[0, 1].tick_params(labelsize=9)
            
            # 3. 相似性损失曲线
            axes[0, 2].plot(df['epoch'], df['similarity_loss'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#F18F01')
            axes[0, 2].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[0, 2].set_ylabel('Similarity Loss', fontsize=11, fontweight='bold')
            axes[0, 2].set_title('(c) Similarity Loss', fontsize=12, fontweight='bold')
            axes[0, 2].grid(True, alpha=0.3, linestyle='--')
            axes[0, 2].tick_params(labelsize=9)
            
            # 4. 多样性损失曲线
            axes[1, 0].plot(df['epoch'], df['diversity_loss'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#C73E1D')
            axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('Diversity Loss', fontsize=11, fontweight='bold')
            axes[1, 0].set_title('(d) Diversity Loss', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, linestyle='--')
            axes[1, 0].tick_params(labelsize=9)
            
            # 5. 梯度范数曲线
            axes[1, 1].plot(df['epoch'], df['grad_norm'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#6A994E')
            axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('Gradient Norm', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('(e) Gradient Norm', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, linestyle='--')
            axes[1, 1].tick_params(labelsize=9)
            
            # 6. 学习率曲线
            axes[1, 2].plot(df['epoch'], df['learning_rate'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#BC4B51')
            axes[1, 2].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[1, 2].set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
            axes[1, 2].set_title('(f) Learning Rate', fontsize=12, fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3, linestyle='--')
            axes[1, 2].tick_params(labelsize=9)
            axes[1, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            plt.tight_layout()
            
            # 保存图片
            save_dir = os.path.dirname(csv_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(save_dir, f"training_curves_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"训练曲线图已保存到: {plot_file}")
            
        except ImportError:
            logger.warning("matplotlib未安装，跳过训练曲线绘制")
        except Exception as e:
            logger.error(f"绘制训练曲线时出错: {e}")

class GraphDataLoader:
    """Load paired original/attacked graphs from disk with basic integrity checks."""

    def __init__(self, graph_dir: Optional[str] = None) -> None:
        if graph_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            graph_suffix = os.environ.get("VGAT_GRAPH_SUFFIX", "").strip()
            graph_dir = os.path.join(script_dir, "..", "convertToGraph", "Graph", f"TrainingSet{graph_suffix}")
        self.graph_dir = os.path.abspath(graph_dir)
        logger.info("图数据加载路径: %s", self.graph_dir)

    def load_graph_data(self) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
        """Load original graphs and their attacked variants if both folders exist."""

        original_dir = os.path.join(self.graph_dir, "Original")
        attacked_dir = os.path.join(self.graph_dir, "Attacked")

        if not os.path.exists(original_dir):
            logger.warning("原始数据目录不存在: %s", original_dir)
            return {}, {}

        original_graphs: Dict[str, Any] = {}
        for filename in sorted(os.listdir(original_dir)):
            if not filename.endswith("_graph.pkl"):
                continue
            graph_name = filename.replace("_graph.pkl", "")
            file_path = os.path.join(original_dir, filename)
            try:
                with open(file_path, "rb") as f:
                    original_graphs[graph_name] = pickle.load(f)
            except Exception as exc:
                logger.error("加载原始图失败 %s: %s", filename, exc)

        if not original_graphs:
            logger.warning("原始图目录存在，但未加载到有效图数据")
            return {}, {}

        attacked_graphs: Dict[str, List[Any]] = {name: [] for name in original_graphs}
        if not os.path.exists(attacked_dir):
            logger.warning("被攻击图目录不存在: %s", attacked_dir)
            return original_graphs, attacked_graphs

        for subdir in sorted(os.listdir(attacked_dir)):
            subdir_path = os.path.join(attacked_dir, subdir)
            if not os.path.isdir(subdir_path) or subdir not in attacked_graphs:
                continue
            for filename in sorted(os.listdir(subdir_path)):
                if not filename.endswith("_graph.pkl"):
                    continue
                file_path = os.path.join(subdir_path, filename)
                try:
                    with open(file_path, "rb") as f:
                        attacked_graphs[subdir].append(pickle.load(f))
                except Exception as exc:
                    logger.error("加载被攻击图失败 %s/%s: %s", subdir, filename, exc)

        logger.info("加载了 %d 个原始图", len(original_graphs))
        logger.info("加载了 %d 个被攻击图", sum(len(v) for v in attacked_graphs.values()))
        try:
            max_attacks_per_class = int(os.environ.get("VGAT_MAX_ATTACKS_PER_CLASS", "0"))
        except Exception:
            max_attacks_per_class = 0
        if max_attacks_per_class > 0:
            for subdir, graphs in attacked_graphs.items():
                if len(graphs) > max_attacks_per_class:
                    attacked_graphs[subdir] = graphs[:max_attacks_per_class]
            logger.info("Limit attacked graphs per class to: %d", max_attacks_per_class)

        return original_graphs, attacked_graphs


def main() -> None:
    """Entry point for step-3 GCN training."""

    global logger
    logger = setup_logging()
    logger.info("＝＝＝ 第三步：GCN模型训练 - 矢量地图零水印鲁棒特征提取 ＝＝＝")

    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        logger.info("PYTORCH_CUDA_ALLOC_CONF=%s", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    except Exception:
        logger.warning("设置 PYTORCH_CUDA_ALLOC_CONF 失败，继续训练")

    try:
        configured_seed = int(os.environ.get("VGAT_RANDOM_SEED", "42"))
    except Exception:
        configured_seed = 42
    set_global_seed(configured_seed)
    logger.info("Random seed: %d", configured_seed)
    pooling_mode = normalize_pooling_mode(os.environ.get("VGAT_POOLING_MODE", "dual"))
    disabled_losses = parse_disabled_losses(
        os.environ.get("VGAT_LOSS_ABLATION", "none"),
        os.environ.get("VGAT_DISABLE_LOSSES", ""),
    )
    graph_suffix = os.environ.get("VGAT_GRAPH_SUFFIX", "").strip()
    model_basename = os.environ.get("VGAT_MODEL_BASENAME", "gcn_model")
    logger.info("Pooling mode: %s", pooling_mode)
    logger.info("Disabled losses: %s", sorted(disabled_losses) if disabled_losses else "[]")
    logger.info("Graph suffix: %s", graph_suffix or "<default>")
    logger.info("Model basename: %s", model_basename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    data_loader = GraphDataLoader()
    original_graphs, attacked_graphs = data_loader.load_graph_data()

    if not original_graphs:
        logger.warning("没有找到原始图数据，请先运行第二步")
        return

    attacked_count = sum(len(graphs) for graphs in attacked_graphs.values())
    if attacked_count == 0:
        logger.warning("没有找到被攻击的图数据，请先运行第二步")
        return

    first_graph = next(iter(original_graphs.values()))
    input_dim = first_graph.x.shape[1]
    logger.info("输入特征维度: %d", input_dim)
    logger.info("目标输出维度: 1024 (32x32)")

    model = GCNModel(input_dim=input_dim, hidden_dim=128, output_dim=1024, dropout=0.2, pooling_mode=pooling_mode)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("模型参数数量: %s", f"{total_params:,}")

    try:
        configured_bs = int(os.environ.get("VGAT_BATCH_SIZE", "4"))
    except Exception:
        configured_bs = 4
        logger.warning("VGAT_BATCH_SIZE 解析失败，使用默认值4")

    try:
        configured_epochs = int(os.environ.get("VGAT_NUM_EPOCHS", "12"))
        logger.info(f"环境变量VGAT_NUM_EPOCHS: {os.environ.get('VGAT_NUM_EPOCHS', 'Not set')}")
        logger.info(f"解析后的训练轮数: {configured_epochs}")
    except Exception as e:
        configured_epochs = 12
        logger.warning(f"VGAT_NUM_EPOCHS 解析失败: {e}，使用默认值12")
    resume_from_checkpoint = os.environ.get("VGAT_RESUME_CHECKPOINT", "1").strip().lower() not in {"0", "false", "no"}
    logger.info("Resume from checkpoint: %s", resume_from_checkpoint)

    trainer = ContrastiveTrainer(
        model,
        device.type,
        use_amp=False,
        batch_size=configured_bs,
        disabled_losses=disabled_losses,
    )
    trainer.train(original_graphs, attacked_graphs, num_epochs=configured_epochs, resume_from_checkpoint=resume_from_checkpoint)

    final_model_path = os.path.join(os.path.dirname(__file__), "models", f"{model_basename}.pth")
    trainer.save_model(final_model_path)

    logger.info("模型训练完成，已保存到: %s", final_model_path)
    logger.info("模型将用于：")
    logger.info("  1. 从原始矢量地图提取鲁棒特征")
    logger.info("  2. 与版权图像结合生成零水印")
    logger.info("  3. 验证阶段提取特征并与零水印结合恢复版权图像")

if __name__ == "__main__":
    main()
