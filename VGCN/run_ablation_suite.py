#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility runner for the paper's ablation experiments."""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("VGAT_PYTHON_EXE", sys.executable)
SUMMARY_CSV = PROJECT_ROOT / "VGCN" / "logs" / "ablation_summary.csv"

ABLATIONS: Dict[str, Dict[str, str]] = {
    "full": {
        "VGAT_MODEL_BASENAME": "ablation_full",
    },
    "no_contrastive": {
        "VGAT_MODEL_BASENAME": "ablation_no_contrastive",
        "VGAT_LOSS_ABLATION": "no_contrastive",
    },
    "no_uniqueness_sep": {
        "VGAT_MODEL_BASENAME": "ablation_no_uniqueness_sep",
        "VGAT_LOSS_ABLATION": "no_uniqueness_sep",
    },
    "mean_pool": {
        "VGAT_MODEL_BASENAME": "ablation_mean_pool",
        "VGAT_POOLING_MODE": "mean",
    },
    "max_pool": {
        "VGAT_MODEL_BASENAME": "ablation_max_pool",
        "VGAT_POOLING_MODE": "max",
    },
    "knn_k6": {
        "VGAT_MODEL_BASENAME": "ablation_knn_k6",
        "VGAT_KNN_K": "6",
        "VGAT_GRAPH_SUFFIX": "_k6",
    },
    "knn_k10": {
        "VGAT_MODEL_BASENAME": "ablation_knn_k10",
        "VGAT_KNN_K": "10",
        "VGAT_GRAPH_SUFFIX": "_k10",
    },
}


def run_step(cmd: List[str], env: Dict[str, str], cwd: Path = PROJECT_ROOT) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def prepare_training_graphs(env: Dict[str, str]) -> None:
    run_step([PYTHON, "convertToGraph-TrainingSet.py"], env, cwd=PROJECT_ROOT / "convertToGraph")


def train_model(env: Dict[str, str]) -> Path:
    run_step([PYTHON, "VGCN/VGCN.py"], env)
    return PROJECT_ROOT / "VGCN" / "models" / f"{env['VGAT_MODEL_BASENAME']}_best.pth"


def evaluate_fig12(env: Dict[str, str]) -> float:
    run_step([PYTHON, "zNC-Test/Fig12.py"], env)
    results_suffix = env.get("VGAT_RESULTS_SUFFIX", "")
    csv_path = PROJECT_ROOT / "zNC-Test" / "NC-Results" / f"Fig12{results_suffix}" / "fig12_compound_seq_nc.csv"
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        label = (row.get("复合攻击(顺序)") or "").strip()
        if label.endswith("Average"):
            return float(row["VGAT"])
    raise RuntimeError(f"Average row not found in {csv_path}")


def append_summary_row(name: str, env: Dict[str, str], average_nc: float) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["experiment", "pooling", "loss_ablation", "knn_k", "graph_suffix", "fig12_average_nc"])
        writer.writerow(
            [
                name,
                env.get("VGAT_POOLING_MODE", "dual"),
                env.get("VGAT_LOSS_ABLATION", "none"),
                env.get("VGAT_KNN_K", "8"),
                env.get("VGAT_GRAPH_SUFFIX", ""),
                f"{average_nc:.6f}",
            ]
        )


def build_env(overrides: Dict[str, str], epochs: int) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("VGAT_RANDOM_SEED", "42")
    env.setdefault("VGAT_NUM_EPOCHS", str(epochs))
    env.setdefault("VGAT_BATCH_SIZE", "1")
    env.setdefault("VGAT_LOSS_ABLATION", "none")
    env.setdefault("VGAT_POOLING_MODE", "dual")
    env.setdefault("VGAT_KNN_K", "8")
    env.setdefault("VGAT_GRAPH_SUFFIX", "")
    env.setdefault("VGAT_RESUME_CHECKPOINT", "0")
    env.update(overrides)
    env.setdefault("VGAT_CHECKPOINT_NAME", f"{env['VGAT_MODEL_BASENAME']}_checkpoint.pth")
    env.setdefault("VGAT_RESULTS_SUFFIX", f"_{env['VGAT_MODEL_BASENAME']}")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments for item 15.")
    parser.add_argument("--list", action="store_true", help="List available experiment names.")
    parser.add_argument("--experiments", nargs="+", default=["full"], help="Experiment names to run.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--skip-fig12", action="store_true", help="Train only, skip extreme composite evaluation.")
    parser.add_argument("--force-prepare-graphs", action="store_true", help="Always rebuild training graphs.")
    args = parser.parse_args()

    if args.list:
        for name in ABLATIONS:
            print(name)
        return

    for name in args.experiments:
        if name not in ABLATIONS:
            raise SystemExit(f"Unknown experiment: {name}")

        env = build_env(ABLATIONS[name], args.epochs)
        if args.force_prepare_graphs or env.get("VGAT_GRAPH_SUFFIX"):
            prepare_training_graphs(env)

        model_path = train_model(env)
        print(f"[done] trained {name}: {model_path.name}")

        if args.skip_fig12:
            continue

        eval_env = env.copy()
        eval_env["VGAT_MODEL_PATH"] = str(model_path)
        average_nc = evaluate_fig12(eval_env)
        append_summary_row(name, env, average_nc)
        print(f"[done] fig12 average NC for {name}: {average_nc:.6f}")


if __name__ == "__main__":
    main()
