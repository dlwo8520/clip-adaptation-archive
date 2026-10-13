# -*- coding: utf-8 -*-
"""
Plot LoRA full-teacher two-teacher selection results.

Supports:
  - random 100-seed CoOp baseline
  - previous CoOp-teacher two-teacher selection
  - new LoRA full-data teacher two-teacher selection

Example:
  python plot_lora_full_teacher_results.py \
    --random-root runs/blue_texture_preserve_100seeds \
    --lora-root runs/lora_full_teacher_twoteacher_texture_v3 \
    --coop-teacher-root runs/twoteacher_texture_diag \
    --out-dir runs/lora_full_teacher_plots
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PLOT_EPOCHS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SHOTS = [1, 2, 4, 8, 16]
SHOT_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    4: "tab:green",
    8: "tab:red",
    16: "tab:purple",
}
METHOD_LABELS = {
    "random": "random few-shot CoOp",
    "coop_teacher": "CoOp-teacher selected",
    "lora_teacher": "LoRA full-teacher selected",
}
METHOD_STYLES = {
    "random": "--",
    "coop_teacher": "-.",
    "lora_teacher": "-",
}
METHOD_WIDTHS = {
    "random": 2.0,
    "coop_teacher": 2.2,
    "lora_teacher": 2.8,
}


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_")


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def metric_accuracy(metric: Dict) -> Optional[float]:
    if not isinstance(metric, dict) or metric.get("accuracy") is None:
        return None
    return float(metric["accuracy"])


def metric_baseline(metric: Dict) -> Optional[float]:
    if not isinstance(metric, dict) or metric.get("zero_shot_baseline") is None:
        return None
    return float(metric["zero_shot_baseline"])


def flatten_record(record: Dict, method: str, source_name: str, replicate: str) -> List[Dict]:
    shot = int(record["shot"])
    epoch = int(record["epoch"])
    target_name = str(record.get("target_name", "texture"))
    teacher = record.get("teacher") or {}

    if teacher:
        replicate = (
            f"teacher_shot_{teacher.get('teacher_shot', shot)}"
            f"_rank_{teacher.get('teacher_rank', 'x')}"
            f"_seed_{teacher.get('sample_seed', 'x')}"
        )

    base = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "source_name": source_name,
        "replicate": replicate,
        "sample_seed": record.get("sample_seed"),
        "shot": shot,
        "epoch": epoch,
    }

    rows: List[Dict] = []
    target = record.get("target", {})
    if metric_accuracy(target) is not None:
        rows.append(
            {
                **base,
                "split": "target",
                "dataset": target_name,
                "accuracy": metric_accuracy(target),
                "zero_shot_baseline": metric_baseline(target),
            }
        )

    preserve = record.get("preserve", {})
    if isinstance(preserve, dict):
        for dataset_name, metric in preserve.items():
            if metric_accuracy(metric) is None:
                continue
            rows.append(
                {
                    **base,
                    "split": "preserve",
                    "dataset": str(dataset_name),
                    "accuracy": metric_accuracy(metric),
                    "zero_shot_baseline": metric_baseline(metric),
                }
            )
    return rows


def load_random(root: Path, seed_start: int, seed_end: int) -> pd.DataFrame:
    rows: List[Dict] = []
    if root is None or not root.exists():
        return pd.DataFrame()
    for sample_seed in range(seed_start, seed_end + 1):
        path = root / f"sample_seed_{sample_seed}" / "curves.jsonl"
        if not path.exists():
            continue
        for record in read_jsonl(path):
            rows.extend(flatten_record(record, "random", root.name, str(sample_seed)))
    return pd.DataFrame(rows)


def load_single_run(root: Path, method: str) -> pd.DataFrame:
    rows: List[Dict] = []
    if root is None or not root.exists():
        return pd.DataFrame()
    curve_path = root / "curves.jsonl"
    if curve_path.exists():
        paths = [curve_path]
    else:
        paths = sorted(root.rglob("history.json"))
    for path in paths:
        if path.name == "curves.jsonl":
            for record in read_jsonl(path):
                rows.extend(flatten_record(record, method, root.name, root.name))
        else:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                replicate = str(path.parent.parent.name if path.parent.name.startswith("student_shot") else path.parent.name)
                for record in obj:
                    if isinstance(record, dict):
                        rows.extend(flatten_record(record, method, root.name, replicate))
    return pd.DataFrame(rows)


def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_from_zero_shot"] = df["accuracy"] - df["zero_shot_baseline"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "method_label", "shot", "epoch", "split", "dataset"]
    s = (
        df.groupby(group_cols, dropna=False)["accuracy"]
        .agg(
            mean_accuracy="mean",
            median_accuracy="median",
            q25_accuracy=lambda x: x.quantile(0.25),
            q75_accuracy=lambda x: x.quantile(0.75),
            std_accuracy="std",
            n="count",
        )
        .reset_index()
    )
    b = (
        df.groupby(group_cols, dropna=False)["zero_shot_baseline"]
        .first()
        .reset_index()
        .rename(columns={"zero_shot_baseline": "baseline"})
    )
    return s.merge(b, on=group_cols)


def apply_ylim(ylim: Optional[List[float]]) -> None:
    if ylim is not None:
        plt.ylim(float(ylim[0]), float(ylim[1]))


def plot_mean_curves(summary: pd.DataFrame, out_dir: Path, plot_epochs: List[int], ylim: Optional[List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in ["random", "coop_teacher", "lora_teacher"] if m in set(summary["method"])]

    for (split, dataset), sub in summary.groupby(["split", "dataset"]):
        plt.figure(figsize=(12, 6.8))
        for shot in SHOTS:
            for method in methods:
                g = sub[(sub["shot"] == shot) & (sub["method"] == method)].sort_values("epoch")
                if g.empty:
                    continue
                color = SHOT_COLORS.get(shot)
                plt.plot(
                    g["epoch"],
                    g["mean_accuracy"],
                    marker="o",
                    linewidth=METHOD_WIDTHS[method],
                    linestyle=METHOD_STYLES[method],
                    color=color,
                    alpha=0.82 if method != "random" else 0.58,
                    label=f"{shot}-shot {METHOD_LABELS[method]}",
                )
                if g["n"].max() > 1:
                    plt.fill_between(
                        g["epoch"].to_numpy(dtype=float),
                        g["q25_accuracy"].to_numpy(dtype=float),
                        g["q75_accuracy"].to_numpy(dtype=float),
                        color=color,
                        alpha=0.08 if method == "random" else 0.12,
                        linewidth=0,
                    )

        baseline = sub["baseline"].dropna()
        if not baseline.empty:
            plt.axhline(float(baseline.iloc[0]), color="black", linestyle=":", linewidth=2.0, alpha=0.65, label="zero-shot baseline")
        plt.xticks(plot_epochs)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"{split} / {dataset}: random vs CoOp-teacher vs LoRA full-teacher")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8, ncol=2)
        apply_ylim(ylim)
        plt.tight_layout()
        save_path = out_dir / f"{split}_{safe_name(dataset)}_mean_curves_compare.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def plot_final_bars(summary: pd.DataFrame, out_dir: Path, final_epoch: int, ylim: Optional[List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = summary[summary["epoch"] == final_epoch].copy()
    if final.empty:
        final_epoch = int(summary["epoch"].max())
        final = summary[summary["epoch"] == final_epoch].copy()
    methods = [m for m in ["random", "coop_teacher", "lora_teacher"] if m in set(final["method"])]
    width = 0.78 / max(1, len(methods))

    for (split, dataset), sub in final.groupby(["split", "dataset"]):
        plt.figure(figsize=(11, 6.3))
        x = list(range(len(SHOTS)))
        for mi, method in enumerate(methods):
            offset = (mi - (len(methods) - 1) / 2) * width
            g = sub[sub["method"] == method].set_index("shot")
            means = [float(g.loc[s, "mean_accuracy"]) if s in g.index else float("nan") for s in SHOTS]
            lows = [float(g.loc[s, "mean_accuracy"] - g.loc[s, "q25_accuracy"]) if s in g.index else 0.0 for s in SHOTS]
            highs = [float(g.loc[s, "q75_accuracy"] - g.loc[s, "mean_accuracy"]) if s in g.index else 0.0 for s in SHOTS]
            xpos = [i + offset for i in x]
            plt.bar(xpos, means, width=width, label=METHOD_LABELS[method], alpha=0.82)
            if g["n"].max() > 1:
                plt.errorbar(xpos, means, yerr=[lows, highs], fmt="none", ecolor="black", capsize=3, linewidth=1.0)

        baseline = sub["baseline"].dropna()
        if not baseline.empty:
            plt.axhline(float(baseline.iloc[0]), color="black", linestyle=":", linewidth=2.0, alpha=0.65, label="zero-shot baseline")
        plt.xticks(x, [str(s) for s in SHOTS])
        plt.xlabel("shot")
        plt.ylabel("accuracy")
        plt.title(f"{split} / {dataset}: final epoch {final_epoch}")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(fontsize=9)
        apply_ylim(ylim)
        plt.tight_layout()
        save_path = out_dir / f"{split}_{safe_name(dataset)}_final_epoch_{final_epoch}_bars.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def plot_lora_spaghetti(df: pd.DataFrame, out_dir: Path, only_shot: Optional[int], plot_epochs: List[int], ylim: Optional[List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lora = df[df["method"] == "lora_teacher"].copy()
    if only_shot is not None:
        lora = lora[lora["shot"] == only_shot]
    if lora.empty:
        return

    for (split, dataset, shot), sub in lora.groupby(["split", "dataset", "shot"]):
        plt.figure(figsize=(11, 6.8))
        for replicate, g in sub.groupby("replicate"):
            g = g.sort_values("epoch")
            plt.plot(g["epoch"], g["accuracy"], linewidth=1.3, alpha=0.35)
        stat = (
            sub.groupby("epoch", as_index=False)
            .agg(mean_accuracy=("accuracy", "mean"), median_accuracy=("accuracy", "median"), n_runs=("replicate", "nunique"))
            .sort_values("epoch")
        )
        plt.plot(stat["epoch"], stat["mean_accuracy"], marker="o", linewidth=2.8, label="mean over runs")
        plt.plot(stat["epoch"], stat["median_accuracy"], marker="s", linestyle="--", linewidth=2.4, label="median over runs")
        baseline = sub["zero_shot_baseline"].dropna()
        if not baseline.empty:
            plt.axhline(float(baseline.iloc[0]), linestyle=":", linewidth=2.0, label="zero-shot baseline")
        plt.xticks(plot_epochs)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"LoRA full-teacher selected\n{split} / {dataset} / {int(shot)}-shot")
        plt.grid(True, alpha=0.25)
        plt.legend()
        apply_ylim(ylim)
        plt.tight_layout()
        save_path = out_dir / f"{split}_{safe_name(dataset)}_{int(shot)}shot_lora_spaghetti.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-root", type=str, required=True)
    parser.add_argument("--random-root", type=str, default=None)
    parser.add_argument("--coop-teacher-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument("--final-epoch", type=int, default=50)
    parser.add_argument("--spaghetti-shot", type=int, default=16)
    parser.add_argument("--include-epoch1", action="store_true")
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.30, 0.75])
    args = parser.parse_args()
    plot_epochs = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] if args.include_epoch1 else DEFAULT_PLOT_EPOCHS

    frames = []
    if args.random_root:
        frames.append(load_random(Path(args.random_root).expanduser().resolve(), args.seed_start, args.seed_end))
    if args.coop_teacher_root:
        frames.append(load_single_run(Path(args.coop_teacher_root).expanduser().resolve(), "coop_teacher"))
    frames.append(load_single_run(Path(args.lora_root).expanduser().resolve(), "lora_teacher"))

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError("No result rows loaded.")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["epoch"].isin(plot_epochs)].copy()
    df = add_delta(df)
    summary = summarize(df)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "lora_full_teacher_flat.csv", index=False)
    summary.to_csv(out_dir / "lora_full_teacher_summary.csv", index=False)

    print("===== loaded =====")
    print(f"methods : {sorted(df['method'].unique())}")
    print(f"shots   : {sorted(df['shot'].unique())}")
    print(f"epochs  : {sorted(df['epoch'].unique())}")
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))

    plot_mean_curves(summary, out_dir / "mean_curves", plot_epochs, args.ylim)
    plot_final_bars(summary, out_dir / "final_bars", args.final_epoch, args.ylim)
    plot_lora_spaghetti(df, out_dir / "lora_spaghetti", args.spaghetti_shot, plot_epochs, args.ylim)

    print("===== done =====")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
