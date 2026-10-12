# -*- coding: utf-8 -*-
"""
Plot JS trade-off results from train_js_tradeoff_coop.py outputs.

Supports:
  - a diagnostic root containing mode folders:
      root/random/curves.jsonl
      root/anchor_entropy/curves.jsonl
      root/js_q1/curves.jsonl ...
  - optional random 100-seed root:
      random_root/sample_seed_0/curves.jsonl ...
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


PLOT_EPOCHS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SHOTS = [1, 2, 4, 8, 16]
MODE_ORDER = ["random", "anchor_entropy", "js_q1", "js_q2", "js_q3", "js_q4"]
MODE_LABELS = {
    "random": "random",
    "anchor_entropy": "anchor entropy",
    "js_q1": "Q1 lowest JS",
    "js_q2": "Q2 low-mid JS",
    "js_q3": "Q3 high-mid JS",
    "js_q4": "Q4 highest JS",
}
MODE_COLORS = {
    "random": "0.35",
    "anchor_entropy": "tab:purple",
    "js_q1": "tab:blue",
    "js_q2": "tab:green",
    "js_q3": "tab:orange",
    "js_q4": "tab:red",
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


def mode_sort_key(mode: str) -> int:
    return MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER)


def flatten_record(record: Dict, mode: str, replicate: str) -> List[Dict]:
    shot = int(record["shot"])
    epoch = int(record["epoch"])
    target_name = str(record.get("target_name", "texture"))
    selection = record.get("selection") or {}
    mode = str(selection.get("selection_mode") or mode)

    base = {
        "mode": mode,
        "mode_label": MODE_LABELS.get(mode, mode),
        "replicate": replicate,
        "shot": shot,
        "epoch": epoch,
        "sample_seed": record.get("sample_seed"),
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
                "delta": metric_accuracy(target) - metric_baseline(target),
            }
        )

    preserve = record.get("preserve", {})
    if isinstance(preserve, dict):
        for dataset, metric in preserve.items():
            if metric_accuracy(metric) is None:
                continue
            rows.append(
                {
                    **base,
                    "split": "preserve",
                    "dataset": str(dataset),
                    "accuracy": metric_accuracy(metric),
                    "zero_shot_baseline": metric_baseline(metric),
                    "delta": metric_accuracy(metric) - metric_baseline(metric),
                }
            )
    return rows


def load_mode_root(root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    if not root.exists():
        return pd.DataFrame()
    for mode_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: mode_sort_key(p.name)):
        curve = mode_dir / "curves.jsonl"
        if not curve.exists():
            continue
        for record in read_jsonl(curve):
            rows.extend(flatten_record(record, mode_dir.name, mode_dir.name))
    return pd.DataFrame(rows)


def load_random_root(root: Path, seed_start: int, seed_end: int) -> pd.DataFrame:
    rows: List[Dict] = []
    if root is None or not root.exists():
        return pd.DataFrame()
    for sample_seed in range(seed_start, seed_end + 1):
        curve = root / f"sample_seed_{sample_seed}" / "curves.jsonl"
        if not curve.exists():
            continue
        for record in read_jsonl(curve):
            rows.extend(flatten_record(record, "random", f"sample_seed_{sample_seed}"))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["mode", "mode_label", "shot", "epoch", "split", "dataset"]
    s = (
        df.groupby(group_cols, dropna=False)["accuracy"]
        .agg(
            mean_accuracy="mean",
            median_accuracy="median",
            q25_accuracy=lambda x: x.quantile(0.25),
            q75_accuracy=lambda x: x.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    d = (
        df.groupby(group_cols, dropna=False)["delta"]
        .agg(mean_delta="mean", median_delta="median")
        .reset_index()
    )
    b = (
        df.groupby(group_cols, dropna=False)["zero_shot_baseline"]
        .first()
        .reset_index()
        .rename(columns={"zero_shot_baseline": "baseline"})
    )
    return s.merge(d, on=group_cols).merge(b, on=group_cols)


def apply_ylim(ylim: Optional[List[float]]) -> None:
    if ylim is not None:
        plt.ylim(float(ylim[0]), float(ylim[1]))


def plot_epoch_curves(summary: pd.DataFrame, out_dir: Path, ylim: Optional[List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (split, dataset, shot), sub in summary.groupby(["split", "dataset", "shot"]):
        plt.figure(figsize=(11.4, 6.6))
        for mode in sorted(sub["mode"].unique(), key=mode_sort_key):
            g = sub[sub["mode"] == mode].sort_values("epoch")
            if g.empty:
                continue
            color = MODE_COLORS.get(mode)
            plt.plot(
                g["epoch"],
                g["mean_accuracy"],
                marker="o",
                linewidth=2.4,
                color=color,
                label=f"{MODE_LABELS.get(mode, mode)}" + (" mean" if int(g["n"].max()) > 1 else ""),
            )
            if int(g["n"].max()) > 1:
                plt.fill_between(
                    g["epoch"].to_numpy(dtype=float),
                    g["q25_accuracy"].to_numpy(dtype=float),
                    g["q75_accuracy"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
        baseline = sub["baseline"].dropna()
        if not baseline.empty:
            plt.axhline(float(baseline.iloc[0]), color="black", linestyle=":", linewidth=2.0, alpha=0.65, label="zero-shot baseline")
        plt.xticks(PLOT_EPOCHS)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"JS trade-off curves\n{split} / {dataset}, {int(shot)}-shot")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8, ncol=2)
        apply_ylim(ylim)
        plt.tight_layout()
        save_path = out_dir / f"{split}_{safe_name(dataset)}_{int(shot)}shot_epoch_curves.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def plot_tradeoff(summary: pd.DataFrame, out_dir: Path, final_epoch: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if final_epoch not in set(summary["epoch"].unique()):
        final_epoch = int(summary["epoch"].max())
    final = summary[summary["epoch"] == final_epoch].copy()
    target = final[final["split"] == "target"].rename(columns={"mean_accuracy": "target_accuracy", "mean_delta": "target_delta"})
    preserve = final[final["split"] == "preserve"].copy()
    preserve_avg = (
        preserve.groupby(["mode", "mode_label", "shot"], dropna=False)
        .agg(preserve_avg_accuracy=("mean_accuracy", "mean"), preserve_avg_delta=("mean_delta", "mean"))
        .reset_index()
    )
    combo = target.merge(preserve_avg, on=["mode", "mode_label", "shot"], how="inner")
    combo.to_csv(out_dir / f"target_preserve_tradeoff_epoch{final_epoch}.csv", index=False)

    for shot in SHOTS:
        sub = combo[combo["shot"] == shot].copy()
        if sub.empty:
            continue
        plt.figure(figsize=(7.5, 6.5))
        for _, row in sub.sort_values("mode", key=lambda s: s.map(mode_sort_key)).iterrows():
            mode = str(row["mode"])
            plt.scatter(
                row["target_accuracy"],
                row["preserve_avg_accuracy"],
                s=95,
                color=MODE_COLORS.get(mode),
                label=MODE_LABELS.get(mode, mode),
            )
            plt.annotate(MODE_LABELS.get(mode, mode), (row["target_accuracy"], row["preserve_avg_accuracy"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
        plt.xlabel("target accuracy")
        plt.ylabel("avg preserve accuracy")
        plt.title(f"JS target-preserve trade-off at epoch {final_epoch}\n{shot}-shot")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        save_path = out_dir / f"{shot}shot_target_preserve_tradeoff_epoch{final_epoch}.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Directory containing random/anchor_entropy/js_q* run folders.")
    parser.add_argument("--random-root", type=str, default=None, help="Optional random 100-seed root.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument("--final-epoch", type=int, default=50)
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.30, 0.75])
    args = parser.parse_args()

    mode_df = load_mode_root(Path(args.root_dir).expanduser().resolve())
    if args.random_root and not mode_df.empty:
        mode_df = mode_df[mode_df["mode"] != "random"].copy()
    frames = [mode_df]
    if args.random_root:
        frames.append(load_random_root(Path(args.random_root).expanduser().resolve(), args.seed_start, args.seed_end))
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError("No curves loaded.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["epoch"].isin(PLOT_EPOCHS)].copy()
    summary = summarize(df)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "js_tradeoff_flat.csv", index=False)
    summary.to_csv(out_dir / "js_tradeoff_summary.csv", index=False)

    print("===== loaded =====")
    print(f"modes  : {sorted(df['mode'].unique(), key=mode_sort_key)}")
    print(f"shots  : {sorted(df['shot'].unique())}")
    print(f"epochs : {sorted(df['epoch'].unique())}")
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))

    plot_epoch_curves(summary, out_dir / "epoch_curves", args.ylim)
    plot_tradeoff(summary, out_dir / "tradeoff", args.final_epoch)
    print("===== done =====")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
