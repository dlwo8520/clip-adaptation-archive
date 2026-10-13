# -*- coding: utf-8 -*-
"""
Plot all individual WC-lambda two-teacher runs as spaghetti curves.

This is for inspecting every curve behind a lambda sweep:
  number of curves per shot = number of lambda values x top-k teachers.

Expected layout:
root/
  twoteacher_texture_diag_wc_0/curves.jsonl
  twoteacher_texture_diag_wc_1e-4/curves.jsonl
  twoteacher_texture_diag_wc_1e-3/curves.jsonl
  ...
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PLOT_EPOCHS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SHOTS = [1, 2, 4, 8, 16]


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_")


def parse_lambda_from_dirname(name: str) -> Optional[float]:
    m = re.search(r"wc_([^/\\]+)$", name)
    if not m:
        return None
    raw = m.group(1).replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def lambda_label(value: float) -> str:
    return "0" if value == 0 else f"{value:g}"


def lambda_sort_key(value: float) -> Tuple[int, float]:
    return (0, -math.inf) if value == 0 else (1, value)


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


def flatten_record(record: Dict, run_dir: Path, lambda_wc: float) -> List[Dict]:
    shot = int(record["shot"])
    epoch = int(record["epoch"])
    target_name = str(record.get("target_name", "texture"))
    teacher = record.get("teacher") or {}
    teacher_shot = int(teacher.get("teacher_shot", shot))
    teacher_rank = int(teacher.get("teacher_rank", -1))
    teacher_seed = int(teacher.get("sample_seed", -1))
    line_id = f"lambda_{lambda_label(lambda_wc)}__teacher_shot_{teacher_shot}_rank_{teacher_rank}_seed_{teacher_seed}"

    base = {
        "run_dir": run_dir.name,
        "lambda_wc": lambda_wc,
        "lambda_label": lambda_label(lambda_wc),
        "line_id": line_id,
        "teacher_shot": teacher_shot,
        "teacher_rank": teacher_rank,
        "teacher_seed": teacher_seed,
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
                }
            )
    return rows


def discover_wc_dirs(root_dir: Path, pattern: str) -> List[Tuple[float, Path]]:
    found: List[Tuple[float, Path]] = []
    for path in sorted(root_dir.glob(pattern)):
        if not path.is_dir():
            continue
        value = parse_lambda_from_dirname(path.name)
        if value is None:
            continue
        if not (path / "curves.jsonl").exists():
            continue
        found.append((value, path))
    found.sort(key=lambda x: lambda_sort_key(x[0]))
    return found


def load_all(root_dir: Path, pattern: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for value, path in discover_wc_dirs(root_dir, pattern):
        for record in read_jsonl(path / "curves.jsonl"):
            rows.extend(flatten_record(record, path, value))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["epoch"].isin(PLOT_EPOCHS)].copy()


def apply_ylim(ylim: Optional[List[float]]) -> None:
    if ylim is not None:
        plt.ylim(float(ylim[0]), float(ylim[1]))


def plot_spaghetti(
    df: pd.DataFrame,
    out_dir: Path,
    split: str,
    dataset: str,
    shot: int,
    ylim: Optional[List[float]],
    line_alpha: float,
    line_width: float,
) -> None:
    sub = df[(df["split"] == split) & (df["dataset"] == dataset) & (df["shot"] == shot)].copy()
    if sub.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11.5, 6.8))

    # One thin line per actual execution: one lambda value + one teacher run.
    # No semantic color/legend is assigned to individual runs, following the
    # plain spaghetti style in [new2]_plot_accuracy_results_gift_auto.py.
    for line_id, g in sub.groupby("line_id"):
        g = g.sort_values("epoch")
        plt.plot(
            g["epoch"],
            g["accuracy"],
            linewidth=line_width,
            alpha=line_alpha,
        )

    stat = (
        sub.groupby("epoch")["accuracy"]
        .agg(mean_accuracy="mean", median_accuracy="median", n_runs=("count"))
        .reset_index()
        .sort_values("epoch")
    )
    plt.plot(
        stat["epoch"],
        stat["mean_accuracy"],
        marker="o",
        linewidth=2.8,
        color="tab:blue",
        label="mean over runs",
    )
    plt.plot(
        stat["epoch"],
        stat["median_accuracy"],
        marker="s",
        linestyle="--",
        linewidth=2.4,
        color="tab:orange",
        label="median over runs",
    )

    baseline = sub["zero_shot_baseline"].dropna()
    if not baseline.empty:
        plt.axhline(float(baseline.iloc[0]), linestyle=":", linewidth=2.3, color="tab:blue", label="zero-shot baseline")

    n_lines = sub["line_id"].nunique()
    plt.xticks(PLOT_EPOCHS)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"{split} / {dataset} / {shot}-shot: all {n_lines} WC lambda x teacher curves")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9, loc="best")
    apply_ylim(ylim)
    plt.tight_layout()

    save_path = out_dir / f"{split}_{safe_name(dataset)}_{shot}shot_wc_lambda_all_spaghetti.png"
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Parent directory containing twoteacher_texture_diag_wc_* folders.")
    parser.add_argument("--wc-pattern", type=str, default="twoteacher_texture_diag_wc_*")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--only-shot", type=int, nargs="*", default=None)
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.30, 0.75])
    parser.add_argument("--line-alpha", type=float, default=0.13)
    parser.add_argument("--line-width", type=float, default=0.8)
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(root_dir, args.wc_pattern)
    if df.empty:
        raise RuntimeError(f"No WC lambda curves found under {root_dir} with pattern {args.wc_pattern}")

    if args.only_shot:
        df = df[df["shot"].isin(args.only_shot)].copy()

    df.to_csv(out_dir / "wc_lambda_all_spaghetti_values.csv", index=False)

    summary = (
        df.groupby(["split", "dataset", "shot", "epoch"], dropna=False)["accuracy"]
        .agg(
            mean_accuracy="mean",
            median_accuracy="median",
            q25_accuracy=lambda x: x.quantile(0.25),
            q75_accuracy=lambda x: x.quantile(0.75),
            min_accuracy="min",
            max_accuracy="max",
            n="count",
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "wc_lambda_all_spaghetti_summary.csv", index=False)

    print("===== loaded =====")
    print(f"rows       : {len(df)}")
    print(f"lines      : {df['line_id'].nunique()}")
    print(f"lambdas    : {sorted(df['lambda_label'].unique(), key=lambda x: lambda_sort_key(float(x)))}")
    print(f"shots      : {sorted(df['shot'].unique())}")
    print(f"epochs     : {sorted(df['epoch'].unique())}")
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))

    for (split, dataset, shot), _ in df.groupby(["split", "dataset", "shot"]):
        plot_spaghetti(
            df,
            out_dir / "spaghetti_all_values",
            split,
            dataset,
            int(shot),
            args.ylim,
            line_alpha=args.line_alpha,
            line_width=args.line_width,
        )

    print("===== done =====")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
