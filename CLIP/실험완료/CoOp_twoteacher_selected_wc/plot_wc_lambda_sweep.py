# -*- coding: utf-8 -*-
"""
Plot WC lambda sweep for two-teacher selected CoOp.

Expected directory layout:

root/
  blue_texture_preserve_100seeds/
    sample_seed_0/curves.jsonl
    sample_seed_1/curves.jsonl
    ...
  twoteacher_texture_diag_wc_0/
    curves.jsonl
  twoteacher_texture_diag_wc_1e-4/
    curves.jsonl
  twoteacher_texture_diag_wc_1e-3/
    curves.jsonl

Outputs:
  - final accuracy vs lambda plots
  - epoch curves by lambda
  - target vs preserve tradeoff plots
  - flat and summary CSV files
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
SHOT_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    4: "tab:green",
    8: "tab:red",
    16: "tab:purple",
}


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_")


def parse_lambda_from_dirname(name: str) -> Optional[float]:
    m = re.search(r"wc_([^/\\]+)$", name)
    if not m:
        return None
    raw = m.group(1)
    try:
        return float(raw)
    except ValueError:
        pass
    raw = raw.replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def lambda_label(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:g}"


def lambda_sort_key(value: float) -> Tuple[int, float]:
    if value == 0:
        return (0, -math.inf)
    return (1, value)


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


def flatten_record(record: Dict, source_kind: str, source_name: str, replicate: str, lambda_wc: Optional[float]) -> List[Dict]:
    shot = int(record["shot"])
    epoch = int(record["epoch"])
    target_name = str(record.get("target_name", "texture"))
    teacher = record.get("teacher") or {}

    if source_kind == "twoteacher" and teacher:
        replicate = (
            f"teacher_shot_{teacher.get('teacher_shot', shot)}"
            f"_rank_{teacher.get('teacher_rank', 'x')}"
            f"_seed_{teacher.get('sample_seed', 'x')}"
        )

    base = {
        "source_kind": source_kind,
        "source_name": source_name,
        "lambda_wc": lambda_wc,
        "lambda_label": "random" if lambda_wc is None else lambda_label(float(lambda_wc)),
        "replicate": replicate,
        "sample_seed": record.get("sample_seed"),
        "teacher_shot": teacher.get("teacher_shot"),
        "teacher_rank": teacher.get("teacher_rank"),
        "teacher_seed": teacher.get("sample_seed"),
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


def load_random(root: Path, seed_start: int, seed_end: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for sample_seed in range(seed_start, seed_end + 1):
        curve_path = root / f"sample_seed_{sample_seed}" / "curves.jsonl"
        if not curve_path.exists():
            continue
        for record in read_jsonl(curve_path):
            rows.extend(flatten_record(record, "random", root.name, str(sample_seed), None))
    return pd.DataFrame(rows)


def load_twoteacher_run(run_dir: Path, lambda_wc: float) -> pd.DataFrame:
    curve_path = run_dir / "curves.jsonl"
    if not curve_path.exists():
        raise FileNotFoundError(f"curves.jsonl not found: {curve_path}")
    rows: List[Dict] = []
    for record in read_jsonl(curve_path):
        rows.extend(flatten_record(record, "twoteacher", run_dir.name, "twoteacher", lambda_wc))
    return pd.DataFrame(rows)


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


def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_from_zero_shot"] = df["accuracy"] - df["zero_shot_baseline"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["source_kind", "source_name", "lambda_wc", "lambda_label", "shot", "epoch", "split", "dataset"]
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
    d = (
        df.groupby(group_cols, dropna=False)["delta_from_zero_shot"]
        .agg(
            mean_delta="mean",
            median_delta="median",
            q25_delta=lambda x: x.quantile(0.25),
            q75_delta=lambda x: x.quantile(0.75),
        )
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


def get_lambda_order(summary: pd.DataFrame) -> List[float]:
    vals = sorted(summary[summary["source_kind"] == "twoteacher"]["lambda_wc"].dropna().unique(), key=lambda x: lambda_sort_key(float(x)))
    return [float(v) for v in vals]


def plot_final_vs_lambda(summary: pd.DataFrame, out_dir: Path, final_epoch: int, ylim: Optional[List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if final_epoch not in set(summary["epoch"].unique()):
        final_epoch = int(summary["epoch"].max())

    lambdas = get_lambda_order(summary)
    x = list(range(len(lambdas)))
    xlabels = [lambda_label(v) for v in lambdas]
    final = summary[summary["epoch"] == final_epoch].copy()

    for (split, dataset), sub_all in final.groupby(["split", "dataset"]):
        plt.figure(figsize=(11.2, 6.4))
        for shot in SHOTS:
            tw = sub_all[(sub_all["source_kind"] == "twoteacher") & (sub_all["shot"] == shot)].copy()
            tw = tw.set_index("lambda_wc")
            if tw.empty:
                continue
            mean = [float(tw.loc[v, "mean_accuracy"]) if v in tw.index else float("nan") for v in lambdas]
            low = [float(tw.loc[v, "mean_accuracy"] - tw.loc[v, "q25_accuracy"]) if v in tw.index else 0.0 for v in lambdas]
            high = [float(tw.loc[v, "q75_accuracy"] - tw.loc[v, "mean_accuracy"]) if v in tw.index else 0.0 for v in lambdas]
            color = SHOT_COLORS.get(shot)
            plt.errorbar(
                x,
                mean,
                yerr=[low, high],
                marker="o",
                linewidth=2.6,
                capsize=4,
                color=color,
                label=f"{shot}-shot two-teacher",
            )

            rnd = sub_all[(sub_all["source_kind"] == "random") & (sub_all["shot"] == shot)]
            if not rnd.empty:
                plt.axhline(
                    float(rnd["mean_accuracy"].iloc[0]),
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.45,
                    label=f"{shot}-shot random mean",
                )

        baseline = sub_all["baseline"].dropna()
        if not baseline.empty:
            plt.axhline(float(baseline.iloc[0]), color="black", linestyle=":", linewidth=2.0, alpha=0.65, label="zero-shot baseline")
        plt.xticks(x, xlabels, rotation=35, ha="right")
        plt.xlabel("lambda_wc")
        plt.ylabel("accuracy")
        plt.title(f"WC lambda sweep at epoch {final_epoch}\n{split} / {dataset}")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(fontsize=8, ncol=2)
        apply_ylim(ylim)
        plt.tight_layout()
        save_path = out_dir / f"{split}_{safe_name(dataset)}_epoch{final_epoch}_lambda_sweep.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def plot_epoch_curves(summary: pd.DataFrame, out_dir: Path, ylim: Optional[List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lambdas = get_lambda_order(summary)
    cmap = plt.get_cmap("viridis")
    lambda_colors = {v: cmap(i / max(1, len(lambdas) - 1)) for i, v in enumerate(lambdas)}

    for (split, dataset, shot), sub_all in summary.groupby(["split", "dataset", "shot"]):
        plt.figure(figsize=(11.2, 6.4))
        rnd = sub_all[sub_all["source_kind"] == "random"].sort_values("epoch")
        if not rnd.empty:
            plt.plot(
                rnd["epoch"],
                rnd["mean_accuracy"],
                color="black",
                linestyle="--",
                linewidth=2.4,
                alpha=0.55,
                label="random mean",
            )
            plt.fill_between(
                rnd["epoch"].to_numpy(dtype=float),
                rnd["q25_accuracy"].to_numpy(dtype=float),
                rnd["q75_accuracy"].to_numpy(dtype=float),
                color="black",
                alpha=0.08,
                linewidth=0,
            )

        for value in lambdas:
            g = sub_all[(sub_all["source_kind"] == "twoteacher") & (sub_all["lambda_wc"] == value)].sort_values("epoch")
            if g.empty:
                continue
            color = lambda_colors[value]
            plt.plot(
                g["epoch"],
                g["mean_accuracy"],
                marker="o",
                linewidth=2.1,
                color=color,
                label=f"lambda={lambda_label(value)}",
            )
            plt.fill_between(
                g["epoch"].to_numpy(dtype=float),
                g["q25_accuracy"].to_numpy(dtype=float),
                g["q75_accuracy"].to_numpy(dtype=float),
                color=color,
                alpha=0.10,
                linewidth=0,
            )

        baseline = sub_all["baseline"].dropna()
        if not baseline.empty:
            plt.axhline(float(baseline.iloc[0]), color="gray", linestyle=":", linewidth=2.0, alpha=0.65, label="zero-shot baseline")
        plt.xticks(PLOT_EPOCHS)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"WC lambda epoch curves\n{split} / {dataset}, {int(shot)}-shot")
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
    final = summary[summary["epoch"] == final_epoch].copy()
    target = final[(final["split"] == "target")].copy()
    preserve = final[(final["split"] == "preserve")].copy()
    preserve_avg = (
        preserve.groupby(["source_kind", "source_name", "lambda_wc", "lambda_label", "shot"], dropna=False)["mean_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"mean_accuracy": "preserve_avg_accuracy"})
    )
    target = target.rename(columns={"mean_accuracy": "target_accuracy"})
    combo = target.merge(preserve_avg, on=["source_kind", "source_name", "lambda_wc", "lambda_label", "shot"], how="inner")
    combo.to_csv(out_dir / f"target_preserve_tradeoff_epoch{final_epoch}.csv", index=False)

    lambdas = get_lambda_order(summary)
    cmap = plt.get_cmap("viridis")
    lambda_colors = {v: cmap(i / max(1, len(lambdas) - 1)) for i, v in enumerate(lambdas)}

    for shot in SHOTS:
        sub = combo[combo["shot"] == shot].copy()
        if sub.empty:
            continue
        plt.figure(figsize=(7.2, 6.4))
        rnd = sub[sub["source_kind"] == "random"]
        if not rnd.empty:
            plt.scatter(
                rnd["target_accuracy"],
                rnd["preserve_avg_accuracy"],
                marker="X",
                s=120,
                color="black",
                alpha=0.75,
                label="random mean",
            )
        tw = sub[sub["source_kind"] == "twoteacher"]
        for _, row in tw.sort_values("lambda_wc").iterrows():
            value = float(row["lambda_wc"])
            plt.scatter(
                row["target_accuracy"],
                row["preserve_avg_accuracy"],
                s=85,
                color=lambda_colors.get(value),
                label=f"lambda={lambda_label(value)}",
            )
            plt.annotate(lambda_label(value), (row["target_accuracy"], row["preserve_avg_accuracy"]), fontsize=8, xytext=(4, 3), textcoords="offset points")

        plt.xlabel("target accuracy")
        plt.ylabel("avg preserve accuracy (ImageNet/CIFAR100)")
        plt.title(f"target-preserve tradeoff at epoch {final_epoch}\n{shot}-shot")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8, loc="best")
        plt.tight_layout()
        save_path = out_dir / f"{shot}shot_target_preserve_tradeoff_epoch{final_epoch}.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Directory containing baseline and WC run folders.")
    parser.add_argument("--baseline-name", type=str, default="blue_texture_preserve_100seeds")
    parser.add_argument("--wc-pattern", type=str, default="twoteacher_texture_diag_wc_*")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument("--final-epoch", type=int, default=50)
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.30, 0.75])
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_root = root_dir / args.baseline_name
    if not baseline_root.exists():
        raise FileNotFoundError(f"baseline directory not found: {baseline_root}")

    wc_dirs = discover_wc_dirs(root_dir, args.wc_pattern)
    if not wc_dirs:
        raise RuntimeError(f"No WC run directories found under {root_dir} with pattern {args.wc_pattern}")

    print("===== discovered WC runs =====")
    for value, path in wc_dirs:
        print(f"lambda={lambda_label(value):>8}  path={path}")

    frames = [load_random(baseline_root, args.seed_start, args.seed_end)]
    for value, path in wc_dirs:
        frames.append(load_twoteacher_run(path, value))

    df = pd.concat(frames, ignore_index=True)
    df = df[df["epoch"].isin(PLOT_EPOCHS)].copy()
    df = add_delta(df)
    summary = summarize(df)

    df.to_csv(out_dir / "wc_lambda_sweep_flat.csv", index=False)
    summary.to_csv(out_dir / "wc_lambda_sweep_summary.csv", index=False)

    print("===== loaded =====")
    print(f"rows       : {len(df)}")
    print(f"lambdas    : {[lambda_label(v) for v, _ in wc_dirs]}")
    print(f"shots      : {sorted(df['shot'].unique())}")
    print(f"epochs     : {sorted(df['epoch'].unique())}")
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))

    plot_final_vs_lambda(summary, out_dir / "final_vs_lambda", args.final_epoch, args.ylim)
    plot_epoch_curves(summary, out_dir / "epoch_curves", args.ylim)
    plot_tradeoff(summary, out_dir / "tradeoff", args.final_epoch)

    print("===== done =====")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
