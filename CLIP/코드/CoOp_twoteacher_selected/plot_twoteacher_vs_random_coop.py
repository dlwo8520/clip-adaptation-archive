# -*- coding: utf-8 -*-
("""
Plot random CoOp vs two-teacher selected CoOp.

Inputs can be either extracted directories or zip files.

Examples:
  python plot_twoteacher_vs_random_coop.py \
    --random-root ${RESEARCH_WORKSPACE}/projects/CoOp_twoteacher_selected/runs/blue_texture_preserve_100seeds \
    --twoteacher-root ${RESEARCH_WORKSPACE}/projects/CoOp_twoteacher_selected/runs/twoteacher_texture_diag \
    --out-dir runs/twoteacher_vs_random_plots

  python plot_twoteacher_vs_random_coop.py \
    --random-root CoOp.zip \
    --twoteacher-root twoteacher_texture_diag.zip \
    --out-dir plots
""").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[3])))

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SHOTS = [1, 2, 4, 8, 16]
COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    4: "tab:green",
    8: "tab:red",
    16: "tab:purple",
}
METHOD_LABELS = {
    "random": "random few-shot",
    "twoteacher": "two-teacher selected",
}
METHOD_STYLE = {
    "random": "--",
    "twoteacher": "-",
}
METHOD_ALPHA = {
    "random": 0.10,
    "twoteacher": 0.18,
}


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_")


def read_jsonl_from_zip(zip_path: Path, pattern: re.Pattern) -> Iterable[tuple[str, Dict]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not pattern.search(name):
                continue
            with zf.open(name, "r") as f:
                for raw in f:
                    line = raw.decode("utf-8").strip()
                    if line:
                        yield name, json.loads(line)


def read_jsonl_from_dir(root: Path, pattern: str) -> Iterable[tuple[str, Dict]]:
    for path in sorted(root.rglob(pattern)):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield str(path), json.loads(line)


def get_acc(metric_obj: Dict) -> Optional[float]:
    if not isinstance(metric_obj, dict):
        return None
    value = metric_obj.get("accuracy")
    return float(value) if value is not None else None


def get_base(metric_obj: Dict) -> Optional[float]:
    if not isinstance(metric_obj, dict):
        return None
    value = metric_obj.get("zero_shot_baseline")
    return float(value) if value is not None else None


def flatten_record(record: Dict, method: str, source: str, fallback_replicate: str) -> List[Dict]:
    shot = int(record["shot"])
    epoch = int(record["epoch"])
    target_name = str(record.get("target_name", "texture"))
    teacher = record.get("teacher") or {}

    if method == "twoteacher" and teacher:
        replicate = f"teacher_shot_{teacher.get('teacher_shot', shot)}_rank_{teacher.get('teacher_rank', 'x')}_seed_{teacher.get('sample_seed', 'x')}"
        teacher_shot = int(teacher.get("teacher_shot", shot))
        teacher_rank = int(teacher.get("teacher_rank", -1))
        teacher_seed = int(teacher.get("sample_seed", -1))
    else:
        replicate = str(record.get("sample_seed", fallback_replicate))
        teacher_shot = None
        teacher_rank = None
        teacher_seed = None

    rows = []
    target = record.get("target", {})
    if get_acc(target) is not None:
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "source": source,
                "replicate": replicate,
                "sample_seed": record.get("sample_seed"),
                "teacher_shot": teacher_shot,
                "teacher_rank": teacher_rank,
                "teacher_seed": teacher_seed,
                "shot": shot,
                "epoch": epoch,
                "split": "target",
                "dataset": target_name,
                "accuracy": get_acc(target),
                "zero_shot_baseline": get_base(target),
            }
        )

    preserve = record.get("preserve", {})
    if isinstance(preserve, dict):
        for dataset_name, metric in preserve.items():
            if get_acc(metric) is None:
                continue
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "source": source,
                    "replicate": replicate,
                    "sample_seed": record.get("sample_seed"),
                    "teacher_shot": teacher_shot,
                    "teacher_rank": teacher_rank,
                    "teacher_seed": teacher_seed,
                    "shot": shot,
                    "epoch": epoch,
                    "split": "preserve",
                    "dataset": str(dataset_name),
                    "accuracy": get_acc(metric),
                    "zero_shot_baseline": get_base(metric),
                }
            )
    return rows


def load_random(root: Path, seed_start: int, seed_end: int) -> pd.DataFrame:
    rows: List[Dict] = []
    if root.is_file() and root.suffix.lower() == ".zip":
        pattern = re.compile(r"blue_texture_preserve_100seeds/sample_seed_(\d+)/curves\.jsonl$")
        for source, record in read_jsonl_from_zip(root, pattern):
            m = re.search(r"sample_seed_(\d+)", source)
            seed = int(m.group(1)) if m else -1
            if seed_start <= seed <= seed_end:
                rows.extend(flatten_record(record, "random", source, str(seed)))
    else:
        for source, record in read_jsonl_from_dir(root, "curves.jsonl"):
            m = re.search(r"sample_seed_(\d+)", source)
            if not m:
                continue
            seed = int(m.group(1))
            if seed_start <= seed <= seed_end:
                rows.extend(flatten_record(record, "random", source, str(seed)))
    return pd.DataFrame(rows)


def load_twoteacher(root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    if root.is_file() and root.suffix.lower() == ".zip":
        pattern = re.compile(r"twoteacher_texture_diag/curves\.jsonl$")
        for source, record in read_jsonl_from_zip(root, pattern):
            rows.extend(flatten_record(record, "twoteacher", source, "twoteacher"))
    else:
        curve_path = root / "curves.jsonl"
        if curve_path.exists():
            sources = read_jsonl_from_dir(root, "curves.jsonl")
        else:
            sources = read_jsonl_from_dir(root, "history.json")
        for source, record in sources:
            rows.extend(flatten_record(record, "twoteacher", source, "twoteacher"))
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


def plot_mean_iqr(summary: pd.DataFrame, out_dir: Path, value: str, ylim: Optional[List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if value == "accuracy":
        mean_col, q25_col, q75_col, y_label = "mean_accuracy", "q25_accuracy", "q75_accuracy", "accuracy"
    else:
        mean_col, q25_col, q75_col, y_label = "mean_delta", "q25_delta", "q75_delta", "delta from zero-shot"

    for (split, dataset), sub in summary.groupby(["split", "dataset"]):
        plt.figure(figsize=(11.5, 6.5))
        for shot in SHOTS:
            for method in ["random", "twoteacher"]:
                g = sub[(sub["shot"] == shot) & (sub["method"] == method)].sort_values("epoch")
                if g.empty:
                    continue
                x = g["epoch"].to_numpy(dtype=float)
                mean = g[mean_col].to_numpy(dtype=float)
                q25 = g[q25_col].to_numpy(dtype=float)
                q75 = g[q75_col].to_numpy(dtype=float)
                label = f"{shot}-shot {METHOD_LABELS[method]} mean"
                plt.plot(
                    x,
                    mean,
                    marker="o",
                    linewidth=2.5 if method == "twoteacher" else 2.0,
                    linestyle=METHOD_STYLE[method],
                    color=COLORS.get(shot),
                    label=label,
                )
                plt.fill_between(x, q25, q75, color=COLORS.get(shot), alpha=METHOD_ALPHA[method], linewidth=0)

        if value == "accuracy":
            baseline = sub["baseline"].dropna()
            if not baseline.empty:
                plt.axhline(
                    float(baseline.iloc[0]),
                    color="black",
                    linestyle=":",
                    linewidth=2.0,
                    alpha=0.65,
                    label="zero-shot baseline",
                )
        else:
            plt.axhline(0.0, color="black", linestyle=":", linewidth=2.0, alpha=0.65, label="zero-shot baseline")

        plt.xlabel("epoch")
        plt.ylabel(y_label)
        plt.title(f"{split} / {dataset}: {METHOD_LABELS['random']} vs {METHOD_LABELS['twoteacher']}")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8, ncol=2)
        apply_ylim(ylim)
        plt.tight_layout()
        save_path = out_dir / f"{split}_{safe_name(dataset)}_{value}_mean_iqr_compare.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def plot_final_bars(summary: pd.DataFrame, out_dir: Path, final_epoch: int, ylim: Optional[List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = summary[summary["epoch"] == final_epoch].copy()
    if final.empty:
        final_epoch = int(summary["epoch"].max())
        final = summary[summary["epoch"] == final_epoch].copy()

    for (split, dataset), sub in final.groupby(["split", "dataset"]):
        plt.figure(figsize=(10.5, 6.2))
        x = range(len(SHOTS))
        width = 0.36
        for offset, method in [(-width / 2, "random"), (width / 2, "twoteacher")]:
            g = sub[sub["method"] == method].set_index("shot")
            means = [float(g.loc[s, "mean_accuracy"]) if s in g.index else float("nan") for s in SHOTS]
            lows = [float(g.loc[s, "mean_accuracy"] - g.loc[s, "q25_accuracy"]) if s in g.index else 0.0 for s in SHOTS]
            highs = [float(g.loc[s, "q75_accuracy"] - g.loc[s, "mean_accuracy"]) if s in g.index else 0.0 for s in SHOTS]
            xpos = [i + offset for i in x]
            plt.bar(xpos, means, width=width, label=METHOD_LABELS[method], alpha=0.82)
            plt.errorbar(xpos, means, yerr=[lows, highs], fmt="none", ecolor="black", capsize=4, linewidth=1.2)

        baseline = sub["baseline"].dropna()
        if not baseline.empty:
            plt.axhline(float(baseline.iloc[0]), color="black", linestyle=":", linewidth=2.0, alpha=0.65, label="zero-shot baseline")
        plt.xticks(list(x), [str(s) for s in SHOTS])
        plt.xlabel("shot")
        plt.ylabel("accuracy")
        plt.title(f"{split} / {dataset}: final epoch {final_epoch} comparison")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(fontsize=9)
        apply_ylim(ylim)
        plt.tight_layout()
        save_path = out_dir / f"{split}_{safe_name(dataset)}_final_epoch_{final_epoch}_bar_compare.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-root", type=str, required=True, help="random 100-seed dir or CoOp.zip")
    parser.add_argument("--twoteacher-root", type=str, required=True, help="two-teacher result dir or twoteacher_texture_diag.zip")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument("--final-epoch", type=int, default=50)
    parser.add_argument("--ylim-accuracy", type=float, nargs=2, default=[0.30, 0.75])
    parser.add_argument("--ylim-delta", type=float, nargs=2, default=[-0.35, 0.25])
    args = parser.parse_args()

    random_root = Path(args.random_root).expanduser().resolve()
    twoteacher_root = Path(args.twoteacher_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    random_df = load_random(random_root, args.seed_start, args.seed_end)
    twoteacher_df = load_twoteacher(twoteacher_root)
    if random_df.empty:
        raise RuntimeError(f"No random CoOp rows loaded from {random_root}")
    if twoteacher_df.empty:
        raise RuntimeError(f"No two-teacher rows loaded from {twoteacher_root}")

    df = pd.concat([random_df, twoteacher_df], ignore_index=True)
    df = df[df["epoch"].isin([0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])].copy()
    df = add_delta(df)
    summary = summarize(df)

    df.to_csv(out_dir / "flat_accuracy_compare.csv", index=False)
    summary.to_csv(out_dir / "summary_accuracy_compare.csv", index=False)

    print("===== loaded =====")
    print(f"random rows     : {len(random_df)}")
    print(f"twoteacher rows : {len(twoteacher_df)}")
    print(f"methods         : {sorted(df['method'].unique())}")
    print(f"shots           : {sorted(df['shot'].unique())}")
    print(f"epochs          : {sorted(df['epoch'].unique())}")
    print(f"datasets:\n{df[['split', 'dataset']].drop_duplicates().to_string(index=False)}")

    plot_mean_iqr(summary, out_dir / "mean_iqr_accuracy", "accuracy", args.ylim_accuracy)
    plot_mean_iqr(summary, out_dir / "mean_iqr_delta", "delta", args.ylim_delta)
    plot_final_bars(summary, out_dir / "final_epoch_bars", args.final_epoch, args.ylim_accuracy)

    print("===== done =====")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
