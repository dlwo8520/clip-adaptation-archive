# -*- coding: utf-8 -*-
("""
Compare LoRA full-teacher selected CoOp against random CoOp and one WC run.

LoRA is treated as a single run: no mean/median label is used for it.
Random and WC runs can contain multiple replicates, so they are shown as
mean curves with optional IQR bands.

Example:
  python plot_lora_random_wc_compare.py \
    --lora-root ${RESEARCH_WORKSPACE}/projects/CoOp_lora_full_teacher/runs/lora_full_teacher_twoteacher_texture_v3 \
    --random-root ${RESEARCH_WORKSPACE}/projects/CoOp_lora_full_teacher/runs/blue_texture_preserve_100seeds \
    --wc-root ${RESEARCH_WORKSPACE}/projects/CoOp_lora_full_teacher/runs/twoteacher_texture_diag_wc_3e-2 \
    --out-dir ${RESEARCH_WORKSPACE}/projects/CoOp_lora_full_teacher/runs/compare_lora_random_wc_3e-2_plots
""").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[3])))

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
METHOD_ORDER = ["random", "wc", "lora"]
METHOD_STYLES = {
    "random": "--",
    "wc": "-.",
    "lora": "-",
}
METHOD_MARKERS = {
    "random": "o",
    "wc": "s",
    "lora": "D",
}
METHOD_COLORS = {
    "random": "0.25",
    "wc": "tab:blue",
    "lora": "tab:red",
}


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_")


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_accuracy(metric: Dict) -> Optional[float]:
    if not isinstance(metric, dict) or metric.get("accuracy") is None:
        return None
    return float(metric["accuracy"])


def metric_baseline(metric: Dict) -> Optional[float]:
    if not isinstance(metric, dict) or metric.get("zero_shot_baseline") is None:
        return None
    return float(metric["zero_shot_baseline"])


def infer_wc_label(path: Path) -> str:
    match = re.search(r"wc_([^/\\]+)$", path.name)
    if not match:
        return "CoOp-teacher + WC"
    return f"CoOp-teacher + WC {match.group(1)}"


def flatten_record(
    record: Dict,
    method: str,
    method_label: str,
    source_name: str,
    replicate: str,
    fallback_shot: Optional[int] = None,
) -> List[Dict]:
    if "shot" not in record:
        if fallback_shot is None:
            return []
        record = {**record, "shot": fallback_shot}

    shot = int(record["shot"])
    epoch = int(record["epoch"])
    target_name = str(record.get("target_name", "texture"))
    teacher = record.get("teacher") or {}

    if method == "wc" and teacher:
        replicate = (
            f"teacher_shot_{teacher.get('teacher_shot', shot)}"
            f"_rank_{teacher.get('teacher_rank', 'x')}"
            f"_seed_{teacher.get('sample_seed', 'x')}"
        )

    base = {
        "method": method,
        "method_label": method_label,
        "source_name": source_name,
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


def load_random(root: Path, label: str, seed_start: int, seed_end: int) -> pd.DataFrame:
    rows: List[Dict] = []
    if not root.exists():
        raise FileNotFoundError(f"random root not found: {root}")
    for sample_seed in range(seed_start, seed_end + 1):
        path = root / f"sample_seed_{sample_seed}" / "curves.jsonl"
        if not path.exists():
            continue
        for record in read_jsonl(path):
            rows.extend(flatten_record(record, "random", label, root.name, f"seed_{sample_seed}"))
    return pd.DataFrame(rows)


def load_wc(root: Path, label: str) -> pd.DataFrame:
    rows: List[Dict] = []
    if not root.exists():
        raise FileNotFoundError(f"WC root not found: {root}")
    curve_path = root / "curves.jsonl"
    if not curve_path.exists():
        raise FileNotFoundError(f"WC curves.jsonl not found: {curve_path}")
    for record in read_jsonl(curve_path):
        rows.extend(flatten_record(record, "wc", label, root.name, root.name))
    return pd.DataFrame(rows)


def load_lora(root: Path, label: str) -> pd.DataFrame:
    rows: List[Dict] = []
    if not root.exists():
        raise FileNotFoundError(f"LoRA root not found: {root}")

    curve_path = root / "curves.jsonl"
    if curve_path.exists():
        for record in read_jsonl(curve_path):
            rows.extend(flatten_record(record, "lora", label, root.name, "lora_single"))
        return pd.DataFrame(rows)

    history_paths = sorted(root.glob("shot_*/history.json"), key=natural_key)
    if not history_paths:
        raise FileNotFoundError(f"No LoRA curves.jsonl or shot_*/history.json found under: {root}")

    for path in history_paths:
        shot_match = re.search(r"shot_(\d+)$", path.parent.name)
        fallback_shot = int(shot_match.group(1)) if shot_match else None
        obj = read_json(path)
        if not isinstance(obj, list):
            continue
        for record in obj:
            if isinstance(record, dict):
                rows.extend(flatten_record(record, "lora", label, root.name, "lora_single", fallback_shot=fallback_shot))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "method_label", "shot", "epoch", "split", "dataset"]
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
    b = (
        df.groupby(group_cols, dropna=False)["zero_shot_baseline"]
        .first()
        .reset_index()
        .rename(columns={"zero_shot_baseline": "baseline"})
    )
    return s.merge(b, on=group_cols)


def method_order_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def apply_ylim(ylim: Optional[List[float]]) -> None:
    if ylim is not None:
        plt.ylim(float(ylim[0]), float(ylim[1]))


def baseline_value(sub: pd.DataFrame) -> Optional[float]:
    baseline = sub["baseline"].dropna() if "baseline" in sub.columns else pd.Series(dtype=float)
    if baseline.empty:
        return None
    return float(baseline.iloc[0])


def line_label(method: str, label: str, n: int) -> str:
    if method == "lora":
        return label
    if method == "random":
        return f"{label} mean over seeds"
    if method == "wc":
        return f"{label} mean over teachers"
    return f"{label} mean" if n > 1 else label


def plot_by_shot(summary: pd.DataFrame, out_dir: Path, plot_epochs: List[int], ylim: Optional[List[float]], show_iqr: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for (split, dataset, shot), sub in summary.groupby(["split", "dataset", "shot"]):
        plt.figure(figsize=(11.2, 6.6))
        for method in sorted(sub["method"].unique(), key=method_order_key):
            g = sub[sub["method"] == method].sort_values("epoch")
            if g.empty:
                continue
            label = str(g["method_label"].iloc[0])
            n = int(g["n"].max())
            color = METHOD_COLORS.get(method)
            plt.plot(
                g["epoch"],
                g["mean_accuracy"],
                marker=METHOD_MARKERS.get(method, "o"),
                linestyle=METHOD_STYLES.get(method, "-"),
                linewidth=2.8 if method == "lora" else 2.3,
                color=color,
                label=line_label(method, label, n),
            )
            if show_iqr and method != "lora" and n > 1:
                plt.fill_between(
                    g["epoch"].to_numpy(dtype=float),
                    g["q25_accuracy"].to_numpy(dtype=float),
                    g["q75_accuracy"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )

        base = baseline_value(sub)
        if base is not None:
            plt.axhline(base, color="black", linestyle=":", linewidth=2.0, alpha=0.65, label="zero-shot baseline")
        plt.xticks(plot_epochs)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"{split} / {dataset} / {int(shot)}-shot: random vs WC vs LoRA")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9, loc="best")
        apply_ylim(ylim)
        plt.tight_layout()
        save_path = out_dir / f"{split}_{safe_name(dataset)}_{int(shot)}shot_compare.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def plot_all_shots(summary: pd.DataFrame, out_dir: Path, plot_epochs: List[int], ylim: Optional[List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for (split, dataset), sub in summary.groupby(["split", "dataset"]):
        plt.figure(figsize=(12.4, 7.0))
        for shot in SHOTS:
            shot_df = sub[sub["shot"] == shot]
            if shot_df.empty:
                continue
            for method in sorted(shot_df["method"].unique(), key=method_order_key):
                g = shot_df[shot_df["method"] == method].sort_values("epoch")
                if g.empty:
                    continue
                label = str(g["method_label"].iloc[0])
                n = int(g["n"].max())
                plt.plot(
                    g["epoch"],
                    g["mean_accuracy"],
                    marker=METHOD_MARKERS.get(method, "o"),
                    linestyle=METHOD_STYLES.get(method, "-"),
                    linewidth=2.5 if method == "lora" else 1.9,
                    color=SHOT_COLORS.get(shot),
                    alpha=0.95 if method == "lora" else 0.64,
                    label=f"{int(shot)}-shot {line_label(method, label, n)}",
                )

        base = baseline_value(sub)
        if base is not None:
            plt.axhline(base, color="black", linestyle=":", linewidth=2.0, alpha=0.65, label="zero-shot baseline")
        plt.xticks(plot_epochs)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"{split} / {dataset}: random vs WC vs LoRA")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=7.5, ncol=2)
        apply_ylim(ylim)
        plt.tight_layout()
        save_path = out_dir / f"{split}_{safe_name(dataset)}_all_shots_compare.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def plot_final_bars(summary: pd.DataFrame, out_dir: Path, final_epoch: int, ylim: Optional[List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = summary[summary["epoch"] == final_epoch].copy()
    if final.empty:
        final_epoch = int(summary["epoch"].max())
        final = summary[summary["epoch"] == final_epoch].copy()

    methods = sorted(final["method"].unique(), key=method_order_key)
    width = 0.78 / max(1, len(methods))

    for (split, dataset), sub in final.groupby(["split", "dataset"]):
        plt.figure(figsize=(11.2, 6.4))
        x = list(range(len(SHOTS)))
        for mi, method in enumerate(methods):
            g = sub[sub["method"] == method].set_index("shot")
            if g.empty:
                continue
            offset = (mi - (len(methods) - 1) / 2) * width
            means = [float(g.loc[s, "mean_accuracy"]) if s in g.index else float("nan") for s in SHOTS]
            xpos = [i + offset for i in x]
            label = str(g["method_label"].iloc[0])
            plt.bar(xpos, means, width=width, label=label, alpha=0.85)
            if method != "lora" and int(g["n"].max()) > 1:
                low = [float(g.loc[s, "mean_accuracy"] - g.loc[s, "q25_accuracy"]) if s in g.index else 0.0 for s in SHOTS]
                high = [float(g.loc[s, "q75_accuracy"] - g.loc[s, "mean_accuracy"]) if s in g.index else 0.0 for s in SHOTS]
                plt.errorbar(xpos, means, yerr=[low, high], fmt="none", ecolor="black", capsize=3, linewidth=1.0)

        base = baseline_value(sub)
        if base is not None:
            plt.axhline(base, color="black", linestyle=":", linewidth=2.0, alpha=0.65, label="zero-shot baseline")
        plt.xticks(x, [str(s) for s in SHOTS])
        plt.xlabel("shot")
        plt.ylabel("accuracy")
        plt.title(f"{split} / {dataset}: final epoch {final_epoch}")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(fontsize=9)
        apply_ylim(ylim)
        plt.tight_layout()
        save_path = out_dir / f"{split}_{safe_name(dataset)}_final_epoch_{final_epoch}_compare.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-root", type=str, required=True)
    parser.add_argument("--random-root", type=str, required=True)
    parser.add_argument("--wc-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--random-label", type=str, default="random few-shot CoOp")
    parser.add_argument("--wc-label", type=str, default=None)
    parser.add_argument("--lora-label", type=str, default="LoRA full-teacher selected")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument("--final-epoch", type=int, default=50)
    parser.add_argument("--only-shot", type=int, nargs="*", default=None)
    parser.add_argument("--include-epoch1", action="store_true")
    parser.add_argument("--no-iqr", action="store_true")
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.30, 0.75])
    args = parser.parse_args()

    lora_root = Path(args.lora_root).expanduser().resolve()
    random_root = Path(args.random_root).expanduser().resolve()
    wc_root = Path(args.wc_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wc_label = args.wc_label if args.wc_label is not None else infer_wc_label(wc_root)
    plot_epochs = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] if args.include_epoch1 else DEFAULT_PLOT_EPOCHS

    frames = [
        load_random(random_root, args.random_label, args.seed_start, args.seed_end),
        load_wc(wc_root, wc_label),
        load_lora(lora_root, args.lora_label),
    ]
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        raise RuntimeError("No result rows loaded.")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["epoch"].isin(plot_epochs)].copy()
    if args.only_shot:
        df = df[df["shot"].isin(args.only_shot)].copy()
    if df.empty:
        raise RuntimeError("No rows remain after epoch/shot filtering.")

    summary = summarize(df)
    df.to_csv(out_dir / "lora_random_wc_compare_flat.csv", index=False)
    summary.to_csv(out_dir / "lora_random_wc_compare_summary.csv", index=False)

    print("===== loaded =====")
    print(f"methods : {sorted(df['method'].unique(), key=method_order_key)}")
    print(f"shots   : {sorted(df['shot'].unique())}")
    print(f"epochs  : {sorted(df['epoch'].unique())}")
    print(df[["method", "method_label", "replicate"]].drop_duplicates().groupby(["method", "method_label"]).size().rename("n_replicates").reset_index().to_string(index=False))
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))

    plot_by_shot(summary, out_dir / "curves_by_shot", plot_epochs, args.ylim, show_iqr=not args.no_iqr)
    plot_all_shots(summary, out_dir / "curves_all_shots", plot_epochs, args.ylim)
    plot_final_bars(summary, out_dir / "final_bars", args.final_epoch, args.ylim)

    print("===== done =====")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
