# -*- coding: utf-8 -*-
"""
CoOp / CoCoOp / CLIP-Adapter 세 모델을 같은 그래프에 비교하는 스크립트.

입력 폴더 구조 (각 모델별 100 seed 결과):
  runs/blue_texture_preserve_100seeds/     sample_seed_0~99/curves.jsonl  (CoOp)
  runs/cocoop_texture_preserve_100seeds/   sample_seed_0~99/curves.jsonl  (CoCoOp)
  runs/adapter_texture_preserve_100seeds/  sample_seed_0~99/curves.jsonl  (CLIP-Adapter)

출력:
  runs/model_comparison/
    target_accuracy/          # target dataset accuracy 비교
    preserve_accuracy/        # preserve dataset accuracy 비교
    tradeoff/                 # target gain vs preserve drop scatter
    final_shot_bar/           # 최종 epoch shot별 bar chart
    summary_comparison.csv    # 수치 요약
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 색상 / 스타일 설정
# ---------------------------------------------------------------------------
MODEL_STYLES: Dict[str, Dict] = {
    "CoOp":         {"color": "#2196F3", "marker": "o", "linestyle": "-"},
    "CoCoOp":       {"color": "#FF5722", "marker": "s", "linestyle": "--"},
    "CLIPAdapter":  {"color": "#4CAF50", "marker": "^", "linestyle": "-."},
}

def safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(x))


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_curves(root_dir: Path, model_name: str, seed_start: int, seed_end: int) -> pd.DataFrame:
    """단일 모델의 100 seed curves.jsonl을 읽어 long-format DataFrame으로 반환."""
    rows = []
    missing = []

    for sample_seed in range(seed_start, seed_end + 1):
        curve_path = root_dir / f"sample_seed_{sample_seed}" / "curves.jsonl"
        if not curve_path.exists():
            missing.append(sample_seed)
            continue

        with open(curve_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                seed = int(r.get("sample_seed", sample_seed))
                base = {
                    "model": model_name,
                    "sample_seed": seed,
                    "shot": int(r["shot"]),
                    "epoch": int(r["epoch"]),
                }
                rows.append({
                    **base,
                    "split": "target",
                    "dataset": r["target_name"],
                    "accuracy": float(r["target"]["accuracy"]),
                    "delta": float(r["target"]["delta_from_zero_shot"]),
                })
                for pname, metric in r["preserve"].items():
                    rows.append({
                        **base,
                        "split": "preserve",
                        "dataset": pname,
                        "accuracy": float(metric["accuracy"]),
                        "delta": float(metric["delta_from_zero_shot"]),
                    })

    if missing:
        print(f"[{model_name}] 누락된 seed {len(missing)}개: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if not rows:
        raise FileNotFoundError(f"[{model_name}] {root_dir}에서 curves.jsonl을 하나도 못 읽음")

    df = pd.DataFrame(rows)
    key_cols = ["model", "sample_seed", "shot", "epoch", "split", "dataset"]
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    return df


def load_all_models(model_dirs: Dict[str, Path], seed_start: int, seed_end: int) -> pd.DataFrame:
    """모든 모델의 데이터를 합쳐서 반환."""
    parts = []
    for model_name, root_dir in model_dirs.items():
        if not root_dir.exists():
            print(f"[경고] {model_name} 폴더 없음: {root_dir} — 건너뜀")
            continue
        df = load_curves(root_dir, model_name, seed_start, seed_end)
        print(f"[{model_name}] {df['sample_seed'].nunique()}개 seed 로드 완료")
        parts.append(df)

    if not parts:
        raise RuntimeError("로드된 모델 데이터가 없음")

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 통계 요약
# ---------------------------------------------------------------------------

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "split", "dataset", "shot", "epoch"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            q25_accuracy=("accuracy", lambda x: x.quantile(0.25)),
            q75_accuracy=("accuracy", lambda x: x.quantile(0.75)),
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
            q25_delta=("delta", lambda x: x.quantile(0.25)),
            q75_delta=("delta", lambda x: x.quantile(0.75)),
            n_seeds=("sample_seed", "nunique"),
        )
    )


# ---------------------------------------------------------------------------
# 그래프 1: target / preserve accuracy 곡선 비교 (shot별 subplot)
# ---------------------------------------------------------------------------

def plot_accuracy_curves(
    summary: pd.DataFrame,
    out_dir: Path,
    split: str,
    dataset: str,
    value: str = "accuracy",
    band: str = "iqr",   # "iqr" or "std"
):
    """
    한 그림 안에 shot별 subplot, 각 subplot에 모델별 mean curve + band.
    band: "iqr" → 25~75%, "std" → mean ± std
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    data = summary[(summary["split"] == split) & (summary["dataset"] == dataset)].copy()
    if data.empty:
        return

    shots = sorted(data["shot"].unique())
    models = sorted(data["model"].unique())
    n_shots = len(shots)

    fig, axes = plt.subplots(1, n_shots, figsize=(5 * n_shots, 5), sharey=True)
    if n_shots == 1:
        axes = [axes]

    for ax, shot in zip(axes, shots):
        for model in models:
            style = MODEL_STYLES.get(model, {"color": "gray", "marker": "o", "linestyle": "-"})
            s = data[(data["model"] == model) & (data["shot"] == shot)].sort_values("epoch")
            if s.empty:
                continue

            x = s["epoch"].to_numpy()
            mean = s[f"mean_{value}"].to_numpy()

            if band == "iqr":
                lo = s[f"q25_{value}"].to_numpy()
                hi = s[f"q75_{value}"].to_numpy()
                band_label = "IQR"
            else:
                std = s[f"std_{value}"].fillna(0).to_numpy()
                lo = mean - std
                hi = mean + std
                band_label = "±std"

            ax.plot(x, mean, label=model,
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=2.2, markersize=5)
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.12)

        # zero-shot baseline (epoch 0)
        baseline_rows = data[data["epoch"] == 0]
        if not baseline_rows.empty:
            baseline = float(baseline_rows[f"mean_{value}"].iloc[0])
            ax.axhline(baseline, color="black", linestyle=":", linewidth=1.5, label="zero-shot baseline")

        ax.set_title(f"{shot}-shot", fontsize=12)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(value)
    axes[0].legend(fontsize=9)

    n_seed = int(data["n_seeds"].max())
    fig.suptitle(
        f"{split} / {dataset}: mean {value} ({band_label}) — {n_seed} seeds",
        fontsize=13, y=1.02
    )
    plt.tight_layout()

    save_path = out_dir / f"{safe_name(split)}_{safe_name(dataset)}_{value}_{band}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


# ---------------------------------------------------------------------------
# 그래프 2: 최종 epoch 기준 shot별 bar chart (모델 비교)
# ---------------------------------------------------------------------------

def plot_final_bar(
    summary: pd.DataFrame,
    out_dir: Path,
    split: str,
    dataset: str,
    value: str = "accuracy",
):
    """
    최종 epoch에서 shot별 모델 비교 bar chart.
    x축: shot 수, bar group: 모델
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    data = summary[(summary["split"] == split) & (summary["dataset"] == dataset)].copy()
    if data.empty:
        return

    max_epoch = int(data["epoch"].max())
    final = data[data["epoch"] == max_epoch].copy()
    if final.empty:
        return

    shots = sorted(final["shot"].unique())
    models = sorted(final["model"].unique())
    n_models = len(models)
    x = np.arange(len(shots))
    width = 0.7 / n_models

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(models):
        style = MODEL_STYLES.get(model, {"color": "gray"})
        mdf = final[final["model"] == model].set_index("shot")
        means = [float(mdf.loc[s, f"mean_{value}"]) if s in mdf.index else np.nan for s in shots]
        stds  = [float(mdf.loc[s, f"std_{value}"]) if s in mdf.index else 0.0 for s in shots]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.92,
               color=style["color"], label=model, alpha=0.85,
               yerr=stds, capsize=3, error_kw={"elinewidth": 1.2})

    # zero-shot baseline
    epoch0 = data[data["epoch"] == 0]
    if not epoch0.empty:
        baseline = float(epoch0[f"mean_{value}"].iloc[0])
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.8, label="zero-shot baseline")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}-shot" for s in shots])
    ax.set_ylabel(value)
    ax.set_title(f"{split} / {dataset}: final epoch {value} by shot (mean ± std)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = out_dir / f"{safe_name(split)}_{safe_name(dataset)}_{value}_final_bar.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"saved: {save_path}")


# ---------------------------------------------------------------------------
# 그래프 3: target gain vs preserve drop trade-off scatter
# ---------------------------------------------------------------------------

def plot_tradeoff_scatter(
    summary: pd.DataFrame,
    out_dir: Path,
    target_dataset: str,
    preserve_datasets: List[str],
):
    """
    x축: target delta (adaptation gain)
    y축: preserve delta (zero-shot preservation, 음수면 하락)
    각 점: (shot, epoch=final)
    모델별로 색깔과 마커가 다름.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    max_epoch = int(summary["epoch"].max())
    final = summary[summary["epoch"] == max_epoch].copy()
    shots = sorted(final["shot"].unique())
    models = sorted(final["model"].unique())

    for preserve_dataset in preserve_datasets:
        fig, ax = plt.subplots(figsize=(8, 6))

        for model in models:
            style = MODEL_STYLES.get(model, {"color": "gray", "marker": "o"})
            xs, ys, labels = [], [], []

            for shot in shots:
                t_row = final[
                    (final["model"] == model) &
                    (final["split"] == "target") &
                    (final["dataset"] == target_dataset) &
                    (final["shot"] == shot)
                ]
                p_row = final[
                    (final["model"] == model) &
                    (final["split"] == "preserve") &
                    (final["dataset"] == preserve_dataset) &
                    (final["shot"] == shot)
                ]
                if t_row.empty or p_row.empty:
                    continue
                xs.append(float(t_row["mean_delta"].iloc[0]))
                ys.append(float(p_row["mean_delta"].iloc[0]))
                labels.append(f"{shot}s")

            if not xs:
                continue

            sc = ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
                            s=120, label=model, zorder=5)
            for xi, yi, lbl in zip(xs, ys, labels):
                ax.annotate(lbl, (xi, yi), textcoords="offset points",
                            xytext=(5, 4), fontsize=8, color=style["color"])
            # shot 수 증가에 따라 선으로 연결
            if len(xs) > 1:
                ax.plot(xs, ys, color=style["color"], alpha=0.4, linewidth=1.2, zorder=4)

        ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
        ax.set_xlabel(f"target ({target_dataset}) delta from zero-shot ↑")
        ax.set_ylabel(f"preserve ({preserve_dataset}) delta from zero-shot ↑")
        ax.set_title(f"Trade-off: target gain vs {preserve_dataset} preservation\n(final epoch, mean over seeds)")
        ax.legend()
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        save_path = out_dir / f"tradeoff_{safe_name(target_dataset)}_vs_{safe_name(preserve_dataset)}.png"
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"saved: {save_path}")


# ---------------------------------------------------------------------------
# 그래프 4: 전체 shot 한눈에 보기 (target + 모든 preserve를 row로)
# ---------------------------------------------------------------------------

def plot_overview_grid(
    summary: pd.DataFrame,
    out_dir: Path,
    target_dataset: str,
    preserve_datasets: List[str],
    value: str = "accuracy",
):
    """
    rows: target + preserve 데이터셋
    cols: shot 수
    각 셀: 모델별 mean curve
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = [("target", target_dataset)] + [("preserve", d) for d in preserve_datasets]
    shots = sorted(summary["shot"].unique())
    n_rows = len(all_datasets)
    n_cols = len(shots)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows), squeeze=False)

    models = sorted(summary["model"].unique())

    for row_idx, (split, dataset) in enumerate(all_datasets):
        for col_idx, shot in enumerate(shots):
            ax = axes[row_idx][col_idx]

            for model in models:
                style = MODEL_STYLES.get(model, {"color": "gray", "marker": "o", "linestyle": "-"})
                s = summary[
                    (summary["model"] == model) &
                    (summary["split"] == split) &
                    (summary["dataset"] == dataset) &
                    (summary["shot"] == shot)
                ].sort_values("epoch")
                if s.empty:
                    continue
                x = s["epoch"].to_numpy()
                mean = s[f"mean_{value}"].to_numpy()
                std = s[f"std_{value}"].fillna(0).to_numpy()

                ax.plot(x, mean, color=style["color"], marker=style["marker"],
                        linestyle=style["linestyle"], linewidth=1.8, markersize=4, label=model)
                ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.10)

            # zero-shot baseline
            epoch0 = summary[
                (summary["split"] == split) &
                (summary["dataset"] == dataset) &
                (summary["epoch"] == 0)
            ]
            if not epoch0.empty:
                baseline = float(epoch0[f"mean_{value}"].iloc[0])
                ax.axhline(baseline, color="black", linestyle=":", linewidth=1.2)

            if row_idx == 0:
                ax.set_title(f"{shot}-shot", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{dataset}\n{value}", fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("epoch", fontsize=9)

            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

    # 범례는 첫 번째 셀에
    axes[0][0].legend(fontsize=8)

    n_seed = int(summary["n_seeds"].max())
    fig.suptitle(f"Model comparison: {value} (mean ± std, {n_seed} seeds)", fontsize=13)
    plt.tight_layout()

    save_path = out_dir / f"overview_grid_{value}.png"
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


# ---------------------------------------------------------------------------
# 수치 요약 출력
# ---------------------------------------------------------------------------

def print_summary_table(summary: pd.DataFrame, out_dir: Path):
    max_epoch = int(summary["epoch"].max())
    final = summary[summary["epoch"] == max_epoch].copy()
    final = final[[
        "model", "split", "dataset", "shot",
        "mean_accuracy", "std_accuracy",
        "mean_delta", "std_delta",
        "n_seeds"
    ]].sort_values(["split", "dataset", "model", "shot"])

    print("\n===== 최종 epoch 수치 요약 =====")
    print(final.to_string(index=False))

    csv_path = out_dir / "summary_comparison.csv"
    final.to_csv(csv_path, index=False)
    print(f"\nsaved: {csv_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CoOp / CoCoOp / CLIP-Adapter 결과를 같은 그래프에 비교"
    )
    parser.add_argument(
        "--coop-dir",
        type=str,
        default="runs/blue_texture_preserve_100seeds",
        help="CoOp 100 seed 결과 루트 폴더",
    )
    parser.add_argument(
        "--cocoop-dir",
        type=str,
        default=None,
        help="CoCoOp 100 seed 결과 루트 폴더 (없으면 비교에서 제외)",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="runs/adapter_texture_preserve_100seeds",
        help="CLIP-Adapter 100 seed 결과 루트 폴더",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end",   type=int, default=99)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/model_comparison",
        help="그래프 저장 폴더",
    )
    parser.add_argument(
        "--band",
        type=str,
        default="std",
        choices=["iqr", "std"],
        help="불확실도 표시 방식: iqr (25~75%) 또는 std (mean ± std)",
    )
    args = parser.parse_args()

    model_dirs: Dict[str, Path] = {}
    if args.coop_dir:
        model_dirs["CoOp"] = Path(args.coop_dir)
    if args.cocoop_dir:
        model_dirs["CoCoOp"] = Path(args.cocoop_dir)
    if args.adapter_dir:
        model_dirs["CLIPAdapter"] = Path(args.adapter_dir)

    if not model_dirs:
        raise ValueError("비교할 모델 폴더를 하나 이상 지정해야 함")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== 데이터 로드 =====")
    df = load_all_models(model_dirs, args.seed_start, args.seed_end)
    summary = make_summary(df)

    # dataset 목록 자동 감지
    target_datasets = df[df["split"] == "target"]["dataset"].unique().tolist()
    preserve_datasets = df[df["split"] == "preserve"]["dataset"].unique().tolist()
    target_dataset = target_datasets[0] if target_datasets else "texture"

    print(f"\ntarget:   {target_datasets}")
    print(f"preserve: {preserve_datasets}")
    print(f"models:   {sorted(df['model'].unique())}")
    print(f"shots:    {sorted(df['shot'].unique())}")
    print(f"epochs:   {sorted(df['epoch'].unique())}")

    # -----------------------------------------------------------------------
    # 1. target accuracy 비교
    # -----------------------------------------------------------------------
    print("\n===== target accuracy 비교 그래프 =====")
    plot_accuracy_curves(
        summary=summary,
        out_dir=out_dir / "target_accuracy",
        split="target",
        dataset=target_dataset,
        value="accuracy",
        band=args.band,
    )
    plot_accuracy_curves(
        summary=summary,
        out_dir=out_dir / "target_accuracy",
        split="target",
        dataset=target_dataset,
        value="delta",
        band=args.band,
    )

    # -----------------------------------------------------------------------
    # 2. preserve accuracy 비교
    # -----------------------------------------------------------------------
    print("\n===== preserve accuracy 비교 그래프 =====")
    for pd_name in preserve_datasets:
        plot_accuracy_curves(
            summary=summary,
            out_dir=out_dir / "preserve_accuracy",
            split="preserve",
            dataset=pd_name,
            value="accuracy",
            band=args.band,
        )
        plot_accuracy_curves(
            summary=summary,
            out_dir=out_dir / "preserve_accuracy",
            split="preserve",
            dataset=pd_name,
            value="delta",
            band=args.band,
        )

    # -----------------------------------------------------------------------
    # 3. bar chart (최종 epoch, shot별 모델 비교)
    # -----------------------------------------------------------------------
    print("\n===== bar chart =====")
    bar_dir = out_dir / "final_shot_bar"
    plot_final_bar(summary, bar_dir, split="target",   dataset=target_dataset, value="accuracy")
    plot_final_bar(summary, bar_dir, split="target",   dataset=target_dataset, value="delta")
    for pd_name in preserve_datasets:
        plot_final_bar(summary, bar_dir, split="preserve", dataset=pd_name, value="accuracy")
        plot_final_bar(summary, bar_dir, split="preserve", dataset=pd_name, value="delta")

    # -----------------------------------------------------------------------
    # 4. trade-off scatter
    # -----------------------------------------------------------------------
    print("\n===== trade-off scatter =====")
    plot_tradeoff_scatter(
        summary=summary,
        out_dir=out_dir / "tradeoff",
        target_dataset=target_dataset,
        preserve_datasets=preserve_datasets,
    )

    # -----------------------------------------------------------------------
    # 5. 전체 overview grid
    # -----------------------------------------------------------------------
    print("\n===== overview grid =====")
    plot_overview_grid(
        summary=summary,
        out_dir=out_dir,
        target_dataset=target_dataset,
        preserve_datasets=preserve_datasets,
        value="accuracy",
    )
    plot_overview_grid(
        summary=summary,
        out_dir=out_dir,
        target_dataset=target_dataset,
        preserve_datasets=preserve_datasets,
        value="delta",
    )

    # -----------------------------------------------------------------------
    # 6. 수치 요약
    # -----------------------------------------------------------------------
    print_summary_table(summary, out_dir)

    print(f"\n===== 완료 =====")
    print(f"결과 폴더: {out_dir}")


if __name__ == "__main__":
    main()
