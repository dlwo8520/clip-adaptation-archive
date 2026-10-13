# -*- coding: utf-8 -*-

("""
CoOp 100-seed 결과에서 epoch 0, 5, 10, 15, ... 만 사용해 그래프를 그림.

입력:
  ${RESEARCH_WORKSPACE}/projects/CoOp/runs/blue_texture_preserve_100seeds/
    sample_seed_0/curves.jsonl
    sample_seed_1/curves.jsonl
    ...
    sample_seed_99/curves.jsonl

출력:
  runs/blue_texture_preserve_100seeds/filtered_epoch_plots/
""").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[3])))

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def keep_plot_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """
    epoch 0과 5 단위 epoch만 남김.
    즉 0, 5, 10, 15, ... 만 사용.
    epoch 1은 제거됨.
    """
    return df[(df["epoch"] == 0) | (df["epoch"] % 5 == 0)].copy()


def load_curves(root_dir: Path, seed_start: int, seed_end: int) -> pd.DataFrame:
    rows = []

    for sample_seed in range(seed_start, seed_end + 1):
        curve_path = root_dir / f"sample_seed_{sample_seed}" / "curves.jsonl"

        if not curve_path.exists():
            print(f"[경고] 없음: {curve_path}")
            continue

        with open(curve_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)

                seed = int(r.get("sample_seed", sample_seed))
                shot = int(r["shot"])
                epoch = int(r["epoch"])

                rows.append(
                    {
                        "sample_seed": seed,
                        "shot": shot,
                        "epoch": epoch,
                        "split": "target",
                        "dataset": r["target_name"],
                        "accuracy": float(r["target"]["accuracy"]),
                        "delta": float(r["target"]["delta_from_zero_shot"]),
                    }
                )

                for preserve_name, metric in r["preserve"].items():
                    rows.append(
                        {
                            "sample_seed": seed,
                            "shot": shot,
                            "epoch": epoch,
                            "split": "preserve",
                            "dataset": preserve_name,
                            "accuracy": float(metric["accuracy"]),
                            "delta": float(metric["delta_from_zero_shot"]),
                        }
                    )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("curves.jsonl 결과를 읽지 못함.")

    key_cols = ["sample_seed", "shot", "epoch", "split", "dataset"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"[경고] 중복 row 제거됨: {before - after}개")

    df = keep_plot_epochs(df)

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["split", "dataset", "shot", "epoch"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            median_accuracy=("accuracy", "median"),
            var_accuracy=("accuracy", "var"),
            std_accuracy=("accuracy", "std"),
            q25_accuracy=("accuracy", lambda x: x.quantile(0.25)),
            q75_accuracy=("accuracy", lambda x: x.quantile(0.75)),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            var_delta=("delta", "var"),
            std_delta=("delta", "std"),
            q25_delta=("delta", lambda x: x.quantile(0.25)),
            q75_delta=("delta", lambda x: x.quantile(0.75)),
            n_seeds=("sample_seed", "nunique"),
        )
    )

    return summary


def plot_mean_iqr(summary: pd.DataFrame, out_dir: Path, split: str, dataset: str, value: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    data = summary[
        (summary["split"] == split)
        & (summary["dataset"] == dataset)
    ].copy()

    if data.empty:
        return

    plt.figure(figsize=(12, 7))

    shots = sorted(data["shot"].unique())

    for shot in shots:
        s = data[data["shot"] == shot].sort_values("epoch")

        x = s["epoch"].to_numpy()
        mean = s[f"mean_{value}"].to_numpy()
        q25 = s[f"q25_{value}"].to_numpy()
        q75 = s[f"q75_{value}"].to_numpy()

        plt.plot(
            x,
            mean,
            marker="o",
            linewidth=2.4,
            label=f"{shot}-shot mean",
        )

        plt.fill_between(
            x,
            q25,
            q75,
            alpha=0.15,
        )

    if value == "delta":
        plt.axhline(0.0, linestyle="--", linewidth=1.5)

    if value == "accuracy":
        epoch0 = data[data["epoch"] == 0]
        if not epoch0.empty:
            baseline = float(epoch0[f"mean_{value}"].iloc[0])
            plt.axhline(baseline, linestyle="--", linewidth=1.5)

    epochs = sorted(data["epoch"].unique())
    plt.xticks(epochs)

    plt.xlabel("epoch")
    plt.ylabel(value)
    plt.title(f"{split} / {dataset}: mean {value} with IQR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = out_dir / f"{split}_{dataset}_{value}_mean_iqr_filtered_epochs.png"
    plt.savefig(save_path, dpi=220)
    plt.close()

    print(f"saved: {save_path}")


def plot_stat(summary: pd.DataFrame, out_dir: Path, split: str, dataset: str, value: str, stat: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    data = summary[
        (summary["split"] == split)
        & (summary["dataset"] == dataset)
    ].copy()

    if data.empty:
        return

    col = f"{stat}_{value}"

    plt.figure(figsize=(12, 7))

    shots = sorted(data["shot"].unique())

    for shot in shots:
        s = data[data["shot"] == shot].sort_values("epoch")

        plt.plot(
            s["epoch"],
            s[col],
            marker="o",
            linewidth=2.2,
            label=f"{shot}-shot",
        )

    if value == "delta" and stat in ["mean", "median"]:
        plt.axhline(0.0, linestyle="--", linewidth=1.5)

    epochs = sorted(data["epoch"].unique())
    plt.xticks(epochs)

    plt.xlabel("epoch")
    plt.ylabel(f"{stat} {value}")
    plt.title(f"{split} / {dataset}: {stat} {value}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = out_dir / f"{split}_{dataset}_{value}_{stat}_filtered_epochs.png"
    plt.savefig(save_path, dpi=220)
    plt.close()

    print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        default=("${RESEARCH_WORKSPACE}/projects/CoOp/runs/blue_texture_preserve_100seeds").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="예: 10으로 주면 epoch 0, 5, 10까지만 그림. 기본값은 전체 epoch.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = root_dir / "filtered_epoch_plots"

    df = load_curves(
        root_dir=root_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )

    if args.max_epoch is not None:
        df = df[df["epoch"] <= args.max_epoch].copy()

    summary = summarize(df)

    flat_csv = out_dir / "all_seed_curves_filtered_epochs.csv"
    summary_csv = out_dir / "seed_epoch_summary_filtered_epochs.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(flat_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"saved: {flat_csv}")
    print(f"saved: {summary_csv}")

    print("\n===== 확인 =====")
    print(f"epochs used: {sorted(df['epoch'].unique())}")
    print(f"num seeds: {df['sample_seed'].nunique()}")
    print(f"shots: {sorted(df['shot'].unique())}")
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))

    combos = summary[["split", "dataset"]].drop_duplicates()

    for _, row in combos.iterrows():
        split = row["split"]
        dataset = row["dataset"]

        for value in ["accuracy", "delta"]:
            plot_mean_iqr(
                summary=summary,
                out_dir=out_dir / "mean_iqr",
                split=split,
                dataset=dataset,
                value=value,
            )

            for stat in ["mean", "median", "var", "std"]:
                plot_stat(
                    summary=summary,
                    out_dir=out_dir / "summary_stats",
                    split=split,
                    dataset=dataset,
                    value=value,
                    stat=stat,
                )

    print("\n===== 완료 =====")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
