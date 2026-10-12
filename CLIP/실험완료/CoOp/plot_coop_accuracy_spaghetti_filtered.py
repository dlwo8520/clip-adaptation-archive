# -*- coding: utf-8 -*-

("""
CoOp 100-seed 결과에서 accuracy만 그림.

특징:
- epoch 0, 5, 10, ..., 50만 사용
- epoch 1은 제거
- seed별 개별 곡선은 흐리게 표시
- mean / median / zero-shot baseline 표시
- target texture, preserve cifar100, preserve imagenet 모두 그림
- shot별로 각각 저장

입력:
  ${RESEARCH_WORKSPACE}/projects/CoOp/runs/blue_texture_preserve_100seeds/
    sample_seed_0/curves.jsonl
    ...
    sample_seed_99/curves.jsonl

출력:
  runs/blue_texture_preserve_100seeds/accuracy_spaghetti_filtered_epochs/
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
    epoch 0, 5, 10, ..., 50만 남김.
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

                # target
                rows.append(
                    {
                        "sample_seed": seed,
                        "shot": shot,
                        "epoch": epoch,
                        "split": "target",
                        "dataset": r["target_name"],
                        "accuracy": float(r["target"]["accuracy"]),
                    }
                )

                # preserve datasets
                for preserve_name, metric in r["preserve"].items():
                    rows.append(
                        {
                            "sample_seed": seed,
                            "shot": shot,
                            "epoch": epoch,
                            "split": "preserve",
                            "dataset": preserve_name,
                            "accuracy": float(metric["accuracy"]),
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


def plot_one(
    df: pd.DataFrame,
    out_dir: Path,
    split: str,
    dataset: str,
    shot: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df[
        (df["split"] == split)
        & (df["dataset"] == dataset)
        & (df["shot"] == shot)
    ].copy()

    if sub.empty:
        print(f"[건너뜀] split={split}, dataset={dataset}, shot={shot}")
        return

    sub = sub.sort_values(["sample_seed", "epoch"])

    plt.figure(figsize=(11, 7))

    # 1. seed별 개별 곡선
    for seed, g in sub.groupby("sample_seed"):
        g = g.sort_values("epoch")
        plt.plot(
            g["epoch"],
            g["accuracy"],
            linewidth=0.8,
            alpha=0.13,
        )

    # 2. mean / median
    stat = (
        sub.groupby("epoch", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            median_accuracy=("accuracy", "median"),
            n_seeds=("sample_seed", "nunique"),
        )
        .sort_values("epoch")
    )

    plt.plot(
        stat["epoch"],
        stat["mean_accuracy"],
        marker="o",
        linewidth=2.8,
        label="mean over seeds",
    )

    plt.plot(
        stat["epoch"],
        stat["median_accuracy"],
        marker="s",
        linestyle="--",
        linewidth=2.4,
        label="median over seeds",
    )

    # 3. zero-shot baseline: epoch 0 평균
    epoch0 = stat[stat["epoch"] == 0]

    if not epoch0.empty:
        baseline = float(epoch0["mean_accuracy"].iloc[0])
        plt.axhline(
            baseline,
            linestyle=":",
            linewidth=2.0,
            label="zero-shot baseline",
        )

    epochs = sorted(sub["epoch"].unique())
    plt.xticks(epochs)

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"{split} / {dataset} / {shot}-shot: accuracy over {sub['sample_seed'].nunique()} sample seeds")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_name = f"{split}_{dataset}_{shot}shot_accuracy_spaghetti_filtered.png"
    save_path = out_dir / save_name

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
        "--only-shot",
        type=int,
        default=None,
        help="예: 16으로 주면 16-shot만 그림. 기본값은 모든 shot.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = root_dir / "accuracy_spaghetti_filtered_epochs"

    df = load_curves(
        root_dir=root_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )

    print("===== loaded =====")
    print(f"root_dir: {root_dir}")
    print(f"epochs used: {sorted(df['epoch'].unique())}")
    print(f"num seeds: {df['sample_seed'].nunique()}")
    print(f"shots: {sorted(df['shot'].unique())}")
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))

    if args.only_shot is not None:
        shots = [args.only_shot]
    else:
        shots = sorted(df["shot"].unique())

    combos = df[["split", "dataset"]].drop_duplicates()

    for _, row in combos.iterrows():
        split = row["split"]
        dataset = row["dataset"]

        for shot in shots:
            plot_one(
                df=df,
                out_dir=out_dir,
                split=split,
                dataset=dataset,
                shot=shot,
            )

    print("\n===== 완료 =====")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
