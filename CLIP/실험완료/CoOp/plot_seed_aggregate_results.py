# -*- coding: utf-8 -*-

"""
sample_seed_0 ~ sample_seed_99 결과를 모아서 그래프를 그림.

입력 폴더 구조:
runs/blue_texture_preserve_100seeds/
  sample_seed_0/
    curves.jsonl
    results.json
    shot_1/
    shot_2/
    ...
  sample_seed_1/
    curves.jsonl
  ...

출력:
runs/blue_texture_preserve_100seeds/
  all_seed_curves_flat.csv
  seed_epoch_summary.csv
  aggregate_plots/
    raw_by_shot/
    summary_stats/
    mean_band/
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def safe_name(name: str) -> str:
    """파일명에 쓰기 어려운 문자를 정리함."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))


def load_all_curves(root_dir: Path, seed_start: int = 0, seed_end: int = 99) -> pd.DataFrame:
    """
    sample_seed_*/curves.jsonl을 전부 읽어서 long-format DataFrame으로 변환함.

    한 row 예시:
    sample_seed | shot | epoch | split | dataset | accuracy | delta
    """
    rows = []

    expected_seeds = set(range(seed_start, seed_end + 1))
    curve_files = []

    for sample_seed in range(seed_start, seed_end + 1):
        curve_path = root_dir / f"sample_seed_{sample_seed}" / "curves.jsonl"
        if curve_path.exists():
            curve_files.append(curve_path)
        else:
            print(f"[경고] curves.jsonl 없음: {curve_path}")

    if len(curve_files) == 0:
        raise FileNotFoundError(
            f"curves.jsonl을 하나도 찾지 못함: {root_dir}/sample_seed_*/curves.jsonl"
        )

    found_seeds = set()

    for curve_path in curve_files:
        folder_name = curve_path.parent.name
        folder_seed = int(folder_name.replace("sample_seed_", ""))
        found_seeds.add(folder_seed)

        with open(curve_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)

                sample_seed = int(r.get("sample_seed", folder_seed))

                base = {
                    "sample_seed": sample_seed,
                    "shot": int(r["shot"]),
                    "epoch": int(r["epoch"]),
                }

                # target 결과
                rows.append(
                    {
                        **base,
                        "split": "target",
                        "dataset": r["target_name"],
                        "accuracy": float(r["target"]["accuracy"]),
                        "delta": float(r["target"]["delta_from_zero_shot"]),
                    }
                )

                # preserve 결과
                for preserve_name, metric in r["preserve"].items():
                    rows.append(
                        {
                            **base,
                            "split": "preserve",
                            "dataset": preserve_name,
                            "accuracy": float(metric["accuracy"]),
                            "delta": float(metric["delta_from_zero_shot"]),
                        }
                    )

    missing = sorted(expected_seeds - found_seeds)
    if missing:
        print(f"[경고] 누락된 sample_seed 있음: {missing[:20]}{' ...' if len(missing) > 20 else ''}")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("읽은 결과가 비어 있음.")

    # 혹시 같은 curves.jsonl 안에 중복 row가 있으면 마지막 것만 사용
    key_cols = ["sample_seed", "shot", "epoch", "split", "dataset"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"[경고] 중복 row 제거됨: {before - after}개")

    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """epoch별 평균, 중앙값, 분산, 표준편차, IQR 계산."""
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
            min_delta=("delta", "min"),
            max_delta=("delta", "max"),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
            n_seeds=("sample_seed", "nunique"),
        )
    )

    return summary


def get_baseline_from_summary(summary_part: pd.DataFrame, value: str):
    """epoch 0 값을 기준선으로 사용."""
    epoch0 = summary_part[summary_part["epoch"] == 0]
    if epoch0.empty:
        return None

    if value == "accuracy":
        return float(epoch0[f"mean_{value}"].iloc[0])

    if value == "delta":
        return 0.0

    return None


def plot_raw_by_shot(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    split: str,
    dataset: str,
    value: str,
):
    """
    shot별로 100개 seed curve를 얇게 그리고,
    평균과 중앙값을 굵게 그림.

    출력 개수:
    dataset 1개 × metric 1개 × shot 5개
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    shots = sorted(df["shot"].unique())

    for shot in shots:
        raw = df[
            (df["split"] == split)
            & (df["dataset"] == dataset)
            & (df["shot"] == shot)
        ].copy()

        stat = summary[
            (summary["split"] == split)
            & (summary["dataset"] == dataset)
            & (summary["shot"] == shot)
        ].sort_values("epoch").copy()

        if raw.empty or stat.empty:
            continue

        plt.figure(figsize=(12, 7))

        # seed별 개별 곡선
        for sample_seed, g in raw.groupby("sample_seed"):
            g = g.sort_values("epoch")
            plt.plot(
                g["epoch"],
                g[value],
                alpha=0.12,
                linewidth=0.9,
            )

        # 평균 곡선
        plt.plot(
            stat["epoch"],
            stat[f"mean_{value}"],
            marker="o",
            linewidth=3.0,
            label=f"mean over seeds",
        )

        # 중앙값 곡선
        plt.plot(
            stat["epoch"],
            stat[f"median_{value}"],
            marker="s",
            linestyle="--",
            linewidth=2.5,
            label=f"median over seeds",
        )

        baseline = get_baseline_from_summary(stat, value)
        if baseline is not None:
            plt.axhline(baseline, linestyle=":", linewidth=2.0, label="zero-shot baseline")

        n_seed = int(stat["n_seeds"].max())

        plt.xlabel("epoch")
        plt.ylabel(value)
        plt.title(f"{split} / {dataset} / {shot}-shot: {value} over {n_seed} sample seeds")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_path = out_dir / f"{safe_name(split)}_{safe_name(dataset)}_{value}_shot_{shot}_raw_mean_median.png"
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"saved: {save_path}")


def plot_stat_across_shots(
    summary: pd.DataFrame,
    out_dir: Path,
    split: str,
    dataset: str,
    value: str,
    stat_name: str,
):
    """
    평균 / 중앙값 / 분산을 shot별로 한 그림에 그림.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    col = f"{stat_name}_{value}"

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

        plt.plot(
            s["epoch"],
            s[col],
            marker="o",
            linewidth=2.3,
            label=f"{shot}-shot",
        )

    if value == "delta" and stat_name in ["mean", "median"]:
        plt.axhline(0.0, linestyle="--", linewidth=2.0, label="zero-shot baseline")

    if value == "accuracy" and stat_name in ["mean", "median"]:
        # accuracy는 epoch 0의 평균값을 기준선으로 둠
        epoch0 = data[data["epoch"] == 0]
        if not epoch0.empty:
            baseline = float(epoch0[f"{stat_name}_{value}"].iloc[0])
            plt.axhline(baseline, linestyle="--", linewidth=2.0, label="zero-shot baseline")

    n_seed = int(data["n_seeds"].max())

    plt.xlabel("epoch")
    plt.ylabel(f"{stat_name} {value}")
    plt.title(f"{split} / {dataset}: {stat_name} {value} over {n_seed} sample seeds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = out_dir / f"{safe_name(split)}_{safe_name(dataset)}_{value}_{stat_name}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"saved: {save_path}")


def plot_mean_with_iqr(
    summary: pd.DataFrame,
    out_dir: Path,
    split: str,
    dataset: str,
    value: str,
):
    """
    평균 곡선 + 25~75% 분위수 영역을 그림.
    seed 변동성이 한눈에 보임.
    """
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
        plt.axhline(0.0, linestyle="--", linewidth=2.0, label="zero-shot baseline")

    if value == "accuracy":
        epoch0 = data[data["epoch"] == 0]
        if not epoch0.empty:
            baseline = float(epoch0[f"mean_{value}"].iloc[0])
            plt.axhline(baseline, linestyle="--", linewidth=2.0, label="zero-shot baseline")

    n_seed = int(data["n_seeds"].max())

    plt.xlabel("epoch")
    plt.ylabel(value)
    plt.title(f"{split} / {dataset}: mean {value} with IQR over {n_seed} sample seeds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = out_dir / f"{safe_name(split)}_{safe_name(dataset)}_{value}_mean_iqr.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"saved: {save_path}")


def plot_mean_with_std(
    summary: pd.DataFrame,
    out_dir: Path,
    split: str,
    dataset: str,
    value: str,
):
    """
    평균 곡선 + 표준편차 영역을 그림.
    분산은 별도 variance 그래프로도 저장됨.
    """
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
        std = s[f"std_{value}"].fillna(0.0).to_numpy()

        plt.plot(
            x,
            mean,
            marker="o",
            linewidth=2.4,
            label=f"{shot}-shot mean",
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.12,
        )

    if value == "delta":
        plt.axhline(0.0, linestyle="--", linewidth=2.0, label="zero-shot baseline")

    if value == "accuracy":
        epoch0 = data[data["epoch"] == 0]
        if not epoch0.empty:
            baseline = float(epoch0[f"mean_{value}"].iloc[0])
            plt.axhline(baseline, linestyle="--", linewidth=2.0, label="zero-shot baseline")

    n_seed = int(data["n_seeds"].max())

    plt.xlabel("epoch")
    plt.ylabel(value)
    plt.title(f"{split} / {dataset}: mean {value} with std over {n_seed} sample seeds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = out_dir / f"{safe_name(split)}_{safe_name(dataset)}_{value}_mean_std.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"saved: {save_path}")


def print_basic_report(df: pd.DataFrame, summary: pd.DataFrame):
    """간단한 점검용 출력."""
    print("\n===== loaded data summary =====")
    print(f"rows: {len(df)}")
    print(f"sample seeds: {df['sample_seed'].min()} ~ {df['sample_seed'].max()}")
    print(f"num sample seeds: {df['sample_seed'].nunique()}")
    print(f"shots: {sorted(df['shot'].unique())}")
    print(f"epochs: {sorted(df['epoch'].unique())}")
    print(f"datasets:")
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))

    print("\n===== n_seeds by split/dataset/shot =====")
    check = (
        df.groupby(["split", "dataset", "shot"])["sample_seed"]
        .nunique()
        .reset_index(name="n_seeds")
    )
    print(check.to_string(index=False))

    print("\n===== final epoch mean delta =====")
    max_epoch = int(df["epoch"].max())
    final = summary[summary["epoch"] == max_epoch].copy()
    final = final[
        ["split", "dataset", "shot", "mean_delta", "median_delta", "var_delta", "n_seeds"]
    ].sort_values(["split", "dataset", "shot"])
    print(final.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        default="runs/blue_texture_preserve_100seeds",
        help="sample_seed_0 ... sample_seed_99가 들어있는 루트 폴더",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = root_dir / "aggregate_plots"

    raw_dir = out_dir / "raw_by_shot"
    stat_dir = out_dir / "summary_stats"
    band_dir = out_dir / "mean_band"

    df = load_all_curves(
        root_dir=root_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )

    summary = make_summary(df)

    # CSV 저장
    flat_csv = root_dir / "all_seed_curves_flat.csv"
    summary_csv = root_dir / "seed_epoch_summary.csv"

    df.to_csv(flat_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"saved: {flat_csv}")
    print(f"saved: {summary_csv}")

    print_basic_report(df, summary)

    combos = summary[["split", "dataset"]].drop_duplicates()

    for _, combo in combos.iterrows():
        split = combo["split"]
        dataset = combo["dataset"]

        for value in ["accuracy", "delta"]:
            # 1. shot별로 100개 seed curve + mean + median
            plot_raw_by_shot(
                df=df,
                summary=summary,
                out_dir=raw_dir,
                split=split,
                dataset=dataset,
                value=value,
            )

            # 2. 평균 / 중앙값 / 분산 / 표준편차를 shot끼리 비교
            for stat_name in ["mean", "median", "var", "std"]:
                plot_stat_across_shots(
                    summary=summary,
                    out_dir=stat_dir,
                    split=split,
                    dataset=dataset,
                    value=value,
                    stat_name=stat_name,
                )

            # 3. mean + IQR 영역
            plot_mean_with_iqr(
                summary=summary,
                out_dir=band_dir,
                split=split,
                dataset=dataset,
                value=value,
            )

            # 4. mean + std 영역
            plot_mean_with_std(
                summary=summary,
                out_dir=band_dir,
                split=split,
                dataset=dataset,
                value=value,
            )

    print("\n===== 완료 =====")
    print(f"plot dir: {out_dir}")


if __name__ == "__main__":
    main()
