# -*- coding: utf-8 -*-

"""
CLIP-Adapter 100-seed 결과 그래프 코드.

그리는 그래프:
1. target / preserve dataset별 mean accuracy + IQR
2. 16-shot 기준 100 seed spaghetti plot
3. 16-shot 기준 best case 4개
   - Best Target DTD Accuracy
   - Best ImageNet Preserve Accuracy
   - Best CIFAR100 Preserve Accuracy
   - Best Average Preserve Accuracy

주의:
- delta는 사용하지 않음.
- epoch 0, 5, 10, ..., 50만 사용함.
- curves.jsonl 구조가 기존 CoOp과 동일하다고 가정함.
"""
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[3])))

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))


def keep_plot_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """
    epoch 0, 5, 10, ..., 50만 남김.
    epoch 1 같은 중간 checkpoint는 제거.
    """
    if df.empty:
        return df

    return df[(df["epoch"] == 0) | (df["epoch"] % 5 == 0)].copy()


def load_curves(root_dir: Path, seed_start: int, seed_end: int) -> pd.DataFrame:
    rows = []

    for sample_seed in range(seed_start, seed_end + 1):
        curve_path = root_dir / f"sample_seed_{sample_seed}" / "curves.jsonl"

        if not curve_path.exists():
            print(f"[경고] curves.jsonl 없음: {curve_path}")
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
        raise RuntimeError("curves.jsonl 결과를 하나도 읽지 못함.")

    key_cols = ["sample_seed", "shot", "epoch", "split", "dataset"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"[경고] 중복 row 제거됨: {before - after}개")

    df = keep_plot_epochs(df)

    return df


def summarize_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["split", "dataset", "shot", "epoch"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            median_accuracy=("accuracy", "median"),
            q25_accuracy=("accuracy", lambda x: x.quantile(0.25)),
            q75_accuracy=("accuracy", lambda x: x.quantile(0.75)),
            std_accuracy=("accuracy", "std"),
            var_accuracy=("accuracy", "var"),
            n_seeds=("sample_seed", "nunique"),
        )
    )

    return summary


def make_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    target = df[df["split"] == "target"].copy()
    preserve = df[df["split"] == "preserve"].copy()

    target_wide = target[
        ["sample_seed", "shot", "epoch", "accuracy"]
    ].rename(columns={"accuracy": "target_accuracy"})

    preserve_wide = preserve.pivot_table(
        index=["sample_seed", "shot", "epoch"],
        columns="dataset",
        values="accuracy",
        aggfunc="last",
    ).reset_index()

    preserve_wide.columns = [
        str(c) if not isinstance(c, tuple) else "_".join(c)
        for c in preserve_wide.columns
    ]

    merged = target_wide.merge(
        preserve_wide,
        on=["sample_seed", "shot", "epoch"],
        how="inner",
    )

    return merged


def find_dataset_col(table: pd.DataFrame, keyword: str) -> str:
    keyword = keyword.lower()

    candidates = [
        c for c in table.columns
        if keyword in c.lower()
    ]

    if not candidates:
        raise RuntimeError(
            f"{keyword} dataset column을 찾지 못함. columns={list(table.columns)}"
        )

    return candidates[0]


def get_baseline_from_summary(data: pd.DataFrame) -> float:
    epoch0 = data[data["epoch"] == 0]
    if epoch0.empty:
        return None

    return float(epoch0["mean_accuracy"].iloc[0])


def apply_accuracy_ylim(ylim):
    """
    accuracy 그래프 y축 범위를 고정함.
    ylim은 (ymin, ymax) 튜플 또는 리스트로 받음.
    None이면 matplotlib 자동 범위를 그대로 사용함.
    """
    if ylim is None:
        return

    ymin, ymax = float(ylim[0]), float(ylim[1])
    plt.ylim(ymin, ymax)


def plot_mean_iqr_accuracy(summary: pd.DataFrame, out_dir: Path, ylim=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = summary[["split", "dataset"]].drop_duplicates()

    for _, row in combos.iterrows():
        split = row["split"]
        dataset = row["dataset"]

        data = summary[
            (summary["split"] == split)
            & (summary["dataset"] == dataset)
        ].copy()

        if data.empty:
            continue

        plt.figure(figsize=(12, 7))

        for shot in sorted(data["shot"].unique()):
            s = data[data["shot"] == shot].sort_values("epoch")

            x = s["epoch"].to_numpy()
            mean = s["mean_accuracy"].to_numpy()
            q25 = s["q25_accuracy"].to_numpy()
            q75 = s["q75_accuracy"].to_numpy()

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

        baseline = get_baseline_from_summary(data)
        if baseline is not None:
            plt.axhline(
                baseline,
                linestyle="--",
                linewidth=1.6,
                label="zero-shot baseline",
            )

        epochs = sorted(data["epoch"].unique())
        plt.xticks(epochs)

        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"{split} / {dataset}: mean accuracy with IQR")
        plt.grid(True, alpha=0.25)
        plt.legend()
        apply_accuracy_ylim(ylim)
        plt.tight_layout()

        save_path = out_dir / f"{safe_name(split)}_{safe_name(dataset)}_mean_accuracy_iqr.png"
        plt.savefig(save_path, dpi=220)
        plt.close()

        print(f"saved: {save_path}")


def plot_spaghetti_accuracy(
    df: pd.DataFrame,
    out_dir: Path,
    only_shot: int = 16,
    ylim=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = df[["split", "dataset"]].drop_duplicates()

    for _, row in combos.iterrows():
        split = row["split"]
        dataset = row["dataset"]

        sub = df[
            (df["split"] == split)
            & (df["dataset"] == dataset)
            & (df["shot"] == only_shot)
        ].copy()

        if sub.empty:
            continue

        sub = sub.sort_values(["sample_seed", "epoch"])

        plt.figure(figsize=(11, 7))

        # seed별 개별 곡선
        for seed, g in sub.groupby("sample_seed"):
            g = g.sort_values("epoch")
            plt.plot(
                g["epoch"],
                g["accuracy"],
                linewidth=0.8,
                alpha=0.13,
            )

        # mean / median
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

        epoch0 = stat[stat["epoch"] == 0]
        if not epoch0.empty:
            baseline = float(epoch0["mean_accuracy"].iloc[0])
            plt.axhline(
                baseline,
                linestyle=":",
                linewidth=2.0,
                label="zero-shot baseline",
            )

        plt.xticks(sorted(sub["epoch"].unique()))
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(
            f"{split} / {dataset} / {only_shot}-shot: "
            f"accuracy over {sub['sample_seed'].nunique()} sample seeds"
        )
        plt.grid(True, alpha=0.25)
        plt.legend()
        apply_accuracy_ylim(ylim)
        plt.tight_layout()

        save_path = out_dir / f"{safe_name(split)}_{safe_name(dataset)}_{only_shot}shot_spaghetti_accuracy.png"
        plt.savefig(save_path, dpi=220)
        plt.close()

        print(f"saved: {save_path}")


def get_other_seed_mean(
    table: pd.DataFrame,
    shot: int,
    selected_seed: int,
) -> pd.DataFrame:
    sub = table[
        (table["shot"] == shot)
        & (table["sample_seed"] != selected_seed)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    imagenet_col = find_dataset_col(sub, "imagenet")
    cifar100_col = find_dataset_col(sub, "cifar100")

    mean_df = (
        sub.groupby("epoch", as_index=False)
        .agg(
            target_accuracy=("target_accuracy", "mean"),
            imagenet_accuracy=(imagenet_col, "mean"),
            cifar100_accuracy=(cifar100_col, "mean"),
        )
        .sort_values("epoch")
    )

    return keep_plot_epochs(mean_df)


def get_selected_curve(
    table: pd.DataFrame,
    shot: int,
    seed: int,
) -> pd.DataFrame:
    sub = table[
        (table["shot"] == shot)
        & (table["sample_seed"] == seed)
    ].copy()

    if sub.empty:
        return sub

    imagenet_col = find_dataset_col(sub, "imagenet")
    cifar100_col = find_dataset_col(sub, "cifar100")

    sub = sub.rename(
        columns={
            imagenet_col: "imagenet_accuracy",
            cifar100_col: "cifar100_accuracy",
        }
    )

    sub = sub.sort_values("epoch")
    sub = keep_plot_epochs(sub)

    return sub


def get_baselines(table: pd.DataFrame, shot: int) -> dict:
    sub = table[
        (table["shot"] == shot)
        & (table["epoch"] == 0)
    ].copy()

    if sub.empty:
        return {}

    row = sub.iloc[0]

    imagenet_col = find_dataset_col(sub, "imagenet")
    cifar100_col = find_dataset_col(sub, "cifar100")

    return {
        "target": float(row["target_accuracy"]),
        "imagenet": float(row[imagenet_col]),
        "cifar100": float(row[cifar100_col]),
    }


def select_best_cases(
    table: pd.DataFrame,
    shot: int,
    selection_epoch: int,
) -> dict:
    sub = table[
        (table["shot"] == shot)
        & (table["epoch"] == selection_epoch)
    ].copy()

    if sub.empty:
        raise RuntimeError(f"{shot}-shot, epoch={selection_epoch} 결과가 없음.")

    imagenet_col = find_dataset_col(sub, "imagenet")
    cifar100_col = find_dataset_col(sub, "cifar100")

    sub["preserve_avg_accuracy"] = sub[[imagenet_col, cifar100_col]].mean(axis=1)

    best_target = sub.sort_values(
        ["target_accuracy", "preserve_avg_accuracy"],
        ascending=[False, False],
    ).iloc[0]

    best_imagenet = sub.sort_values(
        [imagenet_col, "target_accuracy"],
        ascending=[False, False],
    ).iloc[0]

    best_cifar100 = sub.sort_values(
        [cifar100_col, "target_accuracy"],
        ascending=[False, False],
    ).iloc[0]

    best_preserve_avg = sub.sort_values(
        ["preserve_avg_accuracy", "target_accuracy"],
        ascending=[False, False],
    ).iloc[0]

    cases = {
        "best_target": {
            "title": "Best Target DTD Accuracy",
            "seed": int(best_target["sample_seed"]),
        },
        "best_imagenet": {
            "title": "Best ImageNet Preserve Accuracy",
            "seed": int(best_imagenet["sample_seed"]),
        },
        "best_cifar100": {
            "title": "Best CIFAR100 Preserve Accuracy",
            "seed": int(best_cifar100["sample_seed"]),
        },
        "best_preserve_avg": {
            "title": "Best Average Preserve Accuracy",
            "seed": int(best_preserve_avg["sample_seed"]),
        },
    }

    return cases


def make_title_metric_line(
    selected: pd.DataFrame,
    selection_epoch: int,
) -> str:
    score_row = selected[selected["epoch"] == selection_epoch]

    if score_row.empty:
        score_row = selected.iloc[[-1]]

    score_row = score_row.iloc[0]

    return (
        f"Target={score_row['target_accuracy'] * 100:.2f}% / "
        f"ImageNet={score_row['imagenet_accuracy'] * 100:.2f}% / "
        f"CIFAR100={score_row['cifar100_accuracy'] * 100:.2f}%"
    )


def plot_best_case(
    table: pd.DataFrame,
    out_dir: Path,
    shot: int,
    case_key: str,
    case_info: dict,
    selection_epoch: int,
    ylim=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = case_info["seed"]

    selected = get_selected_curve(
        table=table,
        shot=shot,
        seed=seed,
    )

    if selected.empty:
        print(f"[건너뜀] selected curve 없음: case={case_key}, seed={seed}")
        return

    other_mean = get_other_seed_mean(
        table=table,
        shot=shot,
        selected_seed=seed,
    )

    baselines = get_baselines(table, shot)

    title_metric_line = make_title_metric_line(
        selected=selected,
        selection_epoch=selection_epoch,
    )

    plt.figure(figsize=(11, 7))

    # other 99 seeds mean
    if not other_mean.empty:
        plt.plot(
            other_mean["epoch"],
            other_mean["target_accuracy"],
            linestyle="--",
            linewidth=2.2,
            alpha=0.30,
            color="tab:blue",
            label="other 99 seeds mean: DTD",
        )

        plt.plot(
            other_mean["epoch"],
            other_mean["cifar100_accuracy"],
            linestyle="--",
            linewidth=2.2,
            alpha=0.30,
            color="tab:orange",
            label="other 99 seeds mean: CIFAR100",
        )

        plt.plot(
            other_mean["epoch"],
            other_mean["imagenet_accuracy"],
            linestyle="--",
            linewidth=2.2,
            alpha=0.30,
            color="tab:green",
            label="other 99 seeds mean: ImageNet",
        )

    # selected seed
    plt.plot(
        selected["epoch"],
        selected["target_accuracy"],
        marker="o",
        linewidth=3.0,
        color="tab:blue",
        label="selected seed: DTD accuracy",
    )

    plt.plot(
        selected["epoch"],
        selected["cifar100_accuracy"],
        marker="o",
        linewidth=3.0,
        color="tab:orange",
        label="selected seed: CIFAR100 accuracy",
    )

    plt.plot(
        selected["epoch"],
        selected["imagenet_accuracy"],
        marker="o",
        linewidth=3.0,
        color="tab:green",
        label="selected seed: ImageNet accuracy",
    )

    # baselines
    if baselines:
        plt.axhline(
            baselines["target"],
            linestyle="-.",
            linewidth=1.5,
            color="tab:blue",
            alpha=0.35,
            label="DTD zero-shot baseline",
        )

        plt.axhline(
            baselines["cifar100"],
            linestyle=":",
            linewidth=1.8,
            color="tab:orange",
            alpha=0.45,
            label="CIFAR100 zero-shot baseline",
        )

        plt.axhline(
            baselines["imagenet"],
            linestyle=":",
            linewidth=1.8,
            color="tab:green",
            alpha=0.45,
            label="ImageNet zero-shot baseline",
        )

    plt.xticks(sorted(selected["epoch"].unique()))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(
        f"{case_info['title']}\n"
        f"{shot}-shot, selected seed={seed}, epoch={selection_epoch}\n"
        f"{title_metric_line}"
    )
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="lower left")
    apply_accuracy_ylim(ylim)
    plt.tight_layout()

    save_path = out_dir / f"{case_key}_{shot}shot_seed{seed}_accuracy.png"
    plt.savefig(save_path, dpi=220)
    plt.close()

    print(f"saved: {save_path}")


def plot_best_cases(
    table: pd.DataFrame,
    out_dir: Path,
    shot: int = 16,
    selection_epoch: int = 50,
    ylim=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = select_best_cases(
        table=table,
        shot=shot,
        selection_epoch=selection_epoch,
    )

    rows = []
    for case_key, info in cases.items():
        selected = get_selected_curve(table, shot, info["seed"])
        score_row = selected[selected["epoch"] == selection_epoch]
        if score_row.empty:
            score_row = selected.iloc[[-1]]
        score_row = score_row.iloc[0]

        rows.append(
            {
                "case": case_key,
                "title": info["title"],
                "selected_seed": info["seed"],
                "selection_epoch": selection_epoch,
                "target_accuracy": float(score_row["target_accuracy"]),
                "imagenet_accuracy": float(score_row["imagenet_accuracy"]),
                "cifar100_accuracy": float(score_row["cifar100_accuracy"]),
                "preserve_avg_accuracy": float(
                    (score_row["imagenet_accuracy"] + score_row["cifar100_accuracy"]) / 2
                ),
            }
        )

    selected_df = pd.DataFrame(rows)
    selected_csv = out_dir / f"selected_{shot}shot_best_cases_accuracy.csv"
    selected_df.to_csv(selected_csv, index=False)

    print("\n===== selected best cases =====")
    print(selected_df.to_string(index=False))
    print(f"saved: {selected_csv}")

    for case_key, case_info in cases.items():
        plot_best_case(
            table=table,
            out_dir=out_dir / "individual",
            shot=shot,
            case_key=case_key,
            case_info=case_info,
            selection_epoch=selection_epoch,
            ylim=ylim,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        default=("${RESEARCH_WORKSPACE}/projects/CoOp/runs/blue_texture_adapter_100seeds").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)),
        help="sample_seed_0 ... sample_seed_99가 들어있는 CLIP-Adapter 결과 폴더",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument("--best-shot", type=int, default=16)
    parser.add_argument("--selection-epoch", type=int, default=50)

    # 그래프 종류별 y축 고정값.
    # 기본값은 사용자가 지정한 0.30 ~ 0.75임.
    parser.add_argument(
        "--ylim-mean-iqr",
        type=float,
        nargs=2,
        default=[0.30, 0.75],
        metavar=("YMIN", "YMAX"),
        help="mean_iqr_accuracy 그래프 y축 범위",
    )
    parser.add_argument(
        "--ylim-spaghetti",
        type=float,
        nargs=2,
        default=[0.30, 0.75],
        metavar=("YMIN", "YMAX"),
        help="spaghetti_16shot_accuracy 그래프 y축 범위",
    )
    parser.add_argument(
        "--ylim-best-cases",
        type=float,
        nargs=2,
        default=[0.30, 0.75],
        metavar=("YMIN", "YMAX"),
        help="best_16shot_cases_accuracy 그래프 y축 범위",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = root_dir / "adapter_accuracy_plots"

    df = load_curves(
        root_dir=root_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )

    summary = summarize_accuracy(df)
    table = make_wide_table(df)

    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "adapter_all_seed_accuracy_flat.csv", index=False)
    summary.to_csv(out_dir / "adapter_accuracy_summary.csv", index=False)

    print("\n===== loaded =====")
    print(f"root_dir: {root_dir}")
    print(f"epochs used: {sorted(df['epoch'].unique())}")
    print(f"num seeds: {df['sample_seed'].nunique()}")
    print(f"shots: {sorted(df['shot'].unique())}")
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))
    print(f"ylim mean_iqr_accuracy: {args.ylim_mean_iqr}")
    print(f"ylim spaghetti_16shot_accuracy: {args.ylim_spaghetti}")
    print(f"ylim best_16shot_cases_accuracy: {args.ylim_best_cases}")

    plot_mean_iqr_accuracy(
        summary=summary,
        out_dir=out_dir / "mean_iqr_accuracy",
        ylim=args.ylim_mean_iqr,
    )

    plot_spaghetti_accuracy(
        df=df,
        out_dir=out_dir / "spaghetti_16shot_accuracy",
        only_shot=args.best_shot,
        ylim=args.ylim_spaghetti,
    )

    plot_best_cases(
        table=table,
        out_dir=out_dir / "best_16shot_cases_accuracy",
        shot=args.best_shot,
        selection_epoch=args.selection_epoch,
        ylim=args.ylim_best_cases,
    )

    print("\n===== 완료 =====")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()