# -*- coding: utf-8 -*-

"""
16-shot 결과만 사용해서 4가지 best seed case를 그림.

선택 기준:
1. target DTD accuracy가 가장 높은 seed
2. ImageNet accuracy가 가장 높은 seed
3. CIFAR100 accuracy가 가장 높은 seed
4. ImageNet + CIFAR100 평균 accuracy가 가장 높은 seed

그래프:
- accuracy만 그림
- epoch 0, 5, 10, ..., 50만 표시
- selected seed는 실선
- selected seed를 제외한 other 99 seeds mean은 같은 색 점선으로 흐리게 표시
- legend는 왼쪽 아래 고정
- 제목은 3줄:
  1) case title
  2) shot / seed / epoch
  3) Target / ImageNet / CIFAR100 accuracy
"""
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

    return df


def make_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    target = df[df["split"] == "target"].copy()
    preserve = df[df["split"] == "preserve"].copy()

    target_wide = target[
        ["sample_seed", "shot", "epoch", "accuracy"]
    ].rename(
        columns={"accuracy": "target_accuracy"}
    )

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
            "score": float(best_target["target_accuracy"]),
            "score_name": "target_acc",
        },
        "best_imagenet": {
            "title": "Best ImageNet Preserve Accuracy",
            "seed": int(best_imagenet["sample_seed"]),
            "score": float(best_imagenet[imagenet_col]),
            "score_name": "imagenet_acc",
        },
        "best_cifar100": {
            "title": "Best CIFAR100 Preserve Accuracy",
            "seed": int(best_cifar100["sample_seed"]),
            "score": float(best_cifar100[cifar100_col]),
            "score_name": "cifar100_acc",
        },
        "best_preserve_avg": {
            "title": "Best Average Preserve Accuracy",
            "seed": int(best_preserve_avg["sample_seed"]),
            "score": float(best_preserve_avg["preserve_avg_accuracy"]),
            "score_name": "avg_imagenet_cifar100_acc",
        },
    }

    return cases


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


def plot_case(
    table: pd.DataFrame,
    out_dir: Path,
    shot: int,
    case_key: str,
    case_info: dict,
    selection_epoch: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = case_info["seed"]

    selected = get_selected_curve(
        table=table,
        shot=shot,
        seed=seed,
    )

    other_mean = get_other_seed_mean(
        table=table,
        shot=shot,
        selected_seed=seed,
    )

    baselines = get_baselines(table, shot)

    if selected.empty:
        print(f"[건너뜀] selected curve 없음: case={case_key}, seed={seed}")
        return

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

    # zero-shot baselines
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
    plt.tight_layout()

    save_path = out_dir / f"{case_key}_16shot_seed{seed}_accuracy.png"
    plt.savefig(save_path, dpi=220)
    plt.close()

    print(f"saved: {save_path}")


def plot_combined_2x2(
    table: pd.DataFrame,
    out_dir: Path,
    shot: int,
    cases: dict,
    selection_epoch: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = get_baselines(table, shot)

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes = axes.flatten()

    for ax, (case_key, case_info) in zip(axes, cases.items()):
        seed = case_info["seed"]

        selected = get_selected_curve(table, shot, seed)
        other_mean = get_other_seed_mean(table, shot, seed)

        if selected.empty:
            ax.set_title(f"{case_key}: empty")
            continue

        title_metric_line = make_title_metric_line(
            selected=selected,
            selection_epoch=selection_epoch,
        )

        if not other_mean.empty:
            ax.plot(
                other_mean["epoch"],
                other_mean["target_accuracy"],
                linestyle="--",
                linewidth=1.8,
                alpha=0.25,
                color="tab:blue",
                label="other mean: DTD",
            )
            ax.plot(
                other_mean["epoch"],
                other_mean["cifar100_accuracy"],
                linestyle="--",
                linewidth=1.8,
                alpha=0.25,
                color="tab:orange",
                label="other mean: CIFAR100",
            )
            ax.plot(
                other_mean["epoch"],
                other_mean["imagenet_accuracy"],
                linestyle="--",
                linewidth=1.8,
                alpha=0.25,
                color="tab:green",
                label="other mean: ImageNet",
            )

        ax.plot(
            selected["epoch"],
            selected["target_accuracy"],
            marker="o",
            linewidth=2.6,
            color="tab:blue",
            label="selected: DTD",
        )
        ax.plot(
            selected["epoch"],
            selected["cifar100_accuracy"],
            marker="o",
            linewidth=2.6,
            color="tab:orange",
            label="selected: CIFAR100",
        )
        ax.plot(
            selected["epoch"],
            selected["imagenet_accuracy"],
            marker="o",
            linewidth=2.6,
            color="tab:green",
            label="selected: ImageNet",
        )

        if baselines:
            ax.axhline(
                baselines["target"],
                linestyle="-.",
                linewidth=1.2,
                color="tab:blue",
                alpha=0.30,
            )
            ax.axhline(
                baselines["cifar100"],
                linestyle=":",
                linewidth=1.4,
                color="tab:orange",
                alpha=0.40,
            )
            ax.axhline(
                baselines["imagenet"],
                linestyle=":",
                linewidth=1.4,
                color="tab:green",
                alpha=0.40,
            )

        ax.set_xticks(sorted(selected["epoch"].unique()))
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_title(
            f"{case_info['title']}\n"
            f"{shot}-shot, selected seed={seed}, epoch={selection_epoch}\n"
            f"{title_metric_line}"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        f"16-shot best cases by selection metric "
        f"(selection epoch={selection_epoch})",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = out_dir / "combined_2x2_16shot_best_cases_accuracy.png"
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
    parser.add_argument("--shot", type=int, default=16)
    parser.add_argument("--selection-epoch", type=int, default=50)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = root_dir / "best_16shot_cases_accuracy"

    df = load_curves(
        root_dir=root_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )

    table = make_wide_table(df)

    cases = select_best_cases(
        table=table,
        shot=args.shot,
        selection_epoch=args.selection_epoch,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    case_rows = []
    for case_key, info in cases.items():
        case_rows.append(
            {
                "case": case_key,
                "title": info["title"],
                "selected_seed": info["seed"],
                "score_name": info["score_name"],
                "score": info["score"],
                "selection_epoch": args.selection_epoch,
            }
        )

    case_df = pd.DataFrame(case_rows)
    case_csv = out_dir / "selected_16shot_best_cases.csv"
    case_df.to_csv(case_csv, index=False)

    print("\n===== selected cases =====")
    print(case_df.to_string(index=False))
    print(f"\nsaved: {case_csv}")

    for case_key, case_info in cases.items():
        plot_case(
            table=table,
            out_dir=out_dir / "individual",
            shot=args.shot,
            case_key=case_key,
            case_info=case_info,
            selection_epoch=args.selection_epoch,
        )

    plot_combined_2x2(
        table=table,
        out_dir=out_dir,
        shot=args.shot,
        cases=cases,
        selection_epoch=args.selection_epoch,
    )

    print("\n===== 완료 =====")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()