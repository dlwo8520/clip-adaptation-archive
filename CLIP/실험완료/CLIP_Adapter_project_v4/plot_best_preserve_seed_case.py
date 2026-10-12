# -*- coding: utf-8 -*-

"""
shot별로 zero-shot preservation이 가장 좋은 sample_seed를 고르고,
각 shot마다 한 이미지에 2개 그래프를 그림.

이번 버전:
- delta가 아니라 accuracy로 그림
- 그래프에는 epoch 0, 5, 10, 15, ... 만 표시함
- epoch 1은 그래프에서 제외함
- preserve mean curve는 그리지 않음
- CIFAR100 / ImageNet preserve accuracy를 따로 그림
- 왼쪽 그래프에 selected seed + other 99 seeds mean을 함께 표시
- other 99 seeds mean은 같은 색 점선으로 흐리게 표시
- 오른쪽 그래프는 target accuracy vs preserve accuracy trade-off를 그림
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(x))


def keep_plot_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """
    그래프에는 epoch 0과 5 단위 epoch만 표시함.
    즉 0, 5, 10, 15, ... 만 남기고 epoch 1은 제거함.
    """
    if df.empty:
        return df

    return df[
        (df["epoch"] == 0) | (df["epoch"] % 5 == 0)
    ].copy()


def load_all_curves(root_dir: Path, seed_start: int, seed_end: int) -> pd.DataFrame:
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

    return df


def get_dataset_preserve_delta_cols(table: pd.DataFrame) -> list:
    """
    실제 preserve dataset delta column만 반환함.

    포함:
    - cifar100_delta
    - imagenet_delta

    제외:
    - target_delta
    - preserve_mean_delta
    - preserve_min_delta
    - preserve_max_delta
    """
    exclude_cols = {
        "target_delta",
        "preserve_mean_delta",
        "preserve_min_delta",
        "preserve_max_delta",
    }

    return sorted(
        [
            c for c in table.columns
            if c.endswith("_delta") and c not in exclude_cols
        ]
    )


def get_dataset_preserve_accuracy_cols(table: pd.DataFrame) -> list:
    """
    실제 preserve dataset accuracy column만 반환함.

    포함:
    - cifar100_accuracy
    - imagenet_accuracy

    제외:
    - target_accuracy
    - preserve_mean_accuracy
    - preserve_min_accuracy
    - preserve_max_accuracy
    """
    exclude_cols = {
        "target_accuracy",
        "preserve_mean_accuracy",
        "preserve_min_accuracy",
        "preserve_max_accuracy",
    }

    return sorted(
        [
            c for c in table.columns
            if c.endswith("_accuracy") and c not in exclude_cols
        ]
    )


def make_seed_epoch_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    sample_seed, shot, epoch별로 아래 값을 한 row로 정리함.

    - target_accuracy
    - target_delta
    - cifar100_accuracy
    - cifar100_delta
    - imagenet_accuracy
    - imagenet_delta
    """
    target = df[df["split"] == "target"].copy()
    preserve = df[df["split"] == "preserve"].copy()

    target_wide = target[
        ["sample_seed", "shot", "epoch", "accuracy", "delta"]
    ].rename(
        columns={
            "accuracy": "target_accuracy",
            "delta": "target_delta",
        }
    )

    preserve_wide = preserve.pivot_table(
        index=["sample_seed", "shot", "epoch"],
        columns="dataset",
        values=["accuracy", "delta"],
        aggfunc="last",
    )

    preserve_wide.columns = [
        f"{dataset}_{metric}"
        for metric, dataset in preserve_wide.columns
    ]

    preserve_wide = preserve_wide.reset_index()

    merged = target_wide.merge(
        preserve_wide,
        on=["sample_seed", "shot", "epoch"],
        how="inner",
    )

    preserve_delta_cols = get_dataset_preserve_delta_cols(merged)
    preserve_acc_cols = get_dataset_preserve_accuracy_cols(merged)

    if len(preserve_delta_cols) == 0:
        raise RuntimeError("preserve delta column을 찾지 못함.")

    if len(preserve_acc_cols) == 0:
        raise RuntimeError("preserve accuracy column을 찾지 못함.")

    # seed 선택 기준용.
    # 그래프에는 preserve mean을 직접 그리지 않음.
    merged["preserve_mean_delta"] = merged[preserve_delta_cols].mean(axis=1)
    merged["preserve_min_delta"] = merged[preserve_delta_cols].min(axis=1)
    merged["preserve_max_delta"] = merged[preserve_delta_cols].max(axis=1)

    merged["preserve_mean_accuracy"] = merged[preserve_acc_cols].mean(axis=1)
    merged["preserve_min_accuracy"] = merged[preserve_acc_cols].min(axis=1)
    merged["preserve_max_accuracy"] = merged[preserve_acc_cols].max(axis=1)

    return merged


def select_best_seed_by_shot(
    table: pd.DataFrame,
    selection_epoch: int,
    min_target_delta: float,
) -> pd.DataFrame:
    """
    shot별로 preserve_mean_delta가 가장 높은 seed를 선택함.

    preserve_mean_delta는 보통 음수임.
    따라서 0에 가까울수록 zero-shot 성능 하락이 적음.

    그래프는 accuracy로 그리지만,
    seed 선택은 baseline 차이를 제거한 delta 기준으로 하는 것이 더 안전함.
    """
    final = table[table["epoch"] == selection_epoch].copy()

    if final.empty:
        raise ValueError(f"selection_epoch={selection_epoch} 결과가 없음.")

    rows = []

    for shot, g in final.groupby("shot"):
        candidates = g[g["target_delta"] >= min_target_delta].copy()

        if candidates.empty:
            print(
                f"[경고] shot={shot}: target_delta >= {min_target_delta} 조건 만족 seed 없음. "
                f"전체 seed에서 preserve 기준으로 선택함."
            )
            candidates = g.copy()

        best = candidates.sort_values(
            ["preserve_mean_delta", "target_delta"],
            ascending=[False, False],
        ).iloc[0]

        row = {
            "shot": int(shot),
            "selected_sample_seed": int(best["sample_seed"]),
            "selection_epoch": int(selection_epoch),
            "target_accuracy_at_selection": float(best["target_accuracy"]),
            "target_delta_at_selection": float(best["target_delta"]),
            "preserve_mean_accuracy_at_selection": float(best["preserve_mean_accuracy"]),
            "preserve_mean_delta_at_selection": float(best["preserve_mean_delta"]),
            "min_target_delta_filter": float(min_target_delta),
        }

        for col in get_dataset_preserve_accuracy_cols(final):
            row[f"{col}_at_selection"] = float(best[col])

        for col in get_dataset_preserve_delta_cols(final):
            row[f"{col}_at_selection"] = float(best[col])

        rows.append(row)

    return pd.DataFrame(rows).sort_values("shot").reset_index(drop=True)


def get_other_seed_mean_accuracy_for_shot(
    table: pd.DataFrame,
    shot: int,
    selected_seed: int,
) -> pd.DataFrame:
    """
    같은 shot에서 selected_seed를 제외한 나머지 seed들의
    dataset별 accuracy 평균을 계산함.

    selected seed가 1개이면 나머지 99개 seed 평균.
    """
    sub = table[
        (table["shot"] == shot)
        & (table["sample_seed"] != selected_seed)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    preserve_acc_cols = get_dataset_preserve_accuracy_cols(table)

    agg_dict = {
        "target_accuracy": ("target_accuracy", "mean"),
    }

    for col in preserve_acc_cols:
        agg_dict[col] = (col, "mean")

    out = (
        sub.groupby("epoch", as_index=False)
        .agg(**agg_dict)
        .sort_values("epoch")
    )

    return keep_plot_epochs(out)


def get_baselines_for_shot(table: pd.DataFrame, shot: int) -> dict:
    """
    epoch 0 기준으로 target / preserve baseline accuracy를 가져옴.
    """
    sub = table[(table["shot"] == shot) & (table["epoch"] == 0)].copy()

    if sub.empty:
        return {}

    row = sub.iloc[0]

    baselines = {
        "target_accuracy": float(row["target_accuracy"]),
    }

    for col in get_dataset_preserve_accuracy_cols(table):
        baselines[col] = float(row[col])

    return baselines


def plot_one_shot_case(
    table: pd.DataFrame,
    selected_row: pd.Series,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    shot = int(selected_row["shot"])
    seed = int(selected_row["selected_sample_seed"])

    selected = table[
        (table["shot"] == shot)
        & (table["sample_seed"] == seed)
    ].sort_values("epoch").copy()

    selected = keep_plot_epochs(selected)

    if selected.empty:
        print(f"[경고] selected curve 없음: shot={shot}, seed={seed}")
        return

    preserve_acc_cols = get_dataset_preserve_accuracy_cols(selected)
    baselines = get_baselines_for_shot(table, shot)

    other_mean = get_other_seed_mean_accuracy_for_shot(
        table=table,
        shot=shot,
        selected_seed=seed,
    )

    preserve_colors = {
        "cifar100_accuracy": "tab:orange",
        "imagenet_accuracy": "tab:green",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ------------------------------------------------------------
    # 왼쪽 그래프: epoch별 accuracy curve
    # ------------------------------------------------------------
    ax = axes[0]

    # 다른 99개 seed 평균: 같은 색 점선, 흐리게 표시
    if not other_mean.empty:
        ax.plot(
            other_mean["epoch"],
            other_mean["target_accuracy"],
            linestyle="--",
            linewidth=2.2,
            alpha=0.28,
            color="tab:blue",
            zorder=2,
            label="other 99 seeds mean: DTD",
        )

        for col in preserve_acc_cols:
            dataset = col.replace("_accuracy", "")
            color = preserve_colors.get(col, None)

            ax.plot(
                other_mean["epoch"],
                other_mean[col],
                linestyle="--",
                linewidth=2.2,
                alpha=0.28,
                color=color,
                zorder=2,
                label=f"other 99 seeds mean: {dataset}",
            )

    # selected target
    ax.plot(
        selected["epoch"],
        selected["target_accuracy"],
        marker="o",
        linewidth=3.2,
        markersize=6,
        color="tab:blue",
        zorder=8,
        label="selected seed: DTD accuracy",
    )

    # selected preserve datasets
    for col in preserve_acc_cols:
        dataset = col.replace("_accuracy", "")
        color = preserve_colors.get(col, None)

        ax.plot(
            selected["epoch"],
            selected[col],
            marker="o",
            linewidth=3.0,
            markersize=6,
            alpha=1.0,
            color=color,
            zorder=9,
            label=f"selected seed: {dataset} accuracy",
        )

    # zero-shot baseline은 dataset별로 얇게 표시
    if "target_accuracy" in baselines:
        ax.axhline(
            baselines["target_accuracy"],
            linestyle="-.",
            linewidth=1.4,
            color="tab:blue",
            alpha=0.30,
            zorder=1,
            label="DTD zero-shot baseline",
        )

    for col in preserve_acc_cols:
        dataset = col.replace("_accuracy", "")
        color = preserve_colors.get(col, None)

        if col in baselines:
            ax.axhline(
                baselines[col],
                linestyle=":",
                linewidth=1.8,
                color=color,
                alpha=0.40,
                zorder=1,
                label=f"{dataset} zero-shot baseline",
            )

    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(
        f"Fine-tuning curve\n"
        f"{shot}-shot, selected seed={seed}"
    )
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------
    # 오른쪽 그래프: target accuracy vs preserve accuracy
    # ------------------------------------------------------------
    ax = axes[1]

    for col in preserve_acc_cols:
        dataset = col.replace("_accuracy", "")
        color = preserve_colors.get(col, None)

        ax.plot(
            selected["target_accuracy"],
            selected[col],
            marker="o",
            linewidth=3.2,
            markersize=6.5,
            color=color,
            alpha=1.0,
            zorder=8,
            label=f"selected: DTD vs {dataset}",
        )

        # epoch 번호 표시
        for _, r in selected.iterrows():
            epoch = int(r["epoch"])
            if epoch in [0, 5, 10, 20, 30, 40, 50]:
                ax.annotate(
                    str(epoch),
                    (r["target_accuracy"], r[col]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    color=color,
                    zorder=10,
                )

        # zero-shot baseline point
        if "target_accuracy" in baselines and col in baselines:
            ax.scatter(
                [baselines["target_accuracy"]],
                [baselines[col]],
                marker="*",
                s=160,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=12,
                label=f"{dataset} zero-shot point",
            )

    ax.set_xlabel("target DTD accuracy")
    ax.set_ylabel("preserve accuracy")
    ax.set_title(
        "Target-preserve trade-off\n"
        "right-up is better"
    )
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Best preservation sample seed case: {shot}-shot / seed={seed}\n"
        f"final target acc={selected_row['target_accuracy_at_selection']:.4f}, "
        f"final preserve mean acc={selected_row['preserve_mean_accuracy_at_selection']:.4f}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    save_path = out_dir / f"best_preserve_accuracy_case_shot{shot}_seed{seed}.png"
    plt.savefig(save_path, dpi=220)
    plt.close()

    print(f"saved: {save_path}")


def plot_all_selected_summary(
    table: pd.DataFrame,
    selected_df: pd.DataFrame,
    out_dir: Path,
):
    """
    선택된 seed들의 target-preserve accuracy trajectory를 한 그림에 표시.
    CIFAR100 / ImageNet을 따로 그림.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    preserve_acc_cols = get_dataset_preserve_accuracy_cols(table)

    fig, axes = plt.subplots(
        1,
        len(preserve_acc_cols),
        figsize=(7 * len(preserve_acc_cols), 6),
    )

    if len(preserve_acc_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, preserve_acc_cols):
        dataset = col.replace("_accuracy", "")

        for _, row in selected_df.iterrows():
            shot = int(row["shot"])
            seed = int(row["selected_sample_seed"])

            sub = table[
                (table["shot"] == shot)
                & (table["sample_seed"] == seed)
            ].sort_values("epoch").copy()

            sub = keep_plot_epochs(sub)

            ax.plot(
                sub["target_accuracy"],
                sub[col],
                marker="o",
                linewidth=2.5,
                markersize=5.5,
                label=f"{shot}-shot seed {seed}",
                zorder=6,
            )

        ax.set_xlabel("target DTD accuracy")
        ax.set_ylabel(f"{dataset} accuracy")
        ax.set_title(f"Target vs {dataset} accuracy")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    plt.tight_layout()

    save_path = out_dir / "selected_best_preserve_accuracy_tradeoff_all_shots.png"
    plt.savefig(save_path, dpi=220)
    plt.close()

    print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        default="runs/blue_texture_preserve_100seeds",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument(
        "--selection-epoch",
        type=int,
        default=50,
        help="어느 epoch 기준으로 preserve가 제일 좋은 seed를 고를지",
    )
    parser.add_argument(
        "--min-target-delta",
        type=float,
        default=0.0,
        help="target delta가 최소 이 값 이상인 seed만 후보로 사용. 0이면 target 성능이 zero-shot 이상인 seed만 선택.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = root_dir / "best_preserve_seed_case_plots_accuracy"

    df = load_all_curves(
        root_dir=root_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )

    table = make_seed_epoch_table(df)

    selected_df = select_best_seed_by_shot(
        table=table,
        selection_epoch=args.selection_epoch,
        min_target_delta=args.min_target_delta,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / "seed_epoch_accuracy_tradeoff_table.csv"
    selected_path = out_dir / "selected_best_preserve_seed_by_shot.csv"

    table.to_csv(table_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    print(f"saved: {table_path}")
    print(f"saved: {selected_path}")

    print("\n===== selected best preserve seed by shot =====")
    print(selected_df.to_string(index=False))

    for _, selected_row in selected_df.iterrows():
        plot_one_shot_case(
            table=table,
            selected_row=selected_row,
            out_dir=out_dir / "per_shot_cases",
        )

    plot_all_selected_summary(
        table=table,
        selected_df=selected_df,
        out_dir=out_dir,
    )

    print("\n===== 완료 =====")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()