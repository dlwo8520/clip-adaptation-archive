# -*- coding: utf-8 -*-

"""
100 sample seed 결과에서 sample_seed별 target delta / preserve delta / trade-off score 계산.

입력 구조:
runs/blue_texture_preserve_100seeds/
  sample_seed_0/curves.jsonl
  sample_seed_1/curves.jsonl
  ...
  sample_seed_99/curves.jsonl

출력:
runs/blue_texture_preserve_100seeds/tradeoff_analysis/
  seed_tradeoff_scores_epoch50_lambda1.0.csv
  best_worst_by_shot_epoch50_lambda1.0.csv
  topk_by_shot_epoch50_lambda1.0.txt
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def safe_float(x):
    if x is None:
        return None
    return float(x)


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
                        "accuracy": safe_float(r["target"]["accuracy"]),
                        "delta": safe_float(r["target"]["delta_from_zero_shot"]),
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
                            "accuracy": safe_float(metric["accuracy"]),
                            "delta": safe_float(metric["delta_from_zero_shot"]),
                        }
                    )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("읽은 curves 결과가 비어 있음.")

    # 같은 seed/shot/epoch/split/dataset이 중복되면 마지막 값 사용
    key_cols = ["sample_seed", "shot", "epoch", "split", "dataset"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"[경고] 중복 row 제거됨: {before - after}개")

    return df


def choose_epoch(df: pd.DataFrame, epoch_arg: str) -> int:
    epochs = sorted(df["epoch"].unique())

    if epoch_arg == "final":
        return int(max(epochs))

    epoch = int(epoch_arg)

    if epoch not in epochs:
        raise ValueError(f"요청한 epoch={epoch}가 결과에 없음. 사용 가능 epoch: {epochs}")

    return epoch


def make_tradeoff_table(df: pd.DataFrame, epoch: int, lambda_preserve: float) -> pd.DataFrame:
    final_df = df[df["epoch"] == epoch].copy()

    target_df = final_df[final_df["split"] == "target"].copy()
    preserve_df = final_df[final_df["split"] == "preserve"].copy()

    target_wide = target_df[
        ["sample_seed", "shot", "accuracy", "delta"]
    ].rename(
        columns={
            "accuracy": "target_accuracy",
            "delta": "target_delta",
        }
    )

    preserve_wide = preserve_df.pivot_table(
        index=["sample_seed", "shot"],
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
        on=["sample_seed", "shot"],
        how="inner",
    )

    preserve_delta_cols = [
        c for c in merged.columns
        if c.endswith("_delta") and not c.startswith("target_")
    ]

    preserve_acc_cols = [
        c for c in merged.columns
        if c.endswith("_accuracy") and not c.startswith("target_")
    ]

    if len(preserve_delta_cols) == 0:
        raise RuntimeError("preserve delta column을 찾지 못함.")

    merged["preserve_mean_delta"] = merged[preserve_delta_cols].mean(axis=1)
    merged["preserve_min_delta"] = merged[preserve_delta_cols].min(axis=1)
    merged["preserve_max_delta"] = merged[preserve_delta_cols].max(axis=1)

    if len(preserve_acc_cols) > 0:
        merged["preserve_mean_accuracy"] = merged[preserve_acc_cols].mean(axis=1)

    # preserve_mean_delta는 보통 음수임.
    # 점수는 높을수록 좋음.
    # target_delta가 클수록 좋고, preserve_mean_delta가 0에 가까울수록 좋음.
    merged["tradeoff_score"] = (
        merged["target_delta"] + lambda_preserve * merged["preserve_mean_delta"]
    )

    # target은 좋아졌지만 preserve가 크게 망가지는 seed를 보기 위한 penalty
    merged["preserve_drop_abs"] = -merged["preserve_mean_delta"]

    # 보기 좋게 정렬
    preferred_cols = [
        "sample_seed",
        "shot",
        "target_accuracy",
        "target_delta",
    ]

    for c in sorted(preserve_delta_cols):
        preferred_cols.append(c)

    preferred_cols += [
        "preserve_mean_delta",
        "preserve_min_delta",
        "preserve_max_delta",
        "tradeoff_score",
        "preserve_drop_abs",
    ]

    other_cols = [c for c in merged.columns if c not in preferred_cols]
    merged = merged[preferred_cols + other_cols]

    merged = merged.sort_values(["shot", "tradeoff_score"], ascending=[True, False]).reset_index(drop=True)

    return merged


def make_best_worst_table(score_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for shot, g in score_df.groupby("shot"):
        g = g.copy()

        best_tradeoff = g.sort_values("tradeoff_score", ascending=False).iloc[0]
        worst_tradeoff = g.sort_values("tradeoff_score", ascending=True).iloc[0]

        best_target = g.sort_values("target_delta", ascending=False).iloc[0]
        worst_target = g.sort_values("target_delta", ascending=True).iloc[0]

        best_preserve = g.sort_values("preserve_mean_delta", ascending=False).iloc[0]
        worst_preserve = g.sort_values("preserve_mean_delta", ascending=True).iloc[0]

        cases = [
            ("best_tradeoff", best_tradeoff),
            ("worst_tradeoff", worst_tradeoff),
            ("best_target", best_target),
            ("worst_target", worst_target),
            ("best_preserve", best_preserve),
            ("worst_preserve", worst_preserve),
        ]

        for case_name, row in cases:
            rows.append(
                {
                    "shot": int(shot),
                    "case": case_name,
                    "sample_seed": int(row["sample_seed"]),
                    "target_delta": float(row["target_delta"]),
                    "preserve_mean_delta": float(row["preserve_mean_delta"]),
                    "tradeoff_score": float(row["tradeoff_score"]),
                    "preserve_drop_abs": float(row["preserve_drop_abs"]),
                }
            )

    return pd.DataFrame(rows)


def write_topk_report(score_df: pd.DataFrame, out_path: Path, topk: int):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("===== sample seed trade-off ranking =====\n\n")
        f.write("score = target_delta + lambda_preserve * preserve_mean_delta\n")
        f.write("preserve_mean_delta는 보통 음수라서, preserve drop이 클수록 score가 낮아짐.\n")
        f.write("즉 score가 높을수록 target 개선과 zero-shot 보존 균형이 좋음.\n\n")

        for shot, g in score_df.groupby("shot"):
            g = g.sort_values("tradeoff_score", ascending=False)

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"{shot}-shot\n")
            f.write("=" * 80 + "\n\n")

            cols = [
                "sample_seed",
                "target_delta",
                "preserve_mean_delta",
                "tradeoff_score",
                "preserve_drop_abs",
            ]

            preserve_cols = [
                c for c in g.columns
                if c.endswith("_delta") and not c.startswith("target_")
            ]
            cols = cols[:2] + sorted(preserve_cols) + cols[2:]

            f.write(f"[BEST trade-off top {topk}]\n")
            f.write(g.head(topk)[cols].to_string(index=False))
            f.write("\n\n")

            f.write(f"[WORST trade-off bottom {topk}]\n")
            f.write(g.tail(topk).sort_values("tradeoff_score", ascending=True)[cols].to_string(index=False))
            f.write("\n\n")

            f.write(f"[BEST target top {topk}]\n")
            f.write(g.sort_values("target_delta", ascending=False).head(topk)[cols].to_string(index=False))
            f.write("\n\n")

            f.write(f"[BEST preserve top {topk}]\n")
            f.write(g.sort_values("preserve_mean_delta", ascending=False).head(topk)[cols].to_string(index=False))
            f.write("\n\n")


def print_console_summary(best_worst_df: pd.DataFrame):
    print("\n===== best / worst summary =====")
    print(best_worst_df.to_string(index=False))


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
        "--epoch",
        type=str,
        default="final",
        help="'final' 또는 특정 epoch 번호. 예: final, 10, 50",
    )
    parser.add_argument(
        "--lambda-preserve",
        type=float,
        default=1.0,
        help="preserve delta에 주는 가중치. 클수록 zero-shot 보존을 더 중요하게 봄.",
    )
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = root_dir / "tradeoff_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_curves(
        root_dir=root_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )

    epoch = choose_epoch(df, args.epoch)

    print("===== 설정 =====")
    print(f"root_dir: {root_dir}")
    print(f"epoch: {epoch}")
    print(f"lambda_preserve: {args.lambda_preserve}")
    print(f"num seeds: {df['sample_seed'].nunique()}")
    print(f"shots: {sorted(df['shot'].unique())}")
    print(f"datasets:")
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))

    score_df = make_tradeoff_table(
        df=df,
        epoch=epoch,
        lambda_preserve=args.lambda_preserve,
    )

    best_worst_df = make_best_worst_table(score_df)

    lambda_str = str(args.lambda_preserve).replace(".", "p")
    score_csv = out_dir / f"seed_tradeoff_scores_epoch{epoch}_lambda{lambda_str}.csv"
    best_worst_csv = out_dir / f"best_worst_by_shot_epoch{epoch}_lambda{lambda_str}.csv"
    report_txt = out_dir / f"topk_by_shot_epoch{epoch}_lambda{lambda_str}.txt"

    score_df.to_csv(score_csv, index=False)
    best_worst_df.to_csv(best_worst_csv, index=False)
    write_topk_report(score_df, report_txt, topk=args.topk)

    print(f"\nsaved: {score_csv}")
    print(f"saved: {best_worst_csv}")
    print(f"saved: {report_txt}")

    print_console_summary(best_worst_df)


if __name__ == "__main__":
    main()
