# -*- coding: utf-8 -*-
"""
curves.jsonl을 읽어서 그래프를 저장함.

epoch 0은 original CLIP zero-shot이 아니라 initialized CoOp prompt 결과임.
따라서 그래프에는 zero_shot_baselines.json에서 읽은 값을
별도의 dashed line으로 표기함.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_grouped_lines(grouped, key, title, ylabel, out_path,
                       baseline=None, baseline_label=None):
    """
    epoch별 line plot.
    baseline이 주어지면 dashed horizontal line으로 추가함.
    accuracy plot에서는 original CLIP zero-shot accuracy,
    delta plot에서는 0.0을 baseline으로 넣어 사용.
    """
    plt.figure(figsize=(8, 5))
    for shot, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda x: x["epoch"])
        xs = [r["epoch"] for r in rows]
        ys = [r[key] for r in rows]
        plt.plot(xs, ys, marker="o", label=f"{shot}-shot")

    if baseline is not None:
        plt.axhline(
            y=baseline,
            linestyle="--",
            linewidth=2,
            color="black",
            alpha=0.7,
            label=baseline_label or "original CLIP zero-shot",
        )

    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=str, required=True)
    parser.add_argument("--baseline-json", type=str, default=None,
                        help="zero_shot_baselines.json 경로. 주어지면 그래프에 baseline 가로선이 추가됨.")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.curves)

    baselines = None
    if args.baseline_json is not None and Path(args.baseline_json).exists():
        baselines = load_json(args.baseline_json)

    target_acc = defaultdict(list)
    target_delta = defaultdict(list)
    preserve_names = set()

    for row in rows:
        shot = row["shot"]
        target_acc[shot].append({"epoch": row["epoch"], "value": row["target"]["accuracy"]})
        target_delta[shot].append({"epoch": row["epoch"], "value": row["target"]["delta_from_zero_shot"]})
        for name in row["preserve"].keys():
            preserve_names.add(name)

    target_baseline = baselines["target"]["accuracy"] if baselines else None

    plot_grouped_lines(
        target_acc, "value",
        title="Target accuracy under CoOp prompt (per epoch)",
        ylabel="accuracy",
        out_path=out_dir / "target_accuracy_by_epoch.png",
        baseline=target_baseline,
        baseline_label=f"original CLIP zero-shot ({target_baseline*100:.2f}%)" if target_baseline is not None else None,
    )
    plot_grouped_lines(
        target_delta, "value",
        title="Target delta from original CLIP zero-shot (per epoch)",
        ylabel="delta from zero-shot",
        out_path=out_dir / "target_delta_by_epoch.png",
        baseline=0.0,
        baseline_label="zero-shot baseline (delta=0)",
    )

    for preserve_name in sorted(preserve_names):
        grouped_acc = defaultdict(list)
        grouped_delta = defaultdict(list)
        for row in rows:
            shot = row["shot"]
            preserve = row["preserve"][preserve_name]
            grouped_acc[shot].append({"epoch": row["epoch"], "value": preserve["accuracy"]})
            grouped_delta[shot].append({"epoch": row["epoch"], "value": preserve["delta_from_zero_shot"]})

        preserve_baseline = (
            baselines["preserve"][preserve_name]["accuracy"]
            if baselines and preserve_name in baselines.get("preserve", {})
            else None
        )

        plot_grouped_lines(
            grouped_acc, "value",
            title=f"Preserve accuracy under target-trained CoOp prompt: {preserve_name}",
            ylabel="accuracy",
            out_path=out_dir / f"preserve_accuracy_{preserve_name}.png",
            baseline=preserve_baseline,
            baseline_label=f"original CLIP zero-shot ({preserve_baseline*100:.2f}%)" if preserve_baseline is not None else None,
        )
        plot_grouped_lines(
            grouped_delta, "value",
            title=f"Preserve delta from original CLIP zero-shot: {preserve_name}",
            ylabel="delta from zero-shot",
            out_path=out_dir / f"preserve_delta_{preserve_name}.png",
            baseline=0.0,
            baseline_label="zero-shot baseline (delta=0)",
        )

    print(f"그래프 저장 완료: {out_dir}")
    if baselines is None:
        print("[참고] --baseline-json을 주지 않아 baseline 가로선은 표시되지 않음.")


if __name__ == "__main__":
    main()
