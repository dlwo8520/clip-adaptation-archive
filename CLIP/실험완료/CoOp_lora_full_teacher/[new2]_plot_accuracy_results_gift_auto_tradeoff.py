# -*- coding: utf-8 -*-

("""
GIFT / ZSCL-CoOp 계열 100-seed 결과 자동 그래프 코드.

지원하는 결과 구조:
1) 기존 구조
   EXP_DIR/sample_seed_0/curves.jsonl

2) GIFT 구조
   EXP_DIR/sample_seed_0/shot_1/history.json
   EXP_DIR/sample_seed_0/shot_2/history.json
   EXP_DIR/sample_seed_0/shot_4/history.json
   EXP_DIR/sample_seed_0/shot_8/history.json
   EXP_DIR/sample_seed_0/shot_16/history.json

사용 예:
python plot_accuracy_results_gift_auto.py \
  --root-dir ${RESEARCH_WORKSPACE}/runs/coop_gift_BCD_3selection_seed0_49_gpu4/teacher_confident \
  --seed-start 0 --seed-end 49

root-dir에 실험명 폴더(B__synthetic_zscl_like 등)가 여러 개 있으면 자동으로 전부 처리함.
각 결과는 EXP_DIR/accuracy_plots 아래에 저장됨.
""").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[3])))

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# -------------------------
# basic utils
# -------------------------

def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))


def parse_sample_seed(path: Path) -> Optional[int]:
    m = re.search(r"sample_seed_(\d+)", str(path))
    return int(m.group(1)) if m else None


def parse_shot(path: Path) -> Optional[int]:
    m = re.search(r"shot_(\d+)", str(path))
    return int(m.group(1)) if m else None


def keep_plot_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """epoch 0, 5, 10, ..., 50만 남김."""
    if df.empty or "epoch" not in df.columns:
        return df
    return df[(df["epoch"] == 0) | (df["epoch"] % 5 == 0)].copy()


def read_json_any(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_records_from_history_json(path: Path) -> Iterable[Dict[str, Any]]:
    """
    history.json의 형태가 list / dict 모두 가능하다고 보고 최대한 유연하게 읽음.
    """
    obj = read_json_any(path)

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(obj, dict):
        return

    candidate_keys = [
        "history", "records", "curves", "logs", "results", "epochs", "metrics"
    ]
    for key in candidate_keys:
        value = obj.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
            return

    # dict 자체가 한 epoch 결과인 경우
    if any(k in obj for k in ["epoch", "target", "preserve", "target_accuracy"]):
        yield obj


# -------------------------
# metric parsing
# -------------------------

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _get_nested_accuracy(obj: Any) -> Optional[float]:
    """dict 또는 숫자에서 accuracy 값을 뽑음."""
    direct = _to_float(obj)
    if direct is not None:
        return direct

    if not isinstance(obj, dict):
        return None

    for key in ["accuracy", "acc", "top1", "top1_acc", "top1_accuracy"]:
        if key in obj:
            val = _to_float(obj[key])
            if val is not None:
                return val

    return None


def _find_accuracy_by_keywords(record: Dict[str, Any], keywords: List[str]) -> Optional[float]:
    """
    flat dict에서 특정 dataset 관련 accuracy key를 찾음.
    예: imagenet_accuracy, preserve/imagenet/acc 등.
    """
    lowered_keywords = [k.lower() for k in keywords]

    for key, value in record.items():
        key_l = str(key).lower()
        if all(k in key_l for k in lowered_keywords) and ("acc" in key_l or "accuracy" in key_l or "top1" in key_l):
            val = _to_float(value)
            if val is not None:
                return val

    # 한 keyword만 포함하고 accuracy류 단어가 포함된 경우도 허용
    for key, value in record.items():
        key_l = str(key).lower()
        if any(k in key_l for k in lowered_keywords) and ("acc" in key_l or "accuracy" in key_l or "top1" in key_l):
            val = _to_float(value)
            if val is not None:
                return val

    return None


def parse_target_name(record: Dict[str, Any], default: str = "texture") -> str:
    for key in ["target_name", "target_dataset", "dataset", "target"]:
        value = record.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for nk in ["name", "dataset", "target_name"]:
                if isinstance(value.get(nk), str):
                    return value[nk]
    return default


def parse_epoch(record: Dict[str, Any]) -> Optional[int]:
    for key in ["epoch", "step", "eval_epoch"]:
        if key in record:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                pass
    return None


def parse_target_accuracy(record: Dict[str, Any], target_name: str) -> Optional[float]:
    # nested target
    if "target" in record:
        val = _get_nested_accuracy(record["target"])
        if val is not None:
            return val

    # common flat keys
    for key in [
        "target_accuracy", "target_acc", "target_top1", "target_top1_acc",
        f"{target_name}_accuracy", f"{target_name}_acc",
        "dtd_accuracy", "dtd_acc", "texture_accuracy", "texture_acc",
    ]:
        if key in record:
            val = _to_float(record[key])
            if val is not None:
                return val

    return _find_accuracy_by_keywords(record, ["target"])


def parse_preserve_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    preserve: Dict[str, float] = {}

    # nested preserve dict: {imagenet: {accuracy: ...}, cifar100: {accuracy: ...}}
    for preserve_key in ["preserve", "preserve_metrics", "preserve_results"]:
        value = record.get(preserve_key)
        if isinstance(value, dict):
            for dataset_name, metric_obj in value.items():
                acc = _get_nested_accuracy(metric_obj)
                if acc is not None:
                    preserve[str(dataset_name)] = float(acc)

    # flat keys fallback
    known_datasets = {
        "imagenet": ["imagenet"],
        "cifar100": ["cifar100"],
        "cifar-100": ["cifar-100"],
        "cifar": ["cifar"],
    }
    for dataset_name, keywords in known_datasets.items():
        if dataset_name in preserve:
            continue
        acc = _find_accuracy_by_keywords(record, keywords)
        if acc is not None:
            # cifar와 cifar100이 동시에 잡히면 cifar100 우선
            if dataset_name == "cifar" and ("cifar100" in preserve or "cifar-100" in preserve):
                continue
            preserve[dataset_name] = float(acc)

    return preserve


# -------------------------
# loading curves
# -------------------------

def load_curves_jsonl(root_dir: Path, seed_start: int, seed_end: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for sample_seed in range(seed_start, seed_end + 1):
        curve_path = root_dir / f"sample_seed_{sample_seed}" / "curves.jsonl"
        if not curve_path.exists():
            continue

        with open(curve_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)

                seed = int(r.get("sample_seed", sample_seed))
                shot = int(r["shot"])
                epoch = int(r["epoch"])
                target_name = str(r.get("target_name", "texture"))

                rows.append(
                    {
                        "sample_seed": seed,
                        "shot": shot,
                        "epoch": epoch,
                        "split": "target",
                        "dataset": target_name,
                        "accuracy": float(r["target"]["accuracy"]),
                    }
                )

                for preserve_name, metric in r.get("preserve", {}).items():
                    rows.append(
                        {
                            "sample_seed": seed,
                            "shot": shot,
                            "epoch": epoch,
                            "split": "preserve",
                            "dataset": str(preserve_name),
                            "accuracy": float(metric["accuracy"]),
                        }
                    )

    return rows


def load_history_jsons(root_dir: Path, seed_start: int, seed_end: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    history_paths = sorted(root_dir.glob("sample_seed_*/shot_*/history.json"))

    for history_path in history_paths:
        sample_seed = parse_sample_seed(history_path)
        shot = parse_shot(history_path)

        if sample_seed is None or shot is None:
            print(f"[경고] seed/shot 파싱 실패: {history_path}")
            continue
        if not (seed_start <= sample_seed <= seed_end):
            continue

        parsed_count = 0
        for record in iter_records_from_history_json(history_path):
            epoch = parse_epoch(record)
            if epoch is None:
                continue

            target_name = parse_target_name(record, default="texture")
            target_acc = parse_target_accuracy(record, target_name)
            preserve_metrics = parse_preserve_metrics(record)

            if target_acc is not None:
                rows.append(
                    {
                        "sample_seed": sample_seed,
                        "shot": shot,
                        "epoch": epoch,
                        "split": "target",
                        "dataset": target_name,
                        "accuracy": float(target_acc),
                    }
                )
                parsed_count += 1

            for preserve_name, preserve_acc in preserve_metrics.items():
                rows.append(
                    {
                        "sample_seed": sample_seed,
                        "shot": shot,
                        "epoch": epoch,
                        "split": "preserve",
                        "dataset": str(preserve_name),
                        "accuracy": float(preserve_acc),
                    }
                )
                parsed_count += 1

        if parsed_count == 0:
            print(f"[경고] metric 파싱 실패: {history_path}")

    return rows


def load_curves(root_dir: Path, seed_start: int, seed_end: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.extend(load_curves_jsonl(root_dir, seed_start, seed_end))
    rows.extend(load_history_jsons(root_dir, seed_start, seed_end))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"결과를 하나도 읽지 못함: {root_dir}\n"
            "확인할 구조: sample_seed_*/curves.jsonl 또는 sample_seed_*/shot_*/history.json"
        )

    key_cols = ["sample_seed", "shot", "epoch", "split", "dataset"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[경고] 중복 row 제거됨: {before - after}개")

    df = keep_plot_epochs(df)
    return df


# -------------------------
# experiment discovery
# -------------------------

def has_experiment_outputs(path: Path) -> bool:
    return any(path.glob("sample_seed_*/curves.jsonl")) or any(path.glob("sample_seed_*/shot_*/history.json"))


def discover_experiment_dirs(root_dir: Path) -> List[Path]:
    """
    root-dir가 실험 폴더면 [root-dir].
    root-dir가 teacher_confident 같은 상위 폴더면 하위 실험 폴더 자동 탐색.
    """
    root_dir = root_dir.resolve()

    if has_experiment_outputs(root_dir):
        return [root_dir]

    exp_dirs = set()

    for p in root_dir.glob("**/curves.jsonl"):
        # EXP_DIR/sample_seed_x/curves.jsonl
        if p.parent.name.startswith("sample_seed_"):
            exp_dirs.add(p.parent.parent.resolve())

    for p in root_dir.glob("**/history.json"):
        # EXP_DIR/sample_seed_x/shot_y/history.json
        if p.parent.name.startswith("shot_") and p.parent.parent.name.startswith("sample_seed_"):
            exp_dirs.add(p.parent.parent.parent.resolve())

    return sorted(exp_dirs, key=lambda x: str(x))


# -------------------------
# summarization
# -------------------------

def summarize_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    return (
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


def make_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    target = df[df["split"] == "target"].copy()
    preserve = df[df["split"] == "preserve"].copy()

    target_wide = target[["sample_seed", "shot", "epoch", "accuracy"]].rename(
        columns={"accuracy": "target_accuracy"}
    )

    preserve_wide = preserve.pivot_table(
        index=["sample_seed", "shot", "epoch"],
        columns="dataset",
        values="accuracy",
        aggfunc="last",
    ).reset_index()

    preserve_wide.columns = [str(c) for c in preserve_wide.columns]

    merged = target_wide.merge(
        preserve_wide,
        on=["sample_seed", "shot", "epoch"],
        how="inner",
    )

    return merged


def find_dataset_col(table: pd.DataFrame, keyword: str) -> str:
    keyword = keyword.lower()
    candidates = [c for c in table.columns if keyword in c.lower()]
    if not candidates:
        raise RuntimeError(f"{keyword} dataset column을 찾지 못함. columns={list(table.columns)}")
    return candidates[0]


def get_baseline_from_summary(data: pd.DataFrame) -> Optional[float]:
    epoch0 = data[data["epoch"] == 0]
    if epoch0.empty:
        return None
    return float(epoch0["mean_accuracy"].iloc[0])


def apply_accuracy_ylim(ylim: Optional[List[float]]) -> None:
    if ylim is None:
        return
    plt.ylim(float(ylim[0]), float(ylim[1]))


def resolve_selection_epoch(table: pd.DataFrame, shot: int, selection_epoch: int) -> int:
    sub = table[table["shot"] == shot]
    if sub.empty:
        raise RuntimeError(f"{shot}-shot 결과가 없음.")

    available = sorted(int(e) for e in sub["epoch"].unique())
    if selection_epoch in available:
        return selection_epoch

    fallback = max(available)
    print(f"[경고] epoch={selection_epoch} 없음. 대신 마지막 epoch={fallback} 사용.")
    return fallback


# -------------------------
# plotting
# -------------------------

def plot_mean_iqr_accuracy(summary: pd.DataFrame, out_dir: Path, title_prefix: str, ylim=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    combos = summary[["split", "dataset"]].drop_duplicates()

    for _, row in combos.iterrows():
        split = row["split"]
        dataset = row["dataset"]
        data = summary[(summary["split"] == split) & (summary["dataset"] == dataset)].copy()
        if data.empty:
            continue

        plt.figure(figsize=(12, 7))

        for shot in sorted(data["shot"].unique()):
            s = data[data["shot"] == shot].sort_values("epoch")
            x = s["epoch"].to_numpy()
            mean = s["mean_accuracy"].to_numpy()
            q25 = s["q25_accuracy"].to_numpy()
            q75 = s["q75_accuracy"].to_numpy()

            plt.plot(x, mean, marker="o", linewidth=2.4, label=f"{shot}-shot mean")
            plt.fill_between(x, q25, q75, alpha=0.15)

        baseline = get_baseline_from_summary(data)
        if baseline is not None:
            plt.axhline(baseline, linestyle="--", linewidth=1.6, label="zero-shot baseline")

        plt.xticks(sorted(data["epoch"].unique()))
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"{title_prefix}\n{split} / {dataset}: mean accuracy with IQR")
        plt.grid(True, alpha=0.25)
        plt.legend()
        apply_accuracy_ylim(ylim)
        plt.tight_layout()

        save_path = out_dir / f"{safe_name(split)}_{safe_name(dataset)}_mean_accuracy_iqr.png"
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"saved: {save_path}")


def plot_spaghetti_accuracy(df: pd.DataFrame, out_dir: Path, title_prefix: str, only_shot: int = 16, ylim=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    combos = df[["split", "dataset"]].drop_duplicates()

    for _, row in combos.iterrows():
        split = row["split"]
        dataset = row["dataset"]

        sub = df[(df["split"] == split) & (df["dataset"] == dataset) & (df["shot"] == only_shot)].copy()
        if sub.empty:
            continue

        sub = sub.sort_values(["sample_seed", "epoch"])
        plt.figure(figsize=(11, 7))

        for _, g in sub.groupby("sample_seed"):
            g = g.sort_values("epoch")
            plt.plot(g["epoch"], g["accuracy"], linewidth=0.8, alpha=0.13)

        stat = (
            sub.groupby("epoch", as_index=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                median_accuracy=("accuracy", "median"),
                n_seeds=("sample_seed", "nunique"),
            )
            .sort_values("epoch")
        )

        plt.plot(stat["epoch"], stat["mean_accuracy"], marker="o", linewidth=2.8, label="mean over seeds")
        plt.plot(stat["epoch"], stat["median_accuracy"], marker="s", linestyle="--", linewidth=2.4, label="median over seeds")

        epoch0 = stat[stat["epoch"] == 0]
        if not epoch0.empty:
            baseline = float(epoch0["mean_accuracy"].iloc[0])
            plt.axhline(baseline, linestyle=":", linewidth=2.0, label="zero-shot baseline")

        plt.xticks(sorted(sub["epoch"].unique()))
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(
            f"{title_prefix}\n{split} / {dataset} / {only_shot}-shot: "
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


def get_other_seed_mean(table: pd.DataFrame, shot: int, selected_seed: int) -> pd.DataFrame:
    sub = table[(table["shot"] == shot) & (table["sample_seed"] != selected_seed)].copy()
    if sub.empty:
        return pd.DataFrame()

    imagenet_col = find_dataset_col(sub, "imagenet")
    cifar100_col = find_dataset_col(sub, "cifar")

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


def get_selected_curve(table: pd.DataFrame, shot: int, seed: int) -> pd.DataFrame:
    sub = table[(table["shot"] == shot) & (table["sample_seed"] == seed)].copy()
    if sub.empty:
        return sub

    imagenet_col = find_dataset_col(sub, "imagenet")
    cifar100_col = find_dataset_col(sub, "cifar")

    sub = sub.rename(columns={imagenet_col: "imagenet_accuracy", cifar100_col: "cifar100_accuracy"})
    sub = sub.sort_values("epoch")
    return keep_plot_epochs(sub)


def get_baselines(table: pd.DataFrame, shot: int) -> Dict[str, float]:
    sub = table[(table["shot"] == shot) & (table["epoch"] == 0)].copy()
    if sub.empty:
        return {}

    row = sub.iloc[0]
    imagenet_col = find_dataset_col(sub, "imagenet")
    cifar100_col = find_dataset_col(sub, "cifar")

    return {
        "target": float(row["target_accuracy"]),
        "imagenet": float(row[imagenet_col]),
        "cifar100": float(row[cifar100_col]),
    }


def select_best_cases(table: pd.DataFrame, shot: int, selection_epoch: int) -> Dict[str, Dict[str, Any]]:
    sub = table[(table["shot"] == shot) & (table["epoch"] == selection_epoch)].copy()
    if sub.empty:
        raise RuntimeError(f"{shot}-shot, epoch={selection_epoch} 결과가 없음.")

    imagenet_col = find_dataset_col(sub, "imagenet")
    cifar100_col = find_dataset_col(sub, "cifar")
    sub["preserve_avg_accuracy"] = sub[[imagenet_col, cifar100_col]].mean(axis=1)

    best_target = sub.sort_values(["target_accuracy", "preserve_avg_accuracy"], ascending=[False, False]).iloc[0]
    best_imagenet = sub.sort_values([imagenet_col, "target_accuracy"], ascending=[False, False]).iloc[0]
    best_cifar100 = sub.sort_values([cifar100_col, "target_accuracy"], ascending=[False, False]).iloc[0]
    best_preserve_avg = sub.sort_values(["preserve_avg_accuracy", "target_accuracy"], ascending=[False, False]).iloc[0]

    return {
        "best_target": {"title": "Best Target Accuracy", "seed": int(best_target["sample_seed"])},
        "best_imagenet": {"title": "Best ImageNet Preserve Accuracy", "seed": int(best_imagenet["sample_seed"])},
        "best_cifar100": {"title": "Best CIFAR Preserve Accuracy", "seed": int(best_cifar100["sample_seed"])},
        "best_preserve_avg": {"title": "Best Average Preserve Accuracy", "seed": int(best_preserve_avg["sample_seed"])},
    }


def make_title_metric_line(selected: pd.DataFrame, selection_epoch: int) -> str:
    score_row = selected[selected["epoch"] == selection_epoch]
    if score_row.empty:
        score_row = selected.iloc[[-1]]
    score_row = score_row.iloc[0]

    return (
        f"Target={score_row['target_accuracy'] * 100:.2f}% / "
        f"ImageNet={score_row['imagenet_accuracy'] * 100:.2f}% / "
        f"CIFAR={score_row['cifar100_accuracy'] * 100:.2f}%"
    )


def plot_best_case(
    table: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    shot: int,
    case_key: str,
    case_info: Dict[str, Any],
    selection_epoch: int,
    ylim=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = case_info["seed"]

    selected = get_selected_curve(table=table, shot=shot, seed=seed)
    if selected.empty:
        print(f"[건너뜀] selected curve 없음: case={case_key}, seed={seed}")
        return

    other_mean = get_other_seed_mean(table=table, shot=shot, selected_seed=seed)
    baselines = get_baselines(table, shot)
    title_metric_line = make_title_metric_line(selected=selected, selection_epoch=selection_epoch)

    plt.figure(figsize=(11, 7))

    if not other_mean.empty:
        plt.plot(other_mean["epoch"], other_mean["target_accuracy"], linestyle="--", linewidth=2.2, alpha=0.30, color="tab:blue", label="other seeds mean: target")
        plt.plot(other_mean["epoch"], other_mean["cifar100_accuracy"], linestyle="--", linewidth=2.2, alpha=0.30, color="tab:orange", label="other seeds mean: CIFAR")
        plt.plot(other_mean["epoch"], other_mean["imagenet_accuracy"], linestyle="--", linewidth=2.2, alpha=0.30, color="tab:green", label="other seeds mean: ImageNet")

    plt.plot(selected["epoch"], selected["target_accuracy"], marker="o", linewidth=3.0, color="tab:blue", label="selected seed: target accuracy")
    plt.plot(selected["epoch"], selected["cifar100_accuracy"], marker="o", linewidth=3.0, color="tab:orange", label="selected seed: CIFAR accuracy")
    plt.plot(selected["epoch"], selected["imagenet_accuracy"], marker="o", linewidth=3.0, color="tab:green", label="selected seed: ImageNet accuracy")

    if baselines:
        plt.axhline(baselines["target"], linestyle="-.", linewidth=1.5, color="tab:blue", alpha=0.35, label="target zero-shot baseline")
        plt.axhline(baselines["cifar100"], linestyle=":", linewidth=1.8, color="tab:orange", alpha=0.45, label="CIFAR zero-shot baseline")
        plt.axhline(baselines["imagenet"], linestyle=":", linewidth=1.8, color="tab:green", alpha=0.45, label="ImageNet zero-shot baseline")

    plt.xticks(sorted(selected["epoch"].unique()))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(
        f"{title_prefix}\n{case_info['title']}\n"
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


def plot_best_cases(table: pd.DataFrame, out_dir: Path, title_prefix: str, shot: int = 16, selection_epoch: int = 50, ylim=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_epoch = resolve_selection_epoch(table, shot, selection_epoch)

    cases = select_best_cases(table=table, shot=shot, selection_epoch=selection_epoch)

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
                "preserve_avg_accuracy": float((score_row["imagenet_accuracy"] + score_row["cifar100_accuracy"]) / 2),
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
            title_prefix=title_prefix,
            shot=shot,
            case_key=case_key,
            case_info=case_info,
            selection_epoch=selection_epoch,
            ylim=ylim,
        )


def select_top_bottom_seeds_by_target_accuracy(
    table: pd.DataFrame,
    shot: int,
    selection_epoch: int,
    group_size: int = 30,
) -> Tuple[pd.DataFrame, int, Dict[str, str]]:
    selection_epoch = resolve_selection_epoch(table, shot, selection_epoch)
    sub = table[(table["shot"] == shot) & (table["epoch"] == selection_epoch)].copy()
    if sub.empty:
        raise RuntimeError(f"{shot}-shot, epoch={selection_epoch} result not found.")

    cifar100_col = find_dataset_col(sub, "cifar")
    imagenet_col = find_dataset_col(sub, "imagenet")

    ranking = sub[["sample_seed", "shot", "target_accuracy", cifar100_col, imagenet_col]].copy()
    ranking = ranking.rename(
        columns={
            "target_accuracy": "dtd_target_accuracy",
            cifar100_col: "cifar100_accuracy",
            imagenet_col: "imagenet_accuracy",
        }
    )
    ranking["selection_epoch"] = int(selection_epoch)
    ranking = ranking[
        [
            "sample_seed",
            "shot",
            "selection_epoch",
            "dtd_target_accuracy",
            "cifar100_accuracy",
            "imagenet_accuracy",
        ]
    ].drop_duplicates(subset=["sample_seed"], keep="last")

    ranking = ranking.sort_values(["dtd_target_accuracy", "sample_seed"], ascending=[False, True]).reset_index(drop=True)
    n_seeds = len(ranking)
    if n_seeds < 2:
        raise RuntimeError("Need at least two seeds for top/bottom target-preserve trade-off grouping.")

    actual_group_size = min(int(group_size), n_seeds // 2)
    if actual_group_size < group_size:
        print(
            f"[warning] only {n_seeds} seeds available; using top/bottom "
            f"{actual_group_size} instead of {group_size} to avoid overlap."
        )
    if actual_group_size <= 0:
        raise RuntimeError(f"Not enough seeds for grouping: n_seeds={n_seeds}")

    top_seeds = set(int(s) for s in ranking.head(actual_group_size)["sample_seed"])
    bottom_seeds = set(int(s) for s in ranking.tail(actual_group_size)["sample_seed"])
    overlap = top_seeds & bottom_seeds
    if overlap:
        raise RuntimeError(f"Top and bottom seed groups overlap: {sorted(overlap)}")

    def assign_group(seed: int) -> str:
        seed = int(seed)
        if seed in top_seeds:
            return "top30"
        if seed in bottom_seeds:
            return "bottom30"
        return "middle"

    ranking["group"] = ranking["sample_seed"].map(assign_group)

    top_sorted = ranking[ranking["group"] == "top30"].sort_values("dtd_target_accuracy", ascending=False)
    bottom_sorted = ranking[ranking["group"] == "bottom30"].sort_values("dtd_target_accuracy", ascending=True)

    print("\n===== DTD target seed grouping =====")
    print(f"shot={shot}, selection_epoch={selection_epoch}, ranking metric=DTD target accuracy")
    print(f"DTD Top{actual_group_size} seeds   : {top_sorted['sample_seed'].astype(int).tolist()}")
    print(f"DTD Bottom{actual_group_size} seeds: {bottom_sorted['sample_seed'].astype(int).tolist()}")

    metric_cols = {
        "DTD": "target_accuracy",
        "CIFAR100": cifar100_col,
        "ImageNet": imagenet_col,
    }
    return ranking, selection_epoch, metric_cols


def summarize_tradeoff_groups(
    table: pd.DataFrame,
    ranking: pd.DataFrame,
    shot: int,
    metric_cols: Dict[str, str],
) -> pd.DataFrame:
    group_map = ranking.set_index("sample_seed")["group"].to_dict()
    selected_seeds = set(ranking[ranking["group"].isin(["top30", "bottom30"])]["sample_seed"].astype(int))

    sub = table[(table["shot"] == shot) & (table["sample_seed"].isin(selected_seeds))].copy()
    if sub.empty:
        raise RuntimeError(f"No rows found for target-preserve trade-off groups at shot={shot}.")

    sub["group"] = sub["sample_seed"].map(group_map)
    rows: List[Dict[str, Any]] = []
    for metric_name, col in metric_cols.items():
        if col not in sub.columns:
            raise RuntimeError(f"Required metric column not found: {col}")
        metric_df = sub[["group", "sample_seed", "epoch", col]].rename(columns={col: "accuracy"}).copy()
        metric_df["metric"] = metric_name
        rows.append(metric_df)

    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df[long_df["group"].isin(["top30", "bottom30"])].copy()

    summary = (
        long_df.groupby(["group", "metric", "epoch"], as_index=False)
        .agg(
            median_accuracy=("accuracy", "median"),
            q25_accuracy=("accuracy", lambda x: x.quantile(0.25)),
            q75_accuracy=("accuracy", lambda x: x.quantile(0.75)),
            n_seeds=("sample_seed", "nunique"),
        )
        .sort_values(["metric", "group", "epoch"])
        .reset_index(drop=True)
    )
    return summary


def plot_target_preserve_tradeoff_1x3(
    summary: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    shot: int,
    selection_epoch: int,
    ylim=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["DTD", "CIFAR100", "ImageNet"]
    group_styles = {
        "top30": {
            "label": "DTD Top30 seeds",
            "color": "tab:red",
        },
        "bottom30": {
            "label": "DTD Bottom30 seeds",
            "color": "tab:blue",
        },
    }

    epochs = sorted(int(e) for e in summary["epoch"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharex=True, sharey=True)

    for ax, metric in zip(axes, metrics):
        metric_df = summary[summary["metric"] == metric].copy()
        if metric_df.empty:
            ax.set_title(metric)
            ax.grid(True, alpha=0.25)
            continue

        for group in ["top30", "bottom30"]:
            g = metric_df[metric_df["group"] == group].sort_values("epoch")
            if g.empty:
                continue
            style = group_styles[group]
            x = g["epoch"].to_numpy()
            median = g["median_accuracy"].to_numpy()
            q25 = g["q25_accuracy"].to_numpy()
            q75 = g["q75_accuracy"].to_numpy()
            label = style["label"]
            n_unique = sorted(g["n_seeds"].dropna().unique())
            if n_unique:
                label = f"{label} (n={int(n_unique[0])})"

            ax.plot(x, median, marker="o", markersize=4.0, linewidth=1.55, color=style["color"], label=label)
            ax.fill_between(x, q25, q75, color=style["color"], alpha=0.12)

        ax.set_title(metric)
        ax.set_xticks(epochs)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        if ylim is not None:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))

    axes[0].set_ylabel("accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend(handles, labels, fontsize=9, loc="best")

    fig.suptitle(
        f"{title_prefix}\n"
        f"Target-preserve trade-off: DTD target accuracy ranking, "
        f"{shot}-shot, epoch {selection_epoch}, Top30 vs Bottom30",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])

    save_path = out_dir / f"target_preserve_tradeoff_top30_bottom30_1x3_{shot}shot_epoch{selection_epoch}.png"
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    print(f"saved: {save_path}")


def plot_target_preserve_tradeoff_groups(
    table: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    target_shot: int = 16,
    selection_epoch: int = 50,
    ylim=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking, resolved_epoch, metric_cols = select_top_bottom_seeds_by_target_accuracy(
        table=table,
        shot=target_shot,
        selection_epoch=selection_epoch,
        group_size=30,
    )
    summary = summarize_tradeoff_groups(
        table=table,
        ranking=ranking,
        shot=target_shot,
        metric_cols=metric_cols,
    )

    ranking_csv = out_dir / f"target_seed_ranking_{target_shot}shot_epoch{resolved_epoch}.csv"
    summary_csv = out_dir / f"target_preserve_tradeoff_group_summary_{target_shot}shot_epoch{resolved_epoch}.csv"
    ranking.to_csv(ranking_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    print(f"saved: {ranking_csv}")
    print(f"saved: {summary_csv}")

    plot_target_preserve_tradeoff_1x3(
        summary=summary,
        out_dir=out_dir,
        title_prefix=title_prefix,
        shot=target_shot,
        selection_epoch=resolved_epoch,
        ylim=ylim,
    )


# -------------------------
# main process
# -------------------------

def process_one_experiment(exp_dir: Path, args: argparse.Namespace) -> Path:
    out_dir = exp_dir / args.out_name
    title_prefix = exp_dir.name

    print("\n" + "=" * 80)
    print(f"[실험 처리] {exp_dir}")

    df = load_curves(
        root_dir=exp_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )
    summary = summarize_accuracy(df)
    table = make_wide_table(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_seed_accuracy_flat.csv", index=False)
    summary.to_csv(out_dir / "accuracy_summary.csv", index=False)
    table.to_csv(out_dir / "wide_accuracy_table.csv", index=False)

    print("\n===== loaded =====")
    print(f"experiment_dir: {exp_dir}")
    print(f"output_dir    : {out_dir}")
    print(f"epochs used   : {sorted(df['epoch'].unique())}")
    print(f"num seeds     : {df['sample_seed'].nunique()}")
    print(f"shots         : {sorted(df['shot'].unique())}")
    print(df[["split", "dataset"]].drop_duplicates().to_string(index=False))
    print(f"ylim mean_iqr_accuracy       : {args.ylim_mean_iqr}")
    print(f"ylim spaghetti_16shot_accuracy: {args.ylim_spaghetti}")
    print(f"ylim best_16shot_cases_accuracy: {args.ylim_best_cases}")
    print(f"ylim target_preserve_tradeoff : {args.ylim_tradeoff}")
    print(f"target preserve tradeoff shots : {args.target_shot}")

    plot_mean_iqr_accuracy(
        summary=summary,
        out_dir=out_dir / "mean_iqr_accuracy",
        title_prefix=title_prefix,
        ylim=args.ylim_mean_iqr,
    )

    plot_spaghetti_accuracy(
        df=df,
        out_dir=out_dir / f"spaghetti_{args.best_shot}shot_accuracy",
        title_prefix=title_prefix,
        only_shot=args.best_shot,
        ylim=args.ylim_spaghetti,
    )

    plot_best_cases(
        table=table,
        out_dir=out_dir / f"best_{args.best_shot}shot_cases_accuracy",
        title_prefix=title_prefix,
        shot=args.best_shot,
        selection_epoch=args.selection_epoch,
        ylim=args.ylim_best_cases,
    )

    for target_shot in args.target_shot:
        plot_target_preserve_tradeoff_groups(
            table=table,
            out_dir=out_dir / "target_preserve_tradeoff",
            title_prefix=title_prefix,
            target_shot=int(target_shot),
            selection_epoch=args.selection_epoch,
            ylim=args.ylim_tradeoff,
        )

    print("\n===== 완료 =====")
    print(f"output dir: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="실험 폴더 또는 그 상위 폴더. 예: teacher_confident 또는 B__synthetic_zscl_like",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument("--best-shot", type=int, default=16)
    parser.add_argument(
        "--target-shot",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="target-preserve trade-off를 그릴 shot 목록. 예: --target-shot 1 2 4 8 16",
    )
    parser.add_argument("--selection-epoch", type=int, default=50)
    parser.add_argument(
        "--out-name",
        type=str,
        default="accuracy_plots",
        help="각 실험 폴더 안에 생성할 결과 폴더명",
    )
    parser.add_argument(
        "--only-experiment",
        type=str,
        nargs="*",
        default=None,
        help="처리할 실험 폴더명 일부만 지정. 예: B__synthetic_zscl_like C__gift_synthetic",
    )

    parser.add_argument("--ylim-mean-iqr", type=float, nargs=2, default=[0.30, 0.75], metavar=("YMIN", "YMAX"))
    parser.add_argument("--ylim-spaghetti", type=float, nargs=2, default=[0.30, 0.75], metavar=("YMIN", "YMAX"))
    parser.add_argument("--ylim-best-cases", type=float, nargs=2, default=[0.30, 0.75], metavar=("YMIN", "YMAX"))
    parser.add_argument("--ylim-tradeoff", type=float, nargs=2, default=[0.30, 0.75], metavar=("YMIN", "YMAX"))

    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root-dir 없음: {root_dir}")

    exp_dirs = discover_experiment_dirs(root_dir)
    if args.only_experiment:
        wanted = set(args.only_experiment)
        exp_dirs = [p for p in exp_dirs if p.name in wanted or any(w in p.name for w in wanted)]

    if not exp_dirs:
        raise RuntimeError(
            f"처리할 실험 폴더를 찾지 못함: {root_dir}\n"
            "필요 구조: EXP_DIR/sample_seed_*/curves.jsonl 또는 EXP_DIR/sample_seed_*/shot_*/history.json"
        )

    print("\n===== 발견한 실험 폴더 =====")
    for p in exp_dirs:
        print(p)

    output_dirs = []
    failed = []
    for exp_dir in exp_dirs:
        try:
            output_dirs.append(process_one_experiment(exp_dir, args))
        except Exception as e:
            failed.append((exp_dir, e))
            print(f"\n[실패] {exp_dir}")
            print(f"reason: {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print("전체 처리 요약")
    print(f"성공: {len(output_dirs)}개")
    for p in output_dirs:
        print(f"  - {p}")

    if failed:
        print(f"실패: {len(failed)}개")
        for p, e in failed:
            print(f"  - {p}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
