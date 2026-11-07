"""Plot the released ranking metrics and paired deltas."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluation = json.loads(args.summary.read_text(encoding="utf-8"))["evaluation"]
    overall = evaluation["all"]

    fig, (scores, deltas) = plt.subplots(
        1, 2, figsize=(10.8, 4.1), width_ratios=(1.1, 1), layout="constrained"
    )
    fig.suptitle("SciFact reranking on DGX Spark", fontsize=15, weight="bold")
    scores.set_title(f"Held-out queries (n={overall['queries']})", fontsize=11)
    methods = (
        ("bm25", "BM25", "#647489"),
        ("initial", "Random rank head", "#B3BDCA"),
        ("finetuned", "Fine-tuned", "#087F75"),
    )
    for i, (key, label, color) in enumerate(methods):
        bars = scores.barh(
            [j + (i - 1) * 0.21 for j in range(2)],
            [overall["rankings"][key][metric] for metric in ("ndcg_at_10", "mrr_at_10")],
            height=0.18,
            color=color,
            label=label,
        )
        scores.bar_label(bars, fmt="%.3f", padding=4, fontsize=8, color="#25344A")
    scores.set_yticks([0, 1], ["nDCG@10", "MRR@10"])
    scores.set_xlim(0, 1)
    scores.invert_yaxis()
    scores.set_xlabel("Mean score")
    scores.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8, ncol=3)

    deltas.set_title("Fine-tuned vs. BM25", fontsize=11)
    deltas.axvline(0, color="#647489", linewidth=1, linestyle="--")
    labels = []
    groups = (
        ("all", "All queries"),
        ("no_relevant_doc_seen_in_training_pairs", "No relevant doc in\ntraining pairs"),
    )
    for i, (key, label) in enumerate(groups):
        group = evaluation[key]
        labels.append(f"{label} (n={group['queries']})")
        if group["queries"]:
            comparison = group["finetuned_vs_bm25_ndcg_at_10"]
            low, high = comparison["paired_bootstrap_ci95"]
            deltas.hlines(i, low, high, color="#087F75", linewidth=1.8)
            deltas.scatter(comparison["delta"], i, color="#087F75", zorder=3)
    deltas.set_yticks(range(len(groups)), labels, fontsize=8)
    deltas.set_ylim(len(groups) - 0.5, -0.5)
    deltas.set_xlabel("Change in nDCG@10 (paired 95% bootstrap CI)")

    for axis in (scores, deltas):
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#DCE3E9", linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
