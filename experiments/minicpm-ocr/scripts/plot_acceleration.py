"""Plot the recorded OCR generation times, including model prefill."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CASE_LABELS = {
    "case1_cn_document": "Mixed Chinese document",
    "case2_text_putnam": "Math proof text",
    "case3_formula_putnam": "Multiline formula",
    "case4_table_gb_standard": "Structured table",
}


def main():
    evidence = ROOT / "evidence" / "acceleration_benchmark.json"
    results = json.loads(evidence.read_text(encoding="utf-8"))
    cases = results["cases"]
    labels = [CASE_LABELS[case["id"]] for case in cases]

    fig, ax = plt.subplots(figsize=(8, 3.75), layout="constrained")
    fig.suptitle("MiniCPM-OCR generation time on DGX Spark", fontsize=14, weight="bold")
    for key, name, color, offset in (
        ("baseline_elapsed_s", "Greedy baseline", "#657489", -0.16),
        ("dspark_elapsed_s", "DSpark", "#087F75", 0.16),
    ):
        bars = ax.barh(
            [i + offset for i in range(len(cases))],
            [case[key] for case in cases],
            height=0.28,
            color=color,
            label=name,
        )
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9, color="#25344A")

    ax.set_yticks(range(len(cases)), labels)
    ax.set_xlabel("Model generation time, including prefill (s)")
    ax.set_xlim(0, max(case["baseline_elapsed_s"] for case in cases) * 1.17)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DCE3E9", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False)

    output = ROOT / "assets" / "dspark-generation-time.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
