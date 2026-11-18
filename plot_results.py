"""Plot aggregate MinneApple counts without distributing source images."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"learned": "#007d76", "color": "#3d64b1"}
LABELS = {"learned": "Learned centers", "color": "Red components"}
INK = "#152c3e"
MUTED = "#435e73"


def canvas(title, caption, shape=(10, 5)):
    figure, axis = plt.subplots(figsize=shape)
    figure.set_facecolor("#fcfdff")
    axis.set_facecolor(figure.get_facecolor())
    figure.subplots_adjust(left=0.17, right=0.92, top=0.78, bottom=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    figure.text(0.055, 0.925, title, fontsize=19, color=INK, weight="bold")
    figure.text(0.055, 0.045, caption, fontsize=9, color=MUTED)
    return figure, axis


def error_chart(items, title, caption, target):
    figure, axis = canvas(title, caption)
    figure.subplots_adjust(left=0.24)
    y = np.arange(len(items))
    for i, method in enumerate(COLORS):
        values = [summary.get(method, {}).get("mae", np.nan) for _, summary in items]
        bars = axis.barh(y + 0.16 * (2 * i - 1), values, height=0.30, color=COLORS[method], label=LABELS[method])
        axis.bar_label(bars, fmt="%.1f", fontsize=9, padding=4, color=INK)
    axis.set_yticks(y, [f"{label}  ·  n={summary['images']}" for label, summary in items])
    axis.invert_yaxis()
    axis.grid(axis="x", color="#d0dce8")
    axis.set_xlabel("Mean absolute count error (apples per image)")
    axis.margins(x=0.18)
    axis.legend(loc="lower right", bbox_to_anchor=(1, 1.015), ncol=2, frameon=False)
    figure.savefig(target, dpi=175)
    plt.close(figure)


def render(metrics, folder):
    folder.mkdir(parents=True, exist_ok=True)
    rows = [("All groups", metrics["test"])]
    rows.extend((name.replace("dataset", "Row ").replace("_", " "), data)
                for name, data in metrics["by_row_side"].items())
    caption = "MinneApple detection release · custom holdout on orchard rows 3 and 4"
    if metrics["protocol"]["design"] == "official_train_and_detection_test":
        caption = "Official MinneApple detection test subset · equal image counts per available row and side"
    error_chart(rows, "Count error varies across orchard rows", caption,
                folder / "count-error-by-row.png")
    cases = [("Small majority", metrics["by_size"]["median_below_1024_px2"]),
             ("Other size", metrics["by_size"]["other"]),
             ("Box overlap", metrics["by_overlap"]["any_box_iou_above_0_1"]),
             ("No box overlap", metrics["by_overlap"]["other"])]
    error_chart(cases, "Count error by size and box overlap",
                "Small majority · median native box area below 1,024 px²   |   Overlap · any annotated pair with box IoU > 0.10",
                folder / "count-error-by-cohort.png")
    loss = metrics["training_loss"]
    figure, axis = canvas("Center-head training loss",
                          "Mean focal loss per batch · frozen ImageNet features · calibration and test excluded")
    axis.plot(range(1, len(loss) + 1), loss, color=COLORS["learned"], marker="o", linewidth=2)
    axis.set(xlabel="Epoch", ylabel="Mean focal loss", xticks=range(1, len(loss) + 1))
    axis.grid(axis="y", color="#d0dce8")
    figure.savefig(folder / "center-head-training.png", dpi=175)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("reports/minneapple-metrics.json"))
    parser.add_argument("--out", type=Path, default=Path("figures/results"))
    args = parser.parse_args()
    render(json.loads(args.metrics.read_text(encoding="utf-8")), args.out)
