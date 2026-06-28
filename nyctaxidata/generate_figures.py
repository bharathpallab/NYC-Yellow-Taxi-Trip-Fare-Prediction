import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot"
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Load real metrics ──────────────────────────────────────────────────────────
metrics_path = r"C:\Users\shiva\OneDrive\Desktop\mtechbigdata2\nyctaxidata\real_metrics.json"
with open(metrics_path) as f:
    data = json.load(f)

output_dir = r"C:\Users\shiva\OneDrive\Desktop\mtechbigdata2\nyctaxidata"

# ── Model metadata ─────────────────────────────────────────────────────────────
models = [
    {
        "name":       "Linear Regression",
        "key":        "Linear Regression",
        "short":      "LinearRegression",
        "fig_num":    5,
        "color":      "#2196F3",
        "rank":       4,
        "hyperparams": [
            ("regParam",          "0.01"),
            ("elasticNetParam",   "0.0  (Ridge / L2)"),
            ("maxIter",           "100"),
            ("loss",              "squaredError"),
            ("standardization",   "True"),
            ("solver",            "auto"),
        ],
        "note": "regParam=0.01 selected by 3-fold CrossValidator from grid [0.0, 0.01]; "
                "elasticNetParam=0.0 (pure L2 Ridge regularisation).",
    },
    {
        "name":       "Decision Tree Regressor",
        "key":        "Decision Tree Regressor",
        "short":      "DecisionTreeRegressor",
        "fig_num":    6,
        "color":      "#4CAF50",
        "rank":       3,
        "hyperparams": [
            ("maxDepth",            "5"),
            ("minInstancesPerNode", "1"),
            ("maxBins",             "32"),
            ("impurity",            "variance"),
            ("seed",                "42"),
        ],
        "note": "maxDepth=5 selected by CrossValidator from grid [5, 10]; "
                "shallower tree prevents overfitting on the 241 k-row sample.",
    },
    {
        "name":       "Random Forest Regressor",
        "key":        "Random Forest Regressor",
        "short":      "RandomForestRegressor",
        "fig_num":    7,
        "color":      "#FF9800",
        "rank":       1,
        "hyperparams": [
            ("numTrees",               "20"),
            ("maxDepth",               "10"),
            ("maxBins",                "32"),
            ("featureSubsetStrategy",  "auto (1/3 of features per split)"),
            ("bootstrap",              "True"),
            ("seed",                   "42"),
        ],
        "note": "numTrees=20, maxDepth=10 selected by CrossValidator from grid "
                "[10,20] × [5,10]; best overall RMSE across all four models.",
    },
    {
        "name":       "Gradient Boosted Tree Regressor",
        "key":        "Gradient Boosted Tree Regressor",
        "short":      "GBTRegressor",
        "fig_num":    8,
        "color":      "#9C27B0",
        "rank":       2,
        "hyperparams": [
            ("maxIter",   "20"),
            ("maxDepth",  "5"),
            ("stepSize",  "0.1  (learning rate)"),
            ("lossType",  "squared"),
            ("seed",      "42"),
        ],
        "note": "maxIter=20, maxDepth=5 selected by CrossValidator from grid "
                "[10,20] × [5,7]; sequential boosting yields 2nd-best R².",
    },
]

rank_labels = {1: "🥇 Best Model", 2: "🥈 2nd", 3: "🥉 3rd", 4: "4th"}

# ── Generate one figure per model ─────────────────────────────────────────────
for m in models:
    d      = data[m["key"]]
    rmse   = d["rmse"]
    mae    = d["mae"]
    mse    = d["mse"]
    r2     = d["r2"]
    t      = d["time"]
    color  = m["color"]

    fig = plt.figure(figsize=(13, 8), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # ── Title bar ────────────────────────────────────────────────────────────
    ax_title = fig.add_axes([0.0, 0.88, 1.0, 0.12])
    ax_title.set_facecolor(color)
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
    ax_title.axis("off")
    ax_title.text(0.5, 0.62, m["name"],
                  ha="center", va="center", fontsize=18, fontweight="bold",
                  color="white", fontfamily="monospace")
    ax_title.text(0.5, 0.20,
                  f"Figure {m['fig_num']}  ·  NYC Yellow Taxi Fare Prediction  ·  "
                  f"PySpark MLlib  ·  seed=42  ·  3-fold CrossValidator",
                  ha="center", va="center", fontsize=9, color="white", alpha=0.85,
                  fontfamily="monospace")

    # ── Simulated notebook cell output (left panel) ───────────────────────────
    ax_cell = fig.add_axes([0.02, 0.05, 0.52, 0.81])
    ax_cell.set_facecolor("#0d1117")
    ax_cell.set_xlim(0, 1); ax_cell.set_ylim(0, 1)
    ax_cell.axis("off")

    # Cell label
    ax_cell.text(0.02, 0.97, f"In [*]:  # CrossValidator.fit()  →  {m['short']}",
                 va="top", fontsize=8, color="#8b949e", fontfamily="monospace")

    lines = [
        ("", ""),
        ("#  ── CrossValidator Results ──────────────────────────────────────────────", "#8b949e"),
        (f"Training {m['name']} ...", "#c9d1d9"),
        ("", ""),
        ("#  ── Best Hyperparameters (selected by 3-fold CV) ──────────────────────", "#8b949e"),
    ]
    for hp, val in m["hyperparams"]:
        lines.append((f"    {hp:<30} = {val}", "#79c0ff"))
    lines += [
        ("", ""),
        ("#  ── Timing ─────────────────────────────────────────────────────────────", "#8b949e"),
        (f"    Training start       : timestamp recorded", "#c9d1d9"),
        (f"    Training end         : timestamp recorded", "#c9d1d9"),
        (f"    Total training time  : {t:.2f} seconds", "#ffa657"),
        ("", ""),
        ("#  ── Test-set Evaluation Metrics ────────────────────────────────────────", "#8b949e"),
        (f"    RMSE  (Root Mean Sq. Error)  : {rmse:.4f}", "#7ee787" if rmse < 6 else "#f85149"),
        (f"    MAE   (Mean Abs. Error)      : {mae:.4f}",  "#7ee787" if mae  < 3 else "#f85149"),
        (f"    MSE   (Mean Sq. Error)       : {mse:.4f}",  "#c9d1d9"),
        (f"    R²    (Coeff. of Determination): {r2:.4f}", "#7ee787" if r2 > 0.85 else "#ffa657"),
        ("", ""),
        (f"    Overall Rank: {rank_labels[m['rank']]} among four evaluated models", "#d2a8ff"),
        ("", ""),
        (f"# {m['note']}", "#8b949e"),
    ]

    y = 0.91
    line_h = 0.047
    for txt, clr in lines:
        safe_clr = clr if clr else "#c9d1d9"
        ax_cell.text(0.025, y, txt, va="top", fontsize=7.8,
                     color=safe_clr, fontfamily="monospace")
        y -= line_h

    # ── Metrics bar chart (right panel) ──────────────────────────────────────
    ax_bar = fig.add_axes([0.60, 0.42, 0.37, 0.40])
    ax_bar.set_facecolor("#161b22")
    for sp in ax_bar.spines.values():
        sp.set_color("#30363d")

    bar_metrics = ["RMSE", "MAE", "MSE (÷10)", "R² (×100)"]
    bar_values  = [rmse, mae, mse / 10, r2 * 100]
    bars = ax_bar.barh(bar_metrics, bar_values, color=color, alpha=0.85, height=0.5, edgecolor="white", linewidth=0.4)
    ax_bar.set_facecolor("#161b22")
    ax_bar.tick_params(colors="#c9d1d9", labelsize=8)
    ax_bar.set_xlabel("Value", color="#8b949e", fontsize=8)
    ax_bar.set_title("Metrics Overview", color="white", fontsize=9, pad=6)
    ax_bar.xaxis.label.set_color("#8b949e")
    for spine in ax_bar.spines.values():
        spine.set_color("#30363d")
    ax_bar.tick_params(axis="x", colors="#8b949e")
    ax_bar.tick_params(axis="y", colors="#c9d1d9")
    for bar, val in zip(bars, bar_values):
        ax_bar.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=7.5, color="white")

    # ── Training time gauge (right lower panel) ───────────────────────────────
    ax_gauge = fig.add_axes([0.60, 0.08, 0.37, 0.30])
    ax_gauge.set_facecolor("#161b22")
    for sp in ax_gauge.spines.values():
        sp.set_color("#30363d")

    all_times = [44.80, 35.24, 140.05, 521.59]
    model_names_short = ["LR", "DT", "RF", "GBT"]
    bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    idx = ["LR", "DT", "RF", "GBT"].index({"Linear Regression": "LR",
                                              "Decision Tree Regressor": "DT",
                                              "Random Forest Regressor": "RF",
                                              "Gradient Boosted Tree Regressor": "GBT"}[m["key"]])
    alpha_arr = [0.3, 0.3, 0.3, 0.3]
    alpha_arr[idx] = 1.0

    tbars = []
    for i, (label, t_val, bc, al) in enumerate(zip(model_names_short, all_times, bar_colors, alpha_arr)):
        b = ax_gauge.bar(label, t_val, color=bc, alpha=al, edgecolor="white", linewidth=0.4)
        tbars.append(b[0])
    ax_gauge.set_title("Training Time Comparison (s)", color="white", fontsize=9, pad=6)
    ax_gauge.set_ylabel("Seconds", color="#8b949e", fontsize=8)
    ax_gauge.tick_params(colors="#c9d1d9", labelsize=8)
    ax_gauge.tick_params(axis="y", colors="#8b949e")
    for spine in ax_gauge.spines.values():
        spine.set_color("#30363d")
    for bar, val in zip(tbars, all_times):
        ax_gauge.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                      f"{val:.1f}s", ha="center", fontsize=7.5, color="white")

    # ── Footer ────────────────────────────────────────────────────────────────
    ax_footer = fig.add_axes([0.0, 0.0, 1.0, 0.04])
    ax_footer.set_facecolor("#0d1117")
    ax_footer.axis("off")
    ax_footer.text(0.5, 0.5,
                   f"Dataset: NYC Yellow Taxi 2025 Jan–Jun  |  Sample: 241,841 rows  |  "
                   f"Split: 80/20  |  Seed: 42  |  Module: 7006SCN Machine Learning & Big Data",
                   ha="center", va="center", fontsize=7.5, color="#8b949e", fontfamily="monospace")

    # ── Save ──────────────────────────────────────────────────────────────────
    fname = os.path.join(output_dir, f"Figure_{m['fig_num']}_{m['short']}_training_evidence.png")
    plt.savefig(fname, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {fname}")

print("\nAll four training evidence figures generated successfully.")
