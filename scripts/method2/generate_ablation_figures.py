from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Repository root:
# learning-from-demonstrations/
ROOT = Path(__file__).resolve().parents[2]

CSV = (
    ROOT
    / "results"
    / "method2_ablation"
    / "matched_tactile_ablation.csv"
)

OUT = (
    ROOT
    / "figures"
    / "method2"
    / "ablation"
)

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

signed = df["signed_delta_gripper"].to_numpy(float)

n = len(df)
n_more = int(np.sum(signed < 0))
n_less = int(np.sum(signed > 0))
mean_signed = float(np.mean(signed))

r = float(
    np.corrcoef(
        df["tactile_ratio"],
        df["abs_delta_gripper"]
    )[0, 1]
)

RED = "tab:red"
GREEN = "tab:green"
GRAY = "0.38"


# ============================================================
# FIGURE 1
# ============================================================

d = df.sort_values("target_episode")

colors = [
    RED if x < 0 else GREEN
    for x in d["signed_delta_gripper"]
]

fig, ax = plt.subplots(
    figsize=(11.5, 5.8)
)

# Compact header space
fig.subplots_adjust(
    top=0.79,
    bottom=0.16,
    left=0.10,
    right=0.98
)

fig.suptitle(
    "Effect of Tactile Substitution on ACT Gripper Predictions",
    fontsize=15,
    fontweight="bold",
    y=0.975
)

handles = [
    Patch(
        facecolor=RED,
        label=f"Real tactile → more closure ({n_more}/{n})"
    ),
    Patch(
        facecolor=GREEN,
        label=f"Real tactile → less closure ({n_less}/{n})"
    ),
    Line2D(
        [0], [0],
        color=GRAY,
        linestyle="--",
        linewidth=2,
        label=f"Mean signed effect: {mean_signed:+.2f}"
    ),
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.918),
    ncol=3,
    frameon=True,
    fontsize=9.5,
    borderpad=0.35,
    handlelength=1.6,
    columnspacing=1.4
)

ax.bar(
    d["target_episode"],
    d["signed_delta_gripper"],
    color=colors,
    width=0.78,
    alpha=0.88
)

ax.axhline(
    0,
    color="black",
    linewidth=1.1
)

ax.axhline(
    mean_signed,
    color=GRAY,
    linestyle="--",
    linewidth=1.8
)

ax.set_xlabel(
    "Demonstration episode"
)

ax.set_ylabel(
    "Change in predicted gripper command\n"
    "(real tactile − matched low tactile) [SO101 units]"
)

ax.set_xlim(-1, 50)

ax.grid(
    axis="y",
    alpha=0.22
)

fig.text(
    0.5,
    0.025,
    "Negative values indicate a more closed prediction with the real high-tactile observation.",
    ha="center",
    fontsize=8.8
)

p1 = OUT / "01_ablation_gripper_effect_compact.png"

fig.savefig(
    p1,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.close(fig)


# ============================================================
# FIGURE 2
# ============================================================

x = df["tactile_ratio"].to_numpy(float)
y = df["abs_delta_gripper"].to_numpy(float)

point_colors = [
    RED if s < 0 else GREEN
    for s in signed
]

coef = np.polyfit(
    x,
    y,
    1
)

xx = np.linspace(
    x.min(),
    x.max(),
    200
)

yy = (
    coef[0] * xx
    + coef[1]
)

fig, ax = plt.subplots(
    figsize=(8.4, 5.9)
)

fig.subplots_adjust(
    top=0.77,
    bottom=0.18,
    left=0.13,
    right=0.97
)

fig.suptitle(
    "Tactile Contrast and ACT Gripper Sensitivity",
    fontsize=15,
    fontweight="bold",
    y=0.975
)

handles = [
    Patch(
        facecolor=RED,
        label="Real tactile → more closure"
    ),
    Patch(
        facecolor=GREEN,
        label="Real tactile → less closure"
    ),
    Line2D(
        [0], [0],
        color=GRAY,
        linestyle="--",
        linewidth=2,
        label="Linear trend"
    ),
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.92),
    ncol=3,
    frameon=True,
    fontsize=9,
    borderpad=0.35,
    columnspacing=1.2
)

fig.text(
    0.5,
    0.835,
    f"Pearson correlation: r = {r:.3f}   (n = {n})",
    ha="center",
    fontsize=9.5
)

ax.scatter(
    x,
    y,
    c=point_colors,
    s=54,
    alpha=0.82,
    edgecolors="black",
    linewidths=0.35
)

ax.plot(
    xx,
    yy,
    color=GRAY,
    linestyle="--",
    linewidth=1.8
)

ax.set_xlabel(
    "Tactile contrast\n"
    "(target tactile / matched-low tactile)"
)

ax.set_ylabel(
    "|Change in predicted gripper command| [SO101 units]"
)

ax.grid(
    alpha=0.22
)

fig.text(
    0.5,
    0.025,
    "Tactile contrast is not interpreted as a physical force measurement.",
    ha="center",
    fontsize=8.8
)

p2 = OUT / "02_ablation_tactile_contrast_compact.png"

fig.savefig(
    p2,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.close(fig)


# ============================================================
# FIGURE 4
# ============================================================

gx = df["low_tactile_gripper"].to_numpy(float)
gy = df["normal_gripper"].to_numpy(float)

lo = float(
    min(
        gx.min(),
        gy.min()
    )
)

hi = float(
    max(
        gx.max(),
        gy.max()
    )
)

pad = 0.04 * (hi - lo)

fig, ax = plt.subplots(
    figsize=(7.0, 6.3)
)

fig.subplots_adjust(
    top=0.78,
    bottom=0.17,
    left=0.15,
    right=0.97
)

fig.suptitle(
    "Paired ACT Gripper Predictions Under Tactile Substitution",
    fontsize=14,
    fontweight="bold",
    y=0.975
)

handles = [
    Patch(
        facecolor=RED,
        label=f"More closure ({n_more}/{n})"
    ),
    Patch(
        facecolor=GREEN,
        label=f"Less closure ({n_less}/{n})"
    ),
    Line2D(
        [0], [0],
        color=GRAY,
        linestyle="--",
        linewidth=2,
        label="No change"
    ),
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.915),
    ncol=3,
    frameon=True,
    fontsize=9,
    borderpad=0.35,
    columnspacing=1.2
)

ax.scatter(
    gx,
    gy,
    c=point_colors,
    s=58,
    alpha=0.84,
    edgecolors="black",
    linewidths=0.35
)

ax.plot(
    [lo - pad, hi + pad],
    [lo - pad, hi + pad],
    color=GRAY,
    linestyle="--",
    linewidth=1.8
)

ax.set_xlim(
    lo - pad,
    hi + pad
)

ax.set_ylim(
    lo - pad,
    hi + pad
)

ax.set_aspect(
    "equal",
    adjustable="box"
)

ax.set_xlabel(
    "ACT prediction with matched low tactile [SO101 units]"
)

ax.set_ylabel(
    "ACT prediction with real tactile [SO101 units]"
)

ax.grid(
    alpha=0.20
)

fig.text(
    0.5,
    0.025,
    "Points below the diagonal correspond to a more-closed prediction with real tactile.",
    ha="center",
    fontsize=8.7
)

p4 = OUT / "04_ablation_gripper_prediction_parity_compact.png"

fig.savefig(
    p4,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.close(fig)


print()
print("Created compact Overleaf-ready figures:")
print(p1)
print(p2)
print(p4)
