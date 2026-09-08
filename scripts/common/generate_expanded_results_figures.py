from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

# Repository root:
# learning-from-demonstrations/
ROOT = Path(__file__).resolve().parents[2]

BASELINE_PATH = ROOT / "data" / "method1" / "act_no_limiter.csv"
LIMITED_PATH = ROOT / "data" / "method1" / "act_with_safety_limiter.csv"
DIRECT_PATH = ROOT / "data" / "method2" / "successful_rollout.csv"

OUT = ROOT / "figures" / "expanded_results"
OUT.mkdir(parents=True, exist_ok=True)

for path in (BASELINE_PATH, LIMITED_PATH, DIRECT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

baseline = pd.read_csv(BASELINE_PATH)
limited = pd.read_csv(LIMITED_PATH)
direct = pd.read_csv(DIRECT_PATH)


# ============================================================
# COLOR CONVENTION
# ============================================================

BLUE = "tab:blue"
ORANGE = "tab:orange"
GREEN = "tab:green"
RED = "tab:red"
GRAY = "0.35"


# ============================================================
# EVENT DEFINITIONS
# ============================================================

# ------------------------------------------------------------
# Method 1:
# direct logged contact event
# ------------------------------------------------------------

contact_rows = limited[
    limited["contact_detected"] == 1
]

if contact_rows.empty:
    raise RuntimeError(
        "No Method 1 contact_detected event found."
    )

m1_contact_t = float(
    contact_rows.iloc[0]["time_s"]
)


# ------------------------------------------------------------
# Method 2:
# signal-based tactile reference
# ------------------------------------------------------------

peak_eflesh = float(
    direct["eflesh_combined_total"].max()
)

eflesh_threshold = 0.50 * peak_eflesh

mask = (
    direct["eflesh_combined_total"].to_numpy()
    >= eflesh_threshold
)

indices = np.where(mask)[0]

if len(indices) == 0:
    raise RuntimeError(
        "No Method 2 high-tactile interval found."
    )

segments = []

start = indices[0]
previous = indices[0]

for i in indices[1:]:
    if i == previous + 1:
        previous = i
    else:
        segments.append((start, previous))
        start = i
        previous = i

segments.append((start, previous))

dominant_segment = max(
    segments,
    key=lambda segment:
        segment[1] - segment[0]
)

m2_tactile_start = float(
    direct["time_s"].iloc[
        dominant_segment[0]
    ]
)

m2_tactile_end = float(
    direct["time_s"].iloc[
        dominant_segment[1]
    ]
)

m2_peak_idx = (
    direct["eflesh_combined_total"].idxmax()
)

m2_peak_t = float(
    direct.loc[
        m2_peak_idx,
        "time_s"
    ]
)


# ------------------------------------------------------------
# Method 1 baseline reference
#
# Same convention used in our earlier comparison:
# largest current event while gripper_position <= 15,
# then estimated interaction reference 0.25 s earlier.
# ------------------------------------------------------------

low = baseline[
    baseline["gripper_position"] <= 15
].copy()

if low.empty:
    raise RuntimeError(
        "No low-position baseline samples found."
    )

baseline_peak_idx = (
    low["abs_current_raw"].idxmax()
)

baseline_peak_t = float(
    baseline.loc[
        baseline_peak_idx,
        "time_s"
    ]
)

baseline_reference_t = (
    baseline_peak_t - 0.25
)


# ============================================================
# COMMON 2 s COMPARISON WINDOWS
# ============================================================

WINDOW = 2.0

m1_2s = limited[
    (limited["time_s"] >= m1_contact_t)
    & (
        limited["time_s"]
        <= m1_contact_t + WINDOW
    )
].copy()

m2_2s = direct[
    (direct["time_s"] >= m2_tactile_start)
    & (
        direct["time_s"]
        <= m2_tactile_start + WINDOW
    )
].copy()

m1_2s["relative_time_s"] = (
    m1_2s["time_s"] - m1_contact_t
)

m2_2s["relative_time_s"] = (
    m2_2s["time_s"] - m2_tactile_start
)


# ============================================================
# COMMAND-INTERVENTION METRICS
# ============================================================

m1_2s["intervention"] = (
    m1_2s["act_gripper_command"]
    - m1_2s["sent_gripper_command"]
).abs()

m2_2s["intervention"] = (
    m2_2s["policy_cmd.gripper.pos"]
    - m2_2s["sent_cmd.gripper.pos"]
).abs()

m1_mean = float(
    m1_2s["intervention"].mean()
)

m1_max = float(
    m1_2s["intervention"].max()
)

m1_fraction = float(
    (
        m1_2s["intervention"] > 1e-9
    ).mean()
)

m2_mean = float(
    m2_2s["intervention"].mean()
)

m2_max = float(
    m2_2s["intervention"].max()
)

m2_fraction = float(
    (
        m2_2s["intervention"] > 1e-9
    ).mean()
)


# ============================================================
# FIGURE 1
# METHOD 1 — COMPLETE SAFETY-LAYER TIMELINE
# ============================================================

PRE = 0.55
POST = 2.10

w1 = limited[
    (
        limited["time_s"]
        >= m1_contact_t - PRE
    )
    & (
        limited["time_s"]
        <= m1_contact_t + POST
    )
].copy()

w1["relative_time_s"] = (
    w1["time_s"] - m1_contact_t
)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(11.2, 8.4),
    sharex=True,
    gridspec_kw={
        "height_ratios": [
            2.0,
            1.35,
            0.95
        ]
    }
)


# ---------------------------
# Commands / measured motion
# ---------------------------

ax = axes[0]

ax.plot(
    w1["relative_time_s"],
    w1["act_gripper_command"],
    color=BLUE,
    linewidth=2.3,
    label="ACT requested"
)

ax.plot(
    w1["relative_time_s"],
    w1["sent_gripper_command"],
    color=ORANGE,
    linewidth=2.3,
    label="Sent after safety layer"
)

ax.plot(
    w1["relative_time_s"],
    w1["gripper_position"],
    color=GREEN,
    linewidth=2.0,
    label="Measured gripper"
)

valid_boundary = (
    w1["safe_boundary_position"].notna()
)

if valid_boundary.any():

    ax.plot(
        w1.loc[
            valid_boundary,
            "relative_time_s"
        ],
        w1.loc[
            valid_boundary,
            "safe_boundary_position"
        ],
        color=RED,
        linestyle=":",
        linewidth=1.8,
        label="Safe boundary"
    )

post_contact = (
    w1["relative_time_s"] >= 0
)

ax.fill_between(
    w1.loc[
        post_contact,
        "relative_time_s"
    ],
    w1.loc[
        post_contact,
        "act_gripper_command"
    ],
    w1.loc[
        post_contact,
        "sent_gripper_command"
    ],
    color=ORANGE,
    alpha=0.18,
    label="External intervention"
)

ax.axvline(
    0,
    color=RED,
    linestyle="--",
    linewidth=1.6
)

ax.set_ylabel(
    "Gripper command / position\n"
    "[SO101 units]"
)

ax.legend(
    loc="upper right",
    ncol=2
)

ax.grid(
    alpha=0.22
)


# ---------------------------
# Motor current
# ---------------------------

ax = axes[1]

ax.plot(
    w1["relative_time_s"],
    w1["abs_current_raw"],
    color=BLUE,
    linewidth=2.2,
    label="|Motor current|"
)

relief = (
    w1["current_relief"] == 1
)

if relief.any():

    ax.scatter(
        w1.loc[
            relief,
            "relative_time_s"
        ],
        w1.loc[
            relief,
            "abs_current_raw"
        ],
        color=RED,
        s=58,
        zorder=5,
        label="Current-relief active"
    )

ax.axvline(
    0,
    color=RED,
    linestyle="--",
    linewidth=1.6
)

ax.set_ylabel(
    "Motor current [raw]"
)

ax.legend(
    loc="upper right"
)

ax.grid(
    alpha=0.22
)


# ---------------------------
# Logged safety states
# ---------------------------

ax = axes[2]

status_series = [
    (
        "Contact detected",
        "contact_detected",
        BLUE,
        2
    ),
    (
        "Position limit",
        "position_limit_active",
        ORANGE,
        1
    ),
    (
        "Current relief",
        "current_relief",
        RED,
        0
    )
]

for (
    label,
    column,
    color,
    level
) in status_series:

    y = np.where(
        w1[column].to_numpy() > 0,
        level + 0.72,
        level + 0.08
    )

    ax.step(
        w1["relative_time_s"],
        y,
        where="post",
        color=color,
        linewidth=2.0,
        label=label
    )

    ax.fill_between(
        w1["relative_time_s"],
        level + 0.08,
        y,
        step="post",
        color=color,
        alpha=0.18
    )

ax.axvline(
    0,
    color=RED,
    linestyle="--",
    linewidth=1.6
)

ax.set_yticks([
    0.4,
    1.4,
    2.4
])

ax.set_yticklabels([
    "Current relief",
    "Position limit",
    "Contact detected"
])

ax.set_ylim(
    0,
    3
)

ax.set_xlabel(
    "Time relative to detected contact [s]"
)

ax.grid(
    axis="x",
    alpha=0.22
)

fig.suptitle(
    "Method 1: External Safety Layer Responds Immediately After Contact",
    fontsize=15,
    fontweight="bold"
)

fig.text(
    0.5,
    0.012,
    (
        "Shaded command gap shows where the "
        "safety layer overrides the ACT request."
    ),
    ha="center",
    fontsize=10
)

fig.tight_layout(
    rect=[
        0,
        0.035,
        1,
        0.96
    ]
)

fig.savefig(
    OUT
    / "01_method1_safety_layer_timeline.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 2
# METHOD 2 — 10 MAGNETOMETER TACTILE HEATMAP
# ============================================================

heat_start = max(
    float(
        direct["time_s"].min()
    ),
    m2_tactile_start - 2.0
)

heat_end = min(
    float(
        direct["time_s"].max()
    ),
    m2_tactile_end + 1.5
)

wh = direct[
    (
        direct["time_s"]
        >= heat_start
    )
    & (
        direct["time_s"]
        <= heat_end
    )
].copy()

rows = []
row_labels = []

for prefix, side_label in [
    (
        "eflesh_on_motor",
        "Motor side"
    ),
    (
        "eflesh_not_motor",
        "Opposite side"
    )
]:

    for magnet in range(5):

        cols = [
            f"{prefix}.m{magnet}.bx",
            f"{prefix}.m{magnet}.by",
            f"{prefix}.m{magnet}.bz"
        ]

        magnitude = np.linalg.norm(
            wh[cols].to_numpy(
                dtype=float
            ),
            axis=1
        )

        rows.append(
            magnitude
        )

        row_labels.append(
            f"{side_label}  M{magnet}"
        )

heat = np.vstack(
    rows
)

fig, ax = plt.subplots(
    figsize=(11.4, 6.2)
)

im = ax.imshow(
    heat,
    aspect="auto",
    interpolation="nearest",
    origin="upper",
    extent=[
        float(
            wh["time_s"].iloc[0]
        ),
        float(
            wh["time_s"].iloc[-1]
        ),
        9.5,
        -0.5
    ],
    cmap="viridis"
)

ax.set_yticks(
    np.arange(10)
)

ax.set_yticklabels(
    row_labels
)

# White divider between gripper sides
ax.axhline(
    4.5,
    color="white",
    linewidth=1.6,
    alpha=0.9
)

# Tactile event markers
ax.axvline(
    m2_tactile_start,
    color="white",
    linestyle="--",
    linewidth=1.6
)

ax.axvline(
    m2_peak_t,
    color="white",
    linestyle=":",
    linewidth=1.6
)

ax.axvline(
    m2_tactile_end,
    color="white",
    linestyle="--",
    linewidth=1.6
)

ax.text(
    m2_tactile_start + 0.08,
    -0.15,
    "High-response onset",
    color="white",
    fontsize=9,
    va="top"
)

ax.text(
    m2_peak_t + 0.08,
    8.95,
    "Peak",
    color="white",
    fontsize=9,
    va="bottom"
)

ax.text(
    m2_tactile_end - 0.08,
    -0.15,
    "Unloading",
    color="white",
    fontsize=9,
    va="top",
    ha="right"
)

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.015
)

cbar.set_label(
    "Magnetic delta magnitude"
)

ax.set_xlabel(
    "Trial time [s]"
)

ax.set_title(
    (
        "Method 2: Distributed Tactile Response "
        "Across All 10 eFlesh Magnetometers"
    ),
    fontsize=14,
    fontweight="bold"
)

fig.tight_layout()

fig.savefig(
    OUT
    / "02_method2_10mag_tactile_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 3
# METHOD 2 — SUCCESSFUL GRASP TIMELINE
# ============================================================

grasp_start = max(
    float(
        direct["time_s"].min()
    ),
    m2_tactile_start - 2.5
)

grasp_end = min(
    float(
        direct["time_s"].max()
    ),
    m2_tactile_end + 1.5
)

wg = direct[
    (
        direct["time_s"]
        >= grasp_start
    )
    & (
        direct["time_s"]
        <= grasp_end
    )
].copy()

fig, axes = plt.subplots(
    2,
    1,
    figsize=(11.2, 7.4),
    sharex=True,
    gridspec_kw={
        "height_ratios": [
            1.35,
            1.25
        ]
    }
)


# ---------------------------
# Tactile signals
# ---------------------------

ax = axes[0]

ax.plot(
    wg["time_s"],
    wg["eflesh_combined_total"],
    color=BLUE,
    linewidth=2.4,
    label="Combined eFlesh"
)

ax.plot(
    wg["time_s"],
    wg["eflesh_on_motor_total"],
    color=ORANGE,
    linewidth=1.6,
    alpha=0.9,
    label="Motor-side eFlesh"
)

ax.plot(
    wg["time_s"],
    wg["eflesh_not_motor_total"],
    color=GREEN,
    linewidth=1.6,
    alpha=0.9,
    label="Opposite-side eFlesh"
)

ax.axhline(
    eflesh_threshold,
    color=GRAY,
    linestyle=":",
    linewidth=1.4,
    label="50% peak analysis threshold"
)

ax.axvspan(
    m2_tactile_start,
    m2_tactile_end,
    color=BLUE,
    alpha=0.08
)

for x, style in [
    (
        m2_tactile_start,
        "--"
    ),
    (
        m2_peak_t,
        ":"
    ),
    (
        m2_tactile_end,
        "--"
    )
]:

    ax.axvline(
        x,
        color=RED,
        linestyle=style,
        linewidth=1.3
    )

ax.annotate(
    "High tactile-response onset",
    xy=(
        m2_tactile_start,
        eflesh_threshold
    ),
    xytext=(
        m2_tactile_start - 1.95,
        peak_eflesh * 0.68
    ),
    arrowprops={
        "arrowstyle": "->"
    },
    fontsize=9.5
)

ax.annotate(
    (
        "Peak response\n"
        f"{peak_eflesh:,.0f}"
    ),
    xy=(
        m2_peak_t,
        peak_eflesh
    ),
    xytext=(
        m2_peak_t + 0.45,
        peak_eflesh * 0.88
    ),
    arrowprops={
        "arrowstyle": "->"
    },
    fontsize=9.5
)

ax.set_ylabel(
    "eFlesh response"
)

ax.legend(
    loc="upper left",
    ncol=2
)

ax.grid(
    alpha=0.22
)


# ---------------------------
# ACT and measured gripper
# ---------------------------

ax = axes[1]

ax.plot(
    wg["time_s"],
    wg["policy_cmd.gripper.pos"],
    color=BLUE,
    linewidth=2.0,
    label="ACT requested"
)

ax.plot(
    wg["time_s"],
    wg["sent_cmd.gripper.pos"],
    color=ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Sent command"
)

ax.plot(
    wg["time_s"],
    wg["gripper.pos"],
    color=GREEN,
    linewidth=2.0,
    label="Measured gripper"
)

ax.axvspan(
    m2_tactile_start,
    m2_tactile_end,
    color=BLUE,
    alpha=0.08,
    label="Dominant high-tactile interval"
)

for x, style in [
    (
        m2_tactile_start,
        "--"
    ),
    (
        m2_peak_t,
        ":"
    ),
    (
        m2_tactile_end,
        "--"
    )
]:

    ax.axvline(
        x,
        color=RED,
        linestyle=style,
        linewidth=1.3
    )

ax.set_ylabel(
    "Gripper position / command\n"
    "[SO101 units]"
)

ax.set_xlabel(
    "Trial time [s]"
)

ax.legend(
    loc="upper right",
    ncol=2
)

ax.grid(
    alpha=0.22
)

fig.suptitle(
    (
        "Method 2: Tactile Response and ACT "
        "Gripper Behavior During a Successful Grasp"
    ),
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96
    ]
)

fig.savefig(
    OUT
    / "03_method2_successful_grasp_timeline.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 4
# METHOD 2 — BILATERAL TACTILE RESPONSE
# ============================================================

wb = direct[
    (
        direct["time_s"]
        >= m2_tactile_start
    )
    & (
        direct["time_s"]
        <= m2_tactile_end
    )
].copy()

wb["relative_time_s"] = (
    wb["time_s"] - m2_tactile_start
)

total = (
    wb["eflesh_combined_total"]
    .to_numpy(
        dtype=float
    )
)

motor_share = (
    100.0
    * wb[
        "eflesh_on_motor_total"
    ].to_numpy(
        dtype=float
    )
    / total
)

opposite_share = (
    100.0 - motor_share
)

mean_motor = float(
    np.mean(
        motor_share
    )
)

min_motor = float(
    np.min(
        motor_share
    )
)

max_motor = float(
    np.max(
        motor_share
    )
)

fig, ax = plt.subplots(
    figsize=(10.6, 5.8)
)

ax.stackplot(
    wb["relative_time_s"],
    motor_share,
    opposite_share,
    labels=[
        "Motor-side response share",
        "Opposite-side response share"
    ],
    colors=[
        BLUE,
        ORANGE
    ],
    alpha=0.78
)

ax.axhline(
    50,
    color="white",
    linewidth=1.8,
    linestyle="--",
    alpha=0.95
)

ax.text(
    (
        float(
            wb["relative_time_s"].max()
        )
        * 0.98
    ),
    51.5,
    "50 / 50 reference",
    ha="right",
    va="bottom",
    color="white",
    fontsize=9.5,
    fontweight="bold"
)

ax.text(
    0.03,
    0.08,
    (
        f"Motor-side mean share: "
        f"{mean_motor:.1f}%\n"
        f"Observed range: "
        f"{min_motor:.1f}%–"
        f"{max_motor:.1f}%"
    ),
    transform=ax.transAxes,
    fontsize=10.5,
    bbox={
        "boxstyle":
            "round,pad=0.45",
        "facecolor":
            "white",
        "alpha":
            0.88
    }
)

ax.set_ylim(
    0,
    100
)

ax.set_xlim(
    float(
        wb["relative_time_s"].min()
    ),
    float(
        wb["relative_time_s"].max()
    )
)

ax.set_xlabel(
    "Time from high-tactile onset [s]"
)

ax.set_ylabel(
    (
        "Share of combined tactile "
        "response [%]"
    )
)

ax.set_title(
    (
        "Method 2: Bilateral Tactile Response "
        "Distribution During Sustained Grasp"
    ),
    fontsize=14,
    fontweight="bold"
)

ax.legend(
    loc="upper right"
)

ax.grid(
    axis="x",
    alpha=0.20
)

fig.text(
    0.5,
    0.012,
    (
        "Response share reflects eFlesh signal "
        "distribution and is not a calibrated force ratio."
    ),
    ha="center",
    fontsize=9.5
)

fig.tight_layout(
    rect=[
        0,
        0.035,
        1,
        1
    ]
)

fig.savefig(
    OUT
    / "04_method2_bilateral_tactile_balance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 5
# METHOD 1 VS METHOD 2 — CONTROL PHILOSOPHY
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.2, 5.8)
)


# ---------------------------
# Method 1
# ---------------------------

ax = axes[0]

ax.plot(
    m1_2s["relative_time_s"],
    m1_2s["act_gripper_command"],
    color=BLUE,
    linewidth=2.4,
    label="ACT requested"
)

ax.plot(
    m1_2s["relative_time_s"],
    m1_2s["sent_gripper_command"],
    color=ORANGE,
    linewidth=2.4,
    linestyle="--",
    label="Sent command"
)

ax.fill_between(
    m1_2s["relative_time_s"],
    m1_2s["act_gripper_command"],
    m1_2s["sent_gripper_command"],
    color=ORANGE,
    alpha=0.22,
    label="External intervention"
)

ax.set_xlabel(
    "Time after detected contact [s]"
)

ax.set_ylabel(
    "Gripper command [SO101 units]"
)

ax.set_title(
    "Method 1 — External Safety Layer",
    fontweight="bold"
)

ax.grid(
    alpha=0.22
)

ax.legend(
    loc="upper right"
)

ax.text(
    0.04,
    0.08,
    (
        f"Modified samples: "
        f"{m1_fraction*100:.0f}%\n"
        f"Mean modification: "
        f"{m1_mean:.2f}\n"
        f"Peak modification: "
        f"{m1_max:.2f}"
    ),
    transform=ax.transAxes,
    fontsize=10,
    bbox={
        "boxstyle":
            "round,pad=0.4",
        "facecolor":
            "white",
        "alpha":
            0.90
    }
)


# ---------------------------
# Method 2
# ---------------------------

ax = axes[1]

ax.plot(
    m2_2s["relative_time_s"],
    m2_2s[
        "policy_cmd.gripper.pos"
    ],
    color=BLUE,
    linewidth=2.4,
    label="ACT requested"
)

ax.plot(
    m2_2s["relative_time_s"],
    m2_2s[
        "sent_cmd.gripper.pos"
    ],
    color=ORANGE,
    linewidth=2.2,
    linestyle="--",
    label="Sent command"
)

ax.set_xlabel(
    "Time after high-tactile onset [s]"
)

ax.set_ylabel(
    "Gripper command [SO101 units]"
)

ax.set_title(
    "Method 2 — Tactile-Integrated ACT",
    fontweight="bold"
)

ax.grid(
    alpha=0.22
)

# Secondary axis for tactile signal
ax_tactile = ax.twinx()

normalized_tactile = (
    100.0
    * m2_2s[
        "eflesh_combined_total"
    ].to_numpy()
    / peak_eflesh
)

ax_tactile.fill_between(
    m2_2s["relative_time_s"],
    0,
    normalized_tactile,
    color=GREEN,
    alpha=0.11,
    label=(
        "Tactile response "
        "(% of run peak)"
    )
)

ax_tactile.set_ylim(
    0,
    120
)

ax_tactile.set_ylabel(
    (
        "Combined tactile response "
        "[% of run peak]"
    ),
    color=GREEN
)

ax_tactile.tick_params(
    axis="y",
    labelcolor=GREEN
)

lines1, labels1 = (
    ax.get_legend_handles_labels()
)

lines2, labels2 = (
    ax_tactile
    .get_legend_handles_labels()
)

ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right"
)

ax.text(
    0.04,
    0.08,
    (
        f"Modified samples: "
        f"{m2_fraction*100:.0f}%\n"
        f"Mean modification: "
        f"{m2_mean:.2f}\n"
        "Tactile signal active "
        "throughout window"
    ),
    transform=ax.transAxes,
    fontsize=10,
    bbox={
        "boxstyle":
            "round,pad=0.4",
        "facecolor":
            "white",
        "alpha":
            0.90
    }
)

fig.suptitle(
    (
        "Two Safety Strategies: External Override "
        "vs Tactile Information Inside the Policy"
    ),
    fontsize=15,
    fontweight="bold"
)

fig.text(
    0.5,
    0.012,
    (
        "Compare command divergence, not absolute "
        "gripper position; the two methods used "
        "different physical gripper configurations."
    ),
    ha="center",
    fontsize=9.5
)

fig.tight_layout(
    rect=[
        0,
        0.04,
        1,
        0.95
    ]
)

fig.savefig(
    OUT
    / "05_method1_vs_method2_control_strategy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 6
# METHOD 1 — CURRENT PROFILE WITH / WITHOUT LIMITER
# ============================================================

PRE_CURRENT = 0.5
POST_CURRENT = 2.0

baseline_current = baseline[
    (
        baseline["time_s"]
        >= baseline_reference_t
        - PRE_CURRENT
    )
    & (
        baseline["time_s"]
        <= baseline_reference_t
        + POST_CURRENT
    )
].copy()

limited_current = limited[
    (
        limited["time_s"]
        >= m1_contact_t
        - PRE_CURRENT
    )
    & (
        limited["time_s"]
        <= m1_contact_t
        + POST_CURRENT
    )
].copy()

baseline_current[
    "relative_time_s"
] = (
    baseline_current["time_s"]
    - baseline_reference_t
)

limited_current[
    "relative_time_s"
] = (
    limited_current["time_s"]
    - m1_contact_t
)

baseline_post = baseline_current[
    (
        baseline_current[
            "relative_time_s"
        ] >= 0
    )
    & (
        baseline_current[
            "relative_time_s"
        ] <= POST_CURRENT
    )
]

limited_post = limited_current[
    (
        limited_current[
            "relative_time_s"
        ] >= 0
    )
    & (
        limited_current[
            "relative_time_s"
        ] <= POST_CURRENT
    )
]

baseline_peak_current = float(
    baseline_post[
        "abs_current_raw"
    ].max()
)

limited_peak_current = float(
    limited_post[
        "abs_current_raw"
    ].max()
)

current_reduction = (
    100.0
    * (
        baseline_peak_current
        - limited_peak_current
    )
    / baseline_peak_current
)

fig, ax = plt.subplots(
    figsize=(10.6, 5.7)
)

ax.plot(
    baseline_current[
        "relative_time_s"
    ],
    baseline_current[
        "abs_current_raw"
    ],
    color=BLUE,
    linewidth=2.1,
    label="ACT without limiter"
)

ax.plot(
    limited_current[
        "relative_time_s"
    ],
    limited_current[
        "abs_current_raw"
    ],
    color=ORANGE,
    linewidth=2.1,
    label="ACT with limiter"
)

ax.axvline(
    0,
    color=RED,
    linestyle="--",
    linewidth=1.5
)

ax.axvspan(
    0,
    POST_CURRENT,
    color=GRAY,
    alpha=0.06
)

ax.set_xlabel(
    "Time from interaction reference [s]"
)

ax.set_ylabel(
    "|Gripper motor current| [raw]"
)

ax.set_title(
    (
        "Method 1: External Limiter Reduces "
        "the Post-Interaction Current Peak"
    ),
    fontsize=14,
    fontweight="bold"
)

ax.grid(
    alpha=0.22
)

ax.legend(
    loc="upper right"
)

ax.text(
    0.70,
    0.70,
    (
        f"Peak without limiter: "
        f"{baseline_peak_current:.0f} raw\n"
        f"Peak with limiter: "
        f"{limited_peak_current:.0f} raw\n"
        f"Reduction: "
        f"{current_reduction:.1f}%"
    ),
    transform=ax.transAxes,
    fontsize=10.5,
    bbox={
        "boxstyle":
            "round,pad=0.45",
        "facecolor":
            "white",
        "alpha":
            0.90
    }
)

fig.text(
    0.5,
    0.012,
    (
        "Baseline reference follows the previously "
        "used convention: estimated contact = "
        "largest low-position current event − 0.25 s."
    ),
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[
        0,
        0.035,
        1,
        1
    ]
)

fig.savefig(
    OUT
    / "06_method1_current_profile_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = f"""
EXPANDED RESULTS FIGURES
======================================

METHOD 1
--------
Detected contact time:
{m1_contact_t:.3f} s

First 2 s after contact:
Mean external modification:
{m1_mean:.3f} SO101 units

Maximum external modification:
{m1_max:.3f} SO101 units

Samples modified:
{m1_fraction * 100:.1f}%

Peak post-interaction current without limiter:
{baseline_peak_current:.0f} raw

Peak post-interaction current with limiter:
{limited_peak_current:.0f} raw

Peak-current reduction:
{current_reduction:.1f}%


METHOD 2
--------
Signal-based high-tactile onset:
{m2_tactile_start:.3f} s

Peak tactile response time:
{m2_peak_t:.3f} s

End of dominant high-tactile interval:
{m2_tactile_end:.3f} s

Peak combined eFlesh:
{peak_eflesh:.2f}

50% peak threshold:
{eflesh_threshold:.2f}

First 2 s after high-tactile onset:
Mean external modification:
{m2_mean:.3f} SO101 units

Maximum external modification:
{m2_max:.3f} SO101 units

Samples modified:
{m2_fraction * 100:.1f}%

Motor-side tactile response mean share:
{mean_motor:.1f}%

Motor-side tactile response range:
{min_motor:.1f}% to {max_motor:.1f}%


IMPORTANT INTERPRETATION
------------------------
Method 1 uses an external safety layer that modifies
the ACT gripper request after detected contact.

Method 2 places the eFlesh tactile measurements inside
the ACT observation. During the first 2 seconds of the
dominant high-tactile interval of this successful trial,
the policy-requested gripper command and sent gripper
command were identical.

The bilateral tactile-response percentages describe
sensor-response distribution only. They are not
calibrated force ratios.

The two methods used different physical gripper
configurations. Absolute gripper positions and raw
eFlesh magnitudes should therefore not be treated as
directly equivalent physical quantities.
""".strip()

(
    OUT
    / "expanded_results_summary.txt"
).write_text(
    summary
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("EXPANDED RESULTS FIGURES CREATED")
print("=" * 60)

print()
print("OUTPUT FOLDER:")
print(OUT)

print()
print("FIGURES:")

for path in sorted(
    OUT.glob("*.png")
):
    print(
        " ",
        path.name
    )

print()
print("METHOD 1:")
print(
    f"  Mean intervention: "
    f"{m1_mean:.3f}"
)
print(
    f"  Max intervention:  "
    f"{m1_max:.3f}"
)
print(
    f"  Modified samples:  "
    f"{m1_fraction*100:.1f}%"
)
print(
    f"  Current reduction: "
    f"{current_reduction:.1f}%"
)

print()
print("METHOD 2:")
print(
    f"  High tactile onset: "
    f"{m2_tactile_start:.3f} s"
)
print(
    f"  Peak tactile time:  "
    f"{m2_peak_t:.3f} s"
)
print(
    f"  High tactile end:   "
    f"{m2_tactile_end:.3f} s"
)
print(
    f"  Mean intervention:  "
    f"{m2_mean:.3f}"
)
print(
    f"  Max intervention:   "
    f"{m2_max:.3f}"
)
print(
    f"  Modified samples:   "
    f"{m2_fraction*100:.1f}%"
)

print()
print("BILATERAL TACTILE RESPONSE:")
print(
    f"  Motor-side mean: "
    f"{mean_motor:.1f}%"
)
print(
    f"  Range: "
    f"{min_motor:.1f}% - "
    f"{max_motor:.1f}%"
)

print()
print(
    "Summary:",
    OUT / "expanded_results_summary.txt"
)
