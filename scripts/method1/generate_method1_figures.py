from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Repository root:
# learning-from-demonstrations/
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "method1"
EF_PROCESSED = DATA / "calibration"

EF_FIG = ROOT / "figures" / "method1"
ACT_FIG = EF_FIG

REPORT = ROOT / "results"

EF_FIG.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD eFLESH FILES
# ============================================================

motor_path = EF_PROCESSED / "eflesh_motor_aligned.csv"
detail_path = EF_PROCESSED / "deformation_current_detailed.csv"
rigid_path = EF_PROCESSED / "rigid_aligned.csv"
soft_path = EF_PROCESSED / "soft_aligned.csv"

motor = pd.read_csv(motor_path)
detail = pd.read_csv(detail_path)

rigid = pd.read_csv(rigid_path) if rigid_path.exists() else None
soft = pd.read_csv(soft_path) if soft_path.exists() else None


# ============================================================
# FIGURE E1
# RIGID VS SOFT eFLESH RESPONSE
# ============================================================

if (
    rigid is not None
    and soft is not None
    and "signal" in rigid.columns
    and "signal" in soft.columns
    and "gripper_state" in rigid.columns
    and "gripper_state" in soft.columns
):

    plt.figure(figsize=(9, 6))

    plt.scatter(
        rigid["gripper_state"],
        rigid["signal"],
        s=10,
        alpha=0.35,
        label="Rigid reference"
    )

    plt.scatter(
        soft["gripper_state"],
        soft["signal"],
        s=10,
        alpha=0.35,
        label="Soft fish"
    )

    plt.xlabel("Measured gripper position [SO101 units]")
    plt.ylabel("Combined eFlesh response")
    plt.title("eFlesh Response: Rigid Reference vs Soft Fish")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        EF_FIG / "01_rigid_vs_soft_eflesh_response.png",
        dpi=300
    )
    plt.close()


# ============================================================
# FIGURE E2
# eFLESH SIGNAL VS MOTOR CURRENT
# ============================================================

r = motor["combined_total"].corr(motor["abs_current"])

plt.figure(figsize=(8, 6))

plt.scatter(
    motor["combined_total"],
    motor["abs_current"],
    s=12,
    alpha=0.45
)

plt.xlabel("Combined eFlesh response")
plt.ylabel("Absolute gripper motor current [raw]")
plt.title("Relationship Between eFlesh Response and Motor Current")
plt.grid(True, alpha=0.3)

plt.text(
    0.03,
    0.95,
    f"Pearson r = {r:.3f}",
    transform=plt.gca().transAxes,
    va="top",
    bbox=dict(boxstyle="round", alpha=0.15)
)

plt.tight_layout()

plt.savefig(
    EF_FIG / "02_eflesh_vs_motor_current.png",
    dpi=300
)
plt.close()


# ============================================================
# FIGURE E3
# POST-CONTACT CLOSURE VS eFLESH
# ============================================================

plt.figure(figsize=(9, 6))

for squeeze, g in detail.groupby("squeeze"):
    plt.scatter(
        g["actual_extra_closure"],
        g["eflesh_smooth"],
        s=18,
        alpha=0.6,
        label=f"Squeeze {int(squeeze)}"
    )

plt.xlabel("Additional gripper closure after contact [SO101 units]")
plt.ylabel("Combined eFlesh response")
plt.title("eFlesh Response vs Post-Contact Gripper Closure")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    EF_FIG / "03_post_contact_closure_vs_eflesh.png",
    dpi=300
)
plt.close()


# ============================================================
# PER-SQUEEZE SUMMARY
# ============================================================

rows = []

for squeeze, g in detail.groupby("squeeze"):

    g = g.sort_values("time_s")

    rows.append({
        "squeeze": int(squeeze),
        "contact_position": g.iloc[0]["position_smooth"],
        "minimum_position": g["position_smooth"].min(),
        "maximum_post_contact_closure":
            g["actual_extra_closure"].max(),
        "maximum_eflesh":
            g["eflesh_smooth"].max(),
        "maximum_current_raw":
            g["abs_current"].max(),
        "median_current_raw":
            g["abs_current"].median(),
    })

squeeze_summary = pd.DataFrame(rows)

squeeze_summary.to_csv(
    REPORT / "eflesh_per_squeeze_summary.csv",
    index=False
)


# ============================================================
# FIND ACT FILES
# ============================================================

BASELINE = DATA / "act_no_limiter.csv"
LIMITED = DATA / "act_with_safety_limiter.csv"

if not BASELINE.exists():
    raise FileNotFoundError(f"Missing Method 1 baseline telemetry: {BASELINE}")

if not LIMITED.exists():
    raise FileNotFoundError(f"Missing Method 1 limited telemetry: {LIMITED}")

b = pd.read_csv(BASELINE)
l = pd.read_csv(LIMITED)

print("\nACT files used:")
print("Baseline:", BASELINE)
print("Limited :", LIMITED)


# ============================================================
# FIND BASELINE CONTACT
# ============================================================

low = b[b["gripper_position"] <= 15].copy()

peak_idx = low["abs_current_raw"].idxmax()
peak_time = b.loc[peak_idx, "time_s"]

# Search backwards from compression peak for first meaningful
# low-position motor-current response.
pre = b[
    (b["time_s"] >= peak_time - 0.8) &
    (b["time_s"] <= peak_time) &
    (b["gripper_position"] <= 15) &
    (b["abs_current_raw"] >= 5)
]

if len(pre):
    tb = pre.iloc[0]["time_s"]
else:
    tb = peak_time - 0.25


# ============================================================
# LIMITED CONTACT IS EXPLICITLY LOGGED
# ============================================================

contact_rows = l[l["contact_detected"] == 1]

if contact_rows.empty:
    raise RuntimeError(
        "No contact_detected event found in V3 telemetry."
    )

tl = contact_rows.iloc[0]["time_s"]

b["grasp_time"] = b["time_s"] - tb
l["grasp_time"] = l["time_s"] - tl


# ============================================================
# WINDOWS
# ============================================================

bw_position = b[
    (b["grasp_time"] >= -0.5) &
    (b["grasp_time"] <= 2.0)
].copy()

lw_position = l[
    (l["grasp_time"] >= -0.5) &
    (l["grasp_time"] <= 2.0)
].copy()

# Current comparison intentionally starts at contact.
# Pre-contact current is not part of the deformation limiter.
bw_current = b[
    (b["grasp_time"] >= 0.0) &
    (b["grasp_time"] <= 2.0)
].copy()

lw_current = l[
    (l["grasp_time"] >= 0.0) &
    (l["grasp_time"] <= 2.0)
].copy()


# ============================================================
# FIGURE A1
# POST-CONTACT CURRENT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    bw_current["grasp_time"],
    bw_current["abs_current_raw"],
    linewidth=2,
    label="ACT without limiter"
)

plt.plot(
    lw_current["grasp_time"],
    lw_current["abs_current_raw"],
    linewidth=2,
    label="ACT with limiter"
)

plt.axhline(
    9,
    linestyle="--",
    linewidth=1.5,
    label="Bare-gripper safety reference (9 raw)"
)

plt.xlabel("Time after contact [s]")
plt.ylabel("Absolute gripper motor current [raw]")
plt.title("Post-Contact Motor Current During Fish Grasp")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    ACT_FIG / "04_post_contact_current_comparison.png",
    dpi=300
)
plt.close()


# ============================================================
# FIGURE A2
# MEASURED POSITION
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    bw_position["grasp_time"],
    bw_position["gripper_position"],
    linewidth=2,
    label="ACT without limiter"
)

plt.plot(
    lw_position["grasp_time"],
    lw_position["gripper_position"],
    linewidth=2,
    label="ACT with limiter"
)

plt.axvline(
    0,
    linestyle=":",
    linewidth=1.5,
    label="Contact"
)

plt.xlabel("Time relative to contact [s]")
plt.ylabel("Measured gripper position [SO101 units]")
plt.title("Gripper Closure During Fish Grasp")

plt.text(
    0.02,
    0.04,
    "Lower SO101 position = greater gripper closure",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round", alpha=0.15)
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    ACT_FIG / "05_gripper_closure_comparison.png",
    dpi=300
)
plt.close()


# ============================================================
# FIGURE A3
# ACT REQUEST VS SAFETY COMMAND
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    lw_position["grasp_time"],
    lw_position["act_gripper_command"],
    linewidth=2,
    label="ACT requested"
)

plt.plot(
    lw_position["grasp_time"],
    lw_position["sent_gripper_command"],
    linewidth=2,
    label="Command after safety limiter"
)

plt.plot(
    lw_position["grasp_time"],
    lw_position["gripper_position"],
    linewidth=2,
    label="Measured position"
)

plt.axvline(
    0,
    linestyle=":",
    linewidth=1.5,
    label="Contact detected"
)

plt.xlabel("Time relative to contact [s]")
plt.ylabel("Gripper position [SO101 units]")
plt.title("Effect of Safety Layer on ACT Gripper Command")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    ACT_FIG / "06_safety_layer_command_effect.png",
    dpi=300
)
plt.close()


# ============================================================
# SUMMARY NUMBERS
# ============================================================

cp = l["contact_position"].dropna()

summary = {
    "eflesh_current_correlation_r": r,

    "baseline_peak_post_contact_current":
        bw_current["abs_current_raw"].max(),

    "limited_peak_post_contact_current":
        lw_current["abs_current_raw"].max(),

    "baseline_minimum_gripper_position":
        bw_position["gripper_position"].min(),

    "limited_minimum_gripper_position":
        lw_position["gripper_position"].min(),

    "limited_contact_position":
        cp.iloc[0] if len(cp) else np.nan,

    "limited_max_post_contact_closure":
        lw_position["post_contact_closure"].max()
        if "post_contact_closure" in lw_position.columns
        else np.nan,
}

pd.DataFrame(
    list(summary.items()),
    columns=["metric", "value"]
).to_csv(
    REPORT / "final_quantitative_summary.csv",
    index=False
)


# Figures are written directly to figures/method1/.
# Numerical summaries are written to results/.


print("\n====================================================")
print("METHOD 1 FIGURES GENERATED")
print("====================================================")

print(f"\neFlesh/current correlation r = {r:.3f}")

print(
    "Baseline peak post-contact current =",
    summary["baseline_peak_post_contact_current"]
)

print(
    "Limited peak post-contact current =",
    summary["limited_peak_post_contact_current"]
)

print(
    "Baseline minimum position =",
    summary["baseline_minimum_gripper_position"]
)

print(
    "Limited minimum position =",
    summary["limited_minimum_gripper_position"]
)

print("\nGenerated result files:")
for p in sorted(REPORT.iterdir()):
    print(" ", p.name)
