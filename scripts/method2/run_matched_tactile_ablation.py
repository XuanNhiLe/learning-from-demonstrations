from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


# ============================================================
# PATHS AND PUBLIC MODEL/DATASET REFERENCES
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PAIRS_PATH = (
    ROOT / "data" / "method2" / "matched_tactile_pairs.csv"
)

OUT_DIR = (
    ROOT / "results" / "method2_ablation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_CSV = (
    OUT_DIR
    / "matched_tactile_ablation.csv"
)

DATASET_REPO_ID = (
    "Cookieman12/grab_fish_eflesh_act_v1"
)

MODEL_REPO_ID = (
    "Cookieman12/grab_fish_eflesh_act_v1_50k"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 72)
print("MATCHED TACTILE-INPUT SENSITIVITY ANALYSIS")
print("=" * 72)

print()
print("Device:", device)


# ============================================================
# LOAD MATCHED PAIRS
# ============================================================

pairs = pd.read_csv(
    PAIRS_PATH
)

if len(pairs) != 50:
    print(
        f"WARNING: expected 50 pairs, found {len(pairs)}"
    )

print(
    f"Matched pairs loaded: {len(pairs)}"
)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading dataset...")

dataset = LeRobotDataset(
    repo_id=DATASET_REPO_ID,
)

print(
    "Dataset length:",
    len(dataset)
)


# ============================================================
# LOAD ACT
# ============================================================

print()
print("Loading trained ACT policy...")

policy = ACTPolicy.from_pretrained(
    MODEL_REPO_ID
)

policy.to(
    device
)

policy.eval()


# ============================================================
# PRE / POST PROCESSORS
# ============================================================

preprocessor, postprocessor = (
    make_pre_post_processors(
        policy.config,
        dataset_stats=dataset.meta.stats,
    )
)


# ============================================================
# ACTION NAMES
# ============================================================

ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def predict_action(
    target_sample,
    state
):

    obs = {
        "observation.state":
            state.clone()
            .float()
            .unsqueeze(0),

        "observation.images.side":
            target_sample[
                "observation.images.side"
            ]
            .clone()
            .float()
            .unsqueeze(0),

        "observation.images.workspace":
            target_sample[
                "observation.images.workspace"
            ]
            .clone()
            .float()
            .unsqueeze(0),
    }

    obs = preprocessor(
        obs
    )

    for key, value in obs.items():

        if isinstance(
            value,
            torch.Tensor
        ):
            obs[key] = value.to(
                device
            )

    # ACT has an internal action queue.
    # Every condition must begin from exactly the same state.
    policy.reset()

    with torch.inference_mode():

        action = policy.select_action(
            obs
        )

        action = postprocessor(
            action
        )

    if isinstance(
        action,
        torch.Tensor
    ):
        action = (
            action
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        action,
        dtype=np.float64
    ).reshape(-1)


# ============================================================
# RUN ALL 50 MATCHED PAIRS
# ============================================================

results = []

print()
print("Running paired ACT inference...")
print()

for row_number, row in pairs.iterrows():

    target_index = int(
        row["target_dataset_index"]
    )

    replacement_index = int(
        row["replacement_dataset_index"]
    )

    # --------------------------------------------------------
    # Load target observation
    #
    # This provides the camera images and exact target state.
    # --------------------------------------------------------

    target_sample = dataset[
        target_index
    ]

    # --------------------------------------------------------
    # Load replacement observation
    #
    # ONLY its tactile vector [6:36] will be used.
    # Its images and robot joints are discarded.
    # --------------------------------------------------------

    replacement_sample = dataset[
        replacement_index
    ]

    target_state = (
        target_sample[
            "observation.state"
        ]
        .clone()
        .float()
    )

    replacement_state = (
        replacement_sample[
            "observation.state"
        ]
        .clone()
        .float()
    )

    if tuple(
        target_state.shape
    ) != (36,):

        raise RuntimeError(
            f"Target state has wrong shape "
            f"at dataset index {target_index}: "
            f"{tuple(target_state.shape)}"
        )

    if tuple(
        replacement_state.shape
    ) != (36,):

        raise RuntimeError(
            f"Replacement state has wrong shape "
            f"at dataset index {replacement_index}: "
            f"{tuple(replacement_state.shape)}"
        )

    # --------------------------------------------------------
    # CONDITION 1 — NORMAL
    #
    # Full real target observation.
    # --------------------------------------------------------

    normal_state = (
        target_state.clone()
    )

    # --------------------------------------------------------
    # CONDITION 2 — MATCHED LOW TACTILE
    #
    # Keep EXACTLY the target robot configuration [0:6].
    # Replace ONLY tactile channels [6:36].
    # --------------------------------------------------------

    low_tactile_state = (
        target_state.clone()
    )

    low_tactile_state[
        6:36
    ] = replacement_state[
        6:36
    ]

    # --------------------------------------------------------
    # HARD VERIFICATION
    # --------------------------------------------------------

    if not torch.equal(
        normal_state[:6],
        low_tactile_state[:6]
    ):
        raise RuntimeError(
            "Robot state changed during ablation."
        )

    # --------------------------------------------------------
    # ACT PREDICTIONS
    # --------------------------------------------------------

    normal_action = predict_action(
        target_sample,
        normal_state
    )

    low_action = predict_action(
        target_sample,
        low_tactile_state
    )

    if len(normal_action) != 6:
        raise RuntimeError(
            f"Expected 6-D action, got "
            f"{normal_action.shape}"
        )

    delta_signed = (
        normal_action
        - low_action
    )

    delta_abs = np.abs(
        delta_signed
    )

    # --------------------------------------------------------
    # SAVE ROW
    # --------------------------------------------------------

    result = {
        "target_episode":
            int(row["target_episode"]),

        "target_dataset_index":
            target_index,

        "replacement_episode":
            int(row["replacement_episode"]),

        "replacement_dataset_index":
            replacement_index,

        "target_tactile_score":
            float(
                row["target_tactile_score"]
            ),

        "replacement_tactile_score":
            float(
                row[
                    "replacement_tactile_score"
                ]
            ),

        "tactile_ratio":
            float(
                row["target_tactile_score"]
                /
                max(
                    row[
                        "replacement_tactile_score"
                    ],
                    1e-9
                )
            ),

        "target_gripper_state":
            float(
                target_state[5].item()
            ),

        "replacement_source_gripper_state":
            float(
                replacement_state[5].item()
            ),

        "matched_gripper_difference":
            float(
                row["abs_gripper_diff"]
            ),

        "other5_normalized_distance":
            float(
                row[
                    "other5_normalized_distance"
                ]
            ),
    }

    for i, name in enumerate(
        ACTION_NAMES
    ):

        result[
            f"normal_{name}"
        ] = float(
            normal_action[i]
        )

        result[
            f"low_tactile_{name}"
        ] = float(
            low_action[i]
        )

        result[
            f"signed_delta_{name}"
        ] = float(
            delta_signed[i]
        )

        result[
            f"abs_delta_{name}"
        ] = float(
            delta_abs[i]
        )

    results.append(
        result
    )

    print(
        f"[{row_number + 1:02d}/{len(pairs):02d}] "
        f"Episode {int(row['target_episode']):02d} | "
        f"Tactile "
        f"{row['target_tactile_score']:.0f}"
        f" -> "
        f"{row['replacement_tactile_score']:.0f} | "
        f"Gripper prediction "
        f"{normal_action[5]:.3f}"
        f" -> "
        f"{low_action[5]:.3f} | "
        f"delta = "
        f"{delta_signed[5]:+.3f}"
    )


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results = pd.DataFrame(
    results
)

results.to_csv(
    OUT_CSV,
    index=False
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

gripper_signed = (
    results[
        "signed_delta_gripper"
    ].to_numpy()
)

gripper_abs = (
    results[
        "abs_delta_gripper"
    ].to_numpy()
)

# IMPORTANT SO101 convention for this project:
#
# larger command = more open
# smaller command = more closed
#
# signed_delta = normal - low_tactile
#
# positive:
#   real tactile caused MORE OPEN / LESS CLOSED action
#
# negative:
#   real tactile caused MORE CLOSED action

n_less_closure = int(
    np.sum(
        gripper_signed > 0
    )
)

n_more_closure = int(
    np.sum(
        gripper_signed < 0
    )
)

n_negligible = int(
    np.sum(
        np.isclose(
            gripper_signed,
            0,
            atol=1e-6
        )
    )
)


# ============================================================
# CORRELATION WITH TACTILE CONTRAST
# ============================================================

if len(results) >= 3:

    corr_abs = np.corrcoef(
        results[
            "tactile_ratio"
        ],
        results[
            "abs_delta_gripper"
        ]
    )[0, 1]

else:
    corr_abs = np.nan


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 72)
print("MATCHED TACTILE ABLATION COMPLETE")
print("=" * 72)

print()
print("NUMBER OF TESTED OBSERVATIONS:")
print(
    f"  {len(results)}"
)

print()
print("TACTILE CONTRAST:")
print(
    f"  Median target/replacement ratio: "
    f"{results['tactile_ratio'].median():.2f}x"
)

print()
print("GRIPPER PREDICTION SENSITIVITY:")
print(
    f"  Mean |change|:   "
    f"{gripper_abs.mean():.3f} SO101 units"
)

print(
    f"  Median |change|: "
    f"{np.median(gripper_abs):.3f} SO101 units"
)

print(
    f"  90th percentile: "
    f"{np.percentile(gripper_abs, 90):.3f} SO101 units"
)

print(
    f"  Maximum |change|:"
    f" {gripper_abs.max():.3f} SO101 units"
)

print()
print("DIRECTION OF GRIPPER EFFECT:")
print(
    f"  Real tactile -> LESS closure / more open: "
    f"{n_less_closure}/{len(results)}"
)

print(
    f"  Real tactile -> MORE closure:             "
    f"{n_more_closure}/{len(results)}"
)

print(
    f"  Essentially unchanged:                    "
    f"{n_negligible}/{len(results)}"
)

print()
print(
    "Mean signed gripper delta "
    "(normal - matched-low):"
)

print(
    f"  {gripper_signed.mean():+.3f} "
    "SO101 units"
)

print()
print(
    "Correlation between tactile contrast "
    "and |gripper prediction change|:"
)

print(
    f"  r = {corr_abs:.3f}"
)

print()
print("MEAN ABSOLUTE ACTION CHANGE BY JOINT:")

for name in ACTION_NAMES:

    value = results[
        f"abs_delta_{name}"
    ].mean()

    print(
        f"  {name:15s}: "
        f"{value:.3f}"
    )


# ============================================================
# BIGGEST GRIPPER EFFECTS
# ============================================================

print()
print("10 LARGEST GRIPPER PREDICTION CHANGES:")
print()

display_cols = [
    "target_episode",
    "target_tactile_score",
    "replacement_tactile_score",
    "normal_gripper",
    "low_tactile_gripper",
    "signed_delta_gripper",
    "abs_delta_gripper",
]

print(
    results.sort_values(
        "abs_delta_gripper",
        ascending=False
    )[
        display_cols
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

SUMMARY_PATH = (
    OUT_DIR
    / "matched_tactile_ablation_summary.txt"
)

summary = f"""
MATCHED TACTILE-INPUT SENSITIVITY ANALYSIS
===========================================

Number of observations:
{len(results)}

Median target / replacement tactile ratio:
{results['tactile_ratio'].median():.3f}x

Gripper prediction sensitivity
------------------------------
Mean absolute change:
{gripper_abs.mean():.4f} SO101 units

Median absolute change:
{np.median(gripper_abs):.4f} SO101 units

90th percentile:
{np.percentile(gripper_abs, 90):.4f} SO101 units

Maximum:
{gripper_abs.max():.4f} SO101 units

Direction
---------
Real tactile produced less closure / more open:
{n_less_closure}/{len(results)}

Real tactile produced more closure:
{n_more_closure}/{len(results)}

Essentially unchanged:
{n_negligible}/{len(results)}

Mean signed delta (normal - matched-low):
{gripper_signed.mean():+.4f} SO101 units

Correlation between tactile contrast and
absolute gripper prediction change:
r = {corr_abs:.4f}

INTERPRETATION LIMIT
--------------------
This is an offline tactile-input sensitivity analysis.

For each target observation, the camera images and exact
6-D robot state were held fixed. Only the 30 tactile
channels were replaced using a real low-tactile vector
selected from the training dataset with a tightly matched
gripper configuration.

A change in predicted action therefore demonstrates
sensitivity of the trained ACT policy to the tactile input.

It does NOT by itself demonstrate reduced physical
deformation, improved grasp safety, or superior closed-loop
robot performance.
""".strip()

SUMMARY_PATH.write_text(
    summary
)

print()
print("RAW RESULTS SAVED:")
print(OUT_CSV)

print()
print("SUMMARY SAVED:")
print(SUMMARY_PATH)

print()
print("=" * 72)
