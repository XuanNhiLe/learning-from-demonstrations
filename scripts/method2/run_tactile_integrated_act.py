import os
import sys
import csv
import time
import math
import threading
from datetime import datetime
from pathlib import Path

# Keep Hugging Face downloads off the small root drive.
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
OUT_DIR = Path(os.environ.get("ACT_EFLESH_OUT_DIR", "results/method2_rollouts"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_PATH = OUT_DIR / f"act_eflesh_trial_{stamp}.csv"

STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

for prefix in ("eflesh_on_motor", "eflesh_not_motor"):
    for mag in range(5):
        for axis in ("bx", "by", "bz"):
            STATE_KEYS.append(f"{prefix}.m{mag}.{axis}")

ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# ---------------------------------------------------------------------
# Monkey-patch the SOFollower only for this run.
# The normal LeRobot rollout/policy code is otherwise unchanged.
# ---------------------------------------------------------------------
from lerobot.robots.so_follower.so_follower import SOFollower

_original_get_observation = SOFollower.get_observation
_original_send_action = SOFollower.send_action

_lock = threading.Lock()
_csv_file = None
_writer = None
_last_obs = None
_t0 = None
_rows = 0


def eflesh_total(obs, prefix):
    total = 0.0

    for mag in range(5):
        bx = float(obs[f"{prefix}.m{mag}.bx"])
        by = float(obs[f"{prefix}.m{mag}.by"])
        bz = float(obs[f"{prefix}.m{mag}.bz"])

        total += math.sqrt(bx * bx + by * by + bz * bz)

    return total


def logged_get_observation(self):
    global _last_obs

    obs = _original_get_observation(self)

    # Copy only the 36 numeric policy-state features.
    if all(k in obs for k in STATE_KEYS):
        with _lock:
            _last_obs = {k: float(obs[k]) for k in STATE_KEYS}

    return obs


def logged_send_action(self, action):
    global _csv_file, _writer, _t0, _rows

    # This is what ACT requested before SO101 safety clipping.
    policy_action = {
        k: float(action[k]) if k in action else float("nan")
        for k in ACTION_KEYS
    }

    # Let the original SO101 code actually send the command.
    sent_action = _original_send_action(self, action)

    # This is what was actually sent after max_relative_target clipping.
    sent_action_values = {
        k: float(sent_action[k]) if k in sent_action else float("nan")
        for k in ACTION_KEYS
    }

    with _lock:
        obs = dict(_last_obs) if _last_obs is not None else None

    if obs is not None:
        if _csv_file is None:
            _csv_file = open(LOG_PATH, "w", newline="")
            _writer = csv.writer(_csv_file)

            header = (
                ["time_s"]
                + STATE_KEYS
                + [f"policy_cmd.{k}" for k in ACTION_KEYS]
                + [f"sent_cmd.{k}" for k in ACTION_KEYS]
                + [
                    "eflesh_on_motor_total",
                    "eflesh_not_motor_total",
                    "eflesh_combined_total",
                    "eflesh_asymmetry",
                ]
            )

            _writer.writerow(header)
            _csv_file.flush()

        now = time.perf_counter()

        if _t0 is None:
            _t0 = now

        motor_total = eflesh_total(obs, "eflesh_on_motor")
        other_total = eflesh_total(obs, "eflesh_not_motor")
        combined = motor_total + other_total
        asymmetry = motor_total - other_total

        row = (
            [now - _t0]
            + [obs[k] for k in STATE_KEYS]
            + [policy_action[k] for k in ACTION_KEYS]
            + [sent_action_values[k] for k in ACTION_KEYS]
            + [
                motor_total,
                other_total,
                combined,
                asymmetry,
            ]
        )

        _writer.writerow(row)
        _rows += 1

        # Flush approximately once per second.
        if _rows % 30 == 0:
            _csv_file.flush()

    return sent_action


SOFollower.get_observation = logged_get_observation
SOFollower.send_action = logged_send_action


def close_log():
    global _csv_file

    if _csv_file is not None:
        _csv_file.flush()
        _csv_file.close()
        _csv_file = None


# ---------------------------------------------------------------------
# Run the SAME trained ACT policy/setup that already succeeded.
# ---------------------------------------------------------------------
sys.argv = [
    "lerobot-rollout",
    "--strategy.type=base",
    "--policy.path=Cookieman12/grab_fish_eflesh_act_v1_50k",
    "--robot.type=so101_follower",
    "--robot.port=/dev/ttyACM0",
    "--robot.id=my_awesome_follower_arm",
    "--robot.use_degrees=false",
    "--robot.max_relative_target=5",
    "--robot.cameras={side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, workspace: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}",
    "--device=cuda",
    "--fps=30",
    '--task=Grab the fish smoothly',
    "--duration=30",
    "--return_to_initial_position=false",
]

from lerobot.scripts.lerobot_rollout import main

try:
    print()
    print("==============================================")
    print("ACT + eFlesh LOGGED TRIAL")
    print("CSV will be saved to:")
    print(LOG_PATH)
    print("==============================================")
    print()

    main()

finally:
    close_log()

print()
print("==============================================")
print("TRIAL FINISHED")
print("Rows saved:", _rows)
print("CSV:", LOG_PATH)
print("==============================================")

# Add outcome directly to filename for later analysis.
try:
    result = input("Was the grasp successful? [y/n]: ").strip().lower()

    if result in ("y", "yes"):
        new_path = LOG_PATH.with_name(LOG_PATH.stem + "_SUCCESS.csv")
        LOG_PATH.rename(new_path)
        print("Saved as:", new_path)

    elif result in ("n", "no"):
        new_path = LOG_PATH.with_name(LOG_PATH.stem + "_FAIL.csv")
        LOG_PATH.rename(new_path)
        print("Saved as:", new_path)

    else:
        print("Outcome not labelled. File remains:", LOG_PATH)

except Exception as exc:
    print("Could not label outcome:", exc)
