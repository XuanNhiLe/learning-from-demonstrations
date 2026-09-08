import csv
import sys
import time
from datetime import datetime
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower
from lerobot.scripts.lerobot_record import main


# ============================================================
# PARAMETERS
# ============================================================

# Bare-gripper current calibration
CURRENT_LIMIT = 9.0
CURRENT_RELEASE = 6.0

# Bare-gripper region where contact can realistically occur.
# This is only a SEARCH WINDOW, not the deformation criterion.
CONTACT_SEARCH_POSITION = 12.0
CONTACT_CURRENT = 6.0

# eFlesh principle:
# deformation is additional closure AFTER contact.
# Start conservatively with 1 position unit.
MAX_POST_CONTACT_CLOSURE = 1.0

OPEN_THRESHOLD = 50.0
CLOSING_MARGIN = 0.5
RELEASE_MARGIN = 5.0


# ============================================================
# LOGGING
# ============================================================

STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CSV_PATH = Path(f"act_eflesh_limiter_v3_{STAMP}.csv")

f = open(CSV_PATH, "w", newline="", buffering=1)
writer = csv.writer(f)

writer.writerow([
    "time_s",
    "unix_time",
    "act_gripper_command",
    "sent_gripper_command",
    "gripper_position",
    "current_raw",
    "abs_current_raw",
    "system_armed",
    "contact_detected",
    "contact_position",
    "post_contact_closure",
    "safe_boundary_position",
    "current_relief",
    "position_limit_active",
])


# ============================================================
# STATE
# ============================================================

_original_send_action = SO101Follower.send_action

t0 = time.perf_counter()

system_armed = False
contact_detected = False
contact_position = None

current_relief = False


def scalar(x):
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def read_gripper(self, register, normalize=True):
    try:
        x = self.bus.read(
            register,
            "gripper",
            normalize=normalize,
        )
    except TypeError:
        x = self.bus.read(
            register,
            "gripper",
        )
    return scalar(x)


def safe_send_action(self, action):

    global system_armed
    global contact_detected
    global contact_position
    global current_relief

    corrected = dict(action)

    desired = float(action["gripper.pos"])

    position = read_gripper(
        self,
        "Present_Position",
        normalize=True,
    )

    current = read_gripper(
        self,
        "Present_Current",
        normalize=False,
    )

    abs_current = abs(current)

    # Smaller position = more closed.
    closing = desired < (position - CLOSING_MARGIN)


    # ========================================================
    # ARM ONLY AFTER GRIPPER HAS OPENED
    # ========================================================

    if not system_armed:
        if position >= OPEN_THRESHOLD or desired >= OPEN_THRESHOLD:
            system_armed = True

            print("\n========================================")
            print(" SAFETY SYSTEM ARMED")
            print("========================================\n")


    # ========================================================
    # DETECT CONTACT ON CURRENT HARDWARE
    # ========================================================

    if (
        system_armed
        and not contact_detected
        and closing
        and position <= CONTACT_SEARCH_POSITION
        and abs_current >= CONTACT_CURRENT
    ):
        contact_detected = True
        contact_position = position

        print("\n========================================")
        print(" CONTACT DETECTED")
        print("========================================")
        print(f"Contact position : {contact_position:.2f}")
        print(f"Current          : {abs_current:.1f} raw")
        print(f"ACT command      : {desired:.2f}")
        print("========================================\n")


    safe_boundary = None
    post_contact_closure = 0.0
    position_limit_active = False


    # ========================================================
    # eFLESH-DERIVED POSITION PRINCIPLE
    #
    # q_contact becomes zero reference.
    # Prevent excessive additional closure after contact.
    # ========================================================

    if contact_detected:

        post_contact_closure = max(
            0.0,
            contact_position - position
        )

        safe_boundary = (
            contact_position
            - MAX_POST_CONTACT_CLOSURE
        )


        # ----------------------------------------------------
        # CURRENT HYSTERESIS
        # ----------------------------------------------------

        if abs_current >= CURRENT_LIMIT:
            if not current_relief:
                print("\n========================================")
                print(" CURRENT LIMIT REACHED")
                print("========================================")
                print(f"Current          : {abs_current:.1f}")
                print(f"Limit            : {CURRENT_LIMIT:.1f}")
                print(f"Contact position : {contact_position:.2f}")
                print(f"Actual position  : {position:.2f}")
                print(
                    f"Extra closure    : "
                    f"{post_contact_closure:.2f}"
                )
                print("Opening back toward contact.")
                print("========================================\n")

            current_relief = True

        elif current_relief and abs_current <= CURRENT_RELEASE:
            current_relief = False


        # ----------------------------------------------------
        # POSITION CAP
        # ----------------------------------------------------

        # ACT may not command farther closed than the
        # allowed post-contact position.
        if desired < safe_boundary:
            corrected["gripper.pos"] = safe_boundary
            position_limit_active = True


        # ----------------------------------------------------
        # CURRENT EMERGENCY RELIEF
        # ----------------------------------------------------

        # If current becomes excessive, don't freeze where
        # deformation already occurred. Open back to the
        # original contact position.
        if current_relief:
            corrected["gripper.pos"] = contact_position


        # ----------------------------------------------------
        # RESET WHEN ACT REALLY WANTS TO OPEN
        # ----------------------------------------------------

        if desired > contact_position + RELEASE_MARGIN:

            print("\n========================================")
            print(" GRASP SAFETY RESET")
            print("========================================\n")

            contact_detected = False
            contact_position = None
            current_relief = False

            safe_boundary = None
            post_contact_closure = 0.0
            position_limit_active = False


    # ========================================================
    # SEND ACTION
    # ========================================================

    result = _original_send_action(
        self,
        corrected,
    )


    # ========================================================
    # LOG
    # ========================================================

    writer.writerow([
        time.perf_counter() - t0,
        time.time(),
        desired,
        float(corrected["gripper.pos"]),
        position,
        current,
        abs_current,
        int(system_armed),
        int(contact_detected),

        "" if contact_position is None else contact_position,

        post_contact_closure,

        "" if safe_boundary is None else safe_boundary,

        int(current_relief),
        int(position_limit_active),
    ])

    return result


SO101Follower.send_action = safe_send_action


# ============================================================
# RUN SAME ACT POLICY
# ============================================================

sys.argv = [
    "lerobot-record",

    "--robot.type=so101_follower",
    "--robot.port=/dev/ttyACM0",
    "--robot.id=my_awesome_follower_arm",

    "--robot.cameras={ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}",

    "--display_data=true",

    "--policy.path=models/grab_fish_v9_clean_ACT_50k",

    f"--dataset.repo_id=Cookieman12/eval_act_eflesh_v3_{STAMP}",

    "--dataset.single_task=Grab the fish",

    "--dataset.num_episodes=1",
    "--dataset.episode_time_s=60",
    "--dataset.reset_time_s=5",
    "--dataset.push_to_hub=false",
]


print("\n================================================")
print(" ACT + eFLESH-INFORMED LIMITER V3")
print("================================================")
print(f"Current limit              : {CURRENT_LIMIT} raw")
print(f"Contact current            : {CONTACT_CURRENT} raw")
print(f"Contact search position    : <= {CONTACT_SEARCH_POSITION}")
print(
    f"Max post-contact closure   : "
    f"{MAX_POST_CONTACT_CLOSURE}"
)
print("================================================\n")


try:
    main()

finally:
    f.close()

    print("\nTelemetry saved:")
    print(CSV_PATH)
