# Experimental data and CSV dictionary

This directory contains the experimental measurements and intermediate analysis tables used in *Learning from Demonstration*, Engineering Project 2 (Technion, 2026). The descriptions below follow the report sections **eFlesh Sensing Principle**, **Experimental Procedure and Calibration Analysis**, **Method 2 Ablation Study Design**, **Results**, and **Experimental Error and Uncertainty**. Column names are taken from the repository CSVs and analysis scripts.

These files are analysis recordings, not the full image-based training datasets. The Method 2 demonstration dataset and trained checkpoint are available on [Hugging Face (dataset)](https://huggingface.co/datasets/Cookieman12/grab_fish_eflesh_act_v1) and [Hugging Face (model)](https://huggingface.co/Cookieman12/grab_fish_eflesh_act_v1_50k).

## 1. How to read the measurements

| Quantity | Meaning and units |
| --- | --- |
| Time | Seconds; `time_s` is recording-relative time, `unix_time` is a Unix timestamp, and `aligned_time_s` is the common analysis time base. |
| Robot position / action | Calibrated SO-101 position coordinates. Actions are absolute target joint positions, not velocities or currents. Smaller gripper-position values mean greater closure. |
| Post-contact closure | Contact position minus current measured position, in SO-101 units. Positive values indicate further closure. |
| Tactile components | Signed, baseline-relative magnetic-field components from the two eFlesh surfaces (five three-axis sensing locations per surface, 30 channels total). |
| Combined eFlesh response | Sum of the ten Euclidean magnitudes of the baseline-relative three-axis vectors: sum over all locations of sqrt(ΔBx² + ΔBy² + ΔBz²). Reported in processed magnetic-response units, not newtons. |
| Motor current | Measured servo feedback in raw motor-telemetry units, not amperes or commanded current. Absolute current is the magnitude of this feedback. |

The robot and cameras were recorded at 30 Hz. During the separate Method 1 calibration, eFlesh was recorded at approximately 50 Hz. A deliberate full-gripper-closure event provided a synchronization marker; the report documents an offset of approximately 76.36 s for the analyzed calibration recording. The aligned CSVs already contain processed measurements; do not blindly apply the offset again.

## 2. Method 1 — Calibration and external safety limiter

### Calibration data

| File | What it contains / why it matters |
| --- | --- |
| [eflesh_motor_aligned.csv](method1/calibration/eflesh_motor_aligned.csv) | Synchronized tactile response, gripper position and motor telemetry. Used for the eFlesh/current correlation (Pearson r = 0.844). |
| [deformation_current_detailed.csv](method1/calibration/deformation_current_detailed.csv) | Processed post-contact data from the five retained soft-fish squeezes. Used for the response-versus-closure plot and per-squeeze statistics. |
| [rigid_aligned.csv](method1/calibration/rigid_aligned.csv) | Aligned measurements for the rigid 3D-printed PLA reference. |
| [soft_aligned.csv](method1/calibration/soft_aligned.csv) | Aligned measurements for the soft silicone fish. Compared with the rigid reference to examine how contact behavior differs. |

**Calibration column guide**

| Columns | Interpretation |
| --- | --- |
| `gripper_command`, `gripper_position`, `gripper_error` | Target and measured gripper positions; tracking error (command minus measured position). |
| `present_current_raw`, `abs_current` | Signed measured motor current and its absolute magnitude. |
| `present_load_raw`, `abs_load`, `present_voltage_raw` | Auxiliary servo telemetry; raw values, not calibrated mechanical force or converted voltage. |
| `on_motor_total`, `not_motor_total`, `combined_total` | Motor-side and opposite-side tactile response magnitudes, and their sum. |
| `asymmetry` | Auxiliary indicator of response imbalance between jaws. The exact calibration-file definition is not documented in the report; do not assume it matches the Method 2 asymmetry calculation. |
| `squeeze` | Identifier of one of the five retained progressive soft-fish squeezes. |
| `eflesh_smooth`, `current_smooth`, `voltage_smooth`, `position_smooth` | Processed/smoothed traces. The report does not specify the smoothing parameters. |
| `eflesh_above_contact` | Contact-referenced tactile response. |
| `actual_extra_closure` | Additional closure from the squeeze's bilateral-contact position. |
| `rigid_equivalent`, `estimated_deformation` | Intermediate analysis fields; not independently calibrated deformation measurements. Their exact derivation is not specified in the final report. |
| `signal`, `gripper_state` (rigid/soft comparison) | Combined tactile response and measured gripper position used by the plotting script. |

The rigid/soft file entries document the two columns consumed by the plotting script; the full headers of those two large files were not available for inspection.

### ACT execution data

| File | What it contains |
| --- | --- |
| [act_no_limiter.csv](method1/act_no_limiter.csv) | Unrestricted ACT execution with measured gripper position and motor telemetry. |
| [act_with_safety_limiter.csv](method1/act_with_safety_limiter.csv) | Execution of the same trained ACT policy with the external limiter enabled, including the requested and transmitted commands and limiter-state flags. |

| Columns | Interpretation |
| --- | --- |
| `gripper_command`, `act_gripper_command` | Gripper position requested by ACT. |
| `sent_gripper_command` | Gripper target transmitted after the limiter. Subtract the ACT request from this value to obtain the command modification. |
| `gripper_position`, `position_error` | Measured gripper position and tracking error (unrestricted recording includes the latter). |
| `current_raw`, `present_current_raw`, `abs_current_raw` | Measured motor current, signed or absolute as named. |
| `system_armed`, `contact_detected`, `current_relief`, `position_limit_active` | Boolean flags (0/1) for the limiter state. |
| `contact_position`, `safe_boundary_position` | Stored contact position and commanded lower position bound after contact; may be blank before detection. |
| `post_contact_closure` | Additional measured closure after detected contact. |
| `present_load_raw`, `abs_load_raw`, `present_voltage_raw` | Auxiliary raw servo telemetry in the unrestricted log. |

The Method 1 comparison uses a bare gripper during final execution; the calibration recordings use eFlesh-equipped jaws. The configured one-unit closure cap limits the command, not the actual measured physical displacement. The measured maximum additional closure was 1.76 units. The raw current thresholds are specific to the experimental gripper configuration.

Analysis: [generate_method1_figures.py](../scripts/method1/generate_method1_figures.py) and [generate_expanded_results_figures.py](../scripts/common/generate_expanded_results_figures.py).

## 3. Method 2 — Direct tactile integration

### Representative physical rollout

[successful_rollout.csv](method2/successful_rollout.csv) records one successful physical grasp of the fish (approximately 29.6 s at 29.4 Hz). It contains the 36 scalar observation channels and six requested and transmitted target positions, but not the camera images.

| Columns / patterns | Interpretation |
| --- | --- |
| `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos` | Six measured robot-state positions. |
| `eflesh_on_motor.m0.bx` … `eflesh_on_motor.m4.bz` | Fifteen signed motor-side tactile components (five locations, three axes). |
| `eflesh_not_motor.m0.bx` … `eflesh_not_motor.m4.bz` | Fifteen corresponding opposite-side tactile components. |
| `policy_cmd.<joint>.pos`, `sent_cmd.<joint>.pos` | ACT-requested and actually transmitted target positions, for all six joints. |
| `eflesh_on_motor_total`, `eflesh_not_motor_total`, `eflesh_combined_total` | Per-surface and combined tactile magnitudes calculated for analysis, not scalar policy inputs. |
| `eflesh_asymmetry` | Motor-side total minus opposite-side total (signed difference, as implemented in the rollout script). |

The dominant high-tactile interval is the longest continuous interval exceeding 50% of the rollout's maximum combined response (approximately 12.16–19.14 s). This is an analysis convention, not a calibrated contact threshold. Motor-current telemetry is not included in this representative recording.

Analysis: [generate_expanded_results_figures.py](../scripts/common/generate_expanded_results_figures.py). Logger: [run_tactile_integrated_act.py](../scripts/method2/run_tactile_integrated_act.py).

### Offline ablation — Input sensitivity without retraining

[matched_tactile_pairs.csv](method2/matched_tactile_pairs.csv) defines one high-tactile target observation and a matched real low-tactile replacement per demonstration episode (50 pairs). The actual inference comparison replaces only the 30 tactile values; both images and all six target robot-state values stay fixed.

| Columns / patterns | Interpretation |
| --- | --- |
| `target_episode`, `replacement_episode` | Demonstration episode identifiers. |
| `target_dataset_index`, `replacement_dataset_index` | Dataset-wide indices locating the observations for inference. |
| `target_frame_index`, `replacement_frame_index`, `*_time_s` | Frame indices and timestamps for the target and replacement observations. |
| `target_tactile_score`, `replacement_tactile_score` | Combined response magnitudes used to characterize the tactile contrast. |
| `target_gripper`, `replacement_gripper`, `abs_gripper_diff` | Source gripper states and absolute difference (required to be at most two SO-101 units). |
| `other5_normalized_distance`, `<arm_joint>_diff` | Arm-configuration matching distance and individual five arm-joint differences. |

[matched_tactile_ablation.csv](../results/method2_ablation/matched_tactile_ablation.csv) contains the resulting predicted actions. An identical archived copy also exists [in this directory](method2/matched_tactile_ablation.csv); the results-directory copy is the canonical output used by the analysis scripts.

| Columns / patterns | Interpretation |
| --- | --- |
| `tactile_ratio` | Target tactile score divided by replacement tactile score. |
| `target_gripper_state`, `replacement_source_gripper_state`, `matched_gripper_difference` | Gripper-position matching information, not changes to the robot-state input. |
| `normal_<joint>`, `low_tactile_<joint>` | Predicted target position using real versus substituted low-tactile input. |
| `signed_delta_<joint>`, `abs_delta_<joint>` | Real prediction minus low-tactile prediction, and its absolute value. For the gripper, a negative signed difference means greater predicted closure with real tactile input. |

The experiment establishes policy sensitivity to tactile observations, not physical deformation or force reduction. See [run_matched_tactile_ablation.py](../scripts/method2/run_matched_tactile_ablation.py) and [generate_ablation_figures.py](../scripts/method2/generate_ablation_figures.py).

## 4. Derived results and physical ablation

These CSVs live in [results/](../results/) rather than in this directory.

| File | Contents |
| --- | --- |
| [eflesh_per_squeeze_summary.csv](../results/eflesh_per_squeeze_summary.csv) | One row per calibration squeeze: `contact_position`, `minimum_position`, `maximum_post_contact_closure`, `maximum_eflesh`, `maximum_current_raw`, `median_current_raw`. Positions and closure use SO-101 units; eFlesh and current retain their respective processed/raw units. |
| [final_quantitative_summary.csv](../results/final_quantitative_summary.csv) | Method 1 key-value summary, including the tactile/current correlation and limiter performance. |
| [matched_tactile_ablation.csv](../results/method2_ablation/matched_tactile_ablation.csv) | Canonical offline ablation inference results, described above. |
| [clean_trial_metrics.csv](../results/physical_tactile_ablation/clean_trial_metrics.csv) | One row per retained physical rollout (28 total). |
| [summary.csv](../results/physical_tactile_ablation/summary.csv) | Group means, sample standard deviations and paired comparisons by orientation and metric. |

### Controlled physical policy ablation — Retraining with and without tactile input

Both policies used the same 50 demonstrations and 50,000 optimization steps. The eFlesh-equipped gripper was used and tactile measurements were logged for both policies, but only the tactile-integrated policy received those measurements as input. The retained sample contains five matched pairs at 0°, five at 17° and four at 35°; repeat 4 at 35° was excluded as a pair due to data-quality issues. The full frame-level physical-ablation logs are not included in the repository.

| Column | Interpretation |
| --- | --- |
| `angle_deg` | Fish orientation in degrees relative to nominal. |
| `repeat` | Matched repetition number within an orientation. |
| `policy` | `no`: ACT without tactile input; `plus`: ACT with tactile input. |
| `p95_tactile` | 95th percentile of combined eFlesh response within one rollout; less sensitive to isolated spikes than the maximum. |
| `peak_tactile` | Maximum combined eFlesh response in that rollout. |
| `tactile_auc` | Time integral of the combined response (processed response units × s). |
| `p95_current` | 95th percentile of absolute motor current, in raw telemetry units. |
| `n_pairs`, `metric` | Number of retained matched pairs and metric being summarized. |
| `no_mean`, `plus_mean`, `no_sd`, `plus_sd` | Across-rollout means and sample standard deviations for each policy. |
| `change_percent` | 100 × (plus mean / no mean − 1); positive means an increase with tactile input. |
| `plus_lower_pairs` | Count of matched pairs where the tactile-integrated policy had a strictly lower metric. |

See [physical-ablation protocol and exclusions](../docs/physical-ablation.md) and [summarize_physical_ablation.py](../scripts/method2/summarize_physical_ablation.py).

## 5. Interpretation and uncertainty

The report's experimental uncertainty discussion identifies baseline drift, hysteresis, actuator friction/dynamics, contact geometry and residual synchronization error as potential error sources. Across the five calibration squeezes, the maximum tactile response had a coefficient of variation of approximately 6.3%. In the physical ablation, within-condition p95 tactile standard deviations ranged from approximately 3.57 × 10³ to 9.80 × 10³ processed units.

These statistics describe repeatability, not instrument accuracy. Consecutive samples in a time series are correlated; they are not independent trials. The small paired sample and one trained policy per condition limit conclusions. Neither magnetic response nor raw current provides a calibrated measure of contact force, and a reduction in either does not by itself establish less object deformation.

For software setup and reproduction commands, see the [reproduction guide](../docs/reproduction.md).
