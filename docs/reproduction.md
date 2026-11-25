# Reproduction guide

## Offline plots and physical-ablation summaries

Use Python 3.12 (the version documented for the original offline inference environment). The following dependencies cover the plotting and summary scripts, but exact historical NumPy/pandas/Matplotlib versions are unavailable:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/method1/generate_method1_figures.py
python scripts/common/generate_expanded_results_figures.py
python scripts/method2/generate_ablation_figures.py
python scripts/method2/summarize_physical_ablation.py
```

The physical comparison uses the retained trial-level metrics, not raw frame logs. See [experimental design and exclusions](physical-ablation.md).

## Full matched-input ACT inference

The original repository documented successful validation using LeRobot 0.6.1, Python 3.12 and CUDA. Install LeRobot 0.6.1 and a compatible PyTorch build in a separate environment following the upstream instructions; exact PyTorch/CUDA and transitive dependency versions are unavailable. Do not assume that this environment also supports the original hardware scripts.

Dataset: [Cookieman12/grab_fish_eflesh_act_v1](https://huggingface.co/datasets/Cookieman12/grab_fish_eflesh_act_v1).  
Checkpoint: [Cookieman12/grab_fish_eflesh_act_v1_50k](https://huggingface.co/Cookieman12/grab_fish_eflesh_act_v1_50k).

```bash
python scripts/method2/run_matched_tactile_ablation.py
python scripts/method2/generate_ablation_figures.py
```

Output: `results/method2_ablation/matched_tactile_ablation.csv`. The original project reported a bit-for-bit reproduced CSV with SHA-256:

```
edce86cd969ca2731a5b15d4cbffc23a7fc16337a5e9583ac4d46fc992691a1a
```

Verify with `sha256sum results/method2_ablation/matched_tactile_ablation.csv` (Linux) or `Get-FileHash results/method2_ablation/matched_tactile_ablation.csv -Algorithm SHA256` (PowerShell). This revision does not independently rerun GPU inference or assert identical results across versions.

## Physical hardware execution

- [Method 1 runner](../scripts/method1/run_safety_limiter.py): external gripper limiter on the SO-101 follower, using gripper position and measured motor current.
- [Method 2 runner](../scripts/method2/run_tactile_integrated_act.py): logs a tactile-enabled SO follower with 36 scalar state features. The original eFlesh-enabled observation implementation is required; the wrapper alone does not implement sensor acquisition.

These are archived experimental scripts, not plug-and-play hardware examples. Inspect their serial port, camera indices, robot calibration, policy path, data destination and LeRobot API compatibility before executing. Confirm the gripper convention and re-establish contact thresholds for the actual hardware; the experimental thresholds are not a certified safety system. The precise training commands, original custom LeRobot patches, firmware configuration, no-tactile checkpoint and full frame-level physical comparison logs are not committed in this revision.
