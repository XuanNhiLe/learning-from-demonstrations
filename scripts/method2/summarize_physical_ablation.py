"""Summarize matched physical trials from archived, clean trial-level metrics."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "results" / "physical_tactile_ablation"
METRICS = ["p95_tactile", "peak_tactile", "tactile_auc", "p95_current"]
df = pd.read_csv(DIR / "clean_trial_metrics.csv")
assert set(df.policy) == {"no", "plus"}
assert not df.duplicated(["angle_deg", "repeat", "policy"]).any()
assert not ((df.angle_deg == 35) & (df["repeat"] == 4)).any()
assert set(df.angle_deg) == {0, 17, 35}
out = []
for angle, group in df.groupby("angle_deg"):
    expected = {0: {1, 2, 3, 4, 5}, 17: {1, 2, 3, 4, 5}, 35: {1, 2, 3, 5}}[angle]
    no = group[group.policy == "no"].set_index("repeat")
    plus = group[group.policy == "plus"].set_index("repeat")
    assert set(no.index) == set(plus.index) == expected
    for metric in METRICS:
        a, b = no[metric].sort_index(), plus[metric].sort_index()
        out.append(dict(angle_deg=angle, n_pairs=len(a), metric=metric,
                        no_mean=a.mean(), no_sd=a.std(ddof=1),
                        plus_mean=b.mean(), plus_sd=b.std(ddof=1),
                        change_percent=100 * (b.mean() / a.mean() - 1),
                        plus_lower_pairs=int((b < a).sum())))
summary = pd.DataFrame(out)
summary.to_csv(DIR / "summary.csv", index=False, float_format="%.6f")
print(summary.to_string(index=False))
