"""Regenerate docs/RESULTS.md from the JSON metric dumps in results/.

Usage:  python scripts/summarize_results.py

Reads results/metrics_48L.json, results/metrics_6L.json and
results/spectra_48L.json (all produced by the evaluation scripts in
ACC_RMSE_Eval/ and Power_Spectra/) and writes the markdown tables.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# Lead times (in 6-hourly steps, zero-indexed) highlighted in the summary tables
LEADS = [(3, "1 d"), (7, "2 d"), (11, "3 d"), (19, "5 d"), (27, "7 d"),
         (39, "10 d"), (59, "15 d")]

UNITS = {
    "2m_temperature": "K",
    "10m_u_component_of_wind": "m/s",
    "10m_v_component_of_wind": "m/s",
    "mean_sea_level_pressure": "Pa",
    "total_column_water_vapour": "kg/m^2",
    "temperature": "K",
    "geopotential": "m^2/s^2",
    "specific_humidity": "kg/kg",
    "u_component_of_wind": "m/s",
    "v_component_of_wind": "m/s",
}


def unit(var):
    if var in UNITS:
        return UNITS[var]
    return UNITS.get(var.rsplit("_", 1)[0], "")


def skill_horizon(acc, i, threshold=0.6):
    """Lead time in days at which ACC first drops below `threshold`."""
    for step, row in enumerate(acc):
        if row[i] < threshold:
            return step * 6 / 24.0
    return None


# mean_sea_level_pressure is excluded from the aggregate ACC means: the
# WeatherBench 2 climatology used here has no MSLP field, so the evaluation code
# falls back to surface_pressure and the resulting "anomaly" is dominated by a
# static orographic offset. See the Known issues section of the README.
EXCLUDE_FROM_MEAN = {"mean_sea_level_pressure"}


def mean_acc(acc, variables, step):
    vals = [x for i, x in enumerate(acc[step]) if variables[i] not in EXCLUDE_FROM_MEAN]
    return sum(vals) / len(vals)


def fmt(x):
    if abs(x) < 1e-2:
        return f"{x:.3g}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.3g}"


def main():
    with open(os.path.join(RESULTS, "metrics_48L.json")) as f:
        m48 = json.load(f)
    with open(os.path.join(RESULTS, "metrics_6L.json")) as f:
        m6 = json.load(f)

    variables = m48["metadata"]["variables"]
    out = []
    w = out.append

    w("# Detailed results\n")
    w("All numbers are computed over the held-out 2019-2020 ERA5 validation years, "
      "autoregressively rolled out to 60 steps (15 days) from the `ar6` checkpoint. "
      "RMSE is latitude-weighted and reported in physical units; ACC is latitude-weighted "
      "anomaly correlation against the WeatherBench 2 1990-2019 hourly climatology.\n")
    w("Regenerate this file with `python scripts/summarize_results.py`.\n")

    w("\n## Anomaly correlation coefficient (48-layer)\n")
    header = "| Variable | " + " | ".join(lbl for _, lbl in LEADS) + " | Skill horizon (ACC > 0.6) |"
    w(header)
    w("|" + "---|" * (len(LEADS) + 2))
    for i, v in enumerate(variables):
        h = skill_horizon(m48["acc"], i)
        hs = f"{h:.2f} d" if h is not None else "> 15 d"
        w(f"| `{v}` | " + " | ".join(f"{m48['acc'][s][i]:.3f}" for s, _ in LEADS)
          + f" | {hs} |")
    means = [mean_acc(m48["acc"], variables, s) for s, _ in LEADS]
    w("| **mean (19 vars)** | " + " | ".join(f"**{x:.3f}**" for x in means) + " | |")
    w("\n`mean_sea_level_pressure` is reported for completeness but excluded from the "
      "mean: the climatology zarr has no MSLP field and the evaluation falls back to "
      "`surface_pressure`, which inflates its ACC. See Known issues in the README.\n")

    w("\n## Latitude-weighted RMSE (48-layer, physical units)\n")
    w("| Variable | Unit | " + " | ".join(lbl for _, lbl in LEADS) + " |")
    w("|" + "---|" * (len(LEADS) + 2))
    for i, v in enumerate(variables):
        w(f"| `{v}` | {unit(v)} | "
          + " | ".join(fmt(m48["rmse"][s][i]) for s, _ in LEADS) + " |")

    w("\n## Depth ablation: 48 layers vs 6 layers\n")
    w("Identical data, graph, optimiser and curriculum; only `num_bottleneck_layers` differs.\n")
    w("| Variable | RMSE 3 d (48L) | RMSE 3 d (6L) | ACC 3 d (48L) | ACC 3 d (6L) | "
      "Skill horizon 48L | Skill horizon 6L |")
    w("|" + "---|" * 7)
    for i, v in enumerate(variables):
        h48, h6 = skill_horizon(m48["acc"], i), skill_horizon(m6["acc"], i)
        f48 = f"{h48:.2f} d" if h48 is not None else "> 15 d"
        f6 = f"{h6:.2f} d" if h6 is not None else "> 15 d"
        w(f"| `{v}` | {fmt(m48['rmse'][11][i])} | {fmt(m6['rmse'][11][i])} | "
          f"{m48['acc'][11][i]:.3f} | {m6['acc'][11][i]:.3f} | {f48} | {f6} |")
    for label, s in (("1 d", 3), ("3 d", 11), ("5 d", 19), ("7 d", 27)):
        a48 = mean_acc(m48["acc"], variables, s)
        a6 = mean_acc(m6["acc"], variables, s)
        w(f"| **mean ACC @ {label}** | | | **{a48:.3f}** | **{a6:.3f}** | | |")

    spectra_path = os.path.join(RESULTS, "spectra_48L.json")
    if os.path.exists(spectra_path):
        with open(spectra_path) as f:
            sp = json.load(f)
        sv = sp["metadata"]["variables"]
        steps = sp["metadata"]["spectra_steps"]
        w("\n## Spectral energy retention (48-layer)\n")
        w("Ratio of predicted to ERA5 zonal power, averaged over synoptic "
          "(wavenumber 20-59) and sub-synoptic (60-120) bands. 1.0 means the "
          "forecast carries the same variance as the truth at those scales.\n")
        picks = ["2m_temperature", "temperature_850", "u_component_of_wind_250"]
        w("| Lead | " + " | ".join(f"`{p}` syn / sub" for p in picks) + " |")
        w("|" + "---|" * (len(picks) + 1))
        for s in steps:
            pred, truth = sp[f"pred_step_{s}"], sp[f"truth_step_{s}"]
            cells = []
            for p in picks:
                i = sv.index(p)
                mid = [pred[i][k] / truth[i][k] for k in range(20, 60) if truth[i][k] > 0]
                hi = [pred[i][k] / truth[i][k] for k in range(60, 121) if truth[i][k] > 0]
                cells.append(f"{sum(mid)/len(mid):.2f} / {sum(hi)/len(hi):.2f}")
            w(f"| {s * 6} h | " + " | ".join(cells) + " |")

    text = "\n".join(out) + "\n"
    dest = os.path.join(ROOT, "docs", "RESULTS.md")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(text)
    print(f"Wrote {dest} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
