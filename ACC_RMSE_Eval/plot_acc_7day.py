import json
import csv
import numpy as np
import matplotlib.pyplot as plt

JSON_PATH = "evaluate_metrics.json"
OUT_PNG_GRID = "48L_acc_7day_all_vars.png"
OUT_PNG_AVG = "48L_acc_7day_average.png"
OUT_CSV = "48L_acc_7day.csv"

HOURS_PER_DAY = 24
NUM_DAYS = 7
MAX_HOURS = HOURS_PER_DAY * NUM_DAYS  # 168h

with open(JSON_PATH, "r") as f:
    data = json.load(f)

variables = data["metadata"]["variables"]
steps_total = data["metadata"]["steps"]
acc_data = np.array(data["acc"])  # shape [steps, num_vars]

lead_times_all = np.arange(1, steps_total + 1) * 6  # each step = 6h
mask = lead_times_all <= MAX_HOURS
lead_times = lead_times_all[mask]
acc_7day = acc_data[mask, :]

# ---- CSV ----
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lead_time_hours"] + variables + ["average_acc"])
    for i, lt in enumerate(lead_times):
        row = [int(lt)] + acc_7day[i, :].tolist() + [float(np.mean(acc_7day[i, :]))]
        writer.writerow(row)

# ---- Grid plot: ACC per variable, 4x5 ----
fig, axes = plt.subplots(4, 5, figsize=(25, 20), sharex=True)
axes = axes.flatten()
for i, var in enumerate(variables):
    ax = axes[i]
    ax.plot(lead_times, acc_7day[:, i], color="blue", linewidth=2, label="HEAL-ViT 48L")
    ax.set_title(f"ACC: {var}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1.05)
    if i >= 15:
        ax.set_xlabel("Lead Time (Hours)", fontsize=10)
    if i % 5 == 0:
        ax.set_ylabel("ACC", fontsize=10)
fig.suptitle("HEAL-ViT 48 Layers — ACC, 7-Day Forecast", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT_PNG_GRID, dpi=300, bbox_inches="tight")

# ---- Average ACC plot ----
avg_acc = np.mean(acc_7day, axis=1)
fig_avg, ax_avg = plt.subplots(figsize=(10, 6))
ax_avg.plot(lead_times, avg_acc, color="purple", linewidth=2, label="Average ACC")
ax_avg.axhline(y=0.6, color="black", linestyle="--", linewidth=1.5, label="ACC = 0.6")
ax_avg.set_title("HEAL-ViT 48 Layers — Average ACC over All Variables (7-Day)", fontsize=14)
ax_avg.set_xlabel("Lead Time (Hours)", fontsize=12)
ax_avg.set_ylabel("ACC", fontsize=12)
ax_avg.set_ylim(0, 1.05)
ax_avg.set_xlim(0, MAX_HOURS)
ax_avg.grid(True, linestyle="--", alpha=0.6)
ax_avg.legend()
fig_avg.tight_layout()
fig_avg.savefig(OUT_PNG_AVG, dpi=300, bbox_inches="tight")

print(f"Saved: {OUT_PNG_GRID}, {OUT_PNG_AVG}, {OUT_CSV}")
print(f"Lead times covered: {lead_times[0]}h to {lead_times[-1]}h ({len(lead_times)} steps)")
