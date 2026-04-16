import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_plots(json_path="evaluate_metrics.json"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run the evaluation first.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    variables = data["metadata"]["variables"]
    steps = data["metadata"]["steps"]
    
    rmse_data = np.array(data["rmse"]) # Shape [steps, num_vars]
    acc_data = np.array(data["acc"])   # Shape [steps, num_vars]
    
    # X-axis is lead time in hours. Each step is 6 hours.
    lead_times = np.arange(1, steps + 1) * 6
    
    num_vars = len(variables)
    
    # 4 rows, 5 columns (for 20 variables)
    fig_rmse, axes_rmse = plt.subplots(4, 5, figsize=(25, 20), sharex=True)
    fig_acc, axes_acc = plt.subplots(4, 5, figsize=(25, 20), sharex=True)
    
    axes_rmse = axes_rmse.flatten()
    axes_acc = axes_acc.flatten()
    
    for i, var in enumerate(variables):
        # Plot RMSE
        ax = axes_rmse[i]
        ax.plot(lead_times, rmse_data[:, i], color='red', linewidth=2, label="HEAL-ViT")
        ax.set_title(f"RMSE: {var}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        if i >= 15: # Bottom row
            ax.set_xlabel("Lead Time (Hours)", fontsize=10)
        if i % 5 == 0: # Leftmost column
            ax.set_ylabel("RMSE (Physical Units)", fontsize=10)
            
        # Plot ACC
        ax = axes_acc[i]
        ax.plot(lead_times, acc_data[:, i], color='blue', linewidth=2, label="HEAL-ViT")
        ax.set_title(f"ACC: {var}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1.05)
        if i >= 15:
            ax.set_xlabel("Lead Time (Hours)", fontsize=10)
        if i % 5 == 0:
            ax.set_ylabel("ACC", fontsize=10)

    fig_rmse.tight_layout()
    fig_acc.tight_layout()
    
    fig_rmse.savefig("48L_rmse_all_20_vars.png", dpi=300, bbox_inches='tight')
    fig_acc.savefig("48L_acc_all_20_vars.png", dpi=300, bbox_inches='tight')
    
    # Plot Average ACC over all variables
    avg_acc = np.mean(acc_data, axis=1) # Average across variables
    
    fig_avg, ax_avg = plt.subplots(figsize=(10, 6))
    ax_avg.plot(lead_times, avg_acc, color='purple', linewidth=2, label="Average ACC")
    ax_avg.axhline(y=0.6, color='black', linestyle='--', linewidth=1.5, label="ACC = 0.6")
    ax_avg.set_title("Average ACC over all Variables", fontsize=14)
    ax_avg.set_xlabel("Lead Time (Hours)", fontsize=12)
    ax_avg.set_ylabel("ACC", fontsize=12)
    ax_avg.set_ylim(0, 1.05)
    ax_avg.grid(True, linestyle='--', alpha=0.6)
    ax_avg.legend()
    fig_avg.tight_layout()
    fig_avg.savefig("48L_acc_average.png", dpi=300, bbox_inches='tight')
    
    print("Plots saved: 48L_rmse_all_20_vars.png, 48L_acc_all_20_vars.png, 48L_acc_average.png")

if __name__ == "__main__":
    generate_plots()
