import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_overlay_plots(json_6l, json_48l):
    if not os.path.exists(json_6l):
        print(f"Error: {json_6l} not found.")
        return
    if not os.path.exists(json_48l):
        print(f"Error: {json_48l} not found.")
        return

    with open(json_6l, "r") as f:
        data_6 = json.load(f)
    with open(json_48l, "r") as f:
        data_48 = json.load(f)

    variables = data_6["metadata"]["variables"]
    steps = data_6["metadata"]["steps"]
    
    rmse_6 = np.array(data_6["rmse"])
    acc_6  = np.array(data_6["acc"])
    
    rmse_48 = np.array(data_48["rmse"])
    acc_48  = np.array(data_48["acc"])

    lead_times = np.arange(1, steps + 1) * 6
    num_vars = len(variables)
    
    # 4 rows, 5 columns
    fig_rmse, axes_rmse = plt.subplots(4, 5, figsize=(25, 20), sharex=True)
    fig_acc, axes_acc = plt.subplots(4, 5, figsize=(25, 20), sharex=True)
    
    axes_rmse = axes_rmse.flatten()
    axes_acc = axes_acc.flatten()
    
    for i, var in enumerate(variables):
        # Plot RMSE
        ax = axes_rmse[i]
        ax.plot(lead_times, rmse_6[:, i], color='tab:blue', linewidth=2, label="6L Swin", alpha=0.8)
        ax.plot(lead_times, rmse_48[:, i], color='tab:red', linewidth=2, label="48L Swin", alpha=0.8)
        ax.set_title(f"RMSE: {var}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.legend(fontsize=10)
        if i >= 15: # Bottom row
            ax.set_xlabel("Lead Time (Hours)", fontsize=10)
        if i % 5 == 0: # Leftmost column
            ax.set_ylabel("RMSE", fontsize=10)
            
        # Plot ACC
        ax = axes_acc[i]
        ax.plot(lead_times, acc_6[:, i], color='tab:blue', linewidth=2, label="6L Swin", alpha=0.8)
        ax.plot(lead_times, acc_48[:, i], color='tab:red', linewidth=2, label="48L Swin", alpha=0.8)
        ax.set_title(f"ACC: {var}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1.05)
        if i == 0:
            ax.legend(fontsize=10)
        if i >= 15:
            ax.set_xlabel("Lead Time (Hours)", fontsize=10)
        if i % 5 == 0:
            ax.set_ylabel("ACC", fontsize=10)

    fig_rmse.suptitle('RMSE Overlay: 6L (Blue) vs 48L (Red)', fontsize=20, y=1.02)
    fig_acc.suptitle('ACC Overlay: 6L (Blue) vs 48L (Red)', fontsize=20, y=1.02)

    fig_rmse.tight_layout()
    fig_acc.tight_layout()
    
    out_rmse = "rmse_all_20_vars_overlay.png"
    out_acc = "acc_all_20_vars_overlay.png"
    fig_rmse.savefig(out_rmse, dpi=300, bbox_inches='tight')
    fig_acc.savefig(out_acc, dpi=300, bbox_inches='tight')
    
    # Plot Average ACC over all variables
    avg_acc_6 = np.mean(acc_6, axis=1)
    avg_acc_48 = np.mean(acc_48, axis=1)
    
    fig_avg, ax_avg = plt.subplots(figsize=(10, 6))
    ax_avg.plot(lead_times, avg_acc_6, 'tab:blue', linewidth=2.5, label="6L Swin")
    ax_avg.plot(lead_times, avg_acc_48, 'tab:red', linewidth=2.5, label="48L Swin")
    ax_avg.axhline(y=0.6, color='black', linestyle='--', linewidth=1.5, label="Predictability Limit (ACC = 0.6)")
    ax_avg.set_title("Average ACC over all Variables - 6L vs 48L Overlay", fontsize=14)
    ax_avg.set_xlabel("Lead Time (Hours)", fontsize=12)
    ax_avg.set_ylabel("ACC", fontsize=12)
    ax_avg.set_ylim(0, 1.05)
    ax_avg.grid(True, linestyle='--', alpha=0.6)
    ax_avg.legend(fontsize=12)
    fig_avg.tight_layout()
    
    out_avg = "acc_average_overlay.png"
    fig_avg.savefig(out_avg, dpi=300, bbox_inches='tight')
    
    print(f"Plots saved: {out_rmse}, {out_acc}, {out_avg}")

if __name__ == "__main__":
    path_6l = "/storage/bharat/Heal_ViT_Refactored/ACC_RMSE_Eval/evaluate_metrics.json"
    path_48l = "/storage/bharat/Heal_ViT_Refactored_feat_48swin/ACC_RMSE_Eval/evaluate_metrics.json"
    generate_overlay_plots(path_6l, path_48l)
