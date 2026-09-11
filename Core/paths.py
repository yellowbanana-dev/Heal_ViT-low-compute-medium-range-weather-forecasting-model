"""Canonical project paths, so scripts work from any working directory."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(PROJECT_ROOT, "Data_Config", "config.yaml")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "Checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Checkpoint written at the end of the final curriculum stage; the weights all
# evaluation in this repo is run from.
FINAL_CHECKPOINT_STAGE = 6


def checkpoint(ar_steps=FINAL_CHECKPOINT_STAGE):
    """Path to the checkpoint saved after the `ar_steps`-step curriculum stage."""
    return os.path.join(CHECKPOINT_DIR, f"healvit_wb2_ar{ar_steps}.pt")
