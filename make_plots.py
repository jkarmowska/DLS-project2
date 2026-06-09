import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Paths to the summary metrics JSON files
METRICS_A_JSON = Path("subtaskA_restored_metrics.json")
METRICS_B_JSON = Path("subtaskB_restored_metrics.json")

SALIENCY_DIR = "saliency_outputs"  # Directory containing .tsv files from both subtasks
OUTPUT_PLOTS_DIR = "plots_output"  # Directory where final PNG plots will be saved
BAR_VALUES_A = [2.5, 2.7, 1.1]     # Approximate original reference values for Subtask A

def plot_bar_chart_subtask_a(summary_data):
    """Generates a bar chart comparing scores for Subtask A."""
    sequences = summary_data["sequences"]
    sequence_labels = ["Seq 1", "Seq 2", "Seq 3"]
    original_values = BAR_VALUES_A
    corrupted_values = [seq["mean_initial_score"] for seq in sequences]
    repaired_values = [seq["mean_final_score"] for seq in sequences]

    x = np.arange(len(sequence_labels))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, original_values, width, label="Original Reference", color="#2b5c8f")
    plt.bar(x, corrupted_values, width, label="Corrupted (Start)", color="#d95f02")
    plt.bar(x + width, repaired_values, width, label="Repaired (Final)", color="#2ca02c")
    
    plt.xticks(x, sequence_labels, fontsize=11)
    plt.ylabel("Average Ensemble Score (rna_dna_ratio)", fontsize=11)
    plt.title("Subtask A: Performance Comparison (Before vs After Optimization)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = Path(OUTPUT_PLOTS_DIR) / "subtaskA_comparison_bar_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Subtask A bar plot to: {out_path}")


def plot_saliency_heatmaps(summary_data, subtask_label):
    """Generates Signed Saliency heatmaps with an optimized layout for axes and colorbar."""
    sequences = summary_data["sequences"]
    model_names = summary_data["model_names"]
    
    for seq in sequences:
        header = seq["id"]
        safe_header = header.replace("/", "_")
        
        for model_name in model_names:
            for label in ("initial", "final"):
                tsv_path = Path(SALIENCY_DIR) / f"{safe_header}_{model_name}_{label}_saliency.tsv"
                
                # Skip silently if the .tsv file does not exist for a given model/state
                if not tsv_path.exists():
                    continue
                
                positions, saliency_vals = [], []
                with open(tsv_path, "r") as fh:
                    next(fh)  # Skip the TSV header
                    for line in fh:
                        pos, val = line.strip().split("\t")
                        positions.append(int(pos))
                        saliency_vals.append(float(val))
                
                sal = np.array(saliency_vals)

                # Initialize the plot with a balanced aspect ratio setup
                fig, ax = plt.subplots(figsize=(max(6, len(sal)/50), 2.5)) 
                im = ax.imshow(sal[np.newaxis, :], aspect="auto", cmap="seismic", vmin=-1, vmax=1)
                
                # Create a dedicated axis for the vertical colorbar on the right side
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                
                cbar = plt.colorbar(im, cax=cax, orientation="vertical")
                cbar.set_label("signed saliency", labelpad=15, rotation=270)
                
                ax.set_yticks([])
                ax.set_xlabel("nucleotide position", fontsize=10)
                ax.set_title(f"Subtask {subtask_label}: {header} | {model_name} ({label})", fontweight="bold", fontsize=11)
                
                png_path = Path(OUTPUT_PLOTS_DIR) / f"subtask{subtask_label}_{safe_header}_{model_name}_{label}_heatmap.png"
                plt.savefig(png_path, dpi=150, bbox_inches="tight") 
                plt.close(fig)
    print(f"Saliency maps for Subtask {subtask_label} generated successfully.")


def main():
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    
    # 1. PROCESS SUBTASK A
    if METRICS_A_JSON.exists():
        print("--- Processing Subtask A (Bar Plot & Saliency) ---")
        with open(METRICS_A_JSON, "r") as fh:
            data_a = json.load(fh)
        plot_bar_chart_subtask_a(data_a)
        plot_saliency_heatmaps(data_a, subtask_label="A")
    else:
        print(f"Info: {METRICS_A_JSON} not found, skipping Subtask A.")

    print("")  # Console line break

    # 2. PROCESS SUBTASK B
    if METRICS_B_JSON.exists():
        print("--- Processing Subtask B (Saliency Only) ---")
        with open(METRICS_B_JSON, "r") as fh:
            data_b = json.load(fh)
        plot_saliency_heatmaps(data_b, subtask_label="B")
    else:
        print(f"Info: {METRICS_B_JSON} not found, skipping Subtask B.")

    print(f"\nDone! All available plots saved to the '{OUTPUT_PLOTS_DIR}' folder.")

if __name__ == "__main__":
    main()