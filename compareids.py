import pandas as pd
import matplotlib.pyplot as plt

# === Load results ===
binary_df = pd.read_csv("metrics_binary_ml_vs_hybrid.csv")
multi_df = pd.read_csv("metrics_multiclass_ml_vs_hybrid.csv")

# === Metrics to compare ===
metrics = ["Accuracy", "Precision", "Recall", "F1", "FPR"]

# === Plot side-by-side charts ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Binary subplot
binary_plot = binary_df.set_index("Method")[metrics]
binary_plot.plot(kind="bar", ax=axes[0])
axes[0].set_title("Binary IDS Comparison")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)   # all metrics in [0,1]
axes[0].legend(loc="lower right", fontsize=8)

# Multiclass subplot
multi_plot = multi_df.set_index("Method")[metrics]
multi_plot.plot(kind="bar", ax=axes[1])
axes[1].set_title("Multiclass IDS Comparison")
axes[1].set_ylim(0, 1)
axes[1].legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("ids_comparison_combined.png", dpi=300)
plt.close()

print("✅ Saved combined bar chart: ids_comparison_combined.png")
