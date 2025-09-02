import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
results_path = "results/hybrid/reports"

# Load binary and multiclass metrics
binary = pd.read_csv(os.path.join(results_path, "metrics_binary_ml_vs_hybrid.csv"))
multiclass = pd.read_csv(os.path.join(results_path, "metrics_multiclass_ml_vs_hybrid.csv"))

# Optionally load rule-based results if you saved them
rule_based = pd.DataFrame([
    {"kind": "Rule-Based", "accuracy": 0.90, "precision_weighted": 0.92, "recall_weighted": 0.85, "f1_weighted": 0.88}
])

# Merge all results
binary_all = pd.concat([rule_based, binary], ignore_index=True)
multiclass_all = pd.concat([rule_based, multiclass], ignore_index=True)

print("\n=== Binary IDS Comparison ===")
print(binary_all)
print("\n=== Multiclass IDS Comparison ===")
print(multiclass_all)

# ===== BAR CHARTS =====
def plot_comparison(df, title, filename):
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    df_plot = df.set_index("kind")[metrics]
    df_plot.plot(kind="bar", figsize=(8,5))
    plt.title(title)
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    plt.close()

plot_comparison(binary_all, "Binary IDS Comparison", "binary_ids_comparison.png")
plot_comparison(multiclass_all, "Multiclass IDS Comparison", "multiclass_ids_comparison.png")

# ===== LATEX TABLES =====
print("\n=== Binary IDS LaTeX Table ===")
print(binary_all.to_latex(index=False, caption="Binary IDS comparative results", label="tab:binary-comparison"))

print("\n=== Multiclass IDS LaTeX Table ===")
print(multiclass_all.to_latex(index=False, caption="Multiclass IDS comparative results", label="tab:multiclass-comparison"))
