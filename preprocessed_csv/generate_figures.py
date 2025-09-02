import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# 1. Dataset Class Distribution
# -------------------
df = pd.read_csv("../../preprocessed_csv/all_attacks_combined.csv")

plt.figure(figsize=(10,6))
df['attack_name'].value_counts().plot(kind='bar', color="skyblue", edgecolor="black")
plt.title("Class Distribution of Dataset Samples")
plt.ylabel("Number of Samples")
plt.xlabel("Class")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("fig_class_distribution.png", dpi=300)
plt.close()

# -------------------
# 2. Rule vs ML vs Hybrid Comparison
# -------------------
metrics = pd.read_csv("metrics_binary_ml_vs_hybrid.csv")

metrics.set_index("Approach")[["Accuracy","Precision","Recall","F1"]].plot(
    kind='bar', figsize=(10,6), color=["#4c72b0","#55a868","#c44e52","#8172b3"]
)
plt.title("Comparison of IDS Approaches – Rule vs ML vs Hybrid")
plt.ylabel("Score")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("fig_ids_comparison.png", dpi=300)
plt.close()

# -------------------
# 3. Confusion Matrices (Binary + Multiclass)
# -------------------
def plot_confusion_matrix(csv_file, title, output_file):
    cm = pd.read_csv(csv_file, index_col=0)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

plot_confusion_matrix("binary_ML_cm.csv", "Confusion Matrix – ML (Binary)", "fig_binary_ML_cm.png")
plot_confusion_matrix("binary_HYBRID_cm.csv", "Confusion Matrix – Hybrid (Binary)", "fig_binary_HYBRID_cm.png")
plot_confusion_matrix("multiclass_ML_cm.csv", "Confusion Matrix – ML (Multiclass)", "fig_multiclass_ML_cm.png")
plot_confusion_matrix("multiclass_HYBRID_cm.csv", "Confusion Matrix – Hybrid (Multiclass)", "fig_multiclass_HYBRID_cm.png")

# -------------------
# 4. Feature Importance (Optional)
# -------------------
fi = pd.read_csv("feature_importance_multiclass.csv")

plt.figure(figsize=(10,6))
sns.barplot(x="importance", y="feature", data=fi.sort_values(by="importance", ascending=False))
plt.title("Feature Importance – Multiclass Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("fig_feature_importance.png", dpi=300)
plt.close()

print("✅ All figures generated and saved in reports folder!")
