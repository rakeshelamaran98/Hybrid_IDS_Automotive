#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

DEF_IN   = "project/preprocessed_csv/all_with_rules.csv"
DEF_OUTD = "project/results/hybrid"
DEF_MODEL_MC = "project/results/models/RandomForest_multiclass.pkl"
DEF_MODEL_BIN = "project/results/models/RandomForest_binary.pkl"

FEATURES = ["arbitration_id"] + [f"byte{i}" for i in range(8)]

def save_confusion(name, y_true, y_pred, outdir: Path, labels=None, title="Confusion Matrix"):
    # CSV
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm).to_csv(outdir / f"{name}_cm.csv", index=False)

    # PNG plot
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # tick labels if provided
    if labels is not None:
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_cm.png", dpi=150)
    plt.close(fig)

def metrics_table(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {"accuracy":acc, "precision_weighted":prec, "recall_weighted":rec, "f1_weighted":f1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEF_IN, help="dataset with dos_flag & replay_flag")
    ap.add_argument("--model_mc", default=DEF_MODEL_MC, help="multiclass model .pkl")
    ap.add_argument("--model_bin", default=DEF_MODEL_BIN, help="binary model .pkl (optional)")
    ap.add_argument("--outdir", default=DEF_OUTD, help="output directory for reports/plots")
    args = ap.parse_args()

    outdir = Path(args.outdir); (outdir / "reports").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    X = df[FEATURES].values
    y_mc_true = df["label_multiclass"].values
    y_bin_true = df["label_binary"].values

    # -------- ML predictions --------
    clf_mc = load(args.model_mc)
    y_mc_ml = clf_mc.predict(X)

    # Optional feature importance (tree models)
    if hasattr(clf_mc, "feature_importances_"):
        fi = pd.Series(clf_mc.feature_importances_, index=FEATURES).sort_values(ascending=True)
        ax = fi.plot(kind="barh", figsize=(6,4), title="Feature Importance (Multiclass model)")
        ax.figure.tight_layout(); ax.figure.savefig(outdir / "feature_importance_multiclass.png", dpi=150)
        plt.close(ax.figure)
        fi.to_csv(outdir / "reports/feature_importance_multiclass.csv")

    # -------- Hybrid overrides (rules first) --------
    # class map reminder: 0=Normal,1=Spoof,2=Replay,3=DoS,4=Fuzzing,5=Stealth,6=UDS
    y_mc_hybrid = y_mc_ml.copy()
    dos_idx = (df["dos_flag"] == 1).values
    rep_idx = (df["replay_flag"] == 1).values & (~dos_idx)
    y_mc_hybrid[dos_idx] = 3
    y_mc_hybrid[rep_idx] = 2

    # -------- Save metrics (multiclass) --------
    labels_mc = sorted(df["label_multiclass"].unique())
    mt_ml = metrics_table(y_mc_true, y_mc_ml)
    mt_hb = metrics_table(y_mc_true, y_mc_hybrid)

    pd.DataFrame([
        {"kind":"ML", **{k:round(v,5) for k,v in mt_ml.items()}},
        {"kind":"Hybrid", **{k:round(v,5) for k,v in mt_hb.items()}}
    ]).to_csv(outdir / "reports/metrics_multiclass_ml_vs_hybrid.csv", index=False)

    save_confusion("multiclass_ML", y_mc_true, y_mc_ml, outdir/"reports", labels=labels_mc, title="Multiclass (ML)")
    save_confusion("multiclass_HYBRID", y_mc_true, y_mc_hybrid, outdir/"reports", labels=labels_mc, title="Multiclass (Hybrid)")

    # -------- Binary comparison --------
    # If a binary model is available, use it; otherwise derive from multiclass
    try:
        clf_bin = load(args.model_bin)
        y_bin_ml = clf_bin.predict(X)
    except Exception:
        y_bin_ml = (y_mc_ml != 0).astype(int)

    # Hybrid binary: if any rule fires → Attack (1)
    y_bin_hybrid = y_bin_ml.copy()
    y_bin_hybrid[(df["dos_flag"]==1) | (df["replay_flag"]==1)] = 1

    mtb_ml = metrics_table(y_bin_true, y_bin_ml)
    mtb_hb = metrics_table(y_bin_true, y_bin_hybrid)
    pd.DataFrame([
        {"kind":"ML", **{k:round(v,5) for k,v in mtb_ml.items()}},
        {"kind":"Hybrid", **{k:round(v,5) for k,v in mtb_hb.items()}}
    ]).to_csv(outdir / "reports/metrics_binary_ml_vs_hybrid.csv", index=False)

    save_confusion("binary_ML", y_bin_true, y_bin_ml, outdir/"reports", labels=[0,1], title="Binary (ML)")
    save_confusion("binary_HYBRID", y_bin_true, y_bin_hybrid, outdir/"reports", labels=[0,1], title="Binary (Hybrid)")

    print("\n[OK] Wrote:")
    for f in sorted((outdir/"reports").glob("*")):
        print(" -", f.relative_to(outdir))

if __name__ == "__main__":
    main()
