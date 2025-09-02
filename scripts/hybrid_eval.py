#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ---------- Paths resolve relative to this script ----------
ROOT = Path(__file__).resolve().parents[1]  # .../project

DEF_IN        = ROOT / "preprocessed_csv/all_with_rules_with_iat.csv"
DEF_OUTD      = ROOT / "results/hybrid"
DEF_MODEL_MC  = ROOT / "results/models/RandomForest_multiclass.pkl"
DEF_MODEL_BIN = ROOT / "results/models/RandomForest_binary.pkl"

# Our canonical features (superset). The actual columns fed to the model
# will be aligned later to the model's expected set/order.
FEATURES_CANONICAL = ["arbitration_id"] + [f"byte{i}" for i in range(8)] + ["delta"]

CLASS_MAP = {
    0: "Normal", 1: "Spoofing", 2: "Replay", 3: "DoS",
    4: "Fuzzing", 5: "Stealth", 6: "UDS",
}

def save_confusion(name, y_true, y_pred, outdir: Path, labels=None, class_names=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    outdir.mkdir(parents=True, exist_ok=True)

    if class_names is not None and len(class_names) == cm.shape[0]:
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(outdir / f"{name}_cm.csv")
    else:
        pd.DataFrame(cm).to_csv(outdir / f"{name}_cm.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(im)

    if class_names is not None and len(class_names) == cm.shape[0]:
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(outdir / f"{name}_cm.png", dpi=300)
    plt.close(fig)

def metrics_table(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {"accuracy":acc, "precision_weighted":prec, "recall_weighted":rec, "f1_weighted":f1}

def ensure_delta_exists(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure an IAT feature named 'delta' exists. Try aliases, else compute from timestamps per arbitration_id."""
    if "delta" in df.columns:
        return df

    # 1) Try common aliases
    iat_aliases = ["iat", "IAT", "inter_arrival_time", "interarrival", "interarrival_time"]
    for c in iat_aliases:
        if c in df.columns:
            df = df.copy()
            df.rename(columns={c: "delta"}, inplace=True)
            return df

    # 2) Compute from a timestamp column
    ts_candidates = ["timestamp", "time", "time_s", "time_ms", "time_us", "ts"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(
            "Missing required 'delta' feature and no IAT/timestamp column found. "
            f"Tried aliases {iat_aliases} and timestamps {ts_candidates}."
        )

    if "arbitration_id" not in df.columns:
        raise ValueError("Cannot compute 'delta' without 'arbitration_id' column.")

    df = df.copy()

    ts = df[ts_col]
    # Normalize to seconds
    if np.issubdtype(ts.dtype, np.number):
        vmax = float(np.nanmax(ts.values)) if len(ts) else 0.0
        if vmax > 1e12:       # nanoseconds
            scale = 1e-9
        elif vmax > 1e9:      # microseconds
            scale = 1e-6
        elif vmax > 1e6:      # milliseconds
            scale = 1e-3
        else:                 # seconds
            scale = 1.0
        ts_sec = ts.astype(float) * scale
    else:
        ts_sec = pd.to_datetime(ts, errors="coerce").astype("int64") * 1e-9  # ns → s

    df["__ts_sec__"] = ts_sec
    df.sort_values(["arbitration_id", "__ts_sec__"], inplace=True)
    df["delta"] = df.groupby("arbitration_id")["__ts_sec__"].diff().fillna(0.0)
    df["delta"] = df["delta"].clip(lower=0).astype(float)
    df.drop(columns="__ts_sec__", inplace=True)
    return df

def generate_roc_curves(y_true, y_pred_ml_proba, y_pred_hybrid_proba, class_labels, outdir: Path):
    """Comparative micro-average ROC for multiclass."""
    y_true_bin = label_binarize(y_true, classes=class_labels)

    fpr_ml, tpr_ml = roc_curve(y_true_bin.ravel(), y_pred_ml_proba.ravel())[:2]
    auc_ml = auc(fpr_ml, tpr_ml)

    fpr_h, tpr_h = roc_curve(y_true_bin.ravel(), y_pred_hybrid_proba.ravel())[:2]
    auc_h = auc(fpr_h, tpr_h)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr_ml, tpr_ml, linestyle=':', linewidth=3,
            label=f'Standalone ML (Random Forest) micro-average ROC (AUC = {auc_ml:.4f})')
    ax.plot(fpr_h, tpr_h, linestyle='-', linewidth=3,
            label=f'Hybrid IDS micro-average ROC (AUC = {auc_h:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')

    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Comparative ROC Curve: Standalone ML vs. Hybrid IDS', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True)

    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "roc_curve_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC curve plot saved to: {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEF_IN), help="dataset with dos_flag & replay_flag")
    ap.add_argument("--model_mc", default=str(DEF_MODEL_MC), help="multiclass model.pkl")
    ap.add_argument("--model_bin", default=str(DEF_MODEL_BIN), help="binary model.pkl (optional)")
    ap.add_argument("--outdir", default=str(DEF_OUTD), help="output directory for reports/plots")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "reports").mkdir(parents=True, exist_ok=True)

    # -------- Load data --------
    df = pd.read_csv(args.csv)
    df = ensure_delta_exists(df)  # ensure 'delta' exists / is computed

    # Required columns (for rules + labels)
    required_cols = set(
        ["arbitration_id"] + [f"byte{i}" for i in range(8)]
        + ["dos_flag", "replay_flag", "label_multiclass", "label_binary"]
        + (["delta"] if "delta" in df.columns else [])
    )
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # -------- Load model and align features --------
    clf_mc = load(args.model_mc)

    if hasattr(clf_mc, "feature_names_in_"):
        expected_features = list(clf_mc.feature_names_in_)
    else:
        nfi = getattr(clf_mc, "n_features_in_", None)
        if nfi == 10:
            expected_features = ["arbitration_id"] + [f"byte{i}" for i in range(8)] + ["delta"]
        elif nfi == 9:
            expected_features = ["arbitration_id"] + [f"byte{i}" for i in range(8)]  # no delta
        elif nfi == 8:
            expected_features = [f"byte{i}" for i in range(8)]  # bytes only
        else:
            # Fallback: intersect canonical with available columns
            all_feats = ["arbitration_id"] + [f"byte{i}" for i in range(8)] + (["delta"] if "delta" in df.columns else [])
            expected_features = [c for c in all_feats if c in df.columns][: (nfi or len(all_feats))]

    # Make sure expected features exist
    missing_feats = [c for c in expected_features if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Model expects features {expected_features} but CSV is missing {missing_feats}")

    # Build X in the order the model expects
    X_df = df[expected_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X_df.values

    y_mc_true = df["label_multiclass"].astype(int).values
    y_bin_true = df["label_binary"].astype(int).values

    # Sanity check
    if hasattr(clf_mc, "n_features_in_") and clf_mc.n_features_in_ != X.shape[1]:
        raise ValueError(f"Feature count mismatch after alignment: model expects {clf_mc.n_features_in_}, got {X.shape[1]}.")

    # -------- ML predictions (multiclass) --------
    y_mc_ml = clf_mc.predict(X)
    y_mc_ml_proba = clf_mc.predict_proba(X)

    # Optional feature importance
    if hasattr(clf_mc, "feature_importances_"):
        fi = pd.Series(clf_mc.feature_importances_, index=expected_features).sort_values(ascending=True)
        ax = fi.plot(kind="barh", figsize=(6, 4), title="Feature Importance (Multiclass model)")
        ax.figure.tight_layout()
        ax.figure.savefig(outdir / "feature_importance_multiclass.png", dpi=150)
        plt.close(ax.figure)
        fi.to_csv(outdir / "reports/feature_importance_multiclass.csv")

    # -------- Hybrid overrides (rules first) --------
    # Class map: 0=Normal, 1=Spoofing, 2=Replay, 3=DoS, 4=Fuzzing, 5=Stealth, 6=UDS
    y_mc_hybrid = y_mc_ml.copy()
    dos_idx = (df["dos_flag"] == 1).values
    rep_idx = (df["replay_flag"] == 1).values & (~dos_idx)
    y_mc_hybrid[dos_idx] = 3
    y_mc_hybrid[rep_idx] = 2

    # Hybrid probabilities aligned to clf_mc.classes_
    y_mc_hybrid_proba = y_mc_ml_proba.copy()
    class_to_col = {cls: i for i, cls in enumerate(clf_mc.classes_)}
    if 3 in class_to_col:
        y_mc_hybrid_proba[dos_idx, :] = 0.0
        y_mc_hybrid_proba[dos_idx, class_to_col[3]] = 1.0
    if 2 in class_to_col:
        y_mc_hybrid_proba[rep_idx, :] = 0.0
        y_mc_hybrid_proba[rep_idx, class_to_col[2]] = 1.0

    # -------- Save metrics (multiclass) --------
    labels_mc_int = list(clf_mc.classes_)
    class_names_mc = [CLASS_MAP.get(i, str(i)) for i in labels_mc_int]

    mt_ml = metrics_table(y_mc_true, y_mc_ml)
    mt_hb = metrics_table(y_mc_true, y_mc_hybrid)
    pd.DataFrame([
        {"kind":"ML",     **{k:round(v,5) for k,v in mt_ml.items()}},
        {"kind":"Hybrid", **{k:round(v,5) for k,v in mt_hb.items()}}
    ]).to_csv(outdir / "reports/metrics_multiclass_ml_vs_hybrid.csv", index=False)

    save_confusion("multiclass_ML", y_mc_true, y_mc_ml, outdir/"reports",
                   labels=labels_mc_int, class_names=class_names_mc, title="Multiclass (ML)")
    save_confusion("multiclass_HYBRID", y_mc_true, y_mc_hybrid, outdir/"reports",
                   labels=labels_mc_int, class_names=class_names_mc, title="Multiclass (Hybrid)")

    # ROC curve comparison
    generate_roc_curves(y_mc_true, y_mc_ml_proba, y_mc_hybrid_proba, labels_mc_int, outdir / "reports")

    # -------- Binary comparison --------
    try:
        clf_bin = load(args.model_bin)
        if hasattr(clf_bin, "n_features_in_") and clf_bin.n_features_in_ != X.shape[1]:
            raise ValueError(f"[Binary] Feature count mismatch: model expects {clf_bin.n_features_in_}, got {X.shape[1]}.")
        y_bin_ml = clf_bin.predict(X)
    except Exception:
        # Fallback: anything not Normal (0) is Attack
        y_bin_ml = (y_mc_ml != 0).astype(int)

    y_bin_hybrid = y_bin_ml.copy()
    y_bin_hybrid[(df["dos_flag"] == 1) | (df["replay_flag"] == 1)] = 1

    mtb_ml = metrics_table(y_bin_true, y_bin_ml)
    mtb_hb = metrics_table(y_bin_true, y_bin_hybrid)
    pd.DataFrame([
        {"kind":"ML",     **{k:round(v,5) for k,v in mtb_ml.items()}},
        {"kind":"Hybrid", **{k:round(v,5) for k,v in mtb_hb.items()}}
    ]).to_csv(outdir / "reports/metrics_binary_ml_vs_hybrid.csv", index=False)

    save_confusion("binary_ML", y_bin_true, y_bin_ml, outdir/"reports",
                   labels=[0,1], class_names=["Normal","Attack"], title="Binary (ML)")
    save_confusion("binary_HYBRID", y_bin_true, y_bin_hybrid, outdir/"reports",
                   labels=[0,1], class_names=["Normal","Attack"], title="Binary (Hybrid)")

    print("\n[OK] Wrote:")
    for f in sorted((outdir/"reports").glob("*")):
        print(" -", f.relative_to(outdir))

if __name__ == "__main__":
    main()
