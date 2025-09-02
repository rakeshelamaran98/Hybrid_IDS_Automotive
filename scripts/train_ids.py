#!/usr/bin/env python3
import argparse, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump

# --- Optional libs (guarded) ---
HAS_XGB = False
HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    pass
# -------------------------------

def load_dataset(csv_path: Path, target_mode: str, sample: int | None):
    df = pd.read_csv(csv_path)
    if sample is not None and sample > 0:
        df = df.sample(n=min(sample, len(df)), random_state=42)

    feature_cols = ["arbitration_id"] + [f"byte{i}" for i in range(8)]
    X = df[feature_cols].astype(np.float32).values
    y = df["label_binary"].astype(int).values if target_mode == "binary" \
        else df["label_multiclass"].astype(int).values
    return df, X, y, feature_cols

def get_models(include_heavy: bool=False):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=400, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, n_jobs=-1, random_state=42
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, n_jobs=-1, random_state=42
        ),
    }
    if include_heavy and HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=8, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            n_jobs=-1, random_state=42
        )
    if include_heavy and HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1
        )
    return models

def evaluate_models(X, y, models, results_dir: Path, mode: str):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "reports").mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for name, model in models.items():
        y_true_all, y_pred_all = [], []
        for tr, va in skf.split(X, y):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            y_true_all.append(y[va]); y_pred_all.append(pred)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        acc = accuracy_score(y_true_all, y_pred_all)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_all, y_pred_all, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_true_all, y_pred_all)

        # Save confusion matrix & classification report
        cm_path = results_dir / "reports" / f"{name}_{mode}_cm.csv"
        pd.DataFrame(cm).to_csv(cm_path, index=False)
        (results_dir / "reports" / f"{name}_{mode}_report.txt").write_text(
            classification_report(y_true_all, y_pred_all, zero_division=0)
        )

        # Fit on full data & persist
        model.fit(X, y)
        dump(model, results_dir / "models" / f"{name}_{mode}.pkl")

        metrics_rows.append({
            "model": name, "mode": mode,
            "accuracy": round(acc, 5),
            "precision_weighted": round(prec, 5),
            "recall_weighted": round(rec, 5),
            "f1_weighted": round(f1, 5),
            "cm_csv": cm_path.name
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1_weighted", ascending=False)
    return metrics_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="project/preprocessed_csv/all_attacks_combined.csv")
    ap.add_argument("--mode", choices=["binary","multiclass"], default="binary")
    ap.add_argument("--outdir", default="project/results")
    ap.add_argument("--sample", type=int, default=0, help="use only N rows (0 = all)")
    ap.add_argument("--heavy", action="store_true", help="include XGBoost/LightGBM if available")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df, X, y, feats = load_dataset(csv_path, args.mode, args.sample if args.sample>0 else None)
    pd.Series(feats).to_csv(outdir / f"feature_list_{args.mode}.csv", index=False, header=False)

    models = get_models(include_heavy=args.heavy)
    metrics_df = evaluate_models(X, y, models, outdir, args.mode)
    metrics_df.to_csv(outdir / f"metrics_{args.mode}.csv", index=False)
    print(f"\n=== RESULTS ({args.mode}) ===")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()
