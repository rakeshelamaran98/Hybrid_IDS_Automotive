#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def confusion_counts(df, truth_col, pred_col):
    tp = int(((df[truth_col] == 1) & (df[pred_col] == 1)).sum())
    fn = int(((df[truth_col] == 1) & (df[pred_col] == 0)).sum())
    fp = int(((df[truth_col] == 0) & (df[pred_col] == 1)).sum())
    tn = int(((df[truth_col] == 0) & (df[pred_col] == 0)).sum())
    return tp, fn, fp, tn

def main():
    ap = argparse.ArgumentParser(description="V2X detector (rules + optional ML)")
    ap.add_argument("--csv", required=True, help="input CSV with normal+attack rows")
    ap.add_argument("--max_speed", type=float, default=60.0, help="speed rule threshold (m/s)")
    ap.add_argument("--max_jerk", type=float, default=30.0, help="jerk rule threshold (m/s^3)")
    ap.add_argument("--rare_window", type=float, default=3.0, help="seconds; IDs active less than this are 'rare'")
    ap.add_argument("--test_size", type=float, default=0.30, help="test split for ML")
    args = ap.parse_args()

    inp = Path(args.csv)
    df = pd.read_csv(inp)

    # Basic sanity
    need_cols = {"timestamp", "sender_id", "lat", "lon", "speed_mps", "heading_deg"}
    if not need_cols.issubset(df.columns):
        missing = need_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    if "label" not in df.columns:
        # If no labels, create a dummy normal label so rules can still be inspected
        df["label"] = 0

    # --- Feature engineering for rules ---
    df = df.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)
    # time step per sender
    df["dt"] = df.groupby("sender_id")["timestamp"].diff().fillna(0.1)
    df.loc[df["dt"] <= 0, "dt"] = 0.1  # avoid div/0 and negative steps

    # deltas in lon/lat per sender
    df["dlon"] = df.groupby("sender_id")["lon"].diff().fillna(0.0)
    df["dlat"] = df.groupby("sender_id")["lat"].diff().fillna(0.0)

    # rough meters conversion
    # 1 deg lat ~ 111_111 m; 1 deg lon ~ 111_111*cos(lat)
    df["dx_m"] = df["dlon"] * 111_111 * np.cos(np.radians(df["lat"].clip(lower=1e-9)))
    df["dy_m"] = df["dlat"] * 111_111
    df["dist_m"] = np.sqrt(df["dx_m"] ** 2 + df["dy_m"] ** 2)

    # instantaneous speed & jerk
    df["inst_speed_mps"] = (df["dist_m"] / df["dt"]).clip(lower=0, upper=300)
    sp_prev = df.groupby("sender_id")["inst_speed_mps"].shift(1).fillna(df["inst_speed_mps"])
    df["jerk"] = ((df["inst_speed_mps"] - sp_prev) / df["dt"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # --- Rules ---
    # R1: unrealistic speed
    df["rule_speed"] = (df["inst_speed_mps"] > args.max_speed).astype(int)
    # R2: unrealistic jerk
    df["rule_jerk"] = (df["jerk"].abs() > args.max_jerk).astype(int)
    # R3: rare sender id (appears only in a short time window)
    dur = df.groupby("sender_id")["timestamp"].agg(["min", "max"])
    rare_ids = dur[(dur["max"] - dur["min"]) < args.rare_window].index
    df["rule_rare_id"] = df["sender_id"].isin(rare_ids).astype(int)

    # Combined rules OR
    df["rule_attack"] = ((df["rule_speed"] == 1) | (df["rule_jerk"] == 1) | (df["rule_rare_id"] == 1)).astype(int)

    # --- Optional tiny ML (if labels available) ---
    if df["label"].nunique() == 2:
        feats = ["inst_speed_mps", "jerk"]
        X = df[feats].values
        y = df["label"].values

        # Clean NaN/Inf → finite values for sklearn
        X = np.nan_to_num(X, nan=0.0, posinf=200.0, neginf=-200.0)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
        yhat = clf.predict(Xte)

        print("\n[ML] Logistic Regression (V2X) — test split report")
        print(classification_report(yte, yhat, digits=3))

        # Save ML confusion (on test split) as quick counts
        df_ml_eval = pd.DataFrame({"y_true": yte, "y_pred": yhat})
        tp, fn, fp, tn = confusion_counts(df_ml_eval, "y_true", "y_pred")
        print(f"[ML] Confusion (test): TP={tp} FN={fn} FP={fp} TN={tn}")

    else:
        print("[i] Labels not found or not binary; skipping ML evaluation.")

    # --- Rules confusion on FULL data (since rules don't need train/test) ---
    tp, fn, fp, tn = confusion_counts(df, "label", "rule_attack")
    print("\n[Rules] Confusion (full data): TP={} FN={} FP={} TN={}".format(tp, fn, fp, tn))
    print("[Rules] Breakdown:",
          "speed:", int(df["rule_speed"].sum()),
          "jerk:", int(df["rule_jerk"].sum()),
          "rare_id:", int(df["rule_rare_id"].sum()))

    # Save flagged CSV next to input
    out = inp.with_name(inp.stem + "_flags.csv")
    df.to_csv(out, index=False)
    print("Wrote", out)

if __name__ == "__main__":
    main()
