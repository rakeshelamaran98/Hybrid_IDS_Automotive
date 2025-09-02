#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DEF_IN  = "project/preprocessed_csv/all_attacks_combined.csv"
DEF_OUT = "project/preprocessed_csv/all_with_rules.csv"

def add_dos_flag(df: pd.DataFrame, dos_threshold: int) -> pd.DataFrame:
    """
    DoS rule: mark frames whose (arbitration_id, floor(timestamp)) rate exceeds threshold.
    """
    df = df.copy()
    # integer seconds bucket to avoid huge groups
    df["second"] = df["timestamp"].astype(int)
    # per-ID-per-second rate
    rates = (
        df.groupby(["arbitration_id","second"], observed=True)
          .size()
          .rename("per_sec_rate")
          .reset_index()
    )
    df = df.merge(rates, on=["arbitration_id","second"], how="left")
    df["dos_flag"] = (df["per_sec_rate"] > dos_threshold).astype(int)
    return df.drop(columns=["per_sec_rate","second"])

def add_replay_flag(df: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """
    Replay rule: for each (ID,payload), if a repeat occurs within window_s, flag it.
    """
    df = df.copy()
    byte_cols = [f"byte{i}" for i in range(8)]
    # compact payload representation
    df["payload_hex"] = df[byte_cols].apply(lambda r: "".join(f"{int(v):02X}" for v in r), axis=1)
    # sort so we can compare to previous occurrence
    df = df.sort_values(["arbitration_id","payload_hex","timestamp"], kind="mergesort").reset_index(drop=True)
    prev_ts = df.groupby(["arbitration_id","payload_hex"], observed=True)["timestamp"].shift(1)
    dt = df["timestamp"] - prev_ts
    df["replay_flag"] = (dt <= window_s).fillna(False).astype(int)
    return df  # keep payload_hex; it can help debugging

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", default=DEF_IN, help="input CSV (all_attacks_combined.csv)")
    ap.add_argument("--csv_out", default=DEF_OUT, help="output CSV with new rule flags")
    ap.add_argument("--dos_threshold", type=int, default=800,
                    help="frames/sec per ID above which we flag DoS (try 500–1500)")
    ap.add_argument("--replay_window", type=float, default=1.0,
                    help="seconds; if same (ID,payload) repeats within this, flag replay")
    ap.add_argument("--sample", type=int, default=0,
                    help="use only N rows (0 = use full file)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    args = ap.parse_args()

    src = Path(args.csv_in)
    out = Path(args.csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading {src}")
    # read & optional sampling (uniform random to avoid bias)
    df = pd.read_csv(src)
    if args.sample and args.sample > 0 and args.sample < len(df):
        print(f"[i] Sampling {args.sample} / {len(df)} rows (seed={args.seed})")
        df = df.sample(n=args.sample, random_state=args.seed)

    # basic sanity columns
    need_cols = {"timestamp","arbitration_id",*(f"byte{i}" for i in range(8))}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"[+] Adding DoS flag (threshold={args.dos_threshold} frames/sec/ID)")
    df = add_dos_flag(df, args.dos_threshold)

    print(f"[+] Adding Replay flag (window={args.replay_window}s)")
    df = add_replay_flag(df, args.replay_window)

    print(f"[+] Saving {out}")
    df.to_csv(out, index=False)

    # Quick summaries
    print("\n[Summary: flagged counts per class]")
    if "attack_name" in df.columns:
        print(df.groupby("attack_name")[["dos_flag","replay_flag"]].sum())
    else:
        print(df[["dos_flag","replay_flag"]].sum())

if __name__ == "__main__":
    main()
