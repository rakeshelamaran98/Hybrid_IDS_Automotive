#!/usr/bin/env python3
"""
Consolidate candump logs -> tidy frame table (Parquet).

Expected layout:
project/
  logs/
    normal/*.log|*.txt
    spoofing/*.log|*.txt
    ... (replay, dos, fuzz, stealth, uds)

Output:
  project/phase2_frames.parquet
"""

import os, re, glob, sys
import pandas as pd
import numpy as np
from typing import List, Dict

LOG_ROOT = os.environ.get("LOG_ROOT", "logs")
OUT_PATH = os.environ.get("OUT_PATH", "project/phase2_frames.parquet")

# candump classic frame, e.g.:
# (169082.123456) vcan0  123   [8]  11 22 33 44 55 66 77 88
CANDUMP_RE = re.compile(
    r"\((?P<ts>\d+\.\d+)\)\s+"
    r"(?P<bus>\w+)\s+"
    r"(?P<id>[0-9A-Fa-f]+)\s+"
    r"\[(?P<dlc>\d{1,2})\]\s+"
    r"(?P<data>([0-9A-Fa-f]{2}\s*)*)"
)

def parse_line(line: str):
    """
    Return dict for one frame or None if no match.
    Pads bytes to 8 with -1 (so models can treat 'missing' bytes consistently).
    """
    m = CANDUMP_RE.search(line)
    if not m:
        return None
    ts = float(m["ts"])
    bus = m["bus"]
    can_id = int(m["id"], 16)
    dlc = int(m["dlc"])

    data_str = (m["data"] or "").strip()
    data = [int(b, 16) for b in data_str.split()] if data_str else []
    # Clamp DLC to 0..8 for classic CAN; pad to 8 for consistent columns
    dlc = max(0, min(dlc, 8))
    padded = (data + [-1] * 8)[:8]

    row = {
        "ts": ts,
        "bus": bus,
        "can_id": can_id,
        "is_extended": int(can_id > 0x7FF),
        "dlc": dlc,
        **{f"byte_{i}": padded[i] for i in range(8)},
    }
    return row

def parse_file(path: str, label: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            r = parse_line(line)
            if r is None:
                # ignore non-frame lines silently (timestamps, comments, blanks)
                continue
            r["label"] = label
            r["source_file"] = os.path.basename(path)
            rows.append(r)
    return rows

def discover_label_dirs(root: str) -> List[str]:
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

def main():
    if not os.path.isdir(LOG_ROOT):
        print(f"[ERROR] Logs root not found: {LOG_ROOT}", file=sys.stderr)
        sys.exit(1)

    labels = discover_label_dirs(LOG_ROOT)
    if not labels:
        print(f"[ERROR] No label folders under {LOG_ROOT}", file=sys.stderr)
        sys.exit(1)

    all_rows: List[Dict] = []
    total_files = 0

    for label in labels:
        folder = os.path.join(LOG_ROOT, label)
        files = glob.glob(os.path.join(folder, "*.log")) + glob.glob(os.path.join(folder, "*.txt"))
        if not files:
            print(f"[WARN] No .log/.txt files in {folder}")
        for fp in files:
            total_files += 1
            rows = parse_file(fp, label)
            if not rows:
                print(f"[WARN] No frames parsed from {fp}")
            all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] Parsed 0 frames. Check your log formats.", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(all_rows).sort_values(["source_file", "ts"]).reset_index(drop=True)

    # Normalize time per file: ts0 starts at 0 for each source_file
    df["ts0"] = df.groupby("source_file")["ts"].transform(lambda s: s - s.min())

    # Basic sanity fixes
    df["dlc"] = df["dlc"].clip(lower=0, upper=8).astype(int)
    for i in range(8):
        df[f"byte_{i}"] = df[f"byte_{i}"].astype(int)

    # Quick summary to stdout
    print(f"[INFO] Files parsed: {total_files}")
    print(f"[INFO] Total frames : {len(df):,}")
    print("[INFO] Label counts:")
    print(df["label"].value_counts())

    # Ensure output folder exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"[OK] Saved {len(df):,} frames -> {OUT_PATH}")

    # Optional: peek a few rows
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
