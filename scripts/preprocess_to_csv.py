import re, glob, csv
from pathlib import Path
import pandas as pd

# Folders expected under project/logs/
ATTACK_MAP = {
    "normal":0, "spoofing":1, "replay":2, "dos":3,
    "fuzzing":4, "stealth":5, "uds":6
}

LOG_ROOT = Path("project/logs")
OUT_DIR  = Path("project/preprocessed_csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# candump line like: (1692178694.123456) vcan0 244#1122334455667788
LINE = re.compile(r'^\((?P<ts>[\d\.]+)\)\s+\w+\s+(?P<id>[0-9A-Fa-f]+)#(?P<data>[0-9A-Fa-f]{0,16})\s*$')

def process_attack(name:str):
    rows = []
    for f in glob.glob(str(LOG_ROOT/name/"*.log")):
        with open(f, "r") as fh:
            for line in fh:
                m = LINE.match(line.strip())
                if not m: 
                    continue
                ts = float(m.group("ts"))
                can_id = int(m.group("id"), 16)
                data = (m.group("data") or "").ljust(16, "0")  # pad to 8 bytes
                bytes8 = [int(data[i:i+2], 16) for i in range(0, 16, 2)]
                y_mc = ATTACK_MAP[name]
                y_bin = 0 if name == "normal" else 1
                rows.append([ts, can_id, *bytes8, y_bin, y_mc, name])
    if not rows:
        print(f"[WARN] No rows for '{name}' (no logs found in project/logs/{name}/)")
        return None
    out = OUT_DIR/f"{name}.csv"
    with open(out, "w", newline="") as w:
        wr = csv.writer(w)
        wr.writerow(["timestamp","arbitration_id",
                     *[f"byte{i}" for i in range(8)],
                     "label_binary","label_multiclass","attack_name"])
        wr.writerows(rows)
    print(f"[OK] Wrote {out}  ({len(rows)} rows)")
    return out

made = []
for name in ATTACK_MAP.keys():
    out = process_attack(name)
    if out: made.append(out)

if made:
    df = pd.concat([pd.read_csv(p) for p in made], ignore_index=True)
    comb = OUT_DIR/"all_attacks_combined.csv"
    df.to_csv(comb, index=False)
    print(f"[OK] Wrote {comb}  ({len(df)} rows)")
else:
    print("[ERROR] No CSVs were produced. Check your log folders.")
