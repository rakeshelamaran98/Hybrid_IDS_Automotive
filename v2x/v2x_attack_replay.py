import csv
from pathlib import Path

SRC = Path("project/v2x/v2x_normal.csv")
OUT = Path("project/v2x/v2x_attack_replay.csv")

rows = list(csv.DictReader(SRC.open()))
# Take a slice from early in the drive and replay it 10s later
slice_rows = [r for r in rows if 5.0 <= float(r["timestamp"]) <= 10.0 and r["sender_id"]=="vehA"]
replayed=[]
for r in slice_rows:
    r2 = dict(r)
    r2["timestamp"] = f"{float(r['timestamp']) + 10.0:.1f}"  # shift by +10s
    r2["label"] = "1"  # mark as attack
    replayed.append(r2)

rows.extend(replayed)
rows.sort(key=lambda r: (float(r["timestamp"]), r["sender_id"]))
with OUT.open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"[Replay] Wrote {OUT} ({len(rows)} rows)")
