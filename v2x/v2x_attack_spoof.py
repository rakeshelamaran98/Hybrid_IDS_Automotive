import csv, random
from pathlib import Path
random.seed(1)

SRC = Path("project/v2x/v2x_normal.csv")
OUT = Path("project/v2x/v2x_attack_spoof.csv")

rows = list(csv.DictReader(SRC.open()))
# Inject a fake "vehX" that teleports around with impossible motion
for i in range(50, 150):  # between t≈5s and t≈15s
    t = float(rows[i]["timestamp"])
    rows.append({
        "timestamp": f"{t:.1f}",
        "sender_id": "vehX",
        "lat": "12.9800", "lon": "77.6000",
        "speed_mps": "0.1",
        "heading_deg": "0.0",
        "label": "1"  # 1=attack
    })

rows.sort(key=lambda r: (float(r["timestamp"]), r["sender_id"]))
with OUT.open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"[Spoof] Wrote {OUT} ({len(rows)} rows)")
