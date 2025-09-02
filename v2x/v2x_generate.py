import csv, math, random, time
from pathlib import Path
random.seed(42)

OUT = Path("project/v2x/v2x_normal.csv")
VEH_IDS = ["vehA","vehB","vehC"]
STEP = 0.1   # seconds between BSMs per vehicle
DUR  = 60.0  # seconds total

def move(t, base_lat, base_lon, speed_mps, heading_deg):
    # simple kinematics on a plane for demo
    dx = speed_mps * math.cos(math.radians(heading_deg)) * t
    dy = speed_mps * math.sin(math.radians(heading_deg)) * t
    # 1 deg lat ~ 111_111 m; 1 deg lon ~ 111_111*cos(lat) m (rough)
    lat = base_lat + dy/111_111.0
    lon = base_lon + dx/(111_111.0*math.cos(math.radians(base_lat)))
    return lat, lon

rows=[]
t=0.0
configs = {
  "vehA": {"lat":12.9716,"lon":77.5946,"speed":13.9,"head":  0},  # ~50 km/h north
  "vehB": {"lat":12.9706,"lon":77.5940,"speed": 8.3,"head": 90},  # ~30 km/h east
  "vehC": {"lat":12.9720,"lon":77.5950,"speed":11.1,"head":180},  # ~40 km/h south
}
while t <= DUR:
    for vid in VEH_IDS:
        cfg = configs[vid]
        lat, lon = move(t, cfg["lat"], cfg["lon"], cfg["speed"], cfg["head"])
        # add tiny noise
        lat += random.gauss(0, 1e-5); lon += random.gauss(0, 1e-5)
        spd = max(0.0, random.gauss(cfg["speed"], 0.5))  # m/s
        heading = (cfg["head"] + random.gauss(0, 1.0)) % 360
        rows.append([t, vid, lat, lon, spd, heading])
    t = round(t + STEP, 3)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w=csv.writer(f)
    w.writerow(["timestamp","sender_id","lat","lon","speed_mps","heading_deg","label"])
    for r in rows:
        w.writerow(r + [0])  # 0 = normal
print(f"Wrote {OUT} ({len(rows)} rows)")
