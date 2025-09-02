import time, random, re, glob
import can

# Target some real IDs (you can adjust later)
TARGET_IDS = [0x244, 0x166, 0x191]

# Try to seed payloads from latest normal log
baseline_files = sorted(glob.glob("../logs/normal/*.log"))
seed_payloads = {}
pat = re.compile(r'^\((?P<ts>[\d\.]+)\)\s+\w+\s+(?P<id>[0-9A-Fa-f]+)#(?P<data>[0-9A-Fa-f]{0,16})\s*$')

if baseline_files:
    with open(baseline_files[-1]) as f:
        for line in f:
            m = pat.match(line.strip())
            if not m: continue
            cid = int(m.group('id'), 16)
            data = m.group('data').ljust(16, '0')
            if cid in TARGET_IDS and cid not in seed_payloads:
                seed_payloads[cid] = bytearray.fromhex(data)

for cid in TARGET_IDS:
    seed_payloads.setdefault(cid, bytearray(random.getrandbits(8) for _ in range(8)))

bus = can.interface.Bus(channel="vcan0", bustype="socketcan")
print("Semi-structured fuzz running on IDs:", [hex(x) for x in TARGET_IDS])

try:
    while True:
        for cid in TARGET_IDS:
            p = bytearray(seed_payloads[cid])
            for _ in range(random.randint(1,2)):
                idx = random.randrange(8)
                delta = random.choice([-3,-2,-1,1,2,3,16,-16])
                p[idx] = (p[idx] + delta) & 0xFF
            msg = can.Message(arbitration_id=cid, data=p, is_extended_id=False)
            bus.send(msg)
            time.sleep(random.uniform(0.01,0.03))
except KeyboardInterrupt:
    pass
