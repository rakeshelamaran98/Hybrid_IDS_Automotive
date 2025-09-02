import time, can

REQ_ID = 0x7E0   # Diagnostic request
UDS_CMDS = {
    "DiagSessionControl": bytes.fromhex("02 10 03 00 00 00 00 00"),
    "ECUReset": bytes.fromhex("02 11 01 00 00 00 00 00"),
    "ReadDataByID": bytes.fromhex("03 22 F1 90 00 00 00 00"),
}

bus = can.interface.Bus(channel="vcan0", bustype="socketcan")

print("UDS injector started (Ctrl+C to stop)")
try:
    while True:
        for name, payload in UDS_CMDS.items():
            msg = can.Message(arbitration_id=REQ_ID, data=payload, is_extended_id=False)
            bus.send(msg)
            print(f"Sent UDS command: {name} → {payload.hex()}")
            time.sleep(1.5)
except KeyboardInterrupt:
    print("\nUDS injection stopped.")
