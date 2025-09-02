#!/usr/bin/env bash
set -e
say(){ printf "[+] %s\n" "$*"; }
send(){ say "$1"; shift; cansend vcan0 "$@"; sleep 0.2; }
send "DiagSessionControl (0x10 0x01)" 7E0#0210010000000000
send "ECUReset (0x11 0x01)"           7E0#0211010000000000
send "ReadDID F190 (0x22 F1 90)"      7E0#0322F19000000000
for i in 1 2 3; do
  send "DiagSessionControl" 7E0#0210010000000000
  send "ECUReset"           7E0#0211010000000000
  send "ReadDID F190"       7E0#0322F19000000000
done
