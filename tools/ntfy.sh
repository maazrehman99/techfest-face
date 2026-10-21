#!/usr/bin/env bash
# usage: tools/ntfy.sh "title" "message" [priority 1-5: 5 = urgent, makes the phone alert loudly]
T=$(cat ~/.config/techfest_ntfy_topic)
curl -s -m 15 -H "Title: $1" -H "Priority: ${3:-3}" -d "$2" "https://ntfy.sh/$T" >/dev/null && echo sent || echo "ntfy send failed"
