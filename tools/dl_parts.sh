# segmented resumable download: dl_parts.sh <ip[,ip...]> <url> <out> <total-bytes> <n-parts>; a piece below 20 KB/s for 30 s retries on the next connection
# stop it with: PowerShell kill of every process whose command line matches dl_parts.sh, then curl.exe
ips=$1; url=$2; out=$3; total=$4; n=$5
IFS=, read -ra IP <<< "$ips"
seg=$(( (total + n - 1) / n ))
part() {  # $1 = index
  local s=$(( $1 * seg )); local e=$(( s + seg - 1 )); [ $e -ge $total ] && e=$(( total - 1 ))
  local f="$out.part$1"; local want=$(( e - s + 1 ))
  for try in $(seq 1 300); do
    local have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] && return 0
    local ip=${IP[$(( ($1 + try) % ${#IP[@]} ))]}   # alternate connection on every retry
    curl -sL --interface "$ip" -r $(( s + have ))-$e --speed-limit 20000 --speed-time 30 "$url" >> "$f"
    sleep 2
  done
  return 1
}
for i in $(seq 0 $((n-1))); do part $i & done
wait
for i in $(seq 0 $((n-1))); do
  s=$(( i * seg )); e=$(( s + seg - 1 )); [ $e -ge $total ] && e=$(( total - 1 ))
  [ "$(stat -c %s "$out.part$i")" -eq $(( e - s + 1 )) ] || { echo "PART $i INCOMPLETE"; exit 1; }
done
cat $(for i in $(seq 0 $((n-1))); do echo "$out.part$i"; done) > "$out"
got=$(stat -c %s "$out")
[ "$got" -eq "$total" ] && { rm -f "$out".part*; echo "DONE $out $got bytes"; } || echo "SIZE MISMATCH $got vs $total"
