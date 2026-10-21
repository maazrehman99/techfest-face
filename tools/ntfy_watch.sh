#!/usr/bin/env bash
# Phone alerts for the VM run, independent of Claude. Each step (3), the final result (4) and failures (5 = loud).
LOG=${1:-logs_gcp_run.log}; N=tools/ntfy.sh; last=$(date +%s); stalled=0
tail -n 0 -F "$LOG" | while IFS= read -r line; do
  last=$(date +%s)
  case "$line" in
    "=== 0/4"*) $N "VM step 1 of 5" "Install finished. Testing the ORIGINAL model on the held-out people (full pipeline, GPU)." 3;;
    "=== 1/4"*) $N "VM step 2 of 5" "Original model: quick test on cropped faces." 3;;
    "=== 2/4"*) $N "VM step 3 of 5" "Fine-tuning has started on the GPU." 3;;
    "=== 3/4"*) $N "VM step 4 of 5" "Fine-tuning done. Testing the NEW model on the same people." 3;;
    "=== 4/4"*) $N "VM step 5 of 5" "Last test: NEW model, full pipeline. Then results download and the VM is deleted." 3;;
    *"GPU run failed"*) $N "VM: GPU test fell back to CPU" "Slower but fine. No action needed." 3;;
    *"== FAILED"*|*PREFLIGHT*Error*|*"remote command failed"*|*"download failed"*|*"ERROR:"*) $N "VM RUN PROBLEM" "$line" 5;;
    *"== done"*) $N "VM run finished" "Results downloaded to runs/ft1. VM is being deleted." 4;;
    "exit "*) $N "VM script ended" "$line" 4;;
    *"=== 0/4"*|*"PREFLIGHT_OK"*) :;;
  esac
done
