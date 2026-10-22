#!/usr/bin/env bash
# Run the lab-day submission on the GCP VM. This script NEVER deletes the VM (user rule); it can only start and stop it.
#   bash tools/vm_eval.sh setup           # one-time (tonight): create VM if missing, upload project + models, build venv, test
#   bash tools/vm_eval.sh run group10 ~/lab_pack   # lab day: start VM, upload pack, run, download group10.npz into submission/
#   bash tools/vm_eval.sh stop            # stop the VM (stops GPU/CPU billing, disk stays)
# Needs: `gcloud auth login` done. Override with env: PROJECT ZONE VM MACHINE GPU=1 (use ONNX GPU; off by default, see note).
set -euo pipefail
PROJECT="${PROJECT:-r2r-coros-dev}"
ZONE="${ZONE:-asia-south1-b}"
VM="${VM:-maaz-dev-vm}"
MACHINE="${MACHINE:-g2-standard-8}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
G=(gcloud --project "$PROJECT")
SSH=("${G[@]}" compute ssh "$VM" --zone "$ZONE" --quiet --command)

exists() { "${G[@]}" compute instances describe "$VM" --zone "$ZONE" >/dev/null 2>&1; }
status() { "${G[@]}" compute instances describe "$VM" --zone "$ZONE" --format="value(status)"; }

start_wait() {
  [ "$(status)" = "RUNNING" ] || "${G[@]}" compute instances start "$VM" --zone "$ZONE"
  for _ in $(seq 1 60); do "${SSH[@]}" "true" 2>/dev/null && return 0; sleep 5; done
  echo "VM did not accept ssh in 5 min. NOT deleting it; check it by hand." >&2; exit 1
}

case "${1:-}" in
setup)
  if ! exists; then
    echo "== creating $VM ($MACHINE + 1 L4) in $PROJECT $ZONE"
    "${G[@]}" compute instances create "$VM" --zone "$ZONE" --machine-type "$MACHINE" \
      --accelerator type=nvidia-l4,count=1 --maintenance-policy TERMINATE \
      --image-family "${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}" --image-project deeplearning-platform-release \
      --boot-disk-size 100GB --metadata install-nvidia-driver=True
  fi
  start_wait
  echo "== uploading project + models (slow links: this can take a while)"
  tar -C "$ROOT" --exclude=.venv --exclude=data --exclude=runs --exclude=results --exclude=report --exclude=__pycache__ \
      --exclude=.git --exclude='*.npz' -czf /tmp/vm_pack.tgz .
  "${G[@]}" compute scp /tmp/vm_pack.tgz "$VM":~/ --zone "$ZONE" --quiet
  "${SSH[@]}" "set -e; mkdir -p techfest-face && tar -xzf vm_pack.tgz -C techfest-face
    sudo apt-get install -y python3-venv libgl1 >/dev/null
    cd techfest-face && python3 -m venv .venv && . .venv/bin/activate
    pip install -q numpy pillow opencv-python-headless insightface onnxruntime
    ls models/glintr100.onnx models/buffalo_l/w600k_r50.onnx"
  echo "== setup done. Test it with: bash tools/vm_eval.sh run test <a folder with train/ and test/>   then: bash tools/vm_eval.sh stop"
  ;;
run)
  GROUP="${2:?group name, e.g. group10}"; PACK="${3:?folder that contains train/ and test/}"
  [ -d "$PACK/train" ] && [ -d "$PACK/test" ] || { echo "$PACK must contain train/ and test/"; exit 1; }
  exists || { echo "VM $VM does not exist: run setup first"; exit 1; }
  start_wait
  tar -C "$PACK" -czf /tmp/vm_lab.tgz train test
  "${G[@]}" compute scp /tmp/vm_lab.tgz "$VM":~/ --zone "$ZONE" --quiet
  "${SSH[@]}" "set -e; rm -rf lab && mkdir lab && tar -xzf vm_lab.tgz -C lab
    cd techfest-face/submission && . ../.venv/bin/activate
    ${GPU:+FACE_PROVIDER=cuda} python make_submission.py '$GROUP' ~/lab"
  "${G[@]}" compute scp "$VM":~/techfest-face/submission/"$GROUP".npz "$ROOT/submission/" --zone "$ZONE" --quiet
  "$ROOT/.venv/bin/python" -I - "$ROOT/submission/$GROUP.npz" <<'EOF'
import sys, numpy as np
z = np.load(sys.argv[1]); print({k: z[k].shape for k in z})
assert z["train_emb"].shape[1] == z["test_emb"].shape[1]
assert not np.isnan(z["train_emb"]).any() and not np.isnan(z["test_emb"]).any()
print("OK: shapes match, no NaN")
EOF
  echo "== result: $ROOT/submission/$GROUP.npz   (VM is still running; stop it with: bash tools/vm_eval.sh stop)"
  ;;
stop)
  "${G[@]}" compute instances stop "$VM" --zone "$ZONE"
  ;;
*) sed -n 2,7p "$0"; exit 1 ;;
esac
# Note: ONNX Runtime GPU crashed on CUDA 12.9 earlier in this project, so the VM runs on CPU unless GPU=1 is set and works.
