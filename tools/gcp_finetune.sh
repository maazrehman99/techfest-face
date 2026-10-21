#!/usr/bin/env bash
# Fine-tune on a one-GPU (L4) Google Cloud VM, then bring back runs/ft1 and delete the VM.
#
#   bash tools/gcp_finetune.sh check  PROJECT_ID     # read-only: account, project, L4 quota, image family
#   bash tools/gcp_finetune.sh run    PROJECT_ID     # pack data -> create VM -> train -> download -> delete VM
#   bash tools/gcp_finetune.sh resume PROJECT_ID     # re-run the remote steps on the VM that is already up (data already uploaded)
#   bash tools/gcp_finetune.sh delete PROJECT_ID     # delete the VM and its disk by hand (if a run was interrupted)
#
# Needs: `gcloud auth login` done for the account that owns PROJECT_ID, and data/ft_manifest.csv
# (build_ft_data.py + build_imfdb.py already run). Run from the repo root.
#
# What leaves this laptop: the 112x112 aligned crops of halfFace and IMFDB, ft_manifest.csv, the training code and
# models/w600k_r50.onnx (the public InsightFace weights). NOT uploaded: Wikimedia crops (INCLUDE_PKWIKI=0 default),
# original photos, galleries, pk_test. The VM is deleted at the end and by a 3 hour hard stop, so cost stays small.
set -euo pipefail

CMD="${1:-}"; PROJECT="${2:-}"
[ -n "$CMD" ] && [ -n "$PROJECT" ] || { sed -n 2,6p "$0"; exit 1; }

ZONE="${ZONE:-asia-south1-b}"                # Mumbai, L4 available (override with ZONE=...)
VM="${VM:-maaz-dev-vm}"
MACHINE="${MACHINE:-g2-standard-8}"        # g2 = 1x NVIDIA L4 (24 GB), 8 vCPU
IMAGE_FAMILY="${IMAGE_FAMILY:-}"           # empty = pick the newest PyTorch GPU Deep Learning VM family
INCLUDE_PKWIKI="${INCLUDE_PKWIKI:-0}"      # 1 = also upload the Wikimedia crops
EPOCHS="${EPOCHS:-15}"
OUT="runs/ft1"
G=(gcloud --project "$PROJECT")

check() {
  echo "== account / project"; gcloud config get-value account 2>/dev/null; echo "$PROJECT"
  echo "== L4 quota in ${ZONE%-*} (limit must be >= 1)"
  "${G[@]}" compute regions describe "${ZONE%-*}" --format="value(quotas)" | tr ';' '\n' | grep -i -E "NVIDIA_L4|GPUS_ALL_REGIONS" || echo "no L4 quota line found: request it at IAM & Admin > Quotas"
  echo "== L4 available in $ZONE?"
  "${G[@]}" compute accelerator-types list --filter="name=nvidia-l4 AND zone:$ZONE" --format="value(name,zone)"
  echo "== Deep Learning image family"; pick_family
}

pick_family() {
  if [ -z "$IMAGE_FAMILY" ]; then
    IMAGE_FAMILY=$("${G[@]}" compute images list --project deeplearning-platform-release --no-standard-images \
      --filter="family~^pytorch-.*-cu1[0-9]+" --format="value(family)" | sort -uV | tail -1)
  fi
  [ -n "$IMAGE_FAMILY" ] || { echo "no PyTorch image family found, set IMAGE_FAMILY=..."; exit 1; }
  echo "$IMAGE_FAMILY"
}

pack() {
  rm -rf /tmp/ft_pack && mkdir -p /tmp/ft_pack/code /tmp/ft_pack/data /tmp/ft_pack/models
  .venv/bin/python - "$INCLUDE_PKWIKI" <<'PY'
import sys, os, shutil, pandas as pd
inc = sys.argv[1] == "1"
ft = pd.read_csv("data/ft_manifest.csv")
if not inc:
    ft = ft[ft.source != "pk_wiki"]
for c in ft.crop:
    d = os.path.join("/tmp/ft_pack/data", c)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(os.path.join("data", c), d)
ft.to_csv("/tmp/ft_pack/data/ft_manifest.csv", index=False)
print(len(ft), "crops,", ft.identity.nunique(), "people:")
print(ft.groupby(["split", "source"]).agg(images=("crop", "size"), people=("identity", "nunique")))
PY
  cp finetune.py augment.py evalkit.py evaluate.py pipeline.py config.json /tmp/ft_pack/code/
  mkdir -p /tmp/ft_pack/code/tools && cp tools/eval_crops.py /tmp/ft_pack/code/tools/
  # only the detector + recognizer pipeline.py loads (the other buffalo_l files are not used)
  mkdir -p /tmp/ft_pack/code/models/buffalo_l
  cp models/buffalo_l/det_10g.onnx models/buffalo_l/w600k_r50.onnx /tmp/ft_pack/code/models/buffalo_l/
  # original val+test photos for the full-pipeline evaluation (detect -> align -> embed), train photos stay home
  .venv/bin/python - <<'PY'
import os, shutil, pandas as pd
for name, man, root in (("halfface", "data/halfface/manifest_ft.csv", "data/halfface"),
                        ("imfdb", "data/imfdb/manifest_eval.csv", "data/imfdb")):
    df = pd.read_csv(man); df = df[df.split.isin(["val", "test"])]
    for p in df.path:
        d = os.path.join("/tmp/ft_pack/evaldata", name, p); os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(os.path.join(root, p), d)
    df.to_csv(f"/tmp/ft_pack/evaldata/{name}/manifest.csv", index=False)
    print(name, "eval photos:", len(df), "people:", df.identity.nunique())
PY
  tar -C /tmp/ft_pack -czf /tmp/ft_pack.tgz . && ls -lh /tmp/ft_pack.tgz
}

OK=0
cleanup() {  # delete the VM only after a SUCCESSFUL run; on failure keep it so it can be fixed in place (it still self-deletes after 3 h)
  if [ "$OK" = 1 ]; then echo "== deleting VM"; del
  else echo "== run not successful: VM $VM KEPT for fixing (auto-deletes within 3 h of creation). Remove by hand: bash tools/gcp_finetune.sh delete $PROJECT"; fi
}
del() { "${G[@]}" compute instances delete "$VM" --zone "$ZONE" --quiet || true; }

run() {
  [ -f data/ft_manifest.csv ] || { echo "run build_ft_data.py and build_imfdb.py first"; exit 1; }
  pick_family >/dev/null; echo "image family: $IMAGE_FAMILY"
  if [ "${RESUME:-0}" = 1 ]; then
    echo "== RESUME: reusing the running VM $VM and the data already on it"; trap cleanup EXIT
  else
    pack
    echo "== creating VM $VM ($MACHINE + 1 L4, $ZONE). It deletes itself after 3 h at the latest."
    if [ -n "$("${G[@]}" compute instances list --filter="name=$VM" --format="value(name)")" ]; then
      echo "a VM named $VM already exists in $PROJECT: refusing to touch it (set VM=other-name)"; exit 1
    fi
    "${G[@]}" compute instances create "$VM" --zone "$ZONE" --machine-type "$MACHINE" \
      --accelerator type=nvidia-l4,count=1 --maintenance-policy TERMINATE \
      --image-family "$IMAGE_FAMILY" --image-project deeplearning-platform-release \
      --boot-disk-size 100GB --metadata install-nvidia-driver=True \
      --max-run-duration 3h --instance-termination-action DELETE
    trap cleanup EXIT   # only after our own create succeeded
    echo "== waiting for ssh + GPU driver (a few minutes)"
    for i in $(seq 1 40); do
      if "${G[@]}" compute ssh "$VM" --zone "$ZONE" --quiet --command "nvidia-smi -L" 2>/dev/null; then break; fi
      sleep 15
    done
    "${G[@]}" compute scp /tmp/ft_pack.tgz "$VM":~/ --zone "$ZONE" --quiet
  fi
  "${G[@]}" compute ssh "$VM" --zone "$ZONE" --quiet --command "
    set -e; [ -d ft/code ] || { mkdir -p ft && tar -xzf ft_pack.tgz -C ft; }; cd ft
    sudo apt-get update -qq >/dev/null 2>&1; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-full build-essential libgl1 libglib2.0-0 >/dev/null 2>&1 || true
    python3 -m venv --system-site-packages .v; . .v/bin/activate
    pip install -q onnx onnx2torch onnxruntime-gpu matplotlib pandas scikit-learn opencv-python-headless pillow tqdm
    pip install -q insightface || echo 'insightface install failed: full-pipeline stages will fail, crop stages still run'
    python -c 'import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())'
    # preflight: every import the stages need, plus the GPU for onnxruntime. Fails the whole run in a minute, not after 5 wasted stages.
    python -c 'import cv2,onnx,onnx2torch,onnxruntime,insightface,sklearn,pandas,matplotlib;p=onnxruntime.get_available_providers();print(\"PREFLIGHT_OK\",p);assert \"CUDAExecutionProvider\" in p'
    set +e   # from here a failed stage must not stop the others or the download
    cd code
    full() {  # full pipeline on the original val+test photos; \$1=name  \$2=optional extra args (--rec file)
      for d in halfface imfdb; do
        o=../$OUT/eval/full_\$1_\$d; l=../full_\$1_\$d.log
        if ! FACE_PROVIDER=cuda FACE_WORKERS=2 python evaluate.py --manifest ../evaldata/\$d/manifest.csv \
              --root ../evaldata/\$d --out \$o --ablation \$2 > \$l 2>&1; then
          echo \"GPU run failed for \$d, see \$l; retrying on CPU\"; tail -n 5 \$l
          FACE_WORKERS=6 python evaluate.py --manifest ../evaldata/\$d/manifest.csv \
              --root ../evaldata/\$d --out \$o --ablation \$2 > \$l 2>&1
        fi
        tail -n 12 \$l
      done
    }
    echo '=== 0/4 BEFORE, full pipeline (detect+align+embed), GPU'
    full baseline '' || echo 'full-pipeline baseline failed, continuing'
    echo '=== 1/4 BEFORE: frozen baseline on the held-out crops'
    python tools/eval_crops.py --manifest ../data/ft_manifest.csv --root ../data --rec models/buffalo_l/w600k_r50.onnx \
        --name baseline --out ../$OUT/eval 2>&1 | tee ../eval_baseline.log
    echo '=== 2/4 fine-tune'
    python finetune.py --manifest ../data/ft_manifest.csv --root ../data --out ../$OUT \
        --onnx models/buffalo_l/w600k_r50.onnx --allow-halfface --epochs $EPOCHS --workers 4 2>&1 | tee ../train.log
    echo '=== 3/4 AFTER: fine-tuned on the same held-out crops'
    python tools/eval_crops.py --manifest ../data/ft_manifest.csv --root ../data --rec ../$OUT/finetuned.onnx \
        --name finetuned --out ../$OUT/eval 2>&1 | tee ../eval_finetuned.log
    echo '=== 4/4 AFTER, full pipeline, GPU'
    full finetuned \"--rec ../$OUT/finetuned.onnx\" || echo 'full-pipeline finetuned failed'
    mkdir -p ../runs/ft1; cp ../*.log ../runs/ft1/ 2>/dev/null; echo ALL_STAGES_DONE" || echo "== a remote command failed: downloading whatever exists"
  mkdir -p runs && "${G[@]}" compute scp --recurse "$VM":~/ft/$OUT runs/ --zone "$ZONE" --quiet \
    || { trap - EXIT; echo "== download failed: VM left in place, fix and re-run the scp by hand, then: bash tools/gcp_finetune.sh delete $PROJECT"; exit 1; }
  if [ -f "runs/ft1/finetuned.onnx" ]; then
    OK=1; echo "== done: runs/ft1/ (finetuned.onnx, log.txt, history.csv). VM is deleted on exit."
  else
    echo "== FAILED: no runs/ft1/finetuned.onnx came back (see runs/ft1/*.log). VM is deleted on exit."
  fi
}

case "$CMD" in
  check) check;; run) run;; resume) RESUME=1 run;; delete) del;; *) echo "unknown command $CMD"; exit 1;;
esac
