"""Keeps the face database and the waiting list in a PRIVATE Hugging Face dataset, for the online copy of demo/collect.py.

A free Hugging Face Space forgets its files on every restart, so:
  pull()  at start-up copies gallery/ and gallery_inbox/ down from the private dataset
  push()  after every change (a send, a reject, an enrolment) copies them back up, deleting anything removed locally
Nothing happens unless DATA_REPO (and HF_TOKEN, a Space secret) are set, so the laptop version is unchanged.
"""
import os, threading, time

REPO = os.environ.get("DATA_REPO")
FOLDERS = ["gallery", "gallery_inbox"]
SKIP = ["**/.token", "**/.check_token", "**/.admin_token"]     # link codes live in Space secrets, never in the dataset
_lock, _again = threading.Lock(), threading.Event()


def pull(root):
    if not REPO:
        return
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, repo_type="dataset", local_dir=root, allow_patterns=[f + "/**" for f in FOLDERS])
    print("loaded the face database from", REPO, flush=True)


def _upload(root):
    from huggingface_hub import HfApi
    api = HfApi()
    for f in FOLDERS:
        if os.path.isdir(os.path.join(root, f)):
            api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=os.path.join(root, f), path_in_repo=f,
                              ignore_patterns=SKIP, delete_patterns="**", commit_message=f"update {f}")


def push(root):
    """Uploads in the background, so the page answers at once. Several changes close together become one upload."""
    if not REPO:
        return
    _again.set()
    if not _lock.acquire(blocking=False):      # an upload is already running; it will run again after it finishes
        return

    def run():
        try:
            while _again.is_set():
                _again.clear()
                time.sleep(2)
                for attempt in range(3):
                    try:
                        _upload(root); break
                    except Exception as e:      # a failed upload is retried, and the next change uploads everything again
                        print("upload to", REPO, "failed:", e, flush=True); time.sleep(10)
        finally:
            _lock.release()
            if _again.is_set():      # a change arrived just as the upload finished
                push(root)
    threading.Thread(target=run, daemon=True).start()
