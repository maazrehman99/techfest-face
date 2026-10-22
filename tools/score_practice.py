import sys, csv, numpy as np
z = np.load(sys.argv[1]); key = {r["test_id"]: r["person"] for r in csv.DictReader(open(sys.argv[2]))}
n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
S = n(z["test_emb"]) @ n(z["train_emb"]).T; t1 = t5 = 0
for i, t in enumerate(z["test_ids"]):
    names = [z["train_ids"][j].rsplit(".", 1)[0] for j in np.argsort(-S[i])]; r = names.index(key[t]) + 1
    t1 += r == 1; t5 += r <= 5; print(t, key[t], "rank", r)
print("top1", t1, "/", len(key), " top5", t5, "/", len(key))
