"""One-off C sweep on the DEV set only (condition A, dev seed 0). Test data are not touched.

python tune_c.py  ->  results/c_sweep.json ; the chosen C values are then frozen
as defaults in run_baseline.py (--C-word / --C-char).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from run_baseline import ROOT, char_vec, load_data, word_vec
from sklearn.linear_model import LogisticRegression

GRID = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]


def main():
    train_all, _, _ = load_data(ROOT / "data")
    y = np.array([r["label"] for r in train_all])
    tr, dv = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=0)
    tr, dv = sorted(tr), sorted(dv)
    Xtr_t = [train_all[i]["text"] for i in tr]
    Xdv_t = [train_all[i]["text"] for i in dv]
    out = {}
    for kind in sys.argv[1:] or ["word", "char"]:
        t = time.time()
        vec = word_vec() if kind == "word" else char_vec(500_000)
        Xtr, Xdv = vec.fit_transform(Xtr_t), None
        Xdv = vec.transform(Xdv_t)
        res = {}
        for C in GRID:
            clf = LogisticRegression(C=C, solver="liblinear", max_iter=1000, random_state=0).fit(Xtr, y[tr])
            res[C] = float(np.mean(clf.predict(Xdv) == y[dv]))
            print(kind, C, res[C], f"{time.time() - t:.0f}s", flush=True)
        best = max(GRID, key=lambda c: (res[c], -c))
        out[kind] = dict(dev_acc=res, best_C=best)
    Path(ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / "c_sweep.json", "w"), indent=2)
    print(out)


if __name__ == "__main__":
    main()
