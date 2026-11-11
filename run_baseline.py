"""Team 19 baseline: does scrubbing in-text ratings change what a classifier learns?

One command:   python run_baseline.py
Writes results/results.json, results/results.md, frozen ID lists,
results/errors_sample.jsonl and (with annotations/error_categories.csv)
results/error_analysis.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from ratings import (PATTERNS, clean_html, find_ratings, invert_x10, last_x10_score,
                     scrub)

ROOT = Path(__file__).resolve().parent
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ data
def load_data(data_dir: Path):
    """Primary: HuggingFace stanfordnlp/imdb parquet (plain_text). Fallback: aclImdb/ directory."""
    pq_train = data_dir / "train-00000-of-00001.parquet"
    pq_test = data_dir / "test-00000-of-00001.parquet"
    if pq_train.exists() and pq_test.exists():
        import pyarrow.parquet as pq
        out = {}
        for split, p in (("train", pq_train), ("test", pq_test)):
            t = pq.read_table(p)
            texts, labels = t.column("text").to_pylist(), t.column("label").to_pylist()
            out[split] = [dict(id=f"{split}_{i:05d}", label=int(y), stars=None, text=clean_html(x))
                          for i, (x, y) in enumerate(zip(texts, labels))]
        return out["train"], out["test"], "huggingface.co/datasets/stanfordnlp/imdb (plain_text, parquet)"
    root = ensure_extracted(data_dir)
    return load_split(root, "train"), load_split(root, "test"), "aclImdb_v1 directory"


def ensure_extracted(data_dir: Path) -> Path:
    if (data_dir / "aclImdb" / "train" / "pos").is_dir():
        return data_dir / "aclImdb"
    tgz = data_dir / "aclImdb_v1.tar.gz"
    if not tgz.exists():
        sys.exit(f"Data not found in {data_dir}: need the two HF parquet files, aclImdb_v1.tar.gz or aclImdb/")
    log("extracting", tgz)
    with tarfile.open(tgz) as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and re.match(r"^aclImdb/(train|test)/(pos|neg)/\d+_\d+\.txt$", m.name)]
        tf.extractall(data_dir, members=members, filter="data")
    return data_dir / "aclImdb"


def load_split(root: Path, split: str):
    recs = []
    for lab_name, lab in (("neg", 0), ("pos", 1)):
        d = root / split / lab_name
        files = sorted(d.glob("*.txt"), key=lambda p: int(p.stem.split("_")[0]))
        for p in files:
            rid, rating = p.stem.split("_")
            raw = p.read_text(encoding="utf-8")
            recs.append(dict(id=f"{split}/{lab_name}/{p.name}", label=lab, stars=int(rating),
                             text=clean_html(raw)))
    return recs


# ------------------------------------------------------------------ models
def word_vec():
    # token_pattern keeps 1-char tokens so "7/10" -> bigram "7 10"
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b", lowercase=True, dtype=np.float32)


def char_vec(max_features):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                           lowercase=True, max_features=max_features, dtype=np.float32)


def make_lr(C):
    return LogisticRegression(C=C, solver="liblinear", max_iter=1000, random_state=0)


def fit_predict(kind, train_texts, y_train, eval_sets, char_max_features, C):
    vec = word_vec() if kind == "word" else char_vec(char_max_features)
    X = vec.fit_transform(train_texts)
    clf = make_lr(C).fit(X, y_train)
    out = {}
    for name, texts in eval_sets.items():
        Xe = vec.transform(texts)
        out[name] = clf.predict_proba(Xe)[:, 1].astype(np.float64)
    return vec, clf, out, X.shape[1]


def run_one(kind, cond, seed, train_texts, y_tr, evals, ys, char_max_features, C):
    t1 = time.time()
    vec, clf, probs, nfeat = fit_predict(kind, train_texts, y_tr, evals, char_max_features, C)
    res = dict(n_features=nfeat, fit_seconds=round(time.time() - t1, 1))
    for k in ("dev", "test_rf", "test_rated", "test_rated_scrubbed"):
        if k in probs:
            res[k] = float(np.mean((probs[k] > 0.5) == ys[k]))
    if "flip_orig" in probs:
        po, pi = probs["flip_orig"] > 0.5, probs["flip_inv"] > 0.5
        res["flip"] = dict(n=int(len(po)), flips=int(np.sum(po != pi)), rate=float(np.mean(po != pi)),
                           pos_to_neg=int(np.sum(po & ~pi)), neg_to_pos=int(np.sum(~po & pi)))
    feats = None
    if kind == "word" and cond == "A" and seed == 0:
        names = vec.get_feature_names_out()
        coef = clf.coef_[0]
        rating_like = [i for i, f in enumerate(names) if re.fullmatch(r"(?:\d+ )?\d+ (?:10|5)", f)]
        rl = sorted(rating_like, key=lambda i: coef[i])
        order = np.argsort(-np.abs(coef))
        rank = {int(i): k + 1 for k, i in enumerate(order[:5000])}
        feats = dict(
            most_negative=[(names[i], round(float(coef[i]), 2), rank.get(i)) for i in rl[:8]],
            most_positive=[(names[i], round(float(coef[i]), 2), rank.get(i)) for i in rl[::-1][:8]],
            top_overall_neg=[(names[i], round(float(coef[i]), 2)) for i in np.argsort(coef)[:10]],
            top_overall_pos=[(names[i], round(float(coef[i]), 2)) for i in np.argsort(-coef)[:10]],
            note="tuples: (feature, LR coefficient, rank by |coef| among all features if in top 5000)")
    return cond, res, probs["test_rf"], feats


# ------------------------------------------------------------------ stats
def bootstrap_acc(correct: np.ndarray, n_boot=10000, seed=0, chunk=500):
    rng = np.random.default_rng(seed)
    n = len(correct)
    c = correct.astype(np.float32)
    accs = []
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - s), n))
        accs.append(c[idx].mean(axis=1))
    accs = np.concatenate(accs)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def mcnemar_exact(ca: np.ndarray, cb: np.ndarray):
    b = int(np.sum(ca & ~cb))   # A right, B wrong
    c = int(np.sum(~ca & cb))   # A wrong, B right
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, p


def pct(x):
    return round(100.0 * x, 2)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="dev-split seeds; seed 0 is the primary run")
    ap.add_argument("--char-seeds", type=int, nargs="+", default=None,
                    help="seeds on which the char model is run (default: same as --seeds)")
    ap.add_argument("--char-max-features", type=int, default=500_000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--C-word", type=float, default=32.0,
                    help="frozen; best dev acc in tune_c.py (condition A, dev seed 0)")
    ap.add_argument("--C-char", type=float, default=8.0,
                    help="frozen; best dev acc in tune_c.py (condition A, dev seed 0)")
    ap.add_argument("--n-errors", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=2, help="conditions A and B are fitted in parallel")
    args = ap.parse_args()
    char_seeds = args.seeds if args.char_seeds is None else args.char_seeds
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = {"config": dict(seeds=args.seeds, char_seeds=char_seeds,
                        char_max_features=args.char_max_features, n_boot=args.n_boot,
                        word="TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, "
                             "token_pattern=r'(?u)\\b\\w+\\b')",
                        char="TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), min_df=2, "
                             f"sublinear_tf=True, max_features={args.char_max_features})",
                        lr=f"LogisticRegression(C=C, solver='liblinear', random_state=0); C_word={args.C_word}, "
                           f"C_char={args.C_char}: tuned ONCE on dev (condition A, dev seed 0, grid 0.25..128, "
                           "tune_c.py -> results/c_sweep.json) and frozen for all conditions and seeds",
                        preprocessing="'<br />' -> ' ' before everything; ratings scrubbed by ratings.scrub",
                        versions=dict(python=platform.python_version(), numpy=np.__version__,
                                      scipy=scipy.__version__, sklearn=sklearn.__version__))}

    train_all, test, source = load_data(Path(args.data_dir))
    R["data_source"] = source
    log(f"loaded train={len(train_all)} test={len(test)} from {source}")

    # ---- rating detection
    for r in train_all + test:
        ms = find_ratings(r["text"])
        r["matches"] = ms
        r["rated"] = bool(ms)
        r["x10"] = last_x10_score(r["text"]) if ms else None
    log("regex done")

    def counts(recs):
        c = dict(n=len(recs), pos=sum(r["label"] for r in recs))
        c["neg"] = c["n"] - c["pos"]
        rated = [r for r in recs if r["rated"]]
        c["rated"] = len(rated)
        c["rated_pct"] = pct(len(rated) / max(1, len(recs)))
        c["rated_pos"] = sum(r["label"] for r in rated)
        c["rated_neg"] = len(rated) - c["rated_pos"]
        c["rating_free"] = len(recs) - len(rated)
        c["rating_free_pos"] = c["pos"] - c["rated_pos"]
        c["rating_free_neg"] = c["neg"] - c["rated_neg"]
        return c

    regex_stats = {}
    for name, recs in (("train", train_all), ("test", test)):
        per_pat = Counter()
        for r in recs:
            for p in {m.pattern for m in r["matches"]}:
                per_pat[p] += 1
        # position of the LAST match (start offset / length)
        posn = np.array([r["matches"][-1].start / max(1, len(r["text"])) for r in recs if r["rated"]])
        x10 = [(r["x10"], r["label"]) for r in recs if r["x10"] is not None]
        agree = [(1 if x > 5 else 0) == y for x, y in x10]
        regex_stats[name] = dict(
            reviews_with_pattern={p: per_pat.get(p, 0) for p in PATTERNS},
            median_last_match_position_pct=pct(float(np.median(posn))) if len(posn) else None,
            share_last_match_in_last_10pct=pct(float(np.mean(posn >= 0.9))) if len(posn) else None,
            n_x10=len(x10), x10_label_agreement_pct=pct(float(np.mean(agree))) if agree else None)
    both_x10 = [(r["x10"], r["label"]) for r in train_all + test if r["x10"] is not None]
    regex_stats["train+test"] = dict(
        n_x10=len(both_x10),
        x10_label_agreement_pct=pct(float(np.mean([(1 if x > 5 else 0) == y for x, y in both_x10]))))
    regex_stats["x10_rule"] = "predict pos iff last x/10 score > 5"
    R["regex"] = regex_stats

    # ---- frozen test sets
    test_rf = [r for r in test if not r["rated"]]
    test_rated = [r for r in test if r["rated"]]
    (out / "test_rating_free_ids.txt").write_text("\n".join(r["id"] for r in test_rf) + "\n")
    (out / "test_rated_ids.txt").write_text("\n".join(r["id"] for r in test_rated) + "\n")
    y_rf = np.array([r["label"] for r in test_rf])
    y_rated = np.array([r["label"] for r in test_rated])
    rf_texts = [r["text"] for r in test_rf]
    rated_raw = [r["text"] for r in test_rated]
    rated_scrub = [scrub(t) for t in rated_raw]
    flip_recs = [r for r in test_rated if r["x10"] is not None]
    flip_orig = [r["text"] for r in flip_recs]
    flip_inv = [invert_x10(t) for t in flip_orig]
    # sanity: scrubbing the rating-free test set is the identity
    assert all(scrub(t) == t for t in rf_texts[:2000])

    # ---- data example for the slide: shortest rated training review with an x/10 score
    cands = sorted([r for r in train_all if r["x10"] is not None and len(r["matches"]) == 1
                    and 250 <= len(r["text"]) and r["matches"][0].end >= len(r["text"].rstrip()) - 1],
                   key=lambda r: len(r["text"]))
    ex = cands[0]
    R["example"] = dict(id=ex["id"], label="pos" if ex["label"] else "neg",
                        raw=ex["text"], scrubbed=scrub(ex["text"]),
                        matches=[m.text for m in ex["matches"]])

    # ---- seed loop
    runs = {}
    keep = {}
    labels_all = np.array([r["label"] for r in train_all])
    for seed in args.seeds:
        idx = np.arange(len(train_all))
        tr_idx, dev_idx = train_test_split(idx, test_size=0.1, stratify=labels_all, random_state=seed)
        tr = [train_all[i] for i in sorted(tr_idx)]
        dv = [train_all[i] for i in sorted(dev_idx)]
        if seed == 0:
            (out / "dev_ids.txt").write_text("\n".join(r["id"] for r in dv) + "\n")
            R["counts"] = dict(train_full=counts(train_all), train=counts(tr), dev=counts(dv),
                               test=counts(test))
        y_tr = np.array([r["label"] for r in tr])
        y_dv = np.array([r["label"] for r in dv])
        texts = {"A": ([r["text"] for r in tr], [r["text"] for r in dv]),
                 "B": ([scrub(r["text"]) for r in tr], [scrub(r["text"]) for r in dv])}
        n_changed = sum(a != b for a, b in zip(texts["A"][0], texts["B"][0]))
        log(f"seed {seed}: train={len(tr)} dev={len(dv)}; scrub changed {n_changed} train texts")

        # majority class (ties broken towards the lower label, sklearn convention)
        maj = int(np.bincount(y_tr).argmax())
        runs[(seed, "majority", "-")] = dict(
            dev=float(np.mean(y_dv == maj)), test_rf=float(np.mean(y_rf == maj)),
            test_rated=float(np.mean(y_rated == maj)), majority_label=maj)

        for kind in ("word", "char"):
            if kind == "char" and seed not in char_seeds:
                continue
            jobs = []
            for cond in ("A", "B"):
                evals = {"dev": texts[cond][1], "test_rf": rf_texts, "test_rated": rated_raw}
                if seed == 0:
                    evals.update(test_rated_scrubbed=rated_scrub, flip_orig=flip_orig,
                                 flip_inv=flip_inv)
                jobs.append(delayed(run_one)(kind, cond, seed, texts[cond][0], y_tr, evals,
                                             dict(dev=y_dv, test_rf=y_rf, test_rated=y_rated,
                                                  test_rated_scrubbed=y_rated),
                                             args.char_max_features,
                                             args.C_word if kind == "word" else args.C_char))
            for cond, res, p_rf, feats in Parallel(n_jobs=args.n_jobs)(jobs):
                if seed == 0:
                    keep[(kind, cond)] = p_rf
                    if feats is not None:
                        R["rating_features_wordA"] = feats
                runs[(seed, kind, cond)] = res
                log(f"seed {seed} {kind} {cond}: dev={res['dev']:.4f} test_rf={res['test_rf']:.4f} "
                    f"rated={res['test_rated']:.4f} feats={res['n_features']} ({res['fit_seconds']}s)")

    R["runs"] = [dict(seed=s, system=k, condition=c, **v) for (s, k, c), v in runs.items()]

    # ---- seed-0 inference: bootstrap CIs, paired CI, McNemar
    inf = {}
    rng_note = f"{args.n_boot} resamples of the {len(y_rf)} rating-free test reviews, numpy default_rng(0)"
    maj_c = (y_rf == runs[(0, "majority", "-")]["majority_label"])
    inf["majority"] = dict(acc=float(maj_c.mean()), ci95=bootstrap_acc(maj_c, args.n_boot))
    for kind in ("word", "char"):
        if ("char" == kind) and 0 not in char_seeds:
            continue
        ca = (keep[(kind, "A")] > 0.5) == y_rf
        cb = (keep[(kind, "B")] > 0.5) == y_rf
        d = cb.astype(np.int8) - ca.astype(np.int8)
        b, c, p = mcnemar_exact(ca, cb)
        rng = np.random.default_rng(0)
        diffs = []
        for s in range(0, args.n_boot, 500):
            idx = rng.integers(0, len(d), size=(min(500, args.n_boot - s), len(d)))
            diffs.append(d[idx].astype(np.float32).mean(axis=1))
        diffs = np.concatenate(diffs)
        inf[kind] = dict(
            A=dict(acc=float(ca.mean()), ci95=bootstrap_acc(ca, args.n_boot)),
            B=dict(acc=float(cb.mean()), ci95=bootstrap_acc(cb, args.n_boot)),
            B_minus_A=dict(diff=float(d.mean()), ci95=(float(np.percentile(diffs, 2.5)),
                                                       float(np.percentile(diffs, 97.5)))),
            mcnemar=dict(A_right_B_wrong=b, A_wrong_B_right=c, p_exact=p))
    inf["note"] = rng_note + "; same resample indices for every system (paired)"
    R["inference_seed0"] = inf

    # ---- seed spread
    spread = {}
    for kind in ("word", "char"):
        ss = [s for s in args.seeds if (kind == "word" or s in char_seeds)]
        if not ss:
            continue
        a = np.array([runs[(s, kind, "A")]["test_rf"] for s in ss])
        bb = np.array([runs[(s, kind, "B")]["test_rf"] for s in ss])
        dva = np.array([runs[(s, kind, "A")]["dev"] for s in ss])
        dvb = np.array([runs[(s, kind, "B")]["dev"] for s in ss])
        sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        spread[kind] = dict(seeds=ss, A_mean=float(a.mean()), A_std=sd(a), B_mean=float(bb.mean()),
                            B_std=sd(bb), diff_mean=float((bb - a).mean()), diff_std=sd(bb - a),
                            devA_mean=float(dva.mean()), devA_std=sd(dva),
                            devB_mean=float(dvb.mean()), devB_std=sd(dvb))
    R["seed_spread"] = spread

    # ---- errors: word model, condition A, rating-free test
    pw = keep[("word", "A")]
    pred = (pw > 0.5).astype(int)
    err_idx = np.where(pred != y_rf)[0]
    wrong = pred != y_rf
    nwords = np.array([len(r["text"].split()) for r in test_rf])
    qs = np.percentile(nwords, [25, 50, 75])
    edges = [0, qs[0], qs[1], qs[2], np.inf]
    by_len = {}
    for k in range(4):
        m = (nwords > edges[k]) & (nwords <= edges[k + 1]) if k else (nwords <= edges[1])
        by_len[f"Q{k+1} ({int(edges[k]) + (1 if k else 0)}-{'max' if k == 3 else int(edges[k+1])} words)"] = dict(
            n=int(m.sum()), errors=int(np.sum(m & wrong)), error_rate=float(np.mean(wrong[m])))
    conf = np.abs(pw - 0.5) * 2  # 0 = unsure, 1 = certain
    by_conf = {}
    for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        m = (conf >= lo) & (conf < hi)
        by_conf[f"confidence [{lo:.1f}, {min(hi, 1.0):.1f}{']' if hi > 1 else ')'}"] = dict(
            n=int(m.sum()), errors=int(np.sum(m & wrong)),
            share_of_errors=float(np.sum(m & wrong) / max(1, wrong.sum())),
            error_rate=float(np.mean(wrong[m])) if m.sum() else None)
    by_star = {}
    if test_rf[0]["stars"] is not None:
        for s in (1, 2, 3, 4, 7, 8, 9, 10):
            m = np.array([r["stars"] == s for r in test_rf])
            by_star[s] = dict(n=int(m.sum()), errors=int(np.sum(m & wrong)),
                              error_rate=float(np.mean(wrong[m])) if m.sum() else None)
    fn = int(np.sum((pred == 0) & (y_rf == 1)))
    fp = int(np.sum((pred == 1) & (y_rf == 0)))
    R["errors_wordA"] = dict(n_errors=int(len(err_idx)), false_neg=fn, false_pos=fp, by_length_quartile=by_len,
                            by_confidence=by_conf, by_star=by_star)
    rng = np.random.default_rng(0)
    sample = sorted(rng.choice(err_idx, size=min(args.n_errors, len(err_idx)), replace=False).tolist())
    with open(out / "errors_sample.jsonl", "w", encoding="utf-8") as f:
        for k, i in enumerate(sample):
            r = test_rf[i]
            f.write(json.dumps(dict(n=k + 1, id=r["id"], gold="pos" if r["label"] else "neg",
                                    stars=r["stars"], pred="pos" if pred[i] else "neg",
                                    prob_pos=round(float(pw[i]), 4), n_words=len(r["text"].split()),
                                    text=r["text"]), ensure_ascii=False) + "\n")

    R["runtime_seconds"] = round(time.time() - T0, 1)
    with open(out / "results.json", "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o if not isinstance(o, tuple) else list(o))
    write_md(R, out)
    ann = ROOT / "annotations" / "error_categories.csv"
    if ann.exists():
        import make_error_analysis
        make_error_analysis.build(out / "errors_sample.jsonl", ann, out / "results.json",
                                  out / "error_analysis.md")
    log("done; wrote", out)


def write_md(R, out: Path):
    L = []
    c = R["counts"]
    L.append("# Team 19 baseline results\n")
    L.append(f"LR: {R['config']['lr']}.\n")
    L.append(f"Data: {R['data_source']} (same train/test split as Maas et al. 2011 v1.0). "
             f"Runtime: {R['runtime_seconds']} s. Versions: {R['config']['versions']}.\n")
    L.append("## Data counts (after each step)\n")
    L.append("| split | n | pos | neg | rated | rated % | rating-free | rf pos | rf neg |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in ("train_full", "train", "dev", "test"):
        v = c[k]
        L.append(f"| {k} | {v['n']} | {v['pos']} | {v['neg']} | {v['rated']} | {v['rated_pct']} | "
                 f"{v['rating_free']} | {v['rating_free_pos']} | {v['rating_free_neg']} |")
    rx = R["regex"]
    L.append("\n## Regex statistics\n")
    for s in ("train", "test"):
        v = rx[s]
        L.append(f"- {s}: median position of last match {v['median_last_match_position_pct']}% of text; "
                 f"{v['share_last_match_in_last_10pct']}% in last 10%; x/10 reviews {v['n_x10']}, "
                 f"label agreement {v['x10_label_agreement_pct']}%")
        L.append(f"  - reviews per pattern: {v['reviews_with_pattern']}")
    L.append(f"- train+test x/10: n={rx['train+test']['n_x10']}, agreement "
             f"{rx['train+test']['x10_label_agreement_pct']}% ({rx['x10_rule']})")
    inf = R["inference_seed0"]
    run0 = {(r["system"], r["condition"]): r for r in R["runs"] if r["seed"] == 0}
    L.append("\n## Main table (dev seed 0; rating-free test is the frozen primary test set)\n")
    L.append("| system | cond | dev acc | rating-free test acc [95% CI] | rated test acc (raw) | rated test acc (scrubbed) | flip rate |")
    L.append("|---|---|---|---|---|---|---|")
    m = run0[("majority", "-")]
    L.append(f"| majority | - | {pct(m['dev'])} | {pct(inf['majority']['acc'])} "
             f"[{pct(inf['majority']['ci95'][0])}, {pct(inf['majority']['ci95'][1])}] | {pct(m['test_rated'])} | - | - |")
    for kind in ("word", "char"):
        if kind not in inf:
            continue
        for cond in ("A", "B"):
            r = run0[(kind, cond)]
            ci = inf[kind][cond]["ci95"]
            L.append(f"| {kind} TF-IDF+LR | {cond} | {pct(r['dev'])} | {pct(r['test_rf'])} "
                     f"[{pct(ci[0])}, {pct(ci[1])}] | {pct(r['test_rated'])} | {pct(r['test_rated_scrubbed'])} | "
                     f"{pct(r['flip']['rate'])}% ({r['flip']['flips']}/{r['flip']['n']}) |")
    L.append("\n## B - A on rating-free test (seed 0)\n")
    L.append("| system | B-A (points) | paired 95% CI | McNemar b (A only right) | c (B only right) | exact p |")
    L.append("|---|---|---|---|---|---|")
    for kind in ("word", "char"):
        if kind not in inf:
            continue
        d = inf[kind]["B_minus_A"]
        mc = inf[kind]["mcnemar"]
        L.append(f"| {kind} | {100*d['diff']:+.2f} | [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}] | "
                 f"{mc['A_right_B_wrong']} | {mc['A_wrong_B_right']} | {mc['p_exact']:.3g} |")
    L.append(f"\n{inf['note']}.\n")
    L.append("## Seed spread (dev-split seeds; test acc on rating-free test, mean ± std)\n")
    L.append("| system | seeds | A | B | B-A | dev A | dev B |")
    L.append("|---|---|---|---|---|---|---|")
    for kind, v in R["seed_spread"].items():
        L.append(f"| {kind} | {v['seeds']} | {pct(v['A_mean'])} ± {pct(v['A_std'])} | "
                 f"{pct(v['B_mean'])} ± {pct(v['B_std'])} | {100*v['diff_mean']:+.2f} ± {100*v['diff_std']:.2f} | "
                 f"{pct(v['devA_mean'])} ± {pct(v['devA_std'])} | {pct(v['devB_mean'])} ± {pct(v['devB_std'])} |")
    L.append("\n## All runs\n")
    L.append("| seed | system | cond | dev | test_rf | test_rated | n_features | fit s |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in R["runs"]:
        L.append(f"| {r['seed']} | {r['system']} | {r['condition']} | {pct(r['dev'])} | {pct(r['test_rf'])} | "
                 f"{pct(r['test_rated'])} | {r.get('n_features', '-')} | {r.get('fit_seconds', '-')} |")
    if "rating_features_wordA" in R:
        f = R["rating_features_wordA"]
        L.append("\n## Rating-like features of word model A (feature, coef, |coef| rank)\n")
        L.append(f"- most negative: {f['most_negative']}")
        L.append(f"- most positive: {f['most_positive']}")
        L.append(f"- top overall negative: {f['top_overall_neg']}")
        L.append(f"- top overall positive: {f['top_overall_pos']}")
    e = R["errors_wordA"]
    L.append("\n## Word A errors on rating-free test\n")
    L.append(f"Total errors {e['n_errors']} (false negatives {e['false_neg']}, false positives {e['false_pos']}).\n")
    L.append("By review length (word-count quartiles of the rating-free test set):\n")
    L.append("| bucket | n | errors | error rate % |")
    L.append("|---|---|---|---|")
    for s, v in e["by_length_quartile"].items():
        L.append(f"| {s} | {v['n']} | {v['errors']} | {pct(v['error_rate'])} |")
    L.append("\nBy model confidence (|p_pos - 0.5| * 2):\n")
    L.append("| confidence | n | errors | error rate % | share of all errors % |")
    L.append("|---|---|---|---|---|")
    for s, v in e["by_confidence"].items():
        L.append(f"| {s} | {v['n']} | {v['errors']} | {pct(v['error_rate']) if v['error_rate'] is not None else '-'} | {pct(v['share_of_errors'])} |")
    if e["by_star"]:
        L.append("\n| stars | n | errors | error rate % |")
        L.append("|---|---|---|---|")
        for s, v in e["by_star"].items():
            L.append(f"| {s} | {v['n']} | {v['errors']} | {pct(v['error_rate']) if v['error_rate'] is not None else '-'} |")
    ex = R["example"]
    L.append(f"\n## Data example ({ex['id']}, label {ex['label']})\n")
    L.append(f"- raw: {ex['raw']}")
    L.append(f"- scrubbed: {ex['scrubbed']}")
    L.append(f"- matched: {ex['matches']}")
    (out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
