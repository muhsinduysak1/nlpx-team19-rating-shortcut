"""DistilBERT A-vs-B pilot run (mid-project): small-data, CPU-friendly version of the main experiment.

Same data pipeline as run_baseline.py (same HTML cleaning, same rating regexes and scrub(),
same stratified 10% dev split with random_state=0). Differences from the planned main runs
are deliberate and only about size: a stratified training subset (default 2,000 reviews,
the proposal's low-data condition), 1 epoch, max length 256 with head+tail truncation
(head 63 + tail 191 tokens, the proposal's 1:3 head/tail ratio scaled down from 128+382).

A = unmodified training reviews, B = the same reviews with explicit ratings scrubbed.
Everything else is identical: subset, order, seed, hyper-parameters.

Usage:
    python distilbert_pilot.py --model_dir models/distilbert-base-uncased --cond A B
Outputs results/distilbert_pilot.json and results/distilbert_pilot_preds_{A,B}.jsonl
"""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from ratings import has_rating, scrub
from run_baseline import load_data

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_head_tail(tok, texts, max_len):
    """Head+tail truncation: keep the first `head` and last `tail` word-piece tokens."""
    budget = max_len - 2
    head = round(budget * 128 / 510)
    tail = budget - head
    ids_all = tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
    out = []
    for ids in ids_all:
        if len(ids) > budget:
            ids = ids[:head] + ids[-tail:]
        out.append([tok.cls_token_id] + ids + [tok.sep_token_id])
    return out


def batches(seqs, labels, bs, pad_id, shuffle_rng=None):
    order = list(range(len(seqs)))
    if shuffle_rng is not None:
        shuffle_rng.shuffle(order)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        L = max(len(seqs[j]) for j in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            ids[r, :len(seqs[j])] = torch.tensor(seqs[j])
            att[r, :len(seqs[j])] = 1
        yield idx, ids, att, torch.tensor([labels[j] for j in idx])


@torch.no_grad()
def predict(model, seqs, labels, bs, pad_id, dev=torch.device("cpu")):
    model.eval()
    preds = np.zeros(len(seqs), dtype=int)
    probs = np.zeros(len(seqs))
    # sort by length for speed, then scatter back
    order = sorted(range(len(seqs)), key=lambda j: len(seqs[j]))
    s_seqs = [seqs[j] for j in order]
    s_lab = [labels[j] for j in order]
    for idx, ids, att, _ in batches(s_seqs, s_lab, bs, pad_id):
        p = torch.softmax(model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float(),
                          -1)[:, 1].cpu().numpy()
        for r, k in enumerate(idx):
            probs[order[k]] = p[r]
            preds[order[k]] = int(p[r] >= 0.5)
    return preds, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default="models/distilbert-base-uncased")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--cond", nargs="+", default=["A", "B"])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro_bs", type=int, default=4, help="micro-batch for gradient accumulation (memory); effective batch = --bs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto | cpu | mps (Apple GPU) | cuda")
    ap.add_argument("--n_dev", type=int, default=0, help="0 = full dev split")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    dev = torch.device(args.device)
    log("device:", dev)
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    train_all, _test, source = load_data(Path(args.data_dir))
    del _test  # the pilot only uses train/dev; free memory
    labels_all = [r["label"] for r in train_all]
    tr_idx, dev_idx = train_test_split(np.arange(len(train_all)), test_size=0.1,
                                       stratify=labels_all, random_state=0)
    tr = [train_all[i] for i in sorted(tr_idx)]
    dv = [train_all[i] for i in sorted(dev_idx)]
    # fixed stratified pilot subset, identical for A and B
    if args.n_train >= len(tr):
        sub = tr
    else:
        sub_idx, _ = train_test_split(np.arange(len(tr)), train_size=args.n_train,
                                      stratify=[r["label"] for r in tr], random_state=0)
        sub = [tr[i] for i in sorted(sub_idx)]
    if args.n_dev:
        d_idx, _ = train_test_split(np.arange(len(dv)), train_size=args.n_dev,
                                    stratify=[r["label"] for r in dv], random_state=0)
        dv = [dv[i] for i in sorted(d_idx)]
    dev_rf_mask = np.array([not has_rating(r["text"]) for r in dv])
    n_rated_sub = sum(has_rating(r["text"]) for r in sub)
    log(f"source={source} pilot train={len(sub)} (rated {n_rated_sub}) dev={len(dv)} "
        f"(rating-free {int(dev_rf_mask.sum())})")

    del train_all, tr
    gc.collect()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    y_tr = [r["label"] for r in sub]
    y_dv = np.array([r["label"] for r in dv])
    results = dict(source=source, settings=vars(args), n_train=len(sub), n_train_rated=n_rated_sub,
                   n_dev=len(dv), n_dev_rating_free=int(dev_rf_mask.sum()),
                   torch=torch.__version__, runs={})
    res_path = out / "distilbert_pilot.json"
    if res_path.exists():
        results["runs"] = json.loads(res_path.read_text()).get("runs", {})

    for cond in args.cond:
        f = (lambda t: t) if cond == "A" else scrub
        tr_seqs = encode_head_tail(tok, [f(r["text"]) for r in sub], args.max_len)
        dv_seqs = encode_head_tail(tok, [f(r["text"]) for r in dv], args.max_len)
        set_seed(args.seed)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=2).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        steps = args.epochs * ((len(tr_seqs) + args.bs - 1) // args.bs)
        sched = get_linear_schedule_with_warmup(opt, 0, steps)
        rng = random.Random(args.seed)
        t_start, step = time.time(), 0
        for ep in range(args.epochs):
            model.train()
            for _, ids, att, y in batches(tr_seqs, y_tr, args.bs, tok.pad_token_id, rng):
                tot = 0.0
                for k in range(0, len(y), args.micro_bs):
                    sl = slice(k, k + args.micro_bs)
                    l = model(input_ids=ids[sl].to(dev), attention_mask=att[sl].to(dev),
                              labels=y[sl].to(dev)).loss
                    (l * len(y[sl]) / len(y)).backward()
                    tot += l.item() * len(y[sl]) / len(y)
                loss = torch.tensor(tot)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0 or step == steps:
                    el = time.time() - t_start
                    log(f"{cond} ep{ep} step {step}/{steps} loss {loss.item():.4f} "
                        f"({el / step:.1f}s/step, eta {el / step * (steps - step) / 60:.1f} min)")
        train_min = (time.time() - t_start) / 60
        t_eval = time.time()
        preds, probs = predict(model, dv_seqs, list(y_dv), 16, tok.pad_token_id, dev)
        correct = preds == y_dv
        r = dict(dev_acc=float(correct.mean()),
                 dev_rating_free_acc=float(correct[dev_rf_mask].mean()),
                 dev_rated_acc=float(correct[~dev_rf_mask].mean()) if (~dev_rf_mask).any() else None,
                 train_minutes=round(train_min, 1), eval_minutes=round((time.time() - t_eval) / 60, 1),
                 steps=steps, device=str(dev))
        results["runs"][f"{cond}_seed{args.seed}"] = r
        log(cond, r)
        with open(out / f"distilbert_pilot_preds_{cond}_seed{args.seed}.jsonl", "w") as fh:
            for rec, p, pr in zip(dv, preds, probs):
                fh.write(json.dumps(dict(id=rec["id"], gold=rec["label"], pred=int(p),
                                         prob_pos=round(float(pr), 4))) + "\n")
        res_path.write_text(json.dumps(results, indent=2))
        del model, opt, sched
        gc.collect()
    log("done ->", res_path)


if __name__ == "__main__":
    main()
