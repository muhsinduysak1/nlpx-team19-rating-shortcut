"""Build results/error_analysis.md from the 40 sampled errors + manual annotations.

annotations/error_categories.csv columns: n,id,category,note
(one row per error in results/errors_sample.jsonl; categories were assigned by
reading every review in full).  Called automatically by run_baseline.py.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

CATEGORY_DESCRIPTIONS = {}  # filled from annotations/category_descriptions.json if present


def _short(text: str, quote: str | None, max_words=40) -> str:
    s = quote if quote else text
    w = s.split()
    return " ".join(w[:max_words]) + (" ..." if len(w) > max_words else "")


def build(sample_path, ann_path, results_json, out_path):
    sample = {json.loads(l)["id"]: json.loads(l) for l in open(sample_path, encoding="utf-8")}
    rows = list(csv.DictReader(open(ann_path, encoding="utf-8")))
    ids = [r["id"] for r in rows]
    missing = set(sample) - set(ids)
    if missing:
        print(f"[error_analysis] annotations missing for {len(missing)} sampled ids -> skipped", file=sys.stderr)
        return
    rows = [r for r in rows if r["id"] in sample]
    R = json.load(open(results_json))
    desc_path = Path(ann_path).with_name("category_descriptions.json")
    desc = json.load(open(desc_path)) if desc_path.exists() else {}
    cnt = Counter(r["category"] for r in rows)
    n = len(rows)
    L = ["# Error analysis: word 1-2-gram TF-IDF + LR, condition A (raw), rating-free test set\n"]
    e = R["errors_wordA"]
    L.append(f"The model makes {e['n_errors']} errors on the {R['counts']['test']['rating_free']} rating-free test "
             f"reviews ({e['false_neg']} false negatives = gold pos predicted neg, {e['false_pos']} false positives). "
             f"We drew {n} errors at random (numpy default_rng(0), `results/errors_sample.jsonl`) and read every "
             f"review in full, assigning one primary category each. The data come from the HuggingFace mirror, "
             f"which has no star ratings, so 'lukewarm' is judged from the text.\n")
    L.append("| category | count | share | example (id, gold label) |")
    L.append("|---|---|---|---|")
    for cat, c in cnt.most_common():
        ex = next(r for r in rows if r["category"] == cat and r.get("example", "") == "1") if any(
            r["category"] == cat and r.get("example", "") == "1" for r in rows) else next(
            r for r in rows if r["category"] == cat)
        s = sample[ex["id"]]
        q = _short(s["text"], ex.get("quote") or None)
        label = desc.get(cat, cat)
        L.append(f"| {label} | {c} | {100 * c / n:.0f}% | \"{q}\" ({s['id']}, gold {s['gold']}, "
                 f"p_pos={s['prob_pos']:.2f}) |")
    gold = Counter(sample[r["id"]]["gold"] for r in rows)
    L.append(f"\nSample: {n} errors, gold pos {gold['pos']} / gold neg {gold['neg']}.\n")
    leaks = [r for r in rows if r.get("secondary", "").startswith("leak")]
    L.append(f"Secondary tag - rating expression missed by the regexes (leak into the 'rating-free' set): "
             f"{len(leaks)}/{n} sampled errors: " + "; ".join(f"{r['id']} ({r['secondary'][6:]})" for r in leaks) + ".\n")
    L.append("## All rating-free test errors by length and confidence\n")
    L.append("| length bucket (word-count quartiles) | n | errors | error rate % |")
    L.append("|---|---|---|---|")
    for k, v in e["by_length_quartile"].items():
        L.append(f"| {k} | {v['n']} | {v['errors']} | {100 * v['error_rate']:.2f} |")
    L.append("\n| model confidence |p-0.5|*2 | n | errors | error rate % | share of errors % |")
    L.append("|---|---|---|---|---|")
    for k, v in e["by_confidence"].items():
        er = "-" if v["error_rate"] is None else f"{100 * v['error_rate']:.2f}"
        L.append(f"| {k} | {v['n']} | {v['errors']} | {er} | {100 * v['share_of_errors']:.1f} |")
    notes = Path(ann_path).with_name("error_analysis_notes.md")
    if notes.exists():
        L.append("\n" + notes.read_text(encoding="utf-8").strip() + "\n")
    L.append("\n## Per-error labels\n")
    L.append("| # | id | gold | p_pos | words | category | note | secondary |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = sample[r["id"]]
        L.append(f"| {s['n']} | {s['id']} | {s['gold']} | {s['prob_pos']:.2f} | {s['n_words']} | "
                 f"{desc.get(r['category'], r['category'])} | {r['note']} | {r.get('secondary', '')} |")
    Path(out_path).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[error_analysis] wrote {out_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    build(root / "results/errors_sample.jsonl", root / "annotations/error_categories.csv",
          root / "results/results.json", root / "results/error_analysis.md")
