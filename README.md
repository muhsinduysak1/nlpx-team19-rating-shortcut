# Team 19 (NLPX) — Rating shortcut in IMDb sentiment: baselines

CSE 484/654 NLP, Fall 2026. Question: does training on IMDb reviews with explicit
in-text ratings ("7/10", "8 out of 10", "***/****", "3 stars") deleted change accuracy
on the test reviews that contain no such rating?

## One command

```bash
pip install -r requirements.txt
python run_baseline.py          # CPU only, ~10 min on 2 CPUs; writes results/
python tests/test_ratings.py    # regex unit tests
```

Put the data in `data/` first (not committed: the dataset has no stated licence, so we do not
redistribute it). `bash download_data.sh` downloads both files and checks their SHA-256:

* train-00000-of-00001.parquet `db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c`
* test-00000-of-00001.parquet `b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da`

Data sources:

* **Primary (used for the reported numbers):** HuggingFace mirror
  `huggingface.co/datasets/stanfordnlp/imdb`, config `plain_text`, files
  `data/train-00000-of-00001.parquet` and `data/test-00000-of-00001.parquet`
  (25,000 + 25,000 rows, 12,500/12,500 each; same train/test split as
  Maas et al. 2011, v1.0). Review IDs = row index in each parquet
  (`train_00123`, `test_04567`). This mirror has no star ratings.
* Fallback: the original `aclImdb_v1.tar.gz` (or extracted `aclImdb/`) in `data/`;
  IDs are then `test/pos/123_8.txt` and true star ratings are used in the error analysis.

## What it does

1. `<br />` → space; `ratings.py` finds explicit ratings with 11 precision-first regexes
   (x/10, x out of 10, x/100, x/5, x out of 5/4, star symbols, "3 stars", "give it two stars",
   "give it a 7", "Rating: 7", letter grades). `scrub()` deletes the matched expression.
   Unit tests: `python tests/test_ratings.py`.
2. Splits: train 25,000 → stratified 10% dev (2,500, `random_state=0`) + 22,500 train.
   Frozen primary test set = test reviews with no regex match
   (`results/test_rating_free_ids.txt`); rated test subset in `results/test_rated_ids.txt`;
   dev IDs in `results/dev_ids.txt`.
3. Conditions: A = raw text, B = ratings scrubbed in train and dev. The rating-free test
   set is identical under A and B.
4. Systems: majority class; word 1–2-gram TF-IDF + LR; char_wb 2–5-gram TF-IDF + LR
   (`max_features=500,000` cap for memory). TF-IDF: `sublinear_tf=True, min_df=2`;
   word tokens keep 1-character tokens so "7/10" becomes the bigram "7 10".
   LR: `solver="liblinear"`, `C_word=32`, `C_char=8`: tuned **once** on the dev set
   (condition A, dev seed 0, grid 0.25–128; `python tune_c.py` → `results/c_sweep.json`)
   and then frozen for both conditions and all seeds. The test set was not used for tuning.
   A first run with C=1 for both models is kept in `results/sensitivity_C1/`.
5. Metrics: accuracy on rating-free test (primary), dev, rated test (raw and scrubbed).
   95% bootstrap CI (10,000 resamples of test reviews, `numpy.random.default_rng(0)`),
   paired bootstrap CI for B−A (same resample indices), exact McNemar (binomial on
   discordant pairs, `scipy.stats.binomtest`). LR is deterministic, so the seed spread is
   over dev-split seeds 0, 1, 2 (mean ± std of rating-free test accuracy).
6. Flip probe: on rated test reviews with an x/10 rating, x → 10−x (`invert_x10`);
   fraction of predictions that change.
7. Error analysis: 40 random errors (seed 0) of word model A on the rating-free test set
   → `results/errors_sample.jsonl`; manual categories in `annotations/error_categories.csv`;
   `make_error_analysis.py` (called by `run_baseline.py`) writes `results/error_analysis.md`.

## DistilBERT A-vs-B pilot (mid-project)

`distilbert_pilot.py` fine-tunes `distilbert-base-uncased` on condition A and on condition B
with the same split, regexes and `scrub()` as the baselines. Pilot size (to fit a laptop):
a fixed stratified subset of 2,000 training reviews (identical for A and B), 1 epoch,
max length 256 with head+tail truncation (first 63 + last 191 word pieces; the proposal's
128 + 382 scaled down), lr 2e-5, effective batch 16, linear decay, seeds 1–3.
Evaluated on the full dev split (2,500) and its rating-free part (2,174).

```bash
pip install -r requirements-mac.txt   # torch, transformers, ...
# model files in models/distilbert-base-uncased/: config.json, tokenizer_config.json (committed),
# model.safetensors and vocab.txt (download from huggingface.co/distilbert/distilbert-base-uncased)
for s in 1 2 3; do python distilbert_pilot.py --cond A B --device mps --micro_bs 16 --seed $s; done
```

Writes `results/distilbert_pilot.json` (one entry per condition and seed) and per-review
dev predictions. The pilot ran on an Apple M1 GPU (MPS), about 3 min per run.
The main runs (22,500 reviews, 512 tokens = 128 + 382, 2 epochs, fp16, seeds 1–3) use the
same script with `--n_train 22500 --max_len 512 --epochs 2` on a GPU.

## Outputs

`results/results.json`, `results/results.md`, `results/error_analysis.md`,
`results/errors_sample.jsonl`, frozen ID files, `results/run.log`.

## Versions and seeds

Python 3.13, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, pyarrow 26.0.1
(`requirements.txt`). Seeds: dev split 0 (primary), 1, 2; bootstrap 0; error sample 0;
LR `random_state=0`. 2 CPUs, < 7 GB RAM.

## Data licence / citation

IMDb Large Movie Review Dataset v1.0 (Maas et al., ACL 2011); no licence stated, the
authors ask to cite the paper. HF card lists licence "other".

## AI use

Claude (Anthropic) wrote and ran the baseline scripts under the team's protocol; the
error categories were assigned by reading every sampled review.
