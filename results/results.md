# Team 19 baseline results

LR: LogisticRegression(C=C, solver='liblinear', random_state=0); C_word=32.0, C_char=8.0: tuned ONCE on dev (condition A, dev seed 0, grid 0.25..128, tune_c.py -> results/c_sweep.json) and frozen for all conditions and seeds.

Data: huggingface.co/datasets/stanfordnlp/imdb (plain_text, parquet) (same train/test split as Maas et al. 2011 v1.0). Runtime: 601.3 s. Versions: {'python': '3.13.18', 'numpy': '2.5.3', 'scipy': '1.18.1', 'sklearn': '1.9.1'}.

## Data counts (after each step)

| split | n | pos | neg | rated | rated % | rating-free | rf pos | rf neg |
|---|---|---|---|---|---|---|---|---|
| train_full | 25000 | 12500 | 12500 | 3185 | 12.74 | 21815 | 10928 | 10887 |
| train | 22500 | 11250 | 11250 | 2859 | 12.71 | 19641 | 9839 | 9802 |
| dev | 2500 | 1250 | 1250 | 326 | 13.04 | 2174 | 1089 | 1085 |
| test | 25000 | 12500 | 12500 | 3193 | 12.77 | 21807 | 10951 | 10856 |

## Regex statistics

- train: median position of last match 98.32% of text; 75.76% in last 10%; x/10 reviews 2276, label agreement 96.35%
  - reviews per pattern: {'x10_slash': 1491, 'x10_outof': 797, 'x100_slash': 9, 'x5_slash': 36, 'x5_outof': 54, 'star_symbols': 193, 'digit_stars': 194, 'verb_word_stars': 42, 'verb_number': 326, 'label_number': 32, 'letter_grade': 92}
- test: median position of last match 98.34% of text; 76.7% in last 10%; x/10 reviews 2312, label agreement 96.58%
  - reviews per pattern: {'x10_slash': 1463, 'x10_outof': 865, 'x100_slash': 11, 'x5_slash': 48, 'x5_outof': 60, 'star_symbols': 203, 'digit_stars': 168, 'verb_word_stars': 43, 'verb_number': 311, 'label_number': 36, 'letter_grade': 85}
- train+test x/10: n=4588, agreement 96.47% (predict pos iff last x/10 score > 5)

## Main table (dev seed 0; rating-free test is the frozen primary test set)

| system | cond | dev acc | rating-free test acc [95% CI] | rated test acc (raw) | rated test acc (scrubbed) | flip rate |
|---|---|---|---|---|---|---|
| majority | - | 50.0 | 49.78 [49.11, 50.44] | 51.49 | - | - |
| word TF-IDF+LR | A | 90.84 | 90.42 [90.02, 90.81] | 93.08 | 90.48 | 7.66% (177/2312) |
| word TF-IDF+LR | B | 90.2 | 90.53 [90.14, 90.92] | 91.2 | 90.51 | 1.82% (42/2312) |
| char TF-IDF+LR | A | 89.96 | 88.57 [88.14, 88.98] | 90.7 | 88.32 | 6.1% (141/2312) |
| char TF-IDF+LR | B | 89.76 | 88.61 [88.19, 89.03] | 88.82 | 88.47 | 0.48% (11/2312) |

## B - A on rating-free test (seed 0)

| system | B-A (points) | paired 95% CI | McNemar b (A only right) | c (B only right) | exact p |
|---|---|---|---|---|---|
| word | +0.11 | [+0.02, +0.22] | 47 | 72 | 0.0274 |
| char | +0.05 | [-0.05, +0.14] | 49 | 59 | 0.387 |

10000 resamples of the 21807 rating-free test reviews, numpy default_rng(0); same resample indices for every system (paired).

## Seed spread (dev-split seeds; test acc on rating-free test, mean ± std)

| system | seeds | A | B | B-A | dev A | dev B |
|---|---|---|---|---|---|---|
| word | [0, 1, 2] | 90.38 ± 0.04 | 90.5 ± 0.05 | +0.11 ± 0.06 | 90.96 ± 0.12 | 90.44 ± 0.29 |
| char | [0, 1, 2] | 88.57 ± 0.02 | 88.56 ± 0.06 | -0.00 ± 0.05 | 90.19 ± 1.0 | 90.08 ± 1.04 |

## All runs

| seed | system | cond | dev | test_rf | test_rated | n_features | fit s |
|---|---|---|---|---|---|---|---|
| 0 | majority | - | 50.0 | 49.78 | 51.49 | - | - |
| 0 | word | A | 90.84 | 90.42 | 93.08 | 398742 | 26.2 |
| 0 | word | B | 90.2 | 90.53 | 91.2 | 398315 | 26.9 |
| 0 | char | A | 89.96 | 88.57 | 90.7 | 266713 | 141.1 |
| 0 | char | B | 89.76 | 88.61 | 88.82 | 266313 | 141.9 |
| 1 | majority | - | 50.0 | 49.78 | 51.49 | - | - |
| 1 | word | A | 90.96 | 90.38 | 93.11 | 400019 | 22.5 |
| 1 | word | B | 90.36 | 90.44 | 90.79 | 399574 | 23.0 |
| 1 | char | A | 89.32 | 88.59 | 90.7 | 267001 | 110.2 |
| 1 | char | B | 89.24 | 88.58 | 88.47 | 266616 | 108.9 |
| 2 | majority | - | 50.0 | 49.78 | 51.49 | - | - |
| 2 | word | A | 91.08 | 90.34 | 93.3 | 399657 | 22.6 |
| 2 | word | B | 90.76 | 90.51 | 91.23 | 399217 | 22.5 |
| 2 | char | A | 91.28 | 88.54 | 90.76 | 267292 | 106.7 |
| 2 | char | B | 91.24 | 88.49 | 88.57 | 266880 | 103.9 |

## Rating-like features of word model A (feature, coef, |coef| rank)

- most negative: [('4 10', -8.72, 29), ('3 10', -6.18, 83), ('2 10', -4.51, 212), ('1 10', -3.86, 330), ('0 10', -1.17, None), ('1 5', -0.69, None), ('5 10', -0.57, None), ('2 5', -0.48, None)]
- most positive: [('7 10', 8.45, 33), ('10 10', 5.42, 119), ('8 10', 5.0, 157), ('9 10', 3.71, 359), ('7 5', 1.04, None), ('5 5', 0.8, None), ('4 5', 0.61, None), ('8 5', 0.57, None)]
- top overall negative: [('worst', -17.68), ('bad', -17.05), ('awful', -14.52), ('the worst', -13.77), ('boring', -13.15), ('poor', -12.68), ('waste', -11.95), ('terrible', -11.63), ('dull', -10.87), ('worse', -10.69)]
- top overall positive: [('great', 15.14), ('excellent', 13.74), ('7', 12.05), ('perfect', 11.73), ('wonderful', 10.63), ('amazing', 9.47), ('fun', 9.32), ('the best', 9.17), ('today', 9.05), ('8', 8.66)]

## Word A errors on rating-free test

Total errors 2090 (false negatives 1038, false positives 1052).

By review length (word-count quartiles of the rating-free test set):

| bucket | n | errors | error rate % |
|---|---|---|---|
| Q1 (0-124 words) | 5594 | 529 | 9.46 |
| Q2 (125-167 words) | 5374 | 504 | 9.38 |
| Q3 (168-265 words) | 5409 | 539 | 9.96 |
| Q4 (266-max words) | 5430 | 518 | 9.54 |

By model confidence (|p_pos - 0.5| * 2):

| confidence | n | errors | error rate % | share of all errors % |
|---|---|---|---|---|
| confidence [0.0, 0.2) | 1282 | 558 | 43.53 | 26.7 |
| confidence [0.2, 0.4) | 1487 | 471 | 31.67 | 22.54 |
| confidence [0.4, 0.6) | 1895 | 405 | 21.37 | 19.38 |
| confidence [0.6, 0.8) | 3008 | 373 | 12.4 | 17.85 |
| confidence [0.8, 1.0] | 14135 | 283 | 2.0 | 13.54 |

## Data example (train_11670, label neg)

- raw: tries to be funny and fails miserably. The animation is just terrible, looks like a 2 year old threw it together in his sleep. Plot is dull and cliched. IF you have a young child, maybe rent it. but don't waste hard earned money to pay to see it.  1/10
- scrubbed: tries to be funny and fails miserably. The animation is just terrible, looks like a 2 year old threw it together in his sleep. Plot is dull and cliched. IF you have a young child, maybe rent it. but don't waste hard earned money to pay to see it.
- matched: ['1/10']
