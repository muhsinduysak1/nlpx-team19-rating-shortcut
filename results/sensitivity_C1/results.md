# Team 19 baseline results

Data: huggingface.co/datasets/stanfordnlp/imdb (plain_text, parquet) (same train/test split as Maas et al. 2011 v1.0). Runtime: 861.0 s. Versions: {'python': '3.13.18', 'numpy': '2.5.3', 'scipy': '1.18.1', 'sklearn': '1.9.1'}.

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
| word TF-IDF+LR | A | 89.28 | 89.34 [88.93, 89.75] | 91.92 | 89.54 | 6.06% (140/2312) |
| word TF-IDF+LR | B | 88.68 | 89.45 [89.04, 89.85] | 90.17 | 89.6 | 1.73% (40/2312) |
| char TF-IDF+LR | A | 89.24 | 88.65 [88.24, 89.06] | 90.01 | 88.41 | 3.42% (79/2312) |
| char TF-IDF+LR | B | 89.0 | 88.65 [88.23, 89.06] | 88.73 | 88.6 | 0.39% (9/2312) |

## B - A on rating-free test (seed 0)

| system | B-A (points) | paired 95% CI | McNemar b (A only right) | c (B only right) | exact p |
|---|---|---|---|---|---|
| word | +0.11 | [+0.02, +0.19] | 32 | 55 | 0.0178 |
| char | +0.00 | [-0.07, +0.07] | 31 | 31 | 1 |

10000 resamples of the 21807 rating-free test reviews, numpy default_rng(0); same resample indices for every system (paired).

## Seed spread (dev-split seeds; test acc on rating-free test, mean ± std)

| system | seeds | A | B | B-A | dev A | dev B |
|---|---|---|---|---|---|---|
| word | [0, 1, 2] | 89.36 ± 0.02 | 89.44 ± 0.01 | +0.08 ± 0.02 | 89.75 ± 0.55 | 89.27 ± 0.62 |
| char | [0, 1, 2] | 88.71 ± 0.06 | 88.72 ± 0.09 | +0.02 ± 0.04 | 89.6 ± 0.77 | 89.31 ± 0.71 |

## All runs

| seed | system | cond | dev | test_rf | test_rated | n_features | fit s |
|---|---|---|---|---|---|---|---|
| 0 | majority | - | 50.0 | 49.78 | 51.49 | - | - |
| 0 | word | A | 89.28 | 89.34 | 91.92 | 398742 | 24.4 |
| 0 | word | B | 88.68 | 89.45 | 90.17 | 398315 | 27.3 |
| 0 | char | A | 89.24 | 88.65 | 90.01 | 266713 | 140.1 |
| 0 | char | B | 89.0 | 88.65 | 88.73 | 266313 | 142.0 |
| 1 | majority | - | 50.0 | 49.78 | 51.49 | - | - |
| 1 | word | A | 89.6 | 89.38 | 91.86 | 400019 | 36.7 |
| 1 | word | B | 89.2 | 89.45 | 90.1 | 399574 | 37.0 |
| 1 | char | A | 89.08 | 88.71 | 90.1 | 267001 | 223.3 |
| 1 | char | B | 88.8 | 88.69 | 88.54 | 266616 | 223.0 |
| 2 | majority | - | 50.0 | 49.78 | 51.49 | - | - |
| 2 | word | A | 90.36 | 89.37 | 92.01 | 399657 | 41.1 |
| 2 | word | B | 89.92 | 89.43 | 90.23 | 399217 | 41.6 |
| 2 | char | A | 90.48 | 88.76 | 90.01 | 267292 | 217.8 |
| 2 | char | B | 90.12 | 88.82 | 88.35 | 266880 | 215.6 |

## Rating-like features of word model A (feature, coef, |coef| rank)

- most negative: [('4 10', -1.93, 126), ('3 10', -1.58, 211), ('2 10', -1.1, 480), ('1 10', -1.06, 533), ('0 10', -0.39, 3477), ('1 5', -0.24, None), ('2 5', -0.11, None), ('3 5', -0.07, None)]
- most positive: [('7 10', 2.05, 111), ('10 10', 1.73, 160), ('8 10', 1.56, 215), ('9 10', 1.13, 444), ('7 5', 0.31, None), ('8 5', 0.23, None), ('5 5', 0.23, None), ('9 5', 0.14, None)]
- top overall negative: [('bad', -8.16), ('worst', -7.06), ('the worst', -5.76), ('awful', -5.27), ('waste', -4.71), ('boring', -4.7), ('poor', -4.63), ('no', -4.55), ('terrible', -4.32), ('nothing', -4.18)]
- top overall positive: [('great', 6.95), ('excellent', 5.11), ('best', 4.17), ('the best', 4.14), ('wonderful', 4.1), ('perfect', 3.92), ('love', 3.83), ('well', 3.49), ('is a', 3.49), ('amazing', 3.48)]

## Word A errors on rating-free test

Total errors 2324 (false negatives 1073, false positives 1251).

By review length (word-count quartiles of the rating-free test set):

| bucket | n | errors | error rate % |
|---|---|---|---|
| Q1 (0-124 words) | 5594 | 575 | 10.28 |
| Q2 (125-167 words) | 5374 | 565 | 10.51 |
| Q3 (168-265 words) | 5409 | 611 | 11.3 |
| Q4 (266-max words) | 5430 | 573 | 10.55 |

By model confidence (|p_pos - 0.5| * 2):

| confidence | n | errors | error rate % | share of all errors % |
|---|---|---|---|---|
| confidence [0.0, 0.2) | 3481 | 1263 | 36.28 | 54.35 |
| confidence [0.2, 0.4) | 4026 | 664 | 16.49 | 28.57 |
| confidence [0.4, 0.6) | 5106 | 293 | 5.74 | 12.61 |
| confidence [0.6, 0.8) | 5700 | 93 | 1.63 | 4.0 |
| confidence [0.8, 1.0] | 3494 | 11 | 0.31 | 0.47 |

## Data example (train_21158, label pos)

- raw: Excellent episode movie ala Pulp Fiction. 7 days - 7 suicides. It doesnt get more depressing than this. Movie rating: 8/10 Music rating: 10/10
- scrubbed: Excellent episode movie ala Pulp Fiction. 7 days - 7 suicides. It doesnt get more depressing than this. Movie Music
- matched: ['rating: 8/10', 'rating: 10/10']
