# Error analysis: word 1-2-gram TF-IDF + LR, condition A (raw), rating-free test set

The model makes 2090 errors on the 21807 rating-free test reviews (1038 false negatives = gold pos predicted neg, 1052 false positives). We drew 40 errors at random (numpy default_rng(0), `results/errors_sample.jsonl`) and read every review in full, assigning one primary category each. The data come from the HuggingFace mirror, which has no star ratings, so 'lukewarm' is judged from the text.

| category | count | share | example (id, gold label) |
|---|---|---|---|
| Mixed review: praise and criticism of parts, overall verdict lost | 9 | 22% | "This could have been a wonderful movie, but they all blew it up. [...] I loved the first part [...] (especially the scene with the liquor shooting was excellent)" (test_12258, gold neg, p_pos=0.88) |
| Plot/content description dominates (sentiment words describe the story) | 6 | 15% | "The unknown assailant ties Betty to a column, gags her, and places needles under her eyes that will cause incredible damage and pain should Betty close them." (test_22397, gold pos, p_pos=0.42) |
| Sentiment aimed at a non-film target (real person, TV channel, other reviewer, extra) | 6 | 15% | ""Bill Hicks dies of cancer" said the headline in the NME. I felt like someone had punched me in the stomach. Buy this DVD." (test_24124, gold pos, p_pos=0.34) |
| Comparison to other films/episodes | 4 | 10% | "This same exact plot from this movie has been done before. It has been done in Ferris Bueller's Day Off! [...] I like Immature better when they were in House Party 3" (test_00769, gold neg, p_pos=0.77) |
| Text contradicts gold label (likely borderline rating or label noise) | 3 | 8% | "Very predictable ending. The actress that plays "Amy" is actually 33 years old trying to play a teenager!" (test_15550, gold pos, p_pos=0.19) |
| Hedged / understated sentiment, few polar words | 3 | 8% | "I felt that the movie Skammen, directed by Ingmar Bergman, was very dry. [...] I feel that it was not complete or attention holding." (test_01705, gold neg, p_pos=0.57) |
| Contrast turn: long praise, then negative verdict ('Well, bad.', 'However') | 3 | 8% | "And the original Demons is an absolute must-see for fans of eighties B-horror films. So how bad can this be, right? Well, bad." (test_10340, gold neg, p_pos=0.60) |
| 'So bad it's good': negative words used as praise | 3 | 8% | "Stupidly beautiful. This movie epitomizes the 'so bad it's good' genre of films." (test_20103, gold pos, p_pos=0.05) |
| Sarcasm / ironic humour | 2 | 5% | "Great [-] you've just imagined the entirety of the film `India Song.' [...] scream out `Look at me! I am PROFOUND!!'" (test_04246, gold neg, p_pos=0.70) |
| Negation ('a good horror flick this is not') | 1 | 2% | "Unspeakable tries to be sick and disturbing and it does manage to do that as a good horror flick this is not." (test_04171, gold neg, p_pos=0.61) |

Sample: 40 errors, gold pos 21 / gold neg 19.

Secondary tag - rating expression missed by the regexes (leak into the 'rating-free' set): 2/40 sampled errors: test_01887 ('***' star rating before the title line not matched); test_13672 ('I give it ten stars' not matched (verb_word_stars allows only zero-five)).

## All rating-free test errors by length and confidence

| length bucket (word-count quartiles) | n | errors | error rate % |
|---|---|---|---|
| Q1 (0-124 words) | 5594 | 529 | 9.46 |
| Q2 (125-167 words) | 5374 | 504 | 9.38 |
| Q3 (168-265 words) | 5409 | 539 | 9.96 |
| Q4 (266-max words) | 5430 | 518 | 9.54 |

| model confidence |p-0.5|*2 | n | errors | error rate % | share of errors % |
|---|---|---|---|---|
| confidence [0.0, 0.2) | 1282 | 558 | 43.53 | 26.7 |
| confidence [0.2, 0.4) | 1487 | 471 | 31.67 | 22.5 |
| confidence [0.4, 0.6) | 1895 | 405 | 21.37 | 19.4 |
| confidence [0.6, 0.8) | 3008 | 373 | 12.40 | 17.8 |
| confidence [0.8, 1.0] | 14135 | 283 | 2.00 | 13.5 |

## What the errors suggest for DistilBERT

- **Most errors are about the overall verdict, not about vocabulary.** Mixed reviews, contrast turns, comparisons
  and hedged reviews make up 19 of the 40 sampled errors. In these the review has the right polar words, but a
  bag of n-grams cannot tell which part the final verdict follows. A contextual model that tracks discourse
  ("but", "that said", "could have been") should fix many of these. They are also where the head+tail truncation
  matters, since the verdict usually comes at the end.
- **About a third of the errors are hard for any text classifier**: plot or content description, sentiment
  aimed at something other than the film, and reviews whose text contradicts the gold label (15/40). These set a
  ceiling, so we should not expect DistilBERT to close the whole gap, and the A-vs-B difference has to be read
  against these errors, which neither condition can fix.
- **Regex recall matters for the frozen test set.** 2 of the 40 sampled errors still contain a rating ("I give it
  ten stars", a "***" star line before a title). A probe of the 21,807 rating-free test reviews finds up to about
  68 similar strings (23 "six-ten stars", 15 "*** Title (year)" lines, 30 "x of 10", some of which are not
  ratings), i.e. ≤0.3%. A model that learns the shortcut (condition A) could still use them, so we will add
  these patterns before the DistilBERT runs. The current IDs stay frozen for the baseline numbers.


## Per-error labels

| # | id | gold | p_pos | words | category | note | secondary |
|---|---|---|---|---|---|---|---|
| 1 | test_00079 | neg | 0.75 | 346 | Plot/content description dominates (sentiment words describe the story) | long plot/cast description; only verdict is a mild parenthetical at the end |  |
| 2 | test_00397 | neg | 0.53 | 108 | Comparison to other films/episodes | recommends other films (Afrika, 'classic' Stir Crazy); 'critics love this movie' |  |
| 3 | test_00527 | neg | 0.89 | 305 | Text contradicts gold label (likely borderline rating or label noise) | text defends the film ('period charm', 'played quite well'); gold neg - likely borderline rating |  |
| 4 | test_00769 | neg | 0.77 | 95 | Comparison to other films/episodes | plot retold, then 'done before in Ferris Bueller', 'I like Immature better in House Party 3' |  |
| 5 | test_00936 | neg | 0.79 | 212 | Mixed review: praise and criticism of parts, overall verdict lost | counterfactual 'would have been excellent'; praises songs and concert footage, blames editing |  |
| 6 | test_01705 | neg | 0.57 | 151 | Hedged / understated sentiment, few polar words | polite, hedged criticism ('very dry', 'not complete'); no strong negative words |  |
| 7 | test_01887 | neg | 0.77 | 183 | Plot/content description dominates (sentiment words describe the story) | plot and cast history dominate; 'wasted resources' is the only verdict | leak: '***' star rating before the title line not matched |
| 8 | test_02194 | neg | 0.50 | 165 | Comparison to other films/episodes | discursive comparison with Thunderball and other Bond films |  |
| 9 | test_04171 | neg | 0.61 | 116 | Negation ('a good horror flick this is not') | 'as a good horror flick this is not'; 'my favorite movie company', 'interesting' |  |
| 10 | test_04246 | neg | 0.70 | 468 | Sarcasm / ironic humour | ironic 'Great', 'PROFOUND', 'COMICALLY bereft of profundity' |  |
| 11 | test_06632 | neg | 0.71 | 124 | Mixed review: praise and criticism of parts, overall verdict lost | 'two shining moments' praised at length ('incredible', 'spectacular', 'great') |  |
| 12 | test_06890 | neg | 0.60 | 161 | Sentiment aimed at a non-film target (real person, TV channel, other reviewer, extra) | praise is for the real Diane Arbus ('brilliant, talented'), not the film |  |
| 13 | test_07394 | neg | 0.63 | 101 | Sarcasm / ironic humour | humorous understatement ('I tried to love this movie ... it was fourteen') |  |
| 14 | test_07481 | neg | 0.53 | 147 | Hedged / understated sentiment, few polar words | hedged ('maybe it's unfair to dislike'); negativity aimed at depicted bigots |  |
| 15 | test_09490 | neg | 0.54 | 504 | Plot/content description dominates (sentiment words describe the story) | long plot summary; praise for scenery ('awesome', 'glorious') |  |
| 16 | test_10340 | neg | 0.60 | 408 | Contrast turn: long praise, then negative verdict ('Well, bad.', 'However') | long praise of soundtrack, cast and first film, then 'Well, bad.' |  |
| 17 | test_11869 | neg | 0.68 | 341 | Contrast turn: long praise, then negative verdict ('Well, bad.', 'However') | 'decent documentary ... great depiction ... that said, glaring flaws' |  |
| 18 | test_12258 | neg | 0.88 | 166 | Mixed review: praise and criticism of parts, overall verdict lost | 'could have been a wonderful movie'; 'loved the first part', 'excellent', 'perfect' |  |
| 19 | test_12422 | neg | 0.86 | 548 | Contrast turn: long praise, then negative verdict ('Well, bad.', 'However') | praises the two leads, score and visuals, then 'However ... film fails to deliver' |  |
| 20 | test_13341 | pos | 0.10 | 170 | Mixed review: praise and criticism of parts, overall verdict lost | positive verdict, but 'real turkeys' (other films), 'pretty terrible' editing |  |
| 21 | test_13374 | pos | 0.29 | 135 | Mixed review: praise and criticism of parts, overall verdict lost | praise of gags, then 'the only drawback is the plot' |  |
| 22 | test_13672 | pos | 0.05 | 123 | 'So bad it's good': negative words used as praise | 'special effects laughable, acting ridiculous ... that's a good thing' | leak: 'I give it ten stars' not matched (verb_word_stars allows only zero-five) |
| 23 | test_13790 | pos | 0.12 | 226 | Plot/content description dominates (sentiment words describe the story) | plot summary with horror vocabulary ('disembowel', 'nasty') |  |
| 24 | test_14872 | pos | 0.41 | 108 | Hedged / understated sentiment, few polar words | lukewarm 'not the greatest movie ever made but ... something to think about renting' |  |
| 25 | test_15547 | pos | 0.28 | 134 | Text contradicts gold label (likely borderline rating or label noise) | text is negative about the film ('needs to pay you to see', 'save your money'); positive only about the actress |  |
| 26 | test_15550 | pos | 0.19 | 143 | Text contradicts gold label (likely borderline rating or label noise) | plot summary + 'very predictable ending', actress too old; no praise at all; gold pos |  |
| 27 | test_15942 | pos | 0.12 | 122 | Sentiment aimed at a non-film target (real person, TV channel, other reviewer, extra) | production history, actress's career, Reagan/Clinton digression |  |
| 28 | test_16580 | pos | 0.09 | 51 | Mixed review: praise and criticism of parts, overall verdict lost | 'everything else about it was so-so', 'annoying'; positive only because of one player |  |
| 29 | test_17977 | pos | 0.20 | 122 | Mixed review: praise and criticism of parts, overall verdict lost | 'completely in love' but most of the text is about a disturbing plot twist ('bad taste') |  |
| 30 | test_18135 | pos | 0.36 | 142 | 'So bad it's good': negative words used as praise | 'puerile, adolescent, inoffensive, idiotic.. And utterly hilarious' |  |
| 31 | test_19028 | pos | 0.27 | 283 | Mixed review: praise and criticism of parts, overall verdict lost | praises acting and locations but 'utterly bleak', 'disappointed', 'too drained' |  |
| 32 | test_19961 | pos | 0.14 | 122 | Plot/content description dominates (sentiment words describe the story) | black comedy: negative words describe the hospital and the character ('waste', 'cynic') |  |
| 33 | test_20103 | pos | 0.05 | 262 | 'So bad it's good': negative words used as praise | 'Stupidly beautiful ... so bad it's good'; 'ugly pile of celluloid' |  |
| 34 | test_20907 | pos | 0.43 | 146 | Comparison to other films/episodes | other Twilight Zone time-travel episodes 'more to my liking'; one flaw discussed |  |
| 35 | test_21020 | pos | 0.48 | 365 | Mixed review: praise and criticism of parts, overall verdict lost | praises direction, photography and score, but 'particularly weak script all but does the movie in' |  |
| 36 | test_21169 | pos | 0.16 | 206 | Sentiment aimed at a non-film target (real person, TV channel, other reviewer, extra) | mostly about an uncredited extra ('WHO IS THIS GIRL???') |  |
| 37 | test_21365 | pos | 0.39 | 137 | Sentiment aimed at a non-film target (real person, TV channel, other reviewer, extra) | rebuts another commenter ('viciously hidden remarks', 'prejudice') |  |
| 38 | test_22397 | pos | 0.42 | 730 | Plot/content description dominates (sentiment words describe the story) | long plot summary full of violent vocabulary (killer, mutilates, stabs) |  |
| 39 | test_23086 | pos | 0.27 | 110 | Sentiment aimed at a non-film target (real person, TV channel, other reviewer, extra) | complains about a TV channel not airing the show |  |
| 40 | test_24124 | pos | 0.34 | 74 | Sentiment aimed at a non-film target (real person, TV channel, other reviewer, extra) | tribute to Bill Hicks's death ('dies of cancer', 'punched me in the stomach') |  |
