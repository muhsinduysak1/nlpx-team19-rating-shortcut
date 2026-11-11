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
