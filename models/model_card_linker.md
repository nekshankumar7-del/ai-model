# Model card - field linker

Version: 0.1.0  
Generated: 2026-11-16

## What it does

Scores one (text line, field) hypothesis: given a line read off a page and a candidate field from the knowledge file, how likely is it that this line is that field? It re-ranks the candidates the rule matcher produces, it does not replace them.

It is a re-ranker rather than a 107-way classifier on purpose: the knowledge file is designed so you can add a field without retraining, and a flat softmax over field keys would break that every time you edited the file.

## Training data

Candidate hypotheses generated from the labelled pages in the train and validation splits. A hypothesis is positive when the value parsed under that field matches the label file. Splits are grouped by (writer, client) so no hand or person crosses a split.

## Inputs and outputs

Input `features`: float32 [batch, N] in the exact order given by `models.linker.featureNames` in factfind-spec.json, with the scaling in `features.scaling` already applied.

Output `probabilities`: float32 [batch, 2]; column 1 is P(this is the right field).

## Metrics

| metric | value |
| --- | ---: |
| cv_average_precision | 0.9466 |
| cv_roc_auc | 0.9883 |
| cv_folds | 5 |

## Known failures

- Fields whose labels collide (several money fields share the alias 'savings' or 'other') are genuinely ambiguous from the label alone; the model leans on the section heading, so an unheaded note is harder.
- Narrative notes with several facts per sentence are the weakest layout.
- It cannot recover a field whose label the knowledge file does not list: the candidate never gets generated.

## Not to be used for

Not a document classifier and not an OCR model. It assumes the text it is given was read from the page; it has no way to tell that the recogniser misread a word.
