# Model card - confidence

Version: 0.1.0  
Generated: 2026-11-16

## What it does

Predicts whether an extracted value is actually correct, from recogniser confidence, the margin between the top two candidates, fuzzy label score, parser validity, cross-field checks and image quality. Its calibrated output is compared with a per-kind threshold to decide fill versus unsure.

This is the model that controls the wrong-fill rate, which the brief makes priority 1.

## Training data

One example per assigned value on the train split, labelled by whether it matched the label file. A value filled for a field the page never mentioned counts as incorrect. Calibration and all thresholds are fitted on the validation split, never on test.

## Inputs and outputs

Input `features`: float32 [batch, N] in the order given by `models.confidence.featureNames`.

Output `probabilities`: float32 [batch, 2]; column 1 is the raw P(correct). Apply the isotonic knots in `models.confidence.calibration` by linear interpolation to get the calibrated probability, then compare with `decision.thresholds[kind]`.

## Metrics

| metric | value |
| --- | ---: |
| n_train | 4851 |
| n_validation | 1059 |
| train_positive_rate | 0.7438 |
| validation_positive_rate | 0.7734 |
| roc_auc_raw | 0.9660 |
| roc_auc_calibrated | 0.9693 |
| brier_raw | 0.0514 |
| brier_calibrated | 0.0475 |

## Known failures

- A confidently misread value that still parses cleanly is the hard case: a date read as 03/04 instead of 08/04 is valid, plausible and wrong. Cross-field checks catch some of these, not all.
- Calibration is only as good as the validation split; with few writers it will be optimistic on a new hand.
- Fields with very few validation examples get a threshold fitted on thin evidence.

## Not to be used for

The probability is calibrated for the data it was fitted on. Refit the calibration and thresholds whenever the document mix, the writers or the OCR engine change.
