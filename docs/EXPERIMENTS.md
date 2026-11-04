# Experiment and reproducibility notes

These observations come from `Project_12.ipynb`. It has 19 cells, including 11 code cells, and no saved outputs. Dataset sizes, hardware measurements, convergence, and predictive performance have not been verified.

## Choose a training workflow

Cell numbers count all cells from 1.

- **Cell 5:** training-oriented implementation. Its main block calls `train_novel()`. Checkpoint selection uses evaluation metrics; configured validation/test text sets are sampled. It saves reports and confusion matrices.
- **Cell 7:** combined training/evaluation implementation. Defaults to `RUN_TRAIN=False` and `RUN_FUSED_EVAL=True`, loading an existing checkpoint. Its training loop defines `current_score = -avg_train_loss`; the printed `Val Score` therefore means negative training loss. Checkpoint selection and early stopping in this variant do not use held-out validation performance.

Both write `datasets/processed_data/best_novel.pt`. Preserve separate checkpoints when comparing variants. Record the workflow, configuration, dependency versions, seeds, and dataset manifest for each experiment.

## Evaluation scope

The combined evaluation cell defaults to `TEACHER_N=10000`, limiting text test data when larger. Memotion uses its internal test partition. `EXPORT_SAMPLE=None` controls export sampling independently. Set `TEACHER_N=None` for all available text test examples and record actual per-source/class counts.

Report macro-F1, per-class precision/recall/F1, confusion matrices, and per-source metrics. Keep test sets isolated when tuning thresholds, temperature, sampling, or architecture.

## Data and features

- Text is stratified by binary label, not source or three-class target. Exact deduplication does not remove every paraphrase or related example.
- Memotion is shuffled with seed 123 into 80/10/10 partitions without class stratification or grouping by template/image.
- Loaders rebuild paths under `datasets/processed_data/memotion_dataset_7k/images/`; configure this root to ensure vision is used.
- Metadata uses supplied sarcasm/humour labels. Inference expects caller-supplied flags, defaulting to false; assess this distribution difference.
- Cleaning lowercases and normalizes punctuation. EDA on cleaned files cannot recover those original signals.
- Positive binary-only examples become three-class Offensive. The binary head treats both Abusive and Offensive as positive.

## Loss details

The three-class head uses custom soft cross-entropy and smoothed targets. Three-class class weights are computed but are not applied by that custom loss; binary CE does receive binary weights.

The term named `image_only_loss` uses fusion-branch logits for image-bearing samples. This branch still receives text and metadata unless text dropout removes the text. It is not a separate image-only classifier.

The returned gate diagnostic is detached after its use in the logit mixture; the main predictions still allow gradients to reach the gate.

## Exploratory baseline caveat

Cell 12 combines training, validation, and test CSVs for analysis. Cell 14 constructs a separate grouped split, but TF-IDF/SVD/scaling are fitted beforehand. This exposes held-out distribution information to preprocessing. Treat baseline scores as exploratory; for a benchmark, split first and fit learned transformations only on training data.

## Inference behavior

- EasyOCR is the default; PaddleOCR is an alternative backend.
- Successful OCR replaces typed text.
- `FORCE_TEXT_ONLY_WHEN_OCR=True` disables vision when OCR succeeds.
- Training length defaults to 96 tokens; inference defaults to 128.
- Temperature is configurable, with no recorded calibration fitting/result.
- Trained classification heads require a checkpoint; pretrained encoders alone are insufficient.

## Validation for this upload

All 11 code cells were checked for Python syntax with notebook shell commands treated separately. The invalid `train_test_splita` import and a trailing continuation character in Memotion EDA were corrected. Gradio now launches with `share=False`.

Dependency installation, data processing, GPU training, OCR inference, and evaluation were not run because datasets and trained weights were not supplied. Requirement files list imported dependencies and do not represent a reproduced environment.
