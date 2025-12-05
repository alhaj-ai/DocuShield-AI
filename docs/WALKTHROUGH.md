# Project Walkthrough — Step by Step

This document explains, in order, exactly what was built for each problem, why, and which files do what. For final numbers see [results_report.md](results_report.md); this file is about *how* things were built, not just the results.

Environment note: everything runs in Python 3.11 with PyTorch and CUDA acceleration on a local NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB GDDR7 VRAM). Core libraries include `transformers`, `fastapi`, `streamlit`, `spacy`, and `faker` (see `requirements.txt`).

---

## Problem 1: Document Understanding

**Goal:** take a medical lab report image, figure out its layout (header, patient info, results table, footer), and pull out the actual field values (patient name, date of birth, test results, etc.) as structured data.

### Step 1 — Get training data
No public dataset has scanned medical lab reports with the kind of layout + field labels we need, so we generate our own. [`generate_synthetic_docs.py`](../src/problem1_document_understanding/generate_synthetic_docs.py) draws realistic lab report images (English and French) from scratch using PIL: a logo, a lab name/address header, a patient info block, a table of test results, and a physician footer. Because we're the ones drawing the text, we know the exact pixel location of every field — that ground truth gets saved alongside each image as a JSON file. About 30% of documents also get light "scan noise" added (slight rotation, blur, JPEG compression) to imitate a real scanned document. We generated 160 training documents and 80 held-out test documents.

### Step 2 — Turn images into text the model can use
[`ocr_utils.py`](../src/problem1_document_understanding/ocr_utils.py) runs Tesseract OCR (English + French) on an image and returns each detected word with its pixel bounding box.

### Step 3 — Automatically label the OCR output
We don't want to hand-label training data, so [`labels.py`](../src/problem1_document_understanding/labels.py) matches each OCR word's position against the ground-truth field positions we saved in Step 1 (a word "belongs" to a field if its center falls inside that field's box), and assigns it a label like `B-PATIENT_NAME` or `I-PATIENT_NAME` (13 field types total: lab name/address, patient name/DOB/ID, report date, physician name/ID, and the 5 table columns).

### Step 4 — Build the model's training data
[`dataset.py`](../src/problem1_document_understanding/dataset.py) ties Steps 2 and 3 together: for every document it runs OCR, generates labels, caches the result to disk (so we don't re-run OCR every time), and formats everything the way the LayoutLMv3 model expects (words + normalized bounding boxes + image).

### Step 5 — Train the model
[`train_layout_model.py`](../src/problem1_document_understanding/train_layout_model.py) fine-tunes `microsoft/layoutlmv3-base` (a model that reads text, its position on the page, and the image itself, all at once) to predict the field label for every word. 15 epochs, ~8.4 minutes on the RTX 5070. Result: **0.986 token-level F1** on held-out documents.

### Step 6 — Turn model predictions into a structured document
[`layout_extract.py`](../src/problem1_document_understanding/layout_extract.py) is the actual inference pipeline: OCR a new image, run it through the trained model, then post-process the word-level predictions into something useful — a key-value dictionary for single fields, a reconstructed table (grouping table-column predictions into rows by their vertical position), and layout "blocks" (header/patient-info/table/footer), plus a separate OpenCV step that finds the logo image (since the text model can't see it — it isn't text).

### Step 7 — Measure how good it actually is
[`evaluate_extraction.py`](../src/problem1_document_understanding/evaluate_extraction.py) runs the full pipeline on the 80 held-out test documents and compares extracted values against the known ground truth: field accuracy, table accuracy, and processing time per document. Result: **87.0% field accuracy**, 0.55s/document. It also saves the best and worst examples so we could look at real successes and failures.

**Files:** `generate_synthetic_docs.py`, `ocr_utils.py`, `labels.py`, `dataset.py`, `train_layout_model.py`, `layout_extract.py`, `evaluate_extraction.py`
**Outputs:** `outputs/problem1_eval_summary.json`, `problem1_eval_full.json`, `problem1_eval_examples.json`, `outputs/figures/problem1_field_accuracy.png`
**Model:** `models/problem1_layoutlmv3/`

---

## Problem 2: NER & Classification

### Task A — Multilingual disease NER

**Goal:** find disease/medical entity mentions in text, in both English and French.

**Step 1 — Get the data.** We needed NCBI Disease (English) and Quaero French Med (French). Both datasets' official Hugging Face pages use an old "loading script" format that no longer works with the current library version, so we had to find workarounds: NCBI Disease via a re-hosted version (`clinicalnlplab/NCBI_disease_*`, verified its sentence counts exactly match the official corpus so it's genuinely the same data), and Quaero by pulling its raw parquet files directly instead of going through the broken loader.

**Step 2 — Reconcile the label schemes.** NCBI Disease only tags one thing ("Disease"). Quaero tags ten different categories (disorders, anatomy, chemicals, procedures, etc.). We merged NCBI's "Disease" and Quaero's "DISO" (disorders) into one shared `DISEASE` label — the closest match between the two — and kept Quaero's other 9 categories as French-only labels, since English has no equivalent data for them. This whole reconciliation logic lives in [`labels.py`](../src/problem2_ner_classification/labels.py) and [`prepare_data.py`](../src/problem2_ner_classification/prepare_data.py).

**Step 3 — Train two versions, on purpose.** [`train_ner.py`](../src/problem2_ner_classification/train_ner.py) fine-tunes `xlm-roberta-base` (a model trained on 100 languages at once). We trained it twice: once on English only (so we could test how well it transfers to French it never saw), and once on English + French together (the real deliverable). Comparing the two is what lets us measure cross-lingual transfer.

**Step 4 — Evaluate.** [`evaluate_ner.py`](../src/problem2_ner_classification/evaluate_ner.py) scores both models on both languages and produces the transfer comparison. Result: the English-only model gets **0.30 F1** on French it never trained on (real transfer, since XLM-R shares representations across languages) — and jointly training on both languages pushes that to **0.52 F1** on French, while *also* improving English from 0.59 to **0.77 F1**.

### Task B — Cross-domain document classification

**Goal:** given a document, decide if it's Healthcare, Legal, Finance, or a mix of two.

**Step 1 — Build the dataset.** [`prepare_domain_data.py`](../src/problem2_ner_classification/prepare_domain_data.py) collects 50 documents per class: Healthcare comes from Problem 1's OCR'd lab reports (real reuse of earlier work), Legal comes from a European Court of Human Rights case dataset, Finance from a synthetic financial-documents dataset, and "Multi-domain" is built by literally gluing a Healthcare snippet to a Legal or Finance one.

**Step 2 — Train and compare two classifiers.** [`build_domain_classifier.py`](../src/problem2_ner_classification/build_domain_classifier.py) trains a simple TF-IDF + Logistic Regression baseline (90% accuracy) and a fine-tuned XLM-R classifier (100% accuracy) so we could see how much the neural model actually buys us over a simple baseline.

**Step 3 — Combine classification + entity extraction into one output.** [`classify_and_extract.py`](../src/problem2_ner_classification/classify_and_extract.py) loads both the Task A NER model and the Task B classifier and, for a given document, returns one JSON object with both the predicted domain and the entities found in it — this is the specific output format the assignment asked for.

**Files:** `prepare_data.py`, `labels.py`, `train_ner.py`, `evaluate_ner.py`, `prepare_domain_data.py`, `build_domain_classifier.py`, `classify_and_extract.py`
**Outputs:** `outputs/problem2_ner_eval.json`, `problem2_domain_classifier_eval.json`, `problem2_sample_classification_output.json`, figures
**Models:** `models/problem2_xlmr_en_only/`, `models/problem2_xlmr_joint/`, `models/problem2_domain_classifier/`

---

## Problem 3: Privacy-Preserving Processing (Track A)

**Goal:** find PII (names, dates, SSNs, addresses, phones, emails, medical IDs) in EN/FR medical text and redact it three different ways, then check that redaction doesn't wreck the usefulness of the document for downstream tasks.

### Step 1 — Detect PII
[`pii_detectors.py`](../src/problem3_privacy/pii_detectors.py) uses two different approaches on purpose: regex patterns for PII with a predictable shape (emails, phone numbers, SSNs, medical IDs, dates), and a small pretrained NER model (spaCy) for names, since names can't be captured by a pattern. While testing this against a labeled evaluation set, we found and fixed two real bugs: the French address regex only matched *capitalized* street-type words ("Rue") but Faker generates them lowercase ("rue") — recall on addresses was 1.7% until we fixed the case-sensitivity; and running both the English and French name-detection models on every document (regardless of actual language) was causing a lot of false positives, fixed by adding a simple language-guessing step (`guess_language`) so only the matching model runs.

### Step 2 — Redact it three ways
[`redaction.py`](../src/problem3_privacy/redaction.py) implements the three modes the assignment asked for: `strict` (`[REDACTED]` everywhere), `typed` (`[NAME]`, `[SSN]`, etc.), and `surrogate` (replace with a realistic fake value, e.g. a different real-looking name) — with a dictionary that remembers past replacements so the same name always gets the same fake substitute within a document.

### Step 3 — Build a test set with known answers
[`build_eval_set.py`](../src/problem3_privacy/build_eval_set.py) generates 100 synthetic EN/FR sentences from templates, filling in each PII field with Faker and recording exactly where it landed in the text — so we have ground truth to score against.

### Step 4 — Measure detection accuracy
[`evaluate_redaction.py`](../src/problem3_privacy/evaluate_redaction.py) compares detected PII spans against the ground truth and reports precision/recall/F1 per PII type. Final results ranged from **F1 1.00 (email, SSN)** down to **F1 0.63 (names — the hardest case, since names are open-vocabulary)**.

### Step 5 — Check that redaction doesn't break downstream use
[`downstream_utility.py`](../src/problem3_privacy/downstream_utility.py) is the "crucial" part the assignment calls out: it takes Problem 1's real synthetic lab reports, redacts them all three ways, and re-runs Problem 2's classifier and NER model on the redacted versions. Result: domain classification barely notices redaction (stays ~100% correct), but NER only recovers **75-77%** of the entities it found in the original text — and `surrogate` redaction preserves noticeably more of that signal than `strict`, because a realistic fake value disturbs the surrounding sentence less than an opaque `[REDACTED]` marker.

**Files:** `pii_detectors.py`, `redaction.py`, `build_eval_set.py`, `evaluate_redaction.py`, `downstream_utility.py`
**Outputs:** `outputs/problem3_pii_detection_eval.json`, `problem3_downstream_utility.json`, figures

---

## Problem 4: System Design Document

No code for this one — it's a written design doc, [`system_design.md`](system_design.md), covering how this pipeline would actually be deployed: architecture (which model runs when, and in what order), GPU optimization, MLOps/CI-CD, privacy & security controls, and scalability/cost. Importantly, the numbers used in it (processing time, throughput) are the *real* measured numbers from Problems 1-3, not guesses.

---

## Everything else

- [`src/common/make_charts.py`](../src/common/make_charts.py) — generates the bar charts used in the results report, straight from the JSON files in `outputs/`.
- [`architecture.md`](architecture.md), [`model_choices.md`](model_choices.md), [`error_analysis.md`](error_analysis.md), [`results_report.md`](results_report.md) — the write-ups for each problem, with the full numbers and discussion.
- `data/` — generated datasets (gitignored, regenerated by running the scripts above).
- `models/` — the actual fine-tuned model weights for each problem.
