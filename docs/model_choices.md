# Problem 2 — NER & Classification: Model Choices & Results

Domain: **Healthcare**. Code: [src/problem2_ner_classification/](../src/problem2_ner_classification/).

## Task A: Multilingual NER

### Data

- **English**: NCBI Disease corpus (single `Disease` entity type). The canonical `ncbi/ncbi_disease` HF repo uses a deprecated loading script, so we use `clinicalnlplab/NCBI_disease_{train,dev,test}` instead -- verified its sentence counts (5424/923/940) exactly match the official corpus statistics, so it's the same data, just re-hosted in a span-tagged text format that we parse back into BIO tags ([prepare_data.py](../src/problem2_ner_classification/prepare_data.py)).
- **French**: Quaero French Med corpus (10 UMLS semantic-group entity types: DISO, PROC, ANAT, LIVB, CHEM, PHYS, PHEN, GEOG, DEVI, OBJC). Its HF repo also has a broken loading script; we load the raw parquet files directly, bypassing it.
- Sampled 500 English + 500 French training sentences (per the spec's suggested corpus size), evaluated on the full official test sets (940 EN, 1319 FR).

### Taxonomy alignment challenge

NCBI Disease annotates one class; Quaero annotates ten. **We merge NCBI's `Disease` with Quaero's `DISO` (Disorders) into one shared `DISEASE` family** -- the closest semantic match, and the only family with signal in both languages. The other 9 Quaero families are kept as French-only classes. This is a real limitation of combining these two specific datasets (not an implementation gap): **English data has zero examples of ANAT/PROC/LIVB/CHEM/PHYS/PHEN/GEOG/DEVI/OBJC**, so any English-trained model structurally cannot recall those classes, no matter how good its cross-lingual representations are. Final label space: 10 families x B/I + O = 21 labels ([labels.py](../src/problem2_ner_classification/labels.py)).

### Model & training

Fine-tuned `xlm-roberta-base` (chosen for genuine multilingual subword coverage of EN+FR, not "one model per language") for 10 epochs, batch 16, lr 3e-5, fp16, on the RTX 5070 Laptop GPU. Two conditions were trained to enable the cross-lingual comparison:

| Model | Train data | Purpose |
|---|---|---|
| `xlmr_en_only` | 500 EN sentences | zero-shot cross-lingual transfer baseline |
| `xlmr_joint` | 500 EN + 500 FR sentences | main deliverable |

### Results

**Cross-lingual transfer** (DISEASE class -- the only fair comparison, since it's the only class present in both languages):

| Model | Eval set | DISEASE F1 | Overall F1 |
|---|---|---|---|
| EN-only | FR test (zero-shot) | 0.301 | 0.129 |
| Joint EN+FR | FR test | **0.518** | **0.561** |
| EN-only | EN test | 0.585 | 0.585 |
| Joint EN+FR | EN test | **0.769** | **0.766** |

Two findings: (1) XLM-R transfers real disease-recognition signal to French with **zero** French training data (F1 0.301 zero-shot, far above chance) -- genuine cross-lingual transfer. (2) Joint training doesn't just add French capability, it also **improves English DISEASE F1 from 0.585 to 0.769** -- though part of that gain is confounded by the joint model simply seeing more total optimization steps (630 vs. 320), not purely a multilingual-regularization effect.

**Per-entity F1 on French test set** (joint model): CHEM 0.696, LIVB 0.735, PROC 0.550, DISEASE 0.518, ANAT 0.450, GEOG 0.512, PHYS 0.342, DEVI 0.157, **OBJC 0.023, PHEN 0.000**. The two worst classes (OBJC, PHEN) have the least training support (84 and 77 mentions in the 500-sentence French sample) -- a small-sample-size problem, not a modeling failure; more French training data would likely fix this specifically.

Full metrics: `outputs/problem2_ner_eval.json`.

## Task B: Cross-Domain Classification

### Data

50 documents each of Healthcare (OCR'd text from Problem 1's synthetic lab reports), Legal (`mattmdjaga/text-anonymization-benchmark-train`, European Court of Human Rights case documents), and Finance (`gretelai/synthetic_pii_finance_multilingual`, EN+FR), plus 50 synthetic Multi-domain documents built by concatenating a Healthcare snippet with a Legal or Finance one (the spec's own example of "a medical-legal document"). 160 train / 40 test ([prepare_domain_data.py](../src/problem2_ner_classification/prepare_domain_data.py)).

### Model & results

Two models for comparison ([build_domain_classifier.py](../src/problem2_ner_classification/build_domain_classifier.py)):

| Model | Test accuracy |
|---|---|
| TF-IDF + Logistic Regression (baseline) | 90.0% |
| XLM-R-base fine-tuned (8 epochs) | **100.0%** |

Both confusion matrices are clean diagonals (not a degenerate single-class collapse -- verified in `outputs/problem2_domain_classifier_eval.json`). **This task is easier than it looks**: the three source domains are stylistically very distinct (French lab-report OCR text vs. formal English legal prose vs. financial-protocol jargon), so perfect accuracy reflects how separable these particular corpora are, not that domain classification is a solved problem in general -- real production documents from more similar domains (e.g. two kinds of financial forms) would be a harder test.

[classify_and_extract.py](../src/problem2_ner_classification/classify_and_extract.py) combines both Task A and Task B models into the JSON output format the spec asks for (classification + extracted entities per document); examples in `outputs/problem2_sample_classification_output.json`. One notable side effect: the joint NER model, despite DISEASE being its only English-supervised class, correctly tagged the English word "Creatinine" as `CHEM` -- a class it only ever saw labeled in French -- another small piece of cross-lingual transfer evidence.

**How domain detection should shape entity extraction strategy:** a production system should run the domain classifier first and route documents to domain-specific NER models/label sets (e.g. only apply the Healthcare DISEASE/ANAT/CHEM schema to documents classified Healthcare or Multi-domain), rather than running every entity extractor on every document -- this avoids spurious cross-domain false positives (e.g. a legal case number being mis-tagged as a medical ID) and lets each domain's model specialize.
