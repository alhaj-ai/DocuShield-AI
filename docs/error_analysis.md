# Error Analysis — Concrete Examples

Pulled directly from the evaluation runs (not hand-picked to look better or worse than they are); full data in `outputs/`.

## Problem 1 — Document Extraction

Example failure on a rotated/blurred synthetic document (`test_fr_0023`, see `outputs/problem1_eval_examples.json`):

- Ground truth lab name: `"Clinique du Nord - Laboratoire"` → predicted: `"avenue"` (a word from the address line, one line below). The ±3° rotation shifted word bounding boxes just enough that LayoutLMv3 -- which relies heavily on 2D position -- assigned the wrong field label to a nearby word.
- `patient_id` and `physician_name` were predicted empty (missed entirely) on the same document.
- In the same run, `report_date` (`"11/07/2026"`) and `physician_id` (`"FR214644955"`) were both extracted correctly, showing the failure is localized to the rotated header region rather than the whole document.

This is the clearest single finding from Problem 1: **scan noise, not language or vocabulary, is what breaks the model** (see `docs/architecture.md` for the full clean-vs-augmented accuracy comparison: 92.0% vs. 71.1%).

## Problem 2 Task A — NER

Three real misses of the `PHEN` (Phenomenon) class on the French test set:

| Context | Missed word | Gold label | Predicted |
|---|---|---|---|
| "...de données sur la **sécurité** d'emploi" | sécurité (safety) | B-PHEN | O |
| "**Résultats** des incisions transverses" | Résultats (results) | B-PHEN | O |
| "**Génétique** de l'amyotrophie spinale" | Génétique (genetics) | B-PHEN | O |

All three misses are abstract, general-vocabulary words ("safety", "results", "genetics") that only count as a `PHEN` entity in this specific UMLS annotation scheme when used in a clinical-phenomenon sense -- a genuinely hard, context-dependent distinction, and PHEN has the lowest support (77 mentions) of any class in the 500-sentence French training sample. This matches the quantitative result in `docs/model_choices.md`: PHEN and OBJC (the two lowest-support classes) score far worse (F1 0.000 and 0.023) than higher-support classes like CHEM (0.696) -- a data-volume problem, not a fundamental modeling limitation.

## Problem 2 Task B — Domain Classification

The fine-tuned XLM-R classifier had zero test-set errors, so the more informative failures come from the TF-IDF + Logistic Regression baseline it was compared against (`outputs/problem2_domain_classifier_eval.json`):

- 2 `multi_domain` documents were misclassified as `healthcare`, and 1 as `finance` -- the baseline tends to just pick whichever constituent domain has stronger keyword signal, rather than recognizing "contains two domains" as its own pattern.
- 2 `healthcare` documents were misclassified as `multi_domain`.

This makes sense for a bag-of-words model: multi-domain documents are literally built by concatenating two single-domain snippets, so their TF-IDF vector sits between the two source classes, and a linear classifier has no way to represent "this is a mixture" except by leaning toward whichever half has denser vocabulary. The fine-tuned transformer, which can attend to document-wide structure rather than isolated keyword counts, didn't make this mistake.

## Problem 3 — Privacy (Track A)

See `docs/system_design.md` and the Problem 3 section of `docs/results_report.md` for PII detection error examples (regex/model trade-offs are discussed there since they're specific to that pipeline).
