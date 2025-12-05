"""Unified NER label scheme for Problem 2 Task A.

Taxonomy alignment challenge: NCBI Disease (EN) annotates a single "Disease"
class; Quaero French Med (FR) annotates 10 UMLS semantic groups. We merge
NCBI's Disease with Quaero's DISO (Disorders) into one shared "DISEASE"
family -- semantically the closest match, and the only family with signal
in both languages. The other 9 Quaero families (ANAT, PROC, LIVB, CHEM,
PHYS, PHEN, GEOG, DEVI, OBJC) have no English counterpart in this corpus
combination and are kept as French-only classes -- this is a real,
documented limitation of combining these two specific datasets, not an
implementation gap (see docs/model_choices.md).
"""

QUAERO_ID2NAME = {
    0: "O", 1: "DISO", 2: "PROC", 3: "ANAT", 4: "LIVB", 5: "CHEM",
    6: "PHYS", 7: "PHEN", 8: "GEOG", 9: "DEVI", 10: "OBJC",
}
QUAERO_TO_FAMILY = {name: ("DISEASE" if name == "DISO" else name)
                     for name in QUAERO_ID2NAME.values() if name != "O"}

FAMILIES = ["DISEASE", "ANAT", "PROC", "LIVB", "CHEM", "PHYS", "PHEN", "GEOG", "DEVI", "OBJC"]
LABEL_LIST = ["O"] + [f"{p}-{fam}" for fam in FAMILIES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Families with zero support in the English (NCBI) data -- used to flag
# the cross-lingual transfer ceiling in evaluate_ner.py.
EN_ONLY_SUPPORTED_FAMILIES = {"DISEASE"}
