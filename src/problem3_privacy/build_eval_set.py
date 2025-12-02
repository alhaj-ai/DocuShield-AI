"""Builds a small labeled test set for PII detection: synthetic English and
French sentences where we know exactly which PII is where (because we
generated it), so we can measure precision/recall per PII type.

Note: since the test documents are built from the same kind of patterns our
regex detectors look for (e.g. our own MEDICAL_ID format), this measures
"does the pipeline work correctly on well-formed input" rather than
"can it survive an adversarial attacker" -- a fair thing to be honest about
in the write-up (see docs).
"""
import json
import random
from pathlib import Path

from faker import Faker

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "synthetic" / "problem3"

# Each template has one placeholder. We fill it in and record exactly where
# the filled-in value starts and ends in the final sentence.
TEMPLATES_FR = [
    ("Dossier Medical: Patient {v} est suivi depuis 2020.", "NAME"),
    ("Nee le {v}, la patiente a ete admise hier.", "DATE"),
    ("Adresse du patient: {v}.", "ADDRESS"),
    ("Vous pouvez le joindre au {v}.", "PHONE"),
    ("Contact email: {v}.", "EMAIL"),
    ("Numero de Securite Sociale: {v}.", "SSN"),
    ("Le medecin traitant est Dr. {v}.", "NAME"),
    ("ID medical du praticien: {v}.", "MEDICAL_ID"),
]

TEMPLATES_EN = [
    ("Medical Record: Patient {v} has been under care since 2020.", "NAME"),
    ("Born on {v}, the patient was admitted yesterday.", "DATE"),
    ("Patient address: {v}.", "ADDRESS"),
    ("You can reach them at {v}.", "PHONE"),
    ("Contact email: {v}.", "EMAIL"),
    ("Social Security Number: {v}.", "SSN"),
    ("The attending physician is Dr. {v}.", "NAME"),
    ("Practitioner medical ID: {v}.", "MEDICAL_ID"),
]


def _fake_value(pii_type, faker):
    if pii_type == "NAME":
        return faker.name()
    if pii_type == "DATE":
        return faker.date(pattern="%d/%m/%Y")
    if pii_type == "ADDRESS":
        return faker.address().replace("\n", ", ")
    if pii_type == "PHONE":
        return faker.phone_number()
    if pii_type == "EMAIL":
        return faker.email()
    if pii_type == "SSN":
        return faker.ssn()
    if pii_type == "MEDICAL_ID":
        return "FR" + str(faker.random_number(digits=9, fix_len=True))
    raise ValueError(pii_type)


def make_example(language, rng):
    faker = Faker("fr_FR" if language == "fr" else "en_US")
    faker.seed_instance(rng.randint(0, 2**31))

    templates = TEMPLATES_FR if language == "fr" else TEMPLATES_EN
    # pick 4-6 random lines to build one multi-line document, like the
    # spec's own example
    chosen = rng.sample(templates, rng.randint(4, 6))

    lines = []
    gt_spans = []
    offset = 0
    for template, pii_type in chosen:
        value = _fake_value(pii_type, faker)
        line = template.format(v=value)
        start_in_line = template.index("{v}")
        start = offset + start_in_line
        end = start + len(value)
        gt_spans.append({"type": pii_type, "text": value, "start": start, "end": end})
        lines.append(line)
        offset += len(line) + 1  # +1 for the newline we join with

    text = "\n".join(lines)
    return {"text": text, "language": language, "pii_spans": gt_spans}


def build_eval_set(n_per_language=50, seed=123):
    rng = random.Random(seed)
    examples = []
    for language in ["en", "fr"]:
        for _ in range(n_per_language):
            examples.append(make_example(language, rng))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "pii_eval_set.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in examples), encoding="utf-8"
    )
    print(f"Wrote {len(examples)} examples to {out_path}")
    return examples


def load_eval_set():
    path = OUT_DIR / "pii_eval_set.jsonl"
    if not path.exists():
        build_eval_set()
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


if __name__ == "__main__":
    build_eval_set()
