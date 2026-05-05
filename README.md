# GoldenSetAuditor

**Evaluation dataset quality auditor for LLM / RAG applications.**

<p>
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/SidharthKriplani/goldensetauditor/ci.yml?branch=main&label=CI&style=for-the-badge&logo=githubactions&logoColor=white">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/goldensetauditor?style=for-the-badge&logo=pypi&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776ab?style=for-the-badge&logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-22c55e?style=for-the-badge">
</p>

<p>
  <img alt="Checks" src="https://img.shields.io/badge/checks-7-0ea5e9?style=flat-square">
  <img alt="Tests" src="https://img.shields.io/badge/tests-13-22c55e?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/PRD-PASS-22c55e?style=flat-square">
</p>

GoldenSetAuditor audits golden evaluation datasets for LLM and RAG applications **before benchmark scores are trusted**. It is intentionally an auditor, not a scorer — it returns structured `PASS / WARN / FAIL / INSUFFICIENT_INPUT` findings and asks the team to review suspicious examples rather than claiming it can automatically fix evaluation quality.

## About

Golden sets are the foundation of LLM/RAG evaluation. A golden set that contains duplicate prompts, conflicting expected answers, weak reference answers, or ambiguous questions will produce misleading benchmark scores — whether the model improved or not.

These problems are common because golden sets are often assembled manually or by automated pipelines that prioritise coverage over quality. They are typically caught only when scores seem implausible, when a model revision produces unexpected regressions, or when a domain expert manually reviews examples during a post-mortem.

GoldenSetAuditor makes the evaluation dataset review step systematic and reportable. It runs a structured set of checks against the golden set DataFrame, produces a per-finding audit report, and surfaces issues for human review before scores are used for any decision. The truth boundary is explicit in every report: the tool flags suspicious patterns; the evaluation lead confirms whether an example is valid.

## Architecture

```mermaid
flowchart TD
    IN["GoldenSetAuditConfig + DataFrame\ninput_col · expected_col · category_col\nid_col · thresholds"]

    IN --> CD
    IN --> ED
    IN --> WA
    IN --> AP
    IN --> OE
    IN --> ND
    IN --> CC

    CD["Conflicting Duplicate Scan\nsame prompt, different\nexpected answers"]
    ED["Exact Duplicate Scan\nsame prompt, same\nexpected answer"]
    WA["Weak Expected Answer\nfewer meaningful tokens\nthan min_expected_words"]
    AP["Ambiguous Prompt Heuristic\nanaphoric references\nor multiple questions"]
    OE["Over-easy Example\nhigh token Jaccard\nbetween prompt and answer"]
    ND["Near-Duplicate Scan\npairwise token Jaccard\n>= threshold"]
    CC["Category Coverage\nexamples per category\n< min_category_count"]

    CD --> AGG
    ED --> AGG
    WA --> AGG
    AP --> AGG
    OE --> AGG
    ND --> AGG
    CC --> AGG

    AGG["Overall Status\nFAIL > WARN > INSUFFICIENT_INPUT > PASS"]
    AGG --> OUT

    OUT["GoldenSetReport\nJSON · Markdown · HTML\nexplicit truth boundary"]
```

## Why this exists

LLM and RAG benchmark scores are only as reliable as the golden set they are measured against. Common failure modes:

- **Conflicting labels** — the same prompt appears with different expected answers, producing undefined ground truth
- **Exact duplicates** — repeated examples inflate recall metrics and bias fine-tuned models
- **Weak expected answers** — single-token or near-empty reference answers that can't distinguish good from bad model outputs
- **Ambiguous prompts** — anaphoric references or multi-part questions that lack a single correct answer
- **Over-easy examples** — the prompt essentially contains the answer, rewarding retrieval over reasoning
- **Near-duplicates** — paraphrased prompts that reduce coverage diversity without adding signal
- **Category imbalance** — underpopulated categories produce unreliable per-category metrics

GoldenSetAuditor makes that review systematic and reportable before scores inform any decision.

## Truth boundary

GoldenSetAuditor does **not** evaluate model answers. It audits the evaluation dataset itself. Human judgment is required to decide whether a flagged example should be removed, corrected, or kept as-is. It is not a replacement for domain expert review, annotation guidelines, or inter-annotator agreement measurement.

## Install

```bash
pip install goldensetauditor
```

## Quickstart

```python
import pandas as pd
from goldensetauditor import GoldenSetAuditConfig, audit_golden_set

df = pd.read_csv("data/demo_golden_set.csv")

config = GoldenSetAuditConfig(
    input_col="question",
    expected_col="expected_answer",
    category_col="category",
    id_col="id",
)

report = audit_golden_set(df, config)

print(report.status)           # FAIL / WARN / PASS
report.save("outputs/")        # writes JSON, Markdown, HTML
```

## Run the demo

```bash
git clone https://github.com/SidharthKriplani/goldensetauditor
cd goldensetauditor
pip install -e .
python scripts/generate_demo_reports.py
open outputs/goldensetauditor_report.html
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## The 7 checks

| Check | What it detects |
|---|---|
| Conflicting duplicate scan | Same prompt with different expected answers — label conflict |
| Exact duplicate scan | Same prompt with same expected answer — redundant example |
| Weak expected answer | Expected answer with fewer meaningful tokens than threshold |
| Ambiguous prompt heuristic | Anaphoric references or multiple questions in a single prompt |
| Over-easy example | High token Jaccard between prompt and expected answer |
| Near-duplicate scan | Paraphrased prompts with high pairwise token similarity |
| Category coverage | Categories with fewer examples than min_category_count |

## Resume-safe claim

Built **GoldenSetAuditor**, an evaluation dataset quality auditor for LLM/RAG applications that checks golden sets for conflicting expected answers, exact and near-duplicate prompts, weak reference answers, ambiguous questions, over-easy examples, and category coverage gaps, producing structured JSON/Markdown/HTML audit reports with per-finding PASS/WARN/FAIL/INSUFFICIENT_INPUT status and explicit truth boundary.

## License

MIT
