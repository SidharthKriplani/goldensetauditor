# GoldenSetAuditor v0.1.0 — Context Handoff

GoldenSetAuditor was built as the next GenAI/RAG portfolio library after DocIngestQA and DevPulse.

## Why it was built

The RAG/LLM evaluation space is crowded with tools that evaluate generated answers: Ragas, DeepEval, TruLens, Phoenix, and similar systems. The better wedge is upstream: audit the **golden evaluation dataset itself** before trusting model scores.

A weak golden set can make any RAG/LLM system look better or worse than it really is. Duplicates, ambiguous questions, missing expected answers, over-easy examples, and category imbalance all distort evaluation.

## Positioning

GoldenSetAuditor is a **golden evaluation dataset quality auditor**.

It does not grade model answers. It checks whether the test set used for grading is structurally trustworthy.

Correct claim:

> GoldenSetAuditor audits LLM/RAG golden sets for duplicates, near-duplicates, weak expected answers, ambiguous prompts, over-easy examples, conflicting labels, and coverage gaps before teams trust evaluation scores.

## Current v0 features

- Exact duplicate input detection
- Near-duplicate prompt detection using token Jaccard similarity
- Conflicting expected-answer detection
- Missing / too-short expected-answer warnings
- Ambiguous prompt heuristic
- Over-easy example heuristic
- Category coverage warnings
- JSON / Markdown / HTML report export
- Demo golden set with planted issues
- Unit tests

## Truth boundary

GoldenSetAuditor is not a RAG evaluator. It does not compute faithfulness, answer relevance, or groundedness. It audits the dataset used by those systems.

It does not replace Ragas, DeepEval, TruLens, or Phoenix. It complements them upstream.

## Improvements Claude can make

- Add stronger near-duplicate detection with sentence embeddings as optional dependency
- Add domain coverage scoring if category labels are provided
- Add difficulty tags and difficulty-distribution audit
- Add support for JSONL input/output
- Add richer examples for RAG, agents, and structured extraction eval sets
- Add CLI: `goldenset-audit data.csv --input-col question --expected-col expected_answer`
- Add methodology doc examples for each finding type
- Add GitHub-ready README, badges, CI, release, and static HTML demo report
- Keep v0 deterministic; avoid LLM-as-judge unless clearly optional and labeled experimental

## Resume-safe bullet

Built GoldenSetAuditor, a Python library for auditing LLM/RAG golden evaluation datasets for duplicate prompts, conflicting expected answers, weak labels, ambiguous questions, over-easy examples, and category coverage gaps, generating structured JSON/Markdown/HTML quality reports.
