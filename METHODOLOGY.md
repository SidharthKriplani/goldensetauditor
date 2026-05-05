# Methodology

GoldenSetAuditor v0 audits the quality of LLM/RAG golden evaluation datasets.

## Status vocabulary

- `PASS`: no obvious issue found.
- `WARN`: quality risk found; review recommended.
- `FAIL`: severe dataset quality issue found.
- `INSUFFICIENT_INPUT`: required columns are missing or data is too small.

## Checks

### 1. Exact duplicates

Repeated prompts inflate evaluation confidence and can make a dataset appear broader than it is.

### 2. Near duplicates

Token Jaccard similarity is used as a deterministic v0 signal for near-duplicate prompts. This is simple but useful for catching repeated question templates.

### 3. Conflicting labels

If the same prompt appears with materially different expected answers, evaluation results become unstable.

### 4. Weak expected answers

Empty or very short expected answers are flagged because they often cannot support meaningful grading.

### 5. Ambiguity heuristics

Questions with unclear references, multiple questions in one prompt, or vague wording are flagged for review.

### 6. Over-easy examples

If the expected answer overlaps heavily with the input prompt, the example may be too easy and inflate evaluation scores.

### 7. Coverage warnings

If a category column exists, the auditor flags underrepresented categories. This helps prevent a golden set from overrepresenting only easy or common query types.

## Limitations

- v0 uses deterministic heuristics, not LLM judgment.
- Semantic ambiguity is hard to detect without domain knowledge.
- Near-duplicate detection with Jaccard similarity can miss paraphrases.
- Coverage quality depends on user-provided category labels.
