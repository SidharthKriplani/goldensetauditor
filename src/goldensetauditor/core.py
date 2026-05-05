from __future__ import annotations

from dataclasses import dataclass, field, asdict
from html import escape
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import itertools
import json
import math
import re

import numpy as np
import pandas as pd

Status = Literal["PASS", "WARN", "FAIL", "INSUFFICIENT_INPUT"]
Severity = Literal["none", "low", "medium", "high"]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "for", "and", "or", "in",
    "on", "with", "what", "how", "why", "when", "where", "does", "do", "can", "should",
    "please", "explain", "tell", "me", "about", "this", "that", "it"
}

@dataclass
class GoldenSetAuditConfig:
    input_col: str = "question"
    expected_col: str = "expected_answer"
    category_col: Optional[str] = None
    id_col: Optional[str] = None
    predicted_col: Optional[str] = None         # model predictions for semantic similarity check
    min_expected_words: int = 4
    near_duplicate_threshold: float = 0.82
    overeasy_overlap_threshold: float = 0.72
    min_category_count: int = 3
    max_pairwise_checks: int = 25000
    similarity_low_threshold: float = 0.10      # below → possible hallucination / wrong mapping
    similarity_high_threshold: float = 0.98     # above → suspicious verbatim copy
    diversity_entropy_warn: float = 0.55        # normalized Shannon entropy below this → homogeneous
    diversity_ttr_warn: float = 0.30            # type-token ratio below this → repetitive vocab

@dataclass
class GoldenSetFinding:
    check_name: str
    status: Status
    severity: Severity
    row_ids: List[object]
    detail: str
    recommendation: str
    evidence: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

@dataclass
class GoldenSetReport:
    status: Status
    findings: List[GoldenSetFinding]
    summary: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# GoldenSetAuditor Report",
            "",
            f"**Overall status:** `{self.status}`",
            "",
            "## Summary",
            "",
        ]
        for k, v in self.summary.items():
            lines.append(f"- **{k}:** {v}")
        lines += ["", "## Findings", ""]
        if not self.findings:
            lines.append("No findings generated.")
        for i, f in enumerate(self.findings, 1):
            lines += [
                f"### {i}. {f.check_name} — `{f.status}` / {f.severity}",
                f"- **Rows:** `{f.row_ids}`",
                f"- **Detail:** {f.detail}",
                f"- **Recommendation:** {f.recommendation}",
            ]
            if f.evidence:
                ev = ", ".join(f"{k}={v}" for k, v in f.evidence.items())
                lines.append(f"- **Evidence:** {ev}")
            lines.append("")
        lines += [
            "## Interpretation note",
            "",
            "GoldenSetAuditor audits the evaluation dataset, not model outputs. WARN findings should be reviewed before trusting benchmark scores.",
        ]
        return "\n".join(lines)

    def to_html(self) -> str:
        rows = []
        for f in self.findings:
            rows.append(
                "<tr>"
                f"<td>{escape(f.check_name)}</td>"
                f"<td>{escape(f.status)}</td>"
                f"<td>{escape(f.severity)}</td>"
                f"<td>{escape(str(f.row_ids))}</td>"
                f"<td>{escape(f.detail)}</td>"
                f"<td>{escape(f.recommendation)}</td>"
                "</tr>"
            )
        summary_items = "".join(f"<li><b>{escape(str(k))}:</b> {escape(str(v))}</li>" for k, v in self.summary.items())
        return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>GoldenSetAuditor Report</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:40px;line-height:1.45;color:#1f2937}}
.badge{{display:inline-block;padding:6px 10px;border-radius:999px;background:#ecfeff;font-weight:700}}
table{{border-collapse:collapse;width:100%;margin-top:20px}}th,td{{border:1px solid #e5e7eb;padding:10px;text-align:left;vertical-align:top}}th{{background:#f9fafb}}
.note{{background:#f0fdf4;border:1px solid #bbf7d0;padding:14px;border-radius:12px;margin-top:24px}}
</style></head><body>
<h1>GoldenSetAuditor Report</h1>
<p class='badge'>Overall status: {escape(self.status)}</p>
<h2>Summary</h2><ul>{summary_items}</ul>
<h2>Findings</h2>
<table><thead><tr><th>Check</th><th>Status</th><th>Severity</th><th>Rows</th><th>Detail</th><th>Recommendation</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class='note'><b>Truth boundary:</b> This report audits golden-set quality. It does not evaluate model answers.</div>
</body></html>"""

    def save(self, output_dir: str | Path, stem: str = "goldensetauditor_report") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{stem}.json").write_text(self.to_json(), encoding="utf-8")
        (out / f"{stem}.md").write_text(self.to_markdown(), encoding="utf-8")
        (out / f"{stem}.html").write_text(self.to_html(), encoding="utf-8")


def _row_id(df: pd.DataFrame, idx, config: GoldenSetAuditConfig):
    if config.id_col and config.id_col in df.columns:
        return df.loc[idx, config.id_col]
    return int(idx) if isinstance(idx, (int, np.integer)) else str(idx)


def _tokens(text: object) -> set:
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def _jaccard(a: object, b: object) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _build_idf(corpus: List[str]) -> Dict[str, float]:
    """Compute IDF weights for all terms across a corpus of strings."""
    n = len(corpus)
    doc_freq: Dict[str, int] = {}
    for doc in corpus:
        for term in set(re.findall(r"[a-zA-Z0-9]+", doc.lower())):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    return {t: math.log((n + 1) / (df + 1)) + 1.0 for t, df in doc_freq.items()}


def _cosine_sim(text_a: str, text_b: str, idf: Dict[str, float]) -> float:
    """TF-IDF weighted cosine similarity between two text strings (no external deps)."""
    def tf_vec(text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            counts[t] = counts.get(t, 0) + 1
        return counts

    va, vb = tf_vec(str(text_a)), tf_vec(str(text_b))
    all_terms = list(set(va) | set(vb))
    vec_a = [va.get(t, 0) * idf.get(t, 1.0) for t in all_terms]
    vec_b = [vb.get(t, 0) * idf.get(t, 1.0) for t in all_terms]
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return min(dot / (norm_a * norm_b), 1.0)


def _answer_diversity(answers: List[str]) -> Dict[str, float]:
    """Compute three complementary diversity statistics over a list of answer strings.

    Returns:
        normalized_entropy  — Shannon entropy of the token-frequency distribution,
                              normalized by log2(vocab_size) so it sits in [0, 1].
                              Low value → answers share the same small vocabulary.
        type_token_ratio    — unique_tokens / total_tokens (TTR).
                              Low value → high repetition across answers.
        length_cv           — coefficient of variation of answer character lengths
                              (std / mean). Near-zero → all answers are the same length,
                              which can inflate benchmark scores for length-sensitive metrics.
    """
    all_tokens: List[str] = []
    lengths: List[int] = []
    for ans in answers:
        tokens = re.findall(r"[a-zA-Z0-9]+", ans.lower())
        all_tokens.extend(tokens)
        lengths.append(len(ans))

    total = len(all_tokens)
    if total == 0:
        return {"normalized_entropy": 0.0, "type_token_ratio": 0.0, "length_cv": 0.0}

    # Shannon entropy
    freq: Dict[str, int] = {}
    for t in all_tokens:
        freq[t] = freq.get(t, 0) + 1
    vocab_size = len(freq)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    max_entropy = math.log2(vocab_size) if vocab_size > 1 else 1.0
    normalized_entropy = entropy / max_entropy

    # Type-token ratio
    ttr = vocab_size / total

    # Length coefficient of variation
    if len(lengths) < 2:
        length_cv = 0.0
    else:
        mean_len = sum(lengths) / len(lengths)
        if mean_len == 0:
            length_cv = 0.0
        else:
            variance = sum((l - mean_len) ** 2 for l in lengths) / (len(lengths) - 1)
            length_cv = math.sqrt(variance) / mean_len

    return {
        "normalized_entropy": round(normalized_entropy, 4),
        "type_token_ratio": round(ttr, 4),
        "length_cv": round(length_cv, 4),
        "vocab_size": vocab_size,
        "total_tokens": total,
    }


def _overall_status(findings: List[GoldenSetFinding]) -> Status:
    if any(f.status == "FAIL" for f in findings):
        return "FAIL"
    if any(f.status == "WARN" for f in findings):
        return "WARN"
    if findings and all(f.status == "INSUFFICIENT_INPUT" for f in findings):
        return "INSUFFICIENT_INPUT"
    return "PASS"


def audit_golden_set(df: pd.DataFrame, config: GoldenSetAuditConfig = GoldenSetAuditConfig()) -> GoldenSetReport:
    findings: List[GoldenSetFinding] = []
    missing = [c for c in [config.input_col, config.expected_col] if c not in df.columns]
    if missing:
        findings.append(GoldenSetFinding(
            "input_schema", "FAIL", "high", [],
            f"Required column(s) missing: {missing}.",
            "Provide input and expected-answer columns before auditing the golden set.",
        ))
        return GoldenSetReport("FAIL", findings, {"rows": int(len(df)), "columns": int(len(df.columns))})

    # Exact duplicates and conflicting labels.
    input_norm = df[config.input_col].fillna("").astype(str).str.strip().str.lower()
    dup_groups = input_norm[input_norm.duplicated(keep=False)]
    for normalized_value, indices in dup_groups.groupby(dup_groups).groups.items():
        ids = [_row_id(df, i, config) for i in indices]
        answers = df.loc[list(indices), config.expected_col].fillna("").astype(str).str.strip().str.lower().unique()
        status = "FAIL" if len(answers) > 1 else "WARN"
        severity = "high" if len(answers) > 1 else "medium"
        check = "conflicting_expected_answers" if len(answers) > 1 else "exact_duplicate_inputs"
        detail = "Same prompt appears with different expected answers." if len(answers) > 1 else "Same prompt appears multiple times."
        recommendation = "Resolve label conflict before using this example in evaluation." if len(answers) > 1 else "Deduplicate or justify repeated examples."
        findings.append(GoldenSetFinding(check, status, severity, ids, detail, recommendation, {"duplicate_count": len(ids)}))

    # Weak expected answers.
    for idx, val in df[config.expected_col].items():
        words = _tokens(val)
        if pd.isna(val) or len(words) < config.min_expected_words:
            findings.append(GoldenSetFinding(
                "weak_expected_answer", "WARN", "medium", [_row_id(df, idx, config)],
                f"Expected answer has only {len(words)} meaningful token(s).",
                "Add a fuller expected answer or remove the example from scored evaluation.",
                {"meaningful_tokens": len(words), "minimum": config.min_expected_words},
            ))

    # Ambiguous prompts.
    ambiguous_patterns = [r"\bit\b", r"\bthis\b", r"\bthat\b", r"\bthey\b", r"\bthere\b"]
    for idx, q in df[config.input_col].fillna("").astype(str).items():
        q_lower = q.lower()
        question_marks = q.count("?")
        has_ambiguous_ref = any(re.search(p, q_lower) for p in ambiguous_patterns) and len(_tokens(q)) < 7
        has_multi_question = question_marks > 1 or " and " in q_lower and question_marks >= 1
        if has_ambiguous_ref or has_multi_question:
            findings.append(GoldenSetFinding(
                "ambiguous_prompt_heuristic", "WARN", "low", [_row_id(df, idx, config)],
                "Prompt may be ambiguous or contain multiple questions.",
                "Rewrite as a single, self-contained evaluation question.",
                {"question_marks": question_marks, "ambiguous_reference": has_ambiguous_ref},
            ))

    # Over-easy examples.
    for idx, row in df.iterrows():
        overlap = _jaccard(row[config.input_col], row[config.expected_col])
        if overlap >= config.overeasy_overlap_threshold:
            findings.append(GoldenSetFinding(
                "overeasy_example", "WARN", "low", [_row_id(df, idx, config)],
                f"Question and expected answer have high token overlap ({overlap:.3f}).",
                "Review whether this example is too easy or merely repeats the answer in the prompt.",
                {"token_jaccard": round(overlap, 4), "threshold": config.overeasy_overlap_threshold},
            ))

    # Near duplicate pairwise check.
    n = len(df)
    max_pairs = n * (n - 1) // 2
    if max_pairs > config.max_pairwise_checks:
        findings.append(GoldenSetFinding(
            "near_duplicate_inputs", "INSUFFICIENT_INPUT", "low", [],
            f"Dataset has {max_pairs} pairs, exceeding configured max_pairwise_checks.",
            "Increase max_pairwise_checks or run the auditor on stratified batches.",
        ))
    else:
        seen_pairs = set()
        for i, j in itertools.combinations(range(n), 2):
            if input_norm.iloc[i] == input_norm.iloc[j]:
                continue
            score = _jaccard(df.iloc[i][config.input_col], df.iloc[j][config.input_col])
            if score >= config.near_duplicate_threshold:
                pair = tuple(sorted([_row_id(df, df.index[i], config), _row_id(df, df.index[j], config)]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    findings.append(GoldenSetFinding(
                        "near_duplicate_inputs", "WARN", "medium", list(pair),
                        f"Two prompts are near-duplicates by token Jaccard ({score:.3f}).",
                        "Deduplicate, merge, or intentionally mark one as a paraphrase test case.",
                        {"token_jaccard": round(score, 4), "threshold": config.near_duplicate_threshold},
                    ))

    # Category coverage.
    if config.category_col:
        if config.category_col not in df.columns:
            findings.append(GoldenSetFinding(
                "category_coverage", "INSUFFICIENT_INPUT", "low", [],
                f"Configured category column '{config.category_col}' is missing.",
                "Provide a valid category column or omit category_col.",
            ))
        else:
            counts = df[config.category_col].fillna("<missing>").astype(str).value_counts()
            for category, count in counts.items():
                if count < config.min_category_count:
                    ids = [_row_id(df, i, config) for i in df.index[df[config.category_col].fillna("<missing>").astype(str) == category].tolist()]
                    findings.append(GoldenSetFinding(
                        "category_coverage", "WARN", "medium", ids,
                        f"Category '{category}' has only {int(count)} example(s).",
                        "Add more examples or avoid reporting category-level metrics for this group.",
                        {"category": category, "count": int(count), "minimum": config.min_category_count},
                    ))
    else:
        findings.append(GoldenSetFinding(
            "category_coverage", "INSUFFICIENT_INPUT", "low", [],
            "No category column provided.",
            "Provide category_col to audit coverage gaps across evaluation categories.",
        ))

    # Answer diversity: Shannon entropy + type-token ratio + length CV
    answers = df[config.expected_col].fillna("").astype(str).tolist()
    diversity = _answer_diversity(answers)
    low_entropy = diversity["normalized_entropy"] < config.diversity_entropy_warn
    low_ttr = diversity["type_token_ratio"] < config.diversity_ttr_warn
    if low_entropy or low_ttr:
        issues = []
        if low_entropy:
            issues.append(
                f"normalized entropy {diversity['normalized_entropy']:.3f} < {config.diversity_entropy_warn} "
                "(answers share a small, repetitive vocabulary)"
            )
        if low_ttr:
            issues.append(
                f"type-token ratio {diversity['type_token_ratio']:.3f} < {config.diversity_ttr_warn} "
                "(high token repetition across answers)"
            )
        findings.append(GoldenSetFinding(
            "answer_diversity",
            "WARN",
            "medium",
            [],
            "Expected-answer corpus shows low lexical diversity: " + "; ".join(issues) + ".",
            "Expand the vocabulary and phrasing of expected answers. "
            "A homogeneous golden set inflates benchmark scores for length-sensitive and embedding-based metrics.",
            diversity,
        ))
    else:
        findings.append(GoldenSetFinding(
            "answer_diversity",
            "PASS",
            "none",
            [],
            f"Answer corpus entropy {diversity['normalized_entropy']:.3f}, TTR {diversity['type_token_ratio']:.3f}, "
            f"length CV {diversity['length_cv']:.3f} — diversity looks healthy.",
            "Continue monitoring diversity as the golden set grows.",
            diversity,
        ))

    # Semantic similarity: predicted vs. expected answers (requires predicted_col)
    if config.predicted_col:
        if config.predicted_col not in df.columns:
            findings.append(GoldenSetFinding(
                "semantic_similarity", "INSUFFICIENT_INPUT", "low", [],
                f"Configured predicted_col '{config.predicted_col}' is missing from the DataFrame.",
                "Provide model predictions in the predicted_col before running semantic similarity check.",
            ))
        else:
            corpus = (
                df[config.expected_col].fillna("").astype(str).tolist()
                + df[config.predicted_col].fillna("").astype(str).tolist()
            )
            idf = _build_idf(corpus)
            sim_scores: List[float] = []
            for idx, row in df.iterrows():
                expected = str(row[config.expected_col]) if pd.notna(row.get(config.expected_col)) else ""
                predicted = str(row[config.predicted_col]) if pd.notna(row.get(config.predicted_col)) else ""
                sim = _cosine_sim(expected, predicted, idf)
                sim_scores.append(sim)
                if sim < config.similarity_low_threshold:
                    findings.append(GoldenSetFinding(
                        "semantic_similarity", "WARN", "high", [_row_id(df, idx, config)],
                        f"Predicted answer has very low semantic similarity to expected ({sim:.3f}). "
                        "Possible hallucination or wrong-question mapping.",
                        "Review this row manually. The model output may be ungrounded or unrelated.",
                        {"cosine_similarity": round(sim, 4), "threshold": config.similarity_low_threshold},
                    ))
                elif sim > config.similarity_high_threshold:
                    findings.append(GoldenSetFinding(
                        "semantic_similarity", "WARN", "low", [_row_id(df, idx, config)],
                        f"Predicted answer is near-identical to expected ({sim:.3f}). "
                        "Possible verbatim copy or evaluation data contamination.",
                        "Verify the model is not memorizing or copying the expected answer from its context.",
                        {"cosine_similarity": round(sim, 4), "threshold": config.similarity_high_threshold},
                    ))
            if sim_scores:
                avg_sim = sum(sim_scores) / len(sim_scores)
                low_count = sum(1 for s in sim_scores if s < config.similarity_low_threshold)
                findings.append(GoldenSetFinding(
                    "semantic_similarity_summary",
                    "WARN" if low_count > 0 else "PASS",
                    "medium" if low_count > 0 else "none",
                    [],
                    f"Average predicted/expected cosine similarity: {avg_sim:.3f}. "
                    f"Low-similarity rows: {low_count}/{len(sim_scores)}.",
                    "Review all low-similarity rows before using this benchmark to validate model quality.",
                    {
                        "avg_cosine_similarity": round(avg_sim, 4),
                        "low_similarity_count": low_count,
                        "total_rows": len(sim_scores),
                    },
                ))

    if not findings:
        findings.append(GoldenSetFinding(
            "audit_complete", "PASS", "none", [],
            "No obvious golden-set quality risks were detected by v0 checks.",
            "Continue with manual review and domain expert validation before using scores for decisions.",
        ))

    status = _overall_status(findings)
    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "finding_count": int(len(findings)),
        "warn_count": int(sum(f.status == "WARN" for f in findings)),
        "fail_count": int(sum(f.status == "FAIL" for f in findings)),
        "insufficient_input_count": int(sum(f.status == "INSUFFICIENT_INPUT" for f in findings)),
    }
    if config.category_col and config.category_col in df.columns:
        summary["category_count"] = int(df[config.category_col].nunique(dropna=False))
    return GoldenSetReport(status, findings, summary)
