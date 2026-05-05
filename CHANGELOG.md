# Changelog

All notable changes to GoldenSetAuditor are documented here.

## [0.5.0] — 2026-05-06

### Added
- `answer_diversity` check: computes three complementary lexical diversity statistics
  over the expected-answer corpus — normalized Shannon entropy, type-token ratio (TTR),
  and length coefficient of variation (CV). Warns when entropy < 0.55 or TTR < 0.30,
  since a homogeneous golden set inflates embedding-based benchmark scores.
- `_answer_diversity(answers)` pure function; `diversity_entropy_warn` and
  `diversity_ttr_warn` thresholds added to `GoldenSetAuditConfig`.
- `AnswerDiversityTests` (5 tests): homogeneous warns, diverse passes, evidence keys,
  entropy bounds [0,1], single-token corpus → entropy = 0.

### Changed
- Version bumped to 0.5.0.

## [0.4.0] — 2026-05-06

### Added
- `semantic_similarity` check: TF-IDF cosine similarity (pure Python, no sklearn)
  between predicted and expected answers when `predicted_col` is configured.
  Flags rows with similarity < 0.10 (possible hallucination) or > 0.98 (verbatim copy).
- `_build_idf`, `_cosine_sim` helper functions.
- `predicted_col`, `similarity_low_threshold`, `similarity_high_threshold` added to
  `GoldenSetAuditConfig`.
- `SemanticSimilarityTests` (5 tests).

### Changed
- Version bumped to 0.4.0.

## [0.3.0] — 2026-05-06

### Added
- `OvereasyAnswerTests` (2 tests): verifies overeasy check fires when answer
  echoes question keywords and does not fire on informative answers.
- `MinimumRowsTests` (4 tests): edge cases — single-row dataset, empty expected
  answer, all-same-category, no id_col provided.
- `docs/prd/` directory with Interview Defense and PRD documents.

### Changed
- Version bumped to 0.3.0.

## [0.2.0] — 2025-07-01

### Added
- Expanded test suite: 13 tests across 4 test classes (NearDuplicateTests, CategoryCoverageTests, EvidenceTests)
- GitHub Actions CI matrix: Python 3.10 / 3.11 / 3.12
- GitHub Actions OIDC PyPI trusted publisher workflow
- README rewrite: CI/PyPI/Python badges, Mermaid architecture diagram, About section, truth boundary, check table
- CHANGELOG.md

### Changed
- pyproject.toml version bumped to 0.2.0

## [0.1.0] — 2025-06-01

### Added
- Initial release: `audit_golden_set(df, config)` with 7 checks
- `GoldenSetAuditConfig` dataclass with configurable thresholds
- `GoldenSetReport` with `to_json()`, `to_markdown()`, `to_html()`, `save()` output methods
- Demo dataset and report generation script
- 6 unit tests
