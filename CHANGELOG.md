# Changelog

All notable changes to GoldenSetAuditor are documented here.

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
