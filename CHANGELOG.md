# Changelog

All notable changes to `promptgrad` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] — 2024-XX-XX

### Added
- `PromptAnalyzer` — main API for sensitivity analysis
- Six built-in perturbation strategies: synonym substitution, word deletion,
  paraphrase, casing variation, punctuation variation, order shuffle
- Three embedding backends: TF-IDF (zero-dependency), sentence-transformers,
  OpenAI
- `RobustnessReport` dataclass with robustness score, entropy, token importance,
  per-strategy breakdown, and stability label
- `PromptAnalyzer.compare()` for ranking multiple prompts
- `PromptAnalyzer.plot()` with four chart types: gauge, heatmap, bar, radar
- `promptgrad` CLI with `analyze` and `compare` sub-commands
- JSON report export via `--json` flag
- Full pytest test suite (40+ tests)
- GitHub Actions CI/CD with automated PyPI publishing on version tags
