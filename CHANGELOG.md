# Changelog

All notable milestones and release notes for the Evidence-Based Digital Hiring Committee (EDHC) system.

## [1.0.0] - 2026-06-28
### Added
- Created `ExperienceConsistencyAnalyzer` supporting leap-year adjusted tenure calculations (`total_days / 365.25`), chronologically merging overlaps, handling current roles, and verifying stated years of experience.
- Implemented multi-section evidence verification inside the `EvidenceVerifier` matching competencies across skills lists, experience descriptions, project profiles, and summaries.
- Configured 39 detailed candidate features across retrieval, semantic title/summary similarity, career tenure, technical specialization indicators, notice period durations, and quantitative impact densities.
- Standardized LambdaMART LTR training pipeline (`train_ranker.py`) supporting contiguous query groups, validation splits, variance assertion checks, and Ndcg metrics.
- Formulated the unique score calibration offsets inside `ScoreCalibrator` sorting candidates by score and candidate ID, and applying a decreasing step offset to prevent ranking bucketization.
- Integrated deterministic hashing into fallback dense mock retrievers using MD5 hash random seeds, establishing 100% reproducible scoring runs.
- Formulated complete markdown architecture documentation and final repository check guidelines.
- Configured MIT license, contributor covenants, security declarations, and compliant `.gitignore` configurations.
