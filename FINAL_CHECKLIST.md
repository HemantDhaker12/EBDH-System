# Final Submission Checklist (Stage 3 Verification)

This checklist confirms that the EDHC repository is fully compliant with all Stage 3 requirements of the Redrob Candidate Discovery Challenge.

- [x] **Repository Builds Successfully**: Tested package imports, schema validations, and Python runtime dependencies locally under a virtual environment (`.venv`).
- [x] **Dependencies Documented**: Requirements are declared inside `requirements.txt`.
- [x] **README Contains Exact Reproduction Commands**: README.md includes a dedicated highlighted `# Reproducing submission.csv` section detailing commands to train the LTR ranker, run the ranking script, and validate the generated file.
- [x] **submission_metadata.yaml Present**: The metadata file resides in the root directory and complies with the official template schema.
- [x] **Validation Command Documented**: Validation command utilizing `validate_submission.py` is documented in the README.
- [x] **Offline Execution Documented**: Factual guarantees are documented confirming that no external network requests or remote API calls are performed during the ranking process.
- [x] **CPU-Only Execution Documented**: The pipeline is fully configured to execute on CPU-only hosts under 16GB RAM constraints.
- [x] **No Manual Edits Required**: The pipeline runs completely end-to-end autonomously from `candidates.jsonl` to compile the final `submission.csv`.
- [x] **Repository Ready for Stage 3**: The folder structure is clean, temporary artifacts/logs are purged, and all deliverables conform to the Stage 3 specifications.
- [x] **Sandbox Placeholder Documented**: A dedicated sandbox section is present in the README with instructions to test on `sample_candidates.json` and a placeholder URL.
- [x] **No BharatVoice References**: All historical references to the "BharatVoice" project are fully cleaned and replaced with Heaven-Hill (team) and EDHC (project) definitions.
- [x] **FROZEN Codebase Compliance**: The Python scripts, model features, calibration offsets, and reasoning generators have not been modified. Only documentation, metadata, and repository layout adjustments were performed.
