import os
import json
import argparse
from pathlib import Path

from edhc.app.config.settings import settings
from edhc.app.schemas.candidate import CandidateProfileNormalized
from edhc.app.jd.intelligence import JobDescriptionParser
from edhc.app.pipeline.inference import InferencePipeline
from edhc.app.utils.logger import get_logger

logger = get_logger("rank_cli")

def load_candidates(processed_dir: Path) -> list:
    """Load normalized candidate profiles."""
    candidates = []
    for file_path in processed_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            candidates.append(CandidateProfileNormalized(**data))
        except Exception as e:
            logger.error(f"Failed to load candidate {file_path.name}: {e}")
    return candidates

def main():
    parser = argparse.ArgumentParser(description="Rank candidate profiles against a Job Description.")
    parser.add_argument("--candidates-dir", type=str, default=str(settings.PROCESSED_DATA_DIR), help="Folder with normalized candidate json profiles")
    parser.add_argument("--jd-file", type=str, required=False, help="Text file containing the target Job Description")
    parser.add_argument("--model-path", type=str, default=None, help="Optional trained model path override")
    parser.add_argument("--output-format", type=str, choices=["json", "md", "both"], default="both", help="Format to write report")
    
    args = parser.parse_args()
    
    candidates_path = Path(args.candidates_dir)
    
    # 1. Load candidates
    candidates = load_candidates(candidates_path)
    if not candidates:
        logger.error(f"No candidates found in {candidates_path}. Please run preprocess.py first!")
        return

    # 2. Get JD Text
    jd_parser = JobDescriptionParser()
    if args.jd_file and os.path.exists(args.jd_file):
        with open(args.jd_file, "r", encoding="utf-8") as f:
            jd_text = f.read()
        jd_title = Path(args.jd_file).stem.replace("_", " ").title()
    else:
        logger.info("No job description file supplied or path invalid. Falling back to default JD.")
        jd_title = "Senior Search and ML Engineer"
        jd_text = """
        Job Title: Senior Search and Machine Learning Engineer
        We are looking for a Senior Machine Learning Engineer with 5+ years of experience.
        Key responsibilities:
        - Build and optimize search ranking systems using learning-to-rank algorithms.
        - Write clean production code in Python.
        - Develop information retrieval models including BM25 and dense embeddings.
        - Deploy ML microservices to production using Docker and Kubernetes.
        Required skills: Python, Machine Learning, Search, System Architecture, SQL.
        """

    # 3. Parse JD into rubric
    jd = jd_parser.parse(jd_text, jd_title)

    # 4. Execute Pipeline
    pipeline = InferencePipeline(model_path=args.model_path)
    rankings = pipeline.run(candidates, jd)

    # 5. Output Results
    logger.info("--- COMMITTEE RANKING REPORT ---")
    print(f"\nTarget Job Title: {jd.title}")
    print("=" * 60)
    print(f"{'Rank':<5} | {'Candidate Name':<20} | {'Score':<6} | {'Verdict':<15}")
    print("-" * 60)
    for r in rankings:
        print(f"{r['rank']:<5} | {r['candidate_name']:<20} | {r['calibrated_score']:<6.4f} | {r['verdict']:<15}")
    print("=" * 60 + "\n")

    # Ensure output dir exists
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON Report
    if args.output_format in ["json", "both"]:
        json_report_path = settings.OUTPUT_DIR / "ranking_report.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(rankings, f, indent=4)
        logger.info(f"Saved JSON ranking report to: {json_report_path}")

    # Save Markdown Report
    if args.output_format in ["md", "both"]:
        md_report_path = settings.OUTPUT_DIR / "ranking_report.md"
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write(f"# EDHC Ranking Report\n\n")
            f.write(f"**Target Role**: {jd.title}\n\n")
            f.write(f"| Rank | Candidate | Calibrated Score | Verdict |\n")
            f.write(f"| --- | --- | --- | --- |\n")
            for r in rankings:
                f.write(f"| {r['rank']} | {r['candidate_name']} | {r['calibrated_score']:.4f} | {r['verdict']} |\n")
            
            f.write("\n---\n\n## Candidate Justifications\n\n")
            for r in rankings:
                f.write(f"{r['explanation_narrative']}\n\n")
                
        logger.info(f"Saved Markdown ranking report to: {md_report_path}")

if __name__ == "__main__":
    main()
