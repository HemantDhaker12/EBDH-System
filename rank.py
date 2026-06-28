import os
import csv
import json
import argparse
import gzip
from pathlib import Path

# Add project root to python path to avoid import errors
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from edhc.app.config.settings import settings
from edhc.app.schemas.candidate import CandidateProfile
from edhc.app.jd.intelligence import JobDescriptionParser
from edhc.app.pipeline.inference import InferencePipeline
from edhc.app.utils.logger import get_logger

logger = get_logger("rank_root_cli")

def load_candidates_from_jsonl(jsonl_path: Path) -> list:
    """Load candidates from a JSON or JSONL file (either raw or gzipped)."""
    candidates = []
    file_format = "JSONL (gzipped)" if jsonl_path.suffix == ".gz" else ("JSON Array" if jsonl_path.suffix == ".json" else "JSONL")
    logger.info(f"Loading candidate pool from {jsonl_path} (Format: {file_format})...")
    
    open_func = gzip.open if jsonl_path.suffix == ".gz" else open
    mode = "rt" if jsonl_path.suffix == ".gz" else "r"
    
    # Read file content to detect if it is a JSON array or JSONL
    with open_func(jsonl_path, mode, encoding="utf-8") as f:
        first_char = ""
        for char in f.read(100):
            if char.strip():
                first_char = char
                break
                
    # Parse based on structure
    with open_func(jsonl_path, mode, encoding="utf-8") as f:
        if first_char == "[":
            # standard JSON array
            data = json.load(f)
            for item in data:
                candidates.append(CandidateProfile(**item))
        else:
            # JSONL format
            count = 0
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                candidates.append(CandidateProfile(**data))
                count += 1
                if count % 10000 == 0:
                    logger.info(f"Loaded {count} candidates...")
                
    logger.info(f"Loaded total {len(candidates)} candidates.")
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against a job description.")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", type=str, required=True, help="Path to save output submission.csv")
    parser.add_argument("--jd", type=str, required=False, help="Optional path to job description file")
    
    args = parser.parse_args()
    
    candidates_path = Path(args.candidates)
    output_path = Path(args.out)
    
    if not candidates_path.exists():
        logger.error(f"Candidates file not found at {candidates_path}")
        sys.exit(1)
        
    # Ensure parent output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load candidates
    candidates = load_candidates_from_jsonl(candidates_path)
    if not candidates:
        logger.error("No candidates loaded.")
        sys.exit(1)
    logger.info(f"Loaded candidates: {len(candidates)}")
    logger.info(f"After normalization: {len(candidates)}")
        
    # 2. Get JD text
    jd_text = ""
    # Try looking for jd file, fallback to default locations, fallback to hardcoded
    if args.jd and os.path.exists(args.jd):
        with open(args.jd, "r", encoding="utf-8") as f:
            jd_text = f.read()
        jd_title = Path(args.jd).stem.replace("_", " ").title()
    elif os.path.exists("hackathon_assets/job_description.txt"):
        with open("hackathon_assets/job_description.txt", "r", encoding="utf-8") as f:
            jd_text = f.read()
        jd_title = "Senior AI Engineer — Founding Team"
    else:
        logger.info("No job description file supplied or found. Falling back to mock JD.")
        jd_title = "Senior AI Engineer — Founding Team"
        jd_text = """
        Job Title: Senior AI Engineer — Founding Team
        Experience Required: 5–9 years
        We need someone who is comfortable with two things:
        - Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
        - Scrappy product-engineering attitude.
        Required: Production experience with embeddings-based retrieval systems (sentence-transformers), vector databases (Pinecone, FAISS), and strong Python.
        """
        
    # 3. Parse JD into rubric
    jd_parser = JobDescriptionParser()
    jd = jd_parser.parse(jd_text, jd_title)
    
    # 4. Run pipeline
    pipeline = InferencePipeline()
    rankings = pipeline.run(candidates, jd)
    
    # 5. Write submission CSV
    # Required columns: candidate_id, rank, score, reasoning
    logger.info(f"Writing exactly 100 ranked candidates to {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header row
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rankings:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['calibrated_score']:.4f}",
                r["explanation_narrative"]
            ])
            
    logger.info(f"Ranking and CSV generation completed successfully. File saved to {output_path}")

if __name__ == "__main__":
    main()
