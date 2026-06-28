import os
import json
import argparse
import sys
from pathlib import Path

# Add project root to python path to avoid import errors
sys.path.append(str(Path(__file__).resolve().parent.parent))

from edhc.app.config.settings import settings
from edhc.app.schemas.candidate import CandidateProfileRaw
from edhc.app.preprocessing.normalizer import CandidateNormalizer
from edhc.app.utils.logger import get_logger

logger = get_logger("preprocess_cli")

def create_sample_raw_files(raw_dir: Path):
    """Write dummy raw profiles so the pipeline is runnable immediately."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    sample_candidate = {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone": "+1-555-0199",
        "summary": "Senior Machine Learning Engineer with 8 years of experience designing ranking systems, recommender engines, and training deep learning models. Proficient in Python, PyTorch, LightGBM, and FastAPI.",
        "skills": ["Python", "PyTorch", "LightGBM", "FastAPI", "SQL", "Docker", "Kubernetes", "Search Systems"],
        "raw_experience": [
            {
                "job_title": "Senior ML Engineer",
                "company_name": "Google",
                "start_date": "2021-06-01",
                "end_date": "Present",
                "description": "Led a team designing the search ranking algorithm. Built information retrieval system using hybrid retrieval (BM25 + dense embeddings). Wrote production pipelines in Python."
            },
            {
                "job_title": "Software Engineer (ML)",
                "company_name": "Spotify",
                "start_date": "2018-03-01",
                "end_date": "2021-05-30",
                "description": "Implemented recommendation systems using Python and PyTorch. Handled deployment of model pipelines in production."
            }
        ],
        "raw_projects": [
            {
                "title": "Search Rank Engine",
                "description": "Created custom search ranking engine using LightGBM LambdaMART and rank-bm25.",
                "skills_used": ["Python", "LightGBM", "rank-bm25"],
                "url": "https://github.com/janedoe/search-rank"
            }
        ],
        "raw_education": [
            {
                "degree": "MS",
                "major": "Computer Science",
                "institution": "Stanford University",
                "graduation_year": 2018
            }
        ]
    }

    sample_file = raw_dir / "jane_doe_raw.json"
    if not sample_file.exists():
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(sample_candidate, f, indent=4)
        logger.info(f"Created sample raw candidate profile: {sample_file}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess and normalize raw candidate profiles.")
    parser.add_argument("--raw-dir", type=str, default=str(settings.RAW_DATA_DIR), help="Path to raw candidates folder")
    parser.add_argument("--processed-dir", type=str, default=str(settings.PROCESSED_DATA_DIR), help="Path to output processed folder")
    parser.add_argument("--generate-samples", action="store_true", help="Generate sample files if empty")
    
    args = parser.parse_args()
    
    settings.create_directories()
    
    raw_path = Path(args.raw_dir)
    processed_path = Path(args.processed_dir)
    
    if args.generate_samples or not any(raw_path.glob("*.json")):
        create_sample_raw_files(raw_path)

    normalizer = CandidateNormalizer()
    
    raw_files = list(raw_path.glob("*.json"))
    logger.info(f"Found {len(raw_files)} raw profile file(s) to process.")
    
    success_count = 0
    for file_path in raw_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            raw_profile = CandidateProfileRaw(**data)
            normalized_profile = normalizer.normalize(raw_profile)
            
            # Save canonical profile
            output_file = processed_path / f"{normalized_profile.id}.json"
            with open(output_file, "w", encoding="utf-8") as out:
                out.write(normalized_profile.model_dump_json(indent=4))
                
            logger.info(f"Successfully normalized and saved: {file_path.name} -> {output_file.name}")
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            
    logger.info(f"Preprocessing completed. Normalized {success_count}/{len(raw_files)} profiles.")

if __name__ == "__main__":
    main()
