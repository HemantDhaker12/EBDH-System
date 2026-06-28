import os
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to python path to avoid import errors
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from edhc.app.schemas.candidate import CandidateProfile
from edhc.app.jd.intelligence import JobDescriptionParser
from edhc.app.features.generator import FeatureGenerator
from edhc.app.ranking.ranker import LambdaMARTRanker

def load_candidates(candidates_path: Path) -> list:
    """Load candidates from JSON or JSONL file."""
    candidates = []
    if not candidates_path.exists():
        print(f"Candidates file not found: {candidates_path}")
        return []
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        first_char = f.read(1).strip()
        f.seek(0)
        
        if first_char == "[":
            data = json.load(f)
            for item in data:
                candidates.append(CandidateProfile(**item))
        else:
            for line in f:
                if line.strip():
                    candidates.append(CandidateProfile(**json.loads(line)))
    return candidates

def main():
    parser = argparse.ArgumentParser(description="Inspect candidate features and importance.")
    parser.add_argument("--candidates", type=str, default="./hackathon_assets/sample_candidates.json", help="Path to candidates file")
    parser.add_argument("--jd", type=str, default="./hackathon_assets/job_description.txt", help="Path to Job Description file")
    parser.add_argument("--model", type=str, default="./edhc/models/lambdamart_model.pkl", help="Path to trained LambdaMART model")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    jd_path = Path(args.jd)
    model_path = Path(args.model)

    # 1. Load Candidates
    print(f"Loading candidates from {candidates_path}...")
    candidates = load_candidates(candidates_path)
    if not candidates:
        return
    print(f"Successfully loaded {len(candidates)} candidates.")

    # 2. Load and Parse JD
    jd_parser = JobDescriptionParser()
    if jd_path.exists():
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
        jd_title = jd_path.stem.replace("_", " ").title()
    else:
        print(f"Job Description file not found at {jd_path}. Using fallback.")
        jd_title = "Senior ML/Search Engineer"
        jd_text = "Required skills: Python, Machine Learning, Search, System Architecture, SQL."
    
    jd = jd_parser.parse(jd_text, jd_title)
    print(f"Parsed Job Description: {jd.title}")

    # 3. Generate Features Matrix
    feature_generator = FeatureGenerator()
    features_list = []
    print("Generating feature vectors...")
    for cand in candidates:
        features = feature_generator.generate(cand, jd)
        features_list.append(features.to_dict())

    df = pd.DataFrame(features_list)
    numerical_cols = df.columns.drop("candidate_id", errors="ignore")

    # 4. Describe Feature Distributions
    print("\n" + "="*80)
    print(" FEATURE DISTRIBUTIONS & MISSING VALUES")
    print("="*80)
    summary_df = df[numerical_cols].describe().T
    summary_df["missing_count"] = df[numerical_cols].isnull().sum()
    summary_df["missing_pct"] = (summary_df["missing_count"] / len(df)) * 100
    summary_df = summary_df[["mean", "std", "min", "max", "missing_count", "missing_pct"]]
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 100)
    print(summary_df)

    # 5. Correlation Analysis (Top correlated pairs)
    print("\n" + "="*80)
    print(" TOP 10 STRONGEST FEATURE CORRELATIONS")
    print("="*80)
    corr_matrix = df[numerical_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = upper_tri.unstack().dropna().sort_values(ascending=False)
    for i, ((feat1, feat2), val) in enumerate(corr_pairs.items(), 1):
        if i > 10:
            break
        print(f"{i:2d}. {feat1:<30} <-> {feat2:<30} : {val:.4f}")

    # 6. Model Feature Importance (if model exists)
    if model_path.exists():
        print("\n" + "="*80)
        print(" LIGHTGBM LAMBDAMART FEATURE IMPORTANCE")
        print("="*80)
        try:
            ranker = LambdaMARTRanker()
            ranker.load(str(model_path))
            
            # LGBMRanker model exposes feature_importances_
            if hasattr(ranker.model, "feature_importances_"):
                importances = ranker.model.feature_importances_
                feature_names = numerical_cols
                
                # Check if model has fewer features due to backward compatibility slicing
                if len(importances) < len(feature_names):
                    print(f"Model was fitted with {len(importances)} features (legacy model). Slicing names list.")
                    feature_names = feature_names[:len(importances)]
                
                importance_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": importances
                }).sort_values(by="importance", ascending=False)
                
                print(importance_df.to_string(index=False))
            else:
                print("Loaded model does not have feature_importances_ attribute.")
        except Exception as e:
            print(f"Could not calculate model feature importance: {e}")
    else:
        print(f"\nNo LambdaMART model found at {model_path} to evaluate feature importance.")

if __name__ == "__main__":
    main()
