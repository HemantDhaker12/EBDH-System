from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from edhc.app.schemas.jd import JobDescriptionParsed
from edhc.app.schemas.candidate import CandidateProfileRaw, CandidateProfileNormalized
from edhc.app.preprocessing.normalizer import CandidateNormalizer
from edhc.app.jd.intelligence import JobDescriptionParser

router = APIRouter()

# Global Parsers
normalizer = CandidateNormalizer()
jd_parser = JobDescriptionParser()

# In-memory simple storage for demonstration
candidates_db: Dict[str, CandidateProfileNormalized] = {}
jds_db: Dict[str, JobDescriptionParsed] = {}

class JDPayload(BaseModel):
    title: str
    text: str

@router.post("/jd/parse", response_model=JobDescriptionParsed)
def parse_job_description(payload: JDPayload):
    """Parse raw job description text into a structured rubric."""
    try:
        parsed_jd = jd_parser.parse(payload.text, payload.title)
        jds_db[parsed_jd.id] = parsed_jd
        return parsed_jd
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

@router.post("/candidates/normalize", response_model=CandidateProfileNormalized)
def normalize_candidate(raw_candidate: CandidateProfileRaw):
    """Normalize a raw candidate profile into canonical schema representation."""
    try:
        norm_candidate = normalizer.normalize(raw_candidate)
        candidates_db[norm_candidate.id] = norm_candidate
        return norm_candidate
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Normalization failed: {str(e)}")

@router.get("/candidates", response_model=List[CandidateProfileNormalized])
def list_candidates():
    """List all normalized candidates in memory."""
    return list(candidates_db.values())

@router.get("/candidates/{candidate_id}", response_model=CandidateProfileNormalized)
def get_candidate(candidate_id: str):
    """Fetch normalized details for a specific candidate ID."""
    if candidate_id not in candidates_db:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidates_db[candidate_id]

class RankPayload(BaseModel):
    jd_id: str
    candidate_ids: List[str]

@router.post("/rank")
def rank_candidates(payload: RankPayload):
    """Rank list of candidate IDs against a parsed Job Description ID."""
    if payload.jd_id not in jds_db:
        raise HTTPException(status_code=404, detail="Job description not found")
        
    jd = jds_db[payload.jd_id]
    
    missing_cids = [cid for cid in payload.candidate_ids if cid not in candidates_db]
    if missing_cids:
        raise HTTPException(status_code=400, detail=f"Unknown candidate IDs: {missing_cids}")

    # Import pipeline here to avoid circular dependencies
    from edhc.app.pipeline.inference import InferencePipeline
    
    pipeline = InferencePipeline()
    candidates = [candidates_db[cid] for cid in payload.candidate_ids]
    
    try:
        ranked_results = pipeline.run(candidates, jd)
        return ranked_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking execution failed: {str(e)}")
