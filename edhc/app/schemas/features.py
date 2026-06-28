from typing import Dict, List, Any, Optional
import numpy as np
from pydantic import BaseModel, Field, model_validator

class CandidateFeatures(BaseModel):
    """Structured candidate features schema for ranking model input."""
    candidate_id: str = Field(..., description="Reference candidate ID")
    
    # === ORIGINAL 12 FEATURES (Index 0 to 11, locked for backward compatibility) ===
    services_company_ratio: float = Field(default=0.0, description="Ratio of career tenure in services/consulting firms")
    product_company_ratio: float = Field(default=0.0, description="Ratio of career tenure in product companies")
    startup_company_ratio: float = Field(default=0.0, description="Ratio of career tenure in startups (<500 size)")
    enterprise_company_ratio: float = Field(default=0.0, description="Ratio of career tenure in enterprises (>5000 size)")
    career_stability_score: float = Field(default=1.0, description="Nonlinear tenure stability score")
    notice_period_score: float = Field(default=0.5, description="Stated notice period score (higher = immediate)")
    domain_relevance_score: float = Field(default=0.0, description="Semantic/lexical matching of titles/descriptions to AI/ML/IR roles")
    promotion_trajectory_score: float = Field(default=0.0, description="Score indicating responsibility growth over time")
    evidence_strength: float = Field(default=0.0, description="Score representing corroborated evidence across summary, career, projects")
    impact_extraction_score: float = Field(default=0.0, description="Presence and quality of quantitative impact metrics")
    credibility_score: float = Field(default=1.0, description="Probabilistic profile consistency and plausibility score")
    rrf_retrieval_score: float = Field(default=0.0, description="Reciprocal Rank Fusion retrieval relevance score")

    # === NEW RETRIEVAL FEATURES ===
    bm25_retrieval_score: float = Field(default=0.0, description="Raw BM25 score from lexical retrieval stage")
    dense_retrieval_score: float = Field(default=0.0, description="Semantic similarity score from dense transformer model")
    retrieval_rank_percentile: float = Field(default=0.0, description="Percentile rank of candidate in initial retrieval pool")

    # === NEW SEMANTIC FEATURES ===
    jd_title_similarity: float = Field(default=0.0, description="Semantic similarity between JD title and candidate current/recent titles")
    jd_summary_similarity: float = Field(default=0.0, description="Semantic similarity between JD summary and candidate profile summary")
    competency_coverage: float = Field(default=0.0, description="Fraction of rubric competencies with verified evidence")
    skill_overlap_ratio: float = Field(default=0.0, description="Ratio of JD required skills found in candidate's skills")

    # === NEW CAREER FEATURES ===
    relevant_experience_years: float = Field(default=0.0, description="Estimated years of relevant work experience in target domain")
    average_tenure_years: float = Field(default=0.0, description="Average duration in years across all career history roles")
    career_continuity_score: float = Field(default=1.0, description="Score penalizing large chronological gaps between jobs")
    promotion_velocity: float = Field(default=0.0, description="Average number of years taken to achieve promotions")
    leadership_indicator_score: float = Field(default=0.0, description="Strength of management, lead, or architectural keywords in titles")
    technical_specialization_trend: float = Field(default=0.0, description="Trend vector showing increasing specialization in ML/Search over time")

    # === NEW SKILLS FEATURES ===
    num_verified_skills: float = Field(default=0.0, description="Total count of skills backed by work experience or project evidence")
    cross_source_verification_count: float = Field(default=0.0, description="Total count of skills verified across 2+ independent sources")
    skill_diversity_score: float = Field(default=0.0, description="Diversity of verified skill tags across different technical categories")
    skill_recency_score: float = Field(default=0.0, description="Recency score weighting skills used in current/recent roles higher")
    skill_longevity_score: float = Field(default=0.0, description="Total cumulative months of experience across all verified skills")

    # === NEW BEHAVIORAL FEATURES ===
    notice_period_normalized: float = Field(default=0.0, description="Notice period days normalized to a 0-1 scale where immediate is 1.0")
    relocation_willingness: float = Field(default=0.0, description="Binary or scaled score representing candidate's relocation flexibility")
    availability_indicator: float = Field(default=0.0, description="Activity-based platform engagement score indicating immediate availability")

    # === NEW CREDIBILITY FEATURES ===
    timeline_consistency: float = Field(default=1.0, description="Absence of chronological timeline overlaps or gaps")
    contradiction_count: float = Field(default=0.0, description="Raw number of contradictions or alerts triggered during audits")
    profile_completeness: float = Field(default=0.0, description="Overall completeness of candidate's profile sections and signals")
    cross_field_agreement: float = Field(default=1.0, description="Consistency between stated summary achievements and career history details")

    # === NEW IMPACT FEATURES ===
    quantitative_achievement_count: float = Field(default=0.0, description="Count of discrete numerical achievement assertions in descriptions")
    performance_metric_mentions: float = Field(default=0.0, description="Scaled density of search/evaluation metric terms (MRR, NDCG, Latency)")

    # === DEPRECATED BACKWARD COMPATIBILITY FIELDS ===
    job_id: Optional[str] = Field(default=None, description="Deprecated: Job reference ID")
    relevant_years: Optional[float] = Field(default=None, description="Deprecated: use relevant_experience_years")
    confidence: Optional[float] = Field(default=None, description="Deprecated: use evidence_strength")
    credibility: Optional[float] = Field(default=None, description="Deprecated: use credibility_score")

    @model_validator(mode="before")
    @classmethod
    def map_deprecated_fields(cls, data: Any) -> Any:
        """Map legacy feature field names to new canonical ones prior to validation."""
        if isinstance(data, dict):
            # Map legacy credibility to credibility_score
            if "credibility" in data and "credibility_score" not in data:
                data["credibility_score"] = data["credibility"]
            # Map legacy confidence to evidence_strength
            if "confidence" in data and "evidence_strength" not in data:
                data["evidence_strength"] = data["confidence"]
            # Map legacy relevant_years to relevant_experience_years
            if "relevant_years" in data and "relevant_experience_years" not in data:
                data["relevant_experience_years"] = data["relevant_years"]
        return data

    def to_array(self) -> np.ndarray:
        """Convert features to a flat NumPy array for models (excluding candidate_id)."""
        return np.array([
            # Original 12
            self.services_company_ratio,
            self.product_company_ratio,
            self.startup_company_ratio,
            self.enterprise_company_ratio,
            self.career_stability_score,
            self.notice_period_score,
            self.domain_relevance_score,
            self.promotion_trajectory_score,
            self.evidence_strength,
            self.impact_extraction_score,
            self.credibility_score,
            self.rrf_retrieval_score,
            
            # New retrieval (3)
            self.bm25_retrieval_score,
            self.dense_retrieval_score,
            self.retrieval_rank_percentile,
            
            # New semantic (4)
            self.jd_title_similarity,
            self.jd_summary_similarity,
            self.competency_coverage,
            self.skill_overlap_ratio,
            
            # New career (6)
            self.relevant_experience_years,
            self.average_tenure_years,
            self.career_continuity_score,
            self.promotion_velocity,
            self.leadership_indicator_score,
            self.technical_specialization_trend,
            
            # New skills (5)
            self.num_verified_skills,
            self.cross_source_verification_count,
            self.skill_diversity_score,
            self.skill_recency_score,
            self.skill_longevity_score,
            
            # New behavioral (3)
            self.notice_period_normalized,
            self.relocation_willingness,
            self.availability_indicator,
            
            # New credibility (4)
            self.timeline_consistency,
            self.contradiction_count,
            self.profile_completeness,
            self.cross_field_agreement,
            
            # New impact (2)
            self.quantitative_achievement_count,
            self.performance_metric_mentions
        ], dtype=np.float32)

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Return the names of numerical features in the correct order."""
        return [
            # Original 12
            "services_company_ratio",
            "product_company_ratio",
            "startup_company_ratio",
            "enterprise_company_ratio",
            "career_stability_score",
            "notice_period_score",
            "domain_relevance_score",
            "promotion_trajectory_score",
            "evidence_strength",
            "impact_extraction_score",
            "credibility_score",
            "rrf_retrieval_score",
            
            # New retrieval (3)
            "bm25_retrieval_score",
            "dense_retrieval_score",
            "retrieval_rank_percentile",
            
            # New semantic (4)
            "jd_title_similarity",
            "jd_summary_similarity",
            "competency_coverage",
            "skill_overlap_ratio",
            
            # New career (6)
            "relevant_experience_years",
            "average_tenure_years",
            "career_continuity_score",
            "promotion_velocity",
            "leadership_indicator_score",
            "technical_specialization_trend",
            
            # New skills (5)
            "num_verified_skills",
            "cross_source_verification_count",
            "skill_diversity_score",
            "skill_recency_score",
            "skill_longevity_score",
            
            # New behavioral (3)
            "notice_period_normalized",
            "relocation_willingness",
            "availability_indicator",
            
            # New credibility (4)
            "timeline_consistency",
            "contradiction_count",
            "profile_completeness",
            "cross_field_agreement",
            
            # New impact (2)
            "quantitative_achievement_count",
            "performance_metric_mentions"
        ]

    def to_dict(self) -> Dict[str, float]:
        """Convert numerical features into a dictionary."""
        names = self.get_feature_names()
        values = self.to_array().tolist()
        return dict(zip(names, values))
