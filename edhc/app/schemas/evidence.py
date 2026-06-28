from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class EvidenceSourceType(str, Enum):
    SKILL_LIST = "skill_list"
    SUMMARY_DECLARED = "summary_declared"
    EXPERIENCE_DESCRIPTION = "experience_description"
    PROJECT_DESCRIPTION = "project_description"
    EDUCATION_MAJOR = "education_major"
    EXTERNAL_ASSESSMENT = "external_assessment"

class EvidenceToken(BaseModel):
    """An individual unit of evidence confirming a candidate capability."""
    source_type: EvidenceSourceType = Field(..., description="Where the evidence came from")
    source_identifier: str = Field(..., description="Specific experience ID, project title, or index")
    matched_text: str = Field(..., description="The exact matching segment of text or phrase")
    confidence: float = Field(default=1.0, description="Confidence of this specific evidence token, range 0.0 to 1.0")

class CompetencyVerification(BaseModel):
    """The result of auditing a candidate profile against a single competency."""
    competency_name: str = Field(..., description="Name of competency audited")
    verified: bool = Field(..., description="True if minimum evidence requirements are met")
    evidence_tokens: List[EvidenceToken] = Field(default_factory=list, description="All supporting tokens found")
    confidence_score: float = Field(..., description="Aggregate confidence calculation based on token agreement")
    audit_notes: List[str] = Field(default_factory=list, description="Auditing commentary (e.g. gaps found, contradictions)")
    evidence_confidence_level: str = Field(default="Low", description="Confidence level based on multi-source presence (High, Medium, Low)")

class EvidenceLedger(BaseModel):
    """A structured repository of verified capabilities and contradictions for a candidate."""
    candidate_id: str = Field(..., description="Reference candidate ID")
    verifications: Dict[str, CompetencyVerification] = Field(..., description="Competency name mapped to verification result")
    global_contradiction_score: float = Field(default=0.0, description="Overall contradiction rating")
    contradictions: List[str] = Field(default_factory=list, description="List of detected inconsistencies")
    strengths: List[str] = Field(default_factory=list, description="Highly verified competencies")
    weaknesses: List[str] = Field(default_factory=list, description="Missing or weak competencies")
    verified_skill_score: float = Field(default=0.0, description="Sum of confidence scores of verified skills")
    verified_skill_confidence: float = Field(default=0.0, description="Average confidence score of verified skills")
    cross_source_support: List[str] = Field(default_factory=list, description="List of sources found across verified skills")

class ReasoningExplanation(BaseModel):
    """Factual explanation generated for a candidate's suitability."""
    candidate_id: str = Field(..., description="Reference candidate ID")
    overall_verdict: str = Field(..., description="Brief candidate summary evaluation")
    factual_justifications: List[str] = Field(..., description="List of justifications constructed from verified evidence")
    hallucination_safe_explanation: str = Field(..., description="Aggregated markdown textual narrative")
