from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Competency(BaseModel):
    """Represents a specific skill or capability required by the Job Description."""
    name: str = Field(..., description="Name of the competency, e.g., 'Python Programming'")
    description: str = Field(..., description="Details/expectations for this competency")
    weight: float = Field(default=1.0, description="Relative importance weight, normally 0.0 to 1.0")
    semantic_keywords: List[str] = Field(default_factory=list, description="List of related terms for search expansion")

class HiringRubric(BaseModel):
    """The structured hiring rubric mapped from a job description."""
    competencies: List[Competency] = Field(default_factory=list, description="Required competencies")
    min_experience_years: float = Field(default=0.0, description="Minimum years of professional experience requested")
    preferred_education: Optional[str] = Field(None, description="Preferred education degree, e.g., 'BS', 'MS', 'PhD'")
    target_industries: List[str] = Field(default_factory=list, description="Sectors or domains relevant, e.g., 'Finance', 'SaaS'")

class JobDescriptionParsed(BaseModel):
    """The canonical representation of a parsed Job Description."""
    id: str = Field(..., description="Unique job description identifier")
    title: str = Field(..., description="Job Title")
    raw_text: str = Field(..., description="Original, raw job description text")
    rubric: HiringRubric = Field(..., description="Target hiring rubric extracted from the text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata, e.g., parse timestamp, department")
