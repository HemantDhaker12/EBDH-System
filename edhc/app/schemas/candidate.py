from datetime import date
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# === STANDARD HACKATHON CANDIDATE SCHEMA ===

class CandidateProfileField(BaseModel):
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

class CareerHistoryField(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

class EducationField(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: Optional[str] = None

class SkillField(BaseModel):
    name: str
    proficiency: str
    endorsements: int
    duration_months: int

class CertificationField(BaseModel):
    name: str
    issuer: str
    year: int

class LanguageField(BaseModel):
    language: str
    proficiency: str

class SalaryRangeField(BaseModel):
    min: float
    max: float

class RedrobSignalsField(BaseModel):
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: Dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: SalaryRangeField
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

class CandidateProfile(BaseModel):
    candidate_id: str
    profile: CandidateProfileField
    career_history: List[CareerHistoryField] = Field(default_factory=list)
    education: List[EducationField] = Field(default_factory=list)
    skills: List[SkillField] = Field(default_factory=list)
    certifications: List[CertificationField] = Field(default_factory=list)
    languages: List[LanguageField] = Field(default_factory=list)
    redrob_signals: RedrobSignalsField


# === NORMALIZATION AND TEST COMPATIBILITY SCHEMAS ===

class WorkExperience(BaseModel):
    job_title: str
    normalized_title: Optional[str] = None
    company_name: str
    company_tier: Optional[int] = 3
    start_date: date
    end_date: Optional[date] = None
    is_current: bool = False
    description: str
    duration_years: float

class Project(BaseModel):
    title: str
    description: str
    skills_used: List[str] = Field(default_factory=list)
    url: Optional[str] = None

class Education(BaseModel):
    degree: str
    major: str
    institution: str
    graduation_year: Optional[int] = None

class CandidateProfileRaw(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    raw_experience: List[Dict[str, Any]] = Field(default_factory=list)
    raw_projects: List[Dict[str, Any]] = Field(default_factory=list)
    raw_education: List[Dict[str, Any]] = Field(default_factory=list)

class CandidateProfileNormalized(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    headline: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experiences: List[WorkExperience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def normalize_candidate(candidate: Any) -> CandidateProfile:
    """Helper function to normalize and convert a CandidateProfileNormalized (or similar schema)
    into the standard CandidateProfile representation required by the analyzers.
    """
    if isinstance(candidate, CandidateProfile):
        return candidate

    # 1. Candidate ID
    cand_id = getattr(candidate, "candidate_id", getattr(candidate, "id", "unknown")) or "unknown"

    # 2. Extract experiences
    experiences = getattr(candidate, "experiences", []) or []
    if not experiences and hasattr(candidate, "career_history"):
        experiences = candidate.career_history

    # Determine current title and company from experiences
    current_title = ""
    current_company = ""
    current_company_size = "51-200"
    current_industry = "Software"

    # Find current experience
    current_exp = None
    for exp in experiences:
        if getattr(exp, "is_current", False):
            current_exp = exp
            break
    if not current_exp and experiences:
        current_exp = experiences[0]

    if current_exp:
        current_title = getattr(current_exp, "job_title", getattr(current_exp, "title", "")) or ""
        current_company = getattr(current_exp, "company_name", getattr(current_exp, "company", "")) or ""
        tier = getattr(current_exp, "company_tier", 3)
        if tier == 1:
            current_company_size = "10001+"
        elif tier == 2:
            current_company_size = "501-1000"
        else:
            current_company_size = "51-200"
        current_industry = getattr(current_exp, "industry", "Software") or "Software"

    # Sum up years of experience
    total_yoe = 0.0
    for exp in experiences:
        if hasattr(exp, "duration_years"):
            total_yoe += getattr(exp, "duration_years", 0.0) or 0.0
        elif hasattr(exp, "duration_months"):
            total_yoe += (getattr(exp, "duration_months", 0.0) or 0.0) / 12.0

    profile_field = CandidateProfileField(
        anonymized_name=getattr(candidate, "name", "Unknown Name") or "Unknown Name",
        headline=getattr(candidate, "headline", "") or "",
        summary=getattr(candidate, "summary", "") or "",
        location=getattr(candidate, "location", "Unknown") or "Unknown",
        country=getattr(candidate, "country", "Unknown") or "Unknown",
        years_of_experience=total_yoe,
        current_title=current_title,
        current_company=current_company,
        current_company_size=current_company_size,
        current_industry=current_industry
    )

    # 3. Career History
    career_history = []
    for exp in experiences:
        co = getattr(exp, "company_name", getattr(exp, "company", "Unknown Company")) or "Unknown Company"
        ti = getattr(exp, "job_title", getattr(exp, "title", "Unknown Role")) or "Unknown Role"

        start = getattr(exp, "start_date", None)
        if isinstance(start, date):
            start_str = start.isoformat()
        else:
            start_str = str(start or "2023-01-01")

        end = getattr(exp, "end_date", None)
        if isinstance(end, date):
            end_str = end.isoformat()
        elif end is not None:
            end_str = str(end)
        else:
            end_str = None

        dur_m = getattr(exp, "duration_months", None)
        if dur_m is None:
            dur_y = getattr(exp, "duration_years", 0.0) or 0.0
            dur_m = int(dur_y * 12.0)

        is_curr = getattr(exp, "is_current", False)
        ind = getattr(exp, "industry", "Software") or "Software"

        tier = getattr(exp, "company_tier", 3)
        if tier == 1:
            co_size = "10001+"
        elif tier == 2:
            co_size = "501-1000"
        else:
            co_size = "51-200"

        desc = getattr(exp, "description", "") or ""

        career_history.append(CareerHistoryField(
            company=co,
            title=ti,
            start_date=start_str,
            end_date=end_str,
            duration_months=dur_m,
            is_current=is_curr,
            industry=ind,
            company_size=co_size,
            description=desc
        ))

    # 4. Education
    education_fields = []
    edu_list = getattr(candidate, "education", []) or []
    for edu in edu_list:
        inst = getattr(edu, "institution", "Unknown") or "Unknown"
        deg = getattr(edu, "degree", "Unknown") or "Unknown"
        major = getattr(edu, "major", getattr(edu, "field_of_study", "Software Engineering")) or "Software Engineering"
        grad_year = getattr(edu, "graduation_year", getattr(edu, "end_year", 2020)) or 2020
        start_year = getattr(edu, "start_year", grad_year - 4) or (grad_year - 4)

        education_fields.append(EducationField(
            institution=inst,
            degree=deg,
            field_of_study=major,
            start_year=start_year,
            end_year=grad_year,
            grade=getattr(edu, "grade", "A"),
            tier=getattr(edu, "tier", "Tier 3")
        ))

    # 5. Skills
    skills_fields = []
    skills_list = getattr(candidate, "skills", []) or []
    for s in skills_list:
        if isinstance(s, str):
            skills_fields.append(SkillField(
                name=s,
                proficiency="expert",
                endorsements=5,
                duration_months=24
            ))
        else:
            name = getattr(s, "name", "")
            prof = getattr(s, "proficiency", "expert")
            ends = getattr(s, "endorsements", 5)
            dur = getattr(s, "duration_months", 24)
            skills_fields.append(SkillField(
                name=name,
                proficiency=prof,
                endorsements=ends,
                duration_months=dur
            ))

    # 6. Redrob Signals
    if hasattr(candidate, "redrob_signals") and candidate.redrob_signals is not None:
        redrob_signals = candidate.redrob_signals
    else:
        redrob_signals = RedrobSignalsField(
            profile_completeness_score=80.0,
            signup_date="2023-01-01",
            last_active_date="2023-01-01",
            open_to_work_flag=True,
            profile_views_received_30d=10,
            applications_submitted_30d=5,
            recruiter_response_rate=0.8,
            avg_response_time_hours=24.0,
            skill_assessment_scores={},
            connection_count=100,
            endorsements_received=5,
            notice_period_days=30,
            expected_salary_range_inr_lpa=SalaryRangeField(min=10.0, max=20.0),
            preferred_work_mode="hybrid",
            willing_to_relocate=True,
            github_activity_score=50.0,
            search_appearance_30d=5,
            saved_by_recruiters_30d=2,
            interview_completion_rate=0.9,
            offer_acceptance_rate=0.9,
            verified_email=True,
            verified_phone=True,
            linkedin_connected=True
        )

    return CandidateProfile(
        candidate_id=cand_id,
        profile=profile_field,
        career_history=career_history,
        education=education_fields,
        skills=skills_fields,
        redrob_signals=redrob_signals
    )

