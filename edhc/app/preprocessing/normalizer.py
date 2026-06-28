import re
import hashlib
from datetime import date, datetime
from typing import Dict, Any, List, Optional
from edhc.app.schemas.candidate import CandidateProfileRaw, CandidateProfileNormalized, WorkExperience, Project, Education
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class CandidateNormalizer:
    """Standardizes raw parsed resume/profile fields into structured canonical representations."""

    def normalize(self, raw: CandidateProfileRaw) -> CandidateProfileNormalized:
        """Main method to convert raw profile schema to normalized schema."""
        logger.info(f"Normalizing candidate profile: {raw.name}")
        
        # Calculate consistent ID if not provided
        raw_email = raw.email or ""
        profile_hash = hashlib.sha256(f"{raw.name}_{raw_email}".encode("utf-8")).hexdigest()[:16]
        
        # Ingest experiences
        experiences: List[WorkExperience] = []
        for exp in raw.raw_experience:
            try:
                norm_exp = self.normalize_experience(exp)
                experiences.append(norm_exp)
            except Exception as e:
                logger.error(f"Error normalizing experience item {exp}: {e}")

        # Ingest projects
        projects: List[Project] = []
        for proj in raw.raw_projects:
            try:
                norm_proj = self.normalize_project(proj)
                projects.append(norm_proj)
            except Exception as e:
                logger.error(f"Error normalizing project item {proj}: {e}")

        # Ingest education
        education: List[Education] = []
        for edu in raw.raw_education:
            try:
                norm_edu = self.normalize_education(edu)
                education.append(norm_edu)
            except Exception as e:
                logger.error(f"Error normalizing education item {edu}: {e}")

        # Standardize skills
        normalized_skills = [self.clean_text(s).lower() for s in raw.skills if s]

        return CandidateProfileNormalized(
            id=profile_hash,
            name=raw.name,
            email=raw.email,
            headline=self.clean_text(raw.summary[:100]) if raw.summary else None,
            summary=self.clean_text(raw.summary),
            skills=list(set(normalized_skills)),
            experiences=experiences,
            projects=projects,
            education=education,
            metadata={"normalized_at": date.today().isoformat()}
        )

    def clean_text(self, text: str) -> str:
        """Clean string by resolving whitespaces, stripping html tags, and unescaping characters."""
        if not text:
            return ""
        # Remove HTML
        text = re.sub(r"<[^>]*>", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize_experience(self, raw_exp: Dict[str, Any]) -> WorkExperience:
        """Parse, clean, and standardize work experience details."""
        start_date = self.parse_date(raw_exp.get("start_date"))
        end_date_str = raw_exp.get("end_date")
        
        is_current = False
        if not end_date_str or str(end_date_str).lower().strip() in ["present", "current", "now", "ongoing"]:
            is_current = True
            end_date = None
        else:
            end_date = self.parse_date(end_date_str)

        duration_years = self.calculate_duration(start_date, end_date)
        original_title = raw_exp.get("job_title", "Unknown Role")
        
        return WorkExperience(
            job_title=original_title,
            normalized_title=self.standardize_title(original_title),
            company_name=raw_exp.get("company_name", "Unknown Company"),
            company_tier=self.estimate_company_tier(raw_exp.get("company_name", "")),
            start_date=start_date,
            end_date=end_date,
            is_current=is_current,
            description=self.clean_text(raw_exp.get("description", "")),
            duration_years=duration_years
        )

    def normalize_project(self, raw_proj: Dict[str, Any]) -> Project:
        """Normalize project listings."""
        raw_skills = raw_proj.get("skills_used", [])
        if isinstance(raw_skills, str):
            skills = [s.strip() for s in raw_skills.split(",") if s.strip()]
        else:
            skills = [str(s).strip() for s in raw_skills if s]

        return Project(
            title=raw_proj.get("title", "Unnamed Project"),
            description=self.clean_text(raw_proj.get("description", "")),
            skills_used=skills,
            url=raw_proj.get("url")
        )

    def normalize_education(self, raw_edu: Dict[str, Any]) -> Education:
        """Normalize education milestones."""
        grad_year_val = raw_edu.get("graduation_year")
        grad_year = int(grad_year_val) if grad_year_val and str(grad_year_val).isdigit() else None
        
        return Education(
            degree=raw_edu.get("degree", "Unknown Degree"),
            major=raw_edu.get("major", "General"),
            institution=raw_edu.get("institution", "Unknown Institution"),
            graduation_year=grad_year
        )

    def parse_date(self, date_val: Any) -> date:
        """Attempt parsing common date formats or return default fallback date."""
        if not date_val:
            return date.today()
        if isinstance(date_val, (date, datetime)):
            return date_val.date() if isinstance(date_val, datetime) else date_val

        date_str = str(date_val).strip()
        
        # Support formats: "YYYY-MM-DD", "YYYY-MM", "Month YYYY", "YYYY"
        formats = [
            "%Y-%m-%d", "%Y-%m", "%B %Y", "%b %Y", "%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        # Simple regex fallbacks for YYYY
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", date_str)
        if year_match:
            return date(int(year_match.group(1)), 1, 1)

        logger.warning(f"Could not parse date: '{date_str}'. Falling back to current date.")
        return date.today()

    def calculate_duration(self, start: date, end: Optional[date]) -> float:
        """Calculate years of experience between start and end dates."""
        end_resolved = end or date.today()
        days = (end_resolved - start).days
        years = days / 365.25
        return max(0.0, round(years, 2))

    def standardize_title(self, title: str) -> str:
        """Normalize job titles based on predefined rules."""
        t = title.lower()
        if any(w in t for w in ["principal", "staff", "director", "head", "chief", "vp"]):
            lead_pref = "Staff/Principal "
        elif any(w in t for w in ["sr", "senior", "lead"]):
            lead_pref = "Senior "
        elif any(w in t for w in ["jr", "junior", "associate"]):
            lead_pref = "Junior "
        else:
            lead_pref = ""

        # Map domain
        if any(w in t for w in ["machine learning", "ml", "computer vision", "nlp", "deep learning"]):
            return f"{lead_pref}Machine Learning Engineer"
        elif any(w in t for w in ["data scientist", "ds"]):
            return f"{lead_pref}Data Scientist"
        elif any(w in t for w in ["data engineer", "etl"]):
            return f"{lead_pref}Data Engineer"
        elif any(w in t for w in ["backend", "back-end", "python developer"]):
            return f"{lead_pref}Backend Engineer"
        elif any(w in t for w in ["frontend", "front-end", "react"]):
            return f"{lead_pref}Frontend Engineer"
        elif any(w in t for w in ["software engineer", "swe", "developer"]):
            return f"{lead_pref}Software Engineer"
        
        return title.strip()

    def estimate_company_tier(self, company_name: str) -> int:
        """Estimate company tier (1 = top tech/FAANG, 2 = known mid-tier, 3 = general)."""
        name = company_name.lower()
        tier_1 = ["google", "apple", "facebook", "meta", "amazon", "netflix", "microsoft", "openai", "stripe", "uber"]
        tier_2 = ["airbnb", "spotify", "salesforce", "zoom", "twitter", "x.com", "hubspot", "datadog", "adobe"]
        
        if any(t in name for t in tier_1):
            return 1
        elif any(t in name for t in tier_2):
            return 2
        return 3
