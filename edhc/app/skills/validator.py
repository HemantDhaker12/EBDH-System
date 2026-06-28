import re
from typing import List, Dict, Any

from edhc.app.schemas.candidate import CandidateProfile
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

# Synonym Mapping for Terminology Normalization
SYNONYM_MAP = {
    "ml": "machine learning",
    "machine learning": "ml",
    "dl": "deep learning",
    "deep learning": "dl",
    "nlp": "natural language processing",
    "natural language processing": "nlp",
    "k8s": "kubernetes",
    "kubernetes": "k8s",
    "recommender": "recommendation",
    "recommendation": "recommender",
    "vector db": "vector search",
    "vector database": "vector search",
    "ir": "information retrieval",
    "information retrieval": "ir",
    "ltr": "learning to rank",
    "learning to rank": "ltr"
}

class SkillsValidator:
    """Audits skills declared by the candidate by cross-referencing experience, duration, and education."""

    def validate(self, candidate: Any, target_skills: List[str]) -> Dict[str, Dict[str, Any]]:
        """Validate candidate capabilities against target skills and return detailed validations with confidence scores."""
        cid = getattr(candidate, "candidate_id", getattr(candidate, "id", "unknown"))
        logger.info(f"Validating skills for candidate {cid} against target skills: {target_skills}")
        
        results = {}
        for skill in target_skills:
            results[skill] = self.evaluate_skill_evidence(candidate, skill)
            
        return results

    def normalize_and_map_skill(self, skill: str) -> str:
        """Clean skill string and retrieve its standard mapped synonym if it exists."""
        return skill.lower().strip()

    def skill_matches(self, s1: str, s2: str) -> bool:
        """Check if two skill strings match, supporting synonym mapping."""
        s1_clean = s1.lower().strip()
        s2_clean = s2.lower().strip()
        if s1_clean == s2_clean:
            return True
        if SYNONYM_MAP.get(s1_clean) == s2_clean:
            return True
        if SYNONYM_MAP.get(s2_clean) == s1_clean:
            return True
        return False

    def evaluate_skill_evidence(self, candidate: Any, skill: str) -> Dict[str, Any]:
        """Perform cross-referencing checks for a single skill."""
        skill_lower = skill.lower().strip()
        
        # 1. Check direct declared skills list and its duration
        in_declared_list = False
        declared_duration = 0.0
        
        skills_list = getattr(candidate, "skills", [])
        for s in skills_list:
            if isinstance(s, str):
                s_name = s
                s_dur = 0.0
            else:
                s_name = getattr(s, "name", "")
                s_dur = float(getattr(s, "duration_months", 0.0))
                
            if self.skill_matches(skill_lower, s_name):
                in_declared_list = True
                declared_duration = max(declared_duration, s_dur)

        # Build regex pattern for search including synonyms
        search_terms = [re.escape(skill_lower)]
        if skill_lower in SYNONYM_MAP:
            search_terms.append(re.escape(SYNONYM_MAP[skill_lower]))
        # Also find reverse mapping
        for k, v in SYNONYM_MAP.items():
            if v == skill_lower:
                search_terms.append(re.escape(k))
        search_pattern = r"\b(" + "|".join(search_terms) + r")\b"
        search_regex = re.compile(search_pattern, re.IGNORECASE)

        # 2. Check work experiences
        experience_matches = 0
        total_relevant_months = 0.0
        exp_descriptions_matched = []
        
        # Support both CandidateProfile (career_history) and CandidateProfileNormalized (experiences)
        career_history = getattr(candidate, "career_history", [])
        if not career_history and hasattr(candidate, "experiences"):
            career_history = candidate.experiences
            
        for exp in career_history:
            title = getattr(exp, "title", getattr(exp, "job_title", ""))
            company = getattr(exp, "company", getattr(exp, "company_name", "Unknown Company"))
            desc = getattr(exp, "description", "")
            dur_m = getattr(exp, "duration_months", int(getattr(exp, "duration_years", 0.0) * 12))
            
            # Combine job title and description text
            text = f"{title} {desc}"
            
            if search_regex.search(text):
                experience_matches += 1
                total_relevant_months += float(dur_m)
                exp_descriptions_matched.append(company)

        # 3. Check education
        edu_match = False
        education = getattr(candidate, "education", [])
        for edu in education:
            degree = getattr(edu, "degree", "")
            major = getattr(edu, "field_of_study", getattr(edu, "major", ""))
            inst = getattr(edu, "institution", "")
            text = f"{degree} {major} {inst}"
            if search_regex.search(text):
                edu_match = True
                break

        # 4. Check projects
        project_matches = 0
        projects = getattr(candidate, "projects", [])
        for proj in projects:
            title = getattr(proj, "title", "")
            desc = getattr(proj, "description", "")
            skills_used = getattr(proj, "skills_used", [])
            
            text = f"{title} {desc}"
            # Check explicit skills list or search text
            has_explicit = any(self.skill_matches(skill_lower, str(s)) for s in skills_used)
            if has_explicit or search_regex.search(text):
                project_matches += 1

        # Calculate Confidence Score (0.0 to 1.0)
        # Weights: Declared list presence (15%), Declared duration scaling (20%), Experience duration scaling (45%), Education match (10%), Project match (10%)
        declared_score = 0.15 if in_declared_list else 0.0
        
        # Declared duration component (up to 2 years / 24 months for maximum score)
        declared_dur_score = min(0.20, (declared_duration / 24.0) * 0.20) if declared_duration > 0 else 0.0
        
        # Experience component scales with duration (up to 3 years / 36 months for maximum score)
        if total_relevant_months >= 36.0:
            exp_score = 0.55
        elif total_relevant_months > 0:
            exp_score = (total_relevant_months / 36.0) * 0.55
        else:
            exp_score = 0.0
            
        edu_score = 0.10 if edu_match else 0.0
        proj_score = 0.10 if project_matches > 0 else 0.0
        
        confidence_score = declared_score + declared_dur_score + exp_score + edu_score + proj_score
        
        # Compile sources list
        sources = []
        if in_declared_list:
            sources.append("declared_skills")
        if experience_matches > 0:
            sources.append("work_experience")
        if edu_match:
            sources.append("education")
        if project_matches > 0:
            sources.append("projects")

        # Check summary section explicitly for multi-section count
        summary_match = False
        summary_text = getattr(getattr(candidate, "profile", None), "summary", getattr(candidate, "summary", "")) or ""
        if summary_text and search_regex.search(summary_text):
            summary_match = True
            sources.append("summary_declared")

        # Determine evidence confidence levels (High, Medium, Low)
        # High: appears in multiple sections (at least 3 of: career history, projects, summary, skills list)
        # Medium: appears in at least one reliable section (experience or projects)
        # Low: appears only once / weak evidence (e.g. only in declared list or only in education)
        sections_matched = sum([in_declared_list, experience_matches > 0, project_matches > 0, summary_match])
        
        if sections_matched >= 3:
            evidence_confidence_level = "High"
        elif experience_matches > 0 or project_matches > 0:
            evidence_confidence_level = "Medium"
        else:
            evidence_confidence_level = "Low"

        # Adjust confidence based on multi-source alignment (Task 3)
        # Increase confidence when evidence appears across multiple independent sources (multiplier)
        # Reduce confidence when evidence is isolated (only 1 source)
        if len(sources) >= 3:
            confidence_score = min(1.0, confidence_score * 1.1)
        elif len(sources) == 1:
            confidence_score = confidence_score * 0.7

        # Discount using continuous credibility_score (reducing confidence when contradictions exist)
        credibility = getattr(candidate, "credibility_score", getattr(candidate, "credibility", 1.0))
        confidence_score = confidence_score * credibility

        return {
            "confidence_score": round(confidence_score, 2),
            "evidence_confidence_level": evidence_confidence_level,
            "sources_found": list(set(sources)),
            "relevant_experience_years": round(total_relevant_months / 12.0, 2),
            "project_matches_count": project_matches,
            "audit_trail": {
                "in_declared_list": in_declared_list,
                "declared_duration_months": declared_duration,
                "experience_companies_matched": exp_descriptions_matched,
                "education_matched": edu_match
            }
        }
