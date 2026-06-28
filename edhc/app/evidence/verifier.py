import re
from typing import Dict, List, Any
from edhc.app.schemas.candidate import CandidateProfile, normalize_candidate
from edhc.app.schemas.jd import HiringRubric, Competency
from edhc.app.schemas.evidence import EvidenceLedger, CompetencyVerification, EvidenceToken, EvidenceSourceType
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)
# Synonym Mapping for Terminology Normalization
SYNONYM_MAP = {
    "ml": "machine learning",
    "machine learning": "ml",
    "dl": "deep learning",
    "deep learning": "dl",
    "nlp": "natural language processing",
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

def clean_term(t: str) -> str:
    return t.lower().strip()

def keyword_matches(kw: str, skill_name: str) -> bool:
    kw_clean = clean_term(kw)
    sk_clean = clean_term(skill_name)
    if kw_clean == sk_clean:
        return True
    if SYNONYM_MAP.get(kw_clean) == sk_clean:
        return True
    if SYNONYM_MAP.get(sk_clean) == kw_clean:
        return True
    return False

class EvidenceVerifier:
    """Verifies candidate capabilities against rubric competencies using multi-source validation."""

    def verify(self, candidate: Any, rubric: HiringRubric, credibility_score: float = 1.0) -> EvidenceLedger:
        """Evaluate candidate profile against hiring rubric, assembling an Evidence Ledger."""
        candidate = normalize_candidate(candidate)
        logger.info(f"Assembling evidence ledger for candidate {candidate.candidate_id}")
        
        verifications: Dict[str, CompetencyVerification] = {}
        strengths: List[str] = []
        weaknesses: List[str] = []
        
        for comp in rubric.competencies:
            verification = self.verify_competency(candidate, comp, credibility_score)
            verifications[comp.name] = verification
            
            # Categorize into strengths or weaknesses
            if verification.verified and verification.confidence_score >= 0.6:
                strengths.append(comp.name)
            elif not verification.verified or verification.confidence_score < 0.3:
                weaknesses.append(comp.name)

        # Detect basic timeline anomalies to populate ledger contradictions
        contradictions_list = []
        
        sal = candidate.redrob_signals.expected_salary_range_inr_lpa
        if sal.min > sal.max:
            contradictions_list.append("Expected salary min is greater than max.")
            
        expert_zero_dur = sum(1 for s in candidate.skills if s.proficiency == "expert" and s.duration_months == 0)
        if expert_zero_dur >= 3:
            contradictions_list.append(f"Claimed 'expert' in {expert_zero_dur} skills with 0 months duration.")

        global_contradiction = 1.0 if contradictions_list else 0.0

        # Calculate verified skills metrics
        verified_skills = [v for v in verifications.values() if v.verified]
        verified_skill_score = round(sum(v.confidence_score for v in verified_skills), 2)
        verified_skill_confidence = round(verified_skill_score / len(verified_skills), 2) if verified_skills else 0.0
        
        cross_sources = set()
        for v in verified_skills:
            for t in v.evidence_tokens:
                cross_sources.add(t.source_type.value)
        cross_source_support = list(cross_sources)

        return EvidenceLedger(
            candidate_id=candidate.candidate_id,
            verifications=verifications,
            global_contradiction_score=global_contradiction,
            contradictions=contradictions_list,
            strengths=strengths,
            weaknesses=weaknesses,
            verified_skill_score=verified_skill_score,
            verified_skill_confidence=verified_skill_confidence,
            cross_source_support=cross_source_support
        )

    def verify_competency(self, candidate: CandidateProfile, competency: Competency, credibility_score: float = 1.0) -> CompetencyVerification:
        """Cross-checks a competency across independent candidate sources."""
        tokens: List[EvidenceToken] = []
        keywords = competency.semantic_keywords or [competency.name]
        
        # 1. Search Declared Skills
        for skill in candidate.skills:
            if any(keyword_matches(kw, skill.name) for kw in keywords):
                tokens.append(EvidenceToken(
                    source_type=EvidenceSourceType.SKILL_LIST,
                    source_identifier="declared_skills",
                    matched_text=skill.name,
                    confidence=0.8
                ))
                break

        # 2. Search Work Experience Descriptions
        for idx, exp in enumerate(candidate.career_history):
            text = f"{exp.title} {exp.description}".lower()
            for kw in keywords:
                kw_clean = kw.lower().strip()
                search_terms = [re.escape(kw_clean)]
                if kw_clean in SYNONYM_MAP:
                    search_terms.append(re.escape(SYNONYM_MAP[kw_clean]))
                for k, v in SYNONYM_MAP.items():
                    if v == kw_clean:
                        search_terms.append(re.escape(k))
                        
                search_pattern = r"\b(" + "|".join(search_terms) + r")\b"
                if re.search(search_pattern, text):
                    tokens.append(EvidenceToken(
                        source_type=EvidenceSourceType.EXPERIENCE_DESCRIPTION,
                        source_identifier=f"experience_{idx}_{exp.company}",
                        matched_text=f"Matched keyword '{kw}' in role {exp.title} at {exp.company}",
                        confidence=0.9
                    ))
                    break  # Avoid double counting multiple keywords in same role description

        # 3. Search Education field of study/degree
        for idx, edu in enumerate(candidate.education):
            text = f"{edu.degree} {edu.field_of_study} {edu.institution}".lower()
            for kw in keywords:
                kw_clean = kw.lower().strip()
                search_terms = [re.escape(kw_clean)]
                if kw_clean in SYNONYM_MAP:
                    search_terms.append(re.escape(SYNONYM_MAP[kw_clean]))
                for k, v in SYNONYM_MAP.items():
                    if v == kw_clean:
                        search_terms.append(re.escape(k))
                        
                search_pattern = r"\b(" + "|".join(search_terms) + r")\b"
                if re.search(search_pattern, text):
                    tokens.append(EvidenceToken(
                        source_type=EvidenceSourceType.EDUCATION_MAJOR,
                        source_identifier=f"education_{idx}_{edu.institution}",
                        matched_text=f"Matched keyword '{kw}' in education {edu.degree} at {edu.institution}",
                        confidence=0.5
                    ))
                    break

        # 4. Search Projects
        for idx, proj in enumerate(getattr(candidate, "projects", [])):
            text = f"{proj.title} {proj.description} {' '.join(proj.skills_used)}".lower()
            for kw in keywords:
                kw_clean = kw.lower().strip()
                search_terms = [re.escape(kw_clean)]
                if kw_clean in SYNONYM_MAP:
                    search_terms.append(re.escape(SYNONYM_MAP[kw_clean]))
                for k, v in SYNONYM_MAP.items():
                    if v == kw_clean:
                        search_terms.append(re.escape(k))
                search_pattern = r"\b(" + "|".join(search_terms) + r")\b"
                if re.search(search_pattern, text):
                    tokens.append(EvidenceToken(
                        source_type=EvidenceSourceType.PROJECT_DESCRIPTION,
                        source_identifier=f"project_{idx}_{proj.title}",
                        matched_text=f"Matched keyword '{kw}' in project '{proj.title}'",
                        confidence=0.7
                    ))
                    break

        # Evaluate verification status and confidence score
        unique_sources = set(t.source_type for t in tokens)
        
        if not tokens:
            confidence_score = 0.0
            verified = False
            evidence_confidence_level = "Low"
        else:
            base_score = sum(t.confidence for t in tokens) / len(tokens)
            source_multiplier = 0.5 if len(unique_sources) == 1 else (0.85 if len(unique_sources) == 2 else 1.0)
            confidence_score = base_score * source_multiplier
            
            # Apply credibility discounting proportionally
            confidence_score = confidence_score * credibility_score
            verified = confidence_score >= 0.4 and len(unique_sources) >= 1
            
            # Determine evidence confidence levels
            if len(unique_sources) >= 3:
                evidence_confidence_level = "High"
            elif len(unique_sources) == 2 or (len(unique_sources) == 1 and (EvidenceSourceType.EXPERIENCE_DESCRIPTION in unique_sources or EvidenceSourceType.PROJECT_DESCRIPTION in unique_sources)):
                evidence_confidence_level = "Medium"
            else:
                evidence_confidence_level = "Low"

        audit_notes = []
        if len(unique_sources) >= 2:
            audit_notes.append(f"Capability verified across multiple source types.")
        elif len(unique_sources) == 1:
            audit_notes.append(f"Caution: capability only mentioned in one source type: {list(unique_sources)[0].value}.")
        elif not tokens:
            audit_notes.append("No supporting evidence found in candidate profile.")

        return CompetencyVerification(
            competency_name=competency.name,
            verified=verified,
            evidence_tokens=tokens,
            confidence_score=round(confidence_score, 2),
            audit_notes=audit_notes,
            evidence_confidence_level=evidence_confidence_level
        )
