import re
from datetime import datetime, date
from typing import Optional, Dict, Any

from edhc.app.schemas.candidate import CandidateProfile, normalize_candidate
from edhc.app.schemas.jd import JobDescriptionParsed
from edhc.app.schemas.features import CandidateFeatures

from edhc.app.career.analyzer import CareerAnalyzer
from edhc.app.skills.validator import SkillsValidator
from edhc.app.behavior.analyzer import BehavioralAnalyzer
from edhc.app.evidence.verifier import EvidenceVerifier
from edhc.app.consistency.detector import ConsistencyDetector
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureGenerator:
    """Aggregates scores from all specialized analyzers into structured candidate feature vectors.

    === REDROB SIGNALS USAGE AUDIT ===
    1. notice_period_days -> Used (notice_period_score, notice_period_normalized, ScoreCalibrator).
    2. github_activity_score -> Used (BehavioralAnalyzer modifiers and activity_score).
    3. recruiter_response_rate -> Used (BehavioralAnalyzer engagement_score).
    4. interview_completion_rate -> Used (BehavioralAnalyzer engagement_score).
    5. profile_completeness_score -> Used (BehavioralAnalyzer activity_score).
    6. expected_salary_range_inr_lpa -> Used (ConsistencyDetector expected_min > expected_max anomaly).
    7. willing_to_relocate -> Used (BehavioralAnalyzer and relocation_willingness features).
    8. open_to_work_flag -> Used (BehavioralAnalyzer activity_score and availability_indicator).
    9. signup_date -> Ignored. (Reason: Platform-specific timeline value; not correlated with competence/skills).
    10. last_active_date -> Ignored. (Reason: Acts as operational outreach filter, not a technical suitability signal).
    11. profile_views_received_30d -> Ignored. (Reason: Organic views contain high visibility/popularity feedback loops, introducing bias).
    12. applications_submitted_30d -> Ignored. (Reason: Job seeker volume rate does not indicate technical capacity or relevance).
    13. avg_response_time_hours -> Ignored. (Reason: High volatility and timezone variations, operational noise).
    14. skill_assessment_scores -> Ignored. (Reason: Handled via specialized NLP skills verification and project corroborations in SkillsValidator).
    15. connection_count -> Ignored. (Reason: Professional network sizes are uncorrelated with specialized backend/search expertise).
    16. endorsements_received -> Ignored. (Reason: Individual skill endorsements in the skills lists are utilized directly).
    17. preferred_work_mode -> Ignored. (Reason: Operational recruitment filter, does not measure candidate fit quality).
    18. search_appearance_30d -> Ignored. (Reason: Highly skewed by platform search visibility algorithm anomalies).
    19. saved_by_recruiters_30d -> Ignored. (Reason: Popularity feedback metric with zero technical match relevance).
    20. offer_acceptance_rate -> Ignored. (Reason: Operational recruitment conversion rate, uncorrelated with target rubric).
    21. verified_email / verified_phone / linkedin_connected -> Ignored. (Reason: Security and onboarding verification parameters).
    ==================================
    """

    def __init__(self) -> None:
        self.career_analyzer = CareerAnalyzer()
        self.skills_validator = SkillsValidator()
        self.behavior_analyzer = BehavioralAnalyzer()
        self.evidence_verifier = EvidenceVerifier()
        self.consistency_detector = ConsistencyDetector()

    def generate(
        self,
        candidate: Any,
        jd: JobDescriptionParsed,
        semantic_retrieval_score: float = 0.0,
        retrieval_diag: Optional[Dict[str, Any]] = None
    ) -> CandidateFeatures:
        """Execute all analyzer modules and assemble the full numerical features profile."""
        candidate = normalize_candidate(candidate)
        cand_id = getattr(candidate, "candidate_id", getattr(candidate, "id", "unknown"))
        logger.info(f"Extracting features for candidate {cand_id} against Job {jd.id}")
        
        # 1. Career Analysis
        career_stats = self.career_analyzer.analyze(candidate)
        
        # 2. Behavioral scoring
        behavior_stats = self.behavior_analyzer.analyze(candidate)
        
        # 3. Consistency checks
        consistency_stats = self.consistency_detector.analyze(candidate)
        
        # 4. Evidence ledgers (discounted by credibility score)
        cred_score = consistency_stats["credibility_score"]
        ledger = self.evidence_verifier.verify(candidate, jd.rubric, credibility_score=cred_score)
        
        # Global confidence is average confidence score of all competencies in ledger
        verifications_list = list(ledger.verifications.values())
        if verifications_list:
            avg_confidence = sum(v.confidence_score for v in verifications_list) / len(verifications_list)
        else:
            avg_confidence = 0.0

        # === FEATURE COMPUTATION LOBBY ===

        # 13. Lexical retrieval score (BM25)
        bm25_score = retrieval_diag.get("bm25_score", 0.0) if retrieval_diag else 0.0
        
        # 14. Dense retrieval semantic similarity
        dense_similarity = retrieval_diag.get("dense_similarity", 0.0) if retrieval_diag else 0.0
        
        # 15. Retrieval rank percentile (scaled relative to rank in RRF pool)
        rrf_rank = retrieval_diag.get("rrf_rank", 1.0) if retrieval_diag else 1.0
        # Assuming typical pre-filter retrieval pool size of 2000
        retrieval_rank_percentile = max(0.0, 1.0 - (rrf_rank / 2000.0))

        # 16. JD title Jaccard word similarity with candidate titles
        jd_title_words = set(re.findall(r"\w+", jd.title.lower()))
        cand_titles = [candidate.profile.current_title] + [exp.title for exp in candidate.career_history]
        cand_title_words = set(re.findall(r"\w+", " ".join(cand_titles).lower()))
        jd_title_similarity = len(jd_title_words.intersection(cand_title_words)) / len(jd_title_words) if jd_title_words else 0.0

        # 17. JD summary/requirements word similarity with candidate summary
        jd_words = set()
        for comp in jd.rubric.competencies:
            jd_words.update(re.findall(r"\w+", comp.description.lower()))
            if comp.semantic_keywords:
                jd_words.update([kw.lower().strip() for kw in comp.semantic_keywords])
        cand_sum_words = set(re.findall(r"\w+", candidate.profile.summary.lower()))
        jd_summary_similarity = min(1.0, (len(jd_words.intersection(cand_sum_words)) / len(jd_words)) * 5.0) if jd_words else 0.0

        # 18. Competency coverage (ratio of verified hiring competencies)
        num_comps = len(ledger.verifications)
        num_verified = sum(1 for v in ledger.verifications.values() if v.verified)
        competency_coverage = num_verified / num_comps if num_comps > 0 else 0.0

        # 19. Skill overlap ratio
        target_skills = []
        for comp in jd.rubric.competencies:
            target_skills.append(comp.name)
            if comp.semantic_keywords:
                target_skills.extend(comp.semantic_keywords)
        skills_results = self.skills_validator.validate(candidate, target_skills)
        matched_skills = sum(1 for res in skills_results.values() if res["confidence_score"] > 0.0)
        skill_overlap_ratio = matched_skills / len(target_skills) if target_skills else 0.0

        # 20. Stated relevant experience years from career history
        total_relevant_months = 0.0
        for exp in candidate.career_history:
            text = f"{exp.title} {exp.description}".lower()
            # Flag matching ML/Search domains
            is_rel = any(kw in text for kw in ["machine learning", "ml", "search", "ranking", "retrieval", "dense", "vector", "nlp", "llm", "deep learning", "ai"])
            if is_rel:
                total_relevant_months += exp.duration_months
        relevant_experience_years = total_relevant_months / 12.0

        # 21. Average career history job tenure in years
        num_jobs = len(candidate.career_history)
        total_months = sum(exp.duration_months for exp in candidate.career_history)
        average_tenure_years = (total_months / num_jobs) / 12.0 if num_jobs > 0 else 0.0

        # 22. Career continuity score (penalizes chronological gaps)
        gaps_months = 0.0
        sorted_history = sorted([exp for exp in candidate.career_history if exp.start_date], key=lambda x: x.start_date)
        for i in range(len(sorted_history) - 1):
            exp1 = sorted_history[i]
            exp2 = sorted_history[i+1]
            if exp1.end_date and exp2.start_date:
                try:
                    end1 = datetime.strptime(exp1.end_date, "%Y-%m-%d")
                    start2 = datetime.strptime(exp2.start_date, "%Y-%m-%d")
                    if start2 > end1:
                        gap = (start2.year - end1.year) * 12 + (start2.month - end1.month)
                        if gap > 3:  # ignore gaps shorter than 3 months
                            gaps_months += gap
                except Exception:
                    pass
        career_continuity_score = max(0.0, 1.0 - (gaps_months / 24.0))

        # 23. Promotion velocity (frequency of senior roles over total tenure)
        senior_roles = sum(1 for exp in candidate.career_history if any(w in exp.title.lower() for w in ["chief", "director", "head", "cto", "vp", "architect", "principal", "staff", "founder", "senior", "sr", "lead", "manager"]))
        total_years = total_months / 12.0
        promotion_velocity = senior_roles / total_years if total_years > 0.0 else 0.0

        # 24. Leadership indicator score
        leadership_score = 0.0
        for exp in candidate.career_history:
            t = exp.title.lower()
            if any(w in t for w in ["chief", "director", "head", "cto", "vp", "founder"]):
                leadership_score = max(leadership_score, 1.0)
            elif any(w in t for w in ["lead", "manager", "architect", "principal", "staff"]):
                leadership_score = max(leadership_score, 0.6)
            elif "senior" in t or "sr" in t:
                leadership_score = max(leadership_score, 0.3)
        leadership_indicator_score = leadership_score

        # 25. Technical specialization trend (increasing relevance over time)
        technical_specialization_trend = 0.5
        if len(sorted_history) >= 2:
            try:
                latest_exp = sorted_history[-1]
                earliest_exp = sorted_history[0]
                latest_text = f"{latest_exp.title} {latest_exp.description}".lower()
                earliest_text = f"{earliest_exp.title} {earliest_exp.description}".lower()
                
                ml_keywords = ["machine learning", "ml", "search", "ranking", "retrieval", "nlp", "llm", "ai"]
                latest_matches = sum(1 for kw in ml_keywords if kw in latest_text)
                earliest_matches = sum(1 for kw in ml_keywords if kw in earliest_text)
                
                technical_specialization_trend = 0.5 + (latest_matches - earliest_matches) * 0.15
                technical_specialization_trend = max(0.0, min(1.0, technical_specialization_trend))
            except Exception:
                pass

        # 26. Number of verified skills
        candidate_skills = [s.name for s in candidate.skills]
        skills_audit = self.skills_validator.validate(candidate, candidate_skills)
        num_verified_skills = float(sum(1 for audit in skills_audit.values() if len(audit["sources_found"]) >= 2))

        # 27. Cross-source verification count
        cross_source_verification_count = float(sum(1 for audit in skills_audit.values() if len(audit["sources_found"]) >= 2))

        # 28. Skill diversity score across domains
        categories = {
            "ml_nlp": ["pytorch", "tensorflow", "scikit-learn", "xgboost", "lightgbm", "transformers", "fine-tuning", "rag", "embeddings", "nlp", "large language model", "llm", "deep learning", "machine learning", "ai"],
            "backend_cloud": ["python", "fastapi", "django", "flask", "system design", "microservices", "kubernetes", "docker", "aws", "gcp", "sql", "postgresql"],
            "data_eng": ["spark", "airflow", "snowflake", "dbt", "kafka", "hadoop", "etl", "data engineering", "polars", "pandas"],
            "search_ir": ["search", "ranking", "learning to rank", "lambdamart", "bm25", "elasticsearch", "opensearch", "vector search", "hybrid search", "information retrieval", "dense retrieval", "reranking", "vector database", "pinecone", "weaviate", "qdrant", "milvus", "faiss"]
        }
        matched_cats = set()
        for skill_name in skills_audit:
            skill_clean = skill_name.lower().strip()
            for cat, keywords in categories.items():
                if any(kw in skill_clean for kw in keywords):
                    matched_cats.add(cat)
                    break
        skill_diversity_score = len(matched_cats) / len(categories) if categories else 0.0

        # 29. Skill recency score (verified skills in current/latest role)
        recency_weight = 0.0
        if candidate.career_history:
            latest_job = candidate.career_history[0]
            current_jobs = [exp for exp in candidate.career_history if exp.is_current]
            if current_jobs:
                latest_job = current_jobs[0]
            latest_desc = latest_job.description.lower()
            matches = sum(1 for skill_name in skills_audit if skill_name.lower() in latest_desc)
            recency_weight = matches / len(skills_audit) if skills_audit else 0.0
        skill_recency_score = recency_weight

        # 30. Skill longevity score (months of experience across verified skills)
        total_months_sum = sum(audit["relevant_experience_years"] * 12.0 for audit in skills_audit.values())
        skill_longevity_score = min(1.0, total_months_sum / 240.0)

        # 31. Notice period normalized
        days = candidate.redrob_signals.notice_period_days
        notice_period_normalized = max(0.0, 1.0 - (days / 90.0))

        # 32. Relocation willingness
        relocation_willingness = 1.0 if candidate.redrob_signals.willing_to_relocate else 0.0

        # 33. Availability platform engagement indicator
        availability_indicator = 1.0 if candidate.redrob_signals.open_to_work_flag else 0.0

        # 34. Timeline consistency (reverse of timeline credibility penalty)
        timeline_consistency = max(0.01, 1.0 - min(0.9, (consistency_stats["credibility_penalty"] / 2.0)))

        # 35. Contradiction count
        contradiction_count = float(len(consistency_stats["warnings"]))

        # 36. Profile completeness score
        profile_completeness = candidate.redrob_signals.profile_completeness_score / 100.0

        # 37. Cross-field agreement
        has_yoe_warning = any("exceeds timeline" in w or "exceeds stated YOE" in w for w in consistency_stats["warnings"])
        cross_field_agreement = 0.5 if has_yoe_warning else 1.0

        # 38. Quantitative achievements count in descriptions
        patterns = [
            r"\b\d+(?:\.\d+)?%",
            r"\$\d+(?:\.\d+)?\s*(?:k|m|million|billion)?\b",
            r"\b\d+(?:\.\d+)?\s*(?:ms|sec|seconds|hrs|hours|days|weeks|months|years|x|fold)\b",
            r"\b\d+\+?\s*(?:k|m|lakhs?|cr|crores?|million|billion|users|queries|requests|records|gb|tb|pb)\b"
        ]
        desc_text = " ".join([exp.description for exp in candidate.career_history]) + " " + candidate.profile.summary
        desc_lower = desc_text.lower()
        quantitative_achievement_count = float(sum(len(re.findall(pat, desc_lower)) for pat in patterns))

        # 39. Performance metric mentions density
        metric_keywords = ["ndcg", "mrr", "map", "auc", "f1-score", "f1 score", "accuracy", "precision", "recall", "latency", "throughput", "qps", "tps"]
        performance_metric_mentions = float(sum(1 for kw in metric_keywords if kw in desc_lower))

        # Construct and validate Features object
        cand_id = getattr(candidate, "candidate_id", getattr(candidate, "id", "unknown"))
        return CandidateFeatures(
            candidate_id=cand_id,
            services_company_ratio=career_stats["services_company_ratio"],
            product_company_ratio=career_stats["product_company_ratio"],
            startup_company_ratio=career_stats["startup_company_ratio"],
            enterprise_company_ratio=career_stats["enterprise_company_ratio"],
            career_stability_score=career_stats["career_stability_score"],
            notice_period_score=behavior_stats["notice_period_score"],
            domain_relevance_score=career_stats["domain_relevance_score"],
            promotion_trajectory_score=career_stats["promotion_trajectory_score"],
            evidence_strength=round(avg_confidence, 4),
            impact_extraction_score=career_stats["impact_extraction_score"],
            credibility_score=consistency_stats["credibility_score"],
            rrf_retrieval_score=semantic_retrieval_score,
            
            # New retrieval
            bm25_retrieval_score=round(bm25_score, 4),
            dense_retrieval_score=round(dense_similarity, 4),
            retrieval_rank_percentile=round(retrieval_rank_percentile, 4),
            
            # New semantic
            jd_title_similarity=round(jd_title_similarity, 4),
            jd_summary_similarity=round(jd_summary_similarity, 4),
            competency_coverage=round(competency_coverage, 4),
            skill_overlap_ratio=round(skill_overlap_ratio, 4),
            
            # New career
            relevant_experience_years=round(relevant_experience_years, 4),
            average_tenure_years=round(average_tenure_years, 4),
            career_continuity_score=round(career_continuity_score, 4),
            promotion_velocity=round(promotion_velocity, 4),
            leadership_indicator_score=round(leadership_indicator_score, 4),
            technical_specialization_trend=round(technical_specialization_trend, 4),
            
            # New skills
            num_verified_skills=round(num_verified_skills, 4),
            cross_source_verification_count=round(cross_source_verification_count, 4),
            skill_diversity_score=round(skill_diversity_score, 4),
            skill_recency_score=round(skill_recency_score, 4),
            skill_longevity_score=round(skill_longevity_score, 4),
            
            # New behavioral
            notice_period_normalized=round(notice_period_normalized, 4),
            relocation_willingness=round(relocation_willingness, 4),
            availability_indicator=round(availability_indicator, 4),
            
            # New credibility
            timeline_consistency=round(timeline_consistency, 4),
            contradiction_count=round(contradiction_count, 4),
            profile_completeness=round(profile_completeness, 4),
            cross_field_agreement=round(cross_field_agreement, 4),
            
            # New impact
            quantitative_achievement_count=round(quantitative_achievement_count, 4),
            performance_metric_mentions=round(performance_metric_mentions, 4)
        )
