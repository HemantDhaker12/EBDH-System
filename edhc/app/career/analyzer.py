import re
from typing import List, Dict, Any
from edhc.app.schemas.candidate import CandidateProfile, CareerHistoryField, normalize_candidate
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class CareerAnalyzer:
    """Computes career progression metrics, company ratios, stability, and domain relevance from professional history."""

    def is_services_company(self, company_name: str) -> bool:
        """Identify if a company is a services/consulting firm based on keywords."""
        name = company_name.lower().strip()
        services_keywords = [
            "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", 
            "tech mahindra", "l&t", "larsen & toubro", "lti", "deloitte", "pwc", "kpmg", "ey", "ernst & young", 
            "genpact", "mindtree", "mphasis", "persistent systems", "virtusa", "coforge", "hexaware", "consultancy",
            "services", "solutions"
        ]
        for kw in services_keywords:
            if len(kw) <= 4:
                if re.search(r"\b" + re.escape(kw) + r"\b", name):
                    return True
            else:
                if kw in name:
                    return True
        return False

    def analyze(self, candidate: Any) -> Dict[str, Any]:
        """Run all career analysis routines and return a structured dictionary of scores."""
        candidate = normalize_candidate(candidate)
        cand_id = getattr(candidate, "candidate_id", getattr(candidate, "id", "unknown"))
        logger.info(f"Running career analysis for candidate {cand_id}")
        
        history = candidate.career_history
        if not history:
            return {
                "services_company_ratio": 0.0,
                "product_company_ratio": 1.0,
                "startup_company_ratio": 0.0,
                "enterprise_company_ratio": 0.0,
                "career_stability_score": 1.0,
                "promotion_trajectory_score": 0.5,
                "domain_relevance_score": 0.0,
                "impact_extraction_score": 0.0,
                "total_experience_months": 0
            }

        total_months = sum(exp.duration_months for exp in history)
        
        # 1. Company type ratios
        if total_months == 0:
            services_ratio = 0.0
            product_ratio = 1.0
            startup_ratio = 0.0
            enterprise_ratio = 0.0
        else:
            services_months = sum(exp.duration_months for exp in history if self.is_services_company(exp.company))
            services_ratio = services_months / total_months
            product_ratio = 1.0 - services_ratio
            
            startup_sizes = {"1-10", "11-50", "51-200", "201-500"}
            startup_months = sum(exp.duration_months for exp in history if exp.company_size in startup_sizes)
            startup_ratio = startup_months / total_months
            
            enterprise_sizes = {"5001-10000", "10001+"}
            enterprise_months = sum(exp.duration_months for exp in history if exp.company_size in enterprise_sizes)
            enterprise_ratio = enterprise_months / total_months

        # 2. Stability score (nonlinear avg tenure & short job penalties)
        num_jobs = len(history)
        avg_years = (total_months / num_jobs) / 12.0 if num_jobs > 0 else 0.0
        
        if avg_years >= 3.0:
            stability = 1.0
        elif avg_years >= 1.5:
            stability = 0.6 + (avg_years - 1.5) * (0.4 / 1.5)
        elif avg_years >= 0.8:
            stability = 0.2 + (avg_years - 0.8) * (0.4 / 0.7)
        else:
            stability = 0.1
            
        short_jobs = sum(1 for exp in history if not exp.is_current and exp.duration_months < 12)
        stability = max(0.05, stability - (short_jobs * 0.1))

        # 3. Promotion trajectory
        sorted_history = sorted(history, key=lambda x: x.start_date or "")
        seniority_levels = []
        for exp in sorted_history:
            t = exp.title.lower()
            if any(w in t for w in ["chief", "director", "head", "cto", "vp", "architect", "principal", "staff", "founder"]):
                seniority_levels.append(3.0)
            elif any(w in t for w in ["senior", "sr", "lead", "manager"]):
                seniority_levels.append(2.0)
            elif any(w in t for w in ["junior", "jr", "associate", "intern"]):
                seniority_levels.append(0.5)
            else:
                seniority_levels.append(1.0)

        if len(seniority_levels) <= 1:
            trajectory = 0.5
        else:
            increases = sum(1 for i in range(1, len(seniority_levels)) if seniority_levels[i] > seniority_levels[i-1])
            decreases = sum(1 for i in range(1, len(seniority_levels)) if seniority_levels[i] < seniority_levels[i-1])
            trajectory = 0.5 + (increases - decreases) * 0.25
            trajectory = max(0.0, min(1.0, trajectory))

        # 4. Domain relevance score
        text_corpus = f"{candidate.profile.headline} {candidate.profile.summary} {candidate.profile.current_title} {candidate.profile.current_industry}"
        for exp in history:
            text_corpus += f" {exp.title} {exp.description}"
        text_lower = text_corpus.lower()
        
        tier1 = ["search ranking", "learning to rank", "lambdamart", "bm25", "elasticsearch", "opensearch", "vector search", "hybrid search", "information retrieval", "dense retrieval", "reranking", "rerank", "vector database", "pinecone", "weaviate", "qdrant", "milvus", "faiss", "recommender", "recommendation"]
        tier2 = ["machine learning", "deep learning", "nlp", "natural language processing", "llm", "large language model", "pytorch", "tensorflow", "scikit-learn", "xgboost", "lightgbm", "transformers", "fine-tuning", "lora", "qlora", "peft", "rag", "embeddings", "neural network"]
        tier3 = ["python", "fastapi", "django", "flask", "system design", "microservices", "kubernetes", "docker", "aws", "gcp", "sql", "postgresql", "data engineering", "spark", "polars", "pandas"]
        
        tier1_matches = sum(1 for kw in tier1 if kw in text_lower)
        tier2_matches = sum(1 for kw in tier2 if kw in text_lower)
        tier3_matches = sum(1 for kw in tier3 if kw in text_lower)
        
        raw_relevance = (tier1_matches * 1.5) + (tier2_matches * 0.8) + (tier3_matches * 0.2)
        domain_relevance = raw_relevance / (raw_relevance + 5.0) if raw_relevance > 0 else 0.0

        # Title-based soft signals (boost for AI/ML history, discount for irrelevant titles like HR/Design/QA)
        current_title = candidate.profile.current_title.lower()
        irrelevant_keywords = [
            "designer", "graphic", "ui/ux", "illustrator", "photoshop", "creative",
            "recruiter", "hr", "talent acquisition", "human resources",
            "sales", "marketing", "seo", "business development", "account executive",
            "accountant", "audit", "finance",
            "mechanical", "civil", "electrical", "cad", "structural",
            "qa", "testing", "manual tester", "quality assurance", "test engineer"
        ]
        is_irrelevant_title = any(kw in current_title for kw in irrelevant_keywords)
        
        ai_keywords = ["ml", "machine learning", "search", "ranking", "retrieval", "ai", "nlp", "llm", "recommend"]
        has_ai_title = any(any(kw in exp.title.lower() for kw in ai_keywords) for exp in history)
        
        if has_ai_title:
            domain_relevance = min(1.0, domain_relevance + 0.15)
        if is_irrelevant_title:
            domain_relevance = domain_relevance * 0.1

        # 5. Impact extraction score
        desc_text = " ".join([exp.description for exp in history]) + " " + candidate.profile.summary
        desc_lower = desc_text.lower()
        
        patterns = [
            r"\b\d+(?:\.\d+)?%",
            r"\$\d+(?:\.\d+)?\s*(?:k|m|million|billion)?\b",
            r"\b\d+(?:\.\d+)?\s*(?:ms|sec|seconds|hrs|hours|days|weeks|months|years|x|fold)\b",
            r"\b(?:ndcg|mrr|map|auc|f1-score|f1 score|accuracy|precision|recall|latency|throughput|qps|tps|roi)\b",
            r"\b\d+\+?\s*(?:k|m|lakhs?|cr|crores?|million|billion|users|queries|requests|records|gb|tb|pb)\b"
        ]
        
        matches_count = sum(len(re.findall(pat, desc_lower)) for pat in patterns)
        impact_extraction = min(1.0, matches_count / 10.0) if matches_count > 0 else 0.0

        return {
            "services_company_ratio": round(services_ratio, 4),
            "product_company_ratio": round(product_ratio, 4),
            "startup_company_ratio": round(startup_ratio, 4),
            "enterprise_company_ratio": round(enterprise_ratio, 4),
            "career_stability_score": round(stability, 4),
            "promotion_trajectory_score": round(trajectory, 4),
            "domain_relevance_score": round(domain_relevance, 4),
            "impact_extraction_score": round(impact_extraction, 4),
            "total_experience_months": total_months
        }

