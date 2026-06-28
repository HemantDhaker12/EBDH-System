import re
import hashlib
from typing import List, Optional
from edhc.app.schemas.jd import JobDescriptionParsed, HiringRubric, Competency
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class JobDescriptionParser:
    """Parses raw text of Job Descriptions to extract competency frameworks and hiring rubrics."""

    def parse(self, text: str, title: str = "Target Position") -> JobDescriptionParsed:
        """Parse raw job description text into structured rubric and schema."""
        logger.info(f"Parsing job description: {title}")
        
        # Unique identifier for the Job Description
        job_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        
        competencies = self._extract_competencies(text)
        min_exp = self._determine_experience_years(text)
        education = self._determine_education(text)
        industries = self._determine_target_industries(text)
        
        rubric = HiringRubric(
            competencies=competencies,
            min_experience_years=min_exp,
            preferred_education=education,
            target_industries=industries
        )

        return JobDescriptionParsed(
            id=job_id,
            title=title,
            raw_text=text,
            rubric=rubric,
            metadata={"parsed_version": "1.1"}
        )

    def generate_retrieval_query(self, jd: JobDescriptionParsed) -> str:
        """Dynamically construct search query from Job Description competencies, title, and keywords."""
        query_parts = [jd.title]
        for comp in jd.rubric.competencies:
            query_parts.append(comp.name)
            query_parts.extend(comp.semantic_keywords)
            
        # Also extract any specific capitalised tech terms from raw text as fallback (e.g. PyTorch, FAISS)
        tech_pattern = re.compile(r"\b(PyTorch|TensorFlow|LightGBM|XGBoost|FastAPI|Pinecone|Qdrant|Milvus|FAISS|Elasticsearch|Airflow|Spark|dbt|Snowflake|Docker|Kubernetes|REST)\b", re.IGNORECASE)
        matches = tech_pattern.findall(jd.raw_text)
        query_parts.extend(matches)

        # Deduplicate terms while preserving logical flow
        seen = set()
        clean_query = []
        for word in " ".join(query_parts).split():
            # Clean symbols except for special tech characters (+, #, -)
            clean_word = re.sub(r"[^\w\+\-#]", "", word.lower().strip())
            if clean_word and clean_word not in seen:
                seen.add(clean_word)
                clean_query.append(clean_word)
                
        result_query = " ".join(clean_query)
        logger.info(f"Dynamically generated retrieval query: '{result_query}'")
        return result_query

    def _extract_competencies(self, text: str) -> List[Competency]:
        """Extract key competencies from the JD using pattern matching."""
        competency_catalog = {
            "Python Programming": ["python", "pydata", "fastapi", "django", "flask"],
            "Machine Learning / Deep Learning": ["machine learning", "ml", "deep learning", "neural networks", "scikit-learn", "xgboost", "lightgbm", "pytorch", "tensorflow"],
            "Search / Information Retrieval": ["bm25", "elasticsearch", "vector database", "dense retrieval", "lucene", "solr", "hybrid search", "information retrieval", "milvus", "qdrant", "faiss"],
            "System Architecture": ["system design", "microservices", "kubernetes", "docker", "aws", "gcp", "scalability", "distributed systems"],
            "Data Engineering & SQL": ["sql", "etl", "postgresql", "spark", "polars", "pandas", "snowflake", "bigquery"],
            "Leadership & Mentorship": ["lead", "mentor", "scrum", "manage", "architect", "vision", "roadmap", "agile"]
        }

        extracted: List[Competency] = []
        text_lower = text.lower()

        for name, terms in competency_catalog.items():
            # Check if any key term matches the description text
            match_count = 0
            for term in terms:
                if re.search(r"\b" + re.escape(term) + r"\b", text_lower):
                    match_count += 1
            
            if match_count > 0:
                # Calculate simple weight based on frequency or importance keyword
                weight = 1.0 if match_count > 1 else 0.5
                
                extracted.append(Competency(
                    name=name,
                    description=f"Extracted because matched keywords: {[t for t in terms if t in text_lower]}",
                    weight=weight,
                    semantic_keywords=terms
                ))

        # Ensure we always have at least some basic competencies if none matched
        if not extracted:
            extracted.append(Competency(
                name="General Software Development",
                description="Core software engineering skillset",
                weight=1.0,
                semantic_keywords=["software", "programming", "developer", "git"]
            ))

        return extracted

    def _determine_experience_years(self, text: str) -> float:
        """Estimate minimum years of experience from job text."""
        match = re.search(r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience", text.lower())
        if match:
            return float(match.group(1))
            
        range_match = re.search(r"(\d+)\s*-\s*(\d+)\s*years?", text.lower())
        if range_match:
            return float(range_match.group(1))
            
        return 0.0

    def _determine_education(self, text: str) -> Optional[str]:
        """Parse preferred educational levels."""
        t = text.lower()
        if "phd" in t or "ph.d" in t:
            return "PhD"
        elif "master" in t or "ms" in t or "m.s." in t:
            return "MS"
        elif "bachelor" in t or "bs" in t or "b.s." in t:
            return "BS"
        return None

    def _determine_target_industries(self, text: str) -> List[str]:
        """Detect target industries or domains mentioned in the JD."""
        industries_patterns = {
            "Finance": ["finance", "fintech", "banking", "trading"],
            "Healthcare": ["healthcare", "biotech", "medical", "clinical"],
            "E-commerce": ["retail", "ecommerce", "e-commerce", "marketplace"],
            "SaaS": ["saas", "software as a service", "b2b", "enterprise"]
        }
        
        t = text.lower()
        matches = []
        for name, terms in industries_patterns.items():
            if any(term in t for term in terms):
                matches.append(name)
        return matches
