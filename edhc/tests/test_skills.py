from datetime import date
from edhc.app.schemas.candidate import CandidateProfileNormalized, WorkExperience, Project
from edhc.app.skills.validator import SkillsValidator

def test_skills_validation():
    validator = SkillsValidator()
    
    # Setup candidate with python and search skills
    candidate = CandidateProfileNormalized(
        id="test-id",
        name="Candidate A",
        summary="A professional developer",
        skills=["Python", "FastAPI"],
        experiences=[
            WorkExperience(
                job_title="Software Developer",
                company_name="Company A",
                start_date=date(2021, 1, 1),
                end_date=date(2023, 1, 1), # 2.0 years (24 months)
                is_current=False,
                description="Developing REST API endpoints using Python and FastAPI.",
                duration_years=2.0
            )
        ],
        projects=[
            Project(
                title="Search API",
                description="Built a search index query using Python and ElasticSearch.",
                skills_used=["Python", "ElasticSearch"]
            )
        ],
        education=[]
    )

    results = validator.validate(candidate, ["Python", "FastAPI", "Docker"])
    
    # Python is declared, in experience, and in projects. High confidence.
    python_val = results["Python"]
    assert python_val["confidence_score"] > 0.6
    assert "declared_skills" in python_val["sources_found"]
    assert "work_experience" in python_val["sources_found"]
    assert "projects" in python_val["sources_found"]
    assert python_val["relevant_experience_years"] == 2.0
    
    # Docker is not in candidate profile. Zero/low confidence.
    docker_val = results["Docker"]
    assert docker_val["confidence_score"] == 0.0
    assert len(docker_val["sources_found"]) == 0
