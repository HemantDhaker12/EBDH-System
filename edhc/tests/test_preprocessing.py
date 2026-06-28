from datetime import date
from edhc.app.schemas.candidate import CandidateProfileRaw
from edhc.app.preprocessing.normalizer import CandidateNormalizer

def test_clean_text():
    normalizer = CandidateNormalizer()
    assert normalizer.clean_text("   Hello   World!   ") == "Hello World!"
    assert normalizer.clean_text("<p>Hello</p> <b>World</b>") == "Hello World"

def test_parse_date():
    normalizer = CandidateNormalizer()
    # Test valid dates
    assert normalizer.parse_date("2020-05-12") == date(2020, 5, 12)
    assert normalizer.parse_date("2021-06") == date(2021, 6, 1)
    
    # Test year fallback
    assert normalizer.parse_date("Graduated in 2018") == date(2018, 1, 1)
    
    # Test present fallback (returns date.today() via experience handler, but direct parse defaults)
    assert isinstance(normalizer.parse_date("invalid-date-string"), date)

def test_standardize_title():
    normalizer = CandidateNormalizer()
    assert normalizer.standardize_title("Senior Machine Learning Engineer") == "Senior Machine Learning Engineer"
    assert normalizer.standardize_title("lead ml engineer") == "Senior Machine Learning Engineer"
    assert normalizer.standardize_title("jr. developer") == "Junior Software Engineer"
    assert normalizer.standardize_title("backend developer") == "Backend Engineer"

def test_company_tier():
    normalizer = CandidateNormalizer()
    assert normalizer.estimate_company_tier("Google LLC") == 1
    assert normalizer.estimate_company_tier("Spotify Inc") == 2
    assert normalizer.estimate_company_tier("Unknown local shop") == 3

def test_normalize_profile():
    normalizer = CandidateNormalizer()
    raw = CandidateProfileRaw(
        name="John Doe",
        email="john.doe@test.com",
        summary="Experienced Python engineer",
        skills=["Python", "FastAPI"],
        raw_experience=[
            {
                "job_title": "Backend Python Developer",
                "company_name": "Meta",
                "start_date": "2020-01-01",
                "end_date": "Present",
                "description": "Building scaling web APIs in FastAPI and Python."
            }
        ]
    )
    
    norm = normalizer.normalize(raw)
    assert norm.name == "John Doe"
    assert len(norm.experiences) == 1
    assert norm.experiences[0].company_tier == 1
    assert norm.experiences[0].is_current is True
    assert norm.experiences[0].normalized_title == "Backend Engineer"
    assert norm.experiences[0].duration_years > 0.0
