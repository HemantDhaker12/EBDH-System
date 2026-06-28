from edhc.app.jd.intelligence import JobDescriptionParser

def test_parse_jd_competencies():
    parser = JobDescriptionParser()
    text = """
    We are looking for a Senior Developer with 5+ years of experience.
    Must have excellent skills in Python programming and Machine Learning models (specifically PyTorch).
    Familiarity with Search engines and information retrieval models is a plus.
    Target industries: Fintech and E-commerce.
    """
    
    parsed = parser.parse(text, "Senior Machine Learning Developer")
    
    assert parsed.title == "Senior Machine Learning Developer"
    assert parsed.rubric.min_experience_years == 5.0
    
    # Check that competencies are successfully parsed
    competency_names = [c.name for c in parsed.rubric.competencies]
    assert "Python Programming" in competency_names
    assert "Machine Learning / Deep Learning" in competency_names
    assert "Search / Information Retrieval" in competency_names
    
    # Industries checks
    assert "Finance" in parsed.rubric.target_industries
    assert "E-commerce" in parsed.rubric.target_industries

def test_parse_jd_experience_years():
    parser = JobDescriptionParser()
    # Test different text styles
    assert parser._determine_experience_years("Requires 3+ years experience") == 3.0
    assert parser._determine_experience_years("Looking for someone with 2-4 years of industry tenure") == 2.0
    assert parser._determine_experience_years("No experience required") == 0.0

def test_parse_jd_education():
    parser = JobDescriptionParser()
    assert parser._determine_education("Requires a PhD in Computer Science") == "PhD"
    assert parser._determine_education("MS degree preferred") == "MS"
    assert parser._determine_education("Must have Bachelor's or BS degree") == "BS"
