import re
from datetime import datetime, date
from typing import Dict, Any, List

from edhc.app.schemas.candidate import CandidateProfile, WorkExperience
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

import re
from datetime import datetime, date
from typing import Dict, Any, List

from edhc.app.schemas.candidate import CandidateProfile, WorkExperience
from edhc.app.consistency.experience_analyzer import ExperienceConsistencyAnalyzer
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class ConsistencyDetector:
    """Computes a continuous credibility score and detects timeline/skill contradictions in candidate profiles."""

    def analyze(self, candidate: Any) -> Dict[str, Any]:
        """Analyze temporal and textual consistency, returning penalties and a credibility score.

        This replaces binary drops with a continuous credibility score used as a downstream feature.
        """
        logger.info(f"Auditing consistency and credibility for candidate {getattr(candidate, 'candidate_id', getattr(candidate, 'id', 'unknown'))}")
        
        warnings: List[str] = []
        is_honeypot = False
        credibility_penalty = 0.0
        current_date = datetime.now()

        # Handle different schema inputs gracefully
        if hasattr(candidate, "career_history"):
            # CandidateProfile (Hackathon standard)
            career_history = candidate.career_history
            skills_list = candidate.skills
            yoe = candidate.profile.years_of_experience
        else:
            # CandidateProfileNormalized (Pre-processed / Test compatibility)
            # Adapt to match CareerHistoryField structure
            career_history = []
            for exp in getattr(candidate, "experiences", []):
                career_history.append(exp)
            skills_list = getattr(candidate, "skills", [])
            yoe = getattr(getattr(candidate, "profile", None), "years_of_experience", 0.0)

        # 1. Experience Consistency Check
        exp_analyzer = ExperienceConsistencyAnalyzer()
        exp_stats = exp_analyzer.analyze(candidate)
        computed_yoe = exp_stats["computed_years_of_experience"]
        
        if exp_stats["inconsistent"]:
            warnings.append(
                f"Experience mismatch: reported {exp_stats['reported_years_of_experience']} years, "
                f"but chronological history computes to {computed_yoe} years (difference of {exp_stats['experience_difference']} years)."
            )
            credibility_penalty += 0.25  # Small experience mismatch penalty

        # 2. Title Seniority Mismatch Check
        current_title = getattr(candidate.profile, "current_title", "").lower()
        if current_title:
            is_high_seniority = any(w in current_title for w in ["chief", "director", "head", "cto", "vp", "architect", "principal", "staff", "founder", "founding", "lead", "manager"])
            is_med_seniority = any(w in current_title for w in ["senior", "sr"])
            
            if is_high_seniority and computed_yoe < 3.0:
                warnings.append(f"Title seniority mismatch: holding high-seniority title '{current_title}' with only {computed_yoe} computed YOE.")
                credibility_penalty += 0.20
            elif is_med_seniority and computed_yoe < 2.0:
                warnings.append(f"Title seniority mismatch: holding mid-seniority title '{current_title}' with only {computed_yoe} computed YOE.")
                credibility_penalty += 0.15

        # 3. Job Stated Duration vs. Actual Months elapsed & Timeline Chronology Validation
        timeline_consistency_score = 1.0
        
        for exp in career_history:
            start_s = exp.start_date
            end_s = exp.end_date
            
            # Duration can be duration_months (int) or duration_years (float)
            dur_m = getattr(exp, "duration_months", int(getattr(exp, "duration_years", 0.0) * 12))
            company = getattr(exp, "company", getattr(exp, "company_name", "Unknown Company"))
            
            if start_s:
                try:
                    start_dt = datetime.strptime(start_s, "%Y-%m-%d") if isinstance(start_s, str) else datetime.combine(start_s, datetime.min.time())
                    
                    if end_s:
                        end_dt = datetime.strptime(end_s, "%Y-%m-%d") if isinstance(end_s, str) else datetime.combine(end_s, datetime.min.time())
                    else:
                        end_dt = current_date
                        
                    # Check negative duration (Impossible chronology)
                    if start_dt > end_dt:
                        warnings.append(f"Impossible timeline: job at '{company}' has start date ({start_s}) after end date ({end_s or 'Present'}).")
                        credibility_penalty += 0.40
                        timeline_consistency_score -= 0.40
                    
                    # Check future dates
                    if start_dt.date() > date.today() or end_dt.date() > date.today():
                        warnings.append(f"Timeline anomaly: job at '{company}' lists future date (start: {start_s}, end: {end_s}).")
                        credibility_penalty += 0.30
                        timeline_consistency_score -= 0.20
                        
                    actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
                    
                    # If duration_months is way bigger (e.g. 1 year bigger than date range)
                    if dur_m > actual_months + 12:
                        warnings.append(
                            f"Stated duration ({dur_m} months) at job '{company}' "
                            f"significantly exceeds date range limit ({actual_months} months)."
                        )
                        credibility_penalty += 1.0  # Severe timeline inflation
                        is_honeypot = True
                except Exception as e:
                    logger.debug(f"Date parsing failed in duration audit: {e}")

        # 4. Expert proficiency skills with 0 duration
        expert_zero_dur = 0
        for s in skills_list:
            prof = getattr(s, "proficiency", "")
            dur = getattr(s, "duration_months", 1.0) # default positive if not present (e.g. simple list)
            if prof == "expert" and dur == 0:
                expert_zero_dur += 1
                
        if expert_zero_dur >= 3:
            warnings.append(f"Candidate claims 'expert' proficiency in {expert_zero_dur} skills with 0 months of use.")
            credibility_penalty += 1.0  # Severe skill inflation
            is_honeypot = True
        elif expert_zero_dur > 0:
            warnings.append(f"Candidate claims 'expert' proficiency in {expert_zero_dur} skills with 0 months of use.")
            credibility_penalty += 0.4  # Moderate skill mismatch

        # 5. Stated Years of Experience vs. Earliest Job Start Date
        yoe_months = yoe * 12.0
        earliest_dt = None
        for exp in career_history:
            start_s = exp.start_date
            if start_s:
                try:
                    start_dt = datetime.strptime(start_s, "%Y-%m-%d") if isinstance(start_s, str) else datetime.combine(start_s, datetime.min.time())
                    if earliest_dt is None or start_dt < earliest_dt:
                        earliest_dt = start_dt
                except Exception:
                    pass
                    
        if earliest_dt:
            elapsed_months = (current_date.year - earliest_dt.year) * 12 + (current_date.month - earliest_dt.month) + 1
            if yoe_months > elapsed_months + 12:
                warnings.append(
                    f"Stated experience ({yoe} years) exceeds timeline since "
                    f"first job start ({earliest_dt.strftime('%Y-%m-%d')})."
                )
                credibility_penalty += 1.0  # Severe experience timeline mismatch
                is_honeypot = True

        # 6. Stated YOE vs Career History Stated Durations
        total_duration_m = sum(getattr(exp, "duration_months", int(getattr(exp, "duration_years", 0.0) * 12)) for exp in career_history)
        if total_duration_m - yoe_months > 60:  # 5+ years difference
            warnings.append(
                f"Sum of career durations ({total_duration_m} months) significantly "
                f"exceeds stated YOE ({yoe} years)."
            )
            credibility_penalty += 0.5  # Soft inflation warning

        # 7. Expected Salary Range Mismatches (Only if redrob_signals are present)
        if hasattr(candidate, "redrob_signals"):
            sal = candidate.redrob_signals.expected_salary_range_inr_lpa
            if sal.min > sal.max:
                warnings.append(f"Expected salary min ({sal.min}) is greater than max ({sal.max}).")
                credibility_penalty += 0.25  # Minor data anomaly
                is_honeypot = True

        # 8. Honeypot check from helper
        if self._check_honeypots(candidate):
            warnings.append("Honeypot trap skill detected in profile.")
            credibility_penalty += 1.0
            is_honeypot = True

        # 9. Timeline overlaps warning
        normalized_exps = []
        for exp in career_history:
            start = getattr(exp, "start_date", None)
            end = getattr(exp, "end_date", None)
            
            # Convert start and end to date objects
            if isinstance(start, str):
                try:
                    start_val = datetime.strptime(start, "%Y-%m-%d").date()
                except ValueError:
                    start_val = date.today()
            else:
                start_val = start or date.today()
                
            if isinstance(end, str):
                try:
                    end_val = datetime.strptime(end, "%Y-%m-%d").date()
                except ValueError:
                    end_val = None
            else:
                end_val = end
                
            normalized_exps.append(WorkExperience(
                job_title=getattr(exp, "title", getattr(exp, "job_title", "Unknown Role")),
                company_name=getattr(exp, "company", getattr(exp, "company_name", "Unknown Company")),
                start_date=start_val,
                end_date=end_val,
                description=getattr(exp, "description", ""),
                duration_years=getattr(exp, "duration_months", 0.0) / 12.0
            ))
            
        overlaps = self._detect_timeline_overlaps(normalized_exps)
        if overlaps:
            warnings.append(f"Detected timeline overlaps between: {', '.join([f'{p[0]} & {p[1]}' for p in overlaps])}")
            credibility_penalty += 0.2 * len(overlaps)
            timeline_consistency_score -= 0.15 * len(overlaps)

        # 10. Unrealistic tech dates warning
        unrealistic_tech = self._detect_unrealistic_tech_durations(candidate)
        if unrealistic_tech:
            for tech, msg in unrealistic_tech:
                warnings.append(f"Unrealistic date for technology '{tech}': {msg}")
                credibility_penalty += 0.3

        # Clamp scores
        timeline_consistency_score = max(0.0, min(1.0, timeline_consistency_score))
        
        # Calculate a continuous, floor-capped credibility score (range [0.01, 1.0])
        if is_honeypot:
            credibility_score = 0.01
        else:
            credibility_score = max(0.01, 1.0 - (credibility_penalty / 2.0))

        return {
            "credibility_score": round(credibility_score, 4),
            "credibility_penalty": round(credibility_penalty, 2),
            "warnings": warnings,
            "suspicious_profile": credibility_score <= 0.5,
            "computed_years_of_experience": computed_yoe,
            "reported_years_of_experience": exp_stats["reported_years_of_experience"],
            "experience_difference": exp_stats["experience_difference"],
            "experience_consistency_score": exp_stats["experience_consistency_score"],
            "timeline_consistency_score": round(timeline_consistency_score, 2)
        }

    # === HELPER METHODS KEPT FOR TEST COMPATIBILITY ===

    def _detect_timeline_overlaps(self, experiences: List[WorkExperience]) -> List[tuple]:
        """Detect overlapping date periods across candidate's career history."""
        overlaps = []
        valid_exps = [e for e in experiences if e.start_date]
        # Sort by start_date ascending
        sorted_exps = sorted(valid_exps, key=lambda e: e.start_date)
        
        for i in range(len(sorted_exps)):
            for j in range(i + 1, len(sorted_exps)):
                exp1 = sorted_exps[i]
                exp2 = sorted_exps[j]
                
                start1 = exp1.start_date
                end1 = exp1.end_date or date.today()
                start2 = exp2.start_date
                
                # An overlap occurs if start2 is strictly before end1
                # (and start2 is after or equal to start1 by sorting order)
                if start1 <= start2 < end1:
                    # Calculate overlapping days
                    overlap_days = (min(end1, exp2.end_date or date.today()) - start2).days
                    if overlap_days > 30:  # Ignore overlaps shorter than 1 month
                        overlaps.append((
                            f"{exp1.company_name} ({exp1.job_title})",
                            f"{exp2.company_name} ({exp2.job_title})"
                        ))
        return overlaps

    def _detect_unrealistic_tech_durations(self, candidate: Any) -> List[tuple]:
        """Verify candidate does not claim technologies prior to their release dates."""
        unrealistic = []
        # Target technology release years
        release_dates = {
            "fastapi": date(2018, 12, 1),
            "transformers": date(2018, 1, 1),
            "pytorch": date(2016, 9, 1),
            "xgboost": date(2014, 3, 1),
            "lightgbm": date(2017, 1, 1),
            "spacy": date(2015, 1, 1),
            "kubernetes": date(2014, 6, 1),
            "spark": date(2014, 5, 1)
        }
        
        # Check experiences
        experiences = getattr(candidate, "experiences", [])
        if not experiences and hasattr(candidate, "career_history"):
            experiences = candidate.career_history
            
        for exp in experiences:
            desc = getattr(exp, "description", "").lower()
            start = getattr(exp, "start_date", None)
            
            if not start:
                continue
                
            if isinstance(start, str):
                try:
                    start_date_obj = datetime.strptime(start, "%Y-%m-%d").date()
                except ValueError:
                    continue
            else:
                start_date_obj = start
                
            for tech, rel_date in release_dates.items():
                if tech in desc:
                    if start_date_obj < rel_date:
                        unrealistic.append((
                            tech,
                            f"Used in role starting {start_date_obj} (released {rel_date})"
                        ))
        return unrealistic

    def _check_honeypots(self, candidate: Any) -> bool:
        """Scan candidate profile skills for fictitious/honeypot technologies."""
        honeypots = {
            "quantum-cobol", 
            "hyperloop-cobol", 
            "blockchain-fortran", 
            "chatgpt-v1",
            "neuro-pascal",
            "cyber-lisp"
        }
        
        skills = getattr(candidate, "skills", [])
        for skill in skills:
            if isinstance(skill, str):
                name = skill.lower().strip()
            else:
                name = getattr(skill, "name", "").lower().strip()
            
            if name in honeypots:
                return True
        return False
