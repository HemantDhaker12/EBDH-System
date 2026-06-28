import re
from datetime import date, datetime
from typing import List, Dict, Any, Tuple
from edhc.app.schemas.candidate import CandidateProfile, normalize_candidate

class ExperienceConsistencyAnalyzer:
    """Computes candidate experience chronologically, handles career gaps/overlaps,
    and checks consistency against the reported profile years of experience.
    """

    def analyze(self, candidate: Any) -> Dict[str, Any]:
        candidate = normalize_candidate(candidate)
        history = candidate.career_history
        reported_yoe = float(candidate.profile.years_of_experience)
        
        intervals: List[Tuple[date, date]] = []
        current_date = date.today()
        
        for exp in history:
            start_s = exp.start_date
            end_s = exp.end_date
            is_curr = exp.is_current
            
            if not start_s:
                continue
                
            # Parse start date
            try:
                if isinstance(start_s, str):
                    start_val = datetime.strptime(start_s, "%Y-%m-%d").date()
                else:
                    start_val = start_s
            except Exception:
                continue
                
            # Parse end date
            if is_curr or not end_s or str(end_s).lower().strip() in ["present", "current", "now", "ongoing"]:
                end_val = current_date
            else:
                try:
                    if isinstance(end_s, str):
                        end_val = datetime.strptime(end_s, "%Y-%m-%d").date()
                    else:
                        end_val = end_s
                except Exception:
                    end_val = current_date
                    
            if start_val > end_val:
                # Duration mismatch (chronological ordering error), handle gracefully by swapping
                start_val, end_val = end_val, start_val
                
            intervals.append((start_val, end_val))
            
        if not intervals:
            computed_yoe = 0.0
        else:
            # Sort intervals by start_date ascending
            intervals.sort(key=lambda x: x[0])
            
            # Merge overlapping intervals to get non-overlapping active periods
            merged: List[List[date]] = []
            for start, end in intervals:
                if not merged:
                    merged.append([start, end])
                else:
                    prev_start, prev_end = merged[-1]
                    if start <= prev_end:
                        # Overlap: extend the current interval
                        merged[-1][1] = max(prev_end, end)
                    else:
                        # Gap: start a new interval
                        merged.append([start, end])
                        
            total_days = sum((end - start).days for start, end in merged)
            # Standardize 365.25 days per year to account for leap years
            computed_yoe = round(total_days / 365.25, 2)
            
        experience_difference = round(abs(reported_yoe - computed_yoe), 2)
        
        # If difference <= 6 months (0.5 years)
        if experience_difference <= 0.5:
            experience_consistency_score = 1.0
            inconsistent = False
        else:
            # Inconsistent: compute linear score penalty cap
            experience_consistency_score = round(max(0.0, 1.0 - (experience_difference / 10.0)), 2)
            inconsistent = True
            
        return {
            "computed_years_of_experience": computed_yoe,
            "reported_years_of_experience": reported_yoe,
            "experience_difference": experience_difference,
            "experience_consistency_score": experience_consistency_score,
            "inconsistent": inconsistent
        }
