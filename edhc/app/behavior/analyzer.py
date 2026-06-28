from typing import Dict, Any
from edhc.app.schemas.candidate import CandidateProfile, normalize_candidate
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class BehavioralAnalyzer:
    """Computes hireability modifiers, recruiter engagement potential, notice periods, and relocation willingness from Redrob platform signals."""

    def analyze(self, candidate: Any) -> Dict[str, Any]:
        """Analyze candidates' metadata and text descriptors to extract hireability attributes."""
        candidate = normalize_candidate(candidate)
        logger.info(f"Analyzing behavior and hireability for candidate {candidate.candidate_id}")
        
        signals = candidate.redrob_signals
        
        # 1. Parse Notice Period Score
        notice_days = signals.notice_period_days
        notice_score = self._calculate_notice_score(notice_days)
        
        # 2. Parse Relocation and Location Preference
        relocates = signals.willing_to_relocate
        
        # 3. Calculate Recruiter Engagement and Activity Scores
        # Recruiter response rate is between 0.0 and 1.0. Let's make sure it's valid.
        resp_rate = max(0.0, min(1.0, signals.recruiter_response_rate))
        interview_rate = max(0.0, min(1.0, signals.interview_completion_rate))
        engagement_score = (resp_rate * 0.6) + (interview_rate * 0.4)
        
        # Activity score combining open to work flag, completeness score, and github score
        completeness = max(0.0, min(100.0, signals.profile_completeness_score)) / 100.0
        github = signals.github_activity_score
        github_factor = max(0.0, github / 100.0) if github >= 0 else 0.5  # Neutral default if no github linked
        
        open_to_work = 1.0 if signals.open_to_work_flag else 0.4
        activity_score = (open_to_work * 0.4) + (completeness * 0.3) + (github_factor * 0.3)
        
        # 4. Extract Hireability Modifiers (Contributions, community, blogs) from summary text
        modifiers = self._extract_modifiers(candidate)
        
        # Calculate Global Behavioral Score (0.0 to 1.0)
        behavioral_score = (notice_score * 0.4) + (activity_score * 0.2) + (engagement_score * 0.2) + (modifiers["modifier_weight"] * 0.2)
        if relocates:
            behavioral_score = min(1.0, behavioral_score + 0.1)  # Willingness to relocate is positive

        return {
            "behavioral_score": round(behavioral_score, 2),
            "notice_period_days": notice_days,
            "notice_period_score": round(notice_score, 2),
            "relocation_willing": relocates,
            "activity_score": round(activity_score, 2),
            "engagement_score": round(engagement_score, 2),
            "hireability_modifiers": modifiers["tags"],
            "relocation_status": "Willing" if relocates else "Not Indicated"
        }

    def _calculate_notice_score(self, days: int) -> float:
        """Score notice periods (lower notice period = higher immediate hireability)."""
        if days == 0:
            return 1.0  # Immediate joiner
        elif days <= 15:
            return 0.9
        elif days <= 30:
            return 0.75
        elif days <= 60:
            return 0.4
        else:
            return 0.1  # 3 months notice is a major friction point

    def _extract_modifiers(self, candidate: CandidateProfile) -> Dict[str, Any]:
        """Compute modifiers based on open-source activity, writing, or achievements in the profile summary."""
        tags = []
        weight = 0.5  # Neutral default
        
        sum_lower = candidate.profile.summary.lower()
        
        # Check for open source mentions
        if "open source" in sum_lower or "contributor" in sum_lower or "github" in sum_lower:
            tags.append("open_source_contributor")
            weight += 0.15
            
        if "blog" in sum_lower or "writer" in sum_lower or "speaker" in sum_lower or "talks" in sum_lower:
            tags.append("technical_communicator")
            weight += 0.1
            
        if "patent" in sum_lower or "publication" in sum_lower or "paper" in sum_lower or "research" in sum_lower:
            tags.append("research_background")
            weight += 0.1
            
        # Check if github activity score is high
        if candidate.redrob_signals.github_activity_score > 70:
            tags.append("highly_active_github")
            weight += 0.15

        return {
            "tags": tags,
            "modifier_weight": round(min(1.0, weight), 2)
        }

