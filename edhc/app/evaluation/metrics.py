from typing import List
import numpy as np
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class RankingEvaluator:
    """Computes search and ranking performance metrics for candidate selection audit."""

    @staticmethod
    def precision_at_k(recommended_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if not recommended_ids or not ground_truth_ids or k <= 0:
            return 0.0
            
        top_k_rec = recommended_ids[:k]
        hits = sum(1 for cid in top_k_rec if cid in ground_truth_ids)
        return hits / k

    @staticmethod
    def mean_reciprocal_rank(recommendations: List[List[str]], ground_truth: List[List[str]]) -> float:
        """Calculate MRR over a batch of queries."""
        if len(recommendations) != len(ground_truth) or not recommendations:
            return 0.0
            
        rr_list = []
        for rec, gt in zip(recommendations, ground_truth):
            rr = 0.0
            for rank, cid in enumerate(rec, 1):
                if cid in gt:
                    rr = 1.0 / rank
                    break
            rr_list.append(rr)
            
        return float(np.mean(rr_list))

    @staticmethod
    def average_precision(recommended_ids: List[str], ground_truth_ids: List[str]) -> float:
        """Calculate Average Precision (AP) for a single query."""
        if not recommended_ids or not ground_truth_ids:
            return 0.0
            
        hits = 0
        sum_precisions = 0.0
        
        for rank, cid in enumerate(recommended_ids, 1):
            if cid in ground_truth_ids:
                hits += 1
                precision = hits / rank
                sum_precisions += precision
                
        if hits == 0:
            return 0.0
            
        return sum_precisions / min(len(ground_truth_ids), len(recommended_ids))

    @classmethod
    def mean_average_precision(cls, recommendations: List[List[str]], ground_truth: List[List[str]]) -> float:
        """Calculate MAP over a batch of queries."""
        if len(recommendations) != len(ground_truth) or not recommendations:
            return 0.0
            
        ap_list = [cls.average_precision(rec, gt) for rec, gt in zip(recommendations, ground_truth)]
        return float(np.mean(ap_list))

    @staticmethod
    def dcg_at_k(relevance_scores: List[float], k: int) -> float:
        """Calculate Discounted Cumulative Gain at K (DCG@K)."""
        scores = np.asfarray(relevance_scores)[:k]
        if scores.size == 0:
            return 0.0
            
        # DCG formulation: Sum( (2^rel - 1) / log2(idx + 1) )
        return float(np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2))))

    @classmethod
    def ndcg_at_k(cls, recommended_relevance: List[float], ideal_relevance: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K)."""
        dcg = cls.dcg_at_k(recommended_relevance, k)
        idcg = cls.dcg_at_k(sorted(ideal_relevance, reverse=True), k)
        
        if idcg == 0.0:
            return 0.0
            
        return float(dcg / idcg)
