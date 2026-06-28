# Feature Engineering (EDHC)

The EDHC pipeline transforms candidate JSON profiles into a dense numerical feature matrix of shape $(N, 39)$. These features are categorized into 8 domains, preserving the 12 original features for backward compatibility while incorporating 27 advanced retrieval, semantic, career, skills, behavioral, credibility, and impact metrics.

---

## 1. Original Legacy Features (Indices 0–11)

These 12 features are locked for backward compatibility and model mapping:

1. **`services_company_ratio`**: Ratio of total career duration spent at known IT consulting/services firms (e.g. TCS, Infosys, Wipro, Cognizant, Tech Mahindra).
2. **`product_company_ratio`**: Ratio of total career duration spent at product-focused software companies (e.g. Google, Flipkart, Zomato).
3. **`startup_company_ratio`**: Ratio of career duration spent at small/growth-stage startups (company size < 500).
4. **`enterprise_company_ratio`**: Ratio of career duration spent at large enterprise organizations (company size > 5,000).
5. **`career_stability_score`**: A nonlinear scoring metric evaluating average job tenure. It penalizes frequent job-hopping (average tenures < 1.5 years).
6. **`notice_period_score`**: Stated notice period mapped to a 0–1 scale, where immediate availability is `1.0` and long notice periods (e.g. 90-120 days) approach `0.0`.
7. **`domain_relevance_score`**: Lexical matching density of title and description terms against target competency areas.
8. **`promotion_trajectory_score`**: Evaluates responsibility growth and title progression (e.g. Junior -> Senior -> Lead) inside career history.
9. **`evidence_strength`**: Aggregated confidence scores of all verified rubric competencies.
10. **`impact_extraction_score`**: Rate of quantitative achievement statements parsed in career descriptions.
11. **`credibility_score`**: System-wide profile consistency score (from `1.0` down to `0.01` if major timeline anomalies are found).
12. **`rrf_retrieval_score`**: Raw reciprocal rank fusion search score.

---

## 2. Advanced Retrieval Features (Indices 12–14)

13. **`bm25_retrieval_score`**: Raw scoring from BM25 Okapi lexical pre-filtering.
14. **`dense_retrieval_score`**: Cosine semantic similarity of E5 passage/query embeddings.
15. **`retrieval_rank_percentile`**: Percentile rank of candidate in initial RRF pool (scaled to $[0.0, 1.0]$).

---

## 3. Advanced Semantic Features (Indices 15–18)

16. **`jd_title_similarity`**: Word-level Jaccard similarity between candidate current/recent titles and JD title.
17. **`jd_summary_similarity`**: Word-level Jaccard similarity between candidate profile summary and JD rubric descriptions.
18. **`competency_coverage`**: Ratio of target job description competencies backed by at least one verified experience.
19. **`skill_overlap_ratio`**: Overlap ratio of JD required skills found in candidate's skills.

---

## 4. Advanced Career Features (Indices 19–24)

20. **`relevant_experience_years`**: Total years of career history spent in direct ML/NLP/IR domains.
21. **`average_tenure_years`**: Average duration in years across all career history roles.
22. **`career_continuity_score`**: Measures continuity (penalizes long gaps between adjacent job end/start dates).
23. **`promotion_velocity`**: Average years taken to achieve promotions (advances in title seniority).
24. **`leadership_indicator_score`**: Mentions of leadership keywords (Lead, Principal, Mentor, Scrum) in career history.
25. **`technical_specialization_trend`**: Evaluates whether candidate's focus is shifting toward specialized target roles over time.

---

## 5. Advanced Skills Features (Indices 25–29)

26. **`num_verified_skills`**: Number of declared skills validated against experience descriptions.
27. **`cross_source_verification_count`**: Count of skills supported by 2+ independent profile sections.
28. **`skill_diversity_score`**: Categorical diversity of verified technical skills.
29. **`skill_recency_score`**: Weights skills used in current/recent roles higher than older history.
30. **`skill_longevity_score`**: Cumulative duration of experience across all verified skills.

---

## 6. Advanced Behavioral Features (Indices 30–32)

31. **`notice_period_normalized`**: Notice period days mapped to a 0–1 scale (immediate = 1.0).
32. **`relocation_willingness`**: Mapped relocation willingness score.
33. **`availability_indicator`**: Mapped candidate platform activity score.

---

## 7. Advanced Credibility Features (Indices 33–36)

34. **`timeline_consistency`**: Mapped timeline overlaps and gaps.
35. **`contradiction_count`**: Count of anomalies or inconsistency warnings triggered.
36. **`profile_completeness`**: Mapped completeness score from `redrob_signals`.
37. **`cross_field_agreement`**: Consistency between profile summary assertions and chronological career history dates.

---

## 8. Advanced Impact Features (Indices 37–38)

38. **`quantitative_achievement_count`**: Mapped count of numerical percentages, scale indicators, and metrics.
39. **`performance_metric_mentions`**: Mentions of ranking, search, or scalability terms (NDCG, MAP, Latency, latency/p95).
