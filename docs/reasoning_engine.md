# Reasoning Engine (EDHC)

The **EDHC Reasoning Engine** generates objective, factual, and evidence-backed candidate justifications for the final top 100 list. It operates in tandem with the **Experience Consistency Engine** to audit profile timelines, identify anomalies, and prevent contradictions in the generated summaries.

---

## 1. Pipeline Stages

```
[Candidate Profile] ──► 1. Experience Consistency Analyzer
                             ├── Mismatch check (> 6 months)
                             ├── Replace reported experience with computed
                             ▼
                        2. Reasoning Generator (Objective Tone)
                             ├── Purge generic adjectives
                             ├── Deterministic ID-hash template selection
                             ├── Signal & Notice Period check
                             ▼
                        [Factual Committee Summary]
```

---

## 2. Experience Consistency Engine

Before generating summaries, candidate profile timelines are audited by `ExperienceConsistencyAnalyzer`:

1. **Date Resolution**: Candidate career history dates are normalized. Current jobs are resolved using the system date, and missing or invalid end dates default to the current date.
2. **Timeline Sorting & Merging**: Experience duration is calculated by sorting roles chronologically and merging overlapping time intervals to compute exact total months. 
3. **Leap-Year Adjusted Tenure**: Merged total days are divided by **`365.25`** to get exact years.
4. **Mismatch Verification**: The analyzer checks for discrepancies between the candidate's reported experience and their chronological timeline:
   $$\text{Mismatch} = |\text{Stated\_Years} - \text{Computed\_Years}|$$
   - **Inconsistency Rule**: If the mismatch exceeds **6 months (0.5 years)**, the candidate's reported experience is replaced by their verified computed experience in explanations, and a credibility penalty of **`0.25`** is applied.

---

## 3. Reasoning Generator

The `ReasoningGenerator` constructs a professional, non-repetitive candidate review paragraph:

- **Adjective Elimination**: To maintain a neutral, factual tone suitable for an engineering committee review, subjective adjectives (like *"proven"*, *"factual"*, *"strong"*, *"excellent"*, *"outstanding"*, and *"highly skilled"*) are completely purged.
- **Narrative Structure**:
  - **Sentence 1 (Intro)**: Mapped using the verified title, computed years of experience, and prominent companies (e.g. *"spanning Google and Flipkart"*).
  - **Sentence 2 (Domain)**: Detailed summary of target technical area expertise (e.g. Search, NLP, Recommenders, MLOps, or general ML).
  - **Sentence 3 (Achievement/Skills)**: Highlights verified rubric competencies (high and medium confidence evidence matching skills list, career descriptions, summary, or project fields).
  - **Sentence 4 (Signals)**: Mapped to grammatical signals check for notice periods (e.g. *"Available immediately and maintains active open-source contributions on GitHub."*).
- **Template Rotation**: To prevent repetitive summaries, the engine uses a deterministic hash of the candidate's ID to select and rotate template structures:
  $$\text{Seed} = \sum \text{ord}(c) \quad \text{for } c \text{ in } \text{candidate\_id}$$
  This seed determines the variation used for the intro, domain, skills, and ending clauses.
