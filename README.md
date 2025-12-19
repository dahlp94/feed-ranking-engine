# Real-Time Personalized Feed Ranking Engine

**An End-to-End Machine Learning System (Retrieval → Ranking → Evaluation)**

---

## Overview

This project implements an **end-to-end personalized feed ranking engine** inspired by large-scale consumer platforms such as Meta, Twitter, and LinkedIn.

Rather than focusing on a single model, the project emphasizes the **full ML system lifecycle**:

* realistic data generation
* baseline retrieval and ranking
* rigorous offline evaluation
* diagnostics and sanity checks
* business-aligned metrics

The goal is to demonstrate how ranking systems are built, validated, and iterated in real-world production settings.

---

## Business Impact Summary

Personalized feed ranking systems directly influence:

* **User engagement** (CTR, dwell time, session length)
* **Retention** (content relevance reduces churn)
* **Content discovery and creator health**
* **Revenue** (ads, subscriptions, discovery efficiency)

This project shows how ML-driven ranking creates business value by:

### Engagement Uplift Potential

* Baseline heuristics surface only **~5–6% of clicked items within the top-100**.
* Diagnostics reveal strong but underexploited signals (e.g., **~2× CTR uplift for followed authors**).
* Learned retrieval and ranking models can significantly improve **top-K relevance**, which typically translates to **double-digit engagement gains** at scale.

### Efficient Use of User Attention

* Most user value is captured in the **top-10 ranked items**.
* Improvements in ranking metrics such as **NDCG@10** directly correlate with faster user satisfaction and higher session quality.

### Reduced Dependence on Heuristics

Heuristic feeds (popularity, recency):

* Reinforce rich-get-richer dynamics
* Struggle with personalization
* Adapt slowly to changing user interests

This system establishes a principled foundation for **learning-based personalization**, enabling better discovery while maintaining fairness and scalability.

### Scalable Decision-Making

The architecture mirrors real production systems:

* Retrieval → ranking separation
* Offline evaluation aligned with online goals
* Diagnostics to validate data realism before deployment

This allows teams to estimate ROI **before** running costly A/B tests.

---

## Problem Setting

At each session:

* A user is shown **hundreds of candidate posts**
* Feedback is **implicit** (click / no-click)
* Engagement is **sparse and noisy**
* The system must return a ranked feed under latency constraints

This reflects real-world feed ranking challenges where:

* Individual signals are weak
* Learning requires aggregating many small effects
* Evaluation must be carefully designed

---

## System Architecture (Current Scope)

```
Synthetic Event Generator
        ↓
Session Builder (Impressions + Clicks)
        ↓
Baseline Candidate Retrieval
        ↓
Baseline Ranking Function
        ↓
Offline Evaluation (NDCG, Recall, MAP)
        ↓
Diagnostics & Sanity Checks
```

Future iterations extend this pipeline with learned retrieval, learning-to-rank models, and real-time serving.

---

## Data Simulation

To enable controlled experimentation, the project uses a **synthetic but realistic feed simulator**.

### Entities

* **Users:** 2,000
* **Posts:** 5,000
* **Authors:** 800
* **Latent topics:** 20
* **Time span:** 14 days

### User Behavior Model

Each user has a latent topic-preference vector that:

* Evolves over time (taste drift)
* Interacts with post topics

### Click Probability Depends On

* Topic match (user ↔ post)
* Post freshness
* Post popularity
* Whether the user follows the post’s author

The simulator intentionally produces **sparse implicit feedback**, making the ranking task non-trivial.

---

## Baseline Model

### Candidate Retrieval

A heuristic retrieval stage selects top-K candidates using:

* Popularity (smoothed CTR proxy)
* Recency
* Author-follow bonus

### Ranking

Candidates are ranked using a linear scoring function combining:

* Popularity
* Freshness
* Follow signal

This baseline represents a **business-as-usual heuristic feed** and provides a lower bound for performance.

---

## Baseline Results

Evaluated on ~50k sessions with time-based train/validation/test splits from a representative local run:

```
Validation:
  NDCG@10    ≈ 0.01–0.02
  Recall@100 ≈ 3–4%

Test:
  NDCG@10    ≈ 0.01
  Recall@100 ≈ 5–6%
```

### Interpretation

* Ranking is genuinely difficult due to large candidate sets and sparse clicks
* Heuristics capture limited signal
* Significant headroom exists for learned retrieval and ranking

---

## Diagnostics & Data Validation

Before introducing learned models, the simulator and baseline are validated with diagnostics.

### Session-Level Engagement

Summary from a representative local run:

* Mean session CTR ≈ **1.24%**
* Median session CTR ≈ **1.25%**
* 90th percentile CTR ≈ **2%**
* ~400 impressions per session with ~5 clicks on average

This confirms realistic sparsity and avoids trivial ranking scenarios.

### Popularity Signal

A weak-to-moderate relationship exists between popularity and clicks:

* Popularity is informative
* But insufficient for strong personalization

### Follow Bonus Effect

CTR by author-follow status:

* **Followed authors:** ~2.6%
* **Non-followed authors:** ~1.2%
* **Uplift:** ~2.1×

This validates meaningful behavioral structure while preserving room for learning.

<p align="center">
  <img src="assets/figures/ctr_distribution.png" width="30%">
  <img src="assets/figures/popularity_vs_clicks.png" width="30%">
  <img src="assets/figures/follow_bonus_effect.png" width="30%">
</p>

---

## Project Structure

```
feed-ranking-engine/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   └── dev.yaml
├── src/
│   └── frec/
│       ├── data/
│       │   ├── schema.py
│       │   └── simulator.py
│       ├── retrieval/
│       │   └── baseline.py
│       ├── eval/
│       │   └── ranking_metrics.py
│       └── viz/
│           └── diagnostics.py
├── scripts/
│   ├── run_baseline_eval.py
│   └── run_diagnostics.py
├── assets/
│   └── figures/
│       ├── ctr_distribution.png
│       ├── popularity_vs_clicks.png
│       └── follow_bonus_effect.png
└── .gitignore
```

---

## How to Run

### Install dependencies

```bash
pip install numpy pyyaml matplotlib
```

### Run baseline evaluation

```bash
python -m scripts.run_baseline_eval
```

### Run diagnostics and generate figures

```bash
python -m scripts.run_diagnostics
```

*(Figures used in the README are copied into `assets/figures/` for documentation.)*

---

## Key Takeaways

* Feed ranking with implicit feedback is inherently difficult
* Heuristic systems leave significant business value untapped
* Retrieval quality is a major bottleneck
* Diagnostics are essential before model iteration

This establishes a strong, credible foundation for learned ranking systems.

---

## Roadmap

Planned extensions:

1. Co-visitation candidate generation
2. Learning-to-rank (LambdaRank / neural rankers)
3. Two-tower embedding-based retrieval
4. Real-time feature updates
5. Latency optimization & monitoring
6. Offline-to-online evaluation alignment

---

## License

This project is released under the **MIT License**.
