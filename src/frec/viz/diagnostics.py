from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from src.frec.data.schema import SimData, Session, Post

MINUTES_PER_DAY = 24 * 60


@dataclass
class DiagnosticsSummary:
    n_sessions: int
    n_sessions_with_click: int
    mean_clicks_per_session: float
    session_ctr_mean: float
    session_ctr_median: float
    session_ctr_p90: float
    followed_ctr: float
    nonfollowed_ctr: float
    follow_uplift_ratio: float


def _post_lookup(posts: List[Post]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays indexed by post_id:
      - author_id[post_id]
      - topic_id[post_id]
      - created_ts[post_id]
    """
    n = len(posts)
    author_id = np.zeros(n, dtype=np.int32)
    topic_id = np.zeros(n, dtype=np.int32)
    created_ts = np.zeros(n, dtype=np.int32)
    for p in posts:
        author_id[p.post_id] = p.author_id
        topic_id[p.post_id] = p.topic_id
        created_ts[p.post_id] = p.created_ts
    return author_id, topic_id, created_ts


def compute_popularity_24h_from_sessions(
    posts: List[Post],
    sessions: List[Session],
    now_ts: int,
    window_minutes: int = 24 * 60,
    alpha: float = 1.0,
    beta: float = 50.0,
) -> np.ndarray:
    """
    Simple smoothed CTR-like popularity estimate from impression/click counts
    in a trailing time window ending at now_ts.

    popularity[post] = (clicks + alpha) / (impressions + beta)
    """
    n_posts = len(posts)
    imp = np.zeros(n_posts, dtype=np.int32)
    clk = np.zeros(n_posts, dtype=np.int32)

    lo = now_ts - window_minutes
    for s in sessions:
        if lo <= s.ts <= now_ts:
            for pid in s.impression_post_ids:
                imp[pid] += 1
            for pid in s.clicked_post_ids:
                clk[pid] += 1

    pop = (clk.astype(np.float32) + alpha) / (imp.astype(np.float32) + beta)
    return pop


def session_ctr_distribution(sessions: List[Session]) -> np.ndarray:
    """
    Per-session CTR = (#clicks / #impressions).
    """
    ctrs = []
    for s in sessions:
        n_imp = len(s.impression_post_ids)
        if n_imp == 0:
            continue
        ctrs.append(len(s.clicked_post_ids) / n_imp)
    return np.array(ctrs, dtype=np.float32)


def compute_follow_effect(
    data: SimData,
    sessions: List[Session],
) -> Tuple[float, float, float]:
    """
    Computes CTR on followed-author impressions vs non-followed-author impressions.

    Returns:
      followed_ctr, nonfollowed_ctr, uplift_ratio = followed_ctr / (nonfollowed_ctr + eps)
    """
    author_id, _, _ = _post_lookup(data.posts)

    followed_imps = 0
    followed_clicks = 0
    nonfollowed_imps = 0
    nonfollowed_clicks = 0

    for s in sessions:
        u = s.user_id
        follow_set = set(data.follows.get(u, []))
        clicked_set = set(s.clicked_post_ids)

        for pid in s.impression_post_ids:
            a = int(author_id[pid])
            if a in follow_set:
                followed_imps += 1
                if pid in clicked_set:
                    followed_clicks += 1
            else:
                nonfollowed_imps += 1
                if pid in clicked_set:
                    nonfollowed_clicks += 1

    eps = 1e-12
    followed_ctr = followed_clicks / (followed_imps + eps)
    nonfollowed_ctr = nonfollowed_clicks / (nonfollowed_imps + eps)
    uplift_ratio = followed_ctr / (nonfollowed_ctr + eps)
    return float(followed_ctr), float(nonfollowed_ctr), float(uplift_ratio)


def plot_ctr_hist(ctrs: np.ndarray, outpath: Path, title: str = "Session CTR distribution") -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(ctrs, bins=60)
    plt.xlabel("CTR per session (clicks / impressions)")
    plt.ylabel("Number of sessions")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_popularity_vs_clicks(
    data: SimData,
    sessions: List[Session],
    now_ts: int,
    outpath: Path,
    title: str = "Popularity (24h) vs Click Counts",
) -> None:
    """
    Scatter: x=popularity_24h (smoothed CTR), y=click counts in same window.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    n_posts = len(data.posts)
    pop = compute_popularity_24h_from_sessions(data.posts, sessions, now_ts=now_ts)

    # clicks in window (raw counts)
    clk = np.zeros(n_posts, dtype=np.int32)
    lo = now_ts - 24 * 60
    for s in sessions:
        if lo <= s.ts <= now_ts:
            for pid in s.clicked_post_ids:
                clk[pid] += 1

    # downsample for readability if needed
    idx = np.arange(n_posts)
    if n_posts > 5000:
        idx = np.random.choice(idx, size=5000, replace=False)

    x = pop[idx]
    y = clk[idx]

    plt.figure()
    plt.scatter(x, y, s=8)
    plt.xlabel("popularity_24h = (clicks+1)/(impressions+50)")
    plt.ylabel("click count (last 24h)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_follow_effect_bar(
    followed_ctr: float,
    nonfollowed_ctr: float,
    outpath: Path,
    title: str = "Follow Bonus Effect (CTR)",
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.bar(["Followed authors", "Non-followed authors"], [followed_ctr, nonfollowed_ctr])
    plt.ylabel("CTR (clicks / impressions)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def run_diagnostics(
    data: SimData,
    sessions: List[Session],
    figs_dir: Path,
    now_ts: int,
) -> DiagnosticsSummary:
    """
    Produces 3 diagnostic figures + returns summary stats.
    """
    ctrs = session_ctr_distribution(sessions)

    n_sessions = len(sessions)
    n_sessions_with_click = int(np.sum(ctrs > 0))
    mean_clicks_per_session = float(np.mean([len(s.clicked_post_ids) for s in sessions])) if n_sessions else 0.0

    session_ctr_mean = float(np.mean(ctrs)) if ctrs.size else 0.0
    session_ctr_median = float(np.median(ctrs)) if ctrs.size else 0.0
    session_ctr_p90 = float(np.quantile(ctrs, 0.90)) if ctrs.size else 0.0

    followed_ctr, nonfollowed_ctr, uplift_ratio = compute_follow_effect(data, sessions)

    # plots
    plot_ctr_hist(
        ctrs,
        figs_dir / "ctr_distribution.png",
        title="Session CTR distribution (synthetic feed simulator)",
    )
    plot_popularity_vs_clicks(
        data,
        sessions,
        now_ts=now_ts,
        outpath=figs_dir / "popularity_vs_clicks.png",
        title="Popularity proxy vs click counts (last 24h window)",
    )
    plot_follow_effect_bar(
        followed_ctr,
        nonfollowed_ctr,
        outpath=figs_dir / "follow_bonus_effect.png",
        title="Follow bonus effect: CTR by author-follow status",
    )

    return DiagnosticsSummary(
        n_sessions=n_sessions,
        n_sessions_with_click=n_sessions_with_click,
        mean_clicks_per_session=mean_clicks_per_session,
        session_ctr_mean=session_ctr_mean,
        session_ctr_median=session_ctr_median,
        session_ctr_p90=session_ctr_p90,
        followed_ctr=followed_ctr,
        nonfollowed_ctr=nonfollowed_ctr,
        follow_uplift_ratio=uplift_ratio,
    )
