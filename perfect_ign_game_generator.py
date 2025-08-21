"""
Perfect IGN Game Generator

This module searches over (year, platform set, genre set, optional reviewer)
combinations and also optimizes the *title shape* (colon, numerals, length)
that feature pipeline uses, then returns the top-K predicted IGN scores
with suggested names.

Notes
-----
- Reviewer optimization is optional (off by default). If you enable it,
  the generator will pick a historically generous author (per smoothed prior),
  which maximizes predicted scores but is less "agnostic".
- Category (c__*) features default to 0 unless you pass explicit picks.
- Platforms are optimized over 1–2 element sets drawn from the most frequent
  platforms in the data; you can increase `max_platforms_in_combo` or
  `platform_pool_size` for bigger searches.
"""

from __future__ import annotations
import itertools
import math
import random
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# =========================
# Utilities and primitives
# =========================
_CLEAN_RE = re.compile(r"[^\w\s&-]")
_NUM_RE_ROMAN = r"(?:\d+|i{1,3}|iv|v|vi{0,3}|ix|x)"
_IS_DLC_RE = r"\b(?:dlc|expansion|episode|chapter|pack|remaster|definitive|remake|remastered|anniversary)\b"

_PLATFORM_CANON = {
    'pc': 'PC', 'windows': 'PC', 'steam': 'PC',
    'ps5': 'PlayStation 5', 'playstation5': 'PlayStation 5',
    'ps4': 'PlayStation 4', 'playstation4': 'PlayStation 4',
    'psvr': 'PlayStation VR', 'ps_vr': 'PlayStation VR',
    'psvita': 'PlayStation Vita', 'vita': 'PlayStation Vita',
    'xbox_one': 'Xbox One', 'xb1': 'Xbox One',
    'xbox_series_x': 'Xbox Series X/S', 'xbox_series_s': 'Xbox Series X/S',
    'xbox_series': 'Xbox Series X/S', 'series_x': 'Xbox Series X/S', 'series_s': 'Xbox Series X/S',
    'switch': 'Nintendo Switch', 'nintendo_switch': 'Nintendo Switch',
    '3ds': 'Nintendo 3DS', 'nintendo_3ds': 'Nintendo 3DS',
    'wii_u': 'Wii U', 'stadia': 'Google Stadia'
}


def _clean_text(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "unknown"
    s = str(s).lower().strip()
    s = _CLEAN_RE.sub('', s)
    return s.replace(' ', '_') or "unknown"


def _canon_platform(token: str) -> str:
    t = (token or '').strip().lower().replace(' ', '_')
    return _PLATFORM_CANON.get(t, (token or '').strip()) or "Unknown"


# =========================
# Priors from training data
# =========================

def build_priors(df: pd.DataFrame,
                 genre_cols: Sequence[str],
                 m_author: float = 5.0,
                 m_creator: float = 5.0,
                 m_platform: float = 5.0,
                 m_genre: float = 5.0) -> dict:
    """Smoothed historical means for bias-aware features."""
    global_mean = float(df["ign_score"].mean())

    # author
    a = df.groupby('author')["ign_score"].agg(['mean','count'])
    author_prior = ((a['count']*a['mean'] + m_author*global_mean) / (a['count'] + m_author)).to_dict()

    # creator
    c = df.groupby('creator')["ign_score"].agg(['mean','count'])
    creator_prior = ((c['count']*c['mean'] + m_creator*global_mean) / (c['count'] + m_creator)).to_dict()

    # platform (explode if list-like)
    plat = df[['platform','ign_score']].copy()
    mask_listlike = plat['platform'].apply(lambda x: isinstance(x, (list, tuple, set)))
    if not mask_listlike.all():
        plat.loc[~mask_listlike, 'platform'] = (
            plat.loc[~mask_listlike, 'platform'].fillna('').astype(str)
                .apply(lambda s: [p.strip() for p in s.split(',')] if s else [])
        )
    plat = plat.explode('platform')
    plat['platform'] = plat['platform'].fillna('Unknown').astype(str)
    p = plat.groupby('platform')['ign_score'].agg(['mean','count'])
    platform_prior = ((p['count']*p['mean'] + m_platform*global_mean) / (p['count'] + m_platform)).to_dict()

    # genres (from one-hots)
    g_means = {}
    for g in genre_cols:
        on = df[g] == 1
        n = int(on.sum())
        if n == 0:
            g_means[g] = global_mean
        else:
            mean = float(df.loc[on, 'ign_score'].mean())
            g_means[g] = (n*mean + m_genre*global_mean) / (n + m_genre)

    return dict(
        global_mean=global_mean,
        author_prior=author_prior,
        creator_prior=creator_prior,
        platform_prior=platform_prior,
        genre_prior=g_means,
    )


# =========================================
# Single-example feature row for prediction
# =========================================

def make_feature_row(title: str,
                     year: int,
                     platforms: Sequence[str],
                     chosen_genres: Sequence[str],
                     *,
                     author: str = "unknown",
                     creator: str = "unknown",
                     expected_platforms: Sequence[str],
                     genre_cols: Sequence[str],
                     cat_cols: Sequence[str],
                     priors: dict,
                     pre) -> pd.DataFrame:
    """Build a feature row matching `pre.feature_names_in_` exactly."""
    title_str = str(title or '').strip()
    has_colon = (":" in title_str)
    has_num = bool(re.search(rf"\b{_NUM_RE_ROMAN}\b", title_str, flags=re.I))
    is_dlc  = bool(re.search(_IS_DLC_RE, title_str, flags=re.I))
    title_len = len(title_str)

    # categorical
    author_clean = _clean_text(author)
    creator_clean = _clean_text(creator)

    # platforms → canonical one-hots
    plats_canon = [_canon_platform(p) for p in platforms if str(p).strip()]
    platform_count = len(plats_canon)
    platform_ohe = {p: 0 for p in expected_platforms}
    for p in plats_canon:
        if p in platform_ohe:
            platform_ohe[p] = 1

    # genre one-hots (case-insensitive map)
    genre_lookup = {c.replace("g__", "").lower(): c for c in genre_cols}
    genre_ohe = {g: 0.0 for g in genre_cols}
    for g in chosen_genres:
        key = str(g or '').strip().lower()
        col = genre_lookup.get(key)
        if col:
            genre_ohe[col] = 1.0

    # priors
    global_mean = float(priors['global_mean'])
    author_avg  = float(priors['author_prior'].get(author_clean, global_mean))
    creator_avg = float(priors['creator_prior'].get(creator_clean, global_mean))
    plat_vals = [priors['platform_prior'][p] for p in plats_canon if p in priors['platform_prior']]
    platform_avg = float(np.mean(plat_vals)) if len(plat_vals) else global_mean
    if chosen_genres:
        gvals = []
        for g in chosen_genres:
            col = genre_lookup.get(str(g).strip().lower())
            if col and col in priors['genre_prior']:
                gvals.append(priors['genre_prior'][col])
        genre_avg = float(np.mean(gvals)) if gvals else global_mean
    else:
        genre_avg = global_mean

    # assemble raw row
    row = {
        "title": _clean_text(title_str),
        "year": float(year) if year else np.nan,
        "has_colon": float(has_colon),
        "has_num": float(has_num),
        "is_dlc": float(is_dlc),
        "review_lag_days": 0.0,
        "platform_count": float(platform_count),
        "title_len": float(title_len),
        "author": author_clean,
        "creator": creator_clean,
        "author_avg": author_avg,
        "creator_avg": creator_avg,
        "platform_avg": platform_avg,
        "genre_avg": genre_avg,
    }
    row.update(platform_ohe)
    row.update(genre_ohe)

    # include all category one-hots as zeros unless given explicitly
    for c in cat_cols:
        if c not in row:
            row[c] = 0.0

    df_row = pd.DataFrame([row])
    # Ensure every expected input column exists; fill missing numeric with 0
    for col in pre.feature_names_in_:
        if col not in df_row.columns:
            df_row[col] = 0.0
    # Reorder to exact input schema for ColumnTransformer
    df_row = df_row[pre.feature_names_in_]
    return df_row


# =================
# Name suggestions
# =================

_LEFT = [
    "The", "Shadow", "Elden", "Final", "Dragon", "Star",
    "Chronicles", "Legend", "Kingdom", "Arc", "Iron", "Neon",
    "Phantom", "Warrior", "Myth", "Astral", "Valkyrie", "Celestial"
]
_RIGHT = [
    "Crown", "Eclipse", "Odyssey", "Reckoning", "Saga", "Rift",
    "Dominion", "Legacy", "Paradox", "Embers", "Echoes", "Remnant",
    "Frontier", "Horizon", "Sanctum", "Origin"
]


def _roman(n: int) -> str:
    vals = [
        (10, 'X'), (9, 'IX'), (8, 'VIII'), (7, 'VII'), (6, 'VI'), (5, 'V'),
        (4, 'IV'), (3, 'III'), (2, 'II'), (1, 'I')
    ]
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s)
            n -= v
    return ''.join(out) or 'I'


def candidate_names(seed: int | None = None) -> List[str]:
    rnd = random.Random(seed)
    def pick() -> str:
        return f"{rnd.choice(_LEFT)} {rnd.choice(_RIGHT)}"
    base = [pick() for _ in range(6)]
    # Variants target your title features: colon, numerals, length
    variants = []
    for b in base:
        root = b
        variants.extend([
            root,                                        # plain
            f"{root}: Definitive Edition",               # colon + dlc-ish (might increase is_dlc)
            f"{root} {_roman(rnd.randint(2, 10))}",      # sequel numeral
            f"{root} Chronicles",                        # longer title
            root.replace(' ', ''),                        # single word (shorter)
            f"Project {root.split()[0]}",               # different structure
        ])
    # Ensure uniqueness and reasonable length
    uniq = []
    for v in variants:
        v = v.strip()
        if 3 <= len(v) <= 48 and v not in uniq:
            uniq.append(v)
    return uniq[:24]


# ==============================
# Search space & optimization
# ==============================

@dataclass
class SearchConfig:
    top_genres_by_prior: int = 12
    top_platforms_by_freq: int = 6
    max_platforms_in_combo: int = 2  # try 1–2 platforms
    years_back: int = 8              # search last N years observed
    allow_dlc_titles: bool = True
    optimize_reviewer: bool = False
    seed: int | None = 1337


def _distinct_recent_years(df: pd.DataFrame, back: int) -> List[int]:
    years = sorted(df['year'].dropna().astype(int).unique())
    if not years:
        return []
    hi = max(years)
    return [y for y in years if y >= hi - back]


def _platform_pool(df: pd.DataFrame, k: int) -> List[str]:
    plat = df[['platform']].explode('platform').copy()
    plat['platform'] = plat['platform'].fillna('Unknown').astype(str)
    vals = plat['platform'].value_counts().head(k).index.tolist()
    return vals


def _genre_pool_by_prior(priors: dict, k: int) -> List[str]:
    items = sorted(priors['genre_prior'].items(), key=lambda kv: kv[1], reverse=True)
    return [g for g, _ in items[:k]]  # already g__ columns


def _genre_sets(genres: Sequence[str]) -> List[List[str]]:
    # consider 1–2 genre combos (use the prefixless names for display)
    out = []
    short = [g for g in genres]
    for g in short:
        out.append([g])
    for a, b in itertools.combinations(short, 2):
        out.append([a, b])
    return out


def _platform_sets(platforms: Sequence[str], max_k: int) -> List[List[str]]:
    out = [[p] for p in platforms]
    if max_k >= 2:
        out += [list(t) for t in itertools.combinations(platforms, 2)]
    return out


# ===============
# Main generator
# ===============

def generate_perfect_ign_games(df: pd.DataFrame,
                               model,
                               pre,
                               genre_cols: Sequence[str],
                               expected_platforms: Sequence[str],
                               cat_cols: Sequence[str],
                               *,
                               k: int = 20,
                               cfg: SearchConfig | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      [pred, name, year, platforms, genres, reviewer, author_avg, platform_avg, genre_avg]
    Sorted by highest predicted score.
    """
    if cfg is None:
        cfg = SearchConfig()

    rnd = random.Random(cfg.seed)

    # Build priors and pools
    priors = build_priors(df, genre_cols)
    years = _distinct_recent_years(df, cfg.years_back)
    plat_pool = _platform_pool(df, cfg.top_platforms_by_freq)
    genre_pool = _genre_pool_by_prior(priors, cfg.top_genres_by_prior)

    if not years:
        # fallback to median year if release_date was sparse
        years = [int(df['date'].dt.year.median())]

    plat_sets = _platform_sets(plat_pool, cfg.max_platforms_in_combo)
    genre_sets = _genre_sets(genre_pool)

    # Optional: pick a generous reviewer by prior
    reviewer = "unknown"
    if cfg.optimize_reviewer:
        # choose top-3 generous authors with at least a minimal count
        a = df.groupby('author')["ign_score"].agg(['mean','count']).reset_index()
        a = a[a['count'] >= 10].sort_values('mean', ascending=False)
        if not a.empty:
            reviewer = str(a.iloc[0]['author'])

    rows = []

    # Search combinations (sampled to keep runtime bounded)
    # limit total combinations roughly to ~10k before name variants
    max_meta = 10000
    meta_combos = list(itertools.product(years, plat_sets, genre_sets))
    if len(meta_combos) > max_meta:
        meta_combos = rnd.sample(meta_combos, max_meta)

    name_pool = candidate_names(seed=cfg.seed)
    if not cfg.allow_dlc_titles:
        name_pool = [n for n in name_pool if not re.search(_IS_DLC_RE, n, flags=re.I)]
    if not name_pool:
        name_pool = ["Project Crown"]

    for yr, plats, gens in meta_combos:
        # Convert g__ back to display and input names
        # Input expects g__* names; display strips prefix
        input_gen_names = [g for g in gens]
        display_gen_names = [g.replace('g__', '') for g in gens]

        best_local = None
        for name in name_pool:
            # Build feature row
            df_row = make_feature_row(
                title=name,
                year=int(yr),
                platforms=plats,
                chosen_genres=[g.replace('g__','') for g in input_gen_names],
                author=reviewer,
                creator="unknown",
                expected_platforms=expected_platforms,
                genre_cols=genre_cols,
                cat_cols=cat_cols,
                priors=priors,
                pre=pre,
            )
            X_new = pre.transform(df_row)
            pred = float(model.predict(X_new)[0])

            # keep track of priors for explanation
            author_avg = float(priors['author_prior'].get(_clean_text(reviewer), priors['global_mean']))
            plat_vals = [priors['platform_prior'][p] for p in plats if p in priors['platform_prior']]
            platform_avg = float(np.mean(plat_vals)) if plat_vals else float(priors['global_mean'])
            gvals = [priors['genre_prior'].get(g, priors['global_mean']) for g in input_gen_names]
            genre_avg = float(np.mean(gvals)) if gvals else float(priors['global_mean'])

            rec = dict(
                pred=pred,
                name=name,
                year=int(yr),
                platforms=", ".join(plats),
                genres=", ".join(display_gen_names),
                reviewer=reviewer,
                author_avg=author_avg,
                platform_avg=platform_avg,
                genre_avg=genre_avg,
            )

            if (best_local is None) or (pred > best_local['pred']):
                best_local = rec
        if best_local:
            rows.append(best_local)

    # Rank and dedupe by name
    out = pd.DataFrame(rows).sort_values('pred', ascending=False)
    out = out.drop_duplicates(subset=['name'])
    return out.head(k).reset_index(drop=True)


if __name__ == "__main__":
    print("This module is meant to be imported and used via generate_perfect_ign_games(...)")
