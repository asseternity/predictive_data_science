
# predict_ign_cli.py
# Minimal CLI to collect a game's details and predict its IGN score using
# your already-trained `model` and `pre` plus training DataFrame `df`.

import re
import numpy as np
import pandas as pd

# --- helpers to match your cleaning ---
_CLEAN_RE = re.compile(r'[^\w\s&-]')  # keep letters/numbers/underscore/&/-

def _clean_text(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "unknown"
    s = str(s).lower().strip()
    s = _CLEAN_RE.sub('', s)
    # mimic your pipeline: then replace spaces with underscores
    s = s.replace(' ', '_')
    return s or "unknown"

# platform alias → canonical (align to expected_platforms)
_PLATFORM_CANON = {
    'pc': 'PC',
    'windows': 'PC',
    'steam': 'PC',
    'ps5': 'PlayStation 5',
    'playstation5': 'PlayStation 5',
    'ps4': 'PlayStation 4',
    'playstation4': 'PlayStation 4',
    'psvr': 'PlayStation VR',
    'ps_vr': 'PlayStation VR',
    'psvita': 'PlayStation Vita',
    'vita': 'PlayStation Vita',
    'xbox_one': 'Xbox One',
    'xb1': 'Xbox One',
    'xbox_series_x': 'Xbox Series X/S',
    'xbox_series_s': 'Xbox Series X/S',
    'xbox_series': 'Xbox Series X/S',
    'series_x': 'Xbox Series X/S',
    'series_s': 'Xbox Series X/S',
    'switch': 'Nintendo Switch',
    'nintendo_switch': 'Nintendo Switch',
    '3ds': 'Nintendo 3DS',
    'nintendo_3ds': 'Nintendo 3DS',
    'wii_u': 'Wii U',
    'stadia': 'Google Stadia'
}

def _canon_platform(token: str) -> str:
    t = token.strip().lower().replace(' ', '_')
    return _PLATFORM_CANON.get(t, token.strip()) or "Unknown"

# --- build smoothed priors from training df ---
def build_priors(df: pd.DataFrame,
                 genre_cols: list[str],
                 m_author: float = 5.0,
                 m_creator: float = 5.0,
                 m_platform: float = 5.0,
                 m_genre: float = 5.0):
    """
    Returns dicts for author/creator/platform/genre smoothed means using training data.
    Uses global-mean shrinkage: (n*mean + m*global) / (n+m)
    """
    global_mean = float(df["ign_score"].mean())

    def _smoothed_mean(series: pd.Series, m: float) -> pd.Series:
        agg = series.groupby(level=0).agg(['mean','count'])
        return (agg['count']*agg['mean'] + m*global_mean) / (agg['count'] + m)

    # author
    author_stats = df.groupby('author')["ign_score"].agg(['mean','count'])
    author_prior = ((author_stats['count']*author_stats['mean'] + m_author*global_mean) /
                    (author_stats['count'] + m_author)).to_dict()

    # creator
    creator_stats = df.groupby('creator')["ign_score"].agg(['mean','count'])
    creator_prior = ((creator_stats['count']*creator_stats['mean'] + m_creator*global_mean) /
                     (creator_stats['count'] + m_creator)).to_dict()

    # platform (explode if list-like)
    plat = df[['platform','ign_score']].copy()
    # if 'platform' is strings with commas/lists, normalize to lists
    mask_listlike = plat['platform'].apply(lambda x: isinstance(x, (list, tuple, set)))
    if not mask_listlike.all():
        # try to split comma-separated strings into lists
        plat.loc[~mask_listlike, 'platform'] = plat.loc[~mask_listlike, 'platform'].fillna('').astype(str).apply(
            lambda s: [p.strip() for p in s.split(',')] if s else [])
    plat = plat.explode('platform')
    plat['platform'] = plat['platform'].fillna('Unknown').astype(str)
    platform_stats = plat.groupby('platform')['ign_score'].agg(['mean','count'])
    platform_prior = ((platform_stats['count']*platform_stats['mean'] + m_platform*global_mean) /
                      (platform_stats['count'] + m_platform)).to_dict()

    # genre
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
        genre_prior=g_means
    )

# --- feature builders for a single example ---
_NUM_RE_ROMAN = r'(?:\d+|i{1,3}|iv|v|vi{0,3}|ix|x)'
_IS_DLC_RE = r'\b(?:dlc|expansion|episode|chapter|pack|remaster|definitive)\b'

def features_from_inputs(title: str,
                         year: int,
                         platforms_csv: str,
                         author: str,
                         # objects from your training environment:
                         df_train: pd.DataFrame,
                         pre,
                         expected_platforms: list[str],
                         genre_cols: list[str],
                         cat_cols: list[str],
                         priors: dict):
    """
    Build a single-row DataFrame with the exact columns expected by `pre`,
    then transform → predict with `model`.
    """
    title_str = str(title or "").strip()
    t_low = title_str.lower()
    has_colon = (":" in title_str)
    has_num = bool(re.search(rf'\b{_NUM_RE_ROMAN}\b', t_low, flags=re.I))
    is_dlc  = bool(re.search(_IS_DLC_RE, t_low, flags=re.I))
    title_len = len(title_str)
    review_lag_days = 0  # unknown at predict-time

    # categorical
    author_clean = _clean_text(author)
    creator_clean = "unknown"  # not asked; can be extended later

    # platforms → list of canonical tokens
    plats_in = [p.strip() for p in (platforms_csv or "").split(",") if p.strip()]
    plats_canon = [_canon_platform(p) for p in plats_in]
    platform_count = len(plats_canon)

    # platform one-hots (only the expected ones)
    platform_ohe = {p: 0 for p in expected_platforms}
    for p in plats_canon:
        if p in platform_ohe:
            platform_ohe[p] = 1

    # genre one-hots (user didn't enter; default all zeros)
    genre_ohe = {g: 0.0 for g in genre_cols}

    # bias priors
    global_mean = priors['global_mean']
    author_avg  = priors['author_prior'].get(author_clean, global_mean)
    creator_avg = priors['creator_prior'].get(creator_clean, global_mean)

    # platform_avg: mean across provided platforms present in prior dict,
    # else fall back to global_mean
    plat_vals = [priors['platform_prior'][p] for p in plats_canon if p in priors['platform_prior']]
    platform_avg = float(np.mean(plat_vals)) if len(plat_vals) else global_mean

    # genre_avg: none provided → global_mean (kept simple);
    # if you later add a genre prompt, average priors for chosen genres.
    genre_avg = global_mean

    # assemble raw row
    row = {
        "title": _clean_text(title_str),
        "year": float(year) if year else np.nan,
        "has_colon": float(has_colon),
        "has_num": float(has_num),
        "is_dlc": float(is_dlc),
        "review_lag_days": float(review_lag_days),
        "platform_count": float(platform_count),
        "title_len": float(title_len),
        "author": author_clean,
        "creator": creator_clean,
        "author_avg": float(author_avg),
        "creator_avg": float(creator_avg),
        "platform_avg": float(platform_avg),
        "genre_avg": float(genre_avg),
    }
    row.update(platform_ohe)
    row.update(genre_ohe)

    # Ensure all expected columns exist for `pre`
    # `cat_cols` are categorical names used by the OneHotEncoder in `pre`.
    # Numeric columns are everything else `pre` expects via passthrough.
    df_row = pd.DataFrame([row])

    # Some pipelines expect exactly the same numeric schema.
    # Add any missing expected_platform columns explicitly (already ensured).
    # Add any missing genre_cols
    for g in genre_cols:
        if g not in df_row.columns:
            df_row[g] = 0.0

    # Return row as built plus the transformed features for debugging
    return df_row

def cli_predict(df, model, pre, genre_cols, cat_cols, expected_platforms,
                m_author=5.0, m_creator=5.0, m_platform=5.0, m_genre=5.0):
    """
    Launch a tiny terminal Q&A, build features, transform with `pre`, predict with `model`.
    """
    print("\n=== IGN Score Predictor (CLI) ===")
    title = input("Game name? ").strip()
    year_str = input("Game release year? ").strip()
    platforms = input("Game platforms? (comma-separated, e.g., PC, PlayStation 5) ").strip()
    author = input("Reviewer name? ").strip()

    year = int(year_str) if year_str.isdigit() else 0

    # Build prior tables from training frame
    priors = build_priors(df, genre_cols, m_author, m_creator, m_platform, m_genre)

    # Build feature row
    raw_row = features_from_inputs(title, year, platforms, author,
                                   df_train=df, pre=pre,
                                   expected_platforms=expected_platforms,
                                   genre_cols=genre_cols, cat_cols=cat_cols,
                                   priors=priors)
    
    # Build a case-insensitive map once (after you have genre_cols)
    genre_lookup = {c.replace("g__", "").lower(): c for c in genre_cols}
    genres_input = input("Game genres? (comma separated, e.g. Action,RPG): ").strip()
    chosen_genres = [g.strip().lower() for g in genres_input.split(",") if g.strip()]

    # Initialize to 0 for all genres (kept)
    for g in genre_cols:
        raw_row[g] = 0

    # Flip to 1 for recognized genres (case-insensitive)
    for g in chosen_genres:
        col = genre_lookup.get(g)  # e.g., "rpg" -> "g__RPG"
        if col:
            raw_row[col] = 1
        else:
            print(f"⚠️  Unknown genre: {g} (skipped)")
    
    # --- Ensure all expected categorical one-hots exist ---
    for c in cat_cols:
        if c not in raw_row.columns:
            raw_row[c] = 0  # assume absent

    # --- Ensure all expected platform one-hots exist ---
    for p in expected_platforms:
        col = f"p__{p}"
        if col not in raw_row.columns:
            raw_row[col] = 0

    # --- Ensure all expected genre one-hots exist ---
    for g in genre_cols:
        if g not in raw_row.columns:
            raw_row[g] = 0

    # Reorder columns so pre.transform() sees them all
    raw_row = raw_row.reindex(columns=[*pre.feature_names_in_])

    # Transform and predict
    X_new = pre.transform(raw_row[["creator","author"] + [c for c in raw_row.columns if c not in ["creator","author"]]])
    pred = float(model.predict(X_new)[0])

    print(f"\nPredicted IGN score for '{title}': {pred:.2f}\n")
    return pred, raw_row

if __name__ == "__main__":
    print("This module is intended to be imported and called as:\n"
          "  from predict_ign_cli import cli_predict\n"
          "  cli_predict(df, model, pre, genre_cols, cat_cols, expected_platforms)\n")
