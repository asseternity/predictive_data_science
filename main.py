# Idea: 
# I built a pipeline to scrape and structure 2,000 IGN reviews, 
# enriched the data using an API, 
# and trained a model to understand scoring trends. 
# It turns out IGN favors RPGs and penalizes racing games.
# Can't use API because it only serves 10 reviews and limits to 100 requests

# Questions:
# - which genres/platforms/developers get higher scores?
# - which writers give higher scores?
# - did IGN's average scores change over time?
# - are DLCs and expansions higher rated than standalone games?
# - are sequels in general well reviewed?
# - in terms of name, genre, platform - can an LLM predict and generate the "perfect IGN game"?

# ------ 1. Cache the data ------
import json
import os

CACHE_PATH = "reviews_cache.json"

def clean_text(s):
    if pd.isna(s):
        return s
    # Lowercase, strip spaces, remove trailing commas/punctuation
    s = s.lower().strip()
    s = re.sub(r'[^\w\s&-]', '', s)  # keep letters/numbers/underscore/&/-
    return s

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= 2000:
            return data
        return None
    return None

def save_cache(data):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, indent=2)
    print(f"Saved {len(data)} reviews to cache.")

# ------ 2. Scraping Opencritic's IGN page with BeautifulSoup ------
# Note: to not hammer any servers, I will use delays (e.g., time.sleep(2)) if needed.
import re
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

options = Options()
options.add_argument("--headless") # run in background
driver = webdriver.Chrome(options=options)

cached = load_cache()
if cached:
    for item in cached:
        item["date"] = datetime.fromisoformat(item["date"]).date()
    all_games = cached
    print(f"Loaded {len(all_games)} reviews from cache.")
else:
    all_games = []
    for page in range(1, 119):
        url = f"https://opencritic.com/outlet/56/ign?page={page}"
        driver.get(url)
        time.sleep(3) # wait for JS to render
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find each HTML review block
        review_blocks = soup.find_all("div", class_="review-row")
        print(f"Found review blocks: {len(review_blocks)}")

        # See the HTML for one block with BeautifulSoup's prettify
        # print(review_blocks[0].prettify())

        for block in review_blocks:
            # Author
            author_el = block.select_one(".author-name a")
            author = author_el.get_text(strip=True) if author_el else None

            # Score
            score_el = block.select_one(".score-display .score-number-bold")
            if not score_el:
                continue
            score = float(score_el.get_text(strip=True).split("/")[0])

            # Date
            date_el = block.select_one(".date-block")
            if date_el:
                date = datetime.strptime(date_el.get_text(strip=True), "%b %d, %Y").date()
            else:
                date = None

            # Title
            title_el = block.select_one(".score-display a")
            title = title_el.get_text(strip=True) if title_el else None
        
            # Link
            link_element = block.find('a', href=re.compile(r'^/game/\d+/'))
            link = f"https://opencritic.com{link_element['href']}" if link_element else None

            all_games.append({
                "title": title,
                "ign_score": score,
                "date": date,
                "author": author,
                "link": link,
            })
            print(f"{date} | {author} | {title} | {score}")

# ------ 3. Getting Metadata from OpenCritic Game Pages with Selenium ------
if not cached:
    for game in all_games:
        creator = ""
        release_date = None
        platform = ""

        if not game.get("link"):
            # Skip if there's no OpenCritic page
            game.update({
                "creator": creator,
                "release_date": release_date,
                "platform": platform
            })
            continue

        print(f"Fetching metadata for {game['title']}: {game['link']}")
        try:
            driver.get(game["link"])
            time.sleep(2)  # Wait for the page to load

            soup = BeautifulSoup(driver.page_source, "html.parser")

            # --- Creator ---
            creator_el = soup.select_one("div.companies span")
            if creator_el:
                creator = creator_el.get_text(strip=True)

            # --- Release Date & Platform ---
            platform_el = soup.select_one("div.platforms")
            if platform_el:
                raw_text = platform_el.get_text(separator=" ", strip=True)
                # Example: "Release Date: Jul 30, 2024 - PC"
                match = re.search(r"Release Date:\s*([A-Za-z]{3} \d{1,2}, \d{4})\s*-\s*(.+)", raw_text)
                if match:
                    date_str, platform = match.groups()
                    try:
                        release_date = datetime.strptime(date_str, "%b %d, %Y").date()
                    except ValueError:
                        release_date = None

            game.update({
                "creator": creator,
                "release_date": release_date,
                "platform": platform
            })

            print(f"{game['title']}: {creator} | {release_date} | {platform}")

        except Exception as e:
            print(f"[ERROR] Failed to parse {game['title']}: {e}")
            game.update({
                "creator": "",
                "release_date": None,
                "platform": ""
            })

    driver.quit()
    save_cache(all_games)
    print(f"Scraped {len(all_games)} reviews and metadata")

# ------ 4. Clean data ------
import pandas as pd

df = pd.DataFrame(all_games)

# Convert the "date" column to datetime objects (makes date operations easier)
# Errors='coerce' will turn anything non-numeric (like "N/A" or "Unknown") into NaN.
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Convert "release_date" column to datetime too, but if conversion fails, set as NaT (missing datetime)
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

# Add a year column, which just holds the year of release as an integer
df["year"] = df["release_date"].dt.year

# Strip whitespace from strings
df["platform"] = df["platform"].str.strip()
df["creator"] = df["creator"].str.strip()

# Remove duplicate rows where "title" and "ign_score" columns are exactly the same
# Keeps the first occurrence, drops the rest
df = df.drop_duplicates(subset=["title", "ign_score"])

# Fill missing values with "unknown", note: only do this for string data
df["creator"] = df["creator"].fillna("Unknown")
df["platform"] = df["platform"].fillna("Unknown")
df["author"] = df["author"].fillna("Unknown")
df["title"] = df["title"].fillna("Unknown")

# run the RegExp text cleaner
df["creator"] = df["creator"].apply(clean_text)
df["author"] = df["author"].apply(clean_text)

# Errors='coerce' will turn anything non-numeric (like "N/A" or "Unknown") into NaN.
df["ign_score"] = pd.to_numeric(df["ign_score"], errors='coerce')

# Standardize column names and data (lowercase and replace spaces with underscores), note: only do this for string data
df.columns = df.columns.str.lower().str.replace(' ', '_')
# df['platform'] = df['platform'].str.lower().str.replace(' ', '_') --- do not do this, we will need to separate the strings later
df["creator"] = df["creator"].str.lower().str.replace(' ', '_')
df["author"] = df["author"].str.lower().str.replace(' ', '_')
df["title"] = df["title"].str.lower().str.replace(' ', '_')
df["link"] = df["link"].str.lower().str.replace(' ', '_')

# Drop NaN rows (works for non-string data)
df = df.dropna(subset=["ign_score", "date"])

# Remove the link column
df = df.drop(columns=["link"])

# Average score by developer (and how many games)
avg_by_dev_plus_count = df.groupby("creator")["ign_score"].agg(["mean", "count"]).sort_values(by="mean", ascending=False)
print(avg_by_dev_plus_count)

# Average score by author (and how many games)
avg_by_author_plus_count = df.groupby("author")["ign_score"].agg(["mean", "count"]).sort_values(by="mean", ascending=False)
print(avg_by_author_plus_count)

# Average score by year
# Group by the new "year" column, calculate the average "ign_score" for each year, and sort the results by year (chronological order)
avg_by_year = df.groupby("year")["ign_score"].mean().sort_index()
print(avg_by_year)

##########################################################

# ------ WHY THE ABOVE WORKS ------
# 1) You start with df, your full DataFrame — a big table containing all the data.

# 2) When you do df.groupby("creator"), pandas splits this big table into smaller tables, one for each unique creator. 
# This returns a GroupBy object — a kind of lazy object that hasn't done calculations yet.

# 3) Adding ["ign_score"] selects only the "ign_score" column from each smaller table, 
# turning those tables into single-column groups (Series).

# 4) Calling .agg(["mean", "count"]) tells pandas to calculate the average and the count of the scores inside each smaller group.

# 5) Then the .groupby-.agg chain is complete: pandas combines the results from all these smaller tables back into one new table, 
# where each row shows a creator’s average score and how many scores they have.

# 6) .agg() behaves differently:
# .agg() on a regular DataFrame (without .groupby()) calculates summary stats like mean or count for each column in the entire table.
# .agg() on a GroupBy object calculates those stats inside each smaller table (each creator’s data separately).

# CONCLUSION: Data Science is about LOGIC, PATH, and STRATEGIZING over OTHER DEVELOPMENT'S "GETTING IT DONE".

##########################################################

# ------ 5. Function to add genre metadata to all_games ------ 
# The model with just year of release, developer, writer and platform is not beating the baseline.
# Reason: IGN review scores mostly live in a tight 6–9 band, a “guess the median” baseline is surprisingly strong.
# Lesson: Before I do a ML project, I have to BELIEVE/THINK that there is a correlation, not just HOPE there is one.
# Solution: attach metadata to all_games: genre / steam tags, like I planned

# match names of games from all_games with the steam ids from the list
# print for how many games a matching steam id was found and for how many it was not found
import requests

def add_metadata(all_games_list, steam_app_list):
    # Build lookup dict
    steam_lookup = {app["name"].strip().lower(): app["appid"] for app in steam_app_list}
    found, not_found = 0, 0

    for game in all_games_list:
        title = game["title"].strip().lower()
        if title in steam_lookup:
            found += 1
            game["steam_appid"] = steam_lookup[title]
        else:
            not_found += 1
            game["steam_appid"] = None

    print(f"Matched {found} games, {not_found} not found.")
    return all_games_list

# --- Usage ---
steam_ids_url = "https://api.steampowered.com/ISteamApps/GetAppList/v0002/?key=STEAMKEY&format=json"
steam_list_resp = requests.get(steam_ids_url)
steam_app_list = steam_list_resp.json()["applist"]["apps"]

# for each of the games that a matching steam id was found, find more details in the steam_details_url 
# and attach it to all_games under .metadata object
import time    

def add_steam_metadata(all_games_list, sleep_sec=0.0, timeout=10):
    """
    Enrich games that already have 'steam_appid' using Steam appdetails.
    - Adds game['metadata'] with: genres, categories, release_date, developers, publishers.
    - Skips games without steam_appid OR if metadata already present.
    - Caches repeated appid lookups within this call.
    - sleep_sec: optional delay between calls (set >0 if you want to be gentle).
    - timeout: requests timeout in seconds.
    """
    session = requests.Session()
    cache = {}  # appid -> metadata dict (or {})
    found, missing, errors, already = 0, 0, 0, 0

    for game in all_games_list:
        appid = game.get("steam_appid")
        if not appid:
            # No steam match
            game.setdefault("metadata", {})
            missing += 1
            continue

        # ✅ Skip if already has metadata
        if game.get("metadata"):
            already += 1
            continue

        if appid in cache:
            game["metadata"] = cache[appid]
            found += 1 if cache[appid] else 0
            continue

        try:
            url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            # appdetails returns { "<appid>": {"success": bool, "data": {...}} }
            node = data.get(str(appid), {})
            if node.get("success") and isinstance(node.get("data"), dict):
                d = node["data"]
                meta = {
                    "steam_appid": appid,
                    "name": d.get("name"),
                    "release_date": d.get("release_date", {}).get("date"),
                    "developers": d.get("developers", []),
                    "publishers": d.get("publishers", []),
                    "genres": [g.get("description") for g in d.get("genres", []) if isinstance(g, dict)],
                    "categories": [c.get("description") for c in d.get("categories", []) if isinstance(c, dict)],
                }
                cache[appid] = meta
                game["metadata"] = meta
                found += 1
            else:
                cache[appid] = {}
                game["metadata"] = {}
        except Exception as e:
            errors += 1
            game["metadata"] = {}
            # Optional: print(f"[ERROR] appid {appid}: {e}")

        if sleep_sec:
            time.sleep(sleep_sec)

    print(f"Steam metadata: {found} enriched, {missing} without appid, {already} already had metadata, {errors} errors.")
    return all_games_list

# all_games = add_metadata(all_games, steam_app_list)
# all_games = add_steam_metadata(all_games) 
# save_cache(all_games)

# ================================================================
# TERMS
# ================================================================

# DataFrame (DF) - a big table object containing all the data
# Series - single-column groups
# Label - for a column - name, for a row - index

# Fit = train

# Preprocessing = turning non-numeric data into a numeric feature matrix that a machine-learning model can consume

# Variation - deviations from the mean (how far each value is from the average)

# Median - middle value in a sorted list of numbers. less sensitive to extreme values than the average (mean):
# [6, 7, 8, 9, 100] → median = 8, mean ≈ 26.
# So: I set aside 20% of the data that the model did NOT see, so that I can programmatically compare 
# the test data with predictions, and get: 

# - MAE (Mean Absolute Error) - sum of by how off predictions are, divided by total predictions = by how much the predictions 
# are off on average

# - RMSE (Root Mean Squared Error) - same, but sums are squared, then square rooted - to make big mistakes more punishable

# Baseline: I grab all features that you give me. For this particular set of features, the mean is this. Dummy of predictions.

# R^2 = 1− (Our model’s error​ / Baseline’s error), which means how much better are we than that. 
# Like, did we even achieve anything.

# baseline= very simple reference model (like predicting the median) used 
# One-hot / dummy variables — Turn a categorical value into 0/1 columns (one column per category)

# ================================================================
# SYNTAX
# ================================================================

# df["col"] - Column selection by label; returns a Series.
# df.loc[rows, cols] - like "get these rows and these columns and return a new DF. "valid" in parameter means "all"
# .clip(lower=a, upper=b) - Bound values to [a, b].
# .str.len() - Per-string length as integers.
# .apply(func) — Apply func elementwise over a Series (each cell gets passed to func).
# df.fillna(x) — Replace missing values with x.
# df.astype(type) — Cast column(s) to a new dtype (e.g., float).
# df.assign(newcol=...) — Returns a new DataFrame with an added/overwritten column (does not modify in place).
# List comprehension — [c for c in df.columns if c.startswith("g__")] filters column names by a rule.

# ================================================================
# PREP FEATURES + METADATA + TIME SPLIT + EARLY-STOP XGB TRAIN
# ================================================================
# `import` brings modules into scope so you can reference their names.
import re
import numpy as np
import pandas as pd
# from pkg import Name imports specific symbols into the current namespace.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------- A) Title clues (fix warnings, handle NA) ----------
#  In pandas, a “DataFrame” (df) is a 2D table; a “Series” is a 1D column.
#         Access a column by key: df["title"] returns a Series.
#         .fillna(x) replaces missing values (NaN/NaT) with x.
t = df["title"].fillna("")

#  Creating a new column: df["has_colon"] = <Series-like>.
#         .str is the pandas string accessor for vectorized string ops on a Series.
#         .str.contains(pattern, ...) returns booleans where pattern matches.
#         na=False treats missing values as False instead of propagating NaN.
df["has_colon"] = t.str.contains(":", na=False)
df["has_num"] = t.str.contains(r'\b(?:\d+|i{1,3}|iv|v|vi{0,3}|ix|x)\b',
                               case=False, regex=True, na=False)
df["is_dlc"]  = t.str.contains(r'\b(?:dlc|expansion|episode|chapter|pack|remaster|definitive)\b',
                               case=False, regex=True, na=False)

#  .dt is the datetime accessor on Series of dtype datetime64.
#         Subtracting two datetime Series yields a Timedelta; .dt.days extracts ints.
df["review_lag_days"] = (df["date"] - df["release_date"]).dt.days
#  .clip(lower=a, upper=b) limits values to [a, b].
df["review_lag_days"] = df["review_lag_days"].clip(lower=-365, upper=365)

#  .str.len gives string lengths per row; result is numeric.
df["title_len"] = t.str.len().fillna(0)

# ---------- B) Platform one-hots (and ensure expected columns exist) ----------
#  .str.split(",") splits each string into a Python list; results in dtype=object.
df["platform"] = df["platform"].fillna("").str.split(",")

#  .apply(func) calls func on each element of the Series (row-wise for a column).
#         Lambda here trims whitespace from each list element.
df["platform"] = df["platform"].apply(lambda xs: [s.strip() for s in xs] if isinstance(xs, list) else [])

#  Applying len to each list gives per-row counts.
df["platform_count"] = df["platform"].apply(len)

# Goal: turn the "platform" column (list of platforms per game) into 0/1 columns.
# Step 1: .str.join('|') → each row's list ["PC","Xbox"] becomes "PC|Xbox".
# Step 2: .str.get_dummies(sep="|") → for each unique token (e.g., "PC", "Xbox"), 
#         create a dummy column: 1 if the token appears in that row, else 0.
#         (This is multi-label one-hot encoding.)
# Step 3: df.join(...) → append those new dummy columns to the original df,
#         aligning rows by index.
df = df.join(df["platform"].str.join('|').str.get_dummies())

#  A "one-hot" or "dummy" column is a binary 0/1 indicator column used to
#         numerically represent categories. A set of k categories becomes k columns.

#  Define a Python list of expected columns so model inputs are consistent.
expected_platforms = [
    "Google Stadia","Nintendo 3DS","Nintendo Switch","PC",
    "PlayStation 4","PlayStation 5","PlayStation VR","PlayStation Vita",
    "Wii U","Xbox One","Xbox Series X/S"
]
#  If a column doesn't exist, create it with zeros (keeps feature schema fixed).
for col in expected_platforms:
    if col not in df.columns:
        df[col] = 0

# ---------- C) Steam metadata -> genre/category dummies ----------
# Here we want a helper that always returns a list.
# `isinstance(x, list)` → checks if x is already a Python list.
# If it is, return it unchanged; otherwise return an empty list [].
def safe_list(x): return x if isinstance(x, list) else []

#  Building a list of dicts, then DataFrame(...) turns rows of dicts into a table.
meta_rows = []
for g in all_games:
    #  dict.get(key, default) fetches a value or returns default if missing.
    md = g.get("metadata", {}) or {}
    meta_rows.append({
        "title": g.get("title", ""),
        "steam_appid": g.get("steam_appid"),
        "steam_release_date": md.get("release_date"),
        "steam_genres": safe_list(md.get("genres")),
        "steam_categories": safe_list(md.get("categories")),
    })
meta_df = pd.DataFrame(meta_rows)

#  .astype(str) casts dtype; chain .str ops.
#         .str.lower() lowercases; .str.replace(old, new, regex=False) literal replace.
meta_df["title"] = meta_df["title"].astype(str).str.lower().str.replace(' ', '_', regex=False)

#  pd.to_datetime(..., errors="coerce") parses strings; bad parse -> NaT (missing).
meta_df["steam_release_date"] = pd.to_datetime(meta_df["steam_release_date"], errors="coerce")

#  Helper that turns list values into a single pipe ("|") separated string per row.
def list_to_pipe(series):
    return series.apply(lambda lst: "|".join(lst) if isinstance(lst, list) and len(lst) else "")

#  .str.get_dummies expands pipe-separated values into dummy columns.
genre_dummies = list_to_pipe(meta_df["steam_genres"]).str.get_dummies(sep="|")
cat_dummies   = list_to_pipe(meta_df["steam_categories"]).str.get_dummies(sep="|")

TOP_N_GENRES, TOP_N_CATS = 30, 30
# genre_dummies and cat_dummies are wide binary matrices:
# - Each column is a genre/category (e.g., "Action", "RPG").
# - Each row is a game (1 if the game has that genre, else 0).
# If there are too many columns, we keep only the most common ones.

# genre_dummies.shape[1] → number of genre columns.
# genre_dummies.sum() → sum down each column = how many games have that genre.
# .sort_values(ascending=False).head(30).index → top 30 genre names by frequency.
# genre_dummies[top_genres] → keep only those columns.
if genre_dummies.shape[1] > TOP_N_GENRES:
    top_genres = genre_dummies.sum().sort_values(ascending=False).head(TOP_N_GENRES).index
    genre_dummies = genre_dummies[top_genres]
# Same logic for categories (multiplayer, singleplayer, VR support, etc.)
if cat_dummies.shape[1] > TOP_N_CATS:
    top_cats = cat_dummies.sum().sort_values(ascending=False).head(TOP_N_CATS).index
    cat_dummies = cat_dummies[top_cats]

# Add prefixes so we can tell these columns apart later:
# "g__Action" vs "c__Multiplayer".
# Without prefixes, a genre "Racing" and a category "Racing" would collide.
#  .add_prefix("g__") prepends a string to every column name.
genre_dummies = genre_dummies.add_prefix("g__")
cat_dummies   = cat_dummies.add_prefix("c__")

#  pd.concat([...], axis=1) concatenates dataframes column-wise.
meta_df = pd.concat([meta_df[["title","steam_appid","steam_release_date"]],
                     genre_dummies, cat_dummies], axis=1)

#  df.merge(right, on="col", how="left") SQL-style join; keeps all left rows.
df = df.merge(meta_df, on="title", how="left")

#  .fillna(other_series) fills missing with aligned values from another Series.
df["release_date"] = df["release_date"].fillna(df["steam_release_date"])

#  .dt.year extracts the calendar year as integers.
df["year"] = df["release_date"].dt.year

#  Recomputing with same .dt days, then .clip as before.
df["review_lag_days"] = (df["date"] - df["release_date"]).dt.days.clip(-365, 365)

#  List comprehension over column names; .startswith("g__") filters by prefix.
genre_cols = [c for c in df.columns if c.startswith("g__")]
cat_cols   = [c for c in df.columns if c.startswith("c__")]

#  Selecting multiple columns: df[cols].fillna(0).astype(float) applies ops vectorized.
df[genre_cols + cat_cols] = df[genre_cols + cat_cols].fillna(0).astype(float)

# ---------- D) Build ML matrix ----------
#  “categorical” vs “numeric”: Categorical features are discrete labels that need
#         encoding; numeric are numbers you can pass directly to models.
categorical = ["creator", "author"]

#  *list unpacking inserts elements of lists into another list literal.
numeric = [
    "year","has_colon","has_num","is_dlc","review_lag_days",
    "platform_count","title_len",
    *expected_platforms, *genre_cols, *cat_cols
]

#  .notna() gives boolean mask of non-missing values per cell;
#         .all(axis=1) requires all columns in the row to be non-missing.
valid = df[categorical + numeric + ["ign_score"]].notna().all(axis=1)

#  df.loc[rows, cols] is label-based selection.
#         Here we filter rows by boolean mask and pick feature columns.
X = df.loc[valid, categorical + numeric].copy()
y = df.loc[valid, "ign_score"].astype(float).copy()

#  .astype(float) converts dtypes; useful before model ingestion.
X[numeric] = X[numeric].astype(float)

# ---------- E) Time-based split (hold out newest ~10%) ----------
# We want to split train/test by review date (simulate predicting future).
# X is features; df.loc[valid, "date"] are the dates.
# .assign(_d=...) → make a temporary column "_d" containing the dates.
# .sort_values("_d") → sort rows chronologically by review date.
# .index → get the sorted row indices (not the values).
ordered_idx = X.assign(_d=df.loc[valid, "date"]).sort_values("_d").index

# first 90% indices for train, last 10% for test.
#  int(len(...) * 0.90) computes position of 90% split point.
cut = int(len(ordered_idx) * 0.90)
train_idx, test_idx = ordered_idx[:cut], ordered_idx[cut:]

#  Indexing by index arrays returns corresponding subsets.
X_train_raw, X_test_raw = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

# Handle preprocessing for creator and author - final non-numeric columns
# ColumnTransformer allows different preprocessing per column group.
# Each tuple is ("name", transformer, [columns]).
# - ("cat", OneHotEncoder(...), categorical) → one-hot encode the categorical columns.
# - ("num", "passthrough", numeric) → keep numeric columns as they are.
# So the output is one big numeric feature matrix: one-hot columns + raw numeric columns.
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
    ("num", "passthrough", numeric)
])

# pre is the column transformer
# so fit_transform and transform are continuations of ColumnTransformer and are just syntax to apply it

# pre.fit_transform(X_train_raw):
# - .fit() → learns how to transform (e.g., discovers all category levels).
# - .transform() → actually encodes the rows.
# Combined: .fit_transform() does both at once.
X_train = pre.fit_transform(X_train_raw)
# pre.transform(X_test_raw):
# - Use the same learned mappings to transform test rows
#   (no re-fitting, so categories are consistent between train/test).
X_test  = pre.transform(X_test_raw)

# ---------- F) XGBoost (version-safe early stopping)----------
import xgboost as xgb
import numpy as np

#  try/except provides a fallback path if import/name not available.
#         Here we alias root_mean_squared_error to RMSE with a fallback to
#         mean_squared_error(squared=False) on older sklearn.
try:
    from sklearn.metrics import root_mean_squared_error as _rmse
    def RMSE(y_true, y_pred): return _rmse(y_true, y_pred)
except Exception:
    from sklearn.metrics import mean_squared_error
    def RMSE(y_true, y_pred): return mean_squared_error(y_true, y_pred, squared=False)

#  def fit_xgb_version_safe(...): defines a helper function.
#         Docstring """...""" documents the function.
def fit_xgb_version_safe(Xtr, ytr, Xva, yva, es_rounds=100, eval_metric="rmse"):
    """
    Prefer new API (constructor args), fall back to old API (fit kwargs), then callbacks.
    """
    #  XGBRegressor(...) constructs a gradient-boosted trees regressor.
    #         Common args:
    #         - n_estimators: number of boosting rounds/trees
    #         - learning_rate: shrinkage per tree
    #         - max_depth, min_child_weight, subsample, colsample_bytree: tree/row/col controls
    #         - reg_lambda/reg_alpha: L2/L1 regularization
    #         - random_state: RNG seed; n_jobs: parallelism
    #         - objective="reg:squarederror": regression loss
    #         Early stopping:
    #         - early_stopping_rounds: stop if eval metric doesn't improve
    #         - eval_metric: which metric to monitor
    #         - eval_set=[(X_val, y_val)]: data on which to monitor performance
    try:
        model = xgb.XGBRegressor(
            n_estimators=4000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            early_stopping_rounds=es_rounds,
            eval_metric=eval_metric
        )
        #  .fit(X, y, eval_set=[...]) trains the model; eval_set enables early stopping.
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)])  # <- ONLY eval_set here
        return model
    except TypeError:
        pass

#  Call the helper to get a trained model.
model = fit_xgb_version_safe(X_train, y_train, X_test, y_test, es_rounds=100, eval_metric="rmse")

# Evaluate vs a median baseline
#  .predict(X) runs the model to get numeric predictions.
y_pred = model.predict(X_test)

#  np.full_like(y_test, value, dtype=...) makes an array same shape as y_test,
#         filled with a constant — here the training median, a simple baseline.
y_base = np.full_like(y_test, np.median(y_train), dtype=float)

#  Metrics — MAE = mean absolute error; RMSE = root mean squared error;
#         R^2 = coefficient of determination (1 is perfect, can be negative).
from sklearn.metrics import mean_absolute_error, r2_score
print(f"Test MAE – Baseline: {mean_absolute_error(y_test, y_base):.3f}")
print(f"Test MAE – XGB     : {mean_absolute_error(y_test, y_pred):.3f}")
print(f"Test RMSE – Baseline: {RMSE(y_test, y_base):.3f}")
print(f"Test RMSE – XGB     : {RMSE(y_test, y_pred):.3f}")
#  f-strings evaluate expressions inside {...} and format with .3f decimals.
print(f"Test R² – XGB       : {r2_score(y_test, y_pred):.3f}")

# Optional: top importances with feature names
try:
    #  Named transformer retrieval: pre.named_transformers_["cat"].
    #         .get_feature_names_out(categorical) returns output column names after OHE.
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(categorical)
    num_names = np.array(numeric)
    #  np.concatenate joins arrays end-to-end.
    feat_names = np.concatenate([cat_names, num_names])

    #  getattr(obj, "attr", default) safely accesses attribute if present.
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        #  zip pairs names with values; sorted(..., key=..., reverse=True) sorts.
        top = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:25]
        print("\nTop features:")
        #  f-string width specifier {name:35s} pads/truncates to 35 chars.
        for name, val in top:
            print(f"{name:35s} {val:.4f}")
except Exception:
    pass
