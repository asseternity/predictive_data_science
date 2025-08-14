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

# ------ 5. Add clues ------ 
# title-based clues (you lowercased already)
t = df["title"].fillna("")
df["has_colon"] = t.str.contains(":")
df["has_num"]   = t.str.contains(r"\b(\d+|ii|iii|iv|v|vi|vii|viii|ix|x)\b")
df["is_dlc"]    = t.str.contains(r"\b(dlc|expansion|episode|chapter|pack|remaster|definitive)\b")

# review timing vs release
df["review_lag_days"] = (df["date"] - df["release_date"]).dt.days

# how many platforms (proxy for budget/reach)
df["platform_count"] = df["platform"].apply(lambda lst: len(lst) if isinstance(lst, list) else 0)

# (optional) title length
df["title_len"] = df["title"].fillna("").str.len()

# ------ 6. Prepare developer, author and platform columns ------ 
# A. Assign each developer and author an ID using LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Encode developers (creators)
creator_encoder = LabelEncoder()
df['creator_id'] = creator_encoder.fit_transform(df['creator'])

# Encode authors
author_encoder = LabelEncoder()
df['author_id'] = author_encoder.fit_transform(df['author'])

# B. For platforms (since a game can be on multiple platforms) make one column per platform and mark 1 if it’s on that platform
# Split each platform cell into a list
df["platform"] = df["platform"].str.split(",")

# Strip whitespace from each platform in those lists
# - .apply() = pandas func to take each element of a Series / each row/column of a DataFrame and run it through a function
# - in our case, we take a column of a DF, so whatever is in () of apply will run through the list of platforms (separated from a string above) 
# - .apply() takes a FUNC as an argument, but that FUNC must have an argument of its own, and .apply's puts its target into that INNER argument
def strip_row_whitespaces(row):
    stripped_row = []
    for string in row:
        cleaned_string = string.strip()
        stripped_row.append(cleaned_string)
    return stripped_row  
df["platform"] = df["platform"].apply(strip_row_whitespaces)

# One-hot encode into separate columns
df = df.join(df["platform"].str.join('|').str.get_dummies())

# ------ 7. Start doing sample ML ------  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

# GOAL: predict ign_score given a year, a developer, a writer and a list of platforms

# y = target we want ML to predict
y = df["ign_score"]

# features = input variables the model uses to make a prediction
# print([c for c in df.columns]) # list the columns to find specific ones
categorical = ["creator", "author"]
numeric = [
    "year","has_colon","has_num","is_dlc",
    "review_lag_days","platform_count","title_len",
    "Google Stadia","Nintendo 3DS","Nintendo Switch","PC",
    "PlayStation 4","PlayStation 5","PlayStation VR","PlayStation Vita",
    "Wii U","Xbox One","Xbox Series X/S"
]
X = df[categorical + numeric]

# Drop rows with any missing features or missing target
# .all(axis=1) → all means it collapses rows into a single True if NO NaNs at all, False if at least 1
# .notna() → returns a DF of True where the cell is not missing (NaN) and False where it is missing.
valid = df[categorical + numeric + ["ign_score"]].notna().all(axis=1)
X = df.loc[valid, categorical + numeric].copy()
y = df.loc[valid, "ign_score"].copy()

# Ensure only numeric features are numeric; leave strings for OneHotEncoder
X[numeric] = X[numeric].astype(float)

# Create a test / train split syntax:
RANDOM_STATE = 42 # random integer
np.random.seed(RANDOM_STATE) # reproducibility: sets NumPy’s random number gen to start in the same state to get same sequence each time
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=RANDOM_STATE,
    shuffle=True
)
# alternative to the above: “predict future from past” setup - split by year instead of random

# --------- TERMS --------
# Fit = train
# Variation - deviations from the mean (how far each value is from the average)
# Median - middle value in a sorted list of numbers. less sensitive to extreme values than the average (mean):
# [6, 7, 8, 9, 100] → median = 8, mean ≈ 26.
# So: I set aside 20% of the data that the model did NOT see, so that I can programmatically compare the test data with predictions, and get: 
# - MAE (Mean Absolute Error) - sum of by how off predictions are, divided by total predictions = by how much the predictions are off on average
# - RMSE (Root Mean Squared Error) - same, but sums are squared, then square rooted - to make big mistakes more punishable
# Baseline: I grab all features that you give me. For this particular set of features, the mean is this. Dummy of predictions.
# R^2 = 1− (Our model’s error​ / Baseline’s error), which means how much better are we than that. Like, did we even achieve anything.

# baseline/dummy = very simple reference model (like predicting the median) used 
# to check whether your real model is actually learning something useful. 
# If you can’t beat the baseline, revisit data cleaning or feature choices
baseline_model = DummyRegressor(strategy="median")
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)

# Models: “Classifier” predicts categories; “Regressor” predicts a number 
# Here, we make, fit (train) XGBoost "regressor" and make it predict
our_model = XGBRegressor(random_state=RANDOM_STATE)
pre = ColumnTransformer([ ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical), ("num", "passthrough", numeric) ])
pipe = make_pipeline(pre, our_model)
pipe.fit(X_train, y_train)
our_predictions = pipe.predict(X_test)

# # Compare baseline and real model with MAE, RMSE and R^2
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
mae_ours = mean_absolute_error(y_test, our_predictions)
rmse_baseline = root_mean_squared_error(y_test, baseline_predictions)
rmse_ours = root_mean_squared_error(y_test, our_predictions)
r2_ours = r2_score(y_test, our_predictions)

print("Baseline MAE:", mae_baseline)
print("Model MAE:", mae_ours)
print("Baseline RMSE:", rmse_baseline)
print("Model RMSE:", rmse_ours)
print("Model R²:", r2_ours)

# Ask it to make a prediction
# Find ids
print(list(enumerate(creator_encoder.classes_)))
print(list(enumerate(author_encoder.classes_)))
# Pretend you want to predict for this game
new_game = pd.DataFrame([{
    "creator": "obsidian_entertainment",
    "author":  "luke_reilly",
    "year": 2025,
    "review_lag_days": 30,
    "title_len": 5,
    "platform_count": 1,
    "has_colon": True, "has_num": False, "is_dlc": False,
    "Google Stadia": 0, "Nintendo 3DS": 0, "Nintendo Switch": 1, "PC": 0,
    "PlayStation 4": 0, "PlayStation 5": 0, "PlayStation VR": 0, "PlayStation Vita": 0,
    "Wii U": 0, "Xbox One": 0, "Xbox Series X/S": 0
}])
predicted_score = pipe.predict(new_game)
print("Predicted IGN score:", predicted_score[0])

# ------ 8. Add Features ------ 
# The model with just year of release, developer, writer and platform is not beating the baseline.
# Reason: IGN review scores mostly live in a tight 6–9 band, a “guess the median” baseline is surprisingly strong.
# Lesson: Before I do a ML project, I have to BELIEVE/THINK that there is a correlation, not just HOPE there is one.
# Solution: attach metadata to all_games: genre / steam tags, like I planned