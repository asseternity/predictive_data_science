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
            link_element = soup.find('a', href=re.compile(r'^/game/\d+/'))
            link = f"https://opencritic.com/{link_element['href']}" if link_element else None

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
df["date"] = pd.to_datetime(df["date"])

# Convert "release_date" column to datetime too, but if conversion fails, set as NaT (missing datetime)
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

# Strip whitespace from strings
df["platform"] = df["platform"].str.strip()
df["creator"] = df["creator"].str.strip()

# Remove duplicate rows where "title", "ign_score", and "date" columns are exactly the same
# Keeps the first occurrence, drops the rest
df = df.drop_duplicates(subset=["title", "ign_score", "date"])

# Average score by developer (and how many games)
avg_by_dev_plus_count = df.groupby("creator")["ign_score"].agg(["mean", "count"]).sort_values(by="mean", ascending=False)
print(avg_by_dev_plus_count)

# Average score by author (and how many games)
avg_by_author_plus_count = df.groupby("author")["ign_score"].agg(["mean", "count"]).sort_values(by="mean", ascending=False)
print(avg_by_author_plus_count)

# Average score by year
# Extract the year from the "release_date" datetime column into a new integer column "year"
df["year"] = df["release_date"].dt.year
# Group by the new "year" column, calculate the average "ign_score" for each year, and sort the results by year (chronological order)
avg_by_year = df.groupby("year")["ign_score"].mean().sort_index()
print(avg_by_year)

# *WHY THE ABOVE WORKS*
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

# ------ 5. Combine → Train ML with XGBoost ------ 

# A. Prepare the data
# 1) make sure data is clean, replace unknown rows with a default "unknown"
# 2) replace date strings with just a year
# 3) assign each developer and author an ID using LabelEncoder
# 4) Platforms → since a game can be on multiple platforms, 
# make one column per platform: PC, PS5, Xbox, etc., and mark 1 if it’s on that platform.
# use pandas' get_dummies
# 5) Title --- string, so that's harder
# Let's congregate title into just title's length in characters

# B. Split into training data vs test data. Train on 80% of data. Test on remaining 20%.
# Use sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=42) so that results are reproducible.

# C. Choose a model: 
# Linear Regression: tries to draw a straight-line relationship between your inputs (year, developer, etc.) and the score.
# Decision Tree: splits the data based on rules like “if developer = X, go left; else go right.”

# D. Train the model
# E. See how well it does by running it on the test set (compare its prediction vs the real scores)
# F. Improve gradually. If the error is too big: Add better features, Try a stronger model or Tune settings