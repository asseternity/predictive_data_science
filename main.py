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


# Convert dates
df["date"] = pd.to_datetime(df["date"])
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

# Strip whitespace from strings
df["platform"] = df["platform"].str.strip()
df["creator"] = df["creator"].str.strip()

# Drop exact duplicates (some look like they are duplicated reviews)
df = df.drop_duplicates(subset=["title", "ign_score", "date"])

top_games = df.sort_values("ign_score", ascending=False).head(10)
print(top_games[["title", "ign_score", "date"]])

print(df["creator"].value_counts().head(10))
print(df["author"].value_counts().head(10))

# ------ 5. Combine → Train ML with XGBoost ------ 
