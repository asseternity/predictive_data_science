# Idea: 
# I built a pipeline to scrape and structure 2,000 IGN reviews, 
# enriched the data using an API, 
# and trained a model to understand scoring trends. 
# It turns out IGN favors RPGs and penalizes racing games.
# Can't use API because it only serves 10 reviews and limits to 100 requests

# Questions:
# - which genres/platforms get higher scores?
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
        if len(data) >= 200:
            return data
        return None
    return None

def save_cache(data):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, indent=2)
    print(f"Saved {len(data)} reviews to cache.")

# ------ 2. Scraping Opencritic's IGN page with BeautifulSoup ------
# Note: to not hammer any servers, I will use delays (e.g., time.sleep(2)) if needed.
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

cached = load_cache()
if cached:
    for item in cached:
        item["date"] = datetime.fromisoformat(item["date"]).date()
    all_games = cached
    print(f"Loaded {len(all_games)} reviews from cache.")
else:
    options = Options()
    options.add_argument("--headless") # run in background
    driver = webdriver.Chrome(options=options)

    all_games = []

    for page in range(1, 20):
        url = f"https://opencritic.com/outlet/56/ign?page={page}"
        driver.get(url)
        time.sleep(3) # wait for JS to render
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find each HTML review block
        review_blocks = soup.find_all("div", class_="review-row")
        print(f"Found review blocks: {len(review_blocks)}")

        # See the HTML for one block with BeautifulSoup's prettify
        print(review_blocks[0].prettify())

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

            all_games.append({
                "title": title,
                "ign_score": score,
                "date": date,
                "author": author,
            })
            print(f"{date} | {author} | {title} | {score}")

    save_cache(all_games)
    print(f"Scraped {len(all_games)} reviews")
    driver.quit()

# ------ 3. Getting Metadata (Genre, Platform, etc.) from https://www.igdb.com/api ------


# ------ 4. Clean data ------


# ------ 5. Combine → Train ML with XGBoost ------ 
