# Game Review Predictor

## Summary
End-to-end machine learning project predicting video game review scores from OpenCritic and Steam metadata. Demonstrates full data engineering and modeling lifecycle.

## Tech
- **Language:** Python  
- **Libraries:** pandas, XGBoost, scikit-learn  
- **Data Sources:** OpenCritic, Steam API  
- **Scraping:** Selenium, BeautifulSoup  
- **Evaluation:** MAE, RMSE, R²  
- **Repro:** deterministic caching, versioned artifacts

## Highlights
- Scraped and cleaned 6,600+ reviews from multiple outlets  
- Enriched dataset with release year, platform, genre, and developer metadata  
- Feature engineering for time lag, genre popularity, author mean scores  
- Trained XGBoost regression model with temporal validation split  
- Benchmarked against median and linear regression baselines

## Pipeline
1. Scrape and cache reviews (OpenCritic + Steam API)  
2. Clean text, merge metadata, derive new features  
3. Encode categorical features and normalize numerics  
4. Train XGBoost with early stopping and grid search  
5. Evaluate and interpret feature importances  

## Results
| Metric | Baseline | Model |
|--------|-----------|--------|
| MAE | 0.61 | **0.45** |
| RMSE | 0.79 | **0.58** |
| R² | 0.41 | **0.67** |

## Testing
- Unit tests for feature builders and validators  
- Data schema contract tests to prevent drift  

## Artifacts
- `/notebooks/pipeline.ipynb`  
- `/data/processed/games.csv`  
- `/reports/metrics.md`  
- `/models/xgb_model.json`  

## Notebook: https://www.kaggle.com/code/assetn/video-game-review-score-predictor
