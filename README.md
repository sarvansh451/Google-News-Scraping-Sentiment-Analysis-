# Google-News-Scraping-Sentiment-Analysis-
# Google News Scraper with Sentiment Analysis

A Python project that automatically fetches Google News articles for a keyword, extracts the full content, and performs sentiment analysis (positive, negative, neutral) on the news.

## Features

- Fetches latest news articles from Google News RSS.  
- Resolves Google News redirect links to get the real article URL.  
- Extracts full article content using Newspaper3k.  
- Performs sentiment analysis using NLTK VADER.  
- Saves results in JSON format for easy analysis.  
- Can fetch multiple articles in parallel for speed.

## Requirements

- Python 3.8+  
- Libraries: `requests`, `beautifulsoup4`, `newspaper3k`, `feedparser`, `nltk`  
- NLTK data for sentiment analysis (`vader_lexicon` and `punkt`) are automatically downloaded in the code using:

```python
import nltk
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
```
### How to Run

Run the script from the command line:

python scrape_news.py --q "Artificial Intelligence" --hl en-US --gl US --max 10 --workers 3 --output output.json

Arguments

--q : Search query (required)

--hl : Language (default: en-US)

--gl : Country (default: US)

--max : Maximum number of news articles to fetch (default: 10)

--workers : Number of parallel threads (default: 3)

--output : Output JSON file name (default: output.json)
