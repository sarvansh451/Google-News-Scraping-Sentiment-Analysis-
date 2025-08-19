import argparse
import concurrent.futures
import feedparser
import json
import re
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from newspaper import Article
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ✅ FastAPI imports
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ✅ New import
import trafilatura  

# Ensure NLTK data is available
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# ✅ FastAPI app instance
app = FastAPI(
    title="Google News Scraper API",
    description="A powerful API to scrape Google News with sentiment analysis",
    version="1.0.0"
)

# ✅ Pydantic models for API
class NewsSearchRequest(BaseModel):
    query: str
    language: str = "en-US"
    country: str = "US"
    max_results: int = 10
    workers: int = 3

class NewsArticle(BaseModel):
    title: str
    link: str
    published: Optional[str]
    source: Optional[str]
    content: Optional[str]
    article_text: str
    article_title: Optional[str]
    sentiment: Optional[str]
    sentiment_score: Optional[float]

class NewsResponse(BaseModel):
    success: bool
    total_results: int
    articles: List[NewsArticle]
    query: str


@dataclass
class NewsItem:
    title: str
    link: str
    published: Optional[str]
    source: Optional[str]
    content: Optional[str] = None
    article_text: str = ""
    article_title: Optional[str] = "Google News"
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None


def build_google_news_rss_url(q: str, hl: str = "en-US", gl: str = "US") -> str:
    return f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl={hl}&gl={gl}&ceid={gl}:en"


def get_real_article_url(google_news_url: str) -> Optional[str]:
    if not google_news_url or 'news.google.com' not in google_news_url:
        return google_news_url
    try:
        resp = requests.get(google_news_url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        c_wiz = soup.select_one('c-wiz[data-p]')
        if not c_wiz:
            return None
        data = c_wiz.get('data-p')
        obj = json.loads(data.replace('%.@.', '["garturlreq",'))
        payload = {'f.req': json.dumps([[['Fbv4je', json.dumps(obj[:-6] + obj[-2:]), 'null', 'generic']]])}
        headers = {'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
                   'user-agent': 'Mozilla/5.0'}
        url = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        array_string = json.loads(response.text.replace(")]}'", ""))[0][2]
        return json.loads(array_string)[1]
    except:
        return None


def extract_summary_text(summary_html: str) -> str:
    soup = BeautifulSoup(summary_html, "html.parser")
    for a_tag in soup.find_all("a"):
        a_tag.replace_with(a_tag.get_text())
    return re.sub(r'\s+', ' ', soup.get_text(separator=' ', strip=True)).strip()


# ✅ Updated scrape_article_content with multi-fallback
def scrape_article_content(url: str, timeout: float = 15.0) -> dict:
    try:
        # First try newspaper3k
        article = Article(url)
        article.download()
        article.parse()
        if article.text.strip():
            return {"article_text": article.text.strip(), "article_title": article.title}
    except:
        pass

    try:
        # Fallback: trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted:
                return {"article_text": extracted.strip(), "article_title": None}
    except:
        pass

    try:
        # Last fallback: BeautifulSoup paragraph join
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        title = soup.title.string if soup.title else None
        return {"article_text": text.strip(), "article_title": title}
    except:
        return {"article_text": "", "article_title": None}


def fetch_google_news(rss_url: str, max_items: int = 10) -> List[NewsItem]:
    feed = feedparser.parse(rss_url)
    items = []
    for i, entry in enumerate(feed.entries[:max_items]):
        items.append(NewsItem(
            title=entry.get("title", ""),
            link=entry.get("link", ""),
            published=entry.get("published", ""),
            source=getattr(entry.source, 'title', "") if hasattr(entry, 'source') else "",
            content=entry.get("summary", "")
        ))
    return items


class SentimentAnalyzer:
    def __init__(self):  # ✅ Fixed: was _init_
        try:
            self._analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon")
            self._analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> tuple:
        text = text or ""
        scores = self._analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            return "positive", compound
        elif compound <= -0.05:
            return "negative", abs(compound)
        return "neutral", abs(compound)


def process_news_item(item: NewsItem) -> NewsItem:
    real_url = get_real_article_url(item.link)
    if real_url and real_url != item.link:
        article_data = scrape_article_content(real_url)
        item.article_text = article_data["article_text"]
        item.article_title = article_data["article_title"]
        item.link = real_url
    return item


def analyze_sentiment(items: List[NewsItem]) -> List[NewsItem]:
    analyzer = SentimentAnalyzer()
    for item in items:
        sentiment, score = analyzer.analyze(item.article_text or item.title)
        item.sentiment, item.sentiment_score = sentiment, round(score, 4)
    return items


def scrape_google_news(query: str, language: str = "en-US", country: str = "US", max_results: int = 10, workers: int = 3) -> List[dict]:
    rss_url = build_google_news_rss_url(query, hl=language, gl=country)
    news_items = fetch_google_news(rss_url, max_items=max_results)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        news_items = list(executor.map(process_news_item, news_items))
    news_items = analyze_sentiment(news_items)
    return [asdict(item) for item in news_items]


def to_dataframe(news_items: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(news_items, columns=[
        "title", "link", "published", "source", "article_text", "article_title", "sentiment", "sentiment_score"
    ])


# ✅ FastAPI Routes

@app.get("/", tags=["Info"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "Google News Scraper API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This info page",
            "GET /health": "Health check",
            "GET /search": "Search news with query parameters",
            "POST /search": "Search news with JSON body",
            "GET /docs": "API documentation"
        }
    }

@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Google News Scraper API"}

@app.get("/search", response_model=NewsResponse, tags=["News"])
async def search_news_get(
    q: str = Query(..., description="Search query", example="artificial intelligence"),
    hl: str = Query("en-US", description="Language code", example="en-US"),
    gl: str = Query("US", description="Country code", example="US"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    workers: int = Query(3, ge=1, le=10, description="Number of parallel workers")
):
    """
    Search Google News with query parameters
    
    - **q**: Search query (required)
    - **hl**: Language (default: en-US)
    - **gl**: Country (default: US)
    - **max_results**: Number of articles to fetch (1-50, default: 10)
    - **workers**: Parallel processing workers (1-10, default: 3)
    """
    try:
        results = scrape_google_news(q, hl, gl, max_results, workers)
        articles = [NewsArticle(**article) for article in results]
        
        return NewsResponse(
            success=True,
            total_results=len(articles),
            articles=articles,
            query=q
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@app.post("/search", response_model=NewsResponse, tags=["News"])
async def search_news_post(request: NewsSearchRequest):
    """
    Search Google News with JSON request body
    
    Send a POST request with JSON body containing search parameters.
    """
    try:
        results = scrape_google_news(
            request.query, 
            request.language, 
            request.country, 
            request.max_results, 
            request.workers
        )
        articles = [NewsArticle(**article) for article in results]
        
        return NewsResponse(
            success=True,
            total_results=len(articles),
            articles=articles,
            query=request.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@app.get("/search/csv", tags=["Export"])
async def search_news_csv(
    q: str = Query(..., description="Search query"),
    hl: str = Query("en-US", description="Language code"),
    gl: str = Query("US", description="Country code"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    workers: int = Query(3, ge=1, le=10, description="Number of parallel workers")
):
    """Export search results as CSV"""
    try:
        results = scrape_google_news(q, hl, gl, max_results, workers)
        df = to_dataframe(results)
        csv_content = df.to_csv(index=False)
        
        from fastapi.responses import Response
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=news_{q.replace(' ', '_')}.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")


# ✅ CLI functionality (original script behavior)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google News Scraper - CLI or API mode")
    parser.add_argument("--mode", choices=["cli", "api"], default="api", help="Run mode")
    parser.add_argument("--q", help="Search query (CLI mode)")
    parser.add_argument("--hl", default="en-US", help="Language")
    parser.add_argument("--gl", default="US", help="Country")
    parser.add_argument("--max", type=int, default=10, help="Max results")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    parser.add_argument("--output", default="output.json", help="Output file (CLI mode)")
    parser.add_argument("--format", choices=["csv", "json"], default="json", help="Output format (CLI mode)")
    parser.add_argument("--host", default="127.0.0.1", help="API host (API mode)")
    parser.add_argument("--port", type=int, default=8000, help="API port (API mode)")
    args = parser.parse_args()

    if args.mode == "cli":
        if not args.q:
            print("❌ Query is required for CLI mode. Use --q 'your search query'")
            exit(1)
            
        print(f"🔍 Searching for: {args.q}")
        results = scrape_google_news(args.q, args.hl, args.gl, args.max, args.workers)

        # Save based on format
        if args.format == "json" or args.output.endswith(".json"):
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            # Convert to DataFrame and save as CSV
            df = to_dataframe(results)
            df.to_csv(args.output, index=False, encoding="utf-8")
        
        print(f"✅ Results saved to {args.output}")
    
    else:  # API mode
        print(f"🚀 Starting Google News Scraper API on http://{args.host}:{args.port}")
        print(f"📖 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"🔧 Interactive API: http://{args.host}:{args.port}/redoc")
        uvicorn.run(app, host=args.host, port=args.port)
