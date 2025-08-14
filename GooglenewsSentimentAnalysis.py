import argparse
import concurrent.futures
import feedparser
import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from newspaper import Article
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure NLTK data is available
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)


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


def scrape_article_content(url: str, timeout: float = 15.0) -> dict:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {"article_text": article.text.strip(), "article_title": article.title}
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
    def __init__(self):
        self._analyzer = SentimentIntensityAnalyzer()
    def analyze(self, text: str) -> tuple:
        scores = self._analyzer.polarity_scores(text or "")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Google News with real article content")
    parser.add_argument("--q", required=True, help="Search query")
    parser.add_argument("--hl", default="en-US", help="Language")
    parser.add_argument("--gl", default="US", help="Country")
    parser.add_argument("--max", type=int, default=10, help="Max results")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    parser.add_argument("--output", default="output.json", help="Output JSON file")
    args = parser.parse_args()

    results = scrape_google_news(args.q, args.hl, args.gl, args.max, args.workers)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Results saved to {args.output}")
