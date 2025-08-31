import argparse
import concurrent.futures
import feedparser
import json
import re
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import requests
from bs4 import BeautifulSoup
from newspaper import Article
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ✅ New import
import trafilatura  
from json import JSONEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch

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
    vector: Optional[np.ndarray] = None
    vector_id: Optional[int] = None


class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class VectorDB:
    def __init__(self, dimension: int = None):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Get dimension from model if not specified
        if dimension is None:
            sample_text = "Sample text for dimension calculation"
            sample_vector = self.model.encode([sample_text])[0]
            dimension = len(sample_vector)
        
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.stored_items: Dict[int, NewsItem] = {}
        self.current_id = 0
    
    def encode_text(self, text: str) -> np.ndarray:
        try:
            vector = self.model.encode([text])[0]
            return vector.astype(np.float32)
        except Exception as e:
            print(f"Error encoding text: {e}")
            # Return zero vector of correct dimension as fallback
            return np.zeros(self.dimension, dtype=np.float32)
    
    def add_item(self, item: NewsItem) -> NewsItem:
        try:
            text_to_encode = f"{item.title} {item.article_text}"
            vector = self.encode_text(text_to_encode)
            
            # Verify vector dimension
            if len(vector) != self.dimension:
                print(f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}")
                vector = np.zeros(self.dimension, dtype=np.float32)
            
            item.vector = vector
            item.vector_id = self.current_id
            
            self.index.add(vector.reshape(1, -1))
            self.stored_items[self.current_id] = item
            self.current_id += 1
            return item
        except Exception as e:
            print(f"Error adding item to vector database: {e}")
            return item
    
    def search(self, query: str, k: int = 5) -> List[NewsItem]:
        query_vector = self.encode_text(query)
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        for idx in indices[0]:
            if idx != -1 and idx in self.stored_items:
                results.append(self.stored_items[idx])
        return results

    def save(self, filepath: str):
        faiss.write_index(self.index, f"{filepath}.index")
        # Convert items to dict, handling numpy arrays
        items_data = {}
        for k, v in self.stored_items.items():
            item_dict = asdict(v)
            # Convert numpy array to list for JSON serialization
            if item_dict['vector'] is not None:
                item_dict['vector'] = item_dict['vector'].tolist()
            items_data[k] = item_dict
        
        with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
            json.dump(
                {"current_id": self.current_id, "items": items_data},
                f,
                cls=NumpyJSONEncoder,
                ensure_ascii=False,
                indent=2
            )
    
    def load(self, filepath: str):
        self.index = faiss.read_index(f"{filepath}.index")
        with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.current_id = data["current_id"]
            self.stored_items = {
                int(k): NewsItem(**{
                    key: (np.array(val) if key == 'vector' and val is not None else val)
                    for key, val in v.items()
                })
                for k, v in data["items"].items()
            }

    def get_relevant_context(self, query: str, k: int = 3) -> Tuple[List[NewsItem], List[float]]:
        """Get relevant articles and their distances for RAG"""
        query_vector = self.encode_text(query)
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1 and idx in self.stored_items:
                results.append(self.stored_items[idx])
        return results, distances[0]


class RAGEngine:
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
        self.model_name = "t5-small"
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map='auto' if torch.cuda.is_available() else None,
                torch_dtype=torch.float32
            )
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(self, query: str, max_length: int = 256) -> str:
        try:
            # Get relevant articles
            relevant_items, distances = self.vector_db.get_relevant_context(query, k=2)  # Reduced from 3 to 2
            
            # Prepare context more concisely
            context = " ".join([
                f"{item.title}. {item.article_text[:200]}..." 
                for item in relevant_items
            ])
            
            # More focused prompt
            prompt = f"summarize in 2-3 sentences: {query}\n\nContext: {context}"

            # Generate response with tighter constraints
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=50,  # Ensure minimum length
                num_return_sequences=1,
                temperature=0.5,  # Reduced from 0.7 for more focused output
                do_sample=True,
                no_repeat_ngram_size=3,
                length_penalty=1.5,  # Prefer shorter responses
                early_stopping=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"


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
    except Exception:
        return None


def extract_summary_text(summary_html: str) -> str:
    soup = BeautifulSoup(summary_html, "html.parser")
    for a_tag in soup.find_all("a"):
        a_tag.replace_with(a_tag.get_text())
    return re.sub(r'\s+', ' ', soup.get_text(separator=' ', strip=True)).strip()


# ✅ Updated scrape_article_content with multi-fallback
def scrape_article_content(url: str, timeout: float = 15.0) -> dict:
    result = {"article_text": "", "article_title": None, "status": "failure"}
    
    try:
        # First try newspaper3k
        article = Article(url)
        article.download()
        article.parse()
        if article.text.strip():
            return {
                "article_text": article.text.strip(), 
                "article_title": article.title,
                "status": "success"
            }
    except Exception:
        pass

    try:
        # Fallback: trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted:
                return {
                    "article_text": extracted.strip(), 
                    "article_title": None,
                    "status": "success"
                }
    except Exception:
        pass

    try:
        # Last fallback: BeautifulSoup paragraph join
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        if text.strip():
            return {
                "article_text": text.strip(), 
                "article_title": soup.title.string if soup.title else None,
                "status": "success"
            }
    except Exception:
        pass

    return result

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
    def __init__(self):  # ✅ Fixed: was _init
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


def scrape_google_news(query: str, language: str = "en-US", country: str = "US", 
                      max_results: int = 10, workers: int = 3, 
                      vector_db: Optional[VectorDB] = None) -> Tuple[List[dict], float]:
    try:
        rss_url = build_google_news_rss_url(query, hl=language, gl=country)
        news_items = fetch_google_news(rss_url, max_items=max_results)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            news_items = list(executor.map(process_news_item, news_items))
        
        news_items = analyze_sentiment(news_items)
        
        if vector_db:
            news_items = [vector_db.add_item(item) for item in news_items]
        
        # Convert NewsItems to dict and calculate success rate
        results = []
        success_count = 0
        total_count = 0
        
        for item in news_items:
            success = bool(item.article_text.strip())
            if success:
                success_count += 1
            total_count += 1
            
            results.append({
                "title": item.title,
                "link": item.link,
                "published": item.published,
                "source": item.source,
                "article_text": item.article_text,
                "article_title": item.article_title,
                "sentiment": item.sentiment,
                "sentiment_score": item.sentiment_score,
                "status": "success" if success else "failure"
            })
        
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        return results, success_rate
    except Exception as e:
        print(f"Error in scrape_google_news: {e}")
        return [], 0.0


def to_dataframe(news_items: List[dict]) -> pd.DataFrame:
    try:
        # Convert numpy arrays to lists before creating DataFrame
        processed_items = []
        for item in news_items:
            processed_item = item.copy()
            if isinstance(processed_item.get('vector'), np.ndarray):
                processed_item['vector'] = processed_item['vector'].tolist()
            processed_items.append(processed_item)
        
        df = pd.DataFrame(processed_items)
        # Select only the columns we want
        columns = [
            "title", "link", "published", "source", "article_text", 
            "article_title", "sentiment", "sentiment_score"
        ]
        return df[columns]
    except Exception as e:
        print(f"Error in to_dataframe: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Google News with real article content and RAG")
    group = parser.add_mutually_exclusive_group()  # Remove required=True
    group.add_argument("--q", help="Search query for scraping news")
    group.add_argument("--search", help="Search query in existing vector database")
    
    parser.add_argument("--hl", default="en-US", help="Language")
    parser.add_argument("--gl", default="US", help="Country")
    parser.add_argument("--max", type=int, default=10, help="Max results")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    parser.add_argument("--output", default="output.json", help="Output file (CSV or JSON)")
    parser.add_argument("--format", choices=["csv", "json"], default="json", help="Output format")
    parser.add_argument("--vector-db", default="news_vectors", help="Vector database file path")
    parser.add_argument("--rag-query", help="Query for RAG-based answer generation")
    args = parser.parse_args()

    vector_db = VectorDB()
    
    if args.rag_query:
        try:
            vector_db.load(args.vector_db)
            if not args.q and not args.search:
                # If no search query is provided, use the rag_query as search query
                args.search = args.rag_query
            
            if args.q:
                # If scraping new articles
                results = scrape_google_news(
                    args.q, args.hl, args.gl, args.max, 
                    args.workers, vector_db=vector_db
                )
                vector_db.save(args.vector_db)
            
            # Generate RAG response
            rag_engine = RAGEngine(vector_db)
            response = rag_engine.generate_response(args.rag_query)
            print(f"\nQuestion: {args.rag_query}")
            print(f"\nAnswer: {response}\n")
        except FileNotFoundError:
            print("No vector database found. Please run with --q first to scrape some articles.")
            print("Example: python Hackathon.py --q 'artificial intelligence' --max 10")
    elif args.search:
        try:
            vector_db.load(args.vector_db)
            results = vector_db.search(args.search)
            print(f"\nTop similar articles for query: {args.search}")
            for item in results:
                print(f"\nTitle: {item.title}")
                print(f"Sentiment: {item.sentiment} ({item.sentiment_score})")
                print(f"URL: {item.link}\n")
        except FileNotFoundError:
            print("No vector database found. Please run the scraper first.")
    else:
        try:
            results, success_rate = scrape_google_news(
                args.q, args.hl, args.gl, args.max, 
                args.workers, vector_db=vector_db
            )

            # Save vector database (full data including vectors)
            vector_db.save(args.vector_db)

            # Create output with metadata
            output_data = {
                "metadata": {
                    "query": args.q,
                    "success_rate": round(success_rate, 2),
                    "total_articles": len(results),
                    "successful_extractions": sum(1 for r in results if r["status"] == "success")
                },
                "articles": results
            }

            # Save results
            if args.format == "json" or args.output.endswith(".json"):
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
            else:
                df = pd.DataFrame(results)
                df.to_csv(args.output, index=False, encoding="utf-8")
            
            print(f"✅ Results saved to {args.output}")
            print(f"✅ Vector database saved to {args.vector_db}")
            print(f"📊 Article extraction success rate: {round(success_rate, 2)}%")
        except Exception as e:
            print(f"Error: {e}")
