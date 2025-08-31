# 📰 Google News Scraper + RAG Engine

This project scrapes **Google News articles**, extracts **real article content**, performs **sentiment analysis**, and enables **semantic search** & **RAG (Retrieval-Augmented Generation)** responses using **T5 transformers**.

---

## ✨ Features
- Scrape **Google News RSS feeds** with real article content extraction.  
- Multi-fallback scraping using **newspaper3k**, **trafilatura**, and **BeautifulSoup**.  
- Perform **sentiment analysis** with NLTK VADER.  
- Create and search **vector embeddings** with SentenceTransformers + FAISS.  
- Generate **RAG-based answers** with T5-small.  
- Save results in **JSON** or **CSV** format.  
- Track **success rate** of article extraction.  
- Supports **parallel workers** for faster scraping.  
- Stores vectors and metadata in **FAISS + JSON** for reusability.  

---

## ⚙️ Arguments
| Argument        | Description |
|-----------------|-------------|
| `--q`           | Search query for scraping news |
| `--search`      | Search query in existing vector DB |
| `--rag-query`   | Question for RAG answer generation |
| `--hl`          | Language (default: `en-US`) |
| `--gl`          | Country (default: `US`) |
| `--max`         | Max results to fetch (default: `10`) |
| `--workers`     | Number of parallel workers (default: `3`) |
| `--output`      | Output file (default: `output.json`) |
| `--format`      | Output format (`json` or `csv`) |
| `--vector-db`   | Path to vector DB (default: `news_vectors`) |

---

## 🖥️ Example Usage

### 1. Scrape Google News and save results
```bash
python Hackathon.py --q "artificial intelligence" --hl en-US --gl US --max 15 --workers 5 --output results.json --format json --vector-db ai_news_vectors
