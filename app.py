from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import json
import re
import hmac
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
import logging
import functools
import threading
import hashlib
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ── LOGGING ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ── STARTUP VALIDATION ─────────────────────────────────
if not os.getenv("GROQ_API_KEY") and not os.getenv("TOGETHER_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("No LLM provider key found. Set at least one of: GROQ_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY")

app = Flask(__name__, static_folder=None)

# ── SECURITY CONFIG ────────────────────────────────────
app.config['DEBUG'] = False
app.config['TESTING'] = False
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25MB — allows PDF uploads for citation tool

# ── CORS ───────────────────────────────────────────────
CORS(app, origins=[
    "https://lumina-jgnu.onrender.com",
    "http://127.0.0.1:5001",
    "http://localhost:5001",
])

# ── RATE LIMITING ──────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day"],
    storage_uri="memory://",
)

# ── LLM CLIENTS — TRIPLE PROVIDER ──────────────────────
# Groq (primary) → Together AI (fallback 1) → OpenRouter (fallback 2)
# All use the same Llama 3.3 70B model for consistency.
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
except Exception as e:
    logger.error(f"Groq init failed: {e}")
    groq_client = None

try:
    together_client = OpenAI(
        api_key=os.getenv("TOGETHER_API_KEY", ""),
        base_url="https://api.together.xyz/v1",
    ) if os.getenv("TOGETHER_API_KEY") else None
except Exception as e:
    logger.error(f"Together AI init failed: {e}")
    together_client = None

try:
    openrouter_client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
    ) if os.getenv("OPENROUTER_API_KEY") else None
except Exception as e:
    logger.error(f"OpenRouter init failed: {e}")
    openrouter_client = None

GROQ_MODEL       = "llama-3.3-70b-versatile"
TOGETHER_MODEL   = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"


def llm_call(messages: list, max_tokens: int = 1000, temperature: float = 0) -> str:
    """
    Unified LLM call with automatic triple-provider fallback.
    Tries Groq → Together AI → OpenRouter in order.
    Returns response content string. Raises RuntimeError if all fail.
    """
    providers = []
    if groq_client:
        providers.append(("Groq", groq_client, GROQ_MODEL))
    if together_client:
        providers.append(("Together AI", together_client, TOGETHER_MODEL))
    if openrouter_client:
        providers.append(("OpenRouter", openrouter_client, OPENROUTER_MODEL))

    if not providers:
        raise RuntimeError("No LLM provider configured.")

    last_error = None
    for name, provider, model in providers:
        try:
            resp = provider.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if name != "Groq":
                logger.info(f"LLM fallback used: {name}")
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate_limit" in err or "rate limit" in err:
                logger.warning(f"{name} rate limited — trying next provider")
            else:
                logger.warning(f"{name} failed ({e}) — trying next provider")
            last_error = e
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

# ── IN-MEMORY STATS ────────────────────────────────────
# Resets on every Render restart (free tier spins down). Tracks activity
# within the current uptime window only. No DB required.
_stats_lock = threading.Lock()
_stats = {
    "started_at": datetime.now(timezone.utc),
    "total_searches": 0,
    "unique_ip_hashes": set(),
    "searches_by_hour": defaultdict(int),   # "YYYY-MM-DD HH" → count
    "recent_activity": [],                  # last 100 events
}

def _record_search(ip: str, topic: str, result_count: int):
    """Thread-safe. Called after every successful search."""
    ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:12]
    hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:00")
    entry = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ip_hash": ip_hash,
        "topic_length": len(topic),
        "results": result_count,
    }
    with _stats_lock:
        _stats["total_searches"] += 1
        _stats["unique_ip_hashes"].add(ip_hash)
        _stats["searches_by_hour"][hour_key] += 1
        _stats["recent_activity"].insert(0, entry)
        _stats["recent_activity"] = _stats["recent_activity"][:100]


# ── HTTPS REDIRECT ─────────────────────────────────────
@app.before_request
def force_https():
    if not request.is_secure and not request.headers.get('X-Forwarded-Proto', 'http') == 'https':
        if os.getenv('FLASK_ENV') == 'production':
            url = request.url.replace('http://', 'https://', 1)
            return redirect(url, code=301)


# ── SECURITY HEADERS ───────────────────────────────────
@app.after_request
def set_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    if os.getenv('FLASK_ENV') == 'production':
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response


# ── 404 HANDLER ────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


# ── KEYWORD EXTRACTION ─────────────────────────────────
def extract_keywords(query: str) -> str:
    """
    Extract 5-7 core search keywords from a full research topic.
    Removes filler words, keeps subject matter and context terms.
    Falls back to simple truncation if AI call fails.
    """
    prompt = f"""Extract the 5-7 most important search keywords from this research topic.
Return ONLY a short search string with the key terms, no explanation, no quotes.
Remove words like: among, the, and, of, in, a, an, with, for, to, using, study, analysis, effect, impact, influence, relationship, role.
Keep: subject terms, field names, specific concepts, geographic/demographic context.

Topic: "{query}"

Example:
Topic: "Social Media Advertising and Ad Avoidance Behaviour among Nigerian University Undergraduate Students"
Output: social media advertising ad avoidance Nigerian university students

Topic: "The Effect of Climate Change on Agricultural Productivity in Sub-Saharan Africa"
Output: climate change agricultural productivity Sub-Saharan Africa

Now extract keywords for: "{query}"
Output:"""
    try:
        content = llm_call(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0,
        )
        keywords = content.strip('"').strip("'")
        print(f"Keywords extracted: {keywords}")
        return keywords
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        # Simple fallback — remove common filler words
        stop = {'among','the','and','of','in','a','an','with','for','to','using',
                'study','analysis','effect','impact','influence','relationship','role',
                'on','its','their','from','by','at','as','into','through'}
        words = [w for w in query.split() if w.lower() not in stop]
        return " ".join(words[:7])


# ── SEARCH STRATEGY GENERATION ────────────────────────
def generate_search_strategy(topic: str) -> dict:
    """
    Generate a comprehensive multi-angle search strategy from one topic.
    Produces up to 6 targeted queries covering all angles a student needs
    for a thorough literature review:
      1. Exact topic
      2. Thematic angle (themes in other works)
      3. Work/author specific queries
      4. Cited-by / written-about queries
    Returns a dict with:
      - queries: list of {label, query, keywords} objects
      - sub_topics: list of display strings for the UI banner
    """
    prompt = f"""You are an expert academic research librarian helping a student build a comprehensive literature review.

Given this research topic, generate a multi-angle search strategy with up to 6 targeted search queries.
Each query must serve a DIFFERENT purpose so the student gets broad, comprehensive coverage.

Research topic: "{topic}"

Generate queries for ALL applicable angles:
1. EXACT TOPIC — search for the topic exactly as stated
2. THEMATIC — search for the core themes/concepts in other literary texts or contexts (not just this specific work)
3. WORK SPECIFIC — one query per named literary work, author, or text mentioned (if any)
4. AUTHOR SCHOLARSHIP — search for academic papers written BY or ABOUT the named authors
5. COMPARATIVE — search for papers comparing or contextualising the themes across similar works
6. BROADER FIELD — search for foundational theory papers on the core concept (e.g. postcolonial theory, religious extremism in literature)

RULES:
- Each query must be meaningfully different — no near-duplicates
- Keep queries concise and searchable (under 10 words each)
- For literary topics, always include at least one work-specific and one thematic query
- For social science topics, include methodology and theory queries
- Maximum 6 queries total

Return ONLY a valid JSON array. Each item must have:
- "label": short name for this search angle (e.g. "Exact topic", "Radical faith in literature", "Obinna Udenwe's Satan and Shaitans")
- "query": the full search string to use
- "keywords": 4-6 key terms extracted from the query

Example for "radical faith and manipulation in Obinna Udenwe's Satan and Shaitans and Elnathan John's Born on a Tuesday":
[
  {{"label": "Exact topic", "query": "radical faith political manipulation Nigerian fiction Udenwe Elnathan John", "keywords": "radical faith manipulation Nigerian fiction Udenwe Elnathan"}},
  {{"label": "Radical faith in African literature", "query": "radical faith religious extremism African literature fiction", "keywords": "radical faith religious extremism African literature"}},
  {{"label": "Obinna Udenwe Satan and Shaitans", "query": "Obinna Udenwe Satan Shaitans novel", "keywords": "Obinna Udenwe Satan Shaitans"}},
  {{"label": "Elnathan John Born on a Tuesday", "query": "Elnathan John Born on a Tuesday novel", "keywords": "Elnathan John Born Tuesday novel"}},
  {{"label": "Political manipulation Nigerian novels", "query": "political manipulation religion Nigerian contemporary fiction", "keywords": "political manipulation religion Nigerian fiction"}},
  {{"label": "Postcolonial religion African fiction theory", "query": "postcolonial theory religion extremism African novel", "keywords": "postcolonial religion extremism African novel"}}
]

Now generate the search strategy for: "{topic}"
Return ONLY the JSON array. No explanation. No markdown. No backticks.
"""
    try:
        raw = llm_call(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0,
        )
        raw = re.sub(r'```[a-z]*', '', raw)
        raw = re.sub(r'```', '', raw)
        raw = raw.strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        strategy = json.loads(raw)
        if isinstance(strategy, list) and strategy:
            print(f"Search strategy: {len(strategy)} angles")
            for s in strategy:
                print(f"  [{s.get('label')}] → {s.get('query')}")
            return {
                "queries": strategy,
                "sub_topics": [s["label"] for s in strategy]
            }
    except Exception as e:
        print(f"Strategy generation error: {e}")

    # Fallback — single query with extracted keywords
    kw = extract_keywords(topic)
    return {
        "queries": [{"label": "Search", "query": topic, "keywords": kw}],
        "sub_topics": []
    }


# ── ARXIV ──────────────────────────────────────────────
def search_arxiv(query: str, keywords: str, max_results=10) -> list:
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{keywords}", "start": 0, "max_results": max_results, "sortBy": "relevance"}
    try:
        r = requests.get(url, params=params, timeout=12)
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            def t(tag): e = entry.find(f"atom:{tag}", ns); return e.text.strip() if e is not None and e.text else ""
            title = t("title")
            if not title: continue
            papers.append({
                "title": title,
                "authors": [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)[:3] if a.find("atom:name", ns) is not None],
                "abstract": t("summary")[:400],
                "year": t("published")[:4],
                "journal": "arXiv",
                "link": t("id"),
                "source": "arXiv",
            })
        print(f"arXiv returned {len(papers)} papers")
        return papers
    except Exception as e:
        print(f"arXiv error: {e}")
        return []


# ── SEMANTIC SCHOLAR ───────────────────────────────────
def search_semantic_scholar(query: str, keywords: str, max_results=10) -> list:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": keywords,
        "limit": max_results,
        "fields": "title,authors,abstract,year,journal,externalIds,openAccessPdf",
    }
    s2_key = os.getenv("S2_API_KEY", "")
    headers = {
        "User-Agent": "Lumina/1.0 (academic research tool; contact: lumina@research.app)",
    }
    if s2_key:
        headers["x-api-key"] = s2_key
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        print(f"Semantic Scholar status: {r.status_code}")
        if r.status_code == 429:
            print("Semantic Scholar rate limited — skipping")
            return []
        if r.status_code != 200:
            print(f"Semantic Scholar error response: {r.text[:200]}")
            return []
        data = r.json()
        total = data.get("total", 0)
        print(f"Semantic Scholar total matches: {total}")
        papers = []
        for item in data.get("data", []):
            title = item.get("title", "")
            if not title: continue
            abstract = (item.get("abstract") or "")[:400]
            # Skip papers with no abstract — saves ranking tokens on useless results
            if not abstract: continue
            doi = (item.get("externalIds") or {}).get("DOI", "")
            pdf = (item.get("openAccessPdf") or {}).get("url", "")
            link = pdf or (f"https://doi.org/{doi}" if doi else
                          f"https://scholar.google.com/scholar?q={requests.utils.quote(title)}")
            papers.append({
                "title": title,
                "authors": [a.get("name", "") for a in (item.get("authors") or [])[:3]],
                "abstract": abstract,
                "year": str(item.get("year", "")) if item.get("year") else "",
                "journal": (item.get("journal") or {}).get("name", ""),
                "link": link,
                "source": "Semantic Scholar",
            })
        print(f"Semantic Scholar returned {len(papers)} papers")
        return papers
    except Exception as e:
        print(f"Semantic Scholar error: {e}")
        return []


# ── PUBMED ─────────────────────────────────────────────
def search_pubmed(query: str, keywords: str, max_results=10) -> list:
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        r = requests.get(f"{base}/esearch.fcgi", params={"db":"pubmed","term":keywords,"retmax":max_results,"retmode":"json"}, timeout=10)
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids: return []
        r2 = requests.get(f"{base}/efetch.fcgi", params={"db":"pubmed","id":",".join(ids),"retmode":"xml"}, timeout=10)
        root = ET.fromstring(r2.text)
        papers = []
        for article in root.findall(".//PubmedArticle"):
            title = " ".join(t.text or "" for t in article.findall(".//ArticleTitle"))
            if not title: continue
            abstract = " ".join(t.text or "" for t in article.findall(".//AbstractText"))[:400]
            authors = [f"{n.findtext('LastName','')} {n.findtext('ForeName','')}".strip() for n in article.findall(".//Author")[:3]]
            year_el = article.find(".//PubDate/Year")
            year = year_el.text if year_el is not None else ""
            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text if journal_el is not None else ""
            pmid_el = article.find(".//PMID")
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_el.text}/" if pmid_el is not None else ""
            papers.append({"title":title,"authors":authors,"abstract":abstract,"year":year,"journal":journal,"link":link,"source":"PubMed"})
        print(f"PubMed returned {len(papers)} papers")
        return papers
    except Exception as e:
        print(f"PubMed error: {e}")
        return []


# ── CROSSREF ───────────────────────────────────────────
def search_crossref(query: str, keywords: str, max_results=10) -> list:
    url = "https://api.crossref.org/works"
    params = {"query": keywords, "rows": max_results, "select": "title,author,abstract,published,container-title,DOI"}
    try:
        r = requests.get(url, params=params, timeout=12)
        items = r.json().get("message", {}).get("items", [])
        papers = []
        for item in items:
            titles = item.get("title", [])
            title = titles[0] if titles else ""
            if not title: continue
            abstract = re.sub(r'<[^>]+>', '', item.get("abstract", ""))[:400]
            if not abstract: continue
            authors = [f"{a.get('given','')} {a.get('family','')}".strip() for a in (item.get("author") or [])[:3]]
            pub = item.get("published", {}).get("date-parts", [[""]])
            year = str(pub[0][0]) if pub and pub[0] else ""
            journal = (item.get("container-title") or [""])[0]
            doi = item.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else f"https://scholar.google.com/scholar?q={requests.utils.quote(title)}"
            papers.append({"title":title,"authors":authors,"abstract":abstract,"year":year,"journal":journal,"link":link,"source":"CrossRef"})
        print(f"CrossRef returned {len(papers)} papers")
        return papers
    except Exception as e:
        print(f"CrossRef error: {e}")
        return []


# ── OPENALEX (AJOL / General) ──────────────────────────
def search_openalex(query: str, keywords: str, domain_filter: str = "", label: str = "AJOL", max_results=10) -> list:
    url = "https://api.openalex.org/works"
    params = {
        "search": keywords,
        "per_page": max_results,
        "select": "title,authorships,abstract_inverted_index,publication_year,primary_location,doi,open_access",
        "mailto": "lumina@research.app",
        "sort": "relevance_score:desc",
    }
    if domain_filter:
        params["filter"] = domain_filter
    try:
        headers = {"User-Agent": "Lumina/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200: return []
        data = r.json()
        results = data.get("results", [])
        papers = []
        for work in results:
            title = work.get("title", "")
            if not title: continue
            abstract = ""
            inv = work.get("abstract_inverted_index")
            if inv:
                try:
                    pos = {}
                    for word, plist in inv.items():
                        for p in plist: pos[p] = word
                    abstract = " ".join(pos[k] for k in sorted(pos))[:400]
                except: pass
            authors = [a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])[:3]]
            year = str(work.get("publication_year", "")) if work.get("publication_year") else ""
            loc = work.get("primary_location") or {}
            source_info = loc.get("source") or {}
            journal = source_info.get("display_name", "")
            doi = work.get("doi", "") or ""
            oa_url = (work.get("open_access") or {}).get("oa_url", "") or ""
            landing = loc.get("landing_page_url") or ""
            if oa_url and oa_url.startswith("http"): link = oa_url
            elif landing and landing.startswith("http"): link = landing
            elif doi: link = doi if doi.startswith("http") else f"https://doi.org/{doi}"
            else: link = f"https://scholar.google.com/scholar?q={requests.utils.quote(title)}"
            papers.append({"title":title,"authors":authors,"abstract":abstract,"year":year,"journal":journal,"link":link,"source":label})
        print(f"{label} (OpenAlex) returned {len(papers)} papers")
        return papers
    except Exception as e:
        print(f"{label} error: {e}")
        return []


# ── CORE ───────────────────────────────────────────────
def search_core(query: str, keywords: str, max_results=10) -> list:
    api_key = os.getenv("CORE_API_KEY", "")
    url = "https://api.core.ac.uk/v3/search/works"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params = {"q": keywords, "limit": max_results}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code in [401, 403]:
            print("CORE: Invalid or missing API key")
            return []
        data = r.json()
        papers = []
        for item in data.get("results", []):
            title = item.get("title", "")
            if not title: continue
            abstract = (item.get("abstract") or "")[:400]
            if not abstract: continue
            authors = [a.get("name", "") for a in (item.get("authors") or [])[:3]]
            year = str(item.get("yearPublished", "")) if item.get("yearPublished") else ""
            journals = item.get("journals") or []
            journal = journals[0].get("title", "") if journals else ""
            urls = item.get("sourceFulltextUrls") or []
            doi = item.get("doi") or ""
            if urls: link = urls[0]
            elif doi: link = doi if doi.startswith("http") else f"https://doi.org/{doi}"
            else: link = f"https://scholar.google.com/scholar?q={requests.utils.quote(title)}"
            papers.append({"title":title,"authors":authors,"abstract":abstract,"year":year,"journal":journal,"link":link,"source":"CORE"})
        print(f"CORE returned {len(papers)} papers")
        return papers
    except Exception as e:
        print(f"CORE error: {e}")
        return []


# ── DOAJ ───────────────────────────────────────────────
def search_doaj(query: str, keywords: str, max_results=10) -> list:
    url = "https://doaj.org/api/search/articles/" + requests.utils.quote(keywords)
    params = {"pageSize": max_results, "page": 1}
    try:
        headers = {"User-Agent": "Lumina/1.0", "Accept": "application/json"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200: return []
        data = r.json()
        papers = []
        for item in data.get("results", []):
            bib = item.get("bibjson", {})
            title = bib.get("title", "")
            if not title: continue
            abstract = (bib.get("abstract") or "")[:400]
            if not abstract: continue
            authors = [a.get("name", "") for a in (bib.get("author") or [])[:3]]
            year = str(bib.get("year", "")) if bib.get("year") else ""
            journal = (bib.get("journal") or {}).get("title", "")
            links = bib.get("link") or []
            link = next((l["url"] for l in links if l.get("url","").startswith("http")), "")
            if not link:
                doi_list = bib.get("identifier") or []
                for d in doi_list:
                    if d.get("type") == "doi":
                        link = f"https://doi.org/{d['id']}"
                        break
            if not link:
                link = f"https://scholar.google.com/scholar?q={requests.utils.quote(title)}"
            papers.append({"title":title,"authors":authors,"abstract":abstract,"year":year,"journal":journal,"link":link,"source":"DOAJ"})
        print(f"DOAJ returned {len(papers)} papers")
        return papers
    except Exception as e:
        print(f"DOAJ error: {e}")
        return []


# ── AI RANKING ─────────────────────────────────────────
def rank_batch(topic: str, papers: list) -> list:
    papers_text = ""
    for i, p in enumerate(papers):
        abstract_snippet = (p['abstract'] or "")[:150]
        papers_text += f"{i+1}. Title: {p['title']}\n"
        papers_text += f"   Abstract: {abstract_snippet}\n\n"

    prompt = f"""You are a strict academic research assistant. A student is researching: "{topic}"

Score each paper's relevance using this guide:
- 9-10: Topic is the CENTRAL subject — title and abstract are directly about it
- 7-8: Topic is a MAJOR focus — substantially discussed
- 5-6: Topic is RELEVANT but not central — meaningfully connected
- 3-4: Topic appears INCIDENTALLY — mentioned briefly or loosely connected
- 1-2: NOT relevant — only shares a word by coincidence

CRITICAL: If the topic mentions a specific place, institution, person or named subject, papers must be ABOUT that subject to score above 5 — not just mention it in passing.

Return a JSON array. Each object must have:
- "index": 1-based number
- "score": 1-10
- "relevance": 1-2 sentences on why this paper does or doesn't match
- "key_contribution": 1 sentence on what this paper found

Return ONLY the raw JSON array. No markdown, no backticks.

Papers:
{papers_text}"""

    raw = llm_call(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0,
    )
    raw = re.sub(r'^```[a-z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match: raw = match.group(0)

    try:
        ai_data = json.loads(raw)
    except Exception as e:
        print(f"JSON parse error: {e}")
        ai_data = [{"index": i+1, "score": 5, "relevance": "", "key_contribution": ""} for i in range(len(papers))]

    ai_map = {item["index"]: item for item in ai_data}
    enriched = []
    for i, p in enumerate(papers):
        ai = ai_map.get(i + 1, {})
        enriched.append({**p, "score": ai.get("score", 1), "relevance": ai.get("relevance", ""), "key_contribution": ai.get("key_contribution", "")})
    return enriched


def prefilter_papers(topic: str, papers: list) -> list:
    """
    Fast keyword pre-filter — removes papers with zero topic signal
    before sending to the AI. No API calls, pure Python.
    Keeps papers where at least one meaningful topic term appears
    in the title or abstract.
    """
    # Extract meaningful terms from topic — words over 4 chars, skip filler
    stop = {'about','among','their','there','these','those','which','where',
            'using','study','analysis','effect','impact','influence','between',
            'university','undergraduate','students','research','papers','nigeria',
            'nigerian','african','africa'}
    topic_terms = [w.lower() for w in topic.split()
                   if len(w) > 4 and w.lower() not in stop]

    if not topic_terms:
        return papers  # can't filter — return all

    filtered = []
    for p in papers:
        text = (p.get('title','') + ' ' + p.get('abstract','')).lower()
        # Keep if any meaningful topic term appears
        if any(term in text for term in topic_terms):
            filtered.append(p)

    removed = len(papers) - len(filtered)
    print(f"Pre-filter: removed {removed} irrelevant papers, {len(filtered)} remain")
    return filtered


def prioritise_papers(papers: list, strategy_queries: list) -> list:
    """
    Sort papers so the most targeted angles are ranked first.
    This ensures that even if TPM limit hits mid-ranking,
    the most relevant papers (exact topic, work-specific) are
    already scored while broader theory papers get default scores.
    """
    # Exact topic and work-specific angles are highest priority
    priority_labels = {'exact topic', 'exact', 'work specific', 'author scholarship'}
    high = [p for p in papers if p.get('search_angle','').lower() in priority_labels]
    low  = [p for p in papers if p.get('search_angle','').lower() not in priority_labels]
    return high + low


def rank_papers_with_ai(topic: str, papers: list, strategy_queries: list = None) -> list:
    import time
    BATCH_SIZE = 10

    # Step 1 — Pre-filter: remove papers with no topic signal
    papers = prefilter_papers(topic, papers)

    # Step 2 — Prioritise: exact/work-specific angles first
    papers = prioritise_papers(papers, strategy_queries or [])

    print(f"Ranking {len(papers)} papers after pre-filter")

    all_enriched = []
    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i:i + BATCH_SIZE]
        batch_num = i//BATCH_SIZE + 1
        total_batches = (len(papers) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Ranking batch {batch_num}/{total_batches} ({len(batch)} papers)...")

        # Retry with backoff only when a 429 actually hits — no artificial delays
        for attempt in range(3):
            try:
                enriched = rank_batch(topic, batch)
                all_enriched.extend(enriched)
                break
            except Exception as e:
                err = str(e)
                if '429' in err or 'rate_limit' in err:
                    # Groq TPM resets every 60s — wait long enough to clear it
                    wait = 30 if attempt == 0 else 60
                    print(f"TPM limit hit — waiting {wait}s before retry {attempt+1}...")
                    time.sleep(wait)
                elif attempt < 2 and ('Connection' in err or 'timeout' in err.lower()):
                    print(f"Connection error, retrying in 2s...")
                    time.sleep(2)
                else:
                    print(f"Ranking batch failed after retries — using default scores")
                    all_enriched.extend([{**p, "score": 5, "relevance": "", "key_contribution": ""} for p in batch])
                    break

    ranked = sorted(all_enriched, key=lambda x: x["score"], reverse=True)

    for threshold in [5, 4, 3]:
        relevant = [p for p in ranked if p["score"] >= threshold]
        if len(relevant) >= 5:
            print(f"Threshold used: {threshold} — {len(relevant)} papers passed")
            return relevant[:70]

    print("Very niche topic — returning top results")
    return ranked[:5]


# ── PARALLEL SEARCH HELPER ─────────────────────────────
def search_all_sources_for_angle(angle: dict, topic: str) -> list:
    """
    Search all 8 databases for a single angle simultaneously using threads.
    Each database call runs in its own thread — the angle finishes in the
    time of the slowest single database call, not the sum of all of them.
    """
    q   = angle.get("query", topic)
    kw  = angle.get("keywords", extract_keywords(q))
    label = angle.get("label", "Search")
    print(f"\nSearching angle: [{label}]")

    tasks = [
        functools.partial(search_arxiv, q, kw, max_results=8),
        functools.partial(search_semantic_scholar, q, kw, max_results=8),
        functools.partial(search_pubmed, q, kw, max_results=8),
        functools.partial(search_crossref, q, kw, max_results=8),
        functools.partial(search_openalex, q, kw, label="AJOL", max_results=8),
        functools.partial(
            search_openalex, q, kw,
            domain_filter="primary_topic.domain.id:https://openalex.org/domains/2|https://openalex.org/domains/4",
            label="ERIC", max_results=8
        ),
        functools.partial(search_core, q, kw, max_results=8),
        functools.partial(search_doaj, q, kw, max_results=8),
    ]

    angle_papers = []
    with ThreadPoolExecutor(max_workers=8) as db_executor:
        futures = {db_executor.submit(task): task.func.__name__ for task in tasks}
        for future in as_completed(futures):
            try:
                papers = future.result()
                angle_papers.extend(papers)
            except Exception as e:
                print(f"Source error in angle [{label}]: {e}")

    for p in angle_papers:
        p["search_angle"] = label

    print(f"Angle [{label}]: {len(angle_papers)} papers collected")
    return angle_papers


# ── ROUTES ─────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/search', methods=['POST'])
@limiter.limit("30 per hour")
@limiter.limit("5 per minute")
def search():
    if groq_client is None and together_client is None and openrouter_client is None:
        return jsonify({'error': 'AI service unavailable. Please try again later.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid request body'}), 400

    # Sanitise input — strip, cap length, remove non-readable characters
    topic = data.get('topic', '').strip()[:500]
    topic = re.sub(r"[^\w\s'\"\-\.\,\?\!]", ' ', topic)
    topic = re.sub(r'\s+', ' ', topic).strip()
    if len(topic) < 3:
        return jsonify({'error': 'Topic too short — please enter at least 3 characters'}), 400

    try:
        # Step 1 — Generate comprehensive multi-angle search strategy
        strategy = generate_search_strategy(topic)
        queries = strategy["queries"]
        sub_topics = strategy["sub_topics"]

        # Step 2 — Search all angles in parallel (each angle searches all 8
        # databases simultaneously). Total time ≈ slowest single request,
        # not the sum of all 48 requests.
        all_papers = []
        seen_titles = set()

        with ThreadPoolExecutor(max_workers=len(queries)) as angle_executor:
            angle_futures = {
                angle_executor.submit(search_all_sources_for_angle, angle, topic): angle.get("label", "Search")
                for angle in queries
            }
            for future in as_completed(angle_futures):
                label = angle_futures[future]
                try:
                    angle_papers = future.result()
                    all_papers.extend(angle_papers)
                except Exception as e:
                    print(f"Angle [{label}] failed: {e}")

        # Step 3 — Deduplicate by normalised title
        unique_papers = []
        for p in all_papers:
            key = p["title"].lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                unique_papers.append(p)

        # Step 4 — Remove papers with no title or abstract
        unique_papers = [p for p in unique_papers if p.get("title") and p.get("abstract")]
        
        # Cap at 100 papers before ranking — prevents Groq per-minute rate limit
        # With 10 papers per batch, 100 papers = 10 batches = safe within rate limits
        if len(unique_papers) > 100:
            import random
            # Keep diverse sources — sample proportionally rather than just truncating
            unique_papers = unique_papers[:100]
        
        print(f"\nTotal unique papers with abstracts: {len(unique_papers)}")

        # Step 5 — Rank all papers against the original full topic
        ranked = rank_papers_with_ai(topic, unique_papers, strategy_queries=queries)

        # Record activity for admin dashboard
        _record_search(
            ip=request.remote_addr or "unknown",
            topic=topic,
            result_count=len(ranked),
        )

        return jsonify({
            'results': ranked,
            'sub_topics': sub_topics,
            'angles_searched': len(queries)
        })

    except Exception as e:
        err_str = str(e)
        logger.error(f"Search error: {err_str}")
        if '429' in err_str or 'rate_limit' in err_str or 'Rate limit' in err_str:
            return jsonify({'error': 'rate_limit', 'message': 'Groq daily token limit reached. Please try again tomorrow.'}), 429
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500


# ── RATE LIMIT ERROR HANDLER ──────────────────────────
@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({
        'error': 'rate_limit',
        'message': 'Too many searches. You are limited to 5 per minute and 30 per hour. Please wait a moment and try again.'
    }), 429


# ── ADMIN DASHBOARD ────────────────────────────────────
@app.route('/admin')
@limiter.exempt
def admin():
    key = request.args.get('key', '')
    admin_key = os.getenv('ADMIN_KEY', '')
    if not admin_key or not hmac.compare_digest(key, admin_key):
        return jsonify({'error': 'Not authorised'}), 401

    with _stats_lock:
        total        = _stats["total_searches"]
        unique_users = len(_stats["unique_ip_hashes"])
        started_at   = _stats["started_at"].strftime("%Y-%m-%d %H:%M UTC")
        uptime_hrs   = round((datetime.now(timezone.utc) - _stats["started_at"]).total_seconds() / 3600, 1)
        by_hour      = dict(sorted(_stats["searches_by_hour"].items(), reverse=True)[:24])
        recent       = list(_stats["recent_activity"][:20])

    by_hour_rows = "".join(
        f"<tr><td>{h}</td><td>{c}</td></tr>" for h, c in by_hour.items()
    )
    activity_rows = "".join(
        f"<tr><td>{e['time']}</td><td>{e['ip_hash']}</td>"
        f"<td>{e['topic_length']} chars</td><td>{e['results']}</td></tr>"
        for e in recent
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lumina — Admin</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Courier New',monospace;background:#0a0a0a;color:#e0e0e0;padding:2rem}}
  h1{{font-size:1.4rem;color:#fff;margin-bottom:.25rem;letter-spacing:.05em}}
  .sub{{color:#555;font-size:.8rem;margin-bottom:2.5rem}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:2.5rem}}
  .card{{background:#111;border:1px solid #222;border-radius:6px;padding:1.25rem}}
  .card .val{{font-size:2rem;font-weight:700;color:#fff;line-height:1}}
  .card .lbl{{font-size:.7rem;color:#555;margin-top:.4rem;text-transform:uppercase;letter-spacing:.08em}}
  h2{{font-size:.85rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.75rem}}
  table{{width:100%;border-collapse:collapse;margin-bottom:2.5rem;font-size:.8rem}}
  th{{text-align:left;color:#444;padding:.4rem .75rem;border-bottom:1px solid #1a1a1a;font-weight:normal}}
  td{{padding:.4rem .75rem;border-bottom:1px solid #151515;color:#aaa}}
  tr:hover td{{background:#111}}
  .note{{font-size:.72rem;color:#333;margin-top:1rem}}
</style>
</head>
<body>
<h1>LUMINA / ADMIN</h1>
<p class="sub">Session started {started_at} &nbsp;·&nbsp; uptime {uptime_hrs}h &nbsp;·&nbsp; resets on restart</p>

<div class="grid">
  <div class="card"><div class="val">{total}</div><div class="lbl">Total searches</div></div>
  <div class="card"><div class="val">{unique_users}</div><div class="lbl">Unique visitors</div></div>
  <div class="card"><div class="val">{uptime_hrs}h</div><div class="lbl">Uptime</div></div>
  <div class="card"><div class="val">{round(total/max(uptime_hrs,0.01),1)}</div><div class="lbl">Searches / hr</div></div>
</div>

<h2>Activity by hour (last 24)</h2>
<table>
  <tr><th>Hour (UTC)</th><th>Searches</th></tr>
  {by_hour_rows if by_hour_rows else '<tr><td colspan="2" style="color:#333">No data yet</td></tr>'}
</table>

<h2>Recent searches (last 20)</h2>
<table>
  <tr><th>Time (UTC)</th><th>Visitor</th><th>Topic size</th><th>Results</th></tr>
  {activity_rows if activity_rows else '<tr><td colspan="4" style="color:#333">No data yet</td></tr>'}
</table>

<p class="note">IPs are one-way hashed — not recoverable. Topics are not stored, only their character length.</p>
</body>
</html>"""
    return html


# ── CITATION TOOL ──────────────────────────────────────
def extract_metadata_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Extract text from PDF and use AI to identify bibliographic metadata.
    Returns dict with: title, authors, year, journal, volume, issue,
    pages, doi, publisher, url, edition, location.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {"error": "PyMuPDF not installed. Add 'pymupdf' to requirements.txt."}

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # Extract first 3 pages — enough for title page + abstract + references header
        text = ""
        for i in range(min(3, len(doc))):
            text += doc[i].get_text()
        doc.close()
        text = text[:4000]  # Cap tokens sent to AI
    except Exception as e:
        return {"error": f"Could not read PDF: {e}"}

    prompt = f"""You are an expert bibliographer. Extract the complete bibliographic metadata from this academic document text.

Document text (first pages):
{text}

Extract ALL of the following fields that are present. If a field is not found, use null.

Return ONLY a valid JSON object with these exact keys:
{{
  "title": "full title of the work",
  "authors": ["Author One", "Author Two"],
  "year": "publication year as string",
  "journal": "journal or book title",
  "volume": "volume number",
  "issue": "issue number",
  "pages": "page range e.g. 123-145",
  "doi": "DOI without https://doi.org/ prefix",
  "publisher": "publisher name",
  "url": "URL if present",
  "edition": "edition if book",
  "location": "publisher location if book",
  "type": "article OR book OR chapter OR thesis OR conference OR website"
}}

Return ONLY the JSON object. No explanation. No markdown. No backticks."""

    try:
        raw = llm_call(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0,
        )
        raw = re.sub(r'```[a-z]*', '', raw).replace('```', '').strip()
        metadata = json.loads(raw)
        return metadata
    except Exception as e:
        return {"error": f"Metadata extraction failed: {e}"}


def resolve_doi_metadata(doi: str) -> dict:
    """Fetch metadata directly from CrossRef using a DOI."""
    try:
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        r = requests.get(
            f"https://api.crossref.org/works/{clean_doi}",
            headers={"User-Agent": "Lumina/1.0 (lumina@research.app)"},
            timeout=10,
        )
        if r.status_code != 200:
            return {}
        data = r.json().get("message", {})
        authors = []
        for a in data.get("author", [])[:6]:
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(name)
        issued = data.get("issued", {}).get("date-parts", [[None]])
        year = str(issued[0][0]) if issued and issued[0] and issued[0][0] else ""
        journal = ""
        container = data.get("container-title", [])
        if container:
            journal = container[0]
        pages = data.get("page", "")
        volume = data.get("volume", "")
        issue = data.get("issue", "")
        publisher = data.get("publisher", "")
        title_list = data.get("title", [])
        title = title_list[0] if title_list else ""
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "doi": clean_doi,
            "publisher": publisher,
            "url": f"https://doi.org/{clean_doi}",
            "type": "article",
        }
    except Exception as e:
        print(f"CrossRef DOI lookup failed: {e}")
        return {}


def format_citations(meta: dict) -> dict:
    """
    Generate citation strings in 5 academic styles from metadata dict.
    Returns dict with keys: apa, mla, chicago, harvard, ieee, vancouver.
    """
    title   = meta.get("title", "Untitled")
    authors = meta.get("authors", [])
    year    = meta.get("year", "n.d.")
    journal = meta.get("journal", "")
    volume  = meta.get("volume", "")
    issue   = meta.get("issue", "")
    pages   = meta.get("pages", "")
    doi     = meta.get("doi", "")
    publisher = meta.get("publisher", "")
    url     = meta.get("url", "")
    edition = meta.get("edition", "")
    location = meta.get("location", "")
    doc_type = meta.get("type", "article")
    access_date = datetime.now().strftime("%d %B %Y")

    doi_str = f"https://doi.org/{doi}" if doi else url

    def apa_author(name):
        parts = name.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = " ".join(p[0].upper() + "." for p in parts[:-1])
            return f"{last}, {initials}"
        return name

    def mla_author(name, index, total):
        parts = name.strip().split()
        if index == 0 and len(parts) >= 2:
            return f"{parts[-1]}, {' '.join(parts[:-1])}"
        return name

    def chicago_author(name, index):
        parts = name.strip().split()
        if index == 0 and len(parts) >= 2:
            return f"{parts[-1]}, {' '.join(parts[:-1])}"
        return name

    def harvard_author(name):
        parts = name.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = "".join(p[0].upper() + "." for p in parts[:-1])
            return f"{last}, {initials}"
        return name

    def ieee_author(name):
        parts = name.strip().split()
        if len(parts) >= 2:
            initials = ". ".join(p[0].upper() for p in parts[:-1]) + "."
            return f"{initials} {parts[-1]}"
        return name

    # ── APA 7th ────────────────────────────────────────
    if authors:
        if len(authors) == 1:
            apa_auth = apa_author(authors[0])
        elif len(authors) <= 20:
            apa_auth = ", ".join(apa_author(a) for a in authors[:-1]) + f", & {apa_author(authors[-1])}"
        else:
            apa_auth = ", ".join(apa_author(a) for a in authors[:19]) + f", . . . {apa_author(authors[-1])}"
    else:
        apa_auth = "Unknown Author"

    if doc_type == "article":
        apa = f"{apa_auth} ({year}). {title}. *{journal}*"
        if volume: apa += f", *{volume}*"
        if issue:  apa += f"({issue})"
        if pages:  apa += f", {pages}"
        apa += "."
        if doi_str: apa += f" {doi_str}"
    else:
        apa = f"{apa_auth} ({year}). *{title}*"
        if edition: apa += f" ({edition} ed.)"
        apa += f". {publisher}."
        if doi_str: apa += f" {doi_str}"

    # ── MLA 9th ────────────────────────────────────────
    if authors:
        if len(authors) == 1:
            mla_auth = mla_author(authors[0], 0, 1)
        elif len(authors) == 2:
            mla_auth = f"{mla_author(authors[0], 0, 2)}, and {authors[1]}"
        else:
            mla_auth = f"{mla_author(authors[0], 0, len(authors))}, et al."
    else:
        mla_auth = "Unknown Author"

    if doc_type == "article":
        mla = f'{mla_auth}. "{title}." *{journal}*'
        if volume: mla += f", vol. {volume}"
        if issue:  mla += f", no. {issue}"
        if year:   mla += f", {year}"
        if pages:  mla += f", pp. {pages}"
        mla += "."
        if doi_str: mla += f" {doi_str}."
    else:
        mla = f'{mla_auth}. *{title}*.'
        if edition: mla += f" {edition} ed.,"
        mla += f" {publisher}, {year}."

    # ── Chicago 17th (author-date) ──────────────────────
    if authors:
        if len(authors) == 1:
            chi_auth = chicago_author(authors[0], 0)
        elif len(authors) <= 3:
            chi_auth = ", ".join(chicago_author(a, i) for i, a in enumerate(authors))
            chi_auth = chi_auth.rsplit(", ", 1)
            chi_auth = chi_auth[0] + ", and " + chi_auth[1] if len(chi_auth) > 1 else chi_auth[0]
        else:
            chi_auth = f"{chicago_author(authors[0], 0)} et al."
    else:
        chi_auth = "Unknown Author"

    if doc_type == "article":
        chicago = f'{chi_auth}. {year}. "{title}." *{journal}*'
        if volume: chicago += f" {volume}"
        if issue:  chicago += f" ({issue})"
        if pages:  chicago += f": {pages}"
        chicago += "."
        if doi_str: chicago += f" {doi_str}."
    else:
        chicago = f"{chi_auth}. {year}. *{title}*."
        if location and publisher: chicago += f" {location}: {publisher}."
        elif publisher: chicago += f" {publisher}."

    # ── Harvard ────────────────────────────────────────
    if authors:
        if len(authors) == 1:
            harv_auth = harvard_author(authors[0])
        elif len(authors) <= 3:
            harv_auth = ", ".join(harvard_author(a) for a in authors[:-1]) + f" and {harvard_author(authors[-1])}"
        else:
            harv_auth = f"{harvard_author(authors[0])} et al."
    else:
        harv_auth = "Unknown Author"

    if doc_type == "article":
        harvard = f"{harv_auth} ({year}) '{title}', *{journal}*"
        if volume: harvard += f", {volume}"
        if issue:  harvard += f"({issue})"
        if pages:  harvard += f", pp.{pages}"
        harvard += "."
        if doi_str: harvard += f" Available at: {doi_str} (Accessed: {access_date})."
    else:
        harvard = f"{harv_auth} ({year}) *{title}*."
        if edition: harvard += f" {edition} edn."
        if location and publisher: harvard += f" {location}: {publisher}."
        elif publisher: harvard += f" {publisher}."

    # ── IEEE ───────────────────────────────────────────
    if authors:
        ieee_auth = ", ".join(ieee_author(a) for a in authors[:6])
        if len(authors) > 6: ieee_auth += " et al."
    else:
        ieee_auth = "Unknown Author"

    ieee_num = "[1]"
    if doc_type == "article":
        ieee = f'{ieee_num} {ieee_auth}, "{title}," *{journal}*'
        if volume: ieee += f", vol. {volume}"
        if issue:  ieee += f", no. {issue}"
        if pages:  ieee += f", pp. {pages}"
        if year:   ieee += f", {year}"
        ieee += "."
        if doi_str: ieee += f" doi: {doi_str}."
    else:
        ieee = f'{ieee_num} {ieee_auth}, *{title}*.'
        if edition: ieee += f" {edition} ed."
        if location: ieee += f" {location}:"
        if publisher: ieee += f" {publisher},"
        if year: ieee += f" {year}."

    # ── Vancouver ──────────────────────────────────────
    def van_author(name):
        parts = name.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = "".join(p[0].upper() for p in parts[:-1])
            return f"{last} {initials}"
        return name

    if authors:
        van_list = [van_author(a) for a in authors[:6]]
        van_auth = ", ".join(van_list)
        if len(authors) > 6: van_auth += ", et al."
    else:
        van_auth = "Unknown Author"

    if doc_type == "article":
        vancouver = f"{van_auth}. {title}. *{journal}*. {year}"
        if volume: vancouver += f";{volume}"
        if issue:  vancouver += f"({issue})"
        if pages:  vancouver += f":{pages}"
        vancouver += "."
        if doi_str: vancouver += f" doi:{doi_str}."
    else:
        vancouver = f"{van_auth}. {title}."
        if edition: vancouver += f" {edition} ed."
        if location: vancouver += f" {location}:"
        if publisher: vancouver += f" {publisher};"
        if year: vancouver += f" {year}."

    return {
        "apa":       apa,
        "mla":       mla,
        "chicago":   chicago,
        "harvard":   harvard,
        "ieee":      ieee,
        "vancouver": vancouver,
    }


@app.route('/cite', methods=['POST'])
@limiter.limit("20 per hour")
@limiter.limit("5 per minute")
def cite():
    """
    Citation tool endpoint. Accepts:
    - PDF file upload (multipart/form-data, field name: 'file')
    - DOI string (JSON body: {"doi": "10.xxxx/xxxx"})
    Returns structured metadata + citations in 6 formats.
    """
    # ── DOI input path ──────────────────────────────────
    if request.is_json:
        data = request.get_json(silent=True) or {}
        doi = sanitise_short(data.get("doi", ""), max_len=200)
        if not doi:
            return jsonify({"error": "No DOI provided"}), 400
        meta = resolve_doi_metadata(doi)
        if not meta:
            return jsonify({"error": "Could not find paper with that DOI. Check the DOI and try again."}), 404
        citations = format_citations(meta)
        return jsonify({"metadata": meta, "citations": citations})

    # ── PDF upload path ─────────────────────────────────
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded. Send a PDF as multipart/form-data with field name 'file'."}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Validate file type by content, not just extension
    pdf_bytes = file.read(5)
    if not pdf_bytes.startswith(b'%PDF'):
        return jsonify({"error": "File does not appear to be a valid PDF."}), 400

    # Read the rest
    pdf_bytes = pdf_bytes + file.read()
    if len(pdf_bytes) > 20 * 1024 * 1024:
        return jsonify({"error": "PDF too large. Maximum size is 20MB."}), 413

    meta = extract_metadata_from_pdf(pdf_bytes)
    if "error" in meta:
        return jsonify({"error": meta["error"]}), 500

    citations = format_citations(meta)
    return jsonify({"metadata": meta, "citations": citations})


# ── HEALTH CHECK ───────────────────────────────────────
@app.route('/health')
@limiter.exempt
def health():
    return jsonify({'status': 'ok', 'service': 'lumina'}), 200


# ── 413 HANDLER ────────────────────────────────────────
@app.errorhandler(413)
def request_too_large(e):
    return jsonify({'error': 'Request too large'}), 413


if __name__ == '__main__':
    env = os.getenv('FLASK_ENV', 'development')
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=(env == 'development'))
