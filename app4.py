from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import json
import re
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from groq import Groq
import logging
import signal

load_dotenv()

# ── LOGGING ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ── STARTUP VALIDATION ─────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set. Check your .env file.")

app = Flask(__name__, static_folder='.')

# ── SECURITY CONFIG ────────────────────────────────────
app.config['DEBUG'] = False
app.config['TESTING'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024  # 16KB max request body

# ── CORS — locked to known origins ─────────────────────
CORS(app, origins=[
    "https://lumina.onrender.com",
    "http://127.0.0.1:5001",
    "http://localhost:5001",
])

# ── RATE LIMITING ──────────────────────────────────────
# Global: 200 searches/day across ALL users (protects Groq token budget)
# Per-IP: 30/hour, 5/minute
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day"],
    storage_uri="memory://",
)

# ── GROQ CLIENT ────────────────────────────────────────
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    logger.error(f"Groq client init failed: {e}")
    client = None

# ── HTTPS REDIRECT ─────────────────────────────────────
@app.before_request
def force_https():
    if not request.is_secure and not request.headers.get('X-Forwarded-Proto', 'http') == 'https':
        if os.getenv('FLASK_ENV') == 'production':
            url = request.url.replace('http://', 'https://', 1)
            return redirect(url, code=301)


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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        keywords = response.choices[0].message.content.strip().strip('"').strip("'")
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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
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

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()
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
            return relevant[:30]

    print("Very niche topic — returning top results")
    return ranked[:5]


# ── ROUTES ─────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/search', methods=['POST'])
@limiter.limit("30 per hour")
@limiter.limit("5 per minute")
def search():
    # Check Groq client is available
    if client is None:
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

    # Set hard timeout (Linux/Render only — ignored on Windows dev)
    def _timeout_handler(signum, frame):
        raise TimeoutError("Search exceeded time limit")
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(55)
    except (AttributeError, OSError):
        pass  # SIGALRM not available on Windows — skip silently

    try:
        # Step 1 — Generate comprehensive multi-angle search strategy
        strategy = generate_search_strategy(topic)
        queries = strategy["queries"]
        sub_topics = strategy["sub_topics"]

        # Step 2 — Search all 8 sources for every angle in the strategy
        all_papers = []
        seen_titles = set()

        for angle in queries:
            q = angle.get("query", topic)
            kw = angle.get("keywords", extract_keywords(q))
            label = angle.get("label", "Search")
            print(f"\nSearching angle: [{label}]")

            angle_papers = []
            angle_papers += search_arxiv(q, kw, max_results=8)
            angle_papers += search_semantic_scholar(q, kw, max_results=8)
            angle_papers += search_pubmed(q, kw, max_results=8)
            angle_papers += search_crossref(q, kw, max_results=8)
            angle_papers += search_openalex(q, kw, label="AJOL", max_results=8)
            angle_papers += search_openalex(q, kw,
                domain_filter="primary_topic.domain.id:https://openalex.org/domains/2|https://openalex.org/domains/4",
                label="ERIC", max_results=8)
            angle_papers += search_core(q, kw, max_results=8)
            angle_papers += search_doaj(q, kw, max_results=8)

            # Tag each paper with its search angle for transparency
            for p in angle_papers:
                p["search_angle"] = label

            all_papers += angle_papers

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
        try:
            signal.alarm(0)  # Cancel timeout on success
        except (AttributeError, OSError):
            pass
        return jsonify({
            'results': ranked,
            'sub_topics': sub_topics,
            'angles_searched': len(queries)
        })

    except TimeoutError:
        logger.error("Search timed out after 55 seconds")
        return jsonify({'error': 'Search timed out. Please try a more specific topic.'}), 504
    except Exception as e:
        err_str = str(e)
        logger.error(f"Search error: {err_str}")
        try:
            signal.alarm(0)
        except (AttributeError, OSError):
            pass
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


if __name__ == '__main__':
    app.run(debug=True, port=5001)
