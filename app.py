from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import re
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__, static_folder='.')
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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


# ── TOPIC DECOMPOSITION ────────────────────────────────
def decompose_topic(topic: str) -> list:
    """
    Detect compound/complex topics and split into focused sub-topics.
    Returns [original] for simple topics, [sub1, sub2...] for compound ones.
    """
    prompt = f"""You are an expert academic research assistant.

Analyse this research topic and determine if it is COMPOUND (covers multiple distinct subjects, texts, or works joined together).

Research topic: "{topic}"

RULES:
1. SIMPLE topic (one subject/concept) → return original as single item
2. COMPOUND topic (multiple texts, authors, or distinct works joined by "and") → split into focused sub-topics, keeping the thematic angle in each
3. Maximum 4 sub-topics
4. Each sub-topic must be a complete searchable query

Return ONLY a valid JSON array of strings. No explanation. No markdown.

Examples:
"climate change effects on agriculture" → ["climate change effects on agriculture"]
"corruption in Chinua Achebe's Things Fall Apart and Wole Soyinka's Death and the King's Horseman" → ["corruption in Chinua Achebe's Things Fall Apart", "corruption in Wole Soyinka's Death and the King's Horseman"]
"Social Media Advertising and Ad Avoidance Behaviour among Nigerian University Undergraduate Students" → ["Social Media Advertising and Ad Avoidance Behaviour among Nigerian University Undergraduate Students"]

Now analyse: "{topic}"
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        sub_topics = json.loads(raw)
        if isinstance(sub_topics, list) and all(isinstance(s, str) for s in sub_topics) and sub_topics:
            print(f"Decomposed into {len(sub_topics)} sub-topics: {sub_topics}")
            return sub_topics
    except Exception as e:
        print(f"Topic decomposition error: {e}")
    return [topic]


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
    params = {"query": keywords, "limit": max_results, "fields": "title,authors,abstract,year,journal,externalIds,openAccessPdf"}
    try:
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
        papers = []
        for item in data.get("data", []):
            title = item.get("title", "")
            if not title: continue
            doi = (item.get("externalIds") or {}).get("DOI", "")
            pdf = (item.get("openAccessPdf") or {}).get("url", "")
            link = pdf or (f"https://doi.org/{doi}" if doi else f"https://scholar.google.com/scholar?q={requests.utils.quote(title)}")
            papers.append({
                "title": title,
                "authors": [a.get("name", "") for a in (item.get("authors") or [])[:3]],
                "abstract": (item.get("abstract") or "")[:400],
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


def rank_papers_with_ai(topic: str, papers: list) -> list:
    BATCH_SIZE = 10
    all_enriched = []
    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i:i + BATCH_SIZE]
        print(f"Ranking batch {i//BATCH_SIZE + 1} ({len(batch)} papers)...")
        enriched = rank_batch(topic, batch)
        all_enriched.extend(enriched)

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
def search():
    data = request.get_json()
    topic = data.get('topic', '').strip()
    if not topic:
        return jsonify({'error': 'No topic provided'}), 400

    try:
        # Step 1 — Extract keywords and decompose topic simultaneously
        keywords = extract_keywords(topic)
        sub_topics = decompose_topic(topic)

        # Step 2 — For each sub-topic, extract its own keywords and search all sources
        all_papers = []
        seen_titles = set()

        for sub in sub_topics:
            sub_kw = extract_keywords(sub) if len(sub_topics) > 1 else keywords
            sub_papers = []
            sub_papers += search_arxiv(sub, sub_kw, max_results=10)
            sub_papers += search_semantic_scholar(sub, sub_kw, max_results=10)
            sub_papers += search_pubmed(sub, sub_kw, max_results=10)
            sub_papers += search_crossref(sub, sub_kw, max_results=10)
            # OpenAlex general (AJOL)
            sub_papers += search_openalex(sub, sub_kw, label="AJOL", max_results=10)
            # OpenAlex social sciences filter
            sub_papers += search_openalex(sub, sub_kw,
                domain_filter="primary_topic.domain.id:https://openalex.org/domains/2|https://openalex.org/domains/4",
                label="ERIC", max_results=10)
            sub_papers += search_core(sub, sub_kw, max_results=10)
            sub_papers += search_doaj(sub, sub_kw, max_results=10)
            for p in sub_papers:
                p["sub_topic"] = sub
            all_papers += sub_papers

        # Step 3 — Deduplicate by title
        unique_papers = []
        for p in all_papers:
            key = p["title"].lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                unique_papers.append(p)

        # Step 4 — Remove papers with no title or abstract
        unique_papers = [p for p in unique_papers if p.get("title") and p.get("abstract")]
        print(f"Total unique papers with abstracts: {len(unique_papers)}")

        # Step 5 — Rank against original full topic
        ranked = rank_papers_with_ai(topic, unique_papers)
        return jsonify({'results': ranked, 'sub_topics': sub_topics if len(sub_topics) > 1 else []})

    except Exception as e:
        err_str = str(e)
        print(f"Search error: {err_str}")
        if '429' in err_str or 'rate_limit' in err_str or 'Rate limit' in err_str:
            return jsonify({'error': 'rate_limit', 'message': 'Groq daily token limit reached. Please try again in a few minutes.'}), 429
        return jsonify({'error': err_str}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
