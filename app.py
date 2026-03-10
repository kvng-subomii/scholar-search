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


# ── ARXIV ──────────────────────────────────────────────
def search_arxiv(query: str, max_results=5) -> list:
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
            link = entry.find("atom:id", ns).text.strip()
            year = entry.find("atom:published", ns).text[:4]
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            papers.append({
                "title": title,
                "authors": authors[:3],
                "abstract": abstract[:400],
                "year": year,
                "journal": "arXiv",
                "link": link,
                "source": "arXiv",
            })
        return papers
    except Exception as e:
        print(f"arXiv error: {e}")
        return []


# ── SEMANTIC SCHOLAR ───────────────────────────────────
def search_semantic_scholar(query: str, max_results=5) -> list:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,abstract,year,venue,externalIds,openAccessPdf",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        papers = []
        for p in data.get("data", []):
            pdf = p.get("openAccessPdf") or {}
            link = pdf.get("url", "")
            if not link:
                ext = p.get("externalIds", {})
                if ext.get("DOI"):
                    link = f"https://doi.org/{ext['DOI']}"
                elif ext.get("ArXiv"):
                    link = f"https://arxiv.org/abs/{ext['ArXiv']}"
            papers.append({
                "title": p.get("title", ""),
                "authors": [a["name"] for a in p.get("authors", [])[:3]],
                "abstract": (p.get("abstract") or "")[:400],
                "year": str(p.get("year", "")),
                "journal": p.get("venue", "Semantic Scholar"),
                "link": link,
                "source": "Semantic Scholar",
            })
        return papers
    except Exception as e:
        print(f"Semantic Scholar error: {e}")
        return []


# ── PUBMED ─────────────────────────────────────────────
def search_pubmed(query: str, max_results=5) -> list:
    try:
        # Step 1: get IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        r = requests.get(search_url, params=params, timeout=10)
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Step 2: fetch details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        r = requests.get(fetch_url, params=params, timeout=10)
        root = ET.fromstring(r.text)

        papers = []
        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            title = title_el.text if title_el is not None else ""

            abstract_el = article.find(".//AbstractText")
            abstract = abstract_el.text if abstract_el is not None else ""

            year_el = article.find(".//PubDate/Year")
            year = year_el.text if year_el is not None else ""

            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text if journal_el is not None else "PubMed"

            authors = []
            for author in article.findall(".//Author")[:3]:
                last = author.find("LastName")
                fore = author.find("ForeName")
                if last is not None:
                    name = last.text
                    if fore is not None:
                        name += f" {fore.text}"
                    authors.append(name)

            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            papers.append({
                "title": title,
                "authors": authors,
                "abstract": (abstract or "")[:400],
                "year": year,
                "journal": journal,
                "link": link,
                "source": "PubMed",
            })
        return papers
    except Exception as e:
        print(f"PubMed error: {e}")
        return []


# ── CROSSREF ───────────────────────────────────────────
def search_crossref(query: str, max_results=5) -> list:
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": max_results,
        "select": "title,author,abstract,published,container-title,DOI",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        items = r.json().get("message", {}).get("items", [])
        papers = []
        for item in items:
            title = item.get("title", [""])[0]
            authors = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in item.get("author", [])[:3]
            ]
            abstract = re.sub(r'<[^>]+>', '', item.get("abstract", ""))[:400]
            pub = item.get("published", {}).get("date-parts", [[""]])[0]
            year = str(pub[0]) if pub else ""
            journal = item.get("container-title", ["CrossRef"])[0]
            doi = item.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else ""
            papers.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "link": link,
                "source": "CrossRef",
            })
        return papers
    except Exception as e:
        print(f"CrossRef error: {e}")
        return []


# ── AI RANKING ─────────────────────────────────────────
def rank_batch(topic: str, papers: list) -> list:
    """Score a batch of papers with AI."""
    papers_text = ""
    for i, p in enumerate(papers):
        papers_text += f"{i+1}. Title: {p['title']}\n"
        papers_text += f"   Abstract: {p['abstract']}\n\n"

    prompt = f"""You are an academic research assistant. A student is researching: \"{topic}\"

Analyze each paper below and return a JSON array. Each object must have:
- "index": 1-based result number (within this batch)
- "score": relevance score 1-10
- "relevance": 1-2 sentences explaining exactly why this paper is or isn't useful for this topic
- "key_contribution": 1 sentence on what this paper contributes to the field

Return ONLY the raw JSON array. No markdown, no backticks.

Papers:
{papers_text}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=3000,
    )

    import re as _re
    raw = response.choices[0].message.content.strip()
    raw = _re.sub(r'^```[a-z]*\n?', '', raw)
    raw = _re.sub(r'\n?```$', '', raw)
    raw = raw.strip()
    match = _re.search(r'\[.*\]', raw, _re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        import json as _json
        ai_data = _json.loads(raw)
    except Exception as e:
        print(f"JSON parse error in batch: {e}")
        ai_data = [{"index": i+1, "score": 5, "relevance": "", "key_contribution": ""} for i in range(len(papers))]

    ai_map = {item["index"]: item for item in ai_data}
    enriched = []
    for i, p in enumerate(papers):
        ai = ai_map.get(i + 1, {})
        enriched.append({
            **p,
            "score": ai.get("score", 1),
            "relevance": ai.get("relevance", ""),
            "key_contribution": ai.get("key_contribution", ""),
        })
    return enriched


def rank_papers_with_ai(topic: str, papers: list) -> list:
    """Split papers into batches of 10, score each, merge and sort up to 30."""
    BATCH_SIZE = 10
    all_enriched = []
    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i:i + BATCH_SIZE]
        print(f"Ranking batch {i//BATCH_SIZE + 1} ({len(batch)} papers)...")
        enriched = rank_batch(topic, batch)
        all_enriched.extend(enriched)
    ranked = sorted(all_enriched, key=lambda x: x["score"], reverse=True)
# Only return papers with a relevance score of 5 or above
    relevant = [p for p in ranked if p["score"] >= 5]
    return relevant[:30]


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
        all_papers = []
        all_papers += search_arxiv(topic, max_results=10)
        all_papers += search_semantic_scholar(topic, max_results=10)
        all_papers += search_pubmed(topic, max_results=10)
        all_papers += search_crossref(topic, max_results=10)

        print(f"Total raw papers: {len(all_papers)}")

        # Remove papers with no title or abstract
        all_papers = [p for p in all_papers if p["title"] and p["abstract"]]

        ranked = rank_papers_with_ai(topic, all_papers)
        return jsonify({'results': ranked})
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
