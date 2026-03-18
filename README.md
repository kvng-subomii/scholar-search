# Lumina — Academic Research, Finally Made Simple

> *"What used to take my friends days or weeks now takes a single search."*

---

## The Problem That Built This

Final year. Dissertation due. You open Google Scholar and start scrolling.

An hour later you have 40 tabs open, three of them are the same paper, and you still aren't sure if any of them are actually relevant to your topic. You start reading one, it cites another, you chase that one, and suddenly it's 2AM and you've made no real progress.

That was me. That was my friends. That was a lot of people I know.

And then there's the other problem nobody talks about — you end up citing sources without realising they aren't credible. Your supervisor sends back your work. You have to start over.

I finished university in 2025. I built Lumina because no tool existed that did this simply and honestly: *search for papers, read them, and only show you the ones that actually matter for your specific topic.* So I built it myself.

---

## What Lumina Does

Lumina generates a **multi-angle search strategy** from your research topic, searches **8 academic databases simultaneously**, reads every result using an AI, and returns only the papers that genuinely matter — each scored, explained, and linked.

**How it works:**
1. AI analyses your topic and generates up to 6 targeted search angles
2. Each angle is searched across all 8 databases simultaneously
3. Results are deduplicated and pre-filtered by topic relevance in Python
4. AI ranks every paper 1–10 with a relevance explanation and key contribution
5. Only papers scoring 5+ are returned — dynamic threshold handles niche topics

**Sources searched (8 databases):**
- 📄 **arXiv** — preprints across STEM fields
- 🔬 **Semantic Scholar** — 200M+ papers with semantic understanding
- 🏥 **PubMed** — biomedical and life sciences
- 🔗 **CrossRef** — journals and publications across all disciplines
- 🌍 **OpenAlex (AJOL)** — open access papers including African institutional repositories
- 🎓 **OpenAlex Social Sciences** — filtered to social sciences and humanities
- 🔓 **CORE** — 30M+ open access papers from global repositories
- 📂 **DOAJ** — Directory of Open Access Journals, peer-reviewed global coverage

**What you get back for each paper:**
- A relevance score out of 10
- A plain-English explanation of why it matches your topic
- A key contribution summary of what the paper found
- A direct link to read or cite it immediately

---

## Built For

Students writing dissertations and research papers who are tired of:
- Spending days finding papers that might be relevant
- Accidentally using uncredible or unrelated sources
- Reading abstracts that go nowhere
- Paying for tools that do half the job

Particularly built for **African university students** researching niche topics in African literature, social science, humanities, and policy — areas severely underrepresented in mainstream academic search tools.

---

## Features

- **Multi-angle search strategy** — AI detects compound topics and generates up to 6 targeted queries covering exact topic, thematic angles, work-specific searches, author scholarship, and foundational theory
- **AI keyword extraction** — intelligently extracts 5–7 core search terms per angle, preventing API errors from overly long queries
- **Smart pre-filtering** — Python-based keyword matching removes irrelevant papers before AI ranking, saving tokens and improving speed
- **Dynamic relevance threshold** — starts at score 5, drops to 4/3 for niche topics so students always get results
- **Number board trivia game** — 50 randomised research and academic trivia questions to play while the search runs (20–40 seconds)
- **Progress bar** — real-time tracking through Strategy → Searching → Ranking → Done
- **Low-results notice** — when fewer than 8 papers are found, direct links to Google Scholar, JSTOR, and ResearchGate with your query pre-populated
- **Rate limiting** — 5 searches/minute, 30/hour per IP, 200/day global cap
- **Security hardened** — input sanitisation, error message sanitisation, CORS locked, 16KB request cap, startup validation

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3 + Flask |
| AI Inference | Groq — LLaMA 3.3 70B |
| Rate Limiting | Flask-Limiter |
| Search APIs | arXiv, Semantic Scholar, PubMed, CrossRef, OpenAlex, CORE, DOAJ |
| Frontend | Vanilla HTML/CSS/JS |
| Design | Zine-brutalist — Big Shoulders Display + Courier Prime |
| Deployment | Render (production) / ngrok (local sharing) |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/kvng-subomii/scholar-search.git
cd scholar-search
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
CORE_API_KEY=your_core_api_key_here
S2_API_KEY=your_semantic_scholar_key_here
```

- Groq (required): [console.groq.com](https://console.groq.com) — free tier
- CORE (recommended): [core.ac.uk/services/api](https://core.ac.uk/services/api) — free tier
- Semantic Scholar (recommended): [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) — free tier

### 5. Run
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5001`

---

## Deployment

Lumina is deployed on [Render](https://render.com). To deploy your own instance:

1. Connect your GitHub repo to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`
4. Add environment variables: `GROQ_API_KEY`, `CORE_API_KEY`, `FLASK_ENV=production`
5. Set health check path: `/health`

---

## Rate Limits

| Window | Limit |
|---|---|
| Per minute (per IP) | 5 searches |
| Per hour (per IP) | 30 searches |
| Per day (global) | 200 searches |

---

## Roadmap

- [ ] Google Scholar via SerpAPI — surfaces ResearchGate and niche African journal papers
- [ ] Parallel database searching — reduce search time from 5 min to under 1 min
- [ ] Filter results by year range
- [ ] Export results as APA/Harvard/Vancouver reference list
- [ ] Result caching — avoid re-running identical searches
- [ ] Semantic Scholar API key integration — currently rate-limited without one

---

## Known Limitations

- Search takes 3–6 minutes due to sequential API calls across 6 angles
- Groq free tier limits to ~6–8 full searches/day before token limits are hit
- African social science papers on ResearchGate/Academia.edu require a paid SerpAPI key
- Semantic Scholar rate-limits heavily without an API key

---

## Built By

**Kvng_subomii** — a recent graduate who got tired of the way research worked and decided to fix it.

---

## Contributing

Pull requests are welcome. If you're a student who ran into a gap Lumina didn't cover, open an issue and describe it. That's exactly the kind of feedback that makes this better.

---

*Lumina is free and open source. It always will be.*
