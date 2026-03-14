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

Lumina searches **four academic databases simultaneously**, reads every result using an AI, and returns only the papers genuinely relevant to your research topic — each one scored, explained, and linked.

**Sources searched (10 databases):**
- 📄 **arXiv** — preprints across STEM fields
- 🔬 **Semantic Scholar** — 200M+ papers with semantic understanding
- 🏥 **PubMed** — biomedical and life sciences
- 🔗 **CrossRef** — journals and publications across all disciplines
- 🌍 **AJOL (via OpenAlex)** — African academic journals across all disciplines
- 🔓 **CORE** — 30M+ open access papers including African institutional repositories
- 📂 **DOAJ** — Directory of Open Access Journals, peer-reviewed global coverage
- 🗄️ **BASE** — 300M+ documents from 7000+ academic repositories worldwide
- 🎓 **ERIC** — education, communication, social science, and psychology research
- 🧬 **Europe PMC** — life sciences, public health, and psychology

**What you get back for each paper:**
- A relevance score out of 10 — so you know immediately what to prioritise
- A plain-English summary of what the paper actually found
- A clear explanation of *why* it matches your specific topic
- A direct link so you can read or cite it immediately

Papers scoring below 5/10 are filtered out entirely. No noise.

---

## Built For

Students writing dissertations and research papers who are tired of:
- Spending days finding papers that might be relevant
- Accidentally using uncredible or unrelated sources
- Reading abstracts that go nowhere
- Paying for tools that do half the job

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python + Flask |
| Search APIs | arXiv, Semantic Scholar, PubMed, CrossRef |
| AI Ranking | Groq (LLaMA 3.3 70B) |
| Frontend | Vanilla HTML, CSS, JavaScript |

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

### 4. Add your API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Run
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5001`

---

## Roadmap

- [ ] Google Scholar via SerpAPI (optional paid tier for even broader coverage)
- [ ] Filter results by year range
- [ ] Export results as a formatted reference list
- [ ] Save and revisit past searches

---

## Built By

**Kvng_subomii** — a recent graduate who got tired of the way research worked and decided to fix it.

---

## Contributing

Pull requests are welcome. If you're a student who ran into a gap Lumina didn't cover, open an issue and describe it. That's exactly the kind of feedback that makes this better.

---

*Lumina is free and open source. It always will be.*
