# 🚀 Quick Start Guide - Local RAG System

## Zero Installation - SQLite Version!

## Step-by-Step Setup

### ✅ STEP 1: Install Python Dependencies

```bash
cd "c:\Users\sandeepk\Favorites\AGENT PRac\RAG"
pip install -r requirements.txt
```

This installs:

- `pypdf` - PDF text extraction
- `sentence-transformers` - FREE embedding model
- `python-dotenv` - Environment variables

**Expected time:** 1-2 minutes
**Note:** SQLite comes with Python - no database installation needed!

---

### ✅ STEP 2: Install Ollama (LLM)

1. Download: https://ollama.ai
2. Install Ollama
3. Pull a model:

```bash
ollama pull mistral
```

**Alternative models:**

```bash
ollama pull llama2        # Faster, less accurate
ollama pull phi           # Smallest (2GB)
ollama pull mixtral       # Best quality (larger)
```

---

### ✅ STEP 3: Setup Database (Optional)

```bash
python setup_database.py
```

This creates the SQLite database file `rag_database.db` with the documents table.
**Note:** The database is created automatically when you run main.py, so this step is optional.

---

### ✅ STEP 4: Test the System

```bash
python main.py
```

**Interactive Menu:**

```
1. Ingest PDF       ← Start here
2. Ask Question     ← Query your documents
3. Show Statistics  ← View system info
4. Clear Database   ← Reset if needed
5. Exit
```

---

## 🎯 Your First RAG Query

### 1. Prepare a PDF

- Find any PDF document
- Place it in your workspace
- Example: `sample.pdf`

### 2. Run the system

```bash
python main.py
```

### 3. Ingest the PDF

- Select option `1` (Ingest PDF)
- Enter PDF path: `sample.pdf`
- Wait for processing (~10-30 seconds)

### 4. Ask Questions

- Select option `2` (Ask Question)
- Type your question
- Get answer with sources!

---

## 📋 Example Session

### Scenario 1: Local Documents (Good Match)

```
🚀 Local RAG System
============================================================

Options:
  1. Ingest PDF
  2. Ask Question
  3. Show Statistics
  4. Clear Database
  5. Exit

Select option (1-5): 1

Enter PDF file path: systematic_trading.pdf

📥 INGESTING PDF: systematic_trading.pdf
============================================================
📄 Loading PDF: systematic_trading.pdf
   Pages: 324
✅ Extracted 458,230 characters from PDF
✅ Created 892 chunks (sentence-based)
🔄 Generating embeddings...
✅ Successfully ingested 892 chunks
📊 Total documents in DB: 892

Select option (1-5): 2

Enter your question: What is the Sharpe ratio?

❓ QUERY: What is the Sharpe ratio?
============================================================
[SEARCH] Searching local documents...
[SUCCESS] Found 5 relevant documents
[GRADING] Grading document relevance...
[INFO] 4/5 documents graded as relevant
[GENERATE] Generating answer...
[INFO] Hallucination check: PASSED
[INFO] Answer usefulness: USEFUL

============================================================
💡 ANSWER:
The Sharpe ratio is a measure of risk-adjusted return that calculates
the excess return per unit of risk. It's calculated as (Return - Risk-free
Rate) / Standard Deviation. A higher Sharpe ratio indicates better
risk-adjusted performance.
============================================================

📚 SOURCES:

[Source 1] (distance: 0.1845) ← Highly relevant
The Sharpe ratio, developed by William Sharpe, measures the performance
of an investment compared to a risk-free asset, after adjusting for risk...

[Source 2] (distance: 0.2134) ← Highly relevant  
In systematic trading, the Sharpe ratio is crucial for evaluating
strategy performance. A ratio above 1.0 is considered good...

[Source 3] (distance: 0.3421) ← Relevant
Calculation: Sharpe Ratio = (Rp - Rf) / σp where Rp is portfolio return,
Rf is risk-free rate, and σp is standard deviation...

[Source 4] (distance: 0.4567) ← Relevant
Traders use the Sharpe ratio to compare different strategies and
optimize their portfolio allocation...

📊 QUERY STATISTICS:
Data Source: local
Sources Used: 4
============================================================
```

### Scenario 2: Web Search Fallback

```
Select option (1-5): 2

Enter your question: Who won the 2024 Nobel Prize in Physics?

❓ QUERY: Who won the 2024 Nobel Prize in Physics?
============================================================
[SEARCH] Searching local documents...
[SUCCESS] Found 3 relevant documents
[GRADING] Grading document relevance...
[WARNING] 0/3 documents graded as relevant
[INFO] No relevant local documents found
[WEB SEARCH] Searching the web...
[SUCCESS] Found 5 web results
[GENERATE] Generating answer from web sources...

============================================================
💡 ANSWER:
The 2024 Nobel Prize in Physics was awarded to John Hopfield and
Geoffrey Hinton for their foundational work on artificial neural
networks and machine learning that has enabled modern AI systems.
============================================================

🌐 SOURCES (from web):

[Source 1] (distance: 1.0000) ← Web source
**Nobel Prize 2024: Physics Laureates Announced**
Source: https://www.nobelprize.org/prizes/physics/2024/

The Royal Swedish Academy of Sciences awarded the 2024 Nobel Prize
in Physics to John Hopfield and Geoffrey Hinton...

[Source 2] (distance: 1.0000) ← Web source
**AI Pioneers Win Nobel Prize in Physics**
Source: https://www.nature.com/articles/...

Hinton and Hopfield's work on neural networks laid the foundation
for modern deep learning...

[Source 3] (distance: 1.0000) ← Web source
**2024 Physics Nobel: Machine Learning Origins** 
Source: https://www.science.org/...

The prize recognizes their contributions from the 1980s that enabled
today's AI revolution...

📊 QUERY STATISTICS:
Data Source: web
Sources Used: 3
============================================================
```

---

## 🔧 Troubleshooting

### Ollama Not Found

```bash
# Check installation
ollama --version

# Add to PATH if needed
```

### Python Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Database File Permission Error

The SQLite file is created automatically in your project folder. If you get permission errors, run as administrator or check folder permissions.

---

## 💡 What You've Built

```
┌─────────────────────────────────────────────────────┐
│        YOUR ADAPTIVE RAG SYSTEM (Zero Installation!)        │
├─────────────────────────────────────────────────────┤
│                                                           │
│  INGESTION PHASE:                                          │
│  PDF → pypdf (text extraction)                            │
│       ↓                                                   │
│  Text → Sentence-based chunking                           │
│       ↓                                                   │
│  Chunks → sentence-transformers (all-MiniLM-L6-v2)        │
│       ↓                                                   │
│  Vectors → SQLite (384-dim embeddings)                    │
│                                                           │
├─────────────────────────────────────────────────────┤
│  QUERY PHASE (Adaptive RAG):                              │
│                                                           │
│  Query → Embed question (384-dim)                        │
│       ↓                                                   │
│  ┌─────┴──────────────────────────────────────────────┐  │
│  │ ADAPTIVE RAG ROUTER (Intelligent Decisions)      │  │
│  ├──────────────────────────────────────────────┤  │
│  │                                                  │  │
│  │ 1. Vector Search (cosine similarity)          │  │
│  │    → Retrieves top-5 docs with distances       │  │
│  │                                                  │  │
│  │ 2. Grade Documents (LLM relevance check)      │  │
│  │    → Filters irrelevant docs                  │  │
│  │    → Keyword fallback if LLM fails           │  │
│  │                                                  │  │
│  │ 3. Decision Point                             │  │
│  │    ├─ Relevant docs? → Generate Answer      │  │
│  │    └─ No relevant docs? → Web Search       │  │
│  │                                                  │  │
│  │ 4. Generate Answer (Ollama llama3.2:3b)       │  │
│  │                                                  │  │
│  │ 5. Check Hallucinations                       │  │
│  │    → Validates answer vs sources            │  │
│  │                                                  │  │
│  │ 6. Grade Answer (useful?)                     │  │
│  │    ├─ Yes → Return answer                   │  │
│  │    └─ No → Transform query & retry (3x)   │  │
│  │                                                  │  │
│  └──────────────────────────────────────────────┘  │
│       │                                                   │
│  ┌────┴──────────────────────────────────────────────┐  │
│  │ WEB SEARCH FALLBACK (DuckDuckGo)             │  │
│  ├──────────────────────────────────────────────┤  │
│  │ Triggered when:                               │  │
│  │  • No local documents found                   │  │
│  │  • All docs graded irrelevant                 │  │
│  │  • Answer not useful after retries           │  │
│  │                                                  │  │
│  │ Returns: 5 web results (title + URL + text)   │  │
│  │ Distance: 1.0 (indicates external source)     │  │
│  └──────────────────────────────────────────────┘  │
│       ↓                                                   │
│  Final Answer + Sources + Distance Metrics              │
│                                                           │
└─────────────────────────────────────────────────────┘
```

---

## 🎓 Key Concepts

### 1. **Embeddings**

- Converts text → 384-dim vectors
- Similar text = similar vectors
- Model: `all-MiniLM-L6-v2` (FREE)

### 2. **Vector Search**

```python
# Manual cosine similarity calculation
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sqrt(sum(a * a for a in vec1))
    magnitude2 = sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2)
```

Iterates through all stored vectors and finds the most similar ones.

### 3. **RAG Pattern**

1. Retrieve relevant context (similarity search)
2. Augment LLM prompt (add context to question)
3. Generate answer (Ollama with context)

---

## 🚀 Next Steps

1. ✅ **Test with your own PDFs**
   - Try different document types
   - Test with multiple PDFs
   
2. ✅ **Experiment with questions**
   - Ask questions from your PDFs (local search)
   - Ask questions outside PDF scope (web search)
   - Compare distance metrics to understand relevance
   
3. ✅ **Understand distance metrics**
   - Watch for patterns in distance values
   - Distance 1.0 = web source
   - Lower distance = more relevant to your docs
   
4. ✅ **Configure Adaptive RAG**
   - Edit `config.py` or `.env` file
   - Toggle `ENABLE_GRADING`, `ENABLE_WEB_SEARCH`, `ENABLE_HALLUCINATION_CHECK`
   - Adjust `MAX_RETRIES` for query transformations
   - Set `VERBOSE=true` to see detailed decision-making
   
5. ✅ **Try different LLM models**
   - `ollama pull mistral` - More powerful
   - `ollama pull phi` - Faster, lighter
   - `ollama pull mixtral` - Best quality
   
6. ✅ **Optimize chunking**
   - Adjust `CHUNK_SIZE` in config
   - Current: Sentence-based (adaptive)
   - Try fixed sizes: 300, 500, 1000 chars
   
7. ✅ **Monitor query logs**
   - Check `rag_queries.log` for all queries
   - Analyze which queries use local vs web
   - Identify grading patterns
   
8. ✅ **Scale up**
   - For >10k documents, consider PostgreSQL + pgvector
   - Current SQLite works well for <10k docs

---

## 📚 Files Created

- `config.py` - Configuration management (Adaptive RAG settings)
- `database.py` - SQLite operations with cosine similarity
- `embeddings.py` - Sentence-transformers wrapper
- `pdf_loader.py` - PDF text extraction
- `chunking.py` - Sentence-based text splitting
- `llm.py` - Ollama LLM integration
- `adaptive_rag_graph.py` - Intelligent routing & validation (NEW!)
- `rag_pipeline.py` - Main orchestration with distance preservation
- `setup_database.py` - Database setup (optional)
- `main.py` - Interactive CLI
- `requirements.txt` - Python dependencies
- `.env.example` - Configuration template
- `rag_queries.log` - Query logging with routing decisions
- `rag_database.db` - SQLite vector database (auto-created)

---

**🎉 You now have a production-grade RAG system running locally!**

No API keys, no cloud costs, complete control.
