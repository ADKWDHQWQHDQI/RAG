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

Enter PDF file path: sample.pdf

📥 INGESTING PDF: sample.pdf
============================================================
📄 Loading PDF: sample.pdf
   Pages: 10
✅ Extracted 45,230 characters from PDF
✅ Created 95 chunks (size=500, overlap=50)
🔄 Generating embeddings...
✅ Successfully ingested 95 chunks
📊 Total documents in DB: 95

Select option (1-5): 2

Enter your question: What is the main topic of this document?

❓ QUERY: What is the main topic of this document?
============================================================
🔄 Embedding question...
🔍 Searching for top 5 similar documents...
✅ Found 5 relevant documents
🤖 Generating answer...

============================================================
💡 ANSWER:
Based on the provided context, this document discusses...
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
│              YOUR LOCAL RAG SYSTEM                  │
│              (Zero Installation!)                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  PDF → pypdf (text extraction)                      │
│         ↓                                           │
│  Text → Python chunking (500 chars)                 │
│         ↓                                           │
│  Chunks → sentence-transformers (embeddings)        │
│         ↓                                           │
│  Vectors → sentence-transformers (384-dim)          │
│         ↓                                           │
│  Storage → SQLite (rag_database.db)                 │
│         ↓                                           │
│  Query → Cosine similarity (Python calculation)     │
│         ↓                                           │
│  Context → Ollama (LLM generation)                  │
│         ↓                                           │
│  Answer + Sources                                   │
│                                                     │
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
2. ✅ **Try different questions**
3. ✅ **Experiment with chunk sizes**
4. ✅ **Try different LLM models**
5. ✅ **For large datasets (>10k docs), upgrade to PostgreSQL + pgvector**

---

## 📚 Files Created

- `config.py` - Configuration management
- `database.py` - SQLite operations with manual similarity
- `embeddings.py` - Sentence-transformers wrapper
- `pdf_loader.py` - PDF text extraction
- `chunking.py` - Text splitting with overlap
- `llm.py` - Ollama LLM integration
- `rag_pipeline.py` - Main orchestration
- `setup_database.py` - Database setup (optional)
- `main.py` - Interactive CLI
- `requirements.txt` - Python dependencies (only 3!)
- `.env.example` - Configuration template

---

**🎉 You now have a production-grade RAG system running locally!**

No API keys, no cloud costs, complete control.
