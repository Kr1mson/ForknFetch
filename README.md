# Fork&Fetch 🍴

Fork&Fetch is a Retrieval-Augmented Generation (RAG) powered recipe assistant. Search for recipes by dish name, ingredient, or craving — the AI chef finds and presents the best match from a collection of 125,000 recipes.

## 📌 Features

- 🔍 Semantic recipe search using Chroma and HuggingFace sentence-transformers
- 🍳 Intelligent recipe suggestions powered by LLaMA 3.3 70B via OpenRouter
- 🧠 LangChain LCEL pipeline combining retrieval with LLM reasoning
- 🎨 Polished HTML/CSS frontend served by Flask
- 🐳 Fully containerized with Docker

## 📂 Project Structure

```
fork-and-fetch/
├── app.py                  # Flask API + static file server
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── frontend/
│   └── index.html          # HTML frontend
├── stages/
│   ├── preprocessor.py     # Dataset cleaning and preparation
│   ├── pipeline.py         # Runs preprocessing → embedding in sequence
│   └── embedder.py         # Chroma vector store generation
└── data/                   # Add dataset here (see Setup step 0)
```

## 📈 Results

### Automated Metrics

| Metric      | Type       | Base LLM | RAG  | Improvement (%) |
|-------------|------------|----------|------|-----------------|
| **BLEU**    | 1-gram     | 0.69     | 0.76 | +8.9%           |
|             | 2-gram     | 0.62     | 0.69 | +11.2%          |
|             | 3-gram     | 0.55     | 0.64 | +18.1%          |
|             | 4-gram     | 0.49     | 0.61 | +26.7%          |
| **Rouge-1** | F1 score   | 0.69     | 0.85 | +22.95%         |
|             | Precision  | 0.69     | 0.90 | +30.64%         |
|             | Recall     | 0.70     | 0.81 | +15.95%         |
| **Rouge-2** | F1 score   | 0.42     | 0.71 | +69.20%         |
|             | Precision  | 0.41     | 0.76 | +83.09%         |
|             | Recall     | 0.43     | 0.68 | +56.19%         |
| **Rouge-L** | F1 score   | 0.52     | 0.79 | +51.23%         |
|             | Precision  | 0.52     | 0.84 | +61.15%         |
|             | Recall     | 0.54     | 0.75 | +41.03%         |

## 📚 Dataset

Sourced from [Eight Portions](https://eightportions.com/datasets/Recipes/#fn:1) — ~125,000 recipes scraped from various food websites, each including title, ingredients, instructions, and source URL.

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Vector database | Chroma |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM | LLaMA 3.3 70B via OpenRouter |
| RAG framework | LangChain LCEL |
| Backend | Flask |
| Frontend | HTML / CSS / JS |
| Container | Docker + uv |
| GPU acceleration | CUDA (when available) |

## 🚀 Setup

### 0. Download the dataset

1. Download the dataset from [Eight Portions](https://eightportions.com/datasets/Recipes/#fn:1)
2. Unzip and place the CSV in `data/`:

```
data/
└── recipes_cleaned.csv
```

### 1. Configure environment

```bash
cp .env.example .env
# add your OPENROUTER_API_KEY
```

### 2. Build the vector database (once)

```bash
python stages/pipeline.py
```

### 3. Start the app

```bash
docker compose up --build   # first run
docker compose up           # subsequent runs
```

Then open **http://localhost:5000**.

## 📋 Prerequisites

- Python 3.11+
- Docker
- OpenRouter API key
- CUDA-compatible GPU (optional)