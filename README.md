# Fork & Fetch 🍴
Fork & Fetch is a Retrieval-Augmented Generation (RAG) powered recipe assistant. It enables users to search for recipe suggestions based on their queries using a combination of semantic search and generative AI. Think of it as your smart AI-powered cookbook!

## 📌 Features
- 🔍 Semantic Recipe Search using FAISS and HuggingFace sentence-transformers.
- 🍳 Intelligent Recipe Suggestions powered by LLaMA 3 through Ollama.
- 🧠 Uses LangChain's RetrievalQA chain to combine retrieved recipes with LLM reasoning.
- 🎨 Built with Streamlit for an interactive UI.

## 📂 Project Structure
- [**Dataset_preprocessing.ipynb:**](https://github.com/Kr1mson/ForknFetch/blob/main/Dataset_preprocessing.ipynb) Handles the cleaning and preparation of the recipe dataset for embedding and retrieval.
- [**Embedder.py:**](https://github.com/Kr1mson/ForknFetch/blob/main/embedder.py) Demonstrates the process of creating embeddings from the recipe dataset using HuggingFace's sentence transformers.
- [**Fork&Fetch.py:**](https://github.com/Kr1mson/ForknFetch/blob/main/Fork%26Fetch.py) Contains the complete RAG implementation for recipe search and recommendation with a user-friendly Streamlit UI.

## 📈 Results

### Automated Metrics

| Metric   | Type         | Base LLM | RAG  | Improvement (%) |
|----------|--------------|----------|------|-----------------|
| **BLEU** | 1-gram       | 0.69     | 0.76 | +8.9%           |
|          | 2-gram       | 0.62     | 0.69 | +11.2%          |
|          | 3-gram       | 0.55     | 0.64 | +18.1%          |
|          | 4-gram       | 0.49     | 0.61 | +26.7%          |
| **Rouge-1** | F1 score  | 0.69     | 0.85 | +22.95%         |
|          | Precision    | 0.69     | 0.90 | +30.64%         |
|          | Recall       | 0.70     | 0.81 | +15.95%         |
| **Rouge-2** | F1 score  | 0.42     | 0.71 | +69.20%         |
|          | Precision    | 0.41     | 0.76 | +83.09%         |
|          | Recall       | 0.43     | 0.68 | +56.19%         |
| **Rouge-L** | F1 score  | 0.52     | 0.79 | +51.23%         |
|          | Precision    | 0.52     | 0.84 | +61.15%         |
|          | Recall       | 0.54     | 0.75 | +41.03%         |


## 📚 Dataset
The recipe dataset was sourced from [**Eight Portions**](https://eightportions.com/datasets/Recipes/#fn:1) and contains approximately 125,000 recipes scraped from various food websites. Each recipe in the dataset includes:
- Recipe title
- List of ingredients and measurements
- Instructions for preparation
- Source URL
- Picture (in some cases)

## 🛠️ Technology Stack
* **Vector Database**: FAISS for efficient similarity search
* **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
* **LLM**: Ollama with llama3.2 model
* **RAG Framework**: LangChain
* **Frontend**: Streamlit
* **GPU Acceleration**: CUDA support (when available)

## 📋 Prerequisites
* Python 3.8+
* CUDA-compatible GPU (optional, for faster processing)
* Ollama with llama3.2 model installed
