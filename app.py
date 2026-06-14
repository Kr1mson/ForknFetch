import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(embedding_function=embedder, persist_directory="./chroma_db")

template = """
You are an AI chef assistant. The user wants: {question}

Here are some relevant recipes:
{context}

Based on these, suggest the best fitting recipe. Format your response as:

**Recipe name**
A one-sentence description.

**Ingredients**
- List each ingredient

**Instructions**
1. Step-by-step instructions

Keep it concise and practical.
"""

prompt = PromptTemplate(input_variables=["question", "context"], template=template)
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)
retriever = db.as_retriever(search_kwargs={"k": 3})
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    query = (data or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400
    try:
        response = rag_chain.invoke(query)
        return jsonify({"result": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "device": device, "provider": "openrouter"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)