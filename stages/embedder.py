import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

df = pd.read_csv("data/recipes_cleaned.csv")

documents = [
    Document(
        page_content=row["text"],
        metadata={
            "title": row["title"]
        }
    )
    for _, row in df.iterrows()
]
print(f"Loaded {len(documents)} documents for embedding.")

embedder = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

print("Generating embeddings and saving to Chroma database...")
batch_size = 100
batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
db = Chroma(
    embedding_function=embedder,
    persist_directory="./chroma_db"
)

with tqdm(total=len(documents), desc="Embedding & storing", unit="doc") as pbar:
    for batch in batches:
        db.add_documents(batch)
        pbar.update(len(batch))

print("Chroma database saved to ./chroma_db")