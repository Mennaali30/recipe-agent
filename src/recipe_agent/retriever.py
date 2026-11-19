"""Chroma vector index over the processed recipes + semantic search."""
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
RECIPES_PATH = ROOT / "data" / "processed" / "recipes.parquet"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "recipes"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_embedder = None
_recipes = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def load_recipes() -> pd.DataFrame:
    """Processed recipes, indexed by recipe_id (cached)."""
    global _recipes
    if _recipes is None:
        _recipes = pd.read_parquet(RECIPES_PATH).set_index("recipe_id", drop=False)
    return _recipes


def get_recipe(recipe_id: int) -> dict:
    return load_recipes().loc[int(recipe_id)].to_dict()


def build_index(batch_size: int = 256) -> int:
    """(Re)build the Chroma collection from the processed recipes file."""
    df = pd.read_parquet(RECIPES_PATH)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    model = get_embedder()
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        texts = chunk["search_text"].tolist()
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        metadatas = [
            {
                "name": str(n),
                "category": str(c),
                "total_minutes": int(t) if pd.notna(t) else 0,
                "calories": float(k),
            }
            for n, c, t, k in zip(
                chunk["name"], chunk["category"], chunk["total_minutes"], chunk["calories"]
            )
        ]
        collection.add(
            ids=[str(i) for i in chunk["recipe_id"]],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"indexed {min(start + batch_size, len(df)):,}/{len(df):,}")
    return collection.count()


def search(query: str, n_results: int = 10, max_minutes: int | None = None) -> list[dict]:
    """Semantic search. Returns [{recipe_id, name, category, total_minutes, calories, distance}]."""
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
    embedding = get_embedder().encode([query], normalize_embeddings=True).tolist()
    where = {"total_minutes": {"$lte": int(max_minutes)}} if max_minutes else None
    res = collection.query(query_embeddings=embedding, n_results=n_results, where=where)
    hits = []
    for rid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"recipe_id": int(rid), **meta, "distance": round(float(dist), 4)})
    return hits
