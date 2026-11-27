"""Chroma vector index over the processed recipes + semantic search."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RECIPES_PATH = ROOT / "data" / "processed" / "recipes.parquet"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "recipes"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_embedder = None
_recipes = None


def minutes_or_none(value):
    """Cooking time in minutes, or None when the dataset does not know it.
    Missing (NaN / None) and 0 both mean unknown: no real recipe takes 0 minutes."""
    try:
        if value is None or pd.isna(value) or int(value) <= 0:
            return None
    except (TypeError, ValueError):
        return None
    return int(value)


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

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


def _metadata(name, category, total_minutes, calories) -> dict:
    """Chroma metadata. An unknown time is left out (not stored as 0), so a time limit
    never matches a recipe whose time is unknown."""
    meta = {"name": str(name), "category": str(category), "calories": float(calories)}
    minutes = minutes_or_none(total_minutes)
    if minutes is not None:
        meta["total_minutes"] = minutes
    return meta


def build_index(batch_size: int = 256) -> int:
    """(Re)build the Chroma collection from the processed recipes file."""
    import chromadb

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
            _metadata(n, c, t, k)
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
    import chromadb

    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
    embedding = get_embedder().encode([query], normalize_embeddings=True).tolist()
    where = {"total_minutes": {"$lte": int(max_minutes)}} if max_minutes else None
    res = collection.query(query_embeddings=embedding, n_results=n_results, where=where)
    hits = []
    for rid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        minutes = minutes_or_none(meta.get("total_minutes"))  # 0 from an old index = unknown
        if max_minutes and minutes is None:
            continue  # unknown time cannot satisfy a time limit
        hits.append(
            {
                "recipe_id": int(rid),
                **{k: v for k, v in meta.items() if k != "total_minutes"},
                "total_minutes": minutes,
                "distance": round(float(dist), 4),
            }
        )
    return hits
