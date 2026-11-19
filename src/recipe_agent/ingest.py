"""Prepare a clean, sampled subset of the Food.com recipes dataset for the RAG index."""
import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

COLUMNS = [
    "RecipeId", "Name", "CookTime", "PrepTime", "TotalTime", "Description",
    "RecipeCategory", "Keywords", "RecipeIngredientQuantities",
    "RecipeIngredientParts", "AggregatedRating", "ReviewCount", "Calories",
    "FatContent", "CarbohydrateContent", "ProteinContent", "SugarContent",
    "RecipeServings", "RecipeInstructions",
]

_ISO = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


def iso_to_minutes(value):
    """Convert an ISO 8601 duration such as 'PT2H15M' to minutes (None if invalid)."""
    if not isinstance(value, str):
        return None
    match = _ISO.match(value)
    if not match:
        return None
    days, hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    return days * 1440 + hours * 60 + minutes + seconds // 60


def as_list(value):
    """Turn a parquet list cell into a clean list of non-empty strings."""
    if value is None or isinstance(value, str) or not hasattr(value, "__iter__"):
        return []
    return [str(x).strip() for x in value if x is not None and str(x).strip()]


def build_ingredient_lines(quantities, parts):
    """Pair each ingredient with its quantity, e.g. '1/4 granulated sugar'."""
    lines = []
    for i, part in enumerate(parts):
        qty = quantities[i] if i < len(quantities) else ""
        qty = "" if qty == "NA" else qty
        lines.append(f"{qty} {part}".strip())
    return lines


def build_search_text(row) -> str:
    """Text that gets embedded: name, category, ingredients and tags (no instructions)."""
    return (
        f"{row['name']}. Category: {row['category']}. "
        f"Ingredients: {', '.join(row['ingredients'])}. "
        f"Tags: {', '.join(row['keywords'][:10])}."
    )


def load_raw(raw_dir, min_reviews, min_rating) -> pd.DataFrame:
    table = pq.read_table(
        Path(raw_dir) / "recipes.parquet",
        columns=COLUMNS,
        filters=[
            ("ReviewCount", ">=", min_reviews),
            ("AggregatedRating", ">=", min_rating),
        ],
    )
    return table.to_pandas()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(
        subset=["Name", "Calories", "RecipeIngredientParts", "RecipeInstructions"]
    ).copy()

    quantities = df["RecipeIngredientQuantities"].map(as_list)
    parts = df["RecipeIngredientParts"].map(as_list)

    out = pd.DataFrame(
        {
            "recipe_id": df["RecipeId"].astype(int),
            "name": df["Name"].astype(str).str.strip(),
            "description": df["Description"].fillna("").astype(str).str.strip(),
            "category": df["RecipeCategory"].fillna("").astype(str),
            "keywords": df["Keywords"].map(as_list),
            "ingredients": parts.map(lambda xs: [x.lower() for x in xs]),
            "ingredient_lines": [build_ingredient_lines(q, p) for q, p in zip(quantities, parts)],
            "instructions": df["RecipeInstructions"].map(as_list),
            "prep_minutes": df["PrepTime"].map(iso_to_minutes),
            "cook_minutes": df["CookTime"].map(iso_to_minutes),
            "total_minutes": df["TotalTime"].map(iso_to_minutes),
            "servings": df["RecipeServings"],
            "rating": df["AggregatedRating"],
            "review_count": df["ReviewCount"],
            "calories": df["Calories"],
            "fat_g": df["FatContent"],
            "carbs_g": df["CarbohydrateContent"],
            "protein_g": df["ProteinContent"],
            "sugar_g": df["SugarContent"],
        }
    )

    out = out[
        (out["ingredients"].map(len) >= 2)
        & (out["instructions"].map(len) >= 2)
        & (out["calories"] > 0)
        & (out["calories"] <= 3000)
    ]

    out["name_key"] = out["name"].str.lower()
    out = out.drop_duplicates(subset="name_key").drop(columns="name_key")
    return out.reset_index(drop=True)


def run(
    raw_dir,
    out_path="data/processed/recipes.parquet",
    n=20000,
    min_reviews=3,
    min_rating=4.0,
    seed=42,
) -> pd.DataFrame:
    raw = load_raw(raw_dir, min_reviews, min_rating)
    print(f"after quality filter: {len(raw):,}")

    df = clean(raw)
    print(f"after cleaning: {len(df):,}")

    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    df["search_text"] = df.apply(build_search_text, axis=1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"saved {len(df):,} recipes -> {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the recipes dataset")
    parser.add_argument("--raw-dir", required=True, help="Folder containing recipes.parquet")
    parser.add_argument("--out", default="data/processed/recipes.parquet")
    parser.add_argument("--n", type=int, default=20000)
    args = parser.parse_args()
    run(args.raw_dir, args.out, args.n)
