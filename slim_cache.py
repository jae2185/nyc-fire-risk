"""Create a slim model cache without raw_df for Streamlit Cloud deployment"""
import pickle
from pathlib import Path

with open(".cache/model_cache.pkl", "rb") as f:
    data = pickle.load(f)

print("Original keys:", list(data.keys()))
print(f"raw_df: {data['raw_df'].shape} = {data['raw_df'].memory_usage(deep=True).sum() / 1e6:.0f} MB")

# Save without raw_df
slim = {k: v for k, v in data.items() if k != "raw_df"}
slim["raw_df_summary"] = {
    "years": sorted(data["raw_df"]["year"].unique().tolist()),
    "n_records": len(data["raw_df"]),
}

slim_path = Path("data/model_cache_slim.pkl")
with open(slim_path, "wb") as f:
    pickle.dump(slim, f)

print(f"\nSlim cache: {slim_path} ({slim_path.stat().st_size / 1e6:.1f} MB)")
print("Keys:", list(slim.keys()))

