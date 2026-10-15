from cfmp.ingest import cfbd
import pandas as pd, os

os.makedirs("data", exist_ok=True)
df = pd.DataFrame(cfbd.venues())
keep = [c for c in ("id", "name", "latitude", "longitude", "dome") if c in df.columns]
df[keep].to_parquet("data/venues_min.parquet", index=False)
print(f"Saved {len(df)} venues to data/venues_min.parquet")
