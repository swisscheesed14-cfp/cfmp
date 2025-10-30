from cfmp.ingest import cfbd
import pandas as pd, os

os.makedirs("data", exist_ok=True)
df = pd.DataFrame(cfbd.venues())[["id", "name", "latitude", "longitude"]]
df.to_parquet("data/venues_min.parquet", index=False)
print(f"Saved {len(df)} venues to data/venues_min.parquet")
