from dotenv import load_dotenv; load_dotenv()
from pathlib import Path
from pydantic import BaseModel
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / ".data"
CACHE_DIR = ROOT / ".cache"
RUNS_DIR = ROOT / "runs"
for d in (DATA_DIR, CACHE_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    cfbd_key: str | None = os.getenv("CFBD_KEY")
    nws_user_agent: str = os.getenv("NWS_USER_AGENT", "cfmp/0.0.1 (contact@nowhere)")

SETTINGS = Settings()
