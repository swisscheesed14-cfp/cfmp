# College Football Matchup Predictor – Project Checklist

_Last updated: October 29, 2025_

---

## 🧩 Environment & Setup
- ✅ Install latest Python 3.x  
- ✅ Install `pipx`, `uv`, `git`, and GitHub CLI (`gh`)  
- ✅ Set up virtual environment & activate in VSCode  
- ✅ Initialize private GitHub repo (`cfmp`)  
- ✅ Add `.gitignore`, `.env`, `.env.example`, and project scaffolding  
- ✅ Add GitHub Secrets (`CFBD_KEY`, `NWS_USER_AGENT`)  
- ✅ Confirm GitHub Actions workflow (`run.yml`) exists  
- ✅ Push initial commit & trigger first action  

---

## 🌐 APIs and Data Ingestion
- ✅ Generate and verify **CFBD API key** (HTTP 200 verified)  
- ✅ Integrate CFBD API endpoints (`/games`, `/venues`)  
- ✅ Add caching for venues locally  
- ✅ Add asynchronous NWS weather retrieval (`hourly_forecast_many`)  
- ✅ Confirm async run completes successfully (<30s runtime)  
- ✅ Validate weather data populates in CSV  

---

## ⚙️ Automation & Versioning
- ✅ Commit final working pipeline to `main`  
- ✅ Verify GitHub Actions triggers on push  
- ⚙️ Schedule daily 6:00 AM CST GitHub Action run *(reminder set)*  
- ⚙️ Add static `data/venues_min.parquet` for Actions *(reminder set)*  

---

## 📊 Core Logic & Data Outputs
- ✅ Generate weekly slate (`slate.csv`, `slate_preview.json`)  
- ✅ Cache runs under `runs/YYYY-Www/`  
- ✅ Add timing and runtime logs  
- ✅ Add forecast skip flag (`CFMP_WEATHER=off`)  
- ⚙️ Add forecast confidence weighting (48 h = 90%+)  
- ⚙️ Add matchup simulation engine (Monte Carlo)  
- ⚙️ Add long-term rating model (2014–present data)  

---

## 💾 Data Storage & Architecture
- ✅ `.cache/` and `runs/` git-ignored  
- ✅ Confirm no sensitive data committed  
- ⚙️ Plan for 2014→present historical backfill  
- ⚙️ Design compact team-week feature store  
- ⚙️ Establish external storage (NVMe local + optional S3/Drive)  
- ⚙️ Decide on GitHub artifact retention policy  

---

### Legend
✅ = Complete  
⚙️ = In progress / planned

---

### Notes
- Venues cache will soon move to `data/venues_min.parquet` for GitHub Actions independence.  
- Monte Carlo simulation and rating model are next milestones after full automation.  
- Weekly Action cron (6:00 AM CST) will be activated after reminders execute.

---
