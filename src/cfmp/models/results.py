"""Grade the card that was sent, not a spread rebuilt after the game."""

from __future__ import annotations

import pandas as pd


def grade_card(card: pd.DataFrame, played: pd.DataFrame) -> pd.DataFrame:
    """One row per finished game. The worst model miss is first.

    `played` has away, home, and the actual home margin. Positive still
    favors the home team, same as the card.
    """
    need = {"away", "home", "model", "draftkings"}
    if card.empty or not need <= set(card.columns) or played.empty:
        return pd.DataFrame()
    games = card.merge(played, on=["away", "home"], how="inner")
    games = games[pd.to_numeric(games["actual"], errors="coerce").notna()].copy()
    if games.empty:
        return games
    games["model"] = pd.to_numeric(games["model"], errors="coerce")
    games["draftkings"] = pd.to_numeric(games["draftkings"], errors="coerce")
    games["actual"] = pd.to_numeric(games["actual"], errors="coerce")
    games["model_miss"] = (games["model"] - games["actual"]).abs()
    games["dk_miss"] = (games["draftkings"] - games["actual"]).abs()
    games["closer"] = "tie"
    games.loc[games["model_miss"] < games["dk_miss"], "closer"] = "model"
    games.loc[games["dk_miss"] < games["model_miss"], "closer"] = "draftkings"
    likes_home = games["model"] > games["draftkings"]
    likes_away = games["model"] < games["draftkings"]
    games["covered"] = pd.NA
    games.loc[likes_home, "covered"] = games.loc[likes_home, "actual"] > games.loc[likes_home, "draftkings"]
    games.loc[likes_away, "covered"] = games.loc[likes_away, "actual"] < games.loc[likes_away, "draftkings"]
    decided = games["closer"].isin(["model", "draftkings"])
    closer_rate = float((games.loc[decided, "closer"] == "model").mean()) if decided.any() else None
    bets = games["covered"].notna()
    cover_rate = float(games.loc[bets, "covered"].astype(float).mean()) if bets.any() else None
    games["closer_rate"] = None if closer_rate is None else round(100.0 * closer_rate, 1)
    games["cover_rate"] = None if cover_rate is None else round(100.0 * cover_rate, 1)
    games = games.sort_values(["model_miss", "home"], ascending=[False, True])
    columns = [
        "away",
        "home",
        "model",
        "draftkings",
        "actual",
        "model_miss",
        "dk_miss",
        "closer",
        "covered",
        "closer_rate",
        "cover_rate",
    ]
    out = games[columns].copy()
    for column in ("model", "draftkings", "actual", "model_miss", "dk_miss"):
        out[column] = out[column].round(2)
    return out.reset_index(drop=True)
