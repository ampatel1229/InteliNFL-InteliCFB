"""Load and normalize NFL (nflverse) and CFB (CFBD) play-by-play data."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from intelinfl import LEAGUE_CFB, LEAGUE_NFL


def resolve_cfbd_api_key(explicit: str | None = None) -> str | None:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    for env in ("CFBD_API_KEY", "BEARER_TOKEN", "CFBD_BEARER_TOKEN"):
        v = os.environ.get(env)
        if v and str(v).strip():
            return str(v).strip().removeprefix("Bearer ").strip()
    try:
        if hasattr(st, "secrets") and st.secrets.get("CFBD_API_KEY"):
            return str(st.secrets["CFBD_API_KEY"]).strip().removeprefix("Bearer ").strip()
    except (FileNotFoundError, RuntimeError, AttributeError):
        pass
    return None


def cfbd_configuration(api_key: str) -> Any:
    """cfbd v5+ uses ``access_token`` for Bearer auth; ``api_key`` dict is ignored."""
    import cfbd

    configuration = cfbd.Configuration()
    configuration.access_token = api_key
    return configuration


def _cfbd_model_list_to_df(items: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "to_dict"):
            rows.append(item.to_dict())
        else:
            rows.append(dict(item))
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows, sep="_")


@st.cache_data(ttl=86400, show_spinner=False)
def _load_nfl_pbp_raw(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_pbp_data([season])


@st.cache_data(ttl=3600, show_spinner=False)
def load_cfb_games(season: int, api_key: str) -> pd.DataFrame:
    import cfbd
    from cfbd.rest import ApiException

    key = resolve_cfbd_api_key(api_key)
    if not key:
        return pd.DataFrame()
    try:
        with cfbd.ApiClient(cfbd_configuration(key)) as client:
            games_api = cfbd.GamesApi(client)
            games = games_api.get_games(year=season, season_type="regular")
    except ApiException as e:
        if getattr(e, "status", None) in (401, 403):
            raise RuntimeError(
                "CollegeFootballData rejected your API key (401/403). "
                "Set a valid key in `.streamlit/secrets.toml` or env `CFBD_API_KEY`."
            ) from e
        raise
    df = _cfbd_model_list_to_df(games)
    if df.empty:
        return df
    if "homeTeam" in df.columns:
        df = df.rename(columns={"homeTeam": "home_team", "awayTeam": "away_team"})
    hc = df.get("homeClassification", df.get("home_classification"))
    ac = df.get("awayClassification", df.get("away_classification"))
    if hc is not None and ac is not None:
        h = hc.astype(str).str.lower()
        a = ac.astype(str).str.lower()
        df = df[(h == "fbs") | (a == "fbs")]
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_cfb_plays_season(season: int, api_key: str) -> pd.DataFrame:
    import cfbd
    from cfbd.rest import ApiException

    key = resolve_cfbd_api_key(api_key)
    if not key:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    auth_failed = False
    with cfbd.ApiClient(cfbd_configuration(key)) as client:
        plays_api = cfbd.PlaysApi(client)
        for week in range(1, 17):
            try:
                plays = plays_api.get_plays(
                    year=season,
                    week=week,
                    season_type="regular",
                    classification="fbs",
                )
            except ApiException as e:
                if getattr(e, "status", None) in (401, 403):
                    auth_failed = True
                    break
                continue
            except Exception:
                continue
            if not plays:
                continue
            pdf = _cfbd_model_list_to_df(plays)
            if not pdf.empty:
                pdf = pdf.copy()
                pdf["week"] = week
                pdf["season"] = season
                parts.append(pdf)
    if auth_failed:
        raise RuntimeError(
            "CollegeFootballData API authentication failed. "
            "Use a valid key in `.streamlit/secrets.toml` (`CFBD_API_KEY`) or the sidebar field."
        )
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out


def _normalize_cfb_wp_df(df: pd.DataFrame) -> pd.DataFrame:
    """Align CFBD play-by-play WP columns with chart code (snake_case)."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    if "home_win_prob" not in out.columns:
        if "homeWinProbability" in out.columns:
            out["home_win_prob"] = out["homeWinProbability"]
        elif "home_win_probability" in out.columns:
            out["home_win_prob"] = out["home_win_probability"]
    if "play_number" not in out.columns and "playNumber" in out.columns:
        out["play_number"] = out["playNumber"]
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_cfb_wp_game(game_id: int, api_key: str) -> pd.DataFrame:
    import cfbd

    key = resolve_cfbd_api_key(api_key)
    if not key:
        return pd.DataFrame()
    with cfbd.ApiClient(cfbd_configuration(key)) as client:
        metrics = cfbd.MetricsApi(client)
        # cfbd v5+ uses get_win_probability; older docs used get_win_probability_data
        fetch = getattr(metrics, "get_win_probability", None) or getattr(
            metrics, "get_win_probability_data", None
        )
        if fetch is None:
            return pd.DataFrame()
        wps = fetch(game_id=game_id)
    return _normalize_cfb_wp_df(_cfbd_model_list_to_df(wps))


@st.cache_data(ttl=3600, show_spinner=False)
def load_cfb_player_passing_stats(season: int, team: str | None, api_key: str) -> pd.DataFrame:
    import cfbd
    from cfbd.models.season_type import SeasonType

    key = resolve_cfbd_api_key(api_key)
    if not key:
        return pd.DataFrame()
    with cfbd.ApiClient(cfbd_configuration(key)) as client:
        stats_api = cfbd.StatsApi(client)
        stats = stats_api.get_player_season_stats(
            year=season,
            team=team,
            category="passing",
            season_type=SeasonType.REGULAR,
        )
    return _cfbd_model_list_to_df(stats)


def load_cfb_data(season: int, api_key: str) -> pd.DataFrame:
    return load_cfb_plays_season(season, api_key)


def load_data(league: str, season: int, api_key: str = "") -> pd.DataFrame:
    if league == LEAGUE_NFL:
        return _load_nfl_pbp_raw(season)
    if league == LEAGUE_CFB:
        return load_cfb_data(season, api_key)
    raise ValueError(f"Unknown league: {league}")


def normalize_data(df: pd.DataFrame, league: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if league == LEAGUE_NFL:
        out["team"] = out.get("posteam")
        out["opponent"] = out.get("defteam")
        out["epa"] = pd.to_numeric(out.get("epa"), errors="coerce")
        out["play_type"] = out.get("play_type")
        out["yards"] = pd.to_numeric(out.get("yards_gained"), errors="coerce")
        out["_norm_game_id"] = out["game_id"].astype(str) if "game_id" in out.columns else None
    elif league == LEAGUE_CFB:
        yds_src = out["yards_gained"] if "yards_gained" in out.columns else out.get("yardsGained")
        pt_src = out["play_type"] if "play_type" in out.columns else out.get("playType")
        gid_src = out["game_id"] if "game_id" in out.columns else out.get("gameId")
        ytg = out["yards_to_goal"] if "yards_to_goal" in out.columns else out.get("yardsToGoal")
        out["team"] = out.get("offense")
        out["opponent"] = out.get("defense")
        out["yards"] = pd.to_numeric(yds_src, errors="coerce")
        out["play_type"] = pt_src
        if ytg is not None:
            out["yards_to_goal"] = pd.to_numeric(ytg, errors="coerce")
        ppa = pd.to_numeric(out.get("ppa"), errors="coerce")
        out["epa"] = ppa
        if gid_src is not None:
            out["_norm_game_id"] = gid_src.astype(str)
        if "play_id" not in out.columns and "id" in out.columns:
            out["play_id"] = out["id"]
        if "offense_score" not in out.columns and "offenseScore" in out.columns:
            out["offense_score"] = pd.to_numeric(out["offenseScore"], errors="coerce")
        elif "offense_score" in out.columns:
            out["offense_score"] = pd.to_numeric(out["offense_score"], errors="coerce")
        if "defense_score" not in out.columns and "defenseScore" in out.columns:
            out["defense_score"] = pd.to_numeric(out["defenseScore"], errors="coerce")
        elif "defense_score" in out.columns:
            out["defense_score"] = pd.to_numeric(out["defense_score"], errors="coerce")
        if "season" not in out.columns or out["season"].isna().all():
            out["season"] = np.nan
    else:
        raise ValueError(f"Unknown league: {league}")
    return out


def _nfl_offensive_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    pt = df["play_type"].fillna("")
    na = df["play_type"].isna()
    base = df["team"].notna() & (df["team"] != "")
    specials = ~pt.isin(["no_play", "qb_kneel", "qb_spike", "end_of_game", "end_of_half"])
    return base & specials & (na | (pt != ""))


def _cfb_offensive_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    base = df["team"].notna() & (df["team"] != "")
    text = df["play_text"].fillna("").str.lower() if "play_text" in df.columns else pd.Series([""] * len(df))
    kick = text.str.contains("kickoff|timeout|end of|quarter", regex=True)
    if "play_type" in df.columns:
        pt = df["play_type"].fillna("").str.lower()
        kick = kick | pt.str.contains(
            "kickoff|punt|field goal|extra point|timeout|end period|end of half|end of game"
        )
    return base & ~kick


def offensive_plays_mask(df: pd.DataFrame, league: str) -> pd.Series:
    if league == LEAGUE_NFL:
        m = _nfl_offensive_mask(df)
        if "play_type" in df.columns:
            m = m & df["play_type"].isin(["pass", "run"])
        return m
    if league == LEAGUE_CFB:
        m = _cfb_offensive_mask(df)
        if "play_type" not in df.columns:
            return m
        pt = df["play_type"].fillna("").str.lower()
        return m & (pt.str.contains("pass|rush|sack|run") | pt.str.contains("interception|fumble"))
    return pd.Series(True, index=df.index)


def is_pass_play(row: pd.Series, league: str) -> bool:
    pt = str(row.get("play_type") or "").lower()
    if league == LEAGUE_NFL:
        return pt == "pass"
    return "pass" in pt or "sack" in pt or "interception" in pt


def is_rush_play(row: pd.Series, league: str) -> bool:
    pt = str(row.get("play_type") or "").lower()
    if league == LEAGUE_NFL:
        return pt == "run"
    return "rush" in pt or (pt == "run")


def red_zone_mask(df: pd.DataFrame, league: str) -> pd.Series:
    if league == LEAGUE_NFL:
        if "yardline_100" in df.columns:
            return pd.to_numeric(df["yardline_100"], errors="coerce") <= 20
        return pd.Series(False, index=df.index)
    if "yards_to_goal" in df.columns:
        return pd.to_numeric(df["yards_to_goal"], errors="coerce") <= 20
    return pd.Series(False, index=df.index)


def early_down_mask(df: pd.DataFrame) -> pd.Series:
    if "down" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["down"].isin([1, 2])


def filter_game_data(
    df: pd.DataFrame,
    league: str,
    game_id: str | None,
    week: int | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    g = df.copy()
    if week is not None and "week" in g.columns:
        g = g[g["week"] == week]
    if game_id is not None and "_norm_game_id" in g.columns:
        g = g[g["_norm_game_id"].astype(str) == str(game_id)]
    return g


def nfl_games_for_week(norm: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    sub = norm[(norm["season"] == season) & (norm["week"] == week)].copy()
    if sub.empty or "home_team" not in sub.columns:
        return pd.DataFrame()
    cols = ["game_id", "home_team", "away_team"]
    g = sub[cols].drop_duplicates()
    g["label"] = g["away_team"].astype(str) + " @ " + g["home_team"].astype(str)
    g["_norm_game_id"] = g["game_id"].astype(str)
    return g.sort_values("label")


@st.cache_data(ttl=86400, show_spinner=False)
def load_normalized_pbp(league: str, season: int, api_key: str = "") -> pd.DataFrame:
    raw = load_data(league, season, api_key)
    return normalize_data(raw, league)


def turnover_mask(df: pd.DataFrame, league: str) -> pd.Series:
    if league == LEAGUE_NFL:
        inter = pd.to_numeric(df.get("interception", 0), errors="coerce").fillna(0)
        fl = pd.to_numeric(df.get("fumble_lost", 0), errors="coerce").fillna(0)
        return (inter > 0) | (fl > 0)
    if "play_type" not in df.columns:
        return pd.Series(False, index=df.index)
    pt = df["play_type"].fillna("").str.lower()
    txt = df["play_text"].fillna("").str.lower() if "play_text" in df.columns else pd.Series([""] * len(df))
    return pt.str.contains("interception") | pt.str.contains("fumble") | txt.str.contains(
        "interception|fumble.*lost"
    )
