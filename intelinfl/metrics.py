"""Efficiency and tendency metrics on normalized play-by-play."""

from __future__ import annotations

import numpy as np
import pandas as pd

from intelinfl import LEAGUE_CFB, LEAGUE_NFL
from intelinfl.data_loader import (
    early_down_mask,
    is_pass_play,
    offensive_plays_mask,
    red_zone_mask,
    turnover_mask,
)


def _subset_team(df: pd.DataFrame, team: str) -> pd.DataFrame:
    return df[df["team"] == team].copy()


def success_mask(df: pd.DataFrame, league: str) -> pd.Series:
    yds = pd.to_numeric(df["yards"], errors="coerce")
    if league == LEAGUE_NFL:
        epa = pd.to_numeric(df["epa"], errors="coerce")
        return epa.fillna(-1e9) > 0
    return yds.fillna(-1e9) > 0


def calculate_team_game_metrics(full_game_df: pd.DataFrame, team: str, league: str) -> dict:
    df = _subset_team(full_game_df, team)
    m: dict[str, float | int] = {}
    off = df[offensive_plays_mask(df, league)]
    if off.empty:
        return {
            "off_plays": 0,
            "epa_per_play": float("nan"),
            "success_rate": float("nan"),
            "yards_per_play": float("nan"),
            "early_epa": float("nan"),
            "rz_epa": float("nan"),
            "pass_rate": float("nan"),
            "pass_rate_early": float("nan"),
            "pass_rate_rz": float("nan"),
            "turnovers": 0,
            "explosive_rate": float("nan"),
        }
    epa = pd.to_numeric(off["epa"], errors="coerce")
    yds = pd.to_numeric(off["yards"], errors="coerce")
    m["off_plays"] = int(len(off))
    m["yards_per_play"] = float(yds.mean()) if league == LEAGUE_CFB else float(yds.mean())
    m["epa_per_play"] = float(epa.mean()) if epa.notna().any() else float("nan")
    sm = success_mask(off, league)
    m["success_rate"] = float(sm.mean()) if len(off) else float("nan")
    early = off[early_down_mask(off)]
    rz = off[red_zone_mask(off, league)]
    m["early_epa"] = float(pd.to_numeric(early["epa"], errors="coerce").mean()) if not early.empty else float(
        "nan"
    )
    m["rz_epa"] = float(pd.to_numeric(rz["epa"], errors="coerce").mean()) if not rz.empty else float("nan")
    pass_rows = np.array([is_pass_play(off.iloc[i], league) for i in range(len(off))])
    m["pass_rate"] = float(pass_rows.mean())
    if not early.empty:
        pe = np.array([is_pass_play(early.iloc[i], league) for i in range(len(early))])
        m["pass_rate_early"] = float(pe.mean())
    else:
        m["pass_rate_early"] = float("nan")
    if not rz.empty:
        pr = np.array([is_pass_play(rz.iloc[i], league) for i in range(len(rz))])
        m["pass_rate_rz"] = float(pr.mean())
    else:
        m["pass_rate_rz"] = float("nan")
    to = turnover_mask(off, league)
    m["turnovers"] = int(to.sum())
    if league == LEAGUE_NFL:
        explosive = (pd.to_numeric(off["epa"], errors="coerce").fillna(-1e9) >= 0.5) | (
            pd.to_numeric(off["yards"], errors="coerce").fillna(0) >= 15
        )
    else:
        explosive = pd.to_numeric(off["yards"], errors="coerce").fillna(0) >= 15
    m["explosive_rate"] = float(explosive.mean()) if len(off) else float("nan")
    m["early_ypp"] = float(pd.to_numeric(early["yards"], errors="coerce").mean()) if not early.empty else float(
        "nan"
    )
    m["rz_ypp"] = float(pd.to_numeric(rz["yards"], errors="coerce").mean()) if not rz.empty else float("nan")
    return m


def calculate_down_splits(df: pd.DataFrame, team: str, league: str) -> pd.DataFrame:
    sub = _subset_team(df, team)
    off = sub[offensive_plays_mask(sub, league)]
    if off.empty or "down" not in off.columns:
        return pd.DataFrame()
    rows = []
    for d in sorted(off["down"].dropna().unique()):
        chunk = off[off["down"] == d]
        passes = sum(is_pass_play(chunk.iloc[i], league) for i in range(len(chunk)))
        rows.append(
            {
                "down": int(d),
                "plays": len(chunk),
                "pass_rate": passes / len(chunk) if len(chunk) else np.nan,
                "epa_per_play": float(pd.to_numeric(chunk["epa"], errors="coerce").mean()),
                "ypp": float(pd.to_numeric(chunk["yards"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def season_team_profile(norm: pd.DataFrame, team: str, league: str) -> dict:
    df = _subset_team(norm, team)
    off = df[offensive_plays_mask(df, league)]
    out: dict[str, str | float] = {}
    if off.empty:
        return out
    pass_rate = float(np.mean([is_pass_play(off.iloc[i], league) for i in range(len(off))]))
    out["pass_rate"] = pass_rate
    out["run_heavy"] = "Pass-heavy" if pass_rate > 0.55 else ("Balanced" if pass_rate > 0.45 else "Run-heavy")
    if league == LEAGUE_NFL:
        epa = pd.to_numeric(off["epa"], errors="coerce")
        ypp = pd.to_numeric(off["yards"], errors="coerce")
        eff = float(epa.mean())
        expl = float((ypp.fillna(0) >= 15).mean())
        out["eff_epa"] = eff
        out["explosive_rate"] = expl
        out["efficient_vs_explosive"] = (
            "Efficient (EPA-driven)" if eff > 0.02 and expl < 0.12 else "Explosive (big-play)"
        )
        if eff > 0.04:
            out["tempo_style"] = "Aggressive (positive EPA volume)"
        elif eff < -0.01:
            out["tempo_style"] = "Conservative / struggling efficiency"
        else:
            out["tempo_style"] = "Neutral efficiency profile"
    else:
        ypp = float(pd.to_numeric(off["yards"], errors="coerce").mean())
        expl = float((pd.to_numeric(off["yards"], errors="coerce").fillna(0) >= 15).mean())
        out["ypp"] = ypp
        out["explosive_rate"] = expl
        out["efficient_vs_explosive"] = (
            "Efficient (steady gains)" if ypp >= 6.0 and expl < 0.14 else "Explosive (chunk plays)"
        )
        epa = pd.to_numeric(off["epa"], errors="coerce")
        if epa.notna().mean() > 0.5 and float(epa.mean()) > 0.03:
            out["tempo_style"] = "Aggressive (PPA-friendly)"
        elif ypp < 5.0:
            out["tempo_style"] = "Conservative / grinding"
        else:
            out["tempo_style"] = "Neutral yardage profile"
    return out


def nfl_qb_summary(norm: pd.DataFrame, team: str, min_plays: int = 15) -> pd.DataFrame:
    sub = norm[(norm["team"] == team) & (norm["play_type"] == "pass")].copy()
    if "passer_player_name" not in sub.columns or sub.empty:
        return pd.DataFrame()
    sub = sub[sub["passer_player_name"].notna() & (sub["passer_player_name"] != "")]
    rows = []
    for qb, g in sub.groupby("passer_player_name"):
        if len(g) < min_plays:
            continue
        air = pd.to_numeric(g.get("air_yards"), errors="coerce")
        yac = pd.to_numeric(g.get("yards_after_catch"), errors="coerce")
        epa = pd.to_numeric(g["epa"], errors="coerce")
        cpoe = pd.to_numeric(g.get("cpoe"), errors="coerce")
        rows.append(
            {
                "qb": qb,
                "plays": len(g),
                "epa_per_play": float(epa.mean()),
                "cpoe": float(cpoe.mean()) if cpoe.notna().any() else float("nan"),
                "avg_air_yards": float(air.mean()) if air.notna().any() else float("nan"),
                "avg_yac": float(yac.mean()) if yac.notna().any() else float("nan"),
                "cmp_pct": float(pd.to_numeric(g.get("complete_pass"), errors="coerce").mean())
                * 100
                if "complete_pass" in g.columns
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("epa_per_play", ascending=False)


def cfb_qb_passing_table(stats_df: pd.DataFrame, team: str) -> pd.DataFrame:
    if stats_df is None or stats_df.empty:
        return pd.DataFrame()
    s = stats_df[stats_df["team"] == team].copy()
    if s.empty:
        return pd.DataFrame()
    pivot: dict[tuple, dict[str, float]] = {}
    for _, row in s.iterrows():
        pid = row.get("player_id", row.get("playerId"))
        name = row.get("player")
        stype = str(row.get("stat_type") or row.get("statType") or "").upper()
        stat = float(row.get("stat") or 0)
        key = (pid, name)
        pivot.setdefault(key, {})[stype] = stat
    rows = []
    for (pid, name), d in pivot.items():
        att = d.get("ATT", 0) or 0
        if att < 20:
            continue
        cmp_ = d.get("CMP") or d.get("COMP") or d.get("COMPLETIONS") or 0
        yds = d.get("YDS", 0) or d.get("YARDS", 0)
        td = d.get("TD", 0)
        ints = d.get("INT", 0)
        ypa = yds / att if att else float("nan")
        pct = (cmp_ / att * 100) if att else float("nan")
        rows.append(
            {
                "player_id": pid,
                "qb": name,
                "attempts": att,
                "yards": yds,
                "td": td,
                "int": ints,
                "ypa": ypa,
                "cmp_pct": pct,
            }
        )
    return pd.DataFrame(rows).sort_values("ypa", ascending=False)


def _period_label_nfl(qtr: int) -> str:
    if qtr <= 4:
        return f"Q{qtr}"
    ot_n = qtr - 4
    return "OT" if ot_n == 1 else f"{ot_n}OT"


def _nfl_quarter_rows(game_df: pd.DataFrame, home_team: str, away_team: str) -> list[dict]:
    g = game_df.copy()
    if "total_home_score" not in g.columns or "total_away_score" not in g.columns:
        if "home_score" in g.columns and "away_score" in g.columns:
            g["total_home_score"] = pd.to_numeric(g["home_score"], errors="coerce")
            g["total_away_score"] = pd.to_numeric(g["away_score"], errors="coerce")
        else:
            return []
    if "qtr" not in g.columns:
        return []
    g = g[g["qtr"].notna()].copy()
    g["qtr"] = pd.to_numeric(g["qtr"], errors="coerce")
    g = g.dropna(subset=["qtr"])
    sort_col = "play_id" if "play_id" in g.columns else None
    if sort_col:
        g = g.sort_values(sort_col)
    else:
        g = g.sort_index()
    g["total_home_score"] = pd.to_numeric(g["total_home_score"], errors="coerce")
    g["total_away_score"] = pd.to_numeric(g["total_away_score"], errors="coerce")
    rows: list[dict] = []
    prev_h, prev_a = 0, 0
    for q in sorted(g["qtr"].astype(int).unique()):
        sub = g[g["qtr"].astype(int) == int(q)]
        if sub.empty:
            continue
        last = sub.iloc[-1]
        if pd.isna(last["total_home_score"]) or pd.isna(last["total_away_score"]):
            continue
        th = int(last["total_home_score"])
        ta = int(last["total_away_score"])
        dh, da = th - prev_h, ta - prev_a
        label = _period_label_nfl(int(q))
        rows.append(
            {
                "Period": label,
                "away_pts": da,
                "home_pts": dh,
                "away_total": ta,
                "home_total": th,
            }
        )
        prev_h, prev_a = th, ta
    return rows


def _cfb_quarter_rows(game_df: pd.DataFrame, home_team: str, away_team: str) -> list[dict]:
    if "period" not in game_df.columns:
        return []
    off = game_df["offense"] if "offense" in game_df.columns else game_df.get("team")
    if off is None:
        return []
    os_ = pd.to_numeric(
        game_df["offense_score"]
        if "offense_score" in game_df.columns
        else game_df.get("offenseScore"),
        errors="coerce",
    )
    ds_ = pd.to_numeric(
        game_df["defense_score"]
        if "defense_score" in game_df.columns
        else game_df.get("defenseScore"),
        errors="coerce",
    )
    g = game_df.copy()
    g["_off"] = off
    g["_os"] = os_
    g["_ds"] = ds_
    g = g.dropna(subset=["_os", "_ds", "_off"])
    is_home_off = g["_off"].astype(str) == str(home_team)
    g["_home_tot"] = np.where(is_home_off, g["_os"], g["_ds"])
    g["_away_tot"] = np.where(is_home_off, g["_ds"], g["_os"])
    g["period"] = pd.to_numeric(g["period"], errors="coerce")
    g = g.dropna(subset=["period"])
    sort_cols = [c for c in ("play_number", "playNumber", "play_id", "id") if c in g.columns]
    if sort_cols:
        g = g.sort_values(sort_cols)
    rows: list[dict] = []
    prev_h, prev_a = 0, 0
    for p in sorted(g["period"].astype(int).unique()):
        sub = g[g["period"].astype(int) == int(p)]
        if sub.empty:
            continue
        last = sub.iloc[-1]
        th = int(last["_home_tot"])
        ta = int(last["_away_tot"])
        dh, da = th - prev_h, ta - prev_a
        label = _period_label_nfl(int(p))
        rows.append(
            {
                "Period": label,
                "away_pts": da,
                "home_pts": dh,
                "away_total": ta,
                "home_total": th,
            }
        )
        prev_h, prev_a = th, ta
    return rows


def quarter_score_breakdown(
    game_df: pd.DataFrame, home_team: str, away_team: str, league: str
) -> pd.DataFrame:
    """End-of-period cumulative scores and points scored in each period (NFL quarters / CFB periods)."""
    if game_df is None or game_df.empty:
        return pd.DataFrame()
    if league == LEAGUE_NFL:
        rows = _nfl_quarter_rows(game_df, home_team, away_team)
    elif league == LEAGUE_CFB:
        rows = _cfb_quarter_rows(game_df, home_team, away_team)
    else:
        rows = []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

