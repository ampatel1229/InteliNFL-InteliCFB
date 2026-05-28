"""Plotly charts for InteliNFL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from intelinfl import LEAGUE_CFB, LEAGUE_NFL

ACCENT = "#38bdf8"
ACCENT2 = "#f472b6"
GRID = "rgba(148,163,184,0.25)"


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#cbd5e1", size=12),
        margin=dict(l=48, r=24, t=56, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )


def plot_win_prob_nfl(game_df: pd.DataFrame, home: str, away: str) -> go.Figure:
    sub = game_df.copy()
    if "play_id" in sub.columns:
        sub = sub.sort_values("play_id")
    if "wp" in sub.columns:
        sub = sub[sub["wp"].notna()]
    idx = np.arange(len(sub))
    wps = []
    for _, row in sub.iterrows():
        wp = row.get("wp")
        pt = row.get("posteam")
        ht = row.get("home_team")
        if pd.isna(wp) or pd.isna(pt):
            wps.append(np.nan)
            continue
        if pt == ht:
            wps.append(float(wp))
        else:
            wps.append(1.0 - float(wp))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=wps,
            mode="lines",
            name=f"{home} win prob",
            line=dict(color=ACCENT, width=2),
        )
    )
    fig.update_layout(**_base_layout(f"In-game win probability (home: {home})"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title_text="Play sequence")
    return fig


def plot_win_prob_cfb(wp_df: pd.DataFrame, home: str, away: str) -> go.Figure:
    if wp_df is None or wp_df.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Win probability (no data)"))
        return fig
    ycol = None
    for c in ("home_win_prob", "homeWinProbability", "home_win_probability"):
        if c in wp_df.columns:
            ycol = c
            break
    if ycol is None:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Win probability (no data)"))
        return fig
    sub = wp_df.sort_values("play_number" if "play_number" in wp_df.columns else wp_df.index)
    if "playNumber" in sub.columns and "play_number" not in sub.columns:
        sub = sub.rename(columns={"playNumber": "play_number"})
    x = sub["play_number"] if "play_number" in sub.columns else np.arange(len(sub))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=sub[ycol],
            mode="lines",
            name=f"{home} win prob",
            line=dict(color=ACCENT, width=2),
        )
    )
    fig.update_layout(**_base_layout(f"In-game win probability (home: {home}) — {away} @ {home}"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title_text="Play #")
    return fig


def plot_run_pass_bar(home_m: dict, away_m: dict, home: str, away: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=away,
            x=["Pass rate", "Early-down pass", "Red-zone pass"],
            y=[
                away_m.get("pass_rate", np.nan),
                away_m.get("pass_rate_early", np.nan),
                away_m.get("pass_rate_rz", np.nan),
            ],
            marker_color=ACCENT,
        )
    )
    fig.add_trace(
        go.Bar(
            name=home,
            x=["Pass rate", "Early-down pass", "Red-zone pass"],
            y=[
                home_m.get("pass_rate", np.nan),
                home_m.get("pass_rate_early", np.nan),
                home_m.get("pass_rate_rz", np.nan),
            ],
            marker_color=ACCENT2,
        )
    )
    fig.update_layout(
        **_base_layout("Pass-rate tendencies (offensive snaps)"),
        barmode="group",
    )
    fig.update_yaxes(tickformat=".0%", title_text="Share of snaps")
    return fig


def plot_down_pass_rates(down_df: pd.DataFrame, team: str) -> go.Figure:
    if down_df is None or down_df.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(f"Pass rate by down — {team}"))
        return fig
    fig = go.Figure(
        data=[
            go.Bar(
                x=down_df["down"].astype(str),
                y=down_df["pass_rate"],
                marker_color=ACCENT,
                name="Pass rate",
            )
        ]
    )
    fig.update_layout(**_base_layout(f"Pass rate by down — {team}"))
    fig.update_yaxes(tickformat=".0%", title_text="Pass rate")
    return fig


def plot_epa_or_ypp_comparison(home_m: dict, away_m: dict, home: str, away: str, league: str) -> go.Figure:
    key = "epa_per_play" if league == LEAGUE_NFL else "yards_per_play"
    label = "EPA/play" if league == LEAGUE_NFL else "Yards/play"
    fig = go.Figure(
        data=[
            go.Bar(
                x=[away, home],
                y=[away_m.get(key, np.nan), home_m.get(key, np.nan)],
                marker_color=[ACCENT, ACCENT2],
            )
        ]
    )
    fig.update_layout(**_base_layout(f"Core efficiency — {label}"))
    fig.update_yaxes(title_text=label)
    return fig


def plot_qb_scatter_nfl(qb_df: pd.DataFrame) -> go.Figure:
    if qb_df is None or qb_df.empty:
        return go.Figure().update_layout(**_base_layout("QB efficiency (no data)"))
    fig = go.Figure(
        data=[
            go.Scatter(
                x=qb_df["avg_air_yards"],
                y=qb_df["epa_per_play"],
                mode="markers+text",
                text=qb_df["qb"],
                textposition="top center",
                marker=dict(size=10, color=ACCENT, line=dict(width=1, color="#e2e8f0")),
                name="QBs",
            )
        ]
    )
    fig.update_layout(**_base_layout("QB profiles — air yards vs EPA/play"))
    fig.update_xaxes(title_text="Avg air yards")
    fig.update_yaxes(title_text="EPA / dropback")
    return fig


def plot_qb_hist_epa_nfl(qb_df: pd.DataFrame, qb_name: str, raw_game: pd.DataFrame, team: str) -> go.Figure:
    sub = raw_game[
        (raw_game["team"] == team)
        & (raw_game["play_type"] == "pass")
        & (raw_game["passer_player_name"] == qb_name)
    ]
    epa = pd.to_numeric(sub["epa"], errors="coerce").dropna()
    fig = go.Figure(data=[go.Histogram(x=epa, nbinsx=22, marker_color=ACCENT)])
    fig.update_layout(**_base_layout(f"EPA distribution — {qb_name}"))
    fig.update_xaxes(title_text="EPA per play")
    fig.update_yaxes(title_text="Count")
    return fig


def plot_qb_scatter_cfb(qb_df: pd.DataFrame) -> go.Figure:
    if qb_df is None or qb_df.empty:
        return go.Figure().update_layout(**_base_layout("QB passing (no data)"))
    fig = go.Figure(
        data=[
            go.Scatter(
                x=qb_df["cmp_pct"],
                y=qb_df["ypa"],
                mode="markers+text",
                text=qb_df["qb"],
                textposition="top center",
                marker=dict(size=11, color=ACCENT2, line=dict(width=1, color="#e2e8f0")),
            )
        ]
    )
    fig.update_layout(**_base_layout("QB profiles — completion % vs yards/attempt"))
    fig.update_xaxes(title_text="Completion %")
    fig.update_yaxes(title_text="Yards / attempt")
    return fig


def plot_team_profile_scatter(profile_rows: list[dict], league: str) -> go.Figure:
    if not profile_rows:
        return go.Figure().update_layout(**_base_layout("Team profiles"))
    df = pd.DataFrame(profile_rows)
    xcol = "pass_rate"
    ycol = "eff_epa" if league == LEAGUE_NFL else "ypp"
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df[xcol],
                y=df[ycol],
                mode="markers+text",
                text=df["team"],
                textposition="top center",
                marker=dict(size=10, color=ACCENT),
            )
        ]
    )
    ylab = "EPA/play (season)" if league == LEAGUE_NFL else "Yards/play (season)"
    fig.update_layout(**_base_layout("Team styles — pass rate vs efficiency"))
    fig.update_xaxes(title_text="Pass rate (offensive snaps)", tickformat=".0%")
    fig.update_yaxes(title_text=ylab)
    return fig


def plot_wpa_cumulative_nfl(game_df: pd.DataFrame, home: str, away: str) -> go.Figure:
    sub = game_df.sort_values("play_id") if "play_id" in game_df.columns else game_df
    if "wpa" not in sub.columns:
        fig = go.Figure()
        fig.update_layout(**_base_layout("WPA cumulative (minimal data)"))
        return fig
    cum_home = []
    cum_away = []
    ch, ca = 0.0, 0.0
    for _, row in sub.iterrows():
        wpa = float(row["wpa"]) if pd.notna(row.get("wpa")) else 0.0
        pt = row.get("posteam")
        if pt == home:
            ch += wpa
        elif pt == away:
            ca += wpa
        cum_home.append(ch)
        cum_away.append(ca)
    x = np.arange(len(sub))
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=x, y=cum_home, name=f"{home} cumulative WPA", line=dict(color=ACCENT)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=cum_away, name=f"{away} cumulative WPA", line=dict(color=ACCENT2)), secondary_y=False)
    fig.update_layout(**_base_layout("Win probability added (cumulative) — NFL"))
    fig.update_xaxes(title_text="Play sequence")
    return fig


def _team_abbr(label: str) -> str:
    s = (label or "?").strip()
    if not s:
        return "?"
    if len(s) <= 4:
        return s.upper()
    first, *rest = s.split()
    if len(first) >= 3:
        return first[:3].upper()
    if rest:
        pad = (first + rest[0])[:3]
        return pad.upper()
    return first.upper()[:3]


def plot_quarter_points_bar(q_df: pd.DataFrame, home: str, away: str) -> go.Figure:
    """Grouped bars + embedded table in one short figure (avoids separate dataframe scroll)."""
    if q_df is None or q_df.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=100,
            margin=dict(l=16, r=16, t=36, b=16),
            title=dict(text="Score by period (no data)", font=dict(size=13, color="#e2e8f0")),
            font=dict(color="#cbd5e1"),
        )
        return fig

    ha, hh = _team_abbr(away), _team_abbr(home)
    n = len(q_df)
    fig_h = min(420, 260 + 26 * max(n, 4))

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.04,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    fig.add_trace(
        go.Bar(
            name=away,
            x=q_df["Period"],
            y=q_df["away_pts"],
            marker_color=ACCENT,
            legendgroup="away",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name=home,
            x=q_df["Period"],
            y=q_df["home_pts"],
            marker_color=ACCENT2,
            legendgroup="home",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Table(
            columnwidth=[0.18, 0.2, 0.2, 0.2, 0.22],
            header=dict(
                values=[
                    "Prd",
                    f"{ha} +",
                    f"{hh} +",
                    f"{ha} Σ",
                    f"{hh} Σ",
                ],
                fill_color="rgba(30,41,59,0.98)",
                font=dict(color="#e2e8f0", size=11),
                align="center",
                height=26,
            ),
            cells=dict(
                values=[
                    q_df["Period"].astype(str).tolist(),
                    [int(x) for x in q_df["away_pts"]],
                    [int(x) for x in q_df["home_pts"]],
                    [int(x) for x in q_df["away_total"]],
                    [int(x) for x in q_df["home_total"]],
                ],
                fill_color="rgba(15,23,42,0.9)",
                font=dict(color="#cbd5e1", size=11),
                align="center",
                height=22,
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        barmode="group",
        height=fig_h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        margin=dict(l=36, r=20, t=40, b=8),
        font=dict(color="#cbd5e1", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        title=dict(
            text="Score by period",
            font=dict(size=14, color="#e2e8f0"),
            x=0,
            xanchor="left",
        ),
        bargap=0.18,
        bargroupgap=0.08,
    )
    fig.update_yaxes(
        title_text="Pts",
        dtick=1,
        row=1,
        col=1,
        gridcolor=GRID,
        zerolinecolor=GRID,
    )
    fig.update_xaxes(row=1, col=1, gridcolor=GRID, showgrid=False)
    return fig
