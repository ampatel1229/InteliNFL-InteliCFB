"""InteliNFL — Decision intelligence for NFL and College Football (Streamlit)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from intelinfl import LEAGUE_CFB, LEAGUE_NFL
from intelinfl.charts import (
    plot_down_pass_rates,
    plot_epa_or_ypp_comparison,
    plot_qb_hist_epa_nfl,
    plot_qb_scatter_cfb,
    plot_qb_scatter_nfl,
    plot_quarter_points_bar,
    plot_run_pass_bar,
    plot_team_profile_scatter,
    plot_win_prob_cfb,
    plot_win_prob_nfl,
    plot_wpa_cumulative_nfl,
)
from intelinfl.data_loader import (
    filter_game_data,
    load_cfb_games,
    load_cfb_player_passing_stats,
    load_cfb_wp_game,
    load_normalized_pbp,
    nfl_games_for_week,
    resolve_cfbd_api_key,
)
from intelinfl.insights import (
    generate_game_insights,
    generate_playcalling_insights,
    generate_qb_insights,
)
from intelinfl.metrics import (
    calculate_down_splits,
    calculate_team_game_metrics,
    cfb_qb_passing_table,
    nfl_qb_summary,
    quarter_score_breakdown,
    season_team_profile,
)
st.set_page_config(
    page_title="InteliNFL Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

YEAR = datetime.now().year
NFL_SEASONS = list(range(1999, YEAR + 1))
CFB_SEASONS = list(range(2004, YEAR + 1))

ACCENT = "#38bdf8"
STYLES = """
<style>
.block-container { padding-top: 1.25rem; max-width: 1400px; }
div[data-testid="stMetricValue"] { font-size: 1.35rem; }
.insight-card {
    background: linear-gradient(145deg, rgba(30,41,59,0.92), rgba(15,23,42,0.95));
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


def section(title: str, subtitle: str | None = None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def _pbp_home_away_for_game(
    game_df: pd.DataFrame, league: str, schedule_home: str, schedule_away: str
) -> tuple[str, str]:
    """Align labels with play-by-play so scores (CFB especially) match offense_score mapping."""
    if game_df is None or game_df.empty:
        return schedule_home, schedule_away
    if league == LEAGUE_NFL and "home_team" in game_df.columns:
        h = game_df["home_team"].dropna()
        a = game_df["away_team"].dropna()
        if len(h) and len(a):
            return str(h.iloc[0]), str(a.iloc[0])
    if league == LEAGUE_CFB and "home" in game_df.columns:
        h = game_df["home"].dropna()
        a = game_df["away"].dropna()
        if len(h) and len(a):
            return str(h.iloc[0]), str(a.iloc[0])
    return schedule_home, schedule_away


def cfb_games_for_week(games_df: pd.DataFrame, week: int) -> pd.DataFrame:
    if games_df is None or games_df.empty:
        return pd.DataFrame()
    g = games_df[games_df["week"] == week].copy()
    g = g[g["home_team"].notna() & g["away_team"].notna()]
    g["label"] = g["away_team"].astype(str) + " @ " + g["home_team"].astype(str)
    g["_norm_game_id"] = g["id"].astype(str)
    return g.sort_values("label")


def main():
    st.title("InteliNFL Decision Intelligence")
    st.caption("Real play-by-play metrics, adaptive to league — NFL (nflverse) & FBS (CollegeFootballData).")

    with st.sidebar:
        st.markdown("**College Football API**")
        key_hint = resolve_cfbd_api_key(None)
        api_override = st.text_input(
            "CFB key (optional)",
            type="password",
            help="Uses `CFBD_API_KEY` env or `.streamlit/secrets.toml` when left blank.",
        )
        api_key = api_override.strip() if api_override else (key_hint or "")

    g1, g2, g3 = st.columns([1, 1, 1])
    with g1:
        league = st.selectbox("League", [LEAGUE_NFL, LEAGUE_CFB], key="global_league")
    with g2:
        seasons = NFL_SEASONS if league == LEAGUE_NFL else CFB_SEASONS
        season = st.selectbox(
            "Season",
            seasons,
            index=min(len(seasons) - 2, len(seasons) - 1),
            key="global_season",
        )
    with g3:
        max_week = 22 if league == LEAGUE_NFL else 16
        week = st.slider("Week", 1, max_week, min(6, max_week), key="global_week")

    cfb_key_ok = league != LEAGUE_CFB or bool(resolve_cfbd_api_key(api_key))
    if league == LEAGUE_CFB and not cfb_key_ok:
        st.warning(
            "College Football requires a CollegeFootballData API key. "
            "Set `CFBD_API_KEY` in your environment or add `CFBD_API_KEY` to `.streamlit/secrets.toml`."
        )
        st.stop()

    try:
        with st.spinner("Loading season play-by-play…"):
            norm = load_normalized_pbp(league, season, api_key)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    if norm is None or norm.empty:
        st.error(
            "No data returned for this league/season. For College Football, pick a season with completed games, "
            "or use **Settings → Clear cache** if you recently fixed your API key."
        )
        st.stop()

    tab_game, tab_qb, tab_pc, tab_wp, tab_team = st.tabs(
        ["Game Analyzer", "QB Analyzer", "Play Calling", "Win Probability", "Team Profiles"]
    )

    # ----- Game lists -----
    if league == LEAGUE_NFL:
        games_meta = nfl_games_for_week(norm, season, week)
    else:
        with st.spinner("Loading CFB schedule for week…"):
            gdf = load_cfb_games(season, api_key)
        games_meta = cfb_games_for_week(gdf, week)

    if games_meta is None or games_meta.empty:
        st.error("No games found for this week (or schedule not available). Try another week.")
        st.stop()

    labels = games_meta["label"].tolist()

    with tab_game:
        section("Game Analyzer", "Pick a matchup — metrics adapt to NFL (EPA) vs College (yards / PPA).")
        choice = st.selectbox("Game (Away @ Home)", labels, key="game_pick_global")

        row = games_meta[games_meta["label"] == choice].iloc[0]
        game_id = str(row["_norm_game_id"])
        away_team = str(row["away_team"])
        home_team = str(row["home_team"])

        game_df = filter_game_data(norm, league, game_id, week=week)
        mh, ma = _pbp_home_away_for_game(game_df, league, home_team, away_team)
        hm = calculate_team_game_metrics(game_df, mh, league)
        am = calculate_team_game_metrics(game_df, ma, league)

        st.caption(f"{away_team} @ {home_team} — Week {week}, {season}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Away", away_team)
        with c2:
            st.metric("Home", home_team)
        with c3:
            dl = game_df.drop(columns=[c for c in game_df.columns if str(c).startswith("_")], errors="ignore")
            st.download_button(
                "Download filtered plays (CSV)",
                data=dl.to_csv(index=False).encode("utf-8"),
                file_name=f"intelinfl_{league}_{season}_w{week}_gid{game_id}.csv",
                mime="text/csv",
            )

        if game_df.empty:
            st.warning("No plays matched this game in the loaded slice. For CFB, confirm this game is FBS and regular season.")
        else:
            q_df = quarter_score_breakdown(game_df, mh, ma, league)
            if not q_df.empty:
                st.markdown("#### Score by period")
                st.caption(
                    "Bars: points scored each period. Table: running totals (Σ). "
                    "NFL from nflverse; CFB from play scoreboard fields."
                )
                st.plotly_chart(
                    plot_quarter_points_bar(q_df, mh, ma),
                    use_container_width=True,
                    key="game_analyzer_quarter_points_bar",
                    config={"displayModeBar": False},
                )
            else:
                st.warning(
                    "Could not build a period score breakdown (missing quarter/period or score columns on plays)."
                )

            col_a, col_b = st.columns(2)
            with col_a:
                if league == LEAGUE_NFL:
                    st.plotly_chart(
                        plot_win_prob_nfl(game_df, mh, ma),
                        use_container_width=True,
                        key="game_analyzer_wp_nfl",
                    )
                else:
                    gid = int(game_id) if str(game_id).isdigit() else None
                    wp_df = pd.DataFrame()
                    if gid is not None:
                        with st.spinner("Loading CFB win-probability series…"):
                            wp_df = load_cfb_wp_game(gid, api_key)
                    st.plotly_chart(
                        plot_win_prob_cfb(wp_df, mh, ma),
                        use_container_width=True,
                        key="game_analyzer_wp_cfb",
                    )
            with col_b:
                st.plotly_chart(
                    plot_epa_or_ypp_comparison(hm, am, mh, ma, league),
                    use_container_width=True,
                    key="game_analyzer_efficiency_bar",
                )

            st.plotly_chart(
                plot_run_pass_bar(hm, am, mh, ma),
                use_container_width=True,
                key="game_analyzer_run_pass_mix",
            )

            eff_cols = st.columns(2)
            with eff_cols[0]:
                if league == LEAGUE_NFL:
                    st.metric(f"EPA/play ({ma})", f"{am.get('epa_per_play', float('nan')):+.3f}")
                    st.metric("Success rate (away)", f"{am.get('success_rate', float('nan'))*100:.1f}%")
                else:
                    st.metric(f"Yards/play ({ma})", f"{am.get('yards_per_play', float('nan')):.2f}")
                    st.metric("Success rate (away)", f"{am.get('success_rate', float('nan'))*100:.1f}%")
            with eff_cols[1]:
                if league == LEAGUE_NFL:
                    st.metric(f"EPA/play ({mh})", f"{hm.get('epa_per_play', float('nan')):+.3f}")
                    st.metric("Success rate (home)", f"{hm.get('success_rate', float('nan'))*100:.1f}%")
                else:
                    st.metric(f"Yards/play ({mh})", f"{hm.get('yards_per_play', float('nan')):.2f}")
                    st.metric("Success rate (home)", f"{hm.get('success_rate', float('nan'))*100:.1f}%")

            st.markdown("#### Automated insights")
            for line in generate_game_insights(league, mh, ma, hm, am):
                st.markdown(f'<div class="insight-card">{line}</div>', unsafe_allow_html=True)

    with tab_qb:
        section("Quarterback analyzer")
        _teams = sorted(
            set(game_df["team"].dropna().unique()) if not game_df.empty else set(norm["team"].dropna().unique())
        )
        team_pick = st.selectbox("Team", _teams)
        if not team_pick:
            st.info("Select a valid game with plays.")
        elif league == LEAGUE_NFL:
            qb_df = nfl_qb_summary(norm, team_pick)
            if qb_df.empty:
                st.warning("No quarterback dropbacks found for this team in the season slice.")
            else:
                st.plotly_chart(plot_qb_scatter_nfl(qb_df), use_container_width=True, key="qb_analyzer_scatter_nfl")
                qb_name = st.selectbox("QB detail", qb_df["qb"].tolist())
                row = qb_df[qb_df["qb"] == qb_name].iloc[0].to_dict()
                for line in generate_qb_insights(league, row):
                    st.markdown(f'<div class="insight-card">{line}</div>', unsafe_allow_html=True)
                st.plotly_chart(
                    plot_qb_hist_epa_nfl(qb_df, qb_name, norm, team_pick),
                    use_container_width=True,
                    key="qb_analyzer_hist_nfl",
                )
        else:
            with st.spinner("Loading CFB passing stats…"):
                sdf = load_cfb_player_passing_stats(season, team_pick, api_key)
            qb_df = cfb_qb_passing_table(sdf, team_pick)
            if qb_df.empty:
                st.warning("No passing leaders returned for this team (API or eligibility filters).")
            else:
                st.plotly_chart(plot_qb_scatter_cfb(qb_df), use_container_width=True, key="qb_analyzer_scatter_cfb")
                qb_name = st.selectbox("QB detail", qb_df["qb"].tolist())
                row = qb_df[qb_df["qb"] == qb_name].iloc[0].to_dict()
                for line in generate_qb_insights(league, row):
                    st.markdown(f'<div class="insight-card">{line}</div>', unsafe_allow_html=True)

    with tab_pc:
        section("Play-calling tendencies", "Offensive play mix by situation (counted scrimmage snaps).")
        if game_df.empty:
            st.info("Load a game with plays to analyze tendencies.")
        else:
            pc_team = st.selectbox(
                "Focus team",
                [ma, mh],
                format_func=lambda t: f"{t} (visitor)" if t == ma else f"{t} (home)",
                key="pc_team",
            )
            m = calculate_team_game_metrics(game_df, pc_team, league)
            down_df = calculate_down_splits(game_df, pc_team, league)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    plot_down_pass_rates(down_df, pc_team),
                    use_container_width=True,
                    key="play_calling_down_pass_rates",
                )
            with c2:
                st.plotly_chart(
                    plot_run_pass_bar(hm, am, mh, ma),
                    use_container_width=True,
                    key="play_calling_run_pass_mix",
                )
            for line in generate_playcalling_insights(
                league, down_df, pc_team, float(m.get("pass_rate") or 0), m.get("pass_rate_early")
            ):
                st.markdown(f'<div class="insight-card">{line}</div>', unsafe_allow_html=True)

    with tab_wp:
        section("Win probability workspace")
        if league == LEAGUE_NFL:
            st.plotly_chart(
                plot_win_prob_nfl(game_df, mh, ma),
                use_container_width=True,
                key="win_prob_tab_wp_chart",
            )
            st.plotly_chart(
                plot_wpa_cumulative_nfl(game_df, mh, ma),
                use_container_width=True,
                key="win_prob_tab_wpa_cumulative",
            )
            st.caption(
                "WPA sums how much each play moved the needle toward a win for the offense on the field. "
                "This is descriptive of the selected game, not a Monte Carlo simulator."
            )
            shift = st.slider("Hypothetical EPA swing to home team (illustrative)", -7.0, 7.0, 0.0, 0.5)
            if shift != 0:
                approx = float(np.clip(0.5 + shift * 0.04, 0.02, 0.98))
                st.info(
                    f"Illustrative only: a crude mapping suggests a finishing win probability near {approx:.0%} "
                    "if that EPA bundle shifted purely to the home side — real models condition on score and time."
                )
        else:
            st.info(
                "CFB supplies win-probability charts via CollegeFootballData for completed games. "
                "Scenario simulation (counterfactual play sequences) is intentionally limited here."
            )
            gid = int(game_id) if str(game_id).isdigit() else None
            if gid:
                with st.spinner("Fetching CFB win probability…"):
                    wp_df = load_cfb_wp_game(gid, api_key)
                st.plotly_chart(
                    plot_win_prob_cfb(wp_df, mh, ma),
                    use_container_width=True,
                    key="win_prob_tab_wp_cfb",
                )

    with tab_team:
        section("Team profiles (season)", "Styles estimated from your loaded season — EPA/PPA for NFL, yards for CFB.")
        teams = sorted(norm["team"].dropna().unique())
        rows = []
        for t in teams:
            prof = season_team_profile(norm, t, league)
            if prof:
                prof["team"] = t
                rows.append(prof)
        st.plotly_chart(
            plot_team_profile_scatter(rows, league),
            use_container_width=True,
            key="team_profiles_style_scatter",
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
