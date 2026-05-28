"""Rule-based insight generation from computed metrics (no hardcoded game narratives)."""

from __future__ import annotations

import numpy as np

from intelinfl import LEAGUE_CFB, LEAGUE_NFL


def _fmt_f(x: float, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:.{d}f}"


def generate_game_insights(
    league: str,
    home_team: str,
    away_team: str,
    home_m: dict,
    away_m: dict,
) -> list[str]:
    insights: list[str] = []
    if not home_m.get("off_plays") or not away_m.get("off_plays"):
        return ["Insufficient offensive plays in this slice to compare efficiency."]

    def gap(metric: str, higher_is_better: bool = True) -> float | None:
        h, a = home_m.get(metric), away_m.get(metric)
        if h is None or a is None:
            return None
        if isinstance(h, float) and isinstance(a, float) and (np.isnan(h) or np.isnan(a)):
            return None
        return (h - a) if higher_is_better else (a - h)

    if league == LEAGUE_NFL:
        g = gap("epa_per_play")
        if g is not None and abs(g) >= 0.07:
            side = home_team if g > 0 else away_team
            insights.append(
                f"{side} generated a clear EPA edge (~{_fmt_f(abs(g))} EPA/play) in the trenches and execution."
            )
        g = gap("success_rate")
        if g is not None and abs(g) >= 0.08:
            side = home_team if g > 0 else away_team
            insights.append(f"{side} stayed ahead of schedule more often (success rate gap ~{_fmt_f(abs(g) * 100, 1)} pts).")
    else:
        g = gap("yards_per_play")
        if g is not None and abs(g) >= 0.55:
            side = home_team if g > 0 else away_team
            insights.append(
                f"{side} created more consistent displacement (~{_fmt_f(abs(g))} yards/play)."
            )
        g = gap("success_rate")
        if g is not None and abs(g) >= 0.08:
            side = home_team if g > 0 else away_team
            insights.append(f"{side} won more positive-gain snaps (success rate gap ~{_fmt_f(abs(g) * 100, 1)} pts).")

    for label, team, m in [("home", home_team, home_m), ("away", away_team, away_m)]:
        pr = m.get("pass_rate")
        if pr is not None and not (isinstance(pr, float) and np.isnan(pr)):
            if pr >= 0.65:
                insights.append(f"{team} leaned on the pass ({_fmt_f(pr * 100, 1)}% pass snaps on counted plays).")
            elif pr <= 0.38:
                insights.append(f"{team} stayed run-oriented ({_fmt_f((1 - pr) * 100, 1)}% run snaps on counted plays).")

    eh = home_m.get("early_epa") if league == LEAGUE_NFL else home_m.get("early_ypp")
    ea = away_m.get("early_epa") if league == LEAGUE_NFL else away_m.get("early_ypp")
    if (
        eh is not None
        and ea is not None
        and not (np.isnan(eh) or np.isnan(ea))
        and abs(eh - ea) >= (0.12 if league == LEAGUE_NFL else 1.1)
    ):
        side = home_team if eh > ea else away_team
        metric = "early-down EPA" if league == LEAGUE_NFL else "early-down yards/play"
        insights.append(
            f"{side} separated on early downs ({metric} advantage ~{_fmt_f(abs(eh - ea))})."
        )

    rz_h = home_m.get("rz_epa") if league == LEAGUE_NFL else home_m.get("rz_ypp")
    rz_a = away_m.get("rz_epa") if league == LEAGUE_NFL else away_m.get("rz_ypp")
    if (
        rz_h is not None
        and rz_a is not None
        and not (np.isnan(rz_h) or np.isnan(rz_a))
        and abs(rz_h - rz_a) >= (0.25 if league == LEAGUE_NFL else 1.5)
    ):
        side = home_team if rz_h > rz_a else away_team
        insights.append(f"{side} was sharper inside the 20, where games are often decided.")

    th, ta = home_m.get("turnovers", 0), away_m.get("turnovers", 0)
    if isinstance(th, (int, float)) and isinstance(ta, (int, float)) and abs(th - ta) >= 1:
        worse = home_team if th > ta else away_team
        insights.append(
            f"Turnover margin tilted the field — {worse} gave the ball away more in counted offensive plays."
        )

    ex_gap = (home_m.get("explosive_rate") or 0) - (away_m.get("explosive_rate") or 0)
    if abs(ex_gap) >= 0.07:
        side = home_team if ex_gap > 0 else away_team
        insights.append(
            f"{side} manufactured more explosives (approx. {_fmt_f(abs(ex_gap) * 100, 1)} pts higher explosive rate)."
        )

    if len(insights) < 3:
        insights.append(
            "The matchup stayed relatively even on macro efficiency — winner likely leveraged situational execution and finishing drives."
        )

    return insights[:8]


def generate_qb_insights(league: str, row: dict) -> list[str]:
    out: list[str] = []
    if league == LEAGUE_NFL:
        epa = row.get("epa_per_play")
        if epa is not None and epa >= 0.15:
            out.append("High-efficiency passing profile by EPA/play.")
        if epa is not None and epa <= -0.05:
            out.append("Bottom-line EPA suggests pressure, accuracy, or situational negatives stacked up.")
        cpoe = row.get("cpoe")
        if cpoe is not None and not np.isnan(cpoe):
            if cpoe >= 3:
                out.append("Completion percentage over expected (CPOE) looks strong — ball placement is beating coverage.")
            elif cpoe <= -3:
                out.append("CPOE is soft — a lot of throws are off-schedule versus expectation.")
        air = row.get("avg_air_yards")
        yac = row.get("avg_yac")
        if air is not None and yac is not None and not (np.isnan(air) or np.isnan(yac)):
            if air <= 6.5 and yac >= 4:
                out.append("Short-area profile with meaningful YAC — quick game and RAC responsibility.")
            if air >= 9 and (yac is None or yac < 3.5):
                out.append("Aggressive air yards with modest YAC — downfield intent stands out.")
    else:
        ypa = row.get("ypa")
        pct = row.get("cmp_pct")
        if ypa is not None and ypa >= 8.5:
            out.append("Vertical efficiency (yards/attempt) pops relative to typical college passing.")
        if ypa is not None and ypa <= 6.2:
            out.append("Yards/attempt are capped — check sacks, screens, or conservative constraints.")
        if pct is not None and pct >= 68:
            out.append("Completion environment is friendly (high CMP%) — timing and layups may dominate.")
        ints = row.get("int", 0)
        td = row.get("td", 0)
        if isinstance(ints, (int, float)) and isinstance(td, (int, float)) and ints >= 5 and td / max(ints, 1) < 2:
            out.append("Turnover touch rate on scoring throws bears monitoring (TD/INT tradeoff).")
    if not out:
        out.append("Profile reads balanced — use the scatter and distribution tabs for context.")
    return out[:6]


def generate_playcalling_insights(league: str, down_df, team: str, pass_rate: float, early_pr: float | None) -> list[str]:
    ins: list[str] = []
    if early_pr is not None and not np.isnan(early_pr):
        if early_pr <= 0.42:
            ins.append(f"{team} stayed run-heavier on early downs (~{_fmt_f(early_pr * 100, 1)}% pass).")
        elif early_pr >= 0.62:
            ins.append(f"{team} aired it out early (~{_fmt_f(early_pr * 100, 1)}% pass on 1st/2nd down).")
    if not down_df.empty and "down" in down_df.columns:
        third = down_df[down_df["down"] == 3]
        if not third.empty and "pass_rate" in third.columns:
            pr3 = float(third["pass_rate"].iloc[0])
            if pr3 >= 0.8:
                ins.append("Third down is heavily pass-dominant — tendency breakers could matter.")
            elif pr3 <= 0.45:
                ins.append("Third-down run usage is unusually high — constraint throws may be underused.")
    if pass_rate >= 0.58:
        ins.append("Overall pass rate sits above league-typical balance on counted snaps.")
    if not ins:
        ins.append("Play-calling profile looks neutral without sharp situational skews in this slice.")
    return ins[:5]
