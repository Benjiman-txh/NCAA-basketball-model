"""
Backtest vanilla vs silver ELO prediction accuracy.

Trains on 1985-1994 (burn-in), then predicts rolling from 1995 onward.
For each game from 1995+, predicts winner BEFORE updating ELO, then updates.
"""

import math
import numpy as np
import polars as pl
from scipy import integrate, stats


# ── Probability functions ──

def vanilla_prob(team_a_elo, team_b_elo):
    delta = team_a_elo - team_b_elo
    return 1 / (1 + 10 ** (-delta / 400))


def silver_prob(team_a_elo, team_b_elo,
                hca_a=0, dist_a=0, dist_b=0,
                neutral=False, tournament=False):
    home_bonus = hca_a if not neutral else 0
    travel_a = 8 * dist_a ** (1 / 3) if neutral else 0
    travel_b = 8 * dist_b ** (1 / 3)

    delta = team_a_elo + home_bonus - team_b_elo + travel_b - travel_a

    if tournament:
        delta = delta * 1.07

    return 1 / (1 + 10 ** (-delta / 400))


# ── Silver ELO update function ──

def silver_elo_update(team_a_elo, team_b_elo, actual_margin, K,
                      hca_a=0, dist_a=0, dist_b=0,
                      neutral=False, tournament=False):
    if tournament:
        K = K * 1.25

    home_bonus = hca_a if not neutral else 0
    travel_a = 8 * dist_a ** (1 / 3) if neutral else 0
    travel_b = 8 * dist_b ** (1 / 3)

    delta_elo = team_a_elo + home_bonus - team_b_elo + travel_b - travel_a

    if tournament:
        delta_elo = delta_elo * 1.07

    win_prob = 1 / (1 + 10 ** (-delta_elo / 400))
    mu = delta_elo / 27
    sigma = 11

    def integrand(s):
        return (3 + abs(s)) ** 0.85 * stats.norm.pdf(s, mu, sigma)

    N, _ = integrate.quad(integrand, -100, 100)

    S = 1 if actual_margin > 0 else 0
    change = K * (3 + abs(actual_margin)) ** 0.85 / N * (S - win_prob)
    return change


# ── Kalman filter for per-team HCA ──

HCA_INIT = 3.5
HCA_INIT_VAR = 10.0
HCA_Q_GAME = 0.01
HCA_Q_SEASON = 4.0
HCA_R = 121.0


class KalmanHCA:
    def __init__(self):
        self.mean: dict[int, float] = {}
        self.var: dict[int, float] = {}

    def _init(self, tid: int):
        if tid not in self.mean:
            self.mean[tid] = HCA_INIT
            self.var[tid] = HCA_INIT_VAR

    def get_elo(self, tid: int) -> float:
        self._init(tid)
        return self.mean[tid] * 27

    def update(self, home_id: int, actual_margin: float, expected_margin_no_hca: float):
        self._init(home_id)
        pred_mean = self.mean[home_id]
        pred_var = self.var[home_id] + HCA_Q_GAME
        innovation = actual_margin - expected_margin_no_hca - pred_mean
        K = pred_var / (pred_var + HCA_R)
        self.mean[home_id] = pred_mean + K * innovation
        self.var[home_id] = (1 - K) * pred_var

    def season_reset(self):
        for tid in self.mean:
            self.var[tid] += HCA_Q_SEASON


def main():
    # ── Load data ──
    regular = pl.read_csv("MRegularSeasonCompactResults.csv").with_columns(
        pl.lit("regular_season").alias("which")
    )
    tourney = pl.read_csv("MNCAATourneyCompactResults.csv").with_columns(
        pl.lit("march_madness").alias("which")
    )
    teams = pl.read_csv("MTeams.csv")
    all_games = pl.concat([regular, tourney], how="vertical").sort("Season")

    # Conference tourney set
    conf_game = pl.read_csv("MConferenceTourneyGames.csv")
    conf_tourney_set: set[tuple] = set()
    for r in conf_game.iter_rows(named=True):
        conf_tourney_set.add((r["Season"], r["DayNum"], r["WTeamID"], r["LTeamID"]))

    # Team conferences per season
    team_conf_df = pl.read_csv("MTeamConferences.csv")
    team_conference: dict[tuple[int, int], str] = {}
    for r in team_conf_df.iter_rows(named=True):
        team_conference[(r["Season"], r["TeamID"])] = r["ConfAbbrev"]

    # Distance lookup (2010+)
    game_distances = pl.read_csv("game_distances.csv")
    dist_lookup: dict[tuple, tuple[float, float]] = {}
    for r in game_distances.iter_rows(named=True):
        key = (r["Season"], r["DayNum"], r["WTeamID"], r["LTeamID"])
        dist_lookup[key] = (r["WTeamDist"] or 0.0, r["LTeamDist"] or 0.0)

    team_ids = teams["TeamID"].to_list()

    # ── Initialize both models ──
    v_elos = {tid: 1300.0 for tid in team_ids}   # vanilla
    s_elos = {tid: 1500.0 for tid in team_ids}    # silver

    s_game_counts: dict[tuple[int, int], int] = {}
    hca = KalmanHCA()

    def get_k(season: int, team_id: int) -> float:
        n = s_game_counts.get((season, team_id), 0)
        if n < 20:
            return 56 - (56 - 38) * (n / 20)
        return 38.0

    def apply_mean_reversion(elos: dict, season: int):
        conf_teams: dict[str, list[int]] = {}
        for tid in elos:
            conf = team_conference.get((season, tid))
            if conf:
                conf_teams.setdefault(conf, []).append(tid)
        conf_means = {conf: np.mean([elos[t] for t in tids]) for conf, tids in conf_teams.items()}
        for tid in elos:
            conf = team_conference.get((season, tid))
            if conf and conf in conf_means:
                elos[tid] = 0.67 * elos[tid] + 0.33 * conf_means[conf]
            else:
                elos[tid] = 0.67 * elos[tid] + 0.33 * 1500

    PRED_START = 1995
    prev_season = None
    total = len(all_games)

    # Tracking
    v_correct = 0
    v_total = 0
    v_log_loss_sum = 0.0
    s_correct = 0
    s_total = 0
    s_log_loss_sum = 0.0

    # Per-season tracking
    season_results: dict[int, dict] = {}

    # March madness only tracking
    v_mm_correct = 0
    v_mm_total = 0
    s_mm_correct = 0
    s_mm_total = 0

    # Post-2010 tracking
    v_post_correct = 0
    v_post_total = 0
    s_post_correct = 0
    s_post_total = 0
    v_post_mm_correct = 0
    v_post_mm_total = 0
    s_post_mm_correct = 0
    s_post_mm_total = 0

    for i, row in enumerate(all_games.iter_rows()):
        season = row[0]
        day_num = row[1]
        win_id = row[2]
        win_score = row[3]
        lose_id = row[4]
        lose_score = row[5]
        w_loc = row[6]
        which = row[8]

        # ── Season boundary ──
        if season != prev_season:
            if prev_season is not None:
                apply_mean_reversion(s_elos, prev_season)
                hca.season_reset()
            s_game_counts = {k: v for k, v in s_game_counts.items() if k[0] == season}
            prev_season = season

        # ── Game type ──
        is_mm = which == "march_madness"
        is_conf = (season, day_num, win_id, lose_id) in conf_tourney_set
        neutral = is_mm or is_conf or w_loc == "N"
        tournament = is_mm
        margin = win_score - lose_score

        game_key = (season, day_num, win_id, lose_id)
        w_dist, l_dist = dist_lookup.get(game_key, (0.0, 0.0))

        # ══════════════════════════════════════════
        # PREDICT (before updating) — from 1995+
        # ══════════════════════════════════════════
        if season >= PRED_START:
            if season not in season_results:
                season_results[season] = {"v_correct": 0, "v_total": 0,
                                          "s_correct": 0, "s_total": 0}

            # Vanilla prediction: P(winner wins)
            v_p = vanilla_prob(v_elos[win_id], v_elos[lose_id])
            if v_p > 0.5:
                v_correct += 1
                season_results[season]["v_correct"] += 1
            v_total += 1
            season_results[season]["v_total"] += 1
            v_log_loss_sum -= math.log(max(v_p, 1e-15))

            # Silver prediction: P(winner wins), accounting for home/away + HCA
            if w_loc == "A":
                home_hca = hca.get_elo(lose_id)
                s_p_home = silver_prob(
                    s_elos[lose_id], s_elos[win_id],
                    hca_a=home_hca, dist_a=l_dist, dist_b=w_dist,
                    neutral=neutral, tournament=tournament,
                )
                s_p = 1 - s_p_home
            elif w_loc == "H":
                home_hca = hca.get_elo(win_id)
                s_p = silver_prob(
                    s_elos[win_id], s_elos[lose_id],
                    hca_a=home_hca, dist_a=w_dist, dist_b=l_dist,
                    neutral=neutral, tournament=tournament,
                )
            else:
                s_p = silver_prob(
                    s_elos[win_id], s_elos[lose_id],
                    hca_a=0, dist_a=w_dist, dist_b=l_dist,
                    neutral=neutral, tournament=tournament,
                )

            if s_p > 0.5:
                s_correct += 1
                season_results[season]["s_correct"] += 1
            s_total += 1
            season_results[season]["s_total"] += 1
            s_log_loss_sum -= math.log(max(s_p, 1e-15))

            # March madness only
            if is_mm:
                if v_p > 0.5:
                    v_mm_correct += 1
                v_mm_total += 1
                if s_p > 0.5:
                    s_mm_correct += 1
                s_mm_total += 1

            # Post-2010 tracking
            if season >= 2010:
                if v_p > 0.5:
                    v_post_correct += 1
                v_post_total += 1
                if s_p > 0.5:
                    s_post_correct += 1
                s_post_total += 1
                if is_mm:
                    if v_p > 0.5:
                        v_post_mm_correct += 1
                    v_post_mm_total += 1
                    if s_p > 0.5:
                        s_post_mm_correct += 1
                    s_post_mm_total += 1

        # ══════════════════════════════════════════
        # UPDATE ELOs (both models)
        # ══════════════════════════════════════════

        # Vanilla update
        v_win_prob = 1 / (1 + 10 ** ((v_elos[lose_id] - v_elos[win_id]) / 400))
        v_change = 100 * (1 - v_win_prob)
        v_elos[win_id] += v_change
        v_elos[lose_id] -= v_change

        # Silver update + Kalman HCA update
        k_w = get_k(season, win_id)
        k_l = get_k(season, lose_id)
        K = k_w if k_w == k_l else (k_w + k_l) / 2

        if w_loc == "A":
            home_hca = hca.get_elo(lose_id)
            change = silver_elo_update(
                s_elos[lose_id], s_elos[win_id], -margin, K,
                hca_a=home_hca, dist_a=l_dist, dist_b=w_dist,
                neutral=neutral, tournament=tournament,
            )
            s_elos[lose_id] += change
            s_elos[win_id] -= change
            expected_no_hca = (s_elos[lose_id] - s_elos[win_id]) / 27
            hca.update(lose_id, -margin, expected_no_hca)

        elif w_loc == "H":
            home_hca = hca.get_elo(win_id)
            change = silver_elo_update(
                s_elos[win_id], s_elos[lose_id], margin, K,
                hca_a=home_hca, dist_a=w_dist, dist_b=l_dist,
                neutral=neutral, tournament=tournament,
            )
            s_elos[win_id] += change
            s_elos[lose_id] -= change
            expected_no_hca = (s_elos[win_id] - s_elos[lose_id]) / 27
            hca.update(win_id, margin, expected_no_hca)

        else:
            change = silver_elo_update(
                s_elos[win_id], s_elos[lose_id], margin, K,
                hca_a=0, dist_a=w_dist, dist_b=l_dist,
                neutral=neutral, tournament=tournament,
            )
            s_elos[win_id] += change
            s_elos[lose_id] -= change

        s_game_counts[(season, win_id)] = s_game_counts.get((season, win_id), 0) + 1
        s_game_counts[(season, lose_id)] = s_game_counts.get((season, lose_id), 0) + 1

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{total} games...")

    # ── Results ──
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS (predictions from {PRED_START} onward)")
    print(f"{'='*60}")

    print(f"\n── All Games (1995+) ──")
    print(f"  Vanilla:  {v_correct}/{v_total} = {v_correct/v_total:.4f}  "
          f"(log loss: {v_log_loss_sum/v_total:.4f})")
    print(f"  Silver:   {s_correct}/{s_total} = {s_correct/s_total:.4f}  "
          f"(log loss: {s_log_loss_sum/s_total:.4f})")

    print(f"\n── All Games (2010+) ──")
    print(f"  Vanilla:  {v_post_correct}/{v_post_total} = {v_post_correct/v_post_total:.4f}")
    print(f"  Silver:   {s_post_correct}/{s_post_total} = {s_post_correct/s_post_total:.4f}")

    print(f"\n── March Madness (all) ──")
    if v_mm_total > 0:
        print(f"  Vanilla:  {v_mm_correct}/{v_mm_total} = {v_mm_correct/v_mm_total:.4f}")
        print(f"  Silver:   {s_mm_correct}/{s_mm_total} = {s_mm_correct/s_mm_total:.4f}")

    print(f"\n── March Madness (2010+) ──")
    if v_post_mm_total > 0:
        print(f"  Vanilla:  {v_post_mm_correct}/{v_post_mm_total} = {v_post_mm_correct/v_post_mm_total:.4f}")
        print(f"  Silver:   {s_post_mm_correct}/{s_post_mm_total} = {s_post_mm_correct/s_post_mm_total:.4f}")

    print(f"\n── Per-Season Accuracy ──")
    print(f"  {'Season':<8} {'Vanilla':>10} {'Silver':>10}")
    print(f"  {'------':<8} {'-------':>10} {'------':>10}")
    for s in sorted(season_results.keys()):
        sr = season_results[s]
        v_acc = sr["v_correct"] / sr["v_total"] if sr["v_total"] > 0 else 0
        s_acc = sr["s_correct"] / sr["s_total"] if sr["s_total"] > 0 else 0
        print(f"  {s:<8} {v_acc:>10.4f} {s_acc:>10.4f}")


if __name__ == "__main__":
    main()
