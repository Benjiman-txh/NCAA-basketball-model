"""
Compute silver ELO (with Kalman HCA) for every team, snapshotted right before
March Madness starts each season.  Saves to pre_tourney_silver_elo.csv.
"""

import numpy as np
import polars as pl
from scipy import integrate, stats
from collections import defaultdict

# ── silver_elo function (from notebook cell 6) ──

def silver_elo(team_a_elo, team_b_elo, actual_margin, K,
               hca_a=0, dist_a=0, dist_b=0,
               neutral=False, tournament=False):
    if tournament:
        K = K * 1.25
    home_bonus = hca_a if not neutral else 0
    travel_a = 8 * dist_a**(1/3) if neutral else 0
    travel_b = 8 * dist_b**(1/3)
    delta_elo = team_a_elo + home_bonus - team_b_elo + travel_b - travel_a
    if tournament:
        delta_elo = delta_elo * 1.07
    win_prob = 1 / (1 + 10**(-delta_elo / 400))
    mu = delta_elo / 27
    sigma = 11
    def integrand(s):
        return (3 + abs(s))**0.85 * stats.norm.pdf(s, mu, sigma)
    N, _ = integrate.quad(integrand, -100, 100)
    S = 1 if actual_margin > 0 else 0
    change_in_elo = K * (3 + abs(actual_margin)) ** 0.85 / N * (S - win_prob)
    return change_in_elo

# ── Load data ──

regular_season_df = pl.read_csv("MRegularSeasonCompactResults.csv")
regular_season_df = regular_season_df.with_columns(pl.lit("regular_season").alias("which"))
march_madness = pl.read_csv("MNCAATourneyCompactResults.csv")
march_madness = march_madness.with_columns(pl.lit("march_madness").alias("which"))
team_mapping = pl.read_csv("MTeams.csv")
all_records_df = pl.concat((regular_season_df, march_madness), how="vertical").sort("Season", descending=False)

# Distance lookup
game_distances = pl.read_csv("game_distances.csv")
dist_lookup: dict[tuple, tuple[float, float]] = {}
for r in game_distances.iter_rows(named=True):
    key = (r["Season"], r["DayNum"], r["WTeamID"], r["LTeamID"])
    dist_lookup[key] = (r["WTeamDist"] or 0.0, r["LTeamDist"] or 0.0)

# Conference tourney set
conf_game = pl.read_csv("MConferenceTourneyGames.csv")
conf_tourney_set: set[tuple] = set()
for r in conf_game.iter_rows(named=True):
    conf_tourney_set.add((r["Season"], r["DayNum"], r["WTeamID"], r["LTeamID"]))

# Team conference mapping
team_conf_df = pl.read_csv("MTeamConferences.csv")
team_conference: dict[tuple[int, int], str] = {}
for r in team_conf_df.iter_rows(named=True):
    team_conference[(r["Season"], r["TeamID"])] = r["ConfAbbrev"]

# ── Kalman HCA ──

HCA_INIT = 3.5
HCA_INIT_VAR = 10.0
HCA_Q_GAME = 0.01
HCA_Q_SEASON = 4.0
HCA_R = 121.0

hca_mean: dict[int, float] = {}
hca_var: dict[int, float] = {}

def init_hca(team_id: int):
    if team_id not in hca_mean:
        hca_mean[team_id] = HCA_INIT
        hca_var[team_id] = HCA_INIT_VAR

def update_hca(home_team_id: int, actual_margin: float, expected_margin_no_hca: float):
    init_hca(home_team_id)
    pred_mean = hca_mean[home_team_id]
    pred_var = hca_var[home_team_id] + HCA_Q_GAME
    innovation = actual_margin - expected_margin_no_hca - pred_mean
    K = pred_var / (pred_var + HCA_R)
    hca_mean[home_team_id] = pred_mean + K * innovation
    hca_var[home_team_id] = (1 - K) * pred_var

def get_hca_elo(team_id: int) -> float:
    init_hca(team_id)
    return hca_mean[team_id] * 27

def season_reset_hca():
    for tid in hca_mean:
        hca_var[tid] = hca_var[tid] + HCA_Q_SEASON

# ── ELO init ──

elos: dict = {}
for id in team_mapping["TeamID"].to_list():
    elos[id] = 1500.0

game_counts: dict[tuple[int, int], int] = {}

def get_k(season: int, team_id: int) -> float:
    n = game_counts.get((season, team_id), 0)
    if n < 20:
        return 56 - (56 - 38) * (n / 20)
    return 38.0

def apply_mean_reversion(elos: dict, season: int):
    conf_teams: dict[str, list[int]] = {}
    for tid in elos:
        conf = team_conference.get((season, tid))
        if conf:
            conf_teams.setdefault(conf, []).append(tid)
    conf_means: dict[str, float] = {}
    for conf, tids in conf_teams.items():
        conf_means[conf] = np.mean([elos[t] for t in tids])
    for tid in elos:
        conf = team_conference.get((season, tid))
        if conf and conf in conf_means:
            elos[tid] = 0.67 * elos[tid] + 0.33 * conf_means[conf]
        else:
            elos[tid] = 0.67 * elos[tid] + 0.33 * 1500

# ── Main loop ──

prev_season = None
total_games = len(all_records_df)
pre_tourney_snapshots = []  # list of (season, team_id, elo, hca_margin)
mm_snapshot_taken: set[int] = set()

for i, row in enumerate(all_records_df.iter_rows()):
    season = row[0]
    day_num = row[1]
    win_team_id = row[2]
    win_score = row[3]
    lose_team_id = row[4]
    lose_score = row[5]
    w_loc = row[6]
    which = row[8]

    # Mean reversion + reset game counts + HCA uncertainty bump on new season
    if season != prev_season:
        if prev_season is not None:
            apply_mean_reversion(elos, prev_season)
            season_reset_hca()
        game_counts = {k: v for k, v in game_counts.items() if k[0] == season}
        prev_season = season

    is_march_madness = which == "march_madness"

    # Snapshot ELOs right before first MM game of the season
    if is_march_madness and season not in mm_snapshot_taken:
        mm_snapshot_taken.add(season)
        for tid, elo_val in elos.items():
            init_hca(tid)
            pre_tourney_snapshots.append((season, tid, elo_val, hca_mean[tid]))
        print(f"  Snapshotted {len(elos)} teams for season {season}")

    # --- rest of ELO update logic ---
    win_team_elo = elos[win_team_id]
    lose_team_elo = elos[lose_team_id]
    margin = win_score - lose_score

    is_conf_tourney = (season, day_num, win_team_id, lose_team_id) in conf_tourney_set
    neutral = is_march_madness or is_conf_tourney or w_loc == "N"
    tournament = is_march_madness

    k_w = get_k(season, win_team_id)
    k_l = get_k(season, lose_team_id)
    K = k_w if k_w == k_l else (k_w + k_l) / 2

    game_key = (season, day_num, win_team_id, lose_team_id)
    w_dist, l_dist = dist_lookup.get(game_key, (0.0, 0.0))

    if w_loc == "A":
        home_id = lose_team_id
        hca_elo = get_hca_elo(home_id)
        change = silver_elo(lose_team_elo, win_team_elo, -margin, K,
                            hca_a=hca_elo, dist_a=l_dist, dist_b=w_dist,
                            neutral=neutral, tournament=tournament)
        elos[lose_team_id] += change
        elos[win_team_id] -= change
        expected_margin_no_hca = (lose_team_elo - win_team_elo) / 27
        update_hca(home_id, -margin, expected_margin_no_hca)
    elif w_loc == "H":
        home_id = win_team_id
        hca_elo = get_hca_elo(home_id)
        change = silver_elo(win_team_elo, lose_team_elo, margin, K,
                            hca_a=hca_elo, dist_a=w_dist, dist_b=l_dist,
                            neutral=neutral, tournament=tournament)
        elos[win_team_id] += change
        elos[lose_team_id] -= change
        expected_margin_no_hca = (win_team_elo - lose_team_elo) / 27
        update_hca(home_id, margin, expected_margin_no_hca)
    else:
        change = silver_elo(win_team_elo, lose_team_elo, margin, K,
                            hca_a=0, dist_a=w_dist, dist_b=l_dist,
                            neutral=neutral, tournament=tournament)
        elos[win_team_id] += change
        elos[lose_team_id] -= change

    game_counts[(season, win_team_id)] = game_counts.get((season, win_team_id), 0) + 1
    game_counts[(season, lose_team_id)] = game_counts.get((season, lose_team_id), 0) + 1

    if (i + 1) % 10000 == 0:
        print(f"  Processed {i + 1}/{total_games} games...")

# ── Save ──

df = pl.DataFrame({
    "Season": [s[0] for s in pre_tourney_snapshots],
    "TeamID": [s[1] for s in pre_tourney_snapshots],
    "SilverELO": [s[2] for s in pre_tourney_snapshots],
    "HCA_margin": [s[3] for s in pre_tourney_snapshots],
})
df.write_csv("pre_tourney_silver_elo.csv")
print(f"\nSaved {len(df)} rows to pre_tourney_silver_elo.csv")
print(f"Seasons: {sorted(df['Season'].unique().to_list())}")
print(df.head(10))
