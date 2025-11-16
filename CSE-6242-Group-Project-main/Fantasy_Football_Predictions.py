"""
Fantasy Football Predictions System
===================================
This script combines NFL play-by-play data processing with machine learning 
models to predict fantasy football points per week for all positions.

The system:
1. Loads and processes NFL play-by-play data (1999-2024)
2. Calculates ESPN-standard PPR fantasy points
3. Engineers position-specific features
4. Trains Mixture of Experts (MoE) models for each position
5. Generates predictions and model comparison reports
"""

# ==============================================================================
# Package Installation (Uncomment lines to install missing packages)
# ==============================================================================
# If you encounter import errors, uncomment and run the corresponding pip install line below:
#
# pip install nflreadpy
# pip install pandas
# pip install numpy
# pip install scikit-learn
# pip install xgboost

# ==============================================================================
# Imports
# ==============================================================================

import numpy as np
import pandas as pd
import nflreadpy as nfl
import xgboost as xgb
from functools import lru_cache
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    XGBOOST_AVAILABLE = True
    import xgboost as xgb
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available. XGBoost model will be skipped.")


# ==============================================================================
# PART 1: DATA PREPARATION AND SCORING
# ==============================================================================

ALL_SEASONS = list(range(1999, 2025))

#returns max fantasy week
def fantasy_week_max(season: int) -> int:
    return 16 if season <= 2020 else 17

#normalizing team names
def normalize_team(team: str, season: int) -> str:
    t = (team or "").upper()
    if season <= 2015 and t in {"LAR", "LA", "STL"}:
        return "STL"
    if season >= 2016 and t in {"LAR", "LA", "STL"}:
        return "LAR"
    if season <= 2019 and t in {"OAK", "LV"}:
        return "OAK"
    if season >= 2020 and t in {"OAK", "LV"}:
        return "LV"
    if season <= 2016 and t in {"SD", "LAC"}:
        return "SD"
    if season >= 2017 and t in {"SD", "LAC"}:
        return "LAC"
    return t

# ----------------------------------
# Utility Functions
# ----------------------------------

def base_filter(df: pd.DataFrame, week_min: int, week_max: int) -> pd.DataFrame:
    out = df.query("season_type == 'REG' and @week_min <= week <= @week_max").copy()
    if "play_deleted" in out.columns:
        out = out[out["play_deleted"].fillna(0) != 1]
    if "play_type" in out.columns:
        out = out[out["play_type"] != "no_play"]
    return out

def safe_col(df, name, default=0):
    return df[name].fillna(0) if name in df.columns else 0

def _to_py_int_list(values) -> list[int]:
    s = pd.Series(values)
    return (s.dropna().astype(int).astype(object).apply(int).drop_duplicates().tolist())

@lru_cache(maxsize=None)
def load_positions_for(seasons_key: tuple[int, ...]) -> pd.DataFrame:
    seasons = list(seasons_key)
    rosters_pl = nfl.load_rosters(seasons)
    rosters = rosters_pl.to_pandas()

    # Find the player ID column - different versions use different names
    id_col = next((c for c in ["gsis_id", "player_id", "gsis_player_id", "nfl_id", "gsis"] if c in rosters.columns), None)
    if id_col is None:
        raise ValueError("No GSIS id column found. Expected one of: gsis_id/player_id/gsis_player_id/nfl_id/gsis")

    # Find the position column
    pos_col = next((c for c in ["position", "position_group"] if c in rosters.columns), None)
    if pos_col is None:
        raise KeyError("No position column found. Expected 'position' or 'position_group'")

    pos_tbl = rosters[["season", id_col, pos_col]].copy().dropna(subset=[id_col])
    pos_tbl["position"] = pos_tbl[pos_col].astype(str).str.upper().str.strip().replace({"NONE": np.nan})
    # Collapse to one row per season+player (in case they have multiple positions)
    pos_tbl = (pos_tbl.sort_values(["season", id_col, "position"])
               .groupby(["season", id_col], as_index=False)
               .agg(position=("position", "first"))
               .rename(columns={id_col: "fantasy_player_id"}))
    return pos_tbl

def _ensure_cols(df: pd.DataFrame, must_int: list = None, must_float: list = None, 
                 must_str: list = None, alias_map: dict = None) -> pd.DataFrame:
    must_int = must_int or []
    must_float = must_float or []
    must_str = must_str or []
    alias_map = alias_map or {}

    # Apply aliases first (pass -> pass_attempt)
    for old_col, new_col in alias_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # Ensure integer columns exist and are integers
    for c in must_int:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype("int64")

    # Ensure float columns exist and are floats
    for c in must_float:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0).astype("float64")

    # Ensure string columns exist and are lowercase
    for c in must_str:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()
        else:
            df[c] = ""

    return df

#Calculates ESPN PPR fantasy points
def player_ppr(pbp_reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pbp_reg.copy()

    pass_attempt  = (df.get("pass_attempt", df.get("pass", 0)).fillna(0).astype(int) == 1)
    rush_attempt  = (df.get("rush_attempt", df.get("rush", 0)).fillna(0).astype(int) == 1)
    complete_pass = (df.get("complete_pass", 0).fillna(0).astype(int) == 1)

    # Passing yards give 0.04 points per yard, TDs give 4, INTs lose 2
    pass_pts = (safe_col(df, "passing_yards")*0.04 + safe_col(df, "pass_touchdown")*4 - safe_col(df, "interception")*2)
    passing = pd.DataFrame({
        "season": df["season"], "week": df["week"],
        "player_id": df.get("passer_player_id"), "player_name": df.get("passer_player_name"),
        "fp": np.where(pass_attempt, pass_pts, 0.0)
    })

    # Rushing yards give 0.1 points per yard, TDs give 6
    rush_pts = (safe_col(df, "rushing_yards")*0.1 + safe_col(df, "rush_touchdown")*6)
    rushing = pd.DataFrame({
        "season": df["season"], "week": df["week"],
        "player_id": df.get("rusher_player_id"), "player_name": df.get("rusher_player_name"),
        "fp": np.where(rush_attempt, rush_pts, 0.0)
    })

    # Receptions give 1 point (PPR bonus), yards give 0.1 per yard, TDs give 6
    recv_pts = (1.0 + safe_col(df, "receiving_yards")*0.1 + safe_col(df, "pass_touchdown")*6)
    receiving = pd.DataFrame({
        "season": df["season"], "week": df["week"],
        "player_id": df.get("receiver_player_id"), "player_name": df.get("receiver_player_name"),
        "fp": np.where(complete_pass, recv_pts, 0.0)
    })

    # Two-point conversions - these give 2 points to whoever scores
    two_pt_success = (safe_col(df, "two_point_attempt") == 1) & (df.get("two_point_conv_result", "") == "success")
    two_pt_recv = pd.DataFrame({
        "season": df["season"], "week": df["week"],
        "player_id": df.get("receiver_player_id"), "player_name": df.get("receiver_player_name"),
        "fp": np.where(two_pt_success & (safe_col(df, "pass") == 1), 2.0, 0.0)
    })
    two_pt_rush = pd.DataFrame({
        "season": df["season"], "week": df["week"],
        "player_id": df.get("rusher_player_id"), "player_name": df.get("rusher_player_name"),
        "fp": np.where(two_pt_success & (safe_col(df, "rush") == 1), 2.0, 0.0)
    })

    # Fumbles lost - these penalize the player who fumbled
    fum_lost = pd.DataFrame({
        "season": df["season"], "week": df["week"],
        "player_id": df.get("fumbled_1_player_id"), "player_name": None, 
        "fp": np.where(safe_col(df, "fumble_lost") == 1, -2.0, 0.0)
    })

    # Combine all the points
    all_rows = pd.concat([passing, rushing, receiving, two_pt_recv, two_pt_rush, fum_lost], ignore_index=True)
    all_rows = all_rows[all_rows["player_id"].notna()].copy()

    # Aggregate to weekly totals
    weekly = (all_rows.groupby(["season","player_id","player_name","week"], as_index=False)["fp"].sum()
                        .sort_values(["season","player_name","week"]))
    
    # Aggregate to season totals
    leaders = (weekly.groupby(["season","player_id","player_name"], as_index=False)["fp"].sum()
                      .sort_values(["season","fp"], ascending=[True, False]))

    # Calculate games played - only count weeks where they scored points
    # This is more accurate than just counting weeks they appeared in, because
    # a player might be on the field but not get any fantasy-relevant touches
    games = (weekly.loc[weekly["fp"] != 0]
                   .groupby(["season","player_id","player_name"], as_index=False)["week"].nunique()
                   .rename(columns={"week":"games"}))
    leaders = leaders.merge(games, on=["season","player_id","player_name"], how="left")
    leaders["ppg"] = leaders["fp"] / leaders["games"]

    leaders = leaders.rename(columns={"player_id":"fantasy_player_id","player_name":"fantasy_player_name"})
    weekly  = weekly.rename(columns={"player_id":"fantasy_player_id","player_name":"fantasy_player_name"})
    return leaders, weekly

# ----------------------------------
# Defense/Special Teams Scoring
# ----------------------------------

def dst_scoring(pbp_reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates D/ST fantasy points (sacks, INTs, fumbles, return TDs, points allowed)."""
    df = pbp_reg.copy()

    # First, figure out the final scores for each game so we can calculate points allowed
    scores = (df.groupby(["season","game_id"], as_index=False)[["total_home_score","total_away_score","home_team","away_team"]]
                .agg({"total_home_score":"max","total_away_score":"max","home_team":"first","away_team":"first"})
                .rename(columns={"total_home_score":"home_pts_final","total_away_score":"away_pts_final"}))

    # Create rows for both home and away teams with points allowed
    home_rows = scores[["season","game_id","home_team","away_pts_final"]].rename(
        columns={"home_team":"team","away_pts_final":"points_allowed"})
    away_rows = scores[["season","game_id","away_team","home_pts_final"]].rename(
        columns={"away_team":"team","home_pts_final":"points_allowed"})
    pa_tbl = pd.concat([home_rows, away_rows], ignore_index=True)

    # Points-allowed tiers - this is how ESPN scores D/ST
    # Lower points allowed = more fantasy points (up to 10 for a shutout)
    def pa_to_pts(pa: int) -> int:
        if pa == 0: return 10
        if pa <= 6: return 7
        if pa <= 13: return 4
        if pa <= 17: return 1
        if pa <= 27: return 0
        if pa <= 34: return -1
        if pa <= 45: return -4
        return -5

    pa_tbl["pa_pts"] = pa_tbl["points_allowed"].astype(int).map(pa_to_pts)

    # Count defensive stats
    by_def = (df.groupby(["season","game_id","defteam"], as_index=False)
                .agg({"sack":"sum","interception":"sum","safety":"sum","punt_blocked":"sum"})
                .rename(columns={"defteam":"team"}))

    # Blocked field goals
    tmp = df.copy()
    tmp["fg_block"] = (tmp.get("field_goal_result","").astype(str).str.lower() == "blocked").astype(int)
    by_def_fgblk = (tmp.groupby(["season","game_id","defteam"], as_index=False)["fg_block"].sum()
                      .rename(columns={"defteam":"team"}))

    # Fumble recoveries - can happen multiple times per play
    fr_parts = []
    if "fumble_recovery_1_team" in df.columns:
        fr1 = df[df["fumble_recovery_1_team"].notna()][["season","game_id","fumble_recovery_1_team"]].copy()
        fr1["team"] = fr1["fumble_recovery_1_team"]; fr1["fr"] = 1
        fr_parts.append(fr1[["season","game_id","team","fr"]])
    if "fumble_recovery_2_team" in df.columns:
        fr2 = df[df["fumble_recovery_2_team"].notna()][["season","game_id","fumble_recovery_2_team"]].copy()
        fr2["team"] = fr2["fumble_recovery_2_team"]; fr2["fr"] = 1
        fr_parts.append(fr2[["season","game_id","team","fr"]])
    by_fr = (pd.concat(fr_parts, ignore_index=True)
             .groupby(["season","game_id","team"], as_index=False)["fr"].sum()) if fr_parts else \
            pd.DataFrame(columns=["season","game_id","team","fr"])

    # Return touchdowns (kick returns, punt returns, etc.)
    ret = df[(safe_col(df,"return_touchdown") == 1)]
    by_ret_td = (ret.groupby(["season","game_id","return_team"], as_index=False).size()
                   .rename(columns={"return_team":"team","size":"ret_td"}))

    # Merge all the D/ST components together
    dst = pa_tbl[["season","game_id","team","pa_pts"]].copy()
    for part in [
        by_def[["season","game_id","team","sack","interception","safety","punt_blocked"]],
        by_def_fgblk[["season","game_id","team","fg_block"]],
        by_fr[["season","game_id","team","fr"]],
        by_ret_td[["season","game_id","team","ret_td"]],
    ]:
        dst = dst.merge(part, on=["season","game_id","team"], how="left")

    # Fill missing values with 0
    dst = dst.fillna(0)
    for c in ["sack","interception","safety","punt_blocked","fg_block","fr","ret_td"]:
        if c in dst.columns: dst[c] = dst[c].astype(int)

    # Normalize team codes for historical consistency
    dst["team"] = [normalize_team(t, s) for t, s in zip(dst["team"].astype(str), dst["season"].astype(int))]

    # Calculate total D/ST fantasy points per game
    # 1 point per sack, 2 per INT/FR/safety, 2 per blocked kick, 6 per return TD, plus points-allowed points
    dst["dst_fp"] = (dst["sack"]*1 + dst["interception"]*2 + dst["fr"]*2 + dst["safety"]*2
        + dst["ret_td"]*6 + (dst["punt_blocked"] + dst["fg_block"])*2 + dst["pa_pts"])

    per_game = dst.sort_values(["season","game_id","team"]).reset_index(drop=True)
    season = (per_game.groupby(["season","team"], as_index=False)["dst_fp"].sum()
                      .sort_values(["season","dst_fp"], ascending=[True, False]))
    return season, per_game

# ----------------------------------
# Kicker Scoring
# ----------------------------------

def kicker_scoring(pbp_reg: pd.DataFrame, miss_pat_minus_one=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates kicker fantasy points (distance-based FGs: 3-6 pts, -1 for misses, 1 for PATs)."""
    df = pbp_reg.copy()
    # Only keep rows with field goal or PAT attempts
    kick = df[(safe_col(df, "field_goal_attempt") == 1) | (safe_col(df, "extra_point_attempt") == 1)].copy()

    # Standardize result strings to lowercase
    for c in ["field_goal_result","extra_point_result"]:
        if c in kick.columns: 
            kick[c] = kick[c].astype(str).str.lower()

    # Field goal points based on distance
    fg_made = (safe_col(kick, "field_goal_attempt") == 1) & (kick["field_goal_result"] == "made")
    dist = safe_col(kick, "kick_distance").astype(float)
    fg_pts = np.where(fg_made & (dist >= 60), 6,
              np.where(fg_made & (dist >= 50), 5,
              np.where(fg_made & (dist >= 40), 4,
              np.where(fg_made, 3, 0))))

    # Missed field goals (including blocked) cost 1 point
    fg_miss = (safe_col(kick, "field_goal_attempt") == 1) & (kick["field_goal_result"].isin(["missed","blocked"]))
    fg_miss_pts = np.where(fg_miss, -1, 0)

    # Extra points (PATs)
    pat_attempt = (safe_col(kick, "extra_point_attempt") == 1)
    pat_good = pat_attempt & (kick["extra_point_result"].isin(["good","made"]))
    pat_pts = np.where(pat_good, 1, 0)
    pat_missed = pat_attempt & (kick["extra_point_result"].isin(["failed","missed","blocked"]))
    pat_miss_pts = np.where(miss_pat_minus_one & pat_missed, -1, 0)

    kick["k_fp"] = fg_pts + fg_miss_pts + pat_pts + pat_miss_pts

    # Kickers might use different column names in different data versions
    id_col = "kicker_player_id" if "kicker_player_id" in kick.columns else "fantasy_player_id"
    name_col = "kicker_player_name" if "kicker_player_name" in kick.columns else "fantasy_player_name"

    # Aggregate per game per kicker
    per_game = (kick.groupby(["season","game_id", id_col, name_col, "posteam"], as_index=False)
                  .agg(k_fp=("k_fp","sum"),
                       fg_made=("field_goal_attempt", lambda s: int(((s==1) & (kick.loc[s.index,'field_goal_result'].eq('made'))).sum())),
                       fg_att=("field_goal_attempt","sum"),
                       pat_made=("extra_point_attempt", lambda s: int(((s==1) & (kick.loc[s.index,'extra_point_result'].isin(['good','made']))).sum())),
                       pat_att=("extra_point_attempt","sum"))
               )

    # Normalize team codes historically
    per_game["posteam"] = [normalize_team(t, s) for t, s in zip(per_game["posteam"].astype(str), per_game["season"].astype(int))]

    # Aggregate to season totals
    season = (per_game.groupby(["season", id_col, name_col, "posteam"], as_index=False)
                      .agg(total_fp=("k_fp","sum"),
                           games=("game_id","nunique"),
                           fg_made=("fg_made","sum"),
                           fg_att=("fg_att","sum"),
                           pat_made=("pat_made","sum"),
                           pat_att=("pat_att","sum")))
    season["ppg"] = season["total_fp"] / season["games"]
    season = season.sort_values(["season","total_fp","ppg"], ascending=[True, False, False])
    return season, per_game

# ----------------------------------
# Main Data Processing Loop
# ----------------------------------
# Process all seasons: loop through years, calculate fantasy points, build datasets.

print("="*80)
print("PHASE 1: DATA PREPARATION AND SCORING")
print("="*80)
print("Processing NFL play-by-play data for seasons 1999-2024...")
print("This will calculate fantasy points and create position-specific features.\n")

players_all, players_weekly_all = [], []
dst_all, dst_games_all = [], []
k_all, k_games_all = [], []

for yr in ALL_SEASONS:
    print(f"Processing {yr} ...", end=" ")
    # Load play-by-play data via nflreadpy
    pbp_pl = nfl.load_pbp([yr])
    pbp_y  = pbp_pl.to_pandas()

    # Filter to regular season weeks we care about
    wk_max = fantasy_week_max(yr)
    pbp_reg = base_filter(pbp_y, week_min=1, week_max=wk_max)

    # Downcast floats to save memory
    for c in pbp_reg.select_dtypes("float64").columns:
        pbp_reg[c] = pbp_reg[c].astype("float32")
    print(f"Done. (Memory optimized)")

    # Calculate fantasy points for players (QB, RB, WR, TE)
    p_season, p_weekly = player_ppr(pbp_reg)
    players_all.append(p_season)
    players_weekly_all.append(p_weekly)

    # Calculate D/ST fantasy points
    d_season, d_games = dst_scoring(pbp_reg)
    dst_all.append(d_season)
    dst_games_all.append(d_games)

    # Calculate kicker fantasy points
    k_season, k_games = kicker_scoring(pbp_reg, miss_pat_minus_one=False)
    k_all.append(k_season)
    k_games_all.append(k_games)

# Combine all seasons into multi-season dataframes
print("\nCombining multi-season data...")
players_leaders_multi = pd.concat(players_all, ignore_index=True)
players_weekly_multi  = pd.concat(players_weekly_all, ignore_index=True)
dst_season_multi      = pd.concat(dst_all, ignore_index=True)
dst_per_game_multi    = pd.concat(dst_games_all, ignore_index=True)
k_season_multi        = pd.concat(k_all, ignore_index=True)
k_per_game_multi      = pd.concat(k_games_all, ignore_index=True)

# Deduplicate player-season combinations (some players might appear twice)
players_leaders_multi = (
    players_leaders_multi
      .sort_values(['season','fantasy_player_id','fp'], ascending=[True, True, False])
      .groupby(['season','fantasy_player_id'], as_index=False)
      .agg(fp=('fp','first'), games=('games','max'), fantasy_player_name=('fantasy_player_name','first'))
)
assert not players_leaders_multi.duplicated(['season','fantasy_player_id']).any(), "Duplicate player-season keys remain."

print(f"✓ Processed {len(ALL_SEASONS)} seasons of data")
print(f"✓ Found {len(players_leaders_multi)} unique player-season combinations\n")

# ----------------------------------
# Generic Helper Functions for Position Feature Engineering
# ----------------------------------

def _load_season_pbp(season: int, ensure_cols_func) -> pd.DataFrame:
    """Loads and prepares play-by-play data for one season (helper for position features)."""
    pbp_pl = nfl.load_pbp([season])
    pbp = pbp_pl.to_pandas()
    wk_max = fantasy_week_max(season)
    pbp = base_filter(pbp, week_min=1, week_max=wk_max)
    pbp = ensure_cols_func(pbp)
    return pbp

def build_position_features(seasons, engineer_func, position_name: str, filter_func=None):
    """Builds position features across multiple seasons, optionally filters low-volume players."""
    frames = []
    for yr in seasons:
        print(f"  Engineering {position_name} features for {yr} ...")
        frames.append(engineer_func(yr))
    out = pd.concat(frames, ignore_index=True)
    if filter_func:
        out = filter_func(out)
    return out

def merge_position_with_targets(feats: pd.DataFrame, players_leaders_multi: pd.DataFrame, 
                                  position: str) -> pd.DataFrame:
    """Merges features with fantasy points, filters by position and min 4 games."""
    needed = {"season","fantasy_player_id","fp","games"}
    missing = needed - set(players_leaders_multi.columns)
    if missing:
        raise ValueError(f"players_leaders_multi missing columns: {missing}")
    assert not players_leaders_multi.duplicated(["season","fantasy_player_id"]).any(), \
        "players_leaders_multi must be unique per season+player (dedupe earlier)."

    # Load position data from rosters
    positions_tbl = load_positions_for(tuple(sorted(_to_py_int_list(feats["season"].unique()))))
    feats = feats.merge(positions_tbl, on=["season","fantasy_player_id"], how="left", validate="many_to_one")
    pre_n = len(feats)
    
    # Filter to only players who actually played this position
    feats = feats[feats["position"].astype(str).str.upper().eq(position)].copy()
    post_n = len(feats)
    print(f"  {position} position filter: kept {post_n}/{pre_n} rows where position == '{position}'")

    # Merge with fantasy point totals (this gives us our target variable)
    result = feats.drop(columns=["games"], errors="ignore").merge(
        players_leaders_multi[["season","fantasy_player_id","fp","games"]].rename(columns={"games":"games_ppr"}),
        on=["season","fantasy_player_id"], how="left", validate="many_to_one")
    result["fp"] = result["fp"].fillna(0.0)
    result = result.rename(columns={"games_ppr":"games"})
    
    # Only keep players with at least 4 games (reduces noise from small samples)
    result = result[result["games"] >= 4].copy()
    return result

# ----------------------------------
# Position-Specific Feature Engineering
# ----------------------------------
# Now we build position-specific features for each position. These features
# capture statistics that are relevant for predicting fantasy performance at
# each position. For example, QBs get features like completion percentage,
# yards per attempt, TD rate, etc. WRs get features like targets, receptions,
# yards per target, target share, etc.

print("\n" + "="*80)
print("PHASE 2: FEATURE ENGINEERING")
print("="*80)
print("Engineering position-specific features for all positions...\n")

# ================================
# QB Feature Engineering
# ================================
# Quarterback features include passing stats (completion %, YPA, TD rate, INT rate,
# air yards, YAC, EPA), rushing stats (yards, TDs, scrambles), and rate stats
# that normalize for volume (rates, shares).

def _ensure_qb_cols(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_cols(df,
        must_int=["complete_pass","pass_touchdown","interception","sack","rush_touchdown","qb_scramble",
                  "pass_attempt","rush_attempt"],
        must_float=["passing_yards","air_yards","yards_after_catch","epa","rushing_yards"],
        alias_map={"pass":"pass_attempt","rush":"rush_attempt"})

def engineer_qb_features_for_season(season: int) -> pd.DataFrame:
    pbp = _load_season_pbp(season, _ensure_qb_cols)

    # Identify pass and rush attempts
    pass_attempt_flag = (pbp.get('pass_attempt', pbp.get('pass', 0)).fillna(0).astype(int) == 1)
    rush_attempt_flag = (pbp.get('rush_attempt', pbp.get('rush', 0)).fillna(0).astype(int) == 1)

    # Attribute plays to the correct player (passer or rusher)
    pbp['player_id'] = np.where(pass_attempt_flag, pbp.get('passer_player_id'),
                        np.where(rush_attempt_flag, pbp.get('rusher_player_id'), np.nan))
    pbp['player_name'] = np.where(pass_attempt_flag, pbp.get('passer_player_name'),
                          np.where(rush_attempt_flag, pbp.get('rusher_player_name'), np.nan))
    pbp = pbp[pbp['player_id'].notna()].copy()

    # Helper columns for aggregation
    pbp['__pass_'] = pass_attempt_flag.astype(int)
    pbp['__rush_'] = rush_attempt_flag.astype(int)

    # Aggregate stats per QB
    g = (pbp.groupby(['season','player_id','player_name'])
        .agg(games=('week', 'nunique'), plays=('play_id', 'count'),
            pass_plays=('__pass_', 'sum'), rush_plays=('__rush_', 'sum'),
            comp=('complete_pass', 'sum'), pass_yards=('passing_yards', 'sum'),
            pass_td=('pass_touchdown', 'sum'), interceptions=('interception', 'sum'),
            sacks=('sack', 'sum'), air_yards=('air_yards', 'sum'),
            yac=('yards_after_catch', 'sum'), epa_sum=('epa', 'sum'),
            rush_yards=('rushing_yards', 'sum'), rush_td=('rush_touchdown', 'sum'),
            scrambles=('qb_scramble', 'sum'))
        .reset_index())

    # Get the team the QB played for most often
    team_mode = (pbp.groupby(['season','player_id'])['posteam']
        .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) > 0 else np.nan)
        .rename('team').reset_index())
    g = g.merge(team_mode, on=['season','player_id'], how='left')
    if 'team' in g.columns:
        g['team'] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                     for t, s in zip(g['team'], g['season'])]

    # Calculate rate stats
    g['comp_pct'] = np.where(g['pass_plays'] > 0, g['comp'] / g['pass_plays'], 0.0)
    g['ypa'] = np.where(g['pass_plays'] > 0, g['pass_yards'] / g['pass_plays'], 0.0)
    g['td_rate'] = np.where(g['pass_plays'] > 0, g['pass_td'] / g['pass_plays'], 0.0)
    g['int_rate'] = np.where(g['pass_plays'] > 0, g['interceptions'] / g['pass_plays'], 0.0)
    g['sack_rate'] = np.where((g['pass_plays'] + g['sacks']) > 0, g['sacks'] / (g['pass_plays'] + g['sacks']), 0.0)
    g['air_ypa'] = np.where(g['pass_plays'] > 0, g['air_yards'] / g['pass_plays'], 0.0)
    g['yac_ypa'] = np.where(g['pass_plays'] > 0, g['yac'] / g['pass_plays'], 0.0)
    g['epa_per_pass'] = np.where(g['pass_plays'] > 0, g['epa_sum'] / g['pass_plays'], 0.0)
    g['rush_yards_per_att'] = np.where(g['rush_plays'] > 0, g['rush_yards'] / g['rush_plays'], 0.0)
    denom = g['pass_plays'] + g['rush_plays']
    g['passer_share'] = np.where(denom > 0, g['pass_plays'] / denom, 0.0)

    g = g.rename(columns={'player_id': 'fantasy_player_id', 'player_name': 'fantasy_player_name'})
    g = g.drop(columns=[c for c in g.columns if c.startswith('__')], errors='ignore')
    return g

# Build QB features and merge with fantasy points
print("Building QB features...")
qb_feats = build_position_features(ALL_SEASONS, engineer_qb_features_for_season, "QB",
    lambda x: x[(x['pass_plays'] >= 150) | ((x['pass_plays'] >= 100) & (x['passer_share'] >= 0.60))].copy())
qb_df = merge_position_with_targets(qb_feats, players_leaders_multi, "QB")
print(f"✓ QB features complete: {len(qb_df)} records\n")

# ================================
# WR Feature Engineering
# ================================
# Wide receiver features include receiving stats (targets, receptions, yards,
# TDs, air yards, YAC, ADOT), rushing stats (some WRs get carries), and
# rate stats (YPR, YPT, target share).

def _ensure_wr_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures WR-relevant columns exist and are properly typed."""
    return _ensure_cols(df,
        must_int=["complete_pass","pass_touchdown","rush_touchdown","pass_attempt","rush_attempt"],
        must_float=["receiving_yards","rushing_yards","air_yards","yards_after_catch","epa"],
        alias_map={"pass":"pass_attempt","rush":"rush_attempt"})

def engineer_wr_features_for_season(season: int) -> pd.DataFrame:
    """Engineers WR features (targets, receptions, yards, TDs, ADOT, YPR, YPT, target share)."""
    pbp = _load_season_pbp(season, _ensure_wr_cols)

    # Identify pass plays where WR was targeted
    is_pass = (pbp["pass_attempt"] == 1)
    has_tgt = pbp["receiver_player_id"].notna()
    tgt_rows = pbp[is_pass & has_tgt].copy()
    rec_rows = pbp[(pbp["complete_pass"] == 1) & has_tgt].copy()

    # Aggregate receiving stats
    rec_g = (rec_rows.groupby(["season", "receiver_player_id", "receiver_player_name"], as_index=False)
        .agg(receptions=("complete_pass", "sum"), rec_yards=("receiving_yards", "sum"),
            rec_td=("pass_touchdown", "sum"), air_yards=("air_yards", "sum"),
            yac=("yards_after_catch", "sum"), rec_plays=("play_id", "count")))

    # Aggregate targets (all pass attempts, not just completions)
    tgt_g = (tgt_rows.groupby(["season", "receiver_player_id", "receiver_player_name"], as_index=False)
        .agg(targets=("play_id", "count")))

    # Some WRs also get rushing attempts
    is_rush = (pbp["rush_attempt"] == 1)
    rush_rows = pbp[is_rush & pbp["rusher_player_id"].notna()].copy()
    rush_g = (rush_rows.groupby(["season", "rusher_player_id", "rusher_player_name"], as_index=False)
        .agg(wr_rush_att=("rush_attempt", "sum"), wr_rush_yards=("rushing_yards", "sum"),
            wr_rush_td=("rush_touchdown", "sum"))
        .rename(columns={"rusher_player_id": "receiver_player_id", "rusher_player_name": "receiver_player_name"}))

    # Merge all the stats together
    wr = (tgt_g.merge(rec_g, on=["season","receiver_player_id","receiver_player_name"], how="outer")
             .merge(rush_g, on=["season","receiver_player_id","receiver_player_name"], how="left")
             .fillna({"receptions":0, "rec_yards":0.0, "rec_td":0, "air_yards":0.0, "yac":0.0, "rec_plays":0,
                 "wr_rush_att":0, "wr_rush_yards":0.0, "wr_rush_td":0, "targets":0}))

    # Get team information
    team_source = pd.concat([
        tgt_rows[["season","receiver_player_id","posteam"]].rename(columns={"receiver_player_id":"pid", "posteam":"team"}),
        rush_rows[["season","rusher_player_id","posteam"]].rename(columns={"rusher_player_id":"pid", "posteam":"team"}),
    ], ignore_index=True)

    team_mode = (team_source.dropna(subset=["pid","team"])
                   .groupby(["season","pid"])["team"]
                   .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) > 0 else np.nan)
                   .reset_index()
                   .rename(columns={"pid":"receiver_player_id","team":"team"}))

    wr = wr.merge(team_mode, on=["season","receiver_player_id"], how="left")
    if "team" in wr.columns:
        wr["team"] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                      for t, s in zip(wr["team"], wr["season"])]

    # Calculate team total targets to compute target share
    team_tgts = (tgt_rows.groupby(["season","posteam"], as_index=False)
                .agg(team_targets=("play_id","count"))
                .rename(columns={"posteam":"team_raw"}))
    wr = wr.merge(team_tgts, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    wr = wr.drop(columns=["team_raw"], errors="ignore")
    wr["team_targets"] = wr["team_targets"].fillna(0).astype(float)

    # Calculate rate stats
    wr["adot"] = np.where(wr["targets"] > 0, wr["air_yards"] / wr["targets"], 0.0)  # Average depth of target
    wr["ypr"] = np.where(wr["receptions"] > 0, wr["rec_yards"] / wr["receptions"], 0.0)  # Yards per reception
    wr["ypt"] = np.where(wr["targets"] > 0, wr["rec_yards"] / wr["targets"], 0.0)  # Yards per target
    wr["yac_per_rec"] = np.where(wr["receptions"] > 0, wr["yac"] / wr["receptions"], 0.0)  # YAC per reception
    wr["target_share"] = np.where(wr["team_targets"] > 0, wr["targets"] / wr["team_targets"], 0.0)  # % of team targets

    # Calculate games played
    inv_rows = pd.concat([
        tgt_rows[["season","week","receiver_player_id"]].rename(columns={"receiver_player_id":"pid"}),
        rush_rows[["season","week","rusher_player_id"]].rename(columns={"rusher_player_id":"pid"})
    ], ignore_index=True).dropna(subset=["pid"])

    inv_weeks = (inv_rows.groupby(["season","pid"], as_index=False)["week"]
                .nunique()
                .rename(columns={"pid":"receiver_player_id","week":"games"}))

    wr = wr.merge(inv_weeks, on=["season","receiver_player_id"], how="left")
    wr["games"] = wr["games"].fillna(0).astype(float)

    wr = wr.rename(columns={"receiver_player_id":"fantasy_player_id","receiver_player_name":"fantasy_player_name"})

    cols_order = ["season","fantasy_player_id","fantasy_player_name","team",
        "targets","receptions","rec_yards","rec_td","air_yards","yac",
        "adot","ypr","ypt","yac_per_rec", "team_targets","target_share",
        "wr_rush_att","wr_rush_yards","wr_rush_td", "games"]
    wr = wr[[c for c in cols_order if c in wr.columns]]
    return wr

# Build WR features and merge with fantasy points
print("Building WR features...")
wr_feats = build_position_features(ALL_SEASONS, engineer_wr_features_for_season, "WR",
    lambda x: x[(x["targets"] >= 50) | ((x["targets"] >= 30) & (x["games"] >= 4))].copy())
wr_df = merge_position_with_targets(wr_feats, players_leaders_multi, "WR")
print(f"✓ WR features complete: {len(wr_df)} records\n")



# ================================
# RB Feature Engineering  
# ================================
# Running back features include both rushing stats (carries, yards, TDs, goal-line carries)
# and receiving stats (targets, receptions, yards). RBs are unique because they get
# touches in both the running and passing game, so we need to capture both aspects.

# ================================
# RB dataframe construction
# ================================
def _ensure_rb_cols(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_cols(df,
        must_int=["pass_attempt","rush_attempt","complete_pass","rush_touchdown","pass_touchdown","fumble_lost"],
        must_float=["rushing_yards","receiving_yards","air_yards","yards_after_catch","epa","yardline_100"],
        alias_map={"pass":"pass_attempt","rush":"rush_attempt"})

def engineer_rb_features_for_season(season: int) -> pd.DataFrame:
    pbp = _load_season_pbp(season, _ensure_rb_cols)

    rush_rows = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()].copy()
    rush_rows["gl_att"] = (rush_rows["yardline_100"] <= 5).astype(int)
    rush_rows["i10_att"] = (rush_rows["yardline_100"] <= 10).astype(int)
    rush_rows["fl_rush"] = ((rush_rows["fumble_lost"] == 1) &
        (rush_rows["rusher_player_id"] == rush_rows.get("fumbled_1_player_id"))).astype(int)

    rush_g = (rush_rows.groupby(["season", "rusher_player_id", "rusher_player_name"], as_index=False)
        .agg(carries=("rush_attempt", "sum"), rush_yards=("rushing_yards", "sum"),
            rush_td=("rush_touchdown", "sum"), rush_epa_sum=("epa", "sum"),
            gl_carries=("gl_att", "sum"), i10_carries=("i10_att", "sum"),
            fumbles_lost_rush=("fl_rush", "sum"))
        .rename(columns={"rusher_player_id":"player_id", "rusher_player_name":"player_name"}))

    tgt_rows = pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()].copy()
    rec_rows = pbp[(pbp["complete_pass"] == 1) & pbp["receiver_player_id"].notna()].copy()
    rec_rows["fl_rec"] = ((rec_rows["fumble_lost"] == 1) &
        (rec_rows["receiver_player_id"] == rec_rows.get("fumbled_1_player_id"))).astype(int)

    tgt_g = (tgt_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(targets=("play_id","count"))
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"}))

    rec_g = (rec_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(receptions=("complete_pass","sum"), rec_yards=("receiving_yards","sum"),
            rec_td=("pass_touchdown","sum"), air_yards=("air_yards","sum"),
            yac=("yards_after_catch","sum"), rec_epa_sum=("epa","sum"),
            fumbles_lost_rec=("fl_rec","sum"))
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"}))

    rb = (rush_g.merge(tgt_g, on=["season","player_id","player_name"], how="outer")
              .merge(rec_g, on=["season","player_id","player_name"], how="outer")
              .fillna({"carries":0, "rush_yards":0.0, "rush_td":0, "rush_epa_sum":0.0, "gl_carries":0,
                  "i10_carries":0, "fumbles_lost_rush":0, "targets":0, "receptions":0, "rec_yards":0.0,
                  "rec_td":0, "air_yards":0.0, "yac":0.0, "rec_epa_sum":0.0, "fumbles_lost_rec":0}))

    team_source = pd.concat([
        rush_rows[["season","rusher_player_id","posteam"]].rename(columns={"rusher_player_id":"pid","posteam":"team"}),
        tgt_rows[["season","receiver_player_id","posteam"]].rename(columns={"receiver_player_id":"pid","posteam":"team"})
    ], ignore_index=True)

    team_mode = (team_source.dropna(subset=["pid","team"])
                   .groupby(["season","pid"])["team"]
                   .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) > 0 else np.nan)
                   .reset_index()
                   .rename(columns={"pid":"player_id","team":"team"}))

    rb = rb.merge(team_mode, on=["season","player_id"], how="left")
    if "team" in rb.columns:
        rb["team"] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                      for t, s in zip(rb["team"], rb["season"])]

    team_carries = (rush_rows.groupby(["season","posteam"], as_index=False)
                 .agg(team_carries=("rush_attempt","sum"))
                 .rename(columns={"posteam":"team_raw"}))
    team_targets = (tgt_rows.groupby(["season","posteam"], as_index=False)
                .agg(team_targets=("play_id","count"))
                .rename(columns={"posteam":"team_raw"}))

    rb = rb.merge(team_carries, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    rb = rb.merge(team_targets, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    rb = rb.drop(columns=["team_raw"], errors="ignore")
    rb["team_carries"] = rb["team_carries"].fillna(0).astype(float)
    rb["team_targets"] = rb["team_targets"].fillna(0).astype(float)

    rb["ypc"] = np.where(rb["carries"] > 0, rb["rush_yards"] / rb["carries"], 0.0)
    rb["ypr"] = np.where(rb["receptions"] > 0, rb["rec_yards"] / rb["receptions"], 0.0)
    rb["ypt"] = np.where(rb["targets"] > 0, rb["rec_yards"] / rb["targets"], 0.0)
    rb["yac_per_rec"] = np.where(rb["receptions"] > 0, rb["yac"] / rb["receptions"], 0.0)
    rb["scrimmage_yards"] = rb["rush_yards"] + rb["rec_yards"]
    rb["touches"] = rb["carries"] + rb["receptions"]
    rb["opportunities"] = rb["carries"] + rb["targets"]
    rb["yards_per_touch"] = np.where(rb["touches"] > 0, rb["scrimmage_yards"] / rb["touches"], 0.0)
    rb["td_total"] = rb["rush_td"] + rb["rec_td"]
    rb["fumbles_lost_total"] = rb["fumbles_lost_rush"] + rb["fumbles_lost_rec"]
    rb["rush_share"] = np.where(rb["team_carries"] > 0, rb["carries"] / rb["team_carries"], 0.0)
    rb["target_share"] = np.where(rb["team_targets"] > 0, rb["targets"] / rb["team_targets"], 0.0)

    inv_rows = pd.concat([
        rush_rows[["season","week","rusher_player_id"]].rename(columns={"rusher_player_id":"pid"}),
        tgt_rows[["season","week","receiver_player_id"]].rename(columns={"receiver_player_id":"pid"})
    ], ignore_index=True).dropna(subset=["pid"])

    inv_weeks = (inv_rows.groupby(["season","pid"], as_index=False)["week"]
                .nunique()
                .rename(columns={"pid":"player_id","week":"games"}))

    rb = rb.merge(inv_weeks, on=["season","player_id"], how="left")
    rb["games"] = rb["games"].fillna(0).astype(float)

    rb = rb.rename(columns={"player_id":"fantasy_player_id","player_name":"fantasy_player_name"})

    cols_order = ["season","fantasy_player_id","fantasy_player_name","team",
        "carries","rush_yards","rush_td","gl_carries","i10_carries",
        "targets","receptions","rec_yards","rec_td", "ypc","ypr","ypt","yac","air_yards","yac_per_rec",
        "rush_epa_sum","rec_epa_sum", "team_carries","team_targets","rush_share","target_share",
        "scrimmage_yards","touches","opportunities","yards_per_touch","td_total",
        "fumbles_lost_rush","fumbles_lost_rec","fumbles_lost_total", "games"]
    rb = rb[[c for c in cols_order if c in rb.columns]]
    return rb

rb_feats = build_position_features(ALL_SEASONS, engineer_rb_features_for_season, "RB",
    lambda x: x[(x["carries"] >= 100) | ((x["carries"] >= 60) & (x["targets"] >= 30)) |
        (x["opportunities"] >= 120)].copy())
rb_df = merge_position_with_targets(rb_feats, players_leaders_multi, "RB")

# ================================
# TE dataframe construction
# ================================
def _ensure_te_cols(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_cols(df,
        must_int=["pass_attempt","rush_attempt","complete_pass","pass_touchdown","rush_touchdown","fumble_lost"],
        must_float=["receiving_yards","rushing_yards","air_yards","yards_after_catch","epa"],
        alias_map={"pass":"pass_attempt","rush":"rush_attempt"})

def engineer_te_features_for_season(season: int) -> pd.DataFrame:
    pbp = _load_season_pbp(season, _ensure_te_cols)

    is_pass = (pbp["pass_attempt"] == 1)
    has_recv = pbp["receiver_player_id"].notna()
    tgt_rows = pbp[is_pass & has_recv].copy()
    rec_rows = pbp[(pbp["complete_pass"] == 1) & has_recv].copy()
    rec_rows["fl_rec"] = ((rec_rows["fumble_lost"] == 1) &
        (rec_rows["receiver_player_id"] == rec_rows.get("fumbled_1_player_id"))).astype(int)

    tgt_g = (tgt_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(targets=("play_id","count"))
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"}))

    rec_g = (rec_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(receptions=("complete_pass","sum"), rec_yards=("receiving_yards","sum"),
            rec_td=("pass_touchdown","sum"), air_yards=("air_yards","sum"),
            yac=("yards_after_catch","sum"), rec_epa_sum=("epa","sum"),
            fumbles_lost_rec=("fl_rec","sum"))
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"}))

    rush_rows = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()].copy()
    rush_rows["fl_rush"] = ((rush_rows["fumble_lost"] == 1) &
        (rush_rows["rusher_player_id"] == rush_rows.get("fumbled_1_player_id"))).astype(int)

    rush_g = (rush_rows.groupby(["season","rusher_player_id","rusher_player_name"], as_index=False)
        .agg(te_rush_att=("rush_attempt","sum"), te_rush_yards=("rushing_yards","sum"),
            te_rush_td=("rush_touchdown","sum"), fumbles_lost_rush=("fl_rush","sum"))
        .rename(columns={"rusher_player_id":"player_id","rusher_player_name":"player_name"}))

    te = (tgt_g.merge(rec_g, on=["season","player_id","player_name"], how="outer")
             .merge(rush_g, on=["season","player_id","player_name"], how="left")
             .fillna({"targets":0, "receptions":0, "rec_yards":0.0, "rec_td":0,
                 "air_yards":0.0, "yac":0.0, "rec_epa_sum":0.0, "fumbles_lost_rec":0,
                 "te_rush_att":0, "te_rush_yards":0.0, "te_rush_td":0, "fumbles_lost_rush":0}))

    team_source = pd.concat([
        tgt_rows[["season","receiver_player_id","posteam"]].rename(columns={"receiver_player_id":"pid","posteam":"team"}),
        rush_rows[["season","rusher_player_id","posteam"]].rename(columns={"rusher_player_id":"pid","posteam":"team"}),
    ], ignore_index=True)

    team_mode = (team_source.dropna(subset=["pid","team"])
                   .groupby(["season","pid"])["team"]
                   .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) > 0 else np.nan)
                   .reset_index()
                   .rename(columns={"pid":"player_id","team":"team"}))

    te = te.merge(team_mode, on=["season","player_id"], how="left")
    if "team" in te.columns:
        te["team"] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                      for t, s in zip(te["team"], te["season"])]

    team_tgts = (tgt_rows.groupby(["season","posteam"], as_index=False)
                .agg(team_targets=("play_id","count"))
                .rename(columns={"posteam":"team_raw"}))
    te = te.merge(team_tgts, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    te = te.drop(columns=["team_raw"], errors="ignore")
    te["team_targets"] = te["team_targets"].fillna(0).astype(float)

    te["adot"] = np.where(te["targets"] > 0, te["air_yards"] / te["targets"], 0.0)
    te["ypr"] = np.where(te["receptions"] > 0, te["rec_yards"] / te["receptions"], 0.0)
    te["ypt"] = np.where(te["targets"] > 0, te["rec_yards"] / te["targets"], 0.0)
    te["yac_per_rec"] = np.where(te["receptions"] > 0, te["yac"] / te["receptions"], 0.0)
    te["target_share"] = np.where(te["team_targets"] > 0, te["targets"] / te["team_targets"], 0.0)

    inv_rows = pd.concat([
        tgt_rows[["season","week","receiver_player_id"]].rename(columns={"receiver_player_id":"pid"}),
        rush_rows[["season","week","rusher_player_id"]].rename(columns={"rusher_player_id":"pid"})
    ], ignore_index=True).dropna(subset=["pid"])
    inv_weeks = (inv_rows.groupby(["season","pid"], as_index=False)["week"]
                .nunique()
                .rename(columns={"pid":"player_id","week":"games"}))
    te = te.merge(inv_weeks, on=["season","player_id"], how="left")
    te["games"] = te["games"].fillna(0).astype(float)

    te = te.rename(columns={"player_id":"fantasy_player_id","player_name":"fantasy_player_name"})

    cols_order = ["season","fantasy_player_id","fantasy_player_name","team",
        "targets","receptions","rec_yards","rec_td","air_yards","yac",
        "adot","ypr","ypt","yac_per_rec", "team_targets","target_share",
        "te_rush_att","te_rush_yards","te_rush_td", "rec_epa_sum", "games"]
    te = te[[c for c in cols_order if c in te.columns]]
    return te

te_feats = build_position_features(ALL_SEASONS, engineer_te_features_for_season, "TE",
    lambda x: x[(x["targets"] >= 40) | ((x["targets"] >= 25) & (x["games"] >= 4))].copy())
te_df = merge_position_with_targets(te_feats, players_leaders_multi, "TE")

# ================================
# K dataframe construction
# ================================
def _ensure_k_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_cols(df,
        must_int=["field_goal_attempt","extra_point_attempt"],
        must_str=["field_goal_result","extra_point_result"],
        must_float=["kick_distance"])
    if "field_goal_attempt" not in df.columns and "field_goal_result" in df.columns:
        df["field_goal_attempt"] = (df["field_goal_result"].notna()).astype("int64")
    if "extra_point_attempt" not in df.columns and "extra_point_result" in df.columns:
        df["extra_point_attempt"] = (df["extra_point_result"].notna()).astype("int64")
    return df

def _bucket_fg_points(is_made: pd.Series, dist: pd.Series) -> pd.Series:
    """ESPN-like FG made buckets: 0-39 = 3, 40-49 = 4, 50-59 = 5, 60+ = 6."""
    return np.where(is_made & (dist >= 60), 6,
           np.where(is_made & (dist >= 50), 5,
           np.where(is_made & (dist >= 40), 4,
           np.where(is_made, 3, 0))))

def _kicker_name_cols(df: pd.DataFrame) -> tuple[str, str]:
    id_col = "kicker_player_id" if "kicker_player_id" in df.columns else "fantasy_player_id"
    name_col = "kicker_player_name" if "kicker_player_name" in df.columns else "fantasy_player_name"
    return id_col, name_col

def engineer_k_features_for_season(season: int, miss_pat_minus_one: bool = False) -> pd.DataFrame:
    pbp = _load_season_pbp(season, _ensure_k_cols)

    kick = pbp[(pbp["field_goal_attempt"] == 1) | (pbp["extra_point_attempt"] == 1)].copy()
    id_col, name_col = _kicker_name_cols(kick)

    kick["team"] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                    for t, s in zip(kick.get("posteam", ""), kick["season"])]

    dist = kick["kick_distance"].astype(float)
    fg_attempt = (kick["field_goal_attempt"] == 1)
    fg_made = fg_attempt & (kick["field_goal_result"].isin(["made", "good"]))
    fg_missed = fg_attempt & (kick["field_goal_result"].isin(["missed", "blocked"]))

    fg_0_39_made = (fg_made & (dist < 40)).astype(int)
    fg_40_49_made = (fg_made & (40 <= dist) & (dist < 50)).astype(int)
    fg_50_59_made = (fg_made & (50 <= dist) & (dist < 60)).astype(int)
    fg_60p_made   = (fg_made & (dist >= 60)).astype(int)

    pat_attempt = (kick["extra_point_attempt"] == 1)
    pat_good = pat_attempt & (kick["extra_point_result"].isin(["good", "made"]))
    pat_miss = pat_attempt & (kick["extra_point_result"].isin(["failed", "missed", "blocked"]))

    k_fp_per_play = (_bucket_fg_points(fg_made, dist) + np.where(fg_missed, -1, 0)
        + np.where(pat_good, 1, 0) + np.where(miss_pat_minus_one & pat_miss, -1, 0))

    kick["fg_0_39_made"] = fg_0_39_made
    kick["fg_40_49_made"] = fg_40_49_made
    kick["fg_50_59_made"] = fg_50_59_made
    kick["fg_60p_made"] = fg_60p_made
    kick["fg_miss"] = fg_missed.astype(int)
    kick["pat_miss"] = pat_miss.astype(int)
    kick["k_fp_calc"] = k_fp_per_play

    g = (kick.groupby(["season", id_col, name_col, "team"], as_index=False)
            .agg(fg_att=("field_goal_attempt", "sum"),
                fg_made=("field_goal_result", lambda s: int((s.isin(["made", "good"])).sum())),
                fg_0_39_made=("fg_0_39_made", "sum"), fg_40_49_made=("fg_40_49_made", "sum"),
                fg_50_59_made=("fg_50_59_made", "sum"), fg_60p_made=("fg_60p_made", "sum"),
                fg_miss=("fg_miss", "sum"), pat_att=("extra_point_attempt", "sum"),
                pat_made=("extra_point_result", lambda s: int((s.isin(["good", "made"])).sum())),
                pat_miss=("pat_miss", "sum"), k_fp_calc=("k_fp_calc", "sum"),
                games=("game_id", "nunique")))

    g = g.rename(columns={id_col: "fantasy_player_id", name_col: "fantasy_player_name"})
    return g

def build_k_features(seasons, miss_pat_minus_one: bool = False) -> pd.DataFrame:
    out = build_position_features(seasons, lambda y: engineer_k_features_for_season(y, miss_pat_minus_one), "K")

    from nflreadpy import load_rosters
    rosters_pl = load_rosters(list(set(out["season"].astype(int).tolist())))
    rosters = rosters_pl.to_pandas()

    if "gsis_id" not in rosters.columns:
        rosters["gsis_id"] = rosters.get("player_id", None)
    if "position" not in rosters.columns:
        rosters["position"] = ""
    if "season" not in rosters.columns:
        rosters["season"] = rosters.get("year", out["season"].min())

    pos_tbl = (rosters[["season", "gsis_id", "position"]]
        .dropna(subset=["gsis_id"])
        .rename(columns={"gsis_id": "fantasy_player_id"}))
    pos_tbl = (pos_tbl.groupby(["season", "fantasy_player_id"])["position"]
               .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) else "")
               .reset_index())

    out = out.merge(pos_tbl, on=["season", "fantasy_player_id"], how="left", validate="many_to_one")
    before = len(out)
    out = out[out["position"] == "K"].copy()
    print(f"K position filter: kept {len(out)}/{before} rows where position == 'K'")

    out = out[out["games"] >= 4].copy()
    return out

def merge_k_with_totals(k_feats: pd.DataFrame, k_season_multi: pd.DataFrame) -> pd.DataFrame:
    def _dedupe_k_totals(df: pd.DataFrame) -> pd.DataFrame:
        tot = df.rename(columns={"kicker_player_id": "fantasy_player_id",
            "kicker_player_name": "fantasy_player_name", "posteam": "team_ppr", "total_fp": "fp_ppr"}).copy()
        tot = (tot.sort_values(["season","fantasy_player_id","fp_ppr","games"],
                               ascending=[True, True, False, False])
                   .drop_duplicates(["season","fantasy_player_id"], keep="first"))
        return tot

    k_totals = _dedupe_k_totals(k_season_multi)
    right_cols = ["season", "fantasy_player_id", "fantasy_player_name", "team_ppr",
        "fp_ppr", "games", "fg_made", "fg_att", "pat_made", "pat_att", "ppg"]
    right_cols = [c for c in right_cols if c in k_totals.columns]
    k = k_feats.merge(k_totals[right_cols], on=["season", "fantasy_player_id"],
        how="left", validate="many_to_one")

    def _series(df, col):
        return df[col] if col in df.columns else pd.Series(index=df.index, dtype="object")

    name_engineered = _series(k, "fantasy_player_name")
    name_ppr_y = _series(k, "fantasy_player_name_y")
    name_ppr = name_ppr_y if not name_ppr_y.isna().all() else _series(k, "fantasy_player_name")
    name_kicker = _series(k, "kicker_player_name")
    k["fantasy_player_name"] = name_engineered.fillna(name_ppr).fillna(name_kicker)

    if "fp_ppr" in k.columns:
        k = k.rename(columns={"fp_ppr": "fp"})

    keep_order = ["season", "fantasy_player_id", "fantasy_player_name", "team", "team_ppr",
        "fg_att", "fg_made", "pat_att", "pat_made", "fg_0_39_made", "fg_40_49_made", "fg_50_59_made", "fg_60p_made",
        "fg_miss", "pat_miss", "k_fp_calc", "fp", "ppg", "games", "position"]
    keep_order = [c for c in keep_order if c in k.columns]
    k = k[keep_order].copy()

    if "position" in k.columns:
        k = k[k["position"] == "K"].copy()
    if "games" in k.columns:
        k = k[k["games"] >= 4].copy()

    return k

# ================================
# D/ST Feature Engineering (Function Definitions)
# ================================
# These functions are defined here but called later after all position
# features are built. This keeps the code organized.

def _ensure_dst_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_cols(df,
        must_int=["sack","interception","safety","punt_blocked","field_goal_attempt","return_touchdown"],
        must_str=["field_goal_result","extra_point_result"])
    if "return_touchdown" not in df.columns:
        df["return_touchdown"] = 0
    df["fg_block"] = (df["field_goal_result"] == "blocked").astype("int64")
    for c in ["total_home_score","total_away_score","home_team","away_team"]:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["defteam","return_team"]:
        if c not in df.columns:
            df[c] = ""
    for c in ["fumble_recovery_1_team","fumble_recovery_2_team"]:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _pa_to_pts(pa: int) -> int:
    """Points-allowed → fantasy points (ESPN-ish)."""
    if pa == 0: return 10
    if pa <= 6: return 7
    if pa <= 13: return 4
    if pa <= 17: return 1
    if pa <= 27: return 0
    if pa <= 34: return -1
    if pa <= 45: return -4
    return -5

def engineer_dst_features_for_season(season: int) -> pd.DataFrame:
    pbp = _load_season_pbp(season, _ensure_dst_cols)

    scores = (pbp.groupby(["season","game_id"], as_index=False)
                [["total_home_score","total_away_score","home_team","away_team"]]
                .agg({"total_home_score":"max","total_away_score":"max","home_team":"first","away_team":"first"})
                .rename(columns={"total_home_score":"home_pts_final","total_away_score":"away_pts_final"}))

    home_rows = scores[["season","game_id","home_team","away_pts_final"]].rename(
        columns={"home_team":"team","away_pts_final":"points_allowed"})
    away_rows = scores[["season","game_id","away_team","home_pts_final"]].rename(
        columns={"away_team":"team","home_pts_final":"points_allowed"})
    pa_tbl = pd.concat([home_rows, away_rows], ignore_index=True)
    pa_tbl["team"] = [normalize_team(str(t), int(season)) for t in pa_tbl["team"].astype(str)]
    pa_tbl["pa_pts"] = pa_tbl["points_allowed"].astype(int).map(_pa_to_pts)

    by_def = (pbp.groupby(["season","game_id","defteam"], as_index=False)
           .agg({"sack":"sum","interception":"sum","safety":"sum","punt_blocked":"sum"})
           .rename(columns={"defteam":"team"}))
    by_def["team"] = [normalize_team(str(t), int(season)) for t in by_def["team"].astype(str)]

    by_def_fgblk = (pbp.groupby(["season","game_id","defteam"], as_index=False)["fg_block"].sum()
           .rename(columns={"defteam":"team"}))
    by_def_fgblk["team"] = [normalize_team(str(t), int(season)) for t in by_def_fgblk["team"].astype(str)]

    fr_parts = []
    fr1 = pbp[pbp["fumble_recovery_1_team"].notna()][["season","game_id","fumble_recovery_1_team"]].copy()
    if len(fr1):
        fr1 = fr1.rename(columns={"fumble_recovery_1_team":"team"}); fr1["fr"] = 1
        fr_parts.append(fr1[["season","game_id","team","fr"]])
    fr2 = pbp[pbp["fumble_recovery_2_team"].notna()][["season","game_id","fumble_recovery_2_team"]].copy()
    if len(fr2):
        fr2 = fr2.rename(columns={"fumble_recovery_2_team":"team"}); fr2["fr"] = 1
        fr_parts.append(fr2[["season","game_id","team","fr"]])

    if fr_parts:
        by_fr = (pd.concat(fr_parts, ignore_index=True)
                 .groupby(["season","game_id","team"], as_index=False)["fr"].sum())
        by_fr["team"] = [normalize_team(str(t), int(season)) for t in by_fr["team"].astype(str)]
    else:
        by_fr = pd.DataFrame(columns=["season","game_id","team","fr"])

    ret = pbp[(pbp["return_touchdown"] == 1)].copy()
    by_ret_td = (ret.groupby(["season","game_id","return_team"], as_index=False).size()
                   .rename(columns={"return_team":"team","size":"ret_td"}))
    if len(by_ret_td):
        by_ret_td["team"] = [normalize_team(str(t), int(season)) for t in by_ret_td["team"].astype(str)]

    cols_merge = ["season","game_id","team"]
    dst_g = pa_tbl[cols_merge + ["points_allowed","pa_pts"]].copy()

    for part in [
        by_def[cols_merge + ["sack","interception","safety","punt_blocked"]],
        by_def_fgblk[cols_merge + ["fg_block"]],
        by_fr[cols_merge + ["fr"]],
        by_ret_td[cols_merge + ["ret_td"]] if len(by_ret_td) else pd.DataFrame(columns=cols_merge+["ret_td"])
    ]:
        if len(part):
            dst_g = dst_g.merge(part, on=cols_merge, how="left")

    for c in ["sack","interception","safety","punt_blocked","fg_block","fr","ret_td"]:
        if c not in dst_g.columns: dst_g[c] = 0
        dst_g[c] = dst_g[c].fillna(0).astype(int)
    dst_g["pa_pts"] = dst_g["pa_pts"].fillna(0).astype(int)

    dst_g["dst_fp_calc"] = (dst_g["sack"]*1 + dst_g["interception"]*2 + dst_g["fr"]*2 + dst_g["safety"]*2
        + dst_g["ret_td"]*6 + (dst_g["punt_blocked"] + dst_g["fg_block"])*2 + dst_g["pa_pts"])

    agg = (dst_g.groupby(["season","team"], as_index=False)
             .agg(games=("game_id","nunique"), points_allowed_sum=("points_allowed","sum"),
                 sacks=("sack","sum"), ints=("interception","sum"), fr=("fr","sum"),
                 safeties=("safety","sum"), blk_punts=("punt_blocked","sum"),
                 blk_fgs=("fg_block","sum"), ret_td=("ret_td","sum"),
                 pa_pts_sum=("pa_pts","sum"), dst_fp_calc=("dst_fp_calc","sum")))
    agg["ppg_calc"] = agg["dst_fp_calc"] / agg["games"].replace(0, np.nan)
    agg["sacks_pg"] = agg["sacks"] / agg["games"].replace(0, np.nan)
    agg["takeaways"] = agg["ints"] + agg["fr"]
    agg["takeaways_pg"] = agg["takeaways"] / agg["games"].replace(0, np.nan)

    return agg, dst_g

def build_dst_features(seasons) -> pd.DataFrame:
    def _engineer_dst_wrapper(yr):
        season_agg, _ = engineer_dst_features_for_season(yr)
        return season_agg
    out = build_position_features(seasons, _engineer_dst_wrapper, "D/ST",
        lambda x: x[x["games"] >= 4].copy())

    cols = ["season","team","games", "points_allowed_sum","pa_pts_sum",
        "sacks","ints","fr","safeties","blk_punts","blk_fgs","ret_td",
        "takeaways","dst_fp_calc","ppg_calc", "sacks_pg","takeaways_pg"]
    out = out[[c for c in cols if c in out.columns]]
    return out

def merge_dst_with_totals(dst_feats: pd.DataFrame, dst_season_multi: pd.DataFrame) -> pd.DataFrame:
    right = dst_season_multi.rename(columns={"dst_fp":"fp"}).copy()
    right = (right.sort_values(["season","team","fp"], ascending=[True, True, False])
                  .drop_duplicates(["season","team"], keep="first"))

    dst = dst_feats.merge(right[["season","team","fp"]], on=["season","team"], how="left", validate="one_to_one")

    if "dst_fp_calc" in dst.columns and "fp" in dst.columns:
        dst["fp_delta"] = (dst["dst_fp_calc"] - dst["fp"]).round(3)
    return dst

# ================================
# RB Feature Engineering
# ================================
# Running back features include both rushing stats (carries, yards, TDs,
# goal-line carries, inside-10 carries) and receiving stats (targets,
# receptions, yards). RBs are unique because they get touches in both
# the running and passing game, so we need to capture both aspects.

def _ensure_rb_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures RB-relevant columns exist and are properly typed."""
    return _ensure_cols(df,
        must_int=["pass_attempt","rush_attempt","complete_pass","rush_touchdown","pass_touchdown","fumble_lost"],
        must_float=["rushing_yards","receiving_yards","air_yards","yards_after_catch","epa","yardline_100"],
        alias_map={"pass":"pass_attempt","rush":"rush_attempt"})

def engineer_rb_features_for_season(season: int) -> pd.DataFrame:
    """Engineers RB features (carries, rush yards, TDs, goal-line carries, targets, rush/target share)."""
    pbp = _load_season_pbp(season, _ensure_rb_cols)

    # Identify rushing plays
    rush_rows = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()].copy()
    # Goal-line carries (within 5 yards of end zone)
    rush_rows["gl_att"] = (rush_rows["yardline_100"] <= 5).astype(int)
    # Inside-10 carries (within 10 yards of end zone)
    rush_rows["i10_att"] = (rush_rows["yardline_100"] <= 10).astype(int)
    # Fumbles lost on rush plays
    rush_rows["fl_rush"] = ((rush_rows["fumble_lost"] == 1) &
        (rush_rows["rusher_player_id"] == rush_rows.get("fumbled_1_player_id"))).astype(int)

    # Aggregate rushing stats
    rush_g = (rush_rows.groupby(["season", "rusher_player_id", "rusher_player_name"], as_index=False)
        .agg(carries=("rush_attempt", "sum"), rush_yards=("rushing_yards", "sum"),
            rush_td=("rush_touchdown", "sum"), rush_epa_sum=("epa", "sum"),
            gl_carries=("gl_att", "sum"), i10_carries=("i10_att", "sum"),
            fumbles_lost_rush=("fl_rush", "sum"))
        .rename(columns={"rusher_player_id":"player_id", "rusher_player_name":"player_name"}))

    # Identify receiving plays (targets and receptions)
    tgt_rows = pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()].copy()
    rec_rows = pbp[(pbp["complete_pass"] == 1) & pbp["receiver_player_id"].notna()].copy()
    rec_rows["fl_rec"] = ((rec_rows["fumble_lost"] == 1) &
        (rec_rows["receiver_player_id"] == rec_rows.get("fumbled_1_player_id"))).astype(int)

    tgt_g = (tgt_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(targets=("play_id","count"))
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"}))

    rec_g = (rec_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(receptions=("complete_pass","sum"), rec_yards=("receiving_yards","sum"),
            rec_td=("pass_touchdown","sum"), air_yards=("air_yards","sum"),
            yac=("yards_after_catch","sum"), rec_epa_sum=("epa","sum"),
            fumbles_lost_rec=("fl_rec","sum"))
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"}))

    # Merge all stats together
    rb = (rush_g.merge(tgt_g, on=["season","player_id","player_name"], how="outer")
              .merge(rec_g, on=["season","player_id","player_name"], how="outer")
              .fillna({"carries":0, "rush_yards":0.0, "rush_td":0, "rush_epa_sum":0.0, "gl_carries":0,
                  "i10_carries":0, "fumbles_lost_rush":0, "targets":0, "receptions":0, "rec_yards":0.0,
                  "rec_td":0, "air_yards":0.0, "yac":0.0, "rec_epa_sum":0.0, "fumbles_lost_rec":0}))

    # Get team information
    team_source = pd.concat([
        rush_rows[["season","rusher_player_id","posteam"]].rename(columns={"rusher_player_id":"pid","posteam":"team"}),
        tgt_rows[["season","receiver_player_id","posteam"]].rename(columns={"receiver_player_id":"pid","posteam":"team"})
    ], ignore_index=True)

    team_mode = (team_source.dropna(subset=["pid","team"])
                   .groupby(["season","pid"])["team"]
                   .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) > 0 else np.nan)
                   .reset_index()
                   .rename(columns={"pid":"player_id","team":"team"}))

    rb = rb.merge(team_mode, on=["season","player_id"], how="left")
    if "team" in rb.columns:
        rb["team"] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                      for t, s in zip(rb["team"], rb["season"])]

    # Calculate team totals for share calculations
    team_carries = (rush_rows.groupby(["season","posteam"], as_index=False)
                 .agg(team_carries=("rush_attempt","sum"))
                 .rename(columns={"posteam":"team_raw"}))
    team_targets = (tgt_rows.groupby(["season","posteam"], as_index=False)
                .agg(team_targets=("play_id","count"))
                .rename(columns={"posteam":"team_raw"}))

    rb = rb.merge(team_carries, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    rb = rb.merge(team_targets, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    rb = rb.drop(columns=["team_raw"], errors="ignore")
    rb["team_carries"] = rb["team_carries"].fillna(0).astype(float)
    rb["team_targets"] = rb["team_targets"].fillna(0).astype(float)

    # Calculate rate stats
    rb["ypc"] = np.where(rb["carries"] > 0, rb["rush_yards"] / rb["carries"], 0.0)  # Yards per carry
    rb["ypr"] = np.where(rb["receptions"] > 0, rb["rec_yards"] / rb["receptions"], 0.0)  # Yards per reception
    rb["ypt"] = np.where(rb["targets"] > 0, rb["rec_yards"] / rb["targets"], 0.0)  # Yards per target
    rb["yac_per_rec"] = np.where(rb["receptions"] > 0, rb["yac"] / rb["receptions"], 0.0)  # YAC per reception
    rb["scrimmage_yards"] = rb["rush_yards"] + rb["rec_yards"]  # Total yards
    rb["touches"] = rb["carries"] + rb["receptions"]  # Total touches
    rb["opportunities"] = rb["carries"] + rb["targets"]  # Total opportunities
    rb["yards_per_touch"] = np.where(rb["touches"] > 0, rb["scrimmage_yards"] / rb["touches"], 0.0)
    rb["td_total"] = rb["rush_td"] + rb["rec_td"]  # Total touchdowns
    rb["fumbles_lost_total"] = rb["fumbles_lost_rush"] + rb["fumbles_lost_rec"]  # Total fumbles lost
    rb["rush_share"] = np.where(rb["team_carries"] > 0, rb["carries"] / rb["team_carries"], 0.0)  # % of team carries
    rb["target_share"] = np.where(rb["team_targets"] > 0, rb["targets"] / rb["team_targets"], 0.0)  # % of team targets

    # Calculate games played
    inv_rows = pd.concat([
        rush_rows[["season","week","rusher_player_id"]].rename(columns={"rusher_player_id":"pid"}),
        tgt_rows[["season","week","receiver_player_id"]].rename(columns={"receiver_player_id":"pid"})
    ], ignore_index=True).dropna(subset=["pid"])

    inv_weeks = (inv_rows.groupby(["season","pid"], as_index=False)["week"]
                .nunique()
                .rename(columns={"pid":"player_id","week":"games"}))

    rb = rb.merge(inv_weeks, on=["season","player_id"], how="left")
    rb["games"] = rb["games"].fillna(0).astype(float)

    rb = rb.rename(columns={"player_id":"fantasy_player_id","player_name":"fantasy_player_name"})

    cols_order = ["season","fantasy_player_id","fantasy_player_name","team",
        "carries","rush_yards","rush_td","gl_carries","i10_carries",
        "targets","receptions","rec_yards","rec_td", "ypc","ypr","ypt","yac","air_yards","yac_per_rec",
        "rush_epa_sum","rec_epa_sum", "team_carries","team_targets","rush_share","target_share",
        "scrimmage_yards","touches","opportunities","yards_per_touch","td_total",
        "fumbles_lost_rush","fumbles_lost_rec","fumbles_lost_total", "games"]
    rb = rb[[c for c in cols_order if c in rb.columns]]
    return rb

# Build RB features and merge with fantasy points
print("Building RB features...")
rb_feats = build_position_features(ALL_SEASONS, engineer_rb_features_for_season, "RB",
    lambda x: x[(x["carries"] >= 100) | ((x["carries"] >= 60) & (x["targets"] >= 30)) |
        (x["opportunities"] >= 120)].copy())
rb_df = merge_position_with_targets(rb_feats, players_leaders_multi, "RB")
print(f"✓ RB features complete: {len(rb_df)} records\n")

# ================================
# TE Feature Engineering
# ================================
# Tight end features are similar to WR features (targets, receptions, yards,
# TDs, ADOT, target share) but with different filtering thresholds since
# TEs typically get fewer targets than WRs.

def _ensure_te_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures TE-relevant columns exist and are properly typed."""
    return _ensure_cols(df,
        must_int=["pass_attempt","rush_attempt","complete_pass","pass_touchdown","rush_touchdown","fumble_lost"],
        must_float=["receiving_yards","rushing_yards","air_yards","yards_after_catch","epa"],
        alias_map={"pass":"pass_attempt","rush":"rush_attempt"})

def engineer_te_features_for_season(season: int) -> pd.DataFrame:
    """Engineers TE features (similar to WR: targets, receptions, yards, TDs, ADOT, target share)."""
    pbp = _load_season_pbp(season, _ensure_te_cols)

    is_pass = (pbp["pass_attempt"] == 1)
    has_recv = pbp["receiver_player_id"].notna()
    tgt_rows = pbp[is_pass & has_recv].copy()
    rec_rows = pbp[(pbp["complete_pass"] == 1) & has_recv].copy()
    rec_rows["fl_rec"] = ((rec_rows["fumble_lost"] == 1) &
        (rec_rows["receiver_player_id"] == rec_rows.get("fumbled_1_player_id"))).astype(int)

    tgt_g = (tgt_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(targets=("play_id","count"))
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"}))

    rec_g = (rec_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(receptions=("complete_pass","sum"), rec_yards=("receiving_yards","sum"),
            rec_td=("pass_touchdown","sum"), air_yards=("air_yards","sum"),
            yac=("yards_after_catch","sum"), rec_epa_sum=("epa","sum"),
            fumbles_lost_rec=("fl_rec","sum"))
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"}))

    rush_rows = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()].copy()
    rush_rows["fl_rush"] = ((rush_rows["fumble_lost"] == 1) &
        (rush_rows["rusher_player_id"] == rush_rows.get("fumbled_1_player_id"))).astype(int)

    rush_g = (rush_rows.groupby(["season","rusher_player_id","rusher_player_name"], as_index=False)
        .agg(te_rush_att=("rush_attempt","sum"), te_rush_yards=("rushing_yards","sum"),
            te_rush_td=("rush_touchdown","sum"), fumbles_lost_rush=("fl_rush","sum"))
        .rename(columns={"rusher_player_id":"player_id","rusher_player_name":"player_name"}))

    te = (tgt_g.merge(rec_g, on=["season","player_id","player_name"], how="outer")
             .merge(rush_g, on=["season","player_id","player_name"], how="left")
             .fillna({"targets":0, "receptions":0, "rec_yards":0.0, "rec_td":0,
                 "air_yards":0.0, "yac":0.0, "rec_epa_sum":0.0, "fumbles_lost_rec":0,
                 "te_rush_att":0, "te_rush_yards":0.0, "te_rush_td":0, "fumbles_lost_rush":0}))

    team_source = pd.concat([
        tgt_rows[["season","receiver_player_id","posteam"]].rename(columns={"receiver_player_id":"pid","posteam":"team"}),
        rush_rows[["season","rusher_player_id","posteam"]].rename(columns={"rusher_player_id":"pid","posteam":"team"}),
    ], ignore_index=True)

    team_mode = (team_source.dropna(subset=["pid","team"])
                   .groupby(["season","pid"])["team"]
                   .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) > 0 else np.nan)
                   .reset_index()
                   .rename(columns={"pid":"player_id","team":"team"}))

    te = te.merge(team_mode, on=["season","player_id"], how="left")
    if "team" in te.columns:
        te["team"] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                      for t, s in zip(te["team"], te["season"])]

    team_tgts = (tgt_rows.groupby(["season","posteam"], as_index=False)
                .agg(team_targets=("play_id","count"))
                .rename(columns={"posteam":"team_raw"}))
    te = te.merge(team_tgts, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    te = te.drop(columns=["team_raw"], errors="ignore")
    te["team_targets"] = te["team_targets"].fillna(0).astype(float)

    te["adot"] = np.where(te["targets"] > 0, te["air_yards"] / te["targets"], 0.0)
    te["ypr"] = np.where(te["receptions"] > 0, te["rec_yards"] / te["receptions"], 0.0)
    te["ypt"] = np.where(te["targets"] > 0, te["rec_yards"] / te["targets"], 0.0)
    te["yac_per_rec"] = np.where(te["receptions"] > 0, te["yac"] / te["receptions"], 0.0)
    te["target_share"] = np.where(te["team_targets"] > 0, te["targets"] / te["team_targets"], 0.0)

    inv_rows = pd.concat([
        tgt_rows[["season","week","receiver_player_id"]].rename(columns={"receiver_player_id":"pid"}),
        rush_rows[["season","week","rusher_player_id"]].rename(columns={"rusher_player_id":"pid"})
    ], ignore_index=True).dropna(subset=["pid"])
    inv_weeks = (inv_rows.groupby(["season","pid"], as_index=False)["week"]
                .nunique()
                .rename(columns={"pid":"player_id","week":"games"}))
    te = te.merge(inv_weeks, on=["season","player_id"], how="left")
    te["games"] = te["games"].fillna(0).astype(float)

    te = te.rename(columns={"player_id":"fantasy_player_id","player_name":"fantasy_player_name"})

    cols_order = ["season","fantasy_player_id","fantasy_player_name","team",
        "targets","receptions","rec_yards","rec_td","air_yards","yac",
        "adot","ypr","ypt","yac_per_rec", "team_targets","target_share",
        "te_rush_att","te_rush_yards","te_rush_td", "rec_epa_sum", "games"]
    te = te[[c for c in cols_order if c in te.columns]]
    return te

print("Building TE features...")
te_feats = build_position_features(ALL_SEASONS, engineer_te_features_for_season, "TE",
    lambda x: x[(x["targets"] >= 40) | ((x["targets"] >= 25) & (x["games"] >= 4))].copy())
te_df = merge_position_with_targets(te_feats, players_leaders_multi, "TE")
print(f"✓ TE features complete: {len(te_df)} records\n")

# ================================
# K Feature Engineering
# ================================
# Kicker features include field goal attempts/makes by distance bucket
# (0-39, 40-49, 50-59, 60+), PAT attempts/makes, and various rate stats.

# Build K features
print("Building K features...")
k_feats = build_k_features(ALL_SEASONS, miss_pat_minus_one=False)
k_df = merge_k_with_totals(k_feats, k_season_multi)
print(f"✓ K features complete: {len(k_df)} records\n")

# ================================
# D/ST Feature Engineering
# ================================
# Defense/Special Teams features include sacks, interceptions, fumble
# recoveries, safeties, blocked kicks, return TDs, points allowed, and
# rate stats like sacks per game and takeaways per game.

# Build D/ST features
print("Building D/ST features...")
dst_feats = build_dst_features(ALL_SEASONS)
dst_df = merge_dst_with_totals(dst_feats, dst_season_multi)
print(f"✓ D/ST features complete: {len(dst_df)} records\n")

# Save CSVs for visibility
print("\nSaving CSV files for visibility...")
output_dir = "fantasy_features_csv"
os.makedirs(output_dir, exist_ok=True)

qb_df.to_csv(os.path.join(output_dir, "QB_features.csv"), index=False)
rb_df.to_csv(os.path.join(output_dir, "RB_features.csv"), index=False)
wr_df.to_csv(os.path.join(output_dir, "WR_features.csv"), index=False)
te_df.to_csv(os.path.join(output_dir, "TE_features.csv"), index=False)
k_df.to_csv(os.path.join(output_dir, "K_features.csv"), index=False)
dst_df.to_csv(os.path.join(output_dir, "DST_features.csv"), index=False)

print(f"All CSV files saved successfully in: {os.path.abspath(output_dir)}")
print(f"\nData preparation complete! Ready for modeling.\n")

# ==============================================================================
# PART 2: MACHINE LEARNING MODELS
# ==============================================================================
# Uses MoE approach: train multiple models per position, select best by validation MAE.

# Map position names to dataframes
POSITION_DF_MAP = {
    'QB': qb_df,
    'RB': rb_df,
    'WR': wr_df,
    'TE': te_df,
    'K': k_df,
    'DST': dst_df
}

def load_position_data_from_df(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Prepares data for modeling by shifting features to previous season's stats."""
    df = df.copy()
    
    # Ensure position column is set
    if 'position' not in df.columns:
        df['position'] = position
    
    # Calculate fantasy points per week
    if position == 'K' and 'ppg' in df.columns:
        # Kickers provide ppg directly
        df['fp_per_week'] = df['ppg'].fillna(0)
    elif 'fp' in df.columns and 'games' in df.columns:
        df['fp_per_week'] = df['fp'] / df['games'].replace(0, 1)
        df['fp_per_week'] = df['fp_per_week'].fillna(0)
    else:
        available_cols = list(df.columns)
        missing_cols = [col for col in ['fp', 'games'] if col not in df.columns]
        raise ValueError(
            f"Missing required columns for {position}:\n"
            f"  Missing: {missing_cols}\n"
            f"  Available columns: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}"
        )
    
    # Ensure fantasy_player_id exists
    if 'fantasy_player_id' not in df.columns:
        if 'team' in df.columns:
            df['fantasy_player_id'] = df['team'].astype(str)
        else:
            df['fantasy_player_id'] = df.index.astype(str)
    
    # Ensure name column exists
    if 'fantasy_player_name' not in df.columns:
        if 'team' in df.columns:
            df['fantasy_player_name'] = df['team']
        else:
            df['fantasy_player_name'] = df['fantasy_player_id']
    
    # Normalize position labels
    df['position'] = df['position'].fillna(position)
    df = df.sort_values(['fantasy_player_id', 'season']).reset_index(drop=True)
    
    # Columns we don't want to shift
    no_shift_cols = {
        'season', 'fantasy_player_id', 'fantasy_player_name', 'team', 'position',
        'fp', 'games', 'ppg', 'fp_per_week', 'fp_delta', 'k_fp_calc', 'dst_fp_calc'
    }
    
    # Shift all numeric feature columns to use previous season's values
    # This means when predicting for 2024, we use 2023's stats as features
    feature_columns_to_shift = [
        col for col in df.columns
        if col not in no_shift_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    if feature_columns_to_shift:
        df[feature_columns_to_shift] = (
            df.groupby('fantasy_player_id')[feature_columns_to_shift].shift(1)
        )
    
    # Track which season our features came from (for debugging/analysis)
    df['feature_source_season'] = df['season'] - 1
    
    return df

# ----------------------------------
# Mixture of Experts (MoE) Model Class
# ----------------------------------

class MixtureOfExperts:
    """Trains multiple models (experts) and selects best based on validation MAE."""
    
    def __init__(self, position='QB'):
        """Initialize MoE model for a position."""
        self.position = position
        self.experts = {}  # Dictionary to store all the expert models
        self.scalers = {}  # Dictionary to store scalers for models that need scaling
        self.best_model = None  # The best model after training
        self.best_model_name = None  # Name of the best model
        self.expert_weights = {}  # Weights for each expert
        self.trained = False  # Flag to check if model has been trained
        
    def _initialize_experts(self):
        """Initializes expert models (RF, GB, Ridge, Linear, optionally XGBoost)."""
        self.experts = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,  # Number of trees in the forest
                max_depth=10,  # Maximum depth of each tree
                min_samples_split=5,  # Minimum samples to split a node
                min_samples_leaf=2,  # Minimum samples in a leaf
                random_state=42,  # For reproducibility
                n_jobs=-1  # Use all CPU cores
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,  # Number of boosting stages
                max_depth=5,  # Maximum depth of each tree
                learning_rate=0.1,  # Shrinkage parameter
                subsample=0.8,  # Fraction of samples to use for each tree
                random_state=42  # For reproducibility
            ),
            'ridge': Ridge(alpha=1.0),  # L2 regularization with alpha=1.0
            'linear': LinearRegression()  # Simple linear regression
        }
        
        if XGBOOST_AVAILABLE:
            self.experts['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,  # Fraction of features to use per tree
                random_state=42,
                n_jobs=-1
            )
        
        # Initialize scalers for models that need feature scaling
        # Ridge and Linear Regression benefit from scaled features
        for model_name in ['ridge', 'linear']:
            self.scalers[model_name] = StandardScaler()
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Trains all experts, selects best by MAE, retrains best on full training set."""
        # Initialize all expert models
        self._initialize_experts()
        
        # If no validation set provided, split training data
        # We use 80% for training and 20% for validation
        if X_val is None or y_val is None:
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        else:
            X_train_fit, y_train_fit = X_train, y_train
        
        # Train each expert and evaluate performance
        model_scores = {}  # Store scores for each model
        model_predictions = {}  # Store predictions for each model
        
        print(f"Training expert models for {self.position}...")
        for model_name, model in self.experts.items():
            try:
                # Scale features for models that need it (Ridge, Linear)
                if model_name in self.scalers:
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train_fit)
                    X_val_scaled = self.scalers[model_name].transform(X_val)
                    model.fit(X_train_scaled, y_train_fit)
                    y_pred = model.predict(X_val_scaled)
                else:
                    # Tree-based models don't need scaling
                    model.fit(X_train_fit, y_train_fit)
                    y_pred = model.predict(X_val)
                
                # Calculate evaluation metrics
                mse = mean_squared_error(y_val, y_pred)  # Mean Squared Error
                mae = mean_absolute_error(y_val, y_pred)  # Mean Absolute Error
                r2 = r2_score(y_val, y_pred)  # R-squared
                
                # Use negative MAE as score (lower MAE = better, so negative MAE = higher is better)
                model_scores[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'score': -mae  # Negative because we want to maximize this (lower MAE is better)
                }
                
                model_predictions[model_name] = y_pred
                
                # Print model performance
                print(f"  {model_name:20s} - MAE: {mae:.4f}, RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
                # Set score to negative infinity so it won't be selected
                model_scores[model_name] = {'score': -np.inf}
        
        # Select best model based on MAE (lowest MAE = best)
        if model_scores:
            self.best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['score'])
            self.best_model = self.experts[self.best_model_name]
            print(f"\n  Best model for {self.position}: {self.best_model_name}")
            print(f"  Best model metrics - MAE: {model_scores[self.best_model_name]['mae']:.4f}, "
                  f"RMSE: {np.sqrt(model_scores[self.best_model_name]['mse']):.4f}, "
                  f"R²: {model_scores[self.best_model_name]['r2']:.4f}\n")
        
        # Retrain best model on full training set (not just 80%)
        # This gives it more data to learn from
        if self.best_model_name:
            if self.best_model_name in self.scalers:
                X_train_scaled = self.scalers[self.best_model_name].fit_transform(X_train)
                self.best_model.fit(X_train_scaled, y_train)
            else:
                self.best_model.fit(X_train, y_train)
        
        self.trained = True
        return model_scores
    
    def predict(self, X):
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features if needed
        if self.best_model_name in self.scalers:
            X_scaled = self.scalers[self.best_model_name].transform(X)
            return self.best_model.predict(X_scaled)
        else:
            return self.best_model.predict(X)
    
    def get_best_model_info(self):
        """Return information about the best model."""
        return {
            'position': self.position,
            'best_model': self.best_model_name,
            'trained': self.trained
        }

# ----------------------------------
# Feature Preparation and Model Training Functions
# ----------------------------------
#Preps our feature columns, handles missing values
def prepare_features(df, required_features=None):
    # Columns to exclude from features (these are metadata, not features)
    exclude_cols = [
        'season', 'fantasy_player_id', 'fantasy_player_name',
        'position', 'fp', 'games', 'fp_per_week', 'team',
        'ppg', 'feature_source_season', 'fp_delta', 'k_fp_calc', 'dst_fp_calc',
        'team_ppr'
    ]
    
    if required_features is None:
        # Automatically select features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Filter out features with too many missing values or no variance
        valid_features = []
        for col in feature_cols:
            # Only keep features with < 50% missing values
            if df[col].isna().sum() / len(df) < 0.5:  
                # Only keep features with more than 1 unique value
                if df[col].nunique() > 1:
                    valid_features.append(col)
    else:
        valid_features = required_features
    
    # Ensure all required feature columns exist
    missing_cols = [col for col in valid_features if col not in df.columns]
    for col in missing_cols:
        df[col] = np.nan
    
    # Select features in the specified order
    X = df[valid_features].copy()
    
    # Ensure all features are numeric; convert non-numeric to NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Extract target variable
    y = df['fp_per_week'].values
    
    return X, y, valid_features

#Splits data by year
def temporal_split(df, test_year=2024, val_year=None):
    train_df = df[df['season'] < test_year].copy()
    test_df = df[df['season'] == test_year].copy()
    
    # If validation year specified, use that year for validation
    if val_year and val_year < test_year:
        val_df = train_df[train_df['season'] == val_year].copy()
        train_df = train_df[train_df['season'] < val_year].copy()
    else:
        val_df = None  # Will split from training set later
    
    return train_df, val_df, test_df

def evaluate_model(model, X_test, y_test, position):
    """Evaluates model performance (MAE, RMSE, R², MAPE) and returns metrics and predictions."""
    y_pred = model.predict(X_test)
    
    # Calculate various evaluation metrics
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # R-squared
    
    # Mean Absolute Percentage Error (MAPE) - percentage error
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    metrics = {
        'position': position,
        'mae': mae,  # Average error in points
        'rmse': rmse,  # Penalizes large errors more
        'mse': mse,  # Penalizes large errors even more
        'r2': r2,  # Proportion of variance explained (1.0 = perfect, 0.0 = no better than mean)
        'mape': mape  # Percentage error
    }
    
    return metrics, y_pred
#Generates model comparison report across positions with metrics and best models
def generate_model_comparison_report(all_metrics, models):
    print("\n" + "="*80)
    print("MODEL COMPARISON AND ACCURACY REPORT")
    print("="*80)
    
    # Create comparison DataFrame
    comparison_data = []
    for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        if position in all_metrics:
            metrics = all_metrics[position]
            best_model = models[position].best_model_name if position in models else "N/A"
            
            comparison_data.append({
                'Position': position,
                'Best Model': best_model,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2'],
                'MAPE (%)': metrics['mape']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print summary
    print("\nOverall Model Performance by Position:")
    print(comparison_df.to_string(index=False))
    
    # Find best and worst positions
    if len(comparison_df) > 0:
        best_mae_pos = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Position']
        worst_mae_pos = comparison_df.loc[comparison_df['MAE'].idxmax(), 'Position']
        best_r2_pos = comparison_df.loc[comparison_df['R²'].idxmax(), 'Position']
        worst_r2_pos = comparison_df.loc[comparison_df['R²'].idxmin(), 'Position']
        
        print(f"\nBest MAE (lowest error): {best_mae_pos} ({comparison_df.loc[comparison_df['MAE'].idxmin(), 'MAE']:.4f})")
        print(f"Worst MAE (highest error): {worst_mae_pos} ({comparison_df.loc[comparison_df['MAE'].idxmax(), 'MAE']:.4f})")
        print(f"Best R² (highest explained variance): {best_r2_pos} ({comparison_df.loc[comparison_df['R²'].idxmax(), 'R²']:.4f})")
        print(f"Worst R² (lowest explained variance): {worst_r2_pos} ({comparison_df.loc[comparison_df['R²'].idxmin(), 'R²']:.4f})")
        
        # Model type distribution
        model_counts = comparison_df['Best Model'].value_counts()
        print(f"\nBest Model Distribution:")
        for model, count in model_counts.items():
            print(f"  {model}: {count} position(s)")
    
    return comparison_df

def train_position_model_with_df(position, df, test_year=2024):
    print(f"\n{'='*60}")
    print(f"Training MoE model for {position}")
    print(f"{'='*60}")
    
    print(f"Loading data from dataframe...")
    # Prepare data for modeling
    df_prep = load_position_data_from_df(df, position)
    print(f"Loaded {len(df_prep)} records spanning {df_prep['season'].min()}-{df_prep['season'].max()}")
    
    # Temporal split
    print(f"\nPerforming temporal split (test year: {test_year})...")
    train_df, val_df, test_df = temporal_split(df_prep, test_year=test_year)
    
    print(f"Training set: {len(train_df)} records (seasons {train_df['season'].min()}-{train_df['season'].max()})")
    if val_df is not None and len(val_df) > 0:
        print(f"Validation set: {len(val_df)} records (season {val_df['season'].unique()})")
    print(f"Test set: {len(test_df)} records (season {test_year})")
    
    # Handle case where test year doesn't exist
    if len(test_df) == 0:
        print(f"Warning: No test data found for {test_year}.")
        latest_year = df_prep['season'].max()
        if latest_year < test_year:
            print(f"Latest available year is {latest_year}. Using {latest_year} as test set.")
            test_year = latest_year
            train_df, val_df, test_df = temporal_split(df_prep, test_year=test_year)
            print(f"Updated test set: {len(test_df)} records (season {test_year})")
        else:
            raise ValueError(f"No data available for {test_year} or any suitable test year.")
    
    # Check if we have enough training data
    if len(train_df) < 10:
        raise ValueError(f"Insufficient training data: only {len(train_df)} records. Need at least 10.")
    
    # Prepare features
    print(f"\nPreparing features...")
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df, required_features=feature_names)
    
    if val_df is not None and len(val_df) > 0:
        X_val, y_val, _ = prepare_features(val_df, required_features=feature_names)
    else:
        X_val, y_val = None, None
    
    if len(feature_names) == 0:
        raise ValueError("No valid features found after preprocessing. Check data quality.")
    
    print(f"Features: {len(feature_names)}")
    if len(feature_names) <= 20:
        print(f"Feature names: {', '.join(feature_names)}")
    else:
        print(f"Feature names (first 10): {', '.join(feature_names[:10])}...")
        print(f"Total features: {len(feature_names)}")
    
    # Train MoE model and selects best one
    print(f"\nTraining MoE model...")
    moe = MixtureOfExperts(position=position)
    model_scores = moe.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set 
    print(f"Evaluating on test set...")
    metrics, y_pred = evaluate_model(moe, X_test, y_test, position)
    
    print(f"\nTest Set Performance for {position}:")
    print(f"  MAE:  {metrics['mae']:.4f}")  # Average error in points
    print(f"  RMSE: {metrics['rmse']:.4f}")  # Penalizes large errors
    print(f"  R²:   {metrics['r2']:.4f}")  # Explained variance
    print(f"  MAPE: {metrics['mape']:.2f}%")  # Percentage error
    
    # Prepare results
    test_df = test_df.copy()
    test_df['predicted_fp_per_week'] = y_pred
    test_df['actual_fp_per_week'] = y_test
    
    return moe, test_df, metrics, feature_names

# ----------------------------------
# Main Execution
# ----------------------------------
print("\n" + "="*80)
print("PHASE 3: MODEL TRAINING AND PREDICTION")
print("="*80)
print("Training Mixture of Experts models for all positions...\n")

# Storage for models and results
models = {}
all_test_results = {}
all_metrics = {}

# Train models for each position using dataframes directly
for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
    if position not in POSITION_DF_MAP:
        print(f"Warning: No dataframe found for {position}. Skipping...")
        continue
    
    try:
        df = POSITION_DF_MAP[position]
        model, test_results, metrics, features = train_position_model_with_df(
            position, df, test_year=2024
        )
        models[position] = model
        all_test_results[position] = test_results
        all_metrics[position] = metrics
    except Exception as e:
        print(f"Error training {position} model: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Generate model comparison report
comparison_report = generate_model_comparison_report(all_metrics, models)

# Generate final predictions DataFrame
print(f"\n{'='*60}")
print("Generating Final Predictions DataFrame")
print(f"{'='*60}")

predictions_list = []

for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
    if position in all_test_results:
        test_df = all_test_results[position]
        
        for idx, row in test_df.iterrows():
            predictions_list.append({
                'Player Name': row.get('fantasy_player_name', 'Unknown'),
                'Player Position': position,
                'Projected Average Fantasy Points Per Week': row.get('predicted_fp_per_week', 0),
                'Actual Average Fantasy Points Per Week': row.get('actual_fp_per_week', 0),
                'Season': row.get('season', 2024)
            })

# Create final DataFrame
final_predictions = pd.DataFrame(predictions_list)

# Check if we have any predictions
if len(final_predictions) == 0:
    print("\nWARNING: No predictions were generated!")
    print("This likely means no models were successfully trained.")
else:
    # Sort by position and projected points
    final_predictions = final_predictions.sort_values(
        ['Player Position', 'Projected Average Fantasy Points Per Week'], 
        ascending=[True, False]
    )
    
    # Save predictions to CSV files
    output_dir = "fantasy_features_csv"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full version with actuals for comparison
    output_file_full = os.path.join(output_dir, 'fantasy_predictions_2024_full.csv')
    final_predictions.to_csv(output_file_full, index=False)
    print(f"\nFull predictions (with actuals) saved to {output_file_full}")
    
    # Save required output format
    required_columns = ['Player Name', 'Player Position', 'Projected Average Fantasy Points Per Week']
    final_predictions_required = final_predictions[required_columns].copy()
    output_file = os.path.join(output_dir, 'fantasy_predictions_2024.csv')
    final_predictions_required.to_csv(output_file, index=False)
    print(f"Required predictions saved to {output_file}")
    
    # Save model comparison report
    comparison_file = os.path.join(output_dir, 'model_comparison_report.csv')
    comparison_report.to_csv(comparison_file, index=False)
    print(f"Model comparison report saved to {comparison_file}")
    
    # Print top predictions by position
    print(f"\n{'='*60}")
    print("Top 10 Projected Players by Position")
    print(f"{'='*60}")
    
    for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        pos_preds = final_predictions[final_predictions['Player Position'] == position]
        if len(pos_preds) > 0:
            top_10 = pos_preds.head(10)
            print(f"\n{position}:")
            for idx, row in top_10.iterrows():
                print(f"  {row['Player Name']:30s} - "
                      f"Projected: {row['Projected Average Fantasy Points Per Week']:6.2f} | "
                      f"Actual: {row['Actual Average Fantasy Points Per Week']:6.2f}")
        else:
            print(f"\n{position}: No predictions available")

print(f"\n{'='*80}")
print("MODEL TRAINING AND PREDICTION COMPLETE!")
print(f"{'='*80}\n")