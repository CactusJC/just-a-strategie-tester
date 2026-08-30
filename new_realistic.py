"""
================================================================================
XAUUSD - PROP FIRM SIMULATOR + GRID SEARCH OPTIMIZER
================================================================================
Autonomous Quantitative Architecture with VectorBT, Numba, Pandas, Pandas-TA.
================================================================================
"""

import os
import warnings

# Multi-threading deadlock prevention: set single thread environment before importing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

import gc
import glob
import time
import itertools
import pandas as pd
import numpy as np
import pandas_ta as ta
import vectorbt as vbt

import numba
from numba import njit

numba.set_num_threads(1)

# ==============================================================================
# CONFIGURATIE
# ==============================================================================
DEFAULT_CACHE_DIR = "cache"
PRIMARY_TICK_FILE = r"C:\workbase\cache\XAUUSD_202608140100_202608282359_tick.csv"

# Prop Firm Rules
INITIAL_BALANCE    = 100_000.0
MAX_DRAWDOWN_PCT   = 0.08
DAILY_DRAWDOWN_PCT = 0.05
PAYOUT_TARGET_PCT  = 0.05
CHALLENGE_FEE      = 49.0

# Trade Specs
CONTRACT_SIZE      = 100.0
FIXED_LOT_SIZE     = 0.02
COMMISSION_PER_LOT = 5.0
SWAP_LONG          = -5.0
SWAP_SHORT         = -2.0
TRIPLE_SWAP_DAY    = "Wed"

# Time Filters
NO_OVERNIGHT = True
NO_WEEKEND   = True
EOD_HOUR     = 23
EOD_MINUTE   = 55

# ==============================================================================
# GRID SEARCH RUIMTE
# ==============================================================================
PARAM_GRID = {
    "ema_fast":   [10, 15, 20, 25],
    "ema_slow":   [30, 40, 50, 60],
    "rsi_len":    [10, 14, 21],
    "rsi_entry":  [50, 55, 60],
}

MIN_TRADES     = 10
MC_TOP_N       = 5
MC_ITERATIONS  = 1000
MC_SEED        = 42

# ==============================================================================
# 01. LOAD DATA (TICKS OR OHLC BARS)
# ==============================================================================
def resolve_data_file(filepath: str = PRIMARY_TICK_FILE) -> str:
    """Resolves data filepath, falling back to cached CSV files if provided path does not exist."""
    if os.path.exists(filepath):
        return filepath

    if os.path.exists(DEFAULT_CACHE_DIR):
        csv_files = glob.glob(os.path.join(DEFAULT_CACHE_DIR, "*.csv"))
        if csv_files:
            print(f"[DATA] Specified file not found. Using cached file: {csv_files[0]}")
            return csv_files[0]

    raise FileNotFoundError(f"No valid market data file found at '{filepath}' or in '{DEFAULT_CACHE_DIR}/'")


def load_market_data(filepath: str = PRIMARY_TICK_FILE) -> pd.DataFrame:
    """
    Inlading van marktdata (Ticks of OHLC M1/M5 bar data).
    Bouwt een uniforme execution dataframe met Bid en Ask kolommen.
    """
    resolved_path = resolve_data_file(filepath)
    print(f"\n[01] Marktdata inladen uit: {resolved_path}")

    sample = pd.read_csv(resolved_path, sep="\t", nrows=5)
    cols = [c.strip("<>") for c in sample.columns]

    if "BID" in cols and "ASK" in cols:
        df = pd.read_csv(resolved_path, sep="\t")
        df.columns = [c.strip("<>") for c in df.columns]
        df["time"] = pd.to_datetime(
            df["DATE"] + " " + df["TIME"],
            format="%Y.%m.%d %H:%M:%S.%f",
            errors="coerce"
        )
        df = df.dropna(subset=["time"]).set_index("time").sort_index()

        print("[01] Resamplen tick data naar M1 execution dataframe...")
        m1 = df.resample("1min").agg({
            "BID": ["first", "max", "min", "last"],
            "ASK": ["first", "max", "min", "last"]
        }).dropna()
        m1.columns = [
            "Bid_O", "Bid_H", "Bid_L", "Bid_C",
            "Ask_O", "Ask_H", "Ask_L", "Ask_C"
        ]
        return m1

    elif "CLOSE" in cols or "Close" in cols:
        df = pd.read_csv(resolved_path, sep="\t" if "\t" in open(resolved_path).readline() else ",")
        df.columns = [c.strip("<>") for c in df.columns]

        if "DATE" in df.columns and "TIME" in df.columns:
            time_str = df["DATE"].astype(str) + " " + df["TIME"].astype(str)
            df["time"] = pd.to_datetime(time_str, errors="coerce")
        elif "Time" in df.columns:
            df["time"] = pd.to_datetime(df["Time"], errors="coerce")
        else:
            df["time"] = pd.to_datetime(df.index, errors="coerce")

        df = df.dropna(subset=["time"]).set_index("time").sort_index()

        spread_pts = df["SPREAD"].values if "SPREAD" in df.columns else 20.0
        spread_usd = spread_pts * 0.01

        m1 = pd.DataFrame(index=df.index)
        m1["Bid_O"] = df["OPEN"]
        m1["Bid_H"] = df["HIGH"]
        m1["Bid_L"] = df["LOW"]
        m1["Bid_C"] = df["CLOSE"]
        m1["Ask_O"] = df["OPEN"] + spread_usd
        m1["Ask_H"] = df["HIGH"] + spread_usd
        m1["Ask_L"] = df["LOW"] + spread_usd
        m1["Ask_C"] = df["CLOSE"] + spread_usd
        return m1.dropna()

    else:
        raise ValueError("Onbekend dataformaat in CSV (verwacht BID/ASK of OPEN/HIGH/LOW/CLOSE).")


# ==============================================================================
# 02. CREATE SIGNALS (Lagged & Time-Filtered)
# ==============================================================================
def create_m5_signals(
    execution_df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 40,
    rsi_len: int = 14,
    rsi_entry: float = 55.0,
) -> pd.DataFrame:
    """Genereert Entry/Exit signalen met anti-look-ahead (shift 1) en time filters."""

    m5 = execution_df["Bid_C"].resample("5min").last().dropna().to_frame(name="Close")
    m5["EMA_F"] = ta.ema(m5["Close"], length=ema_fast)
    m5["EMA_S"] = ta.ema(m5["Close"], length=ema_slow)
    m5["RSI"]   = ta.rsi(m5["Close"], length=rsi_len)

    m5_entries = (m5["EMA_F"] > m5["EMA_S"]) & (m5["RSI"] > rsi_entry)
    m5_exits   = (m5["EMA_F"] < m5["EMA_S"])

    # Strict anti-look-ahead: shift signals by 1 bar
    m5_entries_safe = m5_entries.shift(1).fillna(False).astype(bool)
    m5_exits_safe   = m5_exits.shift(1).fillna(False).astype(bool)

    raw_entries = m5_entries_safe.reindex(execution_df.index, method="ffill").fillna(False).astype(bool)
    raw_exits   = m5_exits_safe.reindex(execution_df.index, method="ffill").fillna(False).astype(bool)

    mins_from_midnight = execution_df.index.hour * 60 + execution_df.index.minute
    eod_mins = EOD_HOUR * 60 + EOD_MINUTE

    is_eod       = (execution_df.index.hour == EOD_HOUR) & (execution_df.index.minute == EOD_MINUTE)
    is_forbidden = (mins_from_midnight >= eod_mins)
    is_friday    = (execution_df.index.dayofweek == 4)
    is_weekend   = (execution_df.index.dayofweek > 4)

    forced_exits  = pd.Series(False, index=execution_df.index)
    block_entries = pd.Series(False, index=execution_df.index)

    if NO_OVERNIGHT:
        forced_exits  = forced_exits | is_eod
        block_entries = block_entries | is_forbidden

    if NO_WEEKEND:
        forced_exits  = forced_exits | (is_eod & is_friday)
        block_entries = block_entries | (is_forbidden & is_friday) | is_weekend

    signals = pd.DataFrame(index=execution_df.index)
    signals["Exit"]  = raw_exits | forced_exits
    signals["Entry"] = raw_entries & ~block_entries
    signals["Entry"] = signals["Entry"] & ~signals["Entry"].shift(1).fillna(False).astype(bool)

    return signals


# ==============================================================================
# 03. REALISTIC VECTORBT EXECUTION
# ==============================================================================
def run_realistic_execution(execution_df: pd.DataFrame, signals: pd.DataFrame) -> vbt.Portfolio:
    """Executes trades with bid-ask spread friction and commissions."""
    half_spread_pct = ((execution_df["Ask_O"] - execution_df["Bid_O"]) / 2) / execution_df["Bid_O"]
    half_comm_pct   = (COMMISSION_PER_LOT / 2) / (CONTRACT_SIZE * execution_df["Bid_O"])
    total_friction  = half_spread_pct + half_comm_pct

    freq_str = pd.infer_freq(execution_df.index) or "5min"

    return vbt.Portfolio.from_signals(
        close=execution_df["Bid_O"],
        entries=signals["Entry"],
        exits=signals["Exit"],
        size=FIXED_LOT_SIZE * CONTRACT_SIZE,
        size_type="amount",
        fees=total_friction,
        init_cash=INITIAL_BALANCE,
        freq=freq_str,
    )


# ==============================================================================
# 04. SWAP ENGINE
# ==============================================================================
def calculate_accurate_swap(
    entry_dates: pd.Series,
    exit_dates: pd.Series,
    is_long: np.ndarray,
    swap_long: float = SWAP_LONG,
    swap_short: float = SWAP_SHORT,
    triple_day: str = TRIPLE_SWAP_DAY,
    exclude_weekend: bool = True
) -> np.ndarray:
    """Calculates holding swap costs with triple swap day accounting."""
    d_start = entry_dates.dt.normalize().values.astype("datetime64[D]")
    d_end   = exit_dates.dt.normalize().values.astype("datetime64[D]")

    invalid = d_end < d_start
    d_end   = np.where(invalid, d_start, d_end)

    total_nights  = (d_end - d_start).astype("timedelta64[D]").astype(int)
    triple_nights = np.busday_count(d_start, d_end, weekmask=triple_day)

    if exclude_weekend:
        weekday_nights   = np.busday_count(d_start, d_end, weekmask="1111100")
        effective_nights = weekday_nights + (triple_nights * 2)
    else:
        effective_nights = total_nights + (triple_nights * 2)

    rates = np.where(is_long, swap_long, swap_short)
    swap  = effective_nights * rates
    return np.where(invalid, 0.0, swap)


def process_trade_history(pf: vbt.Portfolio) -> pd.DataFrame:
    """Extracts trade records and applies compounding swap fees."""
    trades = pf.trades.records_readable.copy()
    if trades.empty:
        return trades

    is_long = trades["Direction"].str.contains("Long", case=False).values
    volume_lots = trades["Size"].values / CONTRACT_SIZE

    raw_swap = calculate_accurate_swap(
        trades["Entry Timestamp"], trades["Exit Timestamp"], is_long
    )
    trades["Swap"]    = raw_swap * volume_lots
    trades["Net_PnL"] = trades["PnL"] + trades["Swap"]
    return trades


# ==============================================================================
# 05. PROP SIMULATOR (Numba Accelerated)
# ==============================================================================
@njit
def sim_prop_firm(
    pnl_array: np.ndarray,
    day_ids: np.ndarray,
    initial_bal: float,
    max_dd_pct: float,
    daily_dd_pct: float,
    target_pct: float,
    fee: float
):
    """
    Numba loop evaluating static drawdown, daily reset drawdown, payout targets, and reset fees.
    """
    account_bal   = initial_bal
    high_water    = initial_bal
    current_day   = -1
    day_start_bal = initial_bal
    payouts = 0
    payout_val = 0.0
    busts = 0

    target_val   = initial_bal * (1.0 + target_pct)
    max_loss_val = initial_bal * (1.0 - max_dd_pct)

    for i in range(len(pnl_array)):
        trade_pnl = pnl_array[i]
        t_day = day_ids[i]

        if t_day != current_day:
            current_day = t_day
            day_start_bal = account_bal

        daily_loss_val = day_start_bal * (1.0 - daily_dd_pct)
        account_bal += trade_pnl

        if account_bal > high_water:
            high_water = account_bal

        if account_bal <= daily_loss_val or account_bal <= max_loss_val:
            busts += 1
            account_bal = initial_bal
            high_water = initial_bal
            day_start_bal = initial_bal
            continue

        if account_bal >= target_val:
            payouts += 1
            payout_val += (account_bal - initial_bal)
            account_bal = initial_bal
            high_water = initial_bal
            day_start_bal = initial_bal

    return payouts, payout_val, busts, (busts * fee)


def evaluate_prop(trades: pd.DataFrame) -> dict:
    """Evaluates strategy performance under prop firm drawdown & target constraints."""
    if trades.empty or len(trades) < MIN_TRADES:
        return {
            "n_trades": len(trades),
            "payouts": 0,
            "busts": 0,
            "net_profit": -1e9,
            "total_swap": 0.0,
            "pnl_array": np.array([], dtype=np.float64),
            "day_ids": np.array([], dtype=np.int64),
        }

    pnl_array = trades["Net_PnL"].values.astype(np.float64)
    day_ids = trades["Exit Timestamp"].dt.date.apply(lambda d: d.toordinal()).values.astype(np.int64)

    payouts, gross, busts, fees = sim_prop_firm(
        pnl_array, day_ids,
        INITIAL_BALANCE, MAX_DRAWDOWN_PCT, DAILY_DRAWDOWN_PCT,
        PAYOUT_TARGET_PCT, CHALLENGE_FEE
    )
    return {
        "n_trades": len(trades),
        "payouts": payouts,
        "busts": busts,
        "net_profit": gross - fees,
        "total_swap": float(trades["Swap"].sum()),
        "pnl_array": pnl_array,
        "day_ids": day_ids,
    }


# ==============================================================================
# 06. MONTE CARLO ENGINE (Numba Compiled & Memory Efficient)
# ==============================================================================
@njit
def run_monte_carlo_numba(
    pnl_array: np.ndarray,
    iterations: int,
    initial_bal: float,
    max_dd_pct: float,
    daily_dd_pct: float,
    target_pct: float,
    fee: float,
    seed: int
) -> np.ndarray:
    np.random.seed(seed)
    n = len(pnl_array)
    results = np.empty(iterations, dtype=np.float64)
    trades_per_day = max(1, n // max(1, n // 3))
    fake_days = np.arange(n) // trades_per_day

    for i in range(iterations):
        indices = np.random.choice(n, size=n, replace=True)
        shuffled = pnl_array[indices]
        _, val, _, fees = sim_prop_firm(
            shuffled, fake_days,
            initial_bal, max_dd_pct, daily_dd_pct,
            target_pct, fee
        )
        results[i] = val - fees

    return results


def run_monte_carlo(pnl_array: np.ndarray, iterations: int = MC_ITERATIONS, seed: int = MC_SEED) -> dict:
    if len(pnl_array) == 0:
        return {"mean": -1e9, "median": -1e9, "p5": -1e9, "win_prob": 0.0}

    results = run_monte_carlo_numba(
        pnl_array.astype(np.float64), iterations,
        INITIAL_BALANCE, MAX_DRAWDOWN_PCT, DAILY_DRAWDOWN_PCT,
        PAYOUT_TARGET_PCT, CHALLENGE_FEE, seed
    )

    res_mean   = float(np.mean(results))
    res_median = float(np.median(results))
    res_p5     = float(np.percentile(results, 5))
    res_win    = float(np.mean(results > 0) * 100)

    del results
    gc.collect()

    return {
        "mean": res_mean,
        "median": res_median,
        "p5": res_p5,
        "win_prob": res_win,
    }


# ==============================================================================
# 07. GRID SEARCH OPTIMIZER
# ==============================================================================
def generate_param_combos(grid: dict) -> list[dict]:
    """Generates valid parameter combinations (ema_fast < ema_slow)."""
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, values))
        if p["ema_fast"] < p["ema_slow"]:
            combos.append(p)
    return combos


def run_grid_search(execution_df: pd.DataFrame) -> pd.DataFrame:
    """Executes full multi-variable grid search and Monte Carlo validation."""
    combos = generate_param_combos(PARAM_GRID)
    total = len(combos)
    print(f"\n[GRID] Start grid search: {total} geldige combinaties...")

    rows = []
    t0 = time.time()

    for i, params in enumerate(combos, 1):
        signals = create_m5_signals(execution_df, **params)
        pf = run_realistic_execution(execution_df, signals)
        trades = process_trade_history(pf)
        metrics = evaluate_prop(trades)

        rows.append({
            **params,
            "n_trades": metrics["n_trades"],
            "payouts": metrics["payouts"],
            "busts": metrics["busts"],
            "net_profit": metrics["net_profit"],
            "total_swap": metrics["total_swap"],
            "_pnl": metrics["pnl_array"],
        })

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"  [{i}/{total}] {elapsed:.1f}s – laatste net_profit: ${metrics['net_profit']:,.0f}")

    df = pd.DataFrame(rows)
    df = df.sort_values("net_profit", ascending=False).reset_index(drop=True)

    print(f"\n[GRID] Monte Carlo simulatie op top-{MC_TOP_N} kandidaten...")
    mc_means, mc_medians, mc_p5s, mc_wins = [], [], [], []

    for idx, row in df.iterrows():
        if idx < MC_TOP_N and len(row["_pnl"]) >= MIN_TRADES:
            mc = run_monte_carlo(row["_pnl"])
            mc_means.append(mc["mean"])
            mc_medians.append(mc["median"])
            mc_p5s.append(mc["p5"])
            mc_wins.append(mc["win_prob"])
        else:
            mc_means.append(np.nan)
            mc_medians.append(np.nan)
            mc_p5s.append(np.nan)
            mc_wins.append(np.nan)

    df["mc_mean"]   = mc_means
    df["mc_median"] = mc_medians
    df["mc_p5"]     = mc_p5s
    df["mc_win%"]   = mc_wins

    df = df.drop(columns=["_pnl"])
    gc.collect()

    return df


def print_results(df: pd.DataFrame, top: int = 10):
    print("\n" + "=" * 78)
    print(f"TOP {top} PARAMETER COMBINATIES (gesorteerd op single-run net profit)")
    print("=" * 78)

    display_cols = [
        "ema_fast", "ema_slow", "rsi_len", "rsi_entry",
        "n_trades", "payouts", "busts", "net_profit", "total_swap",
        "mc_mean", "mc_median", "mc_p5", "mc_win%"
    ]
    show = df[display_cols].head(top).copy()
    show["net_profit"] = show["net_profit"].map(lambda x: f"${x:,.0f}")
    show["total_swap"] = show["total_swap"].map(lambda x: f"${x:,.1f}")
    show["mc_mean"]    = show["mc_mean"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
    show["mc_median"]  = show["mc_median"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
    show["mc_p5"]      = show["mc_p5"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
    show["mc_win%"]    = show["mc_win%"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")

    print(show.to_string(index=True))
    print("=" * 78)

    best = df.iloc[0]
    print(f"\nBESTE SINGLE-RUN: EMA {int(best.ema_fast)}/{int(best.ema_slow)} | "
          f"RSI({int(best.rsi_len)}) > {best.rsi_entry} → "
          f"Net ${best.net_profit:,.0f} | Trades {int(best.n_trades)}")

    mc_valid = df.dropna(subset=["mc_mean"])
    if not mc_valid.empty:
        best_mc = mc_valid.sort_values("mc_mean", ascending=False).iloc[0]
        print(f"BESTE MC-MEAN  : EMA {int(best_mc.ema_fast)}/{int(best_mc.ema_slow)} | "
              f"RSI({int(best_mc.rsi_len)}) > {best_mc.rsi_entry} → "
              f"MC-mean ${best_mc.mc_mean:,.0f} | Win% {best_mc['mc_win%']:.1f}%")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    try:
        execution_df = load_market_data(PRIMARY_TICK_FILE)
        results = run_grid_search(execution_df)
        print_results(results, top=10)
    except Exception as e:
        print(f"Fout tijdens uitvoering: {e}")
        raise
