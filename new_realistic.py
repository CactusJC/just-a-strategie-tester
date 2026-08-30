"""
================================================================================
XAUUSD - PROP FIRM SIMULATOR + GRID SEARCH OPTIMIZER
================================================================================
Zoekt automatisch de beste EMA-lengtes en RSI-instellingen.
================================================================================
"""

import time
import itertools
import pandas as pd
import numpy as np
import pandas_ta as ta
import vectorbt as vbt
from numba import njit

# ==============================================================================
# CONFIGURATIE
# ==============================================================================
TICK_FILE = r"C:\workbase\cache\XAUUSD_202608140100_202608282359_tick.csv"

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
# Pas deze ranges aan naar wens. Meer combinaties = langer rekenen.
PARAM_GRID = {
    "ema_fast":   [10, 15, 20, 25],
    "ema_slow":   [30, 40, 50, 60],
    "rsi_len":    [10, 14, 21],
    "rsi_entry":  [50, 55, 60],          # RSI > deze waarde voor entry
}

# Minimum aantal trades om een combo serieus te nemen
MIN_TRADES = 10

# Monte Carlo alleen op de top-N kandidaten (scheelt enorm veel tijd)
MC_TOP_N       = 5
MC_ITERATIONS  = 1000
MC_SEED        = 42

# ==============================================================================
# 01. LOAD TICKS
# ==============================================================================
def load_ticks(filepath: str) -> pd.DataFrame:
    print("\n[01] Ticks inladen...")
    df = pd.read_csv(filepath, sep="\t")
    df.columns = [c.strip("<>") for c in df.columns]
    df["time"] = pd.to_datetime(
        df["DATE"] + " " + df["TIME"],
        format="%Y.%m.%d %H:%M:%S.%f",
        errors="coerce"
    )
    return df.dropna(subset=["time"]).set_index("time").sort_index()

# ==============================================================================
# 02. CREATE M1 EXECUTION
# ==============================================================================
def create_m1_execution(tick_df: pd.DataFrame) -> pd.DataFrame:
    print("[02] M1 Execution dataframe bouwen...")
    m1 = tick_df.resample("1min").agg({
        "BID": ["first", "max", "min", "last"],
        "ASK": ["first", "max", "min", "last"]
    }).dropna()
    m1.columns = [
        "Bid_O", "Bid_H", "Bid_L", "Bid_C",
        "Ask_O", "Ask_H", "Ask_L", "Ask_C"
    ]
    return m1

# ==============================================================================
# 03. CREATE M5 SIGNALS (parametriseerbaar + time filters)
# ==============================================================================
def create_m5_signals(
    m1_df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 40,
    rsi_len: int = 14,
    rsi_entry: float = 55.0,
) -> pd.DataFrame:
    """Genereert Entry/Exit signalen met anti-look-ahead en time filters."""

    m5 = m1_df["Bid_C"].resample("5min").last().to_frame(name="Close")
    m5["EMA_F"] = ta.ema(m5["Close"], length=ema_fast)
    m5["EMA_S"] = ta.ema(m5["Close"], length=ema_slow)
    m5["RSI"]   = ta.rsi(m5["Close"], length=rsi_len)

    m5_entries = (m5["EMA_F"] > m5["EMA_S"]) & (m5["RSI"] > rsi_entry)
    m5_exits   = (m5["EMA_F"] < m5["EMA_S"])

    m5_entries_safe = m5_entries.shift(1).fillna(False)
    m5_exits_safe   = m5_exits.shift(1).fillna(False)

    raw_entries = m5_entries_safe.reindex(m1_df.index, method="ffill").fillna(False)
    raw_exits   = m5_exits_safe.reindex(m1_df.index, method="ffill").fillna(False)

    # Time filters
    mins_from_midnight = m1_df.index.hour * 60 + m1_df.index.minute
    eod_mins = EOD_HOUR * 60 + EOD_MINUTE

    is_eod       = (m1_df.index.hour == EOD_HOUR) & (m1_df.index.minute == EOD_MINUTE)
    is_forbidden = (mins_from_midnight >= eod_mins)
    is_friday    = (m1_df.index.dayofweek == 4)
    is_weekend   = (m1_df.index.dayofweek > 4)

    forced_exits  = pd.Series(False, index=m1_df.index)
    block_entries = pd.Series(False, index=m1_df.index)

    if NO_OVERNIGHT:
        forced_exits  = forced_exits | is_eod
        block_entries = block_entries | is_forbidden
    elif NO_WEEKEND:
        forced_exits  = forced_exits | (is_eod & is_friday)
        block_entries = block_entries | (is_forbidden & is_friday) | is_weekend

    signals = pd.DataFrame(index=m1_df.index)
    signals["Exit"]  = raw_exits | forced_exits
    signals["Entry"] = raw_entries & ~block_entries
    signals["Entry"] = signals["Entry"] & ~signals["Entry"].shift(1).fillna(False)

    return signals

# ==============================================================================
# 04-06. REALISTIC EXECUTION
# ==============================================================================
def run_realistic_execution(m1_df: pd.DataFrame, signals: pd.DataFrame) -> vbt.Portfolio:
    half_spread_pct = ((m1_df["Ask_O"] - m1_df["Bid_O"]) / 2) / m1_df["Bid_O"]
    half_comm_pct   = (COMMISSION_PER_LOT / 2) / (CONTRACT_SIZE * m1_df["Bid_O"])
    total_friction  = half_spread_pct + half_comm_pct

    return vbt.Portfolio.from_signals(
        close=m1_df["Bid_O"],
        entries=signals["Entry"],
        exits=signals["Exit"],
        size=FIXED_LOT_SIZE * CONTRACT_SIZE,
        size_type="amount",
        fees=total_friction,
        init_cash=INITIAL_BALANCE,
        freq="1min",
    )

# ==============================================================================
# 07. SWAP ENGINE
# ==============================================================================
def calculate_accurate_swap(
    entry_dates, exit_dates, is_long,
    swap_long=SWAP_LONG, swap_short=SWAP_SHORT,
    triple_day=TRIPLE_SWAP_DAY, exclude_weekend=True
) -> np.ndarray:
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
# 08. PROP SIMULATOR (Numba)
# ==============================================================================
@njit
def sim_prop_firm(pnl_array, day_ids, initial_bal, max_dd_pct, daily_dd_pct, target_pct, fee):
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
    """Single-run prop evaluatie → dict met metrics."""
    if trades.empty or len(trades) < MIN_TRADES:
        return {
            "n_trades": len(trades),
            "payouts": 0,
            "busts": 0,
            "net_profit": -1e9,
            "total_swap": 0.0,
            "pnl_array": np.array([]),
            "day_ids": np.array([]),
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
# 09. MONTE CARLO (alleen voor top kandidaten)
# ==============================================================================
def run_monte_carlo(pnl_array: np.ndarray, iterations: int = MC_ITERATIONS, seed: int = MC_SEED) -> dict:
    if len(pnl_array) == 0:
        return {"mean": -1e9, "median": -1e9, "p5": -1e9, "win_prob": 0.0}

    rng = np.random.default_rng(seed)
    n = len(pnl_array)
    trades_per_day = max(1, n // max(1, n // 3))
    fake_days = np.arange(n) // trades_per_day

    results = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        shuffled = rng.choice(pnl_array, size=n, replace=True)
        _, val, _, fees = sim_prop_firm(
            shuffled, fake_days,
            INITIAL_BALANCE, MAX_DRAWDOWN_PCT, DAILY_DRAWDOWN_PCT,
            PAYOUT_TARGET_PCT, CHALLENGE_FEE
        )
        results[i] = val - fees

    return {
        "mean": float(results.mean()),
        "median": float(np.median(results)),
        "p5": float(np.percentile(results, 5)),
        "win_prob": float(np.mean(results > 0) * 100),
    }

# ==============================================================================
# 10. GRID SEARCH OPTIMIZER
# ==============================================================================
def generate_param_combos(grid: dict) -> list[dict]:
    """Alle geldige combinaties (ema_fast < ema_slow)."""
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, values))
        if p["ema_fast"] < p["ema_slow"]:
            combos.append(p)
    return combos

def run_grid_search(m1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Volledige grid search.
    1) Evalueert elke combo op single-run prop net profit
    2) Draait Monte Carlo alleen op de top-N
    """
    combos = generate_param_combos(PARAM_GRID)
    total = len(combos)
    print(f"\n[GRID] Start grid search: {total} geldige combinaties...")

    rows = []
    t0 = time.time()

    for i, params in enumerate(combos, 1):
        signals = create_m5_signals(m1_df, **params)
        pf = run_realistic_execution(m1_df, signals)
        trades = process_trade_history(pf)
        metrics = evaluate_prop(trades)

        rows.append({
            **params,
            "n_trades": metrics["n_trades"],
            "payouts": metrics["payouts"],
            "busts": metrics["busts"],
            "net_profit": metrics["net_profit"],
            "total_swap": metrics["total_swap"],
            "_pnl": metrics["pnl_array"],   # intern voor latere MC
        })

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"  [{i}/{total}] {elapsed:.1f}s – laatste net_profit: ${metrics['net_profit']:,.0f}")

    df = pd.DataFrame(rows)

    # Sorteer op single-run net profit
    df = df.sort_values("net_profit", ascending=False).reset_index(drop=True)

    # Monte Carlo op top-N
    print(f"\n[GRID] Monte Carlo op top-{MC_TOP_N} kandidaten...")
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

    # Opruimen interne kolom
    df = df.drop(columns=["_pnl"])

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

    # Beste op MC mean (als beschikbaar)
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
        tick_df = load_ticks(TICK_FILE)
        m1_df   = create_m1_execution(tick_df)

        results = run_grid_search(m1_df)
        print_results(results, top=10)

        # Optioneel: bewaar volledige ranking
        # results.to_csv("grid_search_results.csv", index=False)
        # print("\nResultaten opgeslagen in grid_search_results.csv")

    except Exception as e:
        print(f"Fout tijdens uitvoering: {e}")
        raise
