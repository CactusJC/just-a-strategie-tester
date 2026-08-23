import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import itertools
from datetime import datetime, timezone, timedelta

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    mt5 = None
    HAS_MT5 = False

import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
from numba import njit, set_num_threads

set_num_threads(1)

# ==============================================================================
# 0. CONFIGURATIE (AI-FACTORY / VANGUARD CFD $5.000)
# ==============================================================================

SYMBOLS = ["MSFT", "EURUSD", "XAUUSD", "BTCUSD.nx", "USOIL.c", "NACUSD.c"]

# Standaard backtestperiode: de laatste 6 maanden t/m nu.
BACKTEST_MONTHS = 6
TIMEFRAME = mt5.TIMEFRAME_M5 if (HAS_MT5 and mt5 is not None) else 5

# Map waarin cache-bestanden (.parquet / .csv) per symbool worden weggeschreven.
CACHE_DIR = "cache"

# --- Vanguard CFD $5.000 accountstructuur: prop firm risicoregels -----------
INITIAL_CAPITAL = 5000.0
CHALLENGE_FEE = 33.50  # eenmalige challenge-kosten om de challenge te starten

DAILY_DD_PCT = 4.0             # Daily Drawdown: 4% t.o.v. saldo bij opening dag
MAX_OVERALL_LOSS_PCT = 7.0       # Maximum Overall Loss: 7% t.o.v. startkapitaal
MAX_SINGLE_TRADE_LOSS_PCT = 2.0  # Max verlies op één enkele trade: 2%

# --- Payout-eisen -------------------------------------------------------------
MIN_PROFIT_PCT_FOR_PAYOUT = 1.0   # Minimaal 1% winst t.o.v. de baseline
MIN_PROFITABLE_DAYS = 6           # Minimaal 6 winstgevende handelsdagen
MIN_DAILY_PROFIT_PCT = 0.5        # Elk met minimaal 0.5% winst
BEST_DAY_MAX_SHARE_OF_PROFIT = 0.20  # Best Day Rule (soft): max 20% per dag

PAYOUT_SCHEDULE = [100.0, 100.0, 150.0, 200.0, 250.0, 315.0, 375.0]

# --- VANGUARD PROHIBITED BEHAVIOR RULES (GEHANDHAAFD IN ENGINE) --------------
# 1. Geen Grid Trading (geen order-grids rond prijsniveaus)
# 2. Geen DCA / Averaging Out Losses (strikt 1 positie tegelijk, geen layering)
# 3. Geen Tick Scalping of High-Frequency Trading (HFT) (minimaal M5 candles + ATR TP/SL)
# 4. Geen Latency Arbitrage / Hedging


# ==============================================================================
# 1. NUMBA TRADE MANAGEMENT ENGINE (Razendsnelle C-compiler)
# ==============================================================================
@njit
def process_trades_numba(
    close, high, low, raw_entries, sl_points_arr, tp_points_arr
):
    """
    Verwerkt per kolom de regels voor Stop-Loss (SL) en Take-Profit (TP).
    Handhaaft strikt 'Single Position Execution': er kan maximaal EEN positie tegelijk
    openstaan. Geen layering, geen DCA / loss averaging en geen grid orders.
    """
    n_bars, n_combos = raw_entries.shape
    entries = np.zeros((n_bars, n_combos), dtype=np.bool_)
    exits = np.zeros((n_bars, n_combos), dtype=np.bool_)

    for col in range(n_combos):
        sl = sl_points_arr[col]
        tp = tp_points_arr[col]

        in_position = False
        entry_price = 0.0

        for i in range(n_bars):
            if in_position:
                current_high = high[i]
                current_low = low[i]

                if current_low <= (entry_price - sl):
                    exits[i, col] = True
                    in_position = False
                elif current_high >= (entry_price + tp):
                    exits[i, col] = True
                    in_position = False
            else:
                if raw_entries[i, col]:
                    entries[i, col] = True
                    in_position = True
                    entry_price = close[i]

    return entries, exits


# ==============================================================================
# 2. METATRADER 5 DATA HELPER (Specifieke Datum Reeks)
# ==============================================================================
def fetch_mt5_data_by_dates(
    symbol: str, timeframe, date_from: datetime, date_to: datetime
) -> pd.DataFrame:
    """Haalt MT5 historie op tussen twee exacte datums."""
    if not HAS_MT5 or mt5 is None:
        raise RuntimeError("MetaTrader5 package is niet geïnstalleerd of niet beschikbaar in deze omgeving.")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialisatie mislukt. Foutcode: {mt5.last_error()}")

    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise ValueError(f"Symbool '{symbol}' niet gevonden in MT5 Market Watch.")

    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.shutdown()
        raise ValueError(f"Kan symboolinfo voor '{symbol}' niet ophalen.")

    point_size = info.point

    print(
        f" -> MT5 Historie downloaden van {date_from.strftime('%Y-%m-%d')} t/m "
        f"{date_to.strftime('%Y-%m-%d')}..."
    )

    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(
            f"Geen data ontvangen vanuit MT5 voor {symbol} in de opgegeven periode."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        },
        inplace=True,
    )

    df["Spread_Price"] = df["spread"] * point_size
    return df


# ==============================================================================
# 3. CACHING MECHANISME (.parquet met .csv fallback, per symbool)
# ==============================================================================
def _cache_path(symbol: str, ext: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_6m_cache.{ext}")


def load_cached_data(symbol: str):
    """Zoekt een lokaal cache-bestand voor dit symbool."""
    parquet_path = _cache_path(symbol, "parquet")
    csv_path = _cache_path(symbol, "csv")

    if os.path.exists(parquet_path):
        print(f" -> Cache gevonden (parquet): {parquet_path} -> MT5-download overgeslagen.")
        return pd.read_parquet(parquet_path)

    if os.path.exists(csv_path):
        print(f" -> Cache gevonden (csv): {csv_path} -> MT5-download overgeslagen.")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    return None


def save_cached_data(symbol: str, df: pd.DataFrame):
    """Slaat data lokaal op."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    parquet_path = _cache_path(symbol, "parquet")
    try:
        df.to_parquet(parquet_path)
        print(f" -> Data gecached als parquet: {parquet_path}")
    except Exception:
        csv_path = _cache_path(symbol, "csv")
        df.to_csv(csv_path)
        print(f" -> Gecached als csv: {csv_path}")


# ==============================================================================
# 4. PROP FIRM RISICOREGELS & PAYOUT-LOGICA
# ==============================================================================
def compute_daily_stats(equity: pd.Series) -> pd.DataFrame:
    equity = equity.sort_index()
    daily = equity.resample("1D").agg(["first", "last", "min"]).dropna()
    daily.columns = ["day_open", "day_close", "day_low"]

    daily["dd_abs"] = daily["day_open"] - daily["day_low"]
    daily["dd_pct"] = np.where(daily["day_open"] != 0, daily["dd_abs"] / daily["day_open"] * 100.0, 0.0)

    daily["profit_abs"] = daily["day_close"] - daily["day_open"]
    daily["profit_pct"] = np.where(daily["day_open"] != 0, daily["profit_abs"] / daily["day_open"] * 100.0, 0.0)

    return daily


def get_payout_amount(payout_number: int, cycle_profit: float) -> float:
    idx = payout_number - 1
    if idx < len(PAYOUT_SCHEDULE):
        return PAYOUT_SCHEDULE[idx]
    return max(cycle_profit, 0.0)


def evaluate_prop_firm_rules(daily_stats: pd.DataFrame, equity: pd.Series, trades_pnl: pd.Series, initial_capital: float) -> dict:
    equity = equity.sort_index()

    daily_violations = daily_stats[daily_stats["dd_pct"] >= DAILY_DD_PCT]

    exact_daily_violation_times = []
    for day_dt in daily_violations.index:
        day_equity = equity[equity.index.date == day_dt.date()]
        if not day_equity.empty:
            day_open = day_equity.iloc[0]
            breach = day_equity[(day_open - day_equity) / day_open * 100.0 >= DAILY_DD_PCT]
            if not breach.empty:
                exact_daily_violation_times.append(pd.Timestamp(breach.index[0]))
            else:
                exact_daily_violation_times.append(pd.Timestamp(day_dt))

    overall_violation_dates = equity.index[(initial_capital - equity) / initial_capital * 100.0 >= MAX_OVERALL_LOSS_PCT]

    if trades_pnl is not None and not trades_pnl.empty:
        single_trade_violations = trades_pnl[trades_pnl <= -(MAX_SINGLE_TRADE_LOSS_PCT / 100.0) * initial_capital]
    else:
        single_trade_violations = pd.Series(dtype=float)

    blown_candidates = []
    if len(exact_daily_violation_times) > 0:
        blown_candidates.append(exact_daily_violation_times[0])
    if len(overall_violation_dates) > 0:
        blown_candidates.append(pd.Timestamp(overall_violation_dates[0]))
    if len(single_trade_violations) > 0:
        first_bad_trade_date = single_trade_violations.index[0]
        try:
            blown_candidates.append(pd.Timestamp(first_bad_trade_date))
        except Exception:
            blown_candidates.append(pd.Timestamp(equity.index[0]))

    account_blown = len(blown_candidates) > 0
    blown_date = min(blown_candidates) if blown_candidates else None

    return {
        "daily_violations": daily_violations,
        "overall_violation_dates": overall_violation_dates,
        "single_trade_violations": single_trade_violations,
        "account_blown": account_blown,
        "blown_date": blown_date,
    }


def simulate_payouts(equity: pd.Series, position_mask: pd.Series, daily_stats: pd.DataFrame,
                     initial_capital: float, blown_date) -> list:
    equity = equity.sort_index()
    if blown_date is not None:
        equity = equity[equity.index < blown_date]
        daily_stats = daily_stats[daily_stats.index < blown_date]

    if equity.empty or daily_stats.empty:
        return []

    days = daily_stats.index
    payouts = []
    baseline_balance = initial_capital
    day_cursor = 0
    payout_number = 0

    while day_cursor < len(days):
        qualifying_days = []
        request_day_pos = None

        for i in range(day_cursor, len(days)):
            day = days[i]
            if daily_stats.loc[day, "profit_pct"] >= MIN_DAILY_PROFIT_PCT:
                qualifying_days.append(day)

            day_close_balance = daily_stats.loc[day, "day_close"]
            total_profit_pct = (
                (day_close_balance - baseline_balance) / baseline_balance * 100.0
                if baseline_balance else 0.0
            )

            if len(qualifying_days) >= MIN_PROFITABLE_DAYS and total_profit_pct >= MIN_PROFIT_PCT_FOR_PAYOUT:
                request_day_pos = i
                break

        if request_day_pos is None:
            break

        while True:
            window_days = days[day_cursor:request_day_pos + 1]
            window_profit_total = daily_stats.loc[window_days[-1], "day_close"] - baseline_balance
            best_day_profit = daily_stats.loc[window_days, "profit_abs"].max()

            rule_ok = (
                window_profit_total <= 0
                or best_day_profit <= BEST_DAY_MAX_SHARE_OF_PROFIT * window_profit_total
            )
            if rule_ok or request_day_pos + 1 >= len(days):
                break
            request_day_pos += 1

        request_day = days[request_day_pos]

        candidate_times = equity.index[equity.index >= request_day]
        flat_time = None
        for t in candidate_times:
            if not bool(position_mask.get(t, False)):
                flat_time = t
                break
        if flat_time is None:
            break

        request_time = flat_time
        cycle_profit = equity.loc[request_time] - baseline_balance
        if cycle_profit <= 0:
            break

        payout_number += 1
        amount = get_payout_amount(payout_number, cycle_profit)

        payouts.append({
            "payout_number": payout_number,
            "request_time": request_time,
            "cycle_start": days[day_cursor],
            "profitable_days": len(qualifying_days),
            "cycle_profit": cycle_profit,
            "amount": amount,
        })

        baseline_balance = equity.loc[request_time]
        day_cursor = request_day_pos + 1

    return payouts


def print_payout_summary(symbol: str, rule_eval: dict, payouts: list, challenge_fee: float):
    print("\n" + "=" * 70)
    print(f" PAYOUT SUMMARY - {symbol}")
    print("=" * 70)
    print(" ✔ Handhaving regels: Single Position (Geen Grid/DCA), M5 Trend (Geen Scalping/HFT/Arbitrage)")
    if len(rule_eval["daily_violations"]) > 0:
        print(f" ⚠ Daily Drawdown (4%) overtreden op {len(rule_eval['daily_violations'])} dag(en).")
    if len(rule_eval["overall_violation_dates"]) > 0:
        print(f" ⚠ Maximum Overall Loss (7%) overtreden vanaf {rule_eval['overall_violation_dates'][0]}.")
    if len(rule_eval["single_trade_violations"]) > 0:
        print(f" ⚠ Max Single Trade Loss (2%) overtreden op {len(rule_eval['single_trade_violations'])} trade(s).")
    if rule_eval["account_blown"]:
        print(f" -> Account geblazen op {rule_eval['blown_date']}.")

    total_payout = 0.0
    if not payouts:
        print(" -> Geen enkele payout-cyclus voltooid binnen deze periode.")
    else:
        for p in payouts:
            print(f"    Payout #{p['payout_number']:>2} aangevraagd op {p['request_time']} | uitbetaling: ${p['amount']:.2f}")
        total_payout = sum(p["amount"] for p in payouts)
        print(f" -> Totaal uitbetaald: ${total_payout:.2f}")

    net_result = total_payout - challenge_fee
    print(f" -> NETTO RESULTAAT: ${net_result:.2f}")
    print("=" * 70)


# ==============================================================================
# 5. HOOFDSCRIPT
# ==============================================================================
def main():
    print("\n" + "=" * 70)
    print("  MT5 DEEP SCANNER - AUTOMATIC LOOP & VANGUARD RULES")
    print("=" * 70 + "\n")

    # Automatische datum-reeks bepaling (laatste 6 maanden t/m nu)
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=BACKTEST_MONTHS * 30)

    for SYMBOL in SYMBOLS:
        print(f"\n[1/5] Data ophalen voor {SYMBOL}...")

        df = load_cached_data(SYMBOL)
        if df is None:
            try:
                df = fetch_mt5_data_by_dates(SYMBOL, TIMEFRAME, date_from, date_to)
            except Exception as e:
                print(f"Fout tijdens ophalen data voor {SYMBOL}: {e}")
                continue
            save_cached_data(SYMBOL, df)

        print(f" -> Succes! {len(df):,} candles ingeladen.")

        close_arr = df["Close"].values.astype(np.float64)
        high_arr = df["High"].values.astype(np.float64)
        low_arr = df["Low"].values.astype(np.float64)

        print("\n[2/5] Technische indicatoren berekenen...")
        rsi_10_arr = ta.rsi(df["Close"], length=10).fillna(50).values
        rsi_14_arr = ta.rsi(df["Close"], length=14).fillna(50).values

        ema9 = ta.ema(df["Close"], length=9)
        ema21 = ta.ema(df["Close"], length=21)
        cond_ema_9_21 = (ema9 > ema21).values

        ema12 = ta.ema(df["Close"], length=12)
        ema26 = ta.ema(df["Close"], length=26)
        cond_ema_12_26 = (ema12 > ema26).values

        ema20 = ta.ema(df["Close"], length=20)
        ema50 = ta.ema(df["Close"], length=50)
        cond_ema_20_50 = (ema20 > ema50).values

        cond_trend_50 = (df["Close"] > ema50).values
        cond_trend_100 = (df["Close"] > ta.ema(df["Close"], length=100)).values
        cond_trend_200 = (df["Close"] > ta.ema(df["Close"], length=200)).values

        macd = ta.macd(df["Close"])
        cond_macd_arr = (macd["MACDh_12_26_9"] > 0 if "MACDh_12_26_9" in macd.columns else macd.iloc[:, 1] > 0).values

        st = ta.supertrend(df["High"], df["Low"], df["Close"])
        st_col = [c for c in st.columns if c.startswith("SUPERTd")][0]
        cond_st_arr = (st[st_col] == 1).values

        stoch = ta.stoch(df["High"], df["Low"], df["Close"])
        cond_stoch_arr = ((stoch.iloc[:, 0] > stoch.iloc[:, 1]) & (stoch.iloc[:, 0] < 80)).values

        print("\n[3/5] Parameter Grid opbouwen...")
        rsi_len_opts = [10, 14]
        rsi_thresh_opts = [30, 40, 50]
        ema_pair_opts = ["9/21", "12/26", "20/50"]
        ema_trend_opts = [50, 100, 200]
        min_agree_opts = [4, 5, 6]

        last_close = close_arr[-1]
        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr_val = atr_series.dropna().iloc[-1] if atr_series is not None and not atr_series.dropna().empty else last_close * 0.01
        dec_places = 4 if last_close < 10 else 2

        sl_points_opts = [float(np.round(m * atr_val, dec_places)) for m in [0.8, 1.5, 2.5, 4.0]]
        tp_points_opts = [float(np.round(m * atr_val, dec_places)) for m in [1.5, 3.0, 5.0, 8.0]]

        param_grid = list(itertools.product(rsi_len_opts, rsi_thresh_opts, ema_pair_opts, ema_trend_opts, min_agree_opts, sl_points_opts, tp_points_opts))

        print("\n[4/5] Signalen genereren...")
        n_bars = len(df)
        n_combos = len(param_grid)
        raw_entries_matrix = np.zeros((n_bars, n_combos), dtype=np.bool_)
        sl_arr = np.zeros(n_combos, dtype=np.float64)
        tp_arr = np.zeros(n_combos, dtype=np.float64)

        for i, (rsi_len, rsi_val, ema_pair, trend_len, min_agree, sl_pts, tp_pts) in enumerate(param_grid):
            rsi_arr = rsi_10_arr if rsi_len == 10 else rsi_14_arr
            c_rsi = rsi_arr < rsi_val
            c_ema = cond_ema_9_21 if ema_pair == "9/21" else (cond_ema_12_26 if ema_pair == "12/26" else cond_ema_20_50)
            c_trend = cond_trend_50 if trend_len == 50 else (cond_trend_100 if trend_len == 100 else cond_trend_200)

            agreement_score = c_rsi.astype(int) + c_ema.astype(int) + cond_macd_arr.astype(int) + cond_st_arr.astype(int) + cond_stoch_arr.astype(int) + c_trend.astype(int)
            raw_entries_matrix[:, i] = agreement_score >= min_agree
            sl_arr[i] = sl_pts
            tp_arr[i] = tp_pts

        print("\n[5/5] Vectorbt Simulation in Batches...")
        BATCH_SIZE = 1000
        spread_fee_series = df["Spread_Price"] / df["Close"]
        all_summary_list = []
        total_batches = (n_combos + BATCH_SIZE - 1) // BATCH_SIZE

        for b in range(total_batches):
            start_idx = b * BATCH_SIZE
            end_idx = min((b + 1) * BATCH_SIZE, n_combos)

            chunk_grid = param_grid[start_idx:end_idx]
            chunk_raw_entries = raw_entries_matrix[:, start_idx:end_idx]
            chunk_sl = sl_arr[start_idx:end_idx]
            chunk_tp = tp_arr[start_idx:end_idx]

            chunk_entries, chunk_exits = process_trades_numba(close_arr, high_arr, low_arr, chunk_raw_entries, chunk_sl, chunk_tp)
            chunk_index = pd.MultiIndex.from_tuples(chunk_grid, names=["rsi_len", "rsi_max", "ema_pair", "ema_trend", "min_agree", "sl_points", "tp_points"])

            pf_chunk = vbt.Portfolio.from_signals(
                close=df["Close"],
                entries=pd.DataFrame(chunk_entries, index=df.index, columns=chunk_index),
                exits=pd.DataFrame(chunk_exits, index=df.index, columns=chunk_index),
                freq="5m",
                fees=spread_fee_series,
                init_cash=INITIAL_CAPITAL,
            )

            pf_trades = pf_chunk.trades

            try:
                pf_win_rate = (pf_trades.win_rate() * 100.0).round(2)
            except Exception:
                pf_win_rate = pd.Series(0.0, index=chunk_index)

            try:
                pf_pf = pf_trades.profit_factor()
                if isinstance(pf_pf, pd.Series):
                    pf_pf = pf_pf.fillna(0.0).round(2)
                else:
                    pf_pf = np.where(np.isnan(pf_pf), 0.0, pf_pf).round(2)
            except Exception:
                pf_pf = pd.Series(0.0, index=chunk_index)

            res_chunk = pd.DataFrame({
                "Return (%)": (pf_chunk.total_return() * 100.0).round(2),
                "Win Rate (%)": pf_win_rate,
                "Total Trades": pf_trades.count(),
                "Max Drawdown (%)": (pf_chunk.max_drawdown() * 100.0).round(2),
                "Profit Factor": pf_pf,
            })

            all_summary_list.append(res_chunk)
            del pf_chunk
            gc.collect()

        results_df = pd.concat(all_summary_list)
        results_df = results_df[results_df["Total Trades"] > 0]

        if results_df.empty:
            print(f"\n -> Geen trades voor {SYMBOL}. Volgende...\n")
            continue

        rule_compliant = results_df[(results_df["Max Drawdown (%)"].abs() < MAX_OVERALL_LOSS_PCT) & (results_df["Return (%)"] > 0) & (results_df["Total Trades"] >= 5)]

        candidates = rule_compliant.sort_values(by="Return (%)", ascending=False).head(15) if not rule_compliant.empty else results_df.sort_values(by="Return (%)", ascending=False).head(15)

        best_combo = candidates.index[0]
        best_payout_total = -1.0
        best_rule_eval = None
        best_payouts = None

        for candidate_combo in candidates.index:
            c_idx = param_grid.index(candidate_combo)
            c_entries, c_exits = process_trades_numba(
                close_arr, high_arr, low_arr,
                raw_entries_matrix[:, c_idx:c_idx+1],
                sl_arr[c_idx:c_idx+1],
                tp_arr[c_idx:c_idx+1]
            )

            c_pf = vbt.Portfolio.from_signals(
                close=df["Close"],
                entries=pd.DataFrame(c_entries, index=df.index, columns=[candidate_combo]),
                exits=pd.DataFrame(c_exits, index=df.index, columns=[candidate_combo]),
                freq="5m",
                fees=spread_fee_series,
                init_cash=INITIAL_CAPITAL,
            )

            c_equity = c_pf.value()
            if isinstance(c_equity, pd.DataFrame):
                c_equity = c_equity.iloc[:, 0]

            c_trades_readable = c_pf.trades.records_readable
            if not c_trades_readable.empty and "Exit Timestamp" in c_trades_readable.columns and "PnL" in c_trades_readable.columns:
                c_trades_pnl = c_trades_readable.set_index("Exit Timestamp")["PnL"]
                if not isinstance(c_trades_pnl.index, pd.DatetimeIndex):
                    c_trades_pnl.index = pd.to_datetime(c_trades_pnl.index)
            else:
                c_trades_pnl = pd.Series(dtype=float)

            if hasattr(c_pf, "shares"):
                c_pos = c_pf.shares
            elif hasattr(c_pf, "holding_shares"):
                c_pos = c_pf.holding_shares
            else:
                c_pos = pd.Series(0, index=df.index)

            if isinstance(c_pos, pd.DataFrame):
                c_pos = c_pos.iloc[:, 0]

            c_pos_mask = c_pos != 0

            c_daily = compute_daily_stats(c_equity)
            c_eval = evaluate_prop_firm_rules(c_daily, c_equity, c_trades_pnl, INITIAL_CAPITAL)
            c_payouts = simulate_payouts(c_equity, c_pos_mask, c_daily, INITIAL_CAPITAL, c_eval["blown_date"])

            c_total_payout = sum(p["amount"] for p in c_payouts) if c_payouts else 0.0

            if not c_eval["account_blown"] and c_total_payout > best_payout_total:
                best_payout_total = c_total_payout
                best_combo = candidate_combo
                best_rule_eval = c_eval
                best_payouts = c_payouts

        if best_rule_eval is None:
            c_idx = param_grid.index(best_combo)
            c_entries, c_exits = process_trades_numba(
                close_arr, high_arr, low_arr,
                raw_entries_matrix[:, c_idx:c_idx+1],
                sl_arr[c_idx:c_idx+1],
                tp_arr[c_idx:c_idx+1]
            )
            fallback_pf = vbt.Portfolio.from_signals(
                close=df["Close"],
                entries=pd.DataFrame(c_entries, index=df.index, columns=[best_combo]),
                exits=pd.DataFrame(c_exits, index=df.index, columns=[best_combo]),
                freq="5m",
                fees=spread_fee_series,
                init_cash=INITIAL_CAPITAL,
            )
            c_equity = fallback_pf.value()
            if isinstance(c_equity, pd.DataFrame):
                c_equity = c_equity.iloc[:, 0]

            c_trades_readable = fallback_pf.trades.records_readable
            if not c_trades_readable.empty and "Exit Timestamp" in c_trades_readable.columns and "PnL" in c_trades_readable.columns:
                c_trades_pnl = c_trades_readable.set_index("Exit Timestamp")["PnL"]
                if not isinstance(c_trades_pnl.index, pd.DatetimeIndex):
                    c_trades_pnl.index = pd.to_datetime(c_trades_pnl.index)
            else:
                c_trades_pnl = pd.Series(dtype=float)

            if hasattr(fallback_pf, "shares"):
                c_pos = fallback_pf.shares
            elif hasattr(fallback_pf, "holding_shares"):
                c_pos = fallback_pf.holding_shares
            else:
                c_pos = pd.Series(0, index=df.index)

            if isinstance(c_pos, pd.DataFrame):
                c_pos = c_pos.iloc[:, 0]
            c_pos_mask = c_pos != 0

            c_daily = compute_daily_stats(c_equity)
            best_rule_eval = evaluate_prop_firm_rules(c_daily, c_equity, c_trades_pnl, INITIAL_CAPITAL)
            best_payouts = simulate_payouts(c_equity, c_pos_mask, c_daily, INITIAL_CAPITAL, best_rule_eval["blown_date"])

        print(f" -> Gekozen strategie parameters: {best_combo}")
        print_payout_summary(SYMBOL, best_rule_eval, best_payouts, CHALLENGE_FEE)


if __name__ == "__main__":
    main()