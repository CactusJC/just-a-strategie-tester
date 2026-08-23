"""
================================================================================
 MT5 REALISTIC SCANNER – Prop Firm Backtest Engine
================================================================================
Doel:
  Zo dicht mogelijk nabootsen van MetaTrader 5 Strategy Tester + Upcomers
  Vanguard Instant Funding regels, zodat Python-grid resultaten bruikbaar zijn.

Belangrijkste realisme-features:
  1. Risk-based lot sizing (vast % risico per trade i.p.v. full capital)
  2. Entry op open van de VOLGENDE bar (geen lookahead / same-bar fill)
  3. Geen same-bar exit (SL/TP pas vanaf bar na entry)
  4. Slippage bij entry
  5. Breakeven-stop (SL → entry na X·R winst)
  6. ATR trailing stop (dynamische SL die alleen omhoog beweegt)
  7. Multi-cycle prop firm simulator (daily DD, overall loss, single trade loss,
     automatische resets + payout schema)

Originele Python-grid was te optimistisch → deze versie corrigeert dat.
================================================================================
"""

import os
# Beperk threads om deterministische / stabiele runs te krijgen
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

set_num_threads(1)  # Numba ook op 1 thread

# ==============================================================================
# 0. CONFIGURATIE
# ==============================================================================

# --- Symbolen & data ---
SYMBOLS = ["BTCUSD.nx"]          # Start met 1 symbool voor snelle iteratie
# SYMBOLS = ["MSFT", "EURUSD", "XAUUSD", "BTCUSD.nx", "USOIL.c", "NACUSD.c"]

BACKTEST_MONTHS = 6              # Hoeveel maanden historie ophalen
TIMEFRAME = mt5.TIMEFRAME_M5 if (HAS_MT5 and mt5 is not None) else 5
CACHE_DIR = "cache"              # Lokale cache (parquet/csv) om MT5 niet telkens te raken

# --- Account & kosten ---
INITIAL_CAPITAL = 5000.0         # Startkapitaal (Vanguard $5k)
CHALLENGE_FEE = 33.50            # Kosten per challenge / herstart

# --- Upcomers Vanguard Instant Funding regels ---
DAILY_DD_PCT = 4.0               # Max verlies per kalenderdag (equity)
MAX_OVERALL_LOSS_PCT = 5.0       # Max overall loss t.o.v. startkapitaal (trailing → lock)
MAX_SINGLE_TRADE_LOSS_PCT = 2.0  # Harde limiet: geen trade mag meer dan 2% verliezen

# Payout voorwaarden
MIN_PROFIT_PCT_FOR_PAYOUT = 1.0  # Minimale totale winst % voor payout-aanvraag
MIN_PROFITABLE_DAYS = 3          # Eerste payout: min. 3 dagen met ≥ 0.5% winst
MIN_DAILY_PROFIT_PCT = 0.5       # Drempel voor "profitabele dag"
BEST_DAY_MAX_SHARE_OF_PROFIT = 0.25  # Soft best-day rule (25%)

# Payout bedragen per payout-nummer (voor $5k account)
PAYOUT_SCHEDULE = [100.0, 100.0, 150.0, 200.0, 250.0, 315.0, 375.0]

# --- Realisme: entry & fills ---
SLIPPAGE_POINTS = 5.0            # Extra nadelige slippage bij entry (prijs-eenheden)
ENTRY_ON_NEXT_BAR = True         # True = signal bar i → entry op open van bar i+1

# --- Risk-based lot sizing ---
# lot = (capital * risk%) / (SL_distance * CONTRACT_VALUE)
USE_RISK_BASED_LOT = True        # True = risk-based, False = FIXED_LOT gebruiken
FIXED_LOT = 0.10                 # Vaste lot (alleen als USE_RISK_BASED_LOT = False)
RISK_PER_TRADE_PCT = 1.0         # Risico per trade als % van kapitaal (1.0 = 1%)
MAX_LOT = 0.50                   # Hard maximum lot (veiligheid)
MIN_LOT = 0.01                   # Minimum lot
# Voor BTCUSD.nx geldt meestal: 1.0 lot = 1 BTC → $1 prijsbeweging ≈ $1 PnL per lot
CONTRACT_VALUE = 1.0             # $ PnL per 1.0 prijs-unit per 1.0 lot

# --- Dynamische / trailing stop-loss ---
USE_TRAILING_SL = True           # True = ATR trailing stop actief
TRAIL_ATR_MULT = 2.0             # Trailing afstand = ATR × deze multiplier
TRAIL_ACTIVATION_ATR = 1.0       # Trail start pas als winst ≥ ATR × deze waarde

# --- Breakeven-stop ---
USE_BREAKEVEN = True             # True = SL naar breakeven verplaatsen
BE_TRIGGER_R = 1.0               # Na X × initiële SL-afstand winst → SL naar BE
BE_OFFSET = 0.0                  # Buffer boven entry (0 = exact BE, >0 = klein winstje)
# Volgorde per bar: 1) breakeven check  2) trailing update  3) SL/TP hit check


# ==============================================================================
# 1. NUMBA TRADE MANAGEMENT ENGINE
# ==============================================================================
@njit
def process_trades_numba_realistic(
    open_arr, high, low, close,
    atr_arr,                    # ATR-reeks (voor trailing afstand)
    raw_entries,                # Boolean signalen [bar × combo]
    sl_points_arr,              # Initiële SL-afstand per combo
    tp_points_arr,              # TP-afstand per combo
    slippage,                   # Vaste nadelige slippage bij entry
    entry_on_next_bar,          # True → entry op open van bar i+1
    use_trailing,               # ATR trailing aan/uit
    trail_atr_mult,             # Trailing = high - ATR * mult
    trail_activation_atr,       # Trail start na ATR * deze winst
    use_breakeven,              # Breakeven aan/uit
    be_trigger_r,               # BE trigger in multiples van initiële SL
    be_offset                   # Extra buffer boven entry bij BE
):
    """
    Kern van de realistische backtest-engine (Numba voor snelheid).

    Flow per bar, per parameter-combo:
      1. Geen positie → kijk of er een entry-signaal is
         - ENTRY_ON_NEXT_BAR: markeer entry op bar i+1 (open + slippage)
         - Zet initiële SL = entry - sl_points
      2. Wel positie (en niet op de entry-bar zelf):
         a. Update highest high sinds entry
         b. Breakeven-check: bij genoeg winst → SL naar entry + offset
         c. Trailing-check: bij genoeg winst → SL = high - ATR*mult (alleen omhoog)
         d. Exit als low ≤ current_sl of high ≥ entry + tp

    Belangrijk: geen same-bar exit → realistischer dan klassieke OHLC-backtests.
    """
    n_bars, n_combos = raw_entries.shape
    entries = np.zeros((n_bars, n_combos), dtype=np.bool_)
    exits = np.zeros((n_bars, n_combos), dtype=np.bool_)

    for col in range(n_combos):
        init_sl_dist = sl_points_arr[col]   # Vaste initiële risico-afstand
        tp_dist = tp_points_arr[col]

        in_position = False
        entry_price = 0.0
        entry_bar = -1
        current_sl = 0.0                    # Actieve stop-loss prijs
        highest_since_entry = 0.0
        be_done = False                     # True zodra BE eenmaal gezet is

        for i in range(n_bars):
            if in_position:
                # Geen exit-check op de entry-bar zelf (voorkomt same-bar fill)
                if i > entry_bar:
                    current_high = high[i]
                    current_low = low[i]
                    atr_now = atr_arr[i]

                    # Houd de hoogste prijs sinds entry bij (voor trail & BE)
                    if current_high > highest_since_entry:
                        highest_since_entry = current_high

                    profit = highest_since_entry - entry_price

                    # --- 1. BREAKEVEN STOP ---
                    # Verplaats SL naar entry (+ offset) zodra winst ≥ X · R
                    if use_breakeven and (not be_done) and init_sl_dist > 0.0:
                        be_trigger = init_sl_dist * be_trigger_r
                        if profit >= be_trigger:
                            be_sl = entry_price + be_offset
                            if be_sl > current_sl:      # Alleen omhoog
                                current_sl = be_sl
                            be_done = True

                    # --- 2. ATR TRAILING STOP ---
                    # Trek SL verder omhoog op basis van high - ATR*mult
                    if use_trailing and atr_now > 0.0:
                        activation_level = atr_now * trail_activation_atr
                        if profit >= activation_level:
                            new_trail_sl = highest_since_entry - (atr_now * trail_atr_mult)
                            if new_trail_sl > current_sl:  # Alleen omhoog
                                current_sl = new_trail_sl

                    # --- 3. EXIT CHECKS ---
                    hit_sl = current_low <= current_sl
                    hit_tp = current_high >= (entry_price + tp_dist)

                    if hit_sl or hit_tp:
                        exits[i, col] = True
                        in_position = False
                        # Reset state voor volgende trade
                        current_sl = 0.0
                        highest_since_entry = 0.0
                        be_done = False
            else:
                # Geen open positie → kijk naar entry-signaal
                if raw_entries[i, col]:
                    if entry_on_next_bar:
                        # Realistisch: entry pas op de OPEN van de volgende bar
                        if i + 1 < n_bars:
                            entries[i + 1, col] = True
                            in_position = True
                            entry_price = open_arr[i + 1] + slippage
                            entry_bar = i + 1
                            current_sl = entry_price - init_sl_dist
                            highest_since_entry = entry_price
                            be_done = False
                    else:
                        # Optimistische fallback (niet aanbevolen)
                        entries[i, col] = True
                        in_position = True
                        entry_price = close[i] + slippage
                        entry_bar = i
                        current_sl = entry_price - init_sl_dist
                        highest_since_entry = entry_price
                        be_done = False

    return entries, exits


# ==============================================================================
# 2. METATRADER 5 DATA HELPER & CACHING
# ==============================================================================
def fetch_mt5_data_by_dates(
    symbol: str, timeframe, date_from: datetime, date_to: datetime
) -> pd.DataFrame:
    if not HAS_MT5 or mt5 is None:
        raise RuntimeError("MetaTrader5 package is niet geïnstalleerd.")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialisatie mislukt: {mt5.last_error()}")

    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise ValueError(f"Symbool '{symbol}' niet gevonden in MT5.")

    info = mt5.symbol_info(symbol)
    point_size = info.point if info else 0.00001

    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(f"Geen data ontvangen voor {symbol}.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume"
    }, inplace=True)
    df["Spread_Price"] = df["spread"] * point_size
    return df


def _cache_path(symbol: str, ext: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_6m_cache.{ext}")


def load_cached_data(symbol: str):
    parquet_path = _cache_path(symbol, "parquet")
    csv_path = _cache_path(symbol, "csv")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return None


def save_cached_data(symbol: str, df: pd.DataFrame):
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        df.to_parquet(_cache_path(symbol, "parquet"))
    except Exception:
        df.to_csv(_cache_path(symbol, "csv"))


# ==============================================================================
# 3. MULTI-CYCLE PROP FIRM SIMULATOR
# ==============================================================================
def compute_daily_stats(equity: pd.Series) -> pd.DataFrame:
    equity = equity.sort_index()
    daily = equity.resample("1D").agg(["first", "last", "min"]).dropna()
    daily.columns = ["day_open", "day_close", "day_low"]
    daily["dd_abs"] = daily["day_open"] - daily["day_low"]
    daily["dd_pct"] = np.where(
        daily["day_open"] != 0,
        daily["dd_abs"] / daily["day_open"] * 100.0,
        0.0
    )
    daily["profit_abs"] = daily["day_close"] - daily["day_open"]
    daily["profit_pct"] = np.where(
        daily["day_open"] != 0,
        daily["profit_abs"] / daily["day_open"] * 100.0,
        0.0
    )
    return daily


def get_payout_amount(payout_number: int, cycle_profit: float) -> float:
    idx = payout_number - 1
    if idx < len(PAYOUT_SCHEDULE):
        return PAYOUT_SCHEDULE[idx]
    return max(cycle_profit, 0.0)


def calc_lot_size(sl_distance: float, capital: float = INITIAL_CAPITAL) -> float:
    """
    Bereken lot size op basis van vast risico-percentage.

    Formule:
        risk_amount = capital × (RISK_PER_TRADE_PCT / 100)
        lot         = risk_amount / (sl_distance × CONTRACT_VALUE)

    Voorbeeld ($5000, 1% risk, SL = 200, CONTRACT_VALUE = 1):
        risk_amount = 50
        lot         = 50 / 200 = 0.25

    Resultaat wordt afgerond naar beneden op 0.01 en begrensd
    tussen MIN_LOT en MAX_LOT.
    """
    if not USE_RISK_BASED_LOT or sl_distance <= 0:
        return FIXED_LOT

    risk_amount = capital * (RISK_PER_TRADE_PCT / 100.0)
    raw_lot = risk_amount / (sl_distance * CONTRACT_VALUE)

    # Afronden op 0.01 (naar beneden) en hard begrenzen
    lot = np.floor(raw_lot * 100) / 100.0
    lot = max(MIN_LOT, min(lot, MAX_LOT))
    return float(lot)


def simulate_multi_cycle_strategy(
    equity: pd.Series,
    position_mask: pd.Series,
    trades_pnl: pd.Series,
    initial_capital: float
):
    """
    Simuleert de volledige backtest-periode onder prop firm regels.

    - Loopt dag-voor-dag door de equity curve
    - Detecteert breaches: Daily DD, Overall Loss, Max Single Trade Loss
    - Bij breach → challenge "geblazen", nieuwe challenge start de volgende dag
    - Binnen elke (overlevende) cyclus worden payouts berekend volgens
      MIN_PROFITABLE_DAYS / MIN_DAILY_PROFIT_PCT / PAYOUT_SCHEDULE
    - Retourneert lijst van cycles + alle uitgekeerde payouts
    """
    equity = equity.sort_index()
    days = equity.resample("1D").first().dropna().index

    cycles = []
    current_challenge = 1
    cycle_start_time = equity.index[0]

    i = 0
    while i < len(days):
        day_dt = days[i]
        day_equity = equity[equity.index.date == day_dt.date()]
        if day_equity.empty:
            i += 1
            continue

        day_open = day_equity.iloc[0]

        # Daily Drawdown
        daily_min = day_equity.min()
        if (day_open - daily_min) / day_open * 100.0 >= DAILY_DD_PCT:
            breach_time = day_equity[
                (day_open - day_equity) / day_open * 100.0 >= DAILY_DD_PCT
            ].index[0]
            cycles.append({
                "challenge": current_challenge,
                "start": cycle_start_time,
                "blown_date": pd.Timestamp(breach_time),
                "reason": f"Daily Drawdown ({DAILY_DD_PCT}%)",
                "payouts": []
            })
            current_challenge += 1
            next_day_idx = np.where(days.date >= breach_time.date())[0]
            i = next_day_idx[0] + 1 if len(next_day_idx) > 0 else i + 1
            if i < len(days):
                cycle_start_time = days[i]
            continue

        # Overall Loss
        overall_loss_mask = (
            (initial_capital - day_equity) / initial_capital * 100.0
            >= MAX_OVERALL_LOSS_PCT
        )
        if overall_loss_mask.any():
            breach_time = day_equity[overall_loss_mask].index[0]
            cycles.append({
                "challenge": current_challenge,
                "start": cycle_start_time,
                "blown_date": pd.Timestamp(breach_time),
                "reason": f"Overall Loss ({MAX_OVERALL_LOSS_PCT}%)",
                "payouts": []
            })
            current_challenge += 1
            next_day_idx = np.where(days.date >= breach_time.date())[0]
            i = next_day_idx[0] + 1 if len(next_day_idx) > 0 else i + 1
            if i < len(days):
                cycle_start_time = days[i]
            continue

        # Single Trade Loss
        if trades_pnl is not None and not trades_pnl.empty:
            max_loss_abs = (MAX_SINGLE_TRADE_LOSS_PCT / 100.0) * initial_capital
            bad_trades = trades_pnl[
                (trades_pnl.index >= cycle_start_time) &
                (trades_pnl <= -max_loss_abs)
            ]
            if not bad_trades.empty:
                breach_time = bad_trades.index[0]
                cycles.append({
                    "challenge": current_challenge,
                    "start": cycle_start_time,
                    "blown_date": pd.Timestamp(breach_time),
                    "reason": f"Max Single Trade Loss ({MAX_SINGLE_TRADE_LOSS_PCT}%)",
                    "payouts": []
                })
                current_challenge += 1
                next_day_idx = np.where(days.date >= breach_time.date())[0]
                i = next_day_idx[0] + 1 if len(next_day_idx) > 0 else i + 1
                if i < len(days):
                    cycle_start_time = days[i]
                continue

        i += 1

    # Laatste / actieve cyclus
    if not cycles or cycles[-1]["challenge"] < current_challenge:
        cycles.append({
            "challenge": current_challenge,
            "start": cycle_start_time,
            "blown_date": None,
            "reason": "Actief / Voltooid",
            "payouts": []
        })

    # Payouts per cyclus
    total_payouts_list = []
    for cyc in cycles:
        c_start = cyc["start"]
        c_end = cyc["blown_date"] if cyc["blown_date"] is not None else equity.index[-1]

        sub_equity = equity[(equity.index >= c_start) & (equity.index <= c_end)]
        if len(sub_equity) < 2:
            continue

        sub_daily = compute_daily_stats(sub_equity)
        if sub_daily.empty:
            continue

        sub_days = sub_daily.index
        baseline_balance = initial_capital
        day_cursor = 0
        payout_count = 0

        while day_cursor < len(sub_days):
            qualifying_days = []
            request_day_pos = None

            for d_idx in range(day_cursor, len(sub_days)):
                d_val = sub_days[d_idx]
                if sub_daily.loc[d_val, "profit_pct"] >= MIN_DAILY_PROFIT_PCT:
                    qualifying_days.append(d_val)

                total_profit_pct = (
                    (sub_daily.loc[d_val, "day_close"] - baseline_balance)
                    / baseline_balance * 100.0
                ) if baseline_balance else 0.0

                if (len(qualifying_days) >= MIN_PROFITABLE_DAYS and
                        total_profit_pct >= MIN_PROFIT_PCT_FOR_PAYOUT):
                    request_day_pos = d_idx
                    break

            if request_day_pos is None:
                break

            request_day = sub_days[request_day_pos]
            candidate_times = sub_equity.index[sub_equity.index >= request_day]
            flat_time = None
            for t in candidate_times:
                if not bool(position_mask.get(t, False)):
                    flat_time = t
                    break
            if flat_time is None:
                break

            cycle_profit = sub_equity.loc[flat_time] - baseline_balance
            if cycle_profit <= 0:
                break

            payout_count += 1
            amount = get_payout_amount(payout_count, cycle_profit)

            payout_info = {
                "challenge": cyc["challenge"],
                "payout_number": payout_count,
                "request_time": flat_time,
                "amount": amount,
                "cycle_profit": cycle_profit
            }
            cyc["payouts"].append(payout_info)
            total_payouts_list.append(payout_info)

            baseline_balance = sub_equity.loc[flat_time]
            day_cursor = request_day_pos + 1

    return cycles, total_payouts_list


def print_multi_cycle_summary(symbol: str, cycles: list, total_payouts: list, challenge_fee: float, best_lot: float = None):
    print("\n" + "=" * 70)
    print(f" MULTI-CYCLE PAYOUT SUMMARY - {symbol}")
    print("=" * 70)
    if USE_RISK_BASED_LOT:
        print(f" ✔ Risk-based lot: {RISK_PER_TRADE_PCT}% risk per trade (max lot {MAX_LOT})")
        if best_lot is not None:
            print(f" ✔ Berekende lot voor beste combo: {best_lot:.2f}")
    else:
        print(f" ✔ Vaste lot: {FIXED_LOT}")
    if USE_BREAKEVEN:
        print(f" ✔ Breakeven-stop: na {BE_TRIGGER_R}R (offset {BE_OFFSET})")
    if USE_TRAILING_SL:
        print(f" ✔ Dynamische trailing SL: ATR×{TRAIL_ATR_MULT} (activatie na ATR×{TRAIL_ACTIVATION_ATR})")
    else:
        print(" ✔ Vaste SL (geen trailing)")
    print(" ✔ Next-bar entry + No same-bar exit + Slippage")
    print(f" ✔ Regels: Daily {DAILY_DD_PCT}% | Overall {MAX_OVERALL_LOSS_PCT}% | Single Trade {MAX_SINGLE_TRADE_LOSS_PCT}%")

    total_fees = len(cycles) * challenge_fee
    sum_payouts = sum(p["amount"] for p in total_payouts)
    net_result = sum_payouts - total_fees

    print(f" -> Totaal aantal challenges: {len(cycles)}")
    for cyc in cycles:
        status_str = (
            f"Geblazen op {cyc['blown_date']} ({cyc['reason']})"
            if cyc["blown_date"] else "Succesvol doorgelopen"
        )
        print(f"   [Challenge #{cyc['challenge']}] {cyc['start']} → {status_str}")
        if cyc["payouts"]:
            for p in cyc["payouts"]:
                print(f"      → Payout #{p['payout_number']} @ {p['request_time']} | ${p['amount']:.2f}")
        else:
            print("      → Geen payouts")

    print("-" * 70)
    print(f" Totaal uitbetaald : ${sum_payouts:.2f}")
    print(f" Totaal fees       : ${total_fees:.2f}")
    print(f" NETTO RESULTAAT   : ${net_result:.2f}")
    print("=" * 70)


# ==============================================================================
# 4. HOOFDSCRIPT – REALISTIC GRID
# ==============================================================================
def main():
    print("\n" + "=" * 70)
    print("  MT5 REALISTIC SCANNER – Risk-based Lot + Dynamic Trailing SL")
    print("=" * 70)
    if USE_RISK_BASED_LOT:
        print(f"  Risk per trade : {RISK_PER_TRADE_PCT}% van ${INITIAL_CAPITAL:.0f} = ${INITIAL_CAPITAL * RISK_PER_TRADE_PCT / 100:.2f}")
        print(f"  Lot limits     : {MIN_LOT} – {MAX_LOT}")
        print(f"  Contract value : {CONTRACT_VALUE} ($ per prijs-unit per lot)")
    else:
        print(f"  Vaste lot      : {FIXED_LOT}")
    if USE_BREAKEVEN:
        print(f"  Breakeven      : na {BE_TRIGGER_R}R (offset {BE_OFFSET})")
    else:
        print(f"  Breakeven      : UIT")
    if USE_TRAILING_SL:
        print(f"  Trailing SL    : ATR×{TRAIL_ATR_MULT} (start na ATR×{TRAIL_ACTIVATION_ATR} winst)")
    else:
        print(f"  Trailing SL    : UIT (vaste initiële SL)")
    print("=" * 70 + "\n")

    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=BACKTEST_MONTHS * 30)

    for SYMBOL in SYMBOLS:
        print(f"\n[1/5] Data ophalen voor {SYMBOL}...")
        df = load_cached_data(SYMBOL)
        if df is None:
            try:
                df = fetch_mt5_data_by_dates(SYMBOL, TIMEFRAME, date_from, date_to)
            except Exception as e:
                print(f"Fout bij ophalen data voor {SYMBOL}: {e}")
                continue
            save_cached_data(SYMBOL, df)

        print(f" → {len(df):,} candles geladen.")

        open_arr = df["Open"].values.astype(np.float64)
        high_arr = df["High"].values.astype(np.float64)
        low_arr = df["Low"].values.astype(np.float64)
        close_arr = df["Close"].values.astype(np.float64)

        # ATR voor trailing stop (periode 14)
        atr_series_ts = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr_arr = atr_series_ts.bfill().fillna(0.0).values.astype(np.float64)

        print("\n[2/5] Indicatoren berekenen...")
        rsi_8 = ta.rsi(df["Close"], length=8).fillna(50).values
        rsi_10 = ta.rsi(df["Close"], length=10).fillna(50).values
        rsi_14 = ta.rsi(df["Close"], length=14).fillna(50).values
        rsi_21 = ta.rsi(df["Close"], length=21).fillna(50).values

        ema9 = ta.ema(df["Close"], length=9)
        ema21 = ta.ema(df["Close"], length=21)
        cond_ema_9_21 = (ema9 > ema21).values

        ema12 = ta.ema(df["Close"], length=12)
        ema26 = ta.ema(df["Close"], length=26)
        cond_ema_12_26 = (ema12 > ema26).values

        ema20 = ta.ema(df["Close"], length=20)
        ema50 = ta.ema(df["Close"], length=50)
        cond_ema_20_50 = (ema20 > ema50).values

        ema50_s = ta.ema(df["Close"], length=50)
        ema200_s = ta.ema(df["Close"], length=200)
        cond_ema_50_200 = (ema50_s > ema200_s).values

        cond_trend_50 = (df["Close"] > ema50).values
        cond_trend_100 = (df["Close"] > ta.ema(df["Close"], length=100)).values
        cond_trend_200 = (df["Close"] > ta.ema(df["Close"], length=200)).values

        macd = ta.macd(df["Close"])
        cond_macd = (
            macd["MACDh_12_26_9"] > 0
            if "MACDh_12_26_9" in macd.columns
            else macd.iloc[:, 1] > 0
        ).values

        st = ta.supertrend(df["High"], df["Low"], df["Close"])
        st_col = [c for c in st.columns if c.startswith("SUPERTd")][0]
        cond_st = (st[st_col] == 1).values

        stoch = ta.stoch(df["High"], df["Low"], df["Close"])
        cond_stoch = ((stoch.iloc[:, 0] > stoch.iloc[:, 1]) & (stoch.iloc[:, 0] < 80)).values

        print("\n[3/5] Parameter grid opbouwen...")
        # Iets smallere, realistischere grid dan de originele 6912
        rsi_len_opts = [8, 10, 14, 21]
        rsi_thresh_opts = [30, 35, 40, 45]
        ema_pair_opts = ["9/21", "12/26", "20/50", "50/200"]
        ema_trend_opts = [50, 100, 200]
        min_agree_opts = [4, 5]

        last_close = close_arr[-1]
        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr_val = (
            atr_series.dropna().iloc[-1]
            if atr_series is not None and not atr_series.dropna().empty
            else last_close * 0.01
        )
        dec_places = 4 if last_close < 10 else 2

        sl_points_opts = [
            float(np.round(m * atr_val, dec_places))
            for m in [1.0, 1.5, 2.0, 2.5, 3.0]
        ]
        tp_points_opts = [
            float(np.round(m * atr_val, dec_places))
            for m in [2.0, 3.0, 4.0, 5.0, 6.0]
        ]

        param_grid = list(itertools.product(
            rsi_len_opts, rsi_thresh_opts, ema_pair_opts,
            ema_trend_opts, min_agree_opts, sl_points_opts, tp_points_opts
        ))
        print(f" → Totaal combinaties: {len(param_grid):,}")

        print("\n[4/5] Signalen genereren...")
        n_bars = len(df)
        n_combos = len(param_grid)
        raw_entries_matrix = np.zeros((n_bars, n_combos), dtype=np.bool_)
        sl_arr = np.zeros(n_combos, dtype=np.float64)
        tp_arr = np.zeros(n_combos, dtype=np.float64)

        for i, (rsi_len, rsi_val, ema_pair, trend_len, min_agree, sl_pts, tp_pts) in enumerate(param_grid):
            if rsi_len == 8:
                rsi_arr = rsi_8
            elif rsi_len == 10:
                rsi_arr = rsi_10
            elif rsi_len == 14:
                rsi_arr = rsi_14
            else:
                rsi_arr = rsi_21

            c_rsi = rsi_arr < rsi_val

            if ema_pair == "9/21":
                c_ema = cond_ema_9_21
            elif ema_pair == "12/26":
                c_ema = cond_ema_12_26
            elif ema_pair == "20/50":
                c_ema = cond_ema_20_50
            else:
                c_ema = cond_ema_50_200

            c_trend = (
                cond_trend_50 if trend_len == 50
                else (cond_trend_100 if trend_len == 100 else cond_trend_200)
            )

            agreement = (
                c_rsi.astype(int) +
                c_ema.astype(int) +
                cond_macd.astype(int) +
                cond_st.astype(int) +
                cond_stoch.astype(int) +
                c_trend.astype(int)
            )
            raw_entries_matrix[:, i] = agreement >= min_agree
            sl_arr[i] = sl_pts
            tp_arr[i] = tp_pts

        print("\n[5/5] Realistische VectorBT simulatie in batches...")
        BATCH_SIZE = 800
        # Spread als fractionele fee
        spread_fee_series = df["Spread_Price"] / df["Close"]

        all_summary_list = []
        total_batches = (n_combos + BATCH_SIZE - 1) // BATCH_SIZE

        for b in range(total_batches):
            start_idx = b * BATCH_SIZE
            end_idx = min((b + 1) * BATCH_SIZE, n_combos)

            chunk_grid = param_grid[start_idx:end_idx]
            chunk_raw = raw_entries_matrix[:, start_idx:end_idx]
            chunk_sl = sl_arr[start_idx:end_idx]
            chunk_tp = tp_arr[start_idx:end_idx]

            chunk_entries, chunk_exits = process_trades_numba_realistic(
                open_arr, high_arr, low_arr, close_arr,
                atr_arr,
                chunk_raw, chunk_sl, chunk_tp,
                SLIPPAGE_POINTS, ENTRY_ON_NEXT_BAR,
                USE_TRAILING_SL, TRAIL_ATR_MULT, TRAIL_ACTIVATION_ATR,
                USE_BREAKEVEN, BE_TRIGGER_R, BE_OFFSET
            )

            chunk_index = pd.MultiIndex.from_tuples(
                chunk_grid,
                names=["rsi_len", "rsi_max", "ema_pair", "ema_trend",
                       "min_agree", "sl_points", "tp_points"]
            )

            # --- Lot size per combo (risk-based of vast) ---
            # Elke parameter-combo heeft een andere SL → andere lot
            size_data = {}
            for j, combo in enumerate(chunk_grid):
                sl_dist = chunk_sl[j]
                lot = calc_lot_size(sl_dist)
                size_data[combo] = lot

            # DataFrame met constante size per kolom (vectorbt eist series/df)
            size_df = pd.DataFrame(
                {col: size_data[col] for col in chunk_index},
                index=df.index
            )

            # Portfolio simulatie met vaste size, geen pyramiding
            pf_chunk = vbt.Portfolio.from_signals(
                close=df["Close"],
                entries=pd.DataFrame(chunk_entries, index=df.index, columns=chunk_index),
                exits=pd.DataFrame(chunk_exits, index=df.index, columns=chunk_index),
                size=size_df,           # units (bij BTC: 0.25 = 0.25 BTC)
                size_type="amount",
                fees=spread_fee_series, # spread als fractionele fee
                init_cash=INITIAL_CAPITAL,
                freq="5m",
                accumulate=False,       # Max 1 positie tegelijk (prop firm regel)
            )

            pf_trades = pf_chunk.trades

            def _to_series(obj, default=0.0):
                """Zet vectorbt output om naar een platte Series met chunk_index."""
                try:
                    if obj is None:
                        return pd.Series(default, index=chunk_index, dtype=float)
                    if isinstance(obj, pd.DataFrame):
                        # Neem eerste kolom als het een DataFrame is
                        obj = obj.iloc[:, 0] if obj.shape[1] > 0 else pd.Series(default, index=chunk_index)
                    if isinstance(obj, pd.Series):
                        # Als MultiIndex dieper is dan chunk_index → probeer te alignen
                        if isinstance(obj.index, pd.MultiIndex) and obj.index.nlevels > chunk_index.nlevels:
                            # Groepeer op de eerste n levels die overeenkomen
                            obj = obj.groupby(level=list(range(chunk_index.nlevels))).first()
                        s = obj.reindex(chunk_index)
                        s = s.fillna(default).astype(float)
                        s.index = chunk_index
                        return s
                    # Scalar of array
                    arr = np.asarray(obj).astype(float).ravel()
                    if arr.size == 1:
                        return pd.Series(float(arr[0]), index=chunk_index, dtype=float)
                    if arr.size == len(chunk_index):
                        return pd.Series(arr, index=chunk_index, dtype=float)
                    return pd.Series(default, index=chunk_index, dtype=float)
                except Exception:
                    return pd.Series(default, index=chunk_index, dtype=float)

            try:
                ret = _to_series(pf_chunk.total_return() * 100.0)
            except Exception:
                ret = pd.Series(0.0, index=chunk_index, dtype=float)

            try:
                wr = _to_series(pf_trades.win_rate() * 100.0)
            except Exception:
                wr = pd.Series(0.0, index=chunk_index, dtype=float)

            try:
                ntr = _to_series(pf_trades.count(), default=0)
            except Exception:
                ntr = pd.Series(0, index=chunk_index, dtype=float)

            try:
                mdd = _to_series(pf_chunk.max_drawdown() * 100.0)
            except Exception:
                mdd = pd.Series(0.0, index=chunk_index, dtype=float)

            try:
                pf_raw = pf_trades.profit_factor()
                pfac = _to_series(pf_raw)
            except Exception:
                pfac = pd.Series(0.0, index=chunk_index, dtype=float)

            res_chunk = pd.DataFrame({
                "Return (%)": ret.round(2),
                "Win Rate (%)": wr.round(2),
                "Total Trades": ntr.astype(int),
                "Max Drawdown (%)": mdd.round(2),
                "Profit Factor": pfac.round(2),
            }, index=chunk_index)

            all_summary_list.append(res_chunk)
            del pf_chunk
            gc.collect()

            print(f"   Batch {b+1}/{total_batches} klaar")

        results_df = pd.concat(all_summary_list)
        results_df = results_df[results_df["Total Trades"] >= 30]  # filter te weinig trades

        if results_df.empty:
            print(f"\n → Geen bruikbare resultaten voor {SYMBOL}.\n")
            continue

        # Sorteer op Profit Factor + lage DD + voldoende trades
        candidates = (
            results_df
            .sort_values(by=["Profit Factor", "Return (%)"], ascending=False)
            .head(20)
        )

        print("\n--- Top 10 kandidaten (realistische engine) ---")
        print(candidates.head(10).to_string())

        best_combo = candidates.index[0]
        best_cycles = None
        best_total_payouts = None
        max_net_profit = -999999.0

        for candidate_combo in candidates.index:
            c_idx = param_grid.index(candidate_combo)

            c_entries, c_exits = process_trades_numba_realistic(
                open_arr, high_arr, low_arr, close_arr,
                atr_arr,
                raw_entries_matrix[:, c_idx:c_idx+1],
                sl_arr[c_idx:c_idx+1],
                tp_arr[c_idx:c_idx+1],
                SLIPPAGE_POINTS,
                ENTRY_ON_NEXT_BAR,
                USE_TRAILING_SL, TRAIL_ATR_MULT, TRAIL_ACTIVATION_ATR,
                USE_BREAKEVEN, BE_TRIGGER_R, BE_OFFSET
            )

            # Risk-based lot voor deze specifieke SL
            c_sl = sl_arr[c_idx]
            c_lot = calc_lot_size(c_sl)

            c_pf = vbt.Portfolio.from_signals(
                close=df["Close"],
                entries=pd.DataFrame(c_entries, index=df.index, columns=[candidate_combo]),
                exits=pd.DataFrame(c_exits, index=df.index, columns=[candidate_combo]),
                size=c_lot,
                size_type="amount",
                fees=spread_fee_series,
                init_cash=INITIAL_CAPITAL,
                freq="5m",
                accumulate=False,
            )

            c_equity = c_pf.value()
            if isinstance(c_equity, pd.DataFrame):
                c_equity = c_equity.iloc[:, 0]

            c_trades_readable = c_pf.trades.records_readable
            if (not c_trades_readable.empty and
                    "Exit Timestamp" in c_trades_readable.columns and
                    "PnL" in c_trades_readable.columns):
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

            cycles, total_payouts = simulate_multi_cycle_strategy(
                c_equity, c_pos_mask, c_trades_pnl, INITIAL_CAPITAL
            )

            total_fees = len(cycles) * CHALLENGE_FEE
            sum_payouts = sum(p["amount"] for p in total_payouts)
            net_profit = sum_payouts - total_fees

            if net_profit > max_net_profit:
                max_net_profit = net_profit
                best_combo = candidate_combo
                best_cycles = cycles
                best_total_payouts = total_payouts

        best_lot = calc_lot_size(best_combo[5]) if best_combo is not None else None  # index 5 = sl_points
        print(f"\n → Beste combo volgens prop-firm netto: {best_combo}")
        if best_lot is not None:
            print(f" → Risk-based lot size voor deze SL: {best_lot:.2f}")
        print_multi_cycle_summary(SYMBOL, best_cycles, best_total_payouts, CHALLENGE_FEE, best_lot)


if __name__ == "__main__":
    main()
