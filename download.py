import os
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5
import pandas as pd

# ==============================================================================
# CONFIGURATIE
# ==============================================================================
SYMBOLS = ["MSFT", "EURUSD", "XAUUSD", "BTCUSD.nx", "USOIL.c", "NACUSD.c"]
BACKTEST_MONTHS = 6
TIMEFRAME = mt5.TIMEFRAME_M5
CACHE_DIR = "cache"

# ==============================================================================
# DOWNLOAD FUNCTIE
# ==============================================================================
def main():
    print("\n" + "=" * 60)
    print("  MT5 PRE-DOWNLOADER (CACHE BUILDER)")
    print("=" * 60 + "\n")

    # Zorg dat de cache map bestaat
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Verbind met MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialisatie mislukt. Foutcode: {mt5.last_error()}")
        return
    print("✅ Succesvol verbonden met MetaTrader 5.\n")

    # Bepaal datums
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=BACKTEST_MONTHS * 30)

    for symbol in SYMBOLS:
        print(f"Binnenhalen van {symbol}...")
        
        # Selecteer symbool in Market Watch
        if not mt5.symbol_select(symbol, True):
            print(f"  -> ❌ Symbool '{symbol}' niet gevonden in MT5 Market Watch.")
            continue

        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"  -> ❌ Kan symboolinfo voor '{symbol}' niet ophalen.")
            continue

        point_size = info.point

        # Download de rates
        rates = mt5.copy_rates_range(symbol, TIMEFRAME, date_from, date_to)
        
        if rates is None or len(rates) == 0:
            print(f"  -> ❌ Geen data ontvangen voor {symbol}.")
            continue

        # Bouw dataframe op
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"},
            inplace=True,
        )
        
        # Bereken de spread-kosten (nodig voor vectorbt)
        df["Spread_Price"] = df["spread"] * point_size

        # Bestandspaden opbouwen
        parquet_path = os.path.join(CACHE_DIR, f"{symbol}_6m_cache.parquet")
        csv_path = os.path.join(CACHE_DIR, f"{symbol}_6m_cache.csv")

        # Opslaan
        try:
            df.to_parquet(parquet_path)
            print(f"  -> ✅ Opgeslagen als Parquet ({len(df):,} candles)")
        except Exception:
            # Fallback naar CSV als Parquet (pyarrow/fastparquet) niet is geïnstalleerd
            df.to_csv(csv_path)
            print(f"  -> ✅ Opgeslagen als CSV ({len(df):,} candles)")

    # MT5 netjes afsluiten
    mt5.shutdown()
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLEET! Je kunt nu new.py draaien.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()