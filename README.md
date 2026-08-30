# Quant Agent: Prop Firm Simulator & Strategy Optimizer

An autonomous, high-performance quantitative research pipeline and prop-firm backtesting framework built using **Python**, **VectorBT**, **Numba**, **Pandas**, and **Pandas-TA**.

---

## Key Features

- **Modular Quantitative Architecture:** Clear separation of concerns across data ingestion, indicator calculation, execution friction, swap accounting, prop-firm risk simulation, walk-forward grid search, and Monte Carlo validation.
- **Walk-Forward Validation (Train vs Test):** Chronologically splits historical data into **70% Train** (in-sample parameter optimization) and **30% Test** (out-of-sample evaluation) to measure overfit gaps and prevent look-ahead bias.
- **Symmetric Long & Short Execution:** Evaluates both Long and Short trading signals using VectorBT:
  - **Long:** `EMA_F > EMA_S` & `RSI > rsi_entry`
  - **Short:** `EMA_F < EMA_S` & `RSI < (100 - rsi_entry)`
- **Strict Anti-Look-Ahead Integrity:** Enforces `shift(1)` lagging on bar-close indicators (EMA, RSI) before reindexing to lower timeframe execution bars.
- **Numba `@njit` Acceleration:** High-speed compiled loops for evaluating multi-variable parameter grids, daily/static drawdowns, payout targets, reset fees, and Monte Carlo bootstrap resamples.
- **Prop-Firm Compliance & Risk Engine:**
  - **Static Overall Drawdown:** Enforces maximum total account drawdown relative to initial account size.
  - **Daily Reset Drawdown:** Tracks daily starting account balance per date ordinal (`USE_CLOSED_TRADE_DD = True`).
  - **Payout Target & Challenge Fee Accounting:** Tracks hit targets, resets balances upon payout, and deducts reset/challenge fees on account busts.
- **Time Filters:** Flat-before-overnight (`23:55`) and weekend entry blocking and forced exit closures.
- **Accurate Holding & Triple Swaps:** Calculates directional holding swap costs using business day date ordinals (`np.busday_count`), accurately applying triple swap charges on Wednesday nights for Long and Short positions.
- **Trade-Count Filtering & Ranking Score:** Filters out overtrading (`> 5000 trades`) or under-trading (`< 10 trades`) parameter combinations and ranks strategies using a multi-factor score:
  $$\text{Score} = \frac{\text{Net Profit}}{\max(\text{Busts}, 1)}$$

---

## Known Limitations

- **Closed-Trade Daily Drawdown:** Daily drawdown checks are evaluated on closed-trade equity PnL per date ordinal ID. Intra-bar floating equity fluctuations are not simulated in trade-record arrays (`USE_CLOSED_TRADE_DD = True`).
- **Static Overall Drawdown:** Overall max drawdown is evaluated statically against the initial challenge balance ($100,000) as per standard prop firm challenge rules.

---

## Installation & Setup

### Prerequisites

- **Python 3.10+**
- Dependencies listed in `requirements.txt`: `vectorbt`, `numba`, `pandas`, `pandas-ta`, `numpy`, `scipy`, `scikit-learn`, `plotly<6.0.0`

### Installing Dependencies

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
├── cache/                                  # Local cache directory for market data CSVs
│   └── XAUUSD_M5_202408190100_202608282355.csv
├── new_realistic.py                        # Main quantitative pipeline & grid search optimizer
├── requirements.txt                        # Production dependency pins
└── README.md                               # Framework documentation
```

---

## Configuration Options

Inside `new_realistic.py`, key strategy and prop-firm parameters can be configured:

```python
# Prop Firm Rules
INITIAL_BALANCE       = 100_000.0  # Starting capital ($)
MAX_DRAWDOWN_PCT      = 0.08       # 8% static max drawdown
DAILY_DRAWDOWN_PCT    = 0.05       # 5% daily drawdown
PAYOUT_TARGET_PCT     = 0.05       # 5% payout target
CHALLENGE_FEE         = 49.0       # Reset fee per challenge ($)
USE_CLOSED_TRADE_DD   = True       # Closed-trade equity PnL for Daily DD

# Trade Specifications
CONTRACT_SIZE      = 100.0      # XAUUSD lot size multiplier
FIXED_LOT_SIZE     = 0.02       # Trade lot size
COMMISSION_PER_LOT = 5.0        # Commission ($/lot)
SWAP_LONG          = -5.0       # Long swap rate ($/lot/night)
SWAP_SHORT         = -2.0       # Short swap rate ($/lot/night)
TRIPLE_SWAP_DAY    = "Wed"      # Day for 3x swap multiplier

# Time Filters
NO_OVERNIGHT = True             # Close positions prior to overnight rollover
NO_WEEKEND   = True             # Close positions prior to weekend closure
EOD_HOUR     = 23               # Cutoff hour (UTC)
EOD_MINUTE   = 55               # Cutoff minute

# Walk-Forward Split
TRAIN_RATIO  = 0.70             # 70% Train, 30% Test
```

---

## Minimal Parameter Grid for Fast Testing

To run a fast test run, you can reduce `PARAM_GRID` in `new_realistic.py`:

```python
# Fast test grid configuration
PARAM_GRID = {
    "ema_fast":   [10, 20],
    "ema_slow":   [30, 50],
    "rsi_len":    [14],
    "rsi_entry":  [50, 55],
}
```

---

## Execution Instructions

To run the full Walk-Forward grid search optimizer, prop firm simulator, and Monte Carlo engine:

```bash
python new_realistic.py
```

### Expected Output Example

```text
[DATA] Specified file not found or None. Using cached file: cache/XAUUSD_M5_202408190100_202608282355.csv

[01] Marktdata inladen uit: cache/XAUUSD_M5_202408190100_202608282355.csv

[WALK-FORWARD] Data Split:
  Train Period: 2024-08-19 01:00:00 -> 2026-01-20 22:55:00 (100803 bars)
  Test Period : 2026-01-20 23:00:00 -> 2026-08-28 23:55:00 (43202 bars)

[GRID] Start train grid search: 144 geldige combinaties...
  [10/144] 11.4s – laatste train net_profit: $0
  ...
  [144/144] 89.1s – laatste train net_profit: $0

[TEST EVALUATION] Top-5 kandidaten evalueren op Out-Of-Sample Test dataset...

===============================================================================================
TOP 10 PARAMETER COMBINATIES (Walk-Forward: Train In-Sample vs Test Out-of-Sample)
===============================================================================================
   ema_fast  ema_slow  rsi_len  rsi_entry  train_n_trades train_net_profit train_score  test_n_trades test_net_profit overfit_gap mc_mean mc_win%
0        10        30       10         50            3969               $0          $0           1680              $0          $0    $310    6.1%
1        10        30       10         55            3746               $0          $0           1600              $0          $0    $315    6.2%
...
===============================================================================================
```
