# Quant Agent: Prop Firm Simulator & Strategy Optimizer

An autonomous, high-performance quantitative research pipeline and prop-firm backtesting framework built using **Python**, **VectorBT**, **Numba**, **Pandas**, and **Pandas-TA**.

---

## Key Features

- **Modular Quantitative Architecture:** Clear separation of concerns across data ingestion, indicator calculation, execution friction, swap accounting, prop-firm risk simulation, and Monte Carlo validation.
- **Strict Anti-Look-Ahead Integrity:** Enforces `shift(1)` lagging on bar-close indicators (EMA, RSI) before reindexing to lower timeframe execution bars to ensure zero look-ahead bias.
- **Numba `@njit` Acceleration:** High-speed compiled loops for evaluating multi-variable parameter grids, daily/static drawdowns, payout targets, reset fees, and Monte Carlo bootstrap resamples.
- **Prop-Firm Compliance & Risk Engine:**
  - **Static Overall Drawdown:** Enforces maximum total account drawdown relative to initial account size.
  - **Daily Reset Drawdown:** Tracks and resets daily starting account balance per date ordinal.
  - **Payout Target & Challenge Fee Accounting:** Tracks hit targets, resets balances upon payout, and deducts reset/challenge fees on account busts.
- **Time Filters:** Flat-before-overnight (`23:55`) and weekend entry blocking and forced exit closures.
- **Accurate Holding & Triple Swaps:** Calculates holding swap costs using business day date ordinals (`np.busday_count`), accurately applying triple swap charges on Wednesday nights.

---

## Installation & Setup

### Prerequisites

- **Python 3.10+**
- Dependencies: `vectorbt`, `numba`, `pandas`, `pandas-ta`, `numpy`, `scipy`, `scikit-learn`, `plotly<6.0.0`

### Installing Dependencies

```bash
pip install "plotly<6.0.0" vectorbt numba pandas pandas_ta scipy scikit-learn
```

> **Note:** `plotly<6.0.0` is required to maintain compatibility with `vectorbt` layout templates.

---

## Repository Structure

```
├── cache/                                  # Local cache directory for market data CSVs
│   └── XAUUSD_M5_202408190100_202608282355.csv
├── new_realistic.py                        # Main quantitative pipeline & grid search optimizer
└── README.md                               # Framework documentation
```

---

## Configuration Options

Inside `new_realistic.py`, key strategy and prop-firm parameters can be configured:

```python
# Prop Firm Rules
INITIAL_BALANCE    = 100_000.0  # Starting capital ($)
MAX_DRAWDOWN_PCT   = 0.08       # 8% static max drawdown
DAILY_DRAWDOWN_PCT = 0.05       # 5% daily drawdown
PAYOUT_TARGET_PCT  = 0.05       # 5% payout target
CHALLENGE_FEE      = 49.0       # Reset fee per challenge ($)

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
```

---

## Grid Search & Monte Carlo Parameter Optimization

The grid search space evaluates combinations of EMA fast, EMA slow, RSI period, and RSI entry thresholds:

```python
PARAM_GRID = {
    "ema_fast":   [10, 15, 20, 25],
    "ema_slow":   [30, 40, 50, 60],
    "rsi_len":    [10, 14, 21],
    "rsi_entry":  [50, 55, 60],
}
```

- Multi-variable grid search evaluates all valid parameter combinations (`ema_fast < ema_slow`).
- Monte Carlo simulation runs `1,000` bootstrap iterations on top candidates (`MC_TOP_N = 5`) to assess return distributions, 5th percentile outcomes (`p5`), median outcomes, and win probabilities.

---

## Execution Instructions

To run the grid search optimizer and prop firm simulator:

```bash
python new_realistic.py
```

### Expected Output Example

```text
[01] Marktdata inladen uit: cache/XAUUSD_M5_202408190100_202608282355.csv

[GRID] Start grid search: 144 geldige combinaties...
  [10/144] 13.1s – laatste net_profit: $0
  ...
  [144/144] 116.9s – laatste net_profit: $0

[GRID] Monte Carlo simulatie op top-5 kandidaten...

==============================================================================
TOP 10 PARAMETER COMBINATIES (gesorteerd op single-run net profit)
==============================================================================
   ema_fast  ema_slow  rsi_len  rsi_entry  n_trades  payouts  busts net_profit total_swap mc_mean mc_median mc_p5 mc_win%
0        10        30       10         50      2836        0      0         $0      $-1.8    $137        $0    $0    2.7%
...
==============================================================================
```
