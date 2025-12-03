# SBI Multi-Timeframe SVR Trading Model & Streamlit App

## 1. Overview

This project builds a **machine learning–based trading model** for **SBI Bank** using:

* **Multi-timeframe price data**:

  * 15-minute (microstructure)
  * 1-hour (main prediction timeframe)
  * 4-hour (higher-timeframe context)
* A **Support Vector Regression (SVR)** model trained to predict the **next 1-hour return**
* A **Streamlit web app** that:

  * Loads the **pre-trained model from a pickle file (`svr_SBI_multi_tf_1h.pkl`)**
  * Lets the user upload new CSV files (15m / 1H / 4H)
  * Rebuilds all features exactly as in training
  * Generates:

    * Predicted next 1-hour returns
    * Predicted next 1-hour close prices
    * Trading signals (BUY / SELL / FLAT) using:

      * Threshold
      * Quantile
      * Tercile methods
    * Full **backtest results, graphs, and trade list**

The goal is to provide a **quant-style, explainable workflow** that a client can understand end-to-end.

---

## 2. High-Level Workflow

The complete pipeline has **two phases**:

### A. Model Training (done once offline)

1. Load historical **15m, 1H, and 4H** OHLCV data for SBI.
2. Engineer features:

   * Price action (body, wicks, range, direction)
   * Technical indicators (EMA, RSI, MACD, ATR, volatility, volume MA)
   * 15m microstructure aggregated to 1H
   * 4H higher-timeframe context merged into 1H
3. Define the **target**:

   * `future_return_1h = close_1h.shift(-1) / close_1h - 1`
   * This is the **next 1-hour return**.
4. Train an **SVR model** with:

   * Standardized inputs (using `StandardScaler`)
   * `SVR(kernel="rbf")`
5. Evaluate train/test performance (RMSE, basic checks).
6. Save the trained pipeline + metadata as:

   * `svr_SBI_multi_tf_1h.pkl`

> The Streamlit app does *not* retrain. It only uses this pre-trained model.

---

### B. Streamlit App (online inference + backtest)

For any new 15m/1H/4H SBI data that follows the same format:

1. User uploads three CSVs:

   * `SBI_*_1H_...csv`
   * `SBI_*_4H_...csv`
   * `SBI_*_15min_...csv`
2. App rebuilds all features in real time:

   * 1H features
   * 4H features merged into 1H
   * 15m microstructure aggregated to 1H
3. The app loads the **pickled SVR model** and predicts:

   * `predicted_return_1h`
   * `predicted_next_close_1h`
4. Using predicted returns, the app generates trading **signals**:

   * BUY (+1), SELL (-1), or FLAT (0)
     based on one of three modes:
   * **Threshold mode** – fixed numeric return threshold
   * **Quantile mode** – uses upper/lower quantiles of predicted returns
   * **Tercile mode** – uses top 1/3 vs bottom 1/3 of predicted returns
5. The app performs a **1-bar ahead backtest**:

   * Each bar:

     * If signal = BUY → enter at current close, exit next bar’s close
     * If signal = SELL → short at current close, cover at next bar’s close
   * Computes P&L, equity curve, and metrics.
6. The app displays:

   * Performance metrics (win rate, total return, drawdown, Sharpe, profit factor)
   * Price chart with BUY/SELL markers
   * Strategy equity curve
   * Sample trades table
   * Sample predictions table
   * Downloadable CSVs for:

     * Full predictions + signals
     * Trade list

---

## 3. Data Requirements

All three input CSV files must share the same basic schema:

### 3.1 Required Columns (for 15m, 1H, 4H files)

Each CSV must contain:

```text
datetime_ist,timestamp,open,high,low,close,volume
```

* `datetime_ist`

  * String datetime with IST timezone (e.g. `2024-12-03 09:30:00+05:30`)
  * Used as the **time index**.
* `timestamp`

  * Unix timestamp (seconds since epoch)
  * Kept mainly for traceability.
* `open, high, low, close, volume`

  * Standard OHLCV fields.

### 3.2 Timeframes

* **15m file**:

  * 15-minute candles covering the same date range as 1H.
* **1H file**:

  * Primary timeframe for model and backtest.
* **4H file**:

  * Either resampled from 1H or precomputed externally.

---

## 4. Feature Engineering (What the Model “Sees”)

### 4.1 1-Hour Features

For each 1H bar, we compute:

**Price action:**

* `body_1h = close_1h - open_1h`
* `range_1h = high_1h - low_1h`
* `upper_wick_1h = high_1h - max(open_1h, close_1h)`
* `lower_wick_1h = min(open_1h, close_1h) - low_1h`
* `body_ratio_1h = body_1h / range_1h`
* `direction_1h = sign(body_1h)` (+1 = bullish, -1 = bearish, 0 = neutral)

**Returns & volatility:**

* `ret_1h` – 1-bar return
* `ret_1h_3` – 3-bar return
* `volatility_1h_10` – rolling standard deviation of `ret_1h` over 10 bars

**Volume:**

* `vol_ma_1h_20` – 20-bar moving average of 1H volume

**Indicators:**

* `ema_1h_20`, `ema_1h_50` – exponential moving averages
* `rsi_1h_14` – 14-period RSI
* `atr_1h_14` – 14-period Average True Range
* `macd_1h`, `macd_signal_1h`, `macd_hist_1h`

---

### 4.2 4-Hour Features

Similar structure for 4H:

* Price action: `body_4h`, `range_4h`, `upper_wick_4h`, `lower_wick_4h`, `body_ratio_4h`, `direction_4h`
* Returns/volatility: `ret_4h`, `volatility_4h_10`
* Volume: `vol_ma_4h_20`
* Indicators: `ema_4h_50`, `rsi_4h_14`, `atr_4h_14`, `macd_4h`, `macd_signal_4h`, `macd_hist_4h`

These 4H features are **merged into the 1H dataframe** using `merge_asof`:

> Each 1H bar gets the **last fully completed 4H bar** as context.

---

### 4.3 15-Minute Microstructure Features

From the 15m data:

* `body_15m = close - open`
* `range_15m = high - low`
* `is_green_15m = 1 if close > open else 0`

Then we **resample 15m → 1H** and aggregate:

* `vol_sum_15m_in_1h` – sum of 15m volume inside the hour
* `range_mean_15m_in_1h` – average 15m range inside the hour
* `body_mean_15m_in_1h` – average 15m body inside the hour
* `green_ratio_15m_in_1h` – percentage of bullish 15m candles inside the hour

This gives the model a sense of **microstructure pressure** inside each 1H bar.

---

## 5. Model Details

* **Algorithm**: Support Vector Regression (SVR)
* **Library**: scikit-learn
* **Pipeline**:

  * `StandardScaler` (feature standardization)
  * `SVR(kernel="rbf", C=1.0, epsilon≈0.0005)`

**Target:**

```python
future_return_1h = close_1h.shift(-1) / close_1h - 1
```

The model predicts a **continuous return value**, not a discrete BUY/SELL label.
Trading decisions are created in the Streamlit app from these continuous predictions.

The trained pipeline and metadata are stored as:

* `svr_SBI_multi_tf_1h.pkl`

---

## 6. Streamlit App – How It Works

### 6.1 Inputs

* 3 CSV files uploaded via sidebar:

  * 1H CSV
  * 4H CSV
  * 15m CSV
* Choice of **signal mode**:

  * Threshold
  * Quantile
  * Tercile
* Mode-specific parameters:

  * Threshold: numeric return threshold (e.g. 0.0003 = 0.03%)
  * Quantile: lower & upper quantiles (e.g. 0.3 and 0.7)
  * Tercile: fixed (bottom 1/3, top 1/3)

### 6.2 Processing Steps

1. **Load Model**

   * `svr_SBI_multi_tf_1h.pkl` is loaded once (cached with `st.cache_resource`).

2. **Read & Preprocess CSVs**

   * Convert `datetime_ist` to datetime and set as index
   * Sort by time
   * Rename columns for 1H / 4H contexts
   * Compute features for each timeframe (as described above)

3. **Merge Timeframes**

   * `merge_asof` is used twice:

     * 4H → 1H
     * 15m aggregated → 1H

4. **Predict Returns**

   * Features are aligned to the same feature columns used in training (`FEATURE_COLS`)
   * The model predicts:

     * `predicted_return_1h`
     * `predicted_next_close_1h = close_1h * (1 + predicted_return_1h)`

5. **Create Signals**

   * **Threshold mode**:

     * BUY if `predicted_return_1h ≥ threshold`
     * SELL if `predicted_return_1h ≤ -threshold`
     * FLAT otherwise
   * **Quantile mode**:

     * Compute quantiles on the predicted returns within the uploaded file:

       * BUY if prediction ≥ upper quantile (`q_high`)
       * SELL if prediction ≤ lower quantile (`q_low`)
       * FLAT otherwise
   * **Tercile mode**:

     * BUY if prediction is in top 1/3
     * SELL if prediction is in bottom 1/3
     * FLAT otherwise

6. **Backtest Logic (1-Bar Hold Strategy)**

For each 1H bar (except the last):

* If signal = +1 (BUY):

  * Enter long at `close[t]`
  * Exit at `close[t+1]`
  * Return = `(close[t+1]/close[t]) - 1`
* If signal = -1 (SELL/short):

  * Short at `close[t]`, cover at `close[t+1]`
  * Return = `(close[t]/close[t+1]) - 1`
* If signal = 0: no trade

The code builds:

* A `trades_df` with:

  * entry_time, entry_price, direction, exit_time, exit_price, return_pct
* A per-bar P&L series:

  * `bar_pnl_pct = signal * future_return_1h_actual`
* An equity curve:

  * `equity_curve = (1 + bar_pnl_pct).cumprod()`

---

## 7. Performance Metrics

The app computes and displays:

* **Number of trades**
* **Win rate** (% of trades with positive return)
* **Total return** (compounded over all trades)
* **Average trade return**
* **Max drawdown**
* **Sharpe ratio** (approximate, per-bar, annualised using a scaling factor)
* **Profit factor**:

  * Sum of all positive returns ÷ sum of all absolute negative returns

Plus:

* **Price chart** with BUY/SELL markers
* **Equity curve chart**
* **Close vs predicted next close chart**

---

## 8. How to Run the App

### 8.1 Requirements

* Python 3.x
* Recommended libraries:

  * `streamlit`
  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `matplotlib`

Install (example):

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

### 8.2 Files Needed

Place in the same folder:

* `app.py` (the Streamlit app)
* `svr_SBI_multi_tf_1h.pkl` (trained model)
* Your 15m, 1H, and 4H CSV data files (for inference/testing)

### 8.3 Start the App

```bash
streamlit run app.py
```

Then:

1. Go to the URL shown in the terminal (usually `http://localhost:8501`)
2. In the sidebar:

   * Upload 1H / 4H / 15m CSV files
   * Choose `Signal mode` (Threshold, Quantile, or Tercile)
   * Set relevant parameters (e.g., threshold or quantiles)
3. Click **“🚀 Run Prediction + Backtest”**
4. Inspect:

   * Performance metrics
   * Trades table
   * Charts
   * Downloadable CSVs

---

## 9. Interpretation & Limitations

* The model predicts **next 1H return**, not guaranteed profit.
* Predictions are based solely on historical price/volume and technical indicators;
  it does **not** incorporate:

  * News
  * Fundamental data
  * Order book / market microstructure beyond 15m OHLCV agg
* Backtest is a **simple 1-bar hold strategy** with:

  * No transaction costs
  * No slippage
  * No position sizing logic
* Real-world results may differ due to:

  * Execution latency
  * Liquidity
  * Fees and taxes
  * Regime changes in the market

This project is best viewed as a **research and prototyping tool**, not as a production-ready auto-trading system without further validation.

