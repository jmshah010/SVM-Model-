import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="SBI SVR Multi-Timeframe Predictor",
    layout="wide"
)

st.title("📈 SBI Multi-Timeframe SVR Predictor (1H + 4H + 15m)")
st.write(
    "Upload your **1H, 4H, and 15m CSV files** (same structure as training), "
    "and the app will recreate the features and run the trained SVR model "
    "to predict the **next 1H return** for each row, then convert predictions "
    "into BUY/SELL signals using Threshold / Quantile / Tercile modes."
)

# ================== LOAD MODEL ==================

@st.cache_resource
def load_model(path: str = "svr_SBI_multi_tf_1h.pkl"):
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    description = artifact.get("description", "")
    return model, feature_cols, description

try:
    model, FEATURE_COLS, model_desc = load_model()
    st.success("✅ Model loaded successfully")
    if model_desc:
        st.caption(f"Model description: {model_desc}")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()


# ================== INDICATOR HELPERS ==================

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/period, adjust=False).mean()

    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = compute_ema(macd, signal)
    hist = macd - signal_line
    return pd.DataFrame({"macd": macd, "signal": signal_line, "macd_hist": hist})


def compute_atr(high: pd.Series,
                low: pd.Series,
                close: pd.Series,
                period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


# ================== PREPARE DATAFRAMES ==================

def prepare_1h_df(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df.set_index("datetime_ist").sort_index()

    df = df.rename(columns={
        "open": "open_1h",
        "high": "high_1h",
        "low": "low_1h",
        "close": "close_1h",
        "volume": "volume_1h",
        "timestamp": "timestamp_1h",
    })

    # price action
    df["body_1h"] = df["close_1h"] - df["open_1h"]
    df["range_1h"] = df["high_1h"] - df["low_1h"]
    df["upper_wick_1h"] = df["high_1h"] - df[["open_1h", "close_1h"]].max(axis=1)
    df["lower_wick_1h"] = df[["open_1h", "close_1h"]].min(axis=1) - df["low_1h"]
    df["body_ratio_1h"] = df["body_1h"] / (df["range_1h"] + 1e-10)
    df["direction_1h"] = np.sign(df["body_1h"])

    # returns & volatility
    df["ret_1h"] = df["close_1h"].pct_change()
    df["ret_1h_3"] = df["close_1h"].pct_change(3)
    df["volatility_1h_10"] = df["ret_1h"].rolling(10).std()

    # volume MA
    df["vol_ma_1h_20"] = df["volume_1h"].rolling(20).mean()

    # indicators
    df["ema_1h_20"] = compute_ema(df["close_1h"], 20)
    df["ema_1h_50"] = compute_ema(df["close_1h"], 50)
    df["rsi_1h_14"] = compute_rsi(df["close_1h"], 14)
    df["atr_1h_14"] = compute_atr(df["high_1h"], df["low_1h"], df["close_1h"], 14)

    macd_df = compute_macd(df["close_1h"])
    df["macd_1h"] = macd_df["macd"]
    df["macd_signal_1h"] = macd_df["signal"]
    df["macd_hist_1h"] = macd_df["macd_hist"]

    return df


def prepare_4h_df(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df.set_index("datetime_ist").sort_index()

    df = df.rename(columns={
        "open": "open_4h",
        "high": "high_4h",
        "low": "low_4h",
        "close": "close_4h",
        "volume": "volume_4h",
        "timestamp": "timestamp_4h",
    })

    df["body_4h"] = df["close_4h"] - df["open_4h"]
    df["range_4h"] = df["high_4h"] - df["low_4h"]
    df["upper_wick_4h"] = df["high_4h"] - df[["open_4h", "close_4h"]].max(axis=1)
    df["lower_wick_4h"] = df[["open_4h", "close_4h"]].min(axis=1) - df["low_4h"]
    df["body_ratio_4h"] = df["body_4h"] / (df["range_4h"] + 1e-10)
    df["direction_4h"] = np.sign(df["body_4h"])

    df["ret_4h"] = df["close_4h"].pct_change()
    df["volatility_4h_10"] = df["ret_4h"].rolling(10).std()
    df["vol_ma_4h_20"] = df["volume_4h"].rolling(20).mean()

    df["ema_4h_50"] = compute_ema(df["close_4h"], 50)
    df["rsi_4h_14"] = compute_rsi(df["close_4h"], 14)
    df["atr_4h_14"] = compute_atr(df["high_4h"], df["low_4h"], df["close_4h"], 14)

    macd_df = compute_macd(df["close_4h"])
    df["macd_4h"] = macd_df["macd"]
    df["macd_signal_4h"] = macd_df["signal"]
    df["macd_hist_4h"] = macd_df["macd_hist"]

    return df


def prepare_15m_micro_df(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df.set_index("datetime_ist").sort_index()

    df["body_15m"] = df["close"] - df["open"]
    df["range_15m"] = df["high"] - df["low"]
    df["is_green_15m"] = (df["close"] > df["open"]).astype(int)

    agg = {
        "volume": "sum",
        "range_15m": "mean",
        "body_15m": "mean",
        "is_green_15m": "mean",
    }

    # use new-style '1h' to avoid FutureWarning
    df_agg = df.resample("1h").agg(agg)

    df_agg = df_agg.rename(columns={
        "volume": "vol_sum_15m_in_1h",
        "range_15m": "range_mean_15m_in_1h",
        "body_15m": "body_mean_15m_in_1h",
        "is_green_15m": "green_ratio_15m_in_1h",
    })

    return df_agg


def build_features_from_uploaded_files(file_1h, file_4h, file_15m) -> pd.DataFrame:
    df_1h = prepare_1h_df(file_1h)
    df_4h = prepare_4h_df(file_4h)
    df_15m = prepare_15m_micro_df(file_15m)

    # merge 4H into 1H
    df_merged = pd.merge_asof(
        df_1h.sort_index(),
        df_4h.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    )

    # merge 15m microstructure into 1H
    df_merged = pd.merge_asof(
        df_merged.sort_index(),
        df_15m.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    )

    # no target here (future_return_1h) – this is purely for prediction / backtest
    df_merged = df_merged.dropna()

    return df_merged


# ================== SIGNAL + BACKTEST HELPERS ==================

def generate_signals(preds: np.ndarray, mode: str,
                     q_low: float = 0.3, q_high: float = 0.7,
                     manual_thr: float = 0.0003):
    """
    Convert predicted returns to signals:
    +1 (long), -1 (short), 0 (flat)
    """
    thresholds = {}

    if mode.lower() == "threshold":
        long_thr = manual_thr
        short_thr = -manual_thr

    elif mode.lower() == "quantile":
        long_thr = np.quantile(preds, q_high)
        short_thr = np.quantile(preds, q_low)

    elif mode.lower() == "tercile":
        long_thr = np.quantile(preds, 2/3)
        short_thr = np.quantile(preds, 1/3)

    else:
        raise ValueError("mode must be 'threshold', 'quantile', or 'tercile'")

    thresholds["long"] = float(long_thr)
    thresholds["short"] = float(short_thr)

    sig = np.where(preds >= long_thr, 1,
          np.where(preds <= short_thr, -1, 0))

    return sig, thresholds


def backtest_from_df(df: pd.DataFrame):
    """
    df must contain:
    - close_1h
    - signal
    - future_return_1h_actual
    """
    df_bt = df.copy()

    # bar P&L based on signal * next-bar return
    df_bt["bar_pnl_pct"] = df_bt["signal"] * df_bt["future_return_1h_actual"]
    df_bt["equity_curve"] = (1 + df_bt["bar_pnl_pct"]).cumprod()

    # build trade list (1-bar hold)
    trades = []
    closes = df_bt["close_1h"].values
    sigs = df_bt["signal"].values
    idx = df_bt.index.to_list()

    for i in range(len(df_bt) - 1):
        if sigs[i] == 0:
            continue

        entry_time = idx[i]
        exit_time = idx[i + 1]
        entry_price = closes[i]
        exit_price = closes[i + 1]
        direction = "BUY" if sigs[i] == 1 else "SELL"

        if sigs[i] == 1:  # long
            ret = (exit_price / entry_price) - 1
        else:             # short
            ret = (entry_price / exit_price) - 1

        trades.append({
            "entry_time": entry_time,
            "direction": direction,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "return_pct": ret
        })

    trades_df = pd.DataFrame(trades)

    # performance stats
    if len(trades_df) > 0:
        wins = trades_df["return_pct"] > 0
        win_rate = wins.mean()

        total_return = (1 + trades_df["return_pct"]).prod() - 1
        avg_trade_return = trades_df["return_pct"].mean()

        # max drawdown on equity curve
        ec = df_bt["equity_curve"]
        roll_max = ec.cummax()
        drawdown = (roll_max - ec) / roll_max
        max_dd = drawdown.max()

        # sharpe (per bar)
        rets = df_bt["bar_pnl_pct"].dropna()
        if rets.std() > 0:
            sharpe = rets.mean() / rets.std() * np.sqrt(252*6.5)  # approx intraday bars/year
        else:
            sharpe = np.nan

        # profit factor
        gains = trades_df.loc[trades_df["return_pct"] > 0, "return_pct"].sum()
        losses = trades_df.loc[trades_df["return_pct"] < 0, "return_pct"].sum()
        profit_factor = gains / abs(losses) if losses < 0 else np.nan
    else:
        win_rate = 0.0
        total_return = 0.0
        avg_trade_return = 0.0
        max_dd = 0.0
        sharpe = np.nan
        profit_factor = np.nan

    stats = {
        "num_trades": int(len(trades_df)),
        "win_rate": float(win_rate),
        "total_return": float(total_return),
        "avg_trade_return": float(avg_trade_return),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,  # handle nan
        "profit_factor": float(profit_factor) if profit_factor == profit_factor else np.nan
    }

    return trades_df, df_bt, stats


def plot_price_with_trades(df_bt: pd.DataFrame, trades_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_bt.index, df_bt["close_1h"], label="Close 1H", linewidth=1.0)

    buys = trades_df[trades_df["direction"] == "BUY"]
    sells = trades_df[trades_df["direction"] == "SELL"]

    ax.scatter(buys["entry_time"], buys["entry_price"], marker="^", s=60, color="green", label="BUY")
    ax.scatter(sells["entry_time"], sells["entry_price"], marker="v", s=60, color="red", label="SELL")

    ax.set_title("Price with BUY / SELL trades")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


def plot_equity_curve(df_bt: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_bt.index, df_bt["equity_curve"], label="Equity Curve", linewidth=1.2)
    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="gray")

    ax.set_title("Strategy Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


# ================== FILE UPLOAD UI ==================

st.sidebar.header("📂 Upload files")

file_1h = st.sidebar.file_uploader(
    "Upload 1H CSV (datetime_ist,timestamp,open,high,low,close,volume)",
    type=["csv"],
    key="file_1h"
)
file_4h = st.sidebar.file_uploader(
    "Upload 4H CSV (same columns)",
    type=["csv"],
    key="file_4h"
)
file_15m = st.sidebar.file_uploader(
    "Upload 15min CSV (same columns)",
    type=["csv"],
    key="file_15m"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Signal Settings")

signal_mode = st.sidebar.selectbox(
    "Signal mode",
    options=["Threshold", "Quantile", "Tercile"],
    index=1
)

if signal_mode == "Threshold":
    manual_thr = st.sidebar.number_input(
        "Absolute return threshold (e.g. 0.0003 ≈ 0.03%)",
        min_value=0.0001,
        max_value=0.01,
        value=0.0003,
        step=0.0001,
        format="%.4f"
    )
    q_low = q_high = None
elif signal_mode == "Quantile":
    q_low = st.sidebar.slider(
        "Short (lower) quantile",
        min_value=0.0, max_value=0.5, value=0.3, step=0.01
    )
    q_high = st.sidebar.slider(
        "Long (upper) quantile",
        min_value=0.5, max_value=1.0, value=0.7, step=0.01
    )
    manual_thr = None
else:  # Tercile
    manual_thr = None
    q_low = q_high = None
    st.sidebar.caption("Tercile = bottom 1/3 SELL, top 1/3 BUY, middle flat.")

run_button = st.sidebar.button("🚀 Run Prediction + Backtest")

if not (file_1h and file_4h and file_15m):
    st.info("Please upload **all three files (1H, 4H, 15m)** and then click **Run Prediction + Backtest**.")
    st.stop()

if run_button:
    with st.spinner("Building features, running model and backtest..."):
        try:
            df_features = build_features_from_uploaded_files(file_1h, file_4h, file_15m)
        except Exception as e:
            st.error(f"Error while preparing features from uploaded files: {e}")
            st.stop()

        # ensure all required feature columns exist
        missing = [c for c in FEATURE_COLS if c not in df_features.columns]
        if missing:
            st.error(f"The following required feature columns are missing in the prepared data: {missing}")
            st.write(missing)
            st.stop()

        # predictions
        X = df_features[FEATURE_COLS].copy()
        preds = model.predict(X)

        df_out = df_features.copy()
        df_out["predicted_return_1h"] = preds
        df_out["predicted_next_close_1h"] = df_out["close_1h"] * (1 + df_out["predicted_return_1h"])

        # true future return for backtest (1-bar ahead)
        df_out["future_return_1h_actual"] = df_out["close_1h"].shift(-1) / df_out["close_1h"] - 1
        df_out = df_out.dropna(subset=["future_return_1h_actual"])

        # generate signals
        if signal_mode == "Threshold":
            sig, thr = generate_signals(
                df_out["predicted_return_1h"].values,
                mode="threshold",
                manual_thr=manual_thr
            )
        elif signal_mode == "Quantile":
            sig, thr = generate_signals(
                df_out["predicted_return_1h"].values,
                mode="quantile",
                q_low=q_low,
                q_high=q_high
            )
        else:  # Tercile
            sig, thr = generate_signals(
                df_out["predicted_return_1h"].values,
                mode="tercile"
            )

        df_out["signal"] = sig

        # backtest
        trades_df, df_bt, stats = backtest_from_df(df_out)

    st.success("✅ Prediction + Backtest complete")

    # ========== TOP METRICS ==========
    st.subheader("📊 Strategy Performance Summary")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Trades", stats["num_trades"])
    col2.metric("Win rate", f"{stats['win_rate']*100:.2f}%")
    col3.metric("Total return", f"{stats['total_return']*100:.2f}%")

    col4.metric("Avg trade return", f"{stats['avg_trade_return']*100:.3f}%")
    col5.metric("Max drawdown", f"{stats['max_drawdown']*100:.2f}%")
    col6.metric("Sharpe (per bar)", f"{stats['sharpe']:.2f}" if stats["sharpe"] == stats["sharpe"] else "NaN")

    st.caption(f"Profit factor: {stats['profit_factor']:.2f}" if stats["profit_factor"] == stats["profit_factor"] else "Profit factor: NaN")

    st.markdown(f"**Signal mode:** `{signal_mode}` &nbsp;&nbsp; | &nbsp;&nbsp; **Thresholds:** {thr}")

    # ========== SAMPLE TRADES TABLE ==========
    st.subheader("🧾 Sample trades")
    if len(trades_df) > 0:
        st.dataframe(trades_df.head(20))
    else:
        st.write("No trades generated with current settings.")

    # ========== PREDICTION TABLE ==========
    st.subheader("📄 Sample predictions (last 30 rows)")
    st.dataframe(
        df_bt[["timestamp_1h", "close_1h", "predicted_return_1h",
               "predicted_next_close_1h", "signal"]].tail(30)
    )

    # ========== GRAPHS ==========
    st.subheader("📉 Price with BUY / SELL markers")
    if len(trades_df) > 0:
        plot_price_with_trades(df_bt, trades_df)
    else:
        st.write("No trades to plot.")

    st.subheader("📈 Strategy Equity Curve")
    plot_equity_curve(df_bt)

    # simple line chart for prediction vs price (last 200 bars)
    st.subheader("Close vs Predicted Next 1H Close (last 200 bars)")
    chart_df = df_bt[["close_1h", "predicted_next_close_1h"]].tail(200)
    st.line_chart(chart_df)

    # ========== DOWNLOADS ==========
    st.subheader("📥 Download data")

    csv_preds = df_bt.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download full predictions + signals CSV",
        data=csv_preds,
        file_name="SBI_predictions_signals_1h_multi_tf.csv",
        mime="text/csv"
    )

    if len(trades_df) > 0:
        csv_trades = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download trade list CSV",
            data=csv_trades,
            file_name="SBI_trades_1h_multi_tf.csv",
            mime="text/csv"
        )
