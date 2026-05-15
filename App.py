import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import ta as ta_lib
from transformers import pipeline
from io import BytesIO

st.set_page_config(page_title="AI Stock Sentinel", layout="wide", page_icon="🚀")
st.title("🚀 AI Stock Sentinel")

# ── KONFIGŪRACIJA ──────────────────────────────────────────
FUTURES_CONFIG = {
    "🇺🇸 US Futures": {
        "S&P 500":    "ES=F",
        "NASDAQ 100": "NQ=F",
        "Dow Jones":  "YM=F",
        "Russell 2K": "RTY=F",
    },
    "🇪🇺 EU Futures": {
        "DAX":        "FDAX=F",
        "Euro Stoxx": "VGM=F",
        "FTSE 100":   "Z=F",
        "CAC 40":     "FCE=F",
    },
    "📦 Macro": {
        "Gold":       "GC=F",
        "Oil (WTI)":  "CL=F",
        "VIX":        "^VIX",
        "USD/EUR":    "EURUSD=X",
    },
}

DEFAULT_TICKERS = ["NVDA", "MSTR", "AAPL", "TSLA", "SMH", "MSFT", "BTC-USD", "AMD"]

# ── MODELIS ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Kraunamas AI modelis...")
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    )

# ── TECHNINĖ ANALIZĖ ───────────────────────────────────────
def get_technical_analysis(df: pd.DataFrame):
    df = df.copy()
    if len(df) < 50:
        return "⚪ INSUFFICIENT DATA", [], None, df

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # RSI
    df["RSI"] = ta_lib.momentum.RSIIndicator(df["Close"], window=14).rsi()

    # Bollinger Bands
    bb = ta_lib.volatility.BollingerBands(df["Close"], window=20)
    df["BBL_20_2.0"] = bb.bollinger_lband()
    df["BBU_20_2.0"] = bb.bollinger_hband()

    # MACD
    macd = ta_lib.trend.MACD(df["Close"])
    df["MACD_12_26_9"]  = macd.macd()
    df["MACDs_12_26_9"] = macd.macd_signal()

    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    signals, score = [], 0

    if latest["SMA20"] > latest["SMA50"]:
        signals.append("🟢 SMA Bullish (SMA20 > SMA50)")
        score += 2
    else:
        signals.append("🔴 SMA Bearish (SMA20 < SMA50)")
        score -= 2

    rsi = latest["RSI"]
    if rsi < 30:
        signals.append(f"🟢 RSI Oversold ({rsi:.1f})")
        score += 2
    elif rsi > 70:
        signals.append(f"🔴 RSI Overbought ({rsi:.1f})")
        score -= 2
    else:
        signals.append(f"⚪ RSI Neutral ({rsi:.1f})")

    if latest["MACD_12_26_9"] > latest["MACDs_12_26_9"] and prev["MACD_12_26_9"] <= prev["MACDs_12_26_9"]:
        signals.append("🟢 MACD Bull Crossover")
        score += 3
    elif latest["MACD_12_26_9"] < latest["MACDs_12_26_9"] and prev["MACD_12_26_9"] >= prev["MACDs_12_26_9"]:
        signals.append("🔴 MACD Bear Crossover")
        score -= 3

    if latest["Close"] < latest["BBL_20_2.0"]:
        signals.append("🟢 Below Lower Bollinger Band")
        score += 2
    elif latest["Close"] > latest["BBU_20_2.0"]:
        signals.append("🔴 Above Upper Bollinger Band")
        score -= 1

    if score >= 5:      overall = "🟢 STRONG BULLISH"
    elif score >= 2:    overall = "🟡 BULLISH"
    elif score <= -5:   overall = "🔴 STRONG BEARISH"
    elif score <= -2:   overall = "🟠 BEARISH"
    else:               overall = "⚪ NEUTRAL"

    return overall, signals, latest, df

# ── FUTURES KORTELĖ ────────────────────────────────────────
def render_futures_card(name: str, ticker: str):
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="5m")
        if len(hist) < 2:
            st.warning(f"Nėra duomenų: {name}")
            return

        price      = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2]
        chg_pct    = (price - prev_close) / prev_close * 100
        chg_abs    = price - prev_close

        overall, signals, latest, _ = get_technical_analysis(
            yf.Ticker(ticker).history(period="1mo")
        )

        color = "🟢" if chg_pct >= 0 else "🔴"
        st.markdown(f"**{color} {name}**")
        st.markdown(
            f"`{price:,.2f}` &nbsp; "
            f"{'▲' if chg_pct >= 0 else '▼'} {abs(chg_pct):.2f}% "
            f"({chg_abs:+.2f})"
        )
        st.caption(overall)

        # Mini sparkline
        fig = go.Figure(go.Scatter(
            x=hist.index[-60:], y=hist["Close"].iloc[-60:],
            mode="lines",
            line=dict(color="#26a69a" if chg_pct >= 0 else "#ef5350", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(38,166,154,0.1)" if chg_pct >= 0 else "rgba(239,83,80,0.1)",
        ))
        fig.update_layout(
            height=80, margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"spark_{ticker}")

    except Exception as e:
        st.error(f"{name}: {e}")

# ── FUTURES SKYRIUS ────────────────────────────────────────
def render_futures_section():
    for group_name, items in FUTURES_CONFIG.items():
        st.markdown(f"#### {group_name}")
        cols = st.columns(len(items))
        for col, (name, ticker) in zip(cols, items.items()):
            with col:
                render_futures_card(name, ticker)
        st.divider()

# ── EXCEL EKSPORTAS ────────────────────────────────────────
def to_excel_bytes(hist: pd.DataFrame, signals: list) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        hist.to_excel(writer, sheet_name="Price Data")
        pd.DataFrame({"Signal": signals}).to_excel(writer, sheet_name="Signals")
    return buf.getvalue()

# ══════════════════════════════════════════════════════════
# PAGRINDINIS UI
# ══════════════════════════════════════════════════════════

tabs = st.tabs(["📈 Akcijos", "🌍 Futures", "📊 Detali Analizė"])

# ── TAB 1: AKCIJOS ─────────────────────────────────────────
with tabs[0]:
    tickers = st.multiselect(
        "Pasirink akcijas / aktyvus",
        DEFAULT_TICKERS,
        default=["NVDA", "MSTR"],
    )
    refresh_rate = st.slider("Atnaujinimo intervalas (sek)", 5, 60, 15)

    if tickers:
        cols = st.columns(len(tickers))
        for idx, ticker in enumerate(tickers):
            with cols[idx]:
                try:
                    hist = yf.Ticker(ticker).history(period="5d", interval="5m")
                    if len(hist) < 2:
                        st.warning(f"Nepakankamai duomenų: {ticker}")
                        continue

                    price      = hist["Close"].iloc[-1]
                    chg_pct    = (price - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100
                    overall, signals, latest, _ = get_technical_analysis(hist)

                    st.subheader(f"{ticker}")
                    st.metric("Kaina", f"${price:.2f}", f"{chg_pct:+.2f}%")
                    st.write(f"**Signalas:** {overall}")

                    fig = go.Figure(go.Candlestick(
                        x=hist.index, open=hist["Open"], high=hist["High"],
                        low=hist["Low"], close=hist["Close"],
                    ))
                    fig.update_layout(height=220, margin=dict(l=5,r=5,t=5,b=5),
                                      showlegend=False, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True, key=f"mini_{ticker}_{idx}")

                    if st.button(f"🔍 Analizuoti {ticker}", key=f"btn_{ticker}_{idx}"):
                        st.session_state.selected = ticker

                    if "BULLISH" in overall:
                        st.success("🚨 Bullish signalas!")

                except Exception as e:
                    st.error(f"Klaida {ticker}: {e}")

    # Auto-refresh
    st.caption(f"⏱ Atnaujinama kas {refresh_rate}s")
    import time; time.sleep(refresh_rate)
    st.rerun()

# ── TAB 2: FUTURES ─────────────────────────────────────────
with tabs[1]:
    st.subheader("🌍 Real-Time Futures Stebėjimas")
    if st.button("🔄 Atnaujinti Futures"):
        st.rerun()
    render_futures_section()

# ── TAB 3: DETALI ANALIZĖ ──────────────────────────────────
with tabs[2]:
    selected = st.session_state.get("selected", DEFAULT_TICKERS[0])
    chosen   = st.selectbox("Pasirink aktyvą analizei", DEFAULT_TICKERS, 
                            index=DEFAULT_TICKERS.index(selected) if selected in DEFAULT_TICKERS else 0)

    stock = yf.Ticker(chosen)
    hist  = stock.history(period="3mo")
    overall, signals, latest, df_ta = get_technical_analysis(hist)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📊 {chosen} — Techninė Analizė")
        st.write(f"**Bendras signalas:** {overall}")

        # Pilnas grafikas su indikatoriais + MACD subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.03)

        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"], name="Price"), row=1, col=1)

        for col_name, color, label in [
            ("SMA20", "orange", "SMA 20"),
            ("SMA50", "royalblue", "SMA 50"),
            ("BBL_20_2.0", "gray", "BB Lower"),
            ("BBU_20_2.0", "gray", "BB Upper"),
        ]:
            if col_name in df_ta.columns:
                fig.add_trace(go.Scatter(
                    x=df_ta.index, y=df_ta[col_name],
                    name=label, line=dict(color=color, width=1, dash="dot" if "BB" in col_name else "solid"),
                ), row=1, col=1)

        if "MACD_12_26_9" in df_ta.columns:
            fig.add_trace(go.Scatter(
                x=df_ta.index, y=df_ta["MACD_12_26_9"], name="MACD",
                line=dict(color="cyan", width=1.2)), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df_ta.index, y=df_ta["MACDs_12_26_9"], name="Signal",
                line=dict(color="orange", width=1.2)), row=2, col=1)

        fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📡 Signalai")
        for sig in signals:
            st.write(sig)

        # RSI gauge
        if latest is not None and "RSI" in df_ta.columns:
            rsi_val = latest["RSI"]
            fig_rsi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rsi_val,
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color="darkblue"),
                    steps=[
                        dict(range=[0, 30], color="green"),
                        dict(range=[30, 70], color="lightgray"),
                        dict(range=[70, 100], color="red"),
                    ],
                    threshold=dict(line=dict(color="white", width=2), value=rsi_val),
                ),
                title=dict(text="RSI"),
            ))
            fig_rsi.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_rsi, use_container_width=True)

        # Excel eksportas (teisingas būdas)
        excel_bytes = to_excel_bytes(hist, signals)
        st.download_button(
            label="📥 Atsisiųsti Excel ataskaitą",
            data=excel_bytes,
            file_name=f"{chosen}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Naujienos + Sentiment
    st.divider()
    st.subheader("📰 Naujienos + AI Sentiment")
    try:
        sentiment_analyzer = load_sentiment_model()
        news = stock.news[:8]
        for item in news:
            content = item.get("content", {})
            title   = content.get("title") or item.get("title", "")
            link    = content.get("canonicalUrl", {}).get("url") or item.get("link", "#")
            publisher = content.get("provider", {}).get("displayName") or item.get("publisher", "")

            if not title:
                continue

            sentiment = sentiment_analyzer(title[:512])[0]
            label     = sentiment["label"].lower()
            score     = sentiment["score"]

            # Teisingas label tikrinimas
            is_bullish = label == "positive" and score > 0.6
            is_bearish = label == "negative" and score > 0.6

            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**[{title}]({link})**  \n*{publisher}*")
            with c2:
                if is_bullish:
                    st.success("🟢")
                elif is_bearish:
                    st.error("🔴")
                else:
                    st.info("⚪")
    except Exception as e:
        st.error(f"Naujienos nepavyko gauti: {e}")
