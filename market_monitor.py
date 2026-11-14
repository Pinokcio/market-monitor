
# -*- coding: utf-8 -*-
"""
US Market Monitor V3
- Daily US market diagnosis
- CNN Fear & Greed Index
- Indicator scores and explicit signal thresholds
- Short-term changes: 5 / 20 / 60 trading days
- Expandable explanations
- QQQ / SPY / SCHD relative strength
- Market risk and signal conflict detection

Disclaimer:
Scores are heuristic decision-support indicators.
They are NOT statistically calibrated probabilities or automatic buy/sell signals.
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# 1. APP CONFIG
# ============================================================

st.set_page_config(
    page_title="US Market Monitor V3",
    page_icon="📊",
    layout="wide"
)

st.title("📊 US Market Monitor V3")
st.caption(
    "미국 시장의 추세, 시장 폭, 변동성, 신용시장, 금리, "
    "공포·탐욕을 매일 점검하는 투자 의사결정 보조 도구"
)

st.warning(
    "이 프로그램의 점수는 설명 가능한 규칙 기반 참고 점수입니다. "
    "미래 수익률의 확률이나 검증된 매매 신호가 아닙니다. "
    "실제 투자 전에는 가격, 금리, 이벤트 및 데이터의 최신성을 함께 확인하세요."
)


# ============================================================
# 2. CONSTANTS
# ============================================================

TICKERS = [
    "SPY", "RSP", "QQQ", "SCHD", "BIL",
    "HYG", "IEF", "^VIX", "^TNX", "^IRX"
]

CNN_FNG_URL = (
    "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
)

CNN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://edition.cnn.com/markets/fear-and-greed",
    "Origin": "https://edition.cnn.com"
}

PERIOD_OPTIONS = {
    "2년": "2y",
    "5년": "5y",
    "10년": "10y"
}


# ============================================================
# 3. DATA DOWNLOAD
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def download_market_data(period="5y"):
    """
    Yahoo Finance adjusted daily close data.
    """

    data = {}

    for ticker in TICKERS:
        try:
            hist = yf.Ticker(ticker).history(
                period=period,
                interval="1d",
                auto_adjust=True,
                actions=False
            )

            if hist.empty or "Close" not in hist.columns:
                continue

            series = hist["Close"].copy()

            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)

            series = series[~series.index.duplicated(keep="last")]
            series = pd.to_numeric(series, errors="coerce")
            data[ticker] = series

        except Exception:
            continue

    if not data:
        return pd.DataFrame()

    close = pd.DataFrame(data).sort_index()

    # 제한적인 결측치 보정
    close = close.ffill(limit=3)

    return close

def cnn_timestamp_to_et(value):
    """
    CNN timestamp를 미국 동부시간(America/New_York)으로 변환.
    Unix seconds / milliseconds 및 ISO 문자열 지원.
    """

    if value is None:
        return pd.NaT

    try:
        # Unix timestamp
        if isinstance(value, (int, float, np.integer, np.floating)):
            if pd.isna(value):
                return pd.NaT

            unit = "ms" if abs(value) > 1e11 else "s"

            ts = pd.to_datetime(
                value,
                unit=unit,
                utc=True,
                errors="coerce"
            )

        else:
            value = str(value).strip()

            # 날짜만 있는 문자열은 날짜가 바뀌지 않도록 처리
            if len(value) == 10 and value[4] == "-" and value[7] == "-":
                ts = pd.Timestamp(value).tz_localize(
                    "America/New_York"
                )
            else:
                ts = pd.to_datetime(
                    value,
                    utc=True,
                    errors="coerce"
                )

        if pd.isna(ts):
            return pd.NaT

        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        return ts.tz_convert("America/New_York")

    except Exception:
        return pd.NaT


@st.cache_data(ttl=900, show_spinner=False)
def download_fear_greed():

    response = requests.get(
        CNN_FNG_URL,
        headers=CNN_HEADERS,
        timeout=15
    )

    response.raise_for_status()

    raw = response.json()

    current = raw.get("fear_and_greed", {})

    if not current or "score" not in current:
        raise ValueError("CNN 응답에서 Fear & Greed 점수를 찾지 못했습니다.")

    current_timestamp = cnn_timestamp_to_et(
        current.get("timestamp")
    )

    result = {
        "score": float(current["score"]),
        "rating": current.get("rating", "unknown"),
        "timestamp": current_timestamp,
        "previous_close": current.get("previous_close"),
        "previous_1_week": current.get("previous_1_week"),
        "previous_1_month": current.get("previous_1_month"),
        "previous_1_year": current.get("previous_1_year")
    }

    history_data = (
        raw.get("fear_and_greed_historical", {}).get("data", [])
    )

    history = pd.DataFrame()

    if history_data:

        history = pd.DataFrame(history_data)

        if "x" in history.columns and "y" in history.columns:

            history["date"] = history["x"].apply(
                cnn_timestamp_to_et
            )

            history["score"] = pd.to_numeric(
                history["y"],
                errors="coerce"
            )

            history = history.dropna(
                subset=["date", "score"]
            )

            # 미국 동부시간 기준 날짜만 사용
            history["date"] = history["date"].apply(
                lambda x: x.date()
            )

            history = history.drop_duplicates(
                subset=["date"],
                keep="last"
            )

            history = history.set_index("date").sort_index()

            history.index = pd.to_datetime(history.index)

    return result, history

@st.cache_data(ttl=900, show_spinner=False)
def download_fear_greed():
    """
    CNN Fear & Greed Index.
    CNN's endpoint is unofficial and may change.
    """

    response = requests.get(
        CNN_FNG_URL,
        headers=CNN_HEADERS,
        timeout=15
    )

    response.raise_for_status()

    raw = response.json()

    current = raw.get("fear_and_greed", {})

    if not current or "score" not in current:
        raise ValueError("CNN 응답에서 Fear & Greed 점수를 찾지 못했습니다.")
    

    current_timestamp = cnn_timestamp_to_et(
        current.get("timestamp")
    )

    result = {
        "score": float(current["score"]),
        "rating": current.get("rating", "unknown"),
        "timestamp": current_timestamp,
        "previous_close": current.get("previous_close"),
        "previous_1_week": current.get("previous_1_week"),
        "previous_1_month": current.get("previous_1_month"),
        "previous_1_year": current.get("previous_1_year")
    }

    history_data = raw.get("fear_and_greed_historical", {}).get("data", [])

    history = pd.DataFrame()

    if history_data:
        history = pd.DataFrame(history_data)

        if "x" in history.columns and "y" in history.columns:
            history["date"] = pd.to_datetime(
                history["x"],
                unit="s",
                errors="coerce"
            )
            history["score"] = pd.to_numeric(
                history["y"],
                errors="coerce"
            )
            history = history.dropna(subset=["date", "score"])
            history = history.set_index("date").sort_index()

    return result, history


# ============================================================
# 4. COMMON FUNCTIONS
# ============================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def get_latest(series):
    if series is None:
        return np.nan

    s = series.dropna()

    if len(s) == 0:
        return np.nan

    return float(s.iloc[-1])


def change_pct(series, periods):
    """
    Percentage change over trading days.
    """

    s = series.dropna()

    if len(s) <= periods:
        return np.nan

    old = float(s.iloc[-periods - 1])
    new = float(s.iloc[-1])

    if old == 0:
        return np.nan

    return (new / old - 1) * 100


def change_abs(series, periods):
    """
    Absolute change over trading days.
    """

    s = series.dropna()

    if len(s) <= periods:
        return np.nan

    return float(s.iloc[-1] - s.iloc[-periods - 1])


def format_pct(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:+.2f}%"


def format_number(value, digits=2):
    if pd.isna(value):
        return "N/A"

    return f"{value:,.{digits}f}"


def format_score(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:.0f} / 100"


def trend_description(series):
    """
    Interpret short, medium, and longer-term direction.
    """

    c5 = change_pct(series, 5)
    c20 = change_pct(series, 20)
    c60 = change_pct(series, 60)

    if any(pd.isna(x) for x in [c5, c20, c60]):
        return "데이터 부족"

    if c5 > 0 and c20 > 0 and c60 > 0:
        return "상승 지속"

    if c5 < 0 and c20 < 0 and c60 < 0:
        return "하락 지속"

    if c5 > 0 and c20 < 0 and c60 < 0:
        return "단기 반등 시도"

    if c5 < 0 and c20 > 0 and c60 > 0:
        return "중기 상승 중 단기 조정"

    if c5 > 0 and c20 > 0 and c60 < 0:
        return "중기 회복 시도"

    if c5 < 0 and c20 < 0 and c60 > 0:
        return "중기 상승세 둔화"

    return "기간별 방향 혼조"


def score_label(score):
    if pd.isna(score):
        return "데이터 부족"

    if score >= 80:
        return "매우 양호"

    if score >= 65:
        return "양호"

    if score >= 55:
        return "다소 양호"

    if score >= 45:
        return "중립"

    if score >= 35:
        return "주의"

    if score >= 20:
        return "위험"

    return "매우 위험"


# ============================================================
# 5. INDICATOR CALCULATIONS
# ============================================================

def calculate_features(close):

    f = pd.DataFrame(index=close.index)

    # --------------------------------------------------------
    # SPY trend
    # --------------------------------------------------------

    spy = close["SPY"]

    f["SPY"] = spy
    f["SPY_MA50"] = spy.rolling(50).mean()
    f["SPY_MA200"] = spy.rolling(200).mean()

    f["SPY_MA50_SLOPE20"] = (
        f["SPY_MA50"] / f["SPY_MA50"].shift(20) - 1
    ) * 100

    f["SPY_RET5"] = spy.pct_change(5) * 100
    f["SPY_RET20"] = spy.pct_change(20) * 100
    f["SPY_RET60"] = spy.pct_change(60) * 100

    # --------------------------------------------------------
    # Breadth proxy: RSP / SPY
    # --------------------------------------------------------

    f["RSP_SPY"] = close["RSP"] / close["SPY"]
    f["RSP_SPY_MA50"] = f["RSP_SPY"].rolling(50).mean()

    f["BREADTH_RET5"] = f["RSP_SPY"].pct_change(5) * 100
    f["BREADTH_RET20"] = f["RSP_SPY"].pct_change(20) * 100
    f["BREADTH_RET60"] = f["RSP_SPY"].pct_change(60) * 100

    f["RSP_RET20"] = close["RSP"].pct_change(20) * 100
    f["SPY_RET20_RSP"] = close["SPY"].pct_change(20) * 100

    # --------------------------------------------------------
    # VIX
    # --------------------------------------------------------

    f["VIX"] = close["^VIX"]
    f["VIX_RET5"] = f["VIX"].pct_change(5) * 100
    f["VIX_RET20"] = f["VIX"].pct_change(20) * 100
    f["VIX_RET60"] = f["VIX"].pct_change(60) * 100

    # --------------------------------------------------------
    # Credit risk appetite proxy: HYG / IEF
    # --------------------------------------------------------

    f["HYG_IEF"] = close["HYG"] / close["IEF"]
    f["HYG_IEF_MA50"] = f["HYG_IEF"].rolling(50).mean()

    f["CREDIT_RET5"] = f["HYG_IEF"].pct_change(5) * 100
    f["CREDIT_RET20"] = f["HYG_IEF"].pct_change(20) * 100
    f["CREDIT_RET60"] = f["HYG_IEF"].pct_change(60) * 100

    # --------------------------------------------------------
    # Interest rates
    # --------------------------------------------------------

    # Yahoo ^TNX / ^IRX values are generally quoted as
    # yield multiplied by 10. Divide by 10 to get percentage.

    f["YIELD10"] = close["^TNX"] / 10
    f["YIELD3M"] = close["^IRX"] / 10

    f["YIELD10_CHG5"] = f["YIELD10"].diff(5)
    f["YIELD10_CHG20"] = f["YIELD10"].diff(20)
    f["YIELD10_CHG60"] = f["YIELD10"].diff(60)

    f["YIELD_SPREAD_10Y_3M"] = f["YIELD10"] - f["YIELD3M"]

    # --------------------------------------------------------
    # ETF relative strength
    # --------------------------------------------------------

    for ticker in ["QQQ", "SPY", "SCHD"]:
        f[f"{ticker}_RET5"] = close[ticker].pct_change(5) * 100
        f[f"{ticker}_RET20"] = close[ticker].pct_change(20) * 100
        f[f"{ticker}_RET60"] = close[ticker].pct_change(60) * 100
        f[f"{ticker}_MA50"] = close[ticker].rolling(50).mean()

    return f


# ============================================================
# 6. INDICATOR SCORING
# ============================================================

def score_market_trend(row):
    """
    SPY trend score: 0-100.

    Components:
    - Price above MA50: 25
    - Price above MA200: 30
    - MA50 above MA200: 25
    - Positive MA50 slope: 20
    """

    required = [
        row["SPY"],
        row["SPY_MA50"],
        row["SPY_MA200"],
        row["SPY_MA50_SLOPE20"]
    ]

    if any(pd.isna(x) for x in required):
        return np.nan

    score = 0

    if row["SPY"] > row["SPY_MA50"]:
        score += 25

    if row["SPY"] > row["SPY_MA200"]:
        score += 30

    if row["SPY_MA50"] > row["SPY_MA200"]:
        score += 25

    if row["SPY_MA50_SLOPE20"] > 0:
        score += 20

    return score


def score_breadth(row):
    """
    RSP/SPY relative strength proxy.

    This is NOT the actual advance-decline line
    or the percentage of stocks above moving averages.
    """

    ratio = row["RSP_SPY"]
    ma = row["RSP_SPY_MA50"]
    r20 = row["BREADTH_RET20"]
    r60 = row["BREADTH_RET60"]

    if any(pd.isna(x) for x in [ratio, ma, r20, r60]):
        return np.nan

    score = 50

    if ratio > ma:
        score += 20
    else:
        score -= 20

    if r20 > 0:
        score += 15
    else:
        score -= 15

    if r60 > 0:
        score += 15
    else:
        score -= 15

    return float(np.clip(score, 0, 100))


def score_vix(vix):
    """
    Lower VIX generally means lower market stress.
    This is a stability score, not a buy signal.
    """

    if pd.isna(vix):
        return np.nan

    if vix < 15:
        return 95
    elif vix < 18:
        return 85
    elif vix < 20:
        return 75
    elif vix < 25:
        return 55
    elif vix < 30:
        return 35
    elif vix < 40:
        return 15
    else:
        return 5


def score_credit(row):
    """
    HYG / IEF relative strength.
    """

    ratio = row["HYG_IEF"]
    ma = row["HYG_IEF_MA50"]
    r20 = row["CREDIT_RET20"]
    r60 = row["CREDIT_RET60"]

    if any(pd.isna(x) for x in [ratio, ma, r20, r60]):
        return np.nan

    score = 50

    if ratio > ma:
        score += 20
    else:
        score -= 20

    if r20 > 0:
        score += 15
    else:
        score -= 15

    if r60 > 0:
        score += 15
    else:
        score -= 15

    return float(np.clip(score, 0, 100))


def score_rates(change20):
    """
    10Y yield change over 20 trading days.

    Falling yields can reduce valuation pressure,
    but the reason for the decline matters.
    """

    if pd.isna(change20):
        return np.nan

    if change20 <= -0.30:
        return 85
    elif change20 <= -0.15:
        return 75
    elif change20 <= -0.05:
        return 65
    elif change20 <= 0.05:
        return 55
    elif change20 <= 0.15:
        return 45
    elif change20 <= 0.30:
        return 30
    else:
        return 15


def score_fear_greed(score):
    """
    CNN Fear & Greed is a sentiment gauge.
    This function converts it into a sentiment condition,
    NOT a direct bullish/bearish probability.
    """

    if pd.isna(score):
        return np.nan

    if score < 25:
        return 20
    elif score < 45:
        return 40
    elif score <= 55:
        return 60
    elif score <= 75:
        return 75
    else:
        return 90


def fear_greed_label(score):
    if pd.isna(score):
        return "데이터 부족"

    if score < 25:
        return "극단적 공포"
    elif score < 45:
        return "공포"
    elif score <= 55:
        return "중립"
    elif score <= 75:
        return "탐욕"
    else:
        return "극단적 탐욕"


# ============================================================
# 7. MARKET DIAGNOSIS
# ============================================================

def build_market_diagnosis(row, fear_greed_score):

    trend = score_market_trend(row)
    breadth = score_breadth(row)
    vix = score_vix(row["VIX"])
    credit = score_credit(row)
    rates = score_rates(row["YIELD10_CHG20"])

    components = {
        "시장 추세": (trend, 0.30),
        "시장 폭 대용": (breadth, 0.20),
        "VIX 안정도": (vix, 0.20),
        "신용시장": (credit, 0.20),
        "금리 환경": (rates, 0.10)
    }

    valid = {
        name: (score, weight)
        for name, (score, weight) in components.items()
        if not pd.isna(score)
    }

    if len(valid) < 3:
        market_score = np.nan
    else:
        total_weight = sum(weight for _, weight in valid.values())

        market_score = sum(
            score * weight
            for score, weight in valid.values()
        ) / total_weight

    if pd.isna(market_score):
        regime = "데이터 부족"
    elif market_score >= 75:
        regime = "위험자산 우호적"
    elif market_score >= 55:
        regime = "중립적 환경"
    elif market_score >= 35:
        regime = "위험 증가"
    else:
        regime = "방어적 환경"

    return {
        "market_score": market_score,
        "regime": regime,
        "components": components,
        "fear_greed_score": fear_greed_score
    }


# ============================================================
# 8. ETF RELATIVE STRENGTH
# ============================================================

def calculate_asset_strength(row):

    results = {}

    for ticker in ["QQQ", "SPY", "SCHD"]:

        r5 = row.get(f"{ticker}_RET5", np.nan)
        r20 = row.get(f"{ticker}_RET20", np.nan)
        r60 = row.get(f"{ticker}_RET60", np.nan)

        if any(pd.isna(x) for x in [r5, r20, r60]):
            strength = np.nan
        else:
            strength = (
                r5 * 0.20 +
                r20 * 0.45 +
                r60 * 0.35
            )

        results[ticker] = {
            "5일": r5,
            "20일": r20,
            "60일": r60,
            "강도 점수": strength
        }

    return results


# ============================================================
# 9. SIGNAL CONFLICTS
# ============================================================

def detect_conflicts(row, diagnosis, fear_greed_score):

    conflicts = []

    trend = diagnosis["components"]["시장 추세"][0]
    breadth = diagnosis["components"]["시장 폭 대용"][0]
    vix_score = diagnosis["components"]["VIX 안정도"][0]
    credit = diagnosis["components"]["신용시장"][0]

    qqq20 = row["QQQ_RET20"]
    spy20 = row["SPY_RET20"]
    yield20 = row["YIELD10_CHG20"]

    if not pd.isna(trend) and not pd.isna(breadth):
        if trend >= 70 and breadth < 45:
            conflicts.append(
                "SPY 추세는 양호하지만 시장 폭 대용 지표가 약합니다. "
                "상승이 일부 대형주에 집중되고 있을 가능성이 있습니다."
            )

    if not pd.isna(trend) and not pd.isna(vix_score):
        if trend >= 65 and vix_score < 40:
            conflicts.append(
                "가격 추세는 양호하지만 VIX가 높습니다. "
                "시장 변동성 위험이 추세를 뒷받침하지 못하고 있습니다."
            )

    if not pd.isna(trend) and not pd.isna(credit):
        if trend >= 65 and credit < 40:
            conflicts.append(
                "주식시장 추세와 신용시장 위험선호가 엇갈립니다."
            )

    if not pd.isna(qqq20) and not pd.isna(yield20):
        if qqq20 > 0 and yield20 >= 0.20:
            conflicts.append(
                "QQQ가 상승하는 가운데 10년물 금리가 빠르게 상승했습니다. "
                "성장주 밸류에이션 부담이 커질 수 있습니다."
            )

    if not pd.isna(fear_greed_score):
        if fear_greed_score >= 80:
            conflicts.append(
                "공포·탐욕 지수가 극단적 탐욕 구간입니다. "
                "시장 분위기가 강하더라도 추격 매수 위험을 점검해야 합니다."
            )

        elif fear_greed_score <= 20:
            conflicts.append(
                "공포·탐욕 지수가 극단적 공포 구간입니다. "
                "가격 하락과 변동성이 동반되는지 확인해야 합니다. "
                "공포만으로 저점 매수 신호가 되지는 않습니다."
            )

    return conflicts


# ============================================================
# 10. UI HELPERS
# ============================================================

def show_metric_card(title, value, delta=None, help_text=None):

    if delta is None:
        st.metric(title, value, help=help_text)
    else:
        st.metric(title, value, delta=delta, help=help_text)


def show_period_changes(series, inverse=False):

    c5 = change_pct(series, 5)
    c20 = change_pct(series, 20)
    c60 = change_pct(series, 60)

    if inverse:
        c5 = -c5 if not pd.isna(c5) else np.nan
        c20 = -c20 if not pd.isna(c20) else np.nan
        c60 = -c60 if not pd.isna(c60) else np.nan

    a, b, c = st.columns(3)

    a.metric("5거래일 변화", format_pct(c5))
    b.metric("20거래일 변화", format_pct(c20))
    c.metric("60거래일 변화", format_pct(c60))


def show_scorecard(title, score, signal, explanation, thresholds):

    st.subheader(title)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.metric("점수", format_score(score))

        if not pd.isna(score):
            st.progress(int(np.clip(score, 0, 100)))

    with c2:
        st.markdown(f"**현재 신호: {signal}**")
        st.write(explanation)

    with st.expander("지표 설명 / 점수 기준 / 투자 판단 방법"):

        st.markdown("**점수 기준**")
        st.markdown(thresholds)

        st.markdown("**이 지표를 보는 이유**")
        st.write(explanation)

    st.divider()


# ============================================================
# 11. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    st.sidebar.header("설정")

    period_label = st.sidebar.selectbox(
        "데이터 조회 기간",
        list(PERIOD_OPTIONS.keys()),
        index=1
    )

    period = PERIOD_OPTIONS[period_label]

    if st.sidebar.button("데이터 새로고침"):
        download_market_data.clear()
        download_fear_greed.clear()
        st.rerun()

    st.sidebar.caption("데이터 캐시: 15분")

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    with st.spinner("시장 데이터를 불러오는 중입니다..."):
        close = download_market_data(period)

    if close.empty:
        st.error(
            "시장 데이터를 가져오지 못했습니다. "
            "인터넷 연결 또는 Yahoo Finance 응답을 확인하세요."
        )
        return

    required = ["SPY", "RSP", "QQQ", "SCHD", "HYG", "IEF",
                "^VIX", "^TNX", "^IRX"]

    missing = [x for x in required if x not in close.columns]

    if missing:
        st.error(f"필수 데이터가 없습니다: {', '.join(missing)}")
        return

    close = close.dropna(subset=["SPY"])

    features = calculate_features(close)

    latest = features.dropna(subset=["SPY"]).iloc[-1]
    signal_date = features.dropna(subset=["SPY"]).index[-1]

    # --------------------------------------------------------
    # Fear & Greed
    # --------------------------------------------------------

    fng = None
    fng_history = pd.DataFrame()
    fng_error = None

    try:
        fng, fng_history = download_fear_greed()
    except Exception as e:
        fng_error = str(e)

    fng_score = np.nan

    if fng:
        fng_score = fng["score"]

    # --------------------------------------------------------
    # Diagnosis
    # --------------------------------------------------------

    diagnosis = build_market_diagnosis(latest, fng_score)
    strengths = calculate_asset_strength(latest)
    conflicts = detect_conflicts(latest, diagnosis, fng_score)

    market_score = diagnosis["market_score"]

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.caption(
        f"시장 데이터 기준일: {signal_date.strftime('%Y-%m-%d')} "
        f"| 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # --------------------------------------------------------
    # Overall diagnosis
    # --------------------------------------------------------

    st.header("1. 종합 시장 진단")

    c1, c2, c3 = st.columns(3)

    with c1:
        show_metric_card(
            "시장 환경 점수",
            format_score(market_score)
        )

    with c2:
        show_metric_card(
            "시장 국면",
            diagnosis["regime"]
        )

    with c3:
        show_metric_card(
            "CNN 공포·탐욕",
            (
                f"{fng_score:.1f} / 100"
                if not pd.isna(fng_score)
                else "데이터 없음"
            )
        )

    with st.expander("종합 점수 산출 방식"):

        st.markdown(
            """
            종합 점수는 아래 5개 시장 환경 요소를 가중 평균한 값입니다.

            | 구성 요소 | 가중치 |
            |---|---:|
            | SPY 추세 | 30% |
            | 시장 폭 대용 | 20% |
            | VIX 안정도 | 20% |
            | 신용시장 | 20% |
            | 금리 환경 | 10% |

            CNN 공포·탐욕 지수는 종합 점수에 직접 합산하지 않습니다.
            시장의 심리적 과열과 공포를 별도 경고로 확인합니다.

            데이터가 없는 항목은 계산에서 제외하고,
            나머지 가중치를 재조정합니다.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # Fear & Greed section
    # --------------------------------------------------------

    st.header("2. 공포·탐욕 지수")

    if fng:

        fg_label = fear_greed_label(fng_score)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "현재 지수",
                f"{fng_score:.1f}",
                delta=(
                    f"{fng_score - float(fng['previous_close']):+.1f} "
                    "전일 대비"
                    if fng.get("previous_close") is not None
                    else None
                )
            )

        with c2:
            st.metric("심리 상태", fg_label)

        with c3:
            st.metric(
                "1주 전 대비",
                (
                    f"{fng_score - float(fng['previous_1_week']):+.1f}"
                    if fng.get("previous_1_week") is not None
                    else "N/A"
                )
            )

        st.progress(int(np.clip(fng_score, 0, 100)))

        if fng_score < 25:
            st.warning(
                "극단적 공포: 시장 참여자들의 위험 회피 심리가 강합니다. "
                "하락 추세와 신용시장 악화가 동반되는지 확인하세요."
            )
        elif fng_score < 45:
            st.warning(
                "공포: 투자 심리가 위축된 상태입니다. "
                "반등 여부보다 가격 추세와 시장 폭 회복을 먼저 확인하세요."
            )
        elif fng_score <= 55:
            st.info("중립: 시장 심리가 한쪽으로 크게 기울지 않은 상태입니다.")
        elif fng_score <= 75:
            st.success(
                "탐욕: 위험자산 선호가 강한 상태입니다. "
                "추세와 시장 폭이 뒷받침되는지 확인하세요."
            )
        else:
            st.warning(
                "극단적 탐욕: 낙관 심리가 강합니다. "
                "과열 및 추격 매수 위험을 점검하세요."
            )

        with st.expander("공포·탐욕 지수 설명 및 판단 기준"):

            st.markdown(
                """
                CNN Fear & Greed Index는 여러 시장 데이터를 조합해
                미국 주식시장의 투자 심리를 0~100으로 나타냅니다.

                | 점수 | 구간 | 해석 |
                |---:|---|---|
                | 0~24 | 극단적 공포 | 위험 회피 심리 매우 강함 |
                | 25~44 | 공포 | 투자 심리 위축 |
                | 45~55 | 중립 | 심리 균형 |
                | 56~75 | 탐욕 | 위험 선호 강함 |
                | 76~100 | 극단적 탐욕 | 과열 가능성 점검 |

                **주의할 점**

                - 극단적 공포가 곧바로 매수 신호는 아닙니다.
                - 극단적 탐욕이 곧바로 매도 신호도 아닙니다.
                - 가격 추세, VIX, 신용시장과 함께 판단해야 합니다.
                - 심리지수는 가격보다 빠르게 또는 늦게 움직일 수 있습니다.

                CNN 지수는 모멘텀, 주가 강도, 시장 폭, 풋/콜 옵션,
                변동성, 정크본드 수요, 안전자산 수요 등
                여러 요소를 종합합니다.
                """
            )

        if not fng_history.empty:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=fng_history.index,
                    y=fng_history["score"],
                    mode="lines",
                    name="Fear & Greed"
                )
            )

            for y in [25, 45, 55, 75]:
                fig.add_hline(
                    y=y,
                    line_dash="dot",
                    annotation_text=str(y)
                )

            fig.update_layout(
                title="CNN Fear & Greed Index (US Eastern Date)",
                yaxis_title="Score",
                yaxis_range=[0, 100],
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

        fng_date = fng.get("timestamp")

        try:
            if fng_date is None:
                fng_date_text = "확인 불가"
            else:
                fng_date = pd.Timestamp(fng_date)

                if pd.isna(fng_date):
                    fng_date_text = "확인 불가"
                else:
                    fng_date_text = fng_date.strftime("%Y-%m-%d")

        except (TypeError, ValueError, AttributeError):
            fng_date_text = "확인 불가"

        st.caption(
            f"공포·탐욕 지수 기준일: {fng_date_text} (미국 동부시간)"
        )
    else:
        st.warning(
            "CNN 공포·탐욕 지수를 가져오지 못했습니다. "
            "다른 시장 지표 분석은 계속 진행합니다."
        )

        with st.expander("오류 상세"):
            st.code(fng_error or "Unknown error")

    st.divider()

    # --------------------------------------------------------
    # Indicator 1: Trend
    # --------------------------------------------------------

    st.header("3. 핵심 시장 지표")

    trend_score = score_market_trend(latest)

    if trend_score >= 80:
        trend_signal = "강한 상승 추세"
    elif trend_score >= 60:
        trend_signal = "상승 우위"
    elif trend_score >= 40:
        trend_signal = "혼조"
    elif trend_score >= 20:
        trend_signal = "하락 우위"
    else:
        trend_signal = "약한 추세"

    show_scorecard(
        "① SPY 시장 추세",
        trend_score,
        trend_signal,
        (
            f"SPY 현재가 {latest['SPY']:.2f}, "
            f"MA50 {latest['SPY_MA50']:.2f}, "
            f"MA200 {latest['SPY_MA200']:.2f}. "
            f"20일 MA50 기울기 "
            f"{latest['SPY_MA50_SLOPE20']:+.2f}%."
        ),
        """
        | 점수 | 신호 |
        |---:|---|
        | 80~100 | 강한 상승 추세 |
        | 60~79 | 상승 우위 |
        | 40~59 | 혼조 |
        | 20~39 | 하락 우위 |
        | 0~19 | 약한 추세 |

        가격이 MA50과 MA200 위에 있는지,
        MA50이 MA200 위에 있는지,
        MA50의 20일 방향이 상승인지 평가합니다.

        단기 조정만으로 장기 추세가 훼손됐다고 판단하지 말고
        MA50과 MA200의 위치를 함께 확인하세요.
        """
    )

    show_period_changes(close["SPY"])

    st.caption(
        f"5일: {trend_description(close['SPY'])}"
    )

    # --------------------------------------------------------
    # Indicator 2: Breadth
    # --------------------------------------------------------

    breadth_score = score_breadth(latest)

    if breadth_score >= 70:
        breadth_signal = "상승 참여 확산"
    elif breadth_score >= 55:
        breadth_signal = "양호"
    elif breadth_score >= 45:
        breadth_signal = "중립"
    elif breadth_score >= 30:
        breadth_signal = "상승 확산 약화"
    else:
        breadth_signal = "시장 폭 취약"

    show_scorecard(
        "② 시장 폭 대용 지표 (RSP / SPY)",
        breadth_score,
        breadth_signal,
        (
            f"RSP/SPY 비율 {latest['RSP_SPY']:.4f}, "
            f"20일 변화 {latest['BREADTH_RET20']:+.2f}%, "
            f"60일 변화 {latest['BREADTH_RET60']:+.2f}%."
        ),
        """
        | 점수 | 신호 |
        |---:|---|
        | 70~100 | 상승 참여 확산 |
        | 55~69 | 양호 |
        | 45~54 | 중립 |
        | 30~44 | 상승 확산 약화 |
        | 0~29 | 시장 폭 취약 |

        RSP는 S&P500 동일가중 ETF이고 SPY는 시가총액가중 ETF입니다.

        RSP/SPY가 상승하면 상대적으로 중소형 S&P500 구성 종목들의
        기여도가 개선되는 흐름으로 해석할 수 있습니다.

        **중요:** 이 지표는 실제 상승 종목 수나 Advance-Decline Line이
        아닙니다. 시장 폭을 간접적으로 추정하는 대용 지표입니다.
        """
    )

    show_period_changes(features["RSP_SPY"])

    # --------------------------------------------------------
    # Indicator 3: VIX
    # --------------------------------------------------------

    vix_score = score_vix(latest["VIX"])

    vix = latest["VIX"]

    if pd.isna(vix):
        vix_signal = "데이터 부족"
    elif vix < 15:
        vix_signal = "매우 안정"
    elif vix < 20:
        vix_signal = "안정"
    elif vix < 25:
        vix_signal = "주의"
    elif vix < 30:
        vix_signal = "위험 증가"
    else:
        vix_signal = "높은 위험"

    show_scorecard(
        "③ VIX 변동성",
        vix_score,
        vix_signal,
        (
            f"현재 VIX {format_number(vix)}, "
            f"5일 변화 {format_pct(change_pct(close['^VIX'], 5))}, "
            f"20일 변화 {format_pct(change_pct(close['^VIX'], 20))}."
        ),
        """
        | VIX | 안정도 점수 | 신호 |
        |---:|---:|---|
        | 15 미만 | 95 | 매우 안정 |
        | 15~18 | 85 | 안정 |
        | 18~20 | 75 | 대체로 안정 |
        | 20~25 | 55 | 주의 |
        | 25~30 | 35 | 위험 증가 |
        | 30~40 | 15 | 높은 위험 |
        | 40 이상 | 5 | 극심한 위험 |

        VIX는 S&P500 옵션 가격에서 추정한 약 30일 기대 변동성입니다.

        VIX가 급등하면 시장의 헤지 수요와 불확실성이 커졌을 수 있습니다.

        VIX가 낮다는 이유만으로 매수하지는 마세요.
        VIX가 급락하는지, 높은 수준에서 유지되는지도 중요합니다.
        """
    )

    show_period_changes(close["^VIX"], inverse=True)

    # --------------------------------------------------------
    # Indicator 4: Credit
    # --------------------------------------------------------

    credit_score = score_credit(latest)

    if credit_score >= 70:
        credit_signal = "위험선호 개선"
    elif credit_score >= 55:
        credit_signal = "양호"
    elif credit_score >= 45:
        credit_signal = "중립"
    elif credit_score >= 30:
        credit_signal = "위험선호 약화"
    else:
        credit_signal = "위험회피"

    show_scorecard(
        "④ 신용시장 위험선호 (HYG / IEF)",
        credit_score,
        credit_signal,
        (
            f"HYG/IEF 비율 {latest['HYG_IEF']:.4f}, "
            f"20일 변화 {latest['CREDIT_RET20']:+.2f}%, "
            f"60일 변화 {latest['CREDIT_RET60']:+.2f}%."
        ),
        """
        | 점수 | 신호 |
        |---:|---|
        | 70~100 | 위험선호 개선 |
        | 55~69 | 양호 |
        | 45~54 | 중립 |
        | 30~44 | 위험선호 약화 |
        | 0~29 | 위험회피 |

        HYG는 하이일드 회사채 ETF이고,
        IEF는 미국 7~10년 국채 ETF입니다.

        HYG/IEF가 상승하면 상대적으로 위험자산 선호가 강해지는
        흐름으로 해석할 수 있습니다.

        **주의:** 이 비율은 순수 신용 스프레드가 아닙니다.
        IEF의 금리 민감도와 채권 가격 변화도 포함됩니다.
        """
    )

    show_period_changes(features["HYG_IEF"])

    # --------------------------------------------------------
    # Indicator 5: Rates
    # --------------------------------------------------------

    rates_score = score_rates(latest["YIELD10_CHG20"])

    yield_change = latest["YIELD10_CHG20"]

    if pd.isna(yield_change):
        rates_signal = "데이터 부족"
    elif yield_change <= -0.15:
        rates_signal = "금리 부담 완화"
    elif yield_change <= 0.05:
        rates_signal = "대체로 중립"
    elif yield_change <= 0.15:
        rates_signal = "금리 상승 부담"
    else:
        rates_signal = "금리 급등 부담"

    show_scorecard(
        "⑤ 미국 10년물 금리",
        rates_score,
        rates_signal,
        (
            f"10년물 {latest['YIELD10']:.2f}%, "
            f"3개월물 {latest['YIELD3M']:.2f}%, "
            f"10년물 20일 변화 {yield_change:+.2f}%p, "
            f"10Y-3M 스프레드 "
            f"{latest['YIELD_SPREAD_10Y_3M']:+.2f}%p."
        ),
        """
        | 10년물 20일 변화 | 점수 | 해석 |
        |---:|---:|---|
        | -0.30%p 이하 | 85 | 금리 부담 크게 완화 |
        | -0.30~-0.15%p | 75 | 금리 부담 완화 |
        | -0.15~-0.05%p | 65 | 완만한 하락 |
        | -0.05~+0.05%p | 55 | 중립 |
        | +0.05~+0.15%p | 45 | 상승 부담 |
        | +0.15~+0.30%p | 30 | 부담 확대 |
        | +0.30%p 초과 | 15 | 급등 부담 |

        금리 상승은 성장주의 미래 현금흐름 할인율을 높여
        QQQ에 부담을 줄 수 있습니다.

        그러나 금리 상승 원인이 경기 개선인지,
        인플레이션 우려인지에 따라 주식시장 영향은 달라집니다.

        따라서 금리 점수만으로 매매하지 말고
        QQQ 상대강도와 시장 추세를 같이 확인하세요.
        """
    )

    show_period_changes(features["YIELD10"])

    # --------------------------------------------------------
    # Indicator 6: Fear & Greed
    # --------------------------------------------------------

    st.subheader("⑥ CNN Fear & Greed 심리 구간")

    if not pd.isna(fng_score):

        st.metric(
            "현재 심리 구간",
            fear_greed_label(fng_score),
            f"{fng_score:.1f} / 100"
        )

        fg_signal_score = score_fear_greed(fng_score)

        st.metric(
            "심리 상태 점수",
            format_score(fg_signal_score)
        )

        with st.expander("심리 점수의 해석 방법"):

            st.markdown(
                """
                이 점수는 시장 환경의 우호도를 직접 측정하는 점수가 아닙니다.

                공포와 탐욕이라는 심리 상태를 구간별로 수치화한 것입니다.

                - 극단적 공포: 위험 회피가 강한 상태
                - 공포: 투자 심리 위축
                - 중립: 심리 균형
                - 탐욕: 위험 선호 강함
                - 극단적 탐욕: 과열 위험 점검

                극단적 공포는 역발상 투자 기회일 수도 있지만,
                하락 추세가 지속되는 상황에서는 추가 하락 위험도 큽니다.
                """
            )

    else:
        st.info("CNN Fear & Greed 데이터가 없어 심리 점수를 계산하지 않았습니다.")

    st.divider()

    # --------------------------------------------------------
    # ETF relative strength
    # --------------------------------------------------------

    st.header("4. QQQ / SPY / SCHD 상대강도")

    strength_rows = []

    for ticker, values in strengths.items():
        strength_rows.append({
            "ETF": ticker,
            "5일 수익률": values["5일"],
            "20일 수익률": values["20일"],
            "60일 수익률": values["60일"],
            "상대강도 점수": values["강도 점수"]
        })

    strength_df = pd.DataFrame(strength_rows)

    st.dataframe(
        strength_df.style.format({
            "5일 수익률": "{:+.2f}%",
            "20일 수익률": "{:+.2f}%",
            "60일 수익률": "{:+.2f}%",
            "상대강도 점수": "{:+.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    if strength_df["상대강도 점수"].notna().any():

        best = strength_df.loc[
            strength_df["상대강도 점수"].idxmax(),
            "ETF"
        ]

        st.info(
            f"현재 상대강도 1위: **{best}**. "
            "이는 최근 가격 강도가 상대적으로 높다는 뜻이지, "
            "독립적인 매수 신호는 아닙니다."
        )

    with st.expander("ETF 상대강도 산출 방식"):

        st.markdown(
            """
            상대강도 점수는 다음과 같이 계산합니다.

            - 5거래일 수익률: 20%
            - 20거래일 수익률: 45%
            - 60거래일 수익률: 35%

            20거래일을 가장 중요하게 보는 이유는
            투자 판단의 주된 관찰 기간을 1~2주로 두되,
            단기 변동에 과도하게 반응하지 않기 위해서입니다.

            이 점수는 미래 기대수익률이 아닙니다.
            """
        )

    # --------------------------------------------------------
    # Signal conflicts
    # --------------------------------------------------------

    st.header("5. 지표 간 충돌 및 위험 경고")

    if conflicts:

        for i, conflict in enumerate(conflicts, start=1):
            st.warning(f"{i}. {conflict}")

    else:
        st.success(
            "현재 규칙에서 탐지된 주요 신호 충돌은 없습니다. "
            "모든 위험이 없다는 의미는 아닙니다."
        )

    with st.expander("충돌 신호를 확인해야 하는 이유"):

        st.markdown(
            """
            단일 지표만 보면 잘못된 결론을 내릴 수 있습니다.

            예를 들어:

            - SPY는 상승하지만 RSP/SPY가 하락한다면
              대형주에 상승이 집중될 수 있습니다.
            - SPY는 상승하지만 VIX가 급등한다면
              시장 내부의 불확실성이 커졌을 수 있습니다.
            - 주식은 상승하지만 신용시장이 약화된다면
              위험자산 선호가 충분히 뒷받침되지 않을 수 있습니다.
            - QQQ가 상승하면서 금리도 급등한다면
              성장주의 밸류에이션 부담이 커질 수 있습니다.

            따라서 시장 추세와 위험 신호를 분리해서 확인해야 합니다.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    st.header("6. 시장 데이터 차트")

    tab1, tab2, tab3, tab4 = st.tabs([
        "SPY 추세",
        "ETF 비교",
        "VIX",
        "시장 내부 지표"
    ])

    with tab1:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features["SPY"],
                name="SPY"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features["SPY_MA50"],
                name="MA50"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features["SPY_MA200"],
                name="MA200"
            )
        )

        fig.update_layout(
            title="SPY / MA50 / MA200",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:

        etf_close = close[["QQQ", "SPY", "SCHD"]].dropna(how="all")

        normalized = etf_close / etf_close.iloc[0] * 100

        fig = go.Figure()

        for ticker in normalized.columns:
            fig.add_trace(
                go.Scatter(
                    x=normalized.index,
                    y=normalized[ticker],
                    name=ticker
                )
            )

        fig.update_layout(
            title="ETF Relative Performance (Start = 100)",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features["VIX"],
                name="VIX"
            )
        )

        for level in [15, 20, 25, 30]:
            fig.add_hline(
                y=level,
                line_dash="dot",
                annotation_text=str(level)
            )

        fig.update_layout(
            title="VIX",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab4:

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                "RSP / SPY",
                "HYG / IEF"
            ],
            vertical_spacing=0.12
        )

        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features["RSP_SPY"],
                name="RSP/SPY"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features["RSP_SPY_MA50"],
                name="RSP/SPY MA50"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features["HYG_IEF"],
                name="HYG/IEF"
            ),
            row=2,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features["HYG_IEF_MA50"],
                name="HYG/IEF MA50"
            ),
            row=2,
            col=1
        )

        fig.update_layout(height=650)

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # Recent data table
    # --------------------------------------------------------

    st.header("7. 최근 지표 데이터")

    with st.expander("계산된 전체 지표 보기"):

        st.dataframe(
            features.tail(30).sort_index(ascending=False),
            use_container_width=True
        )

    # --------------------------------------------------------
    # CSV download
    # --------------------------------------------------------

    csv_data = features.tail(300).to_csv(
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    st.download_button(
        label="최근 300거래일 지표 CSV 다운로드",
        data=csv_data,
        file_name="market_monitor_features.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    st.header("8. 데이터 품질")

    quality_rows = []

    for ticker in TICKERS:

        if ticker not in close.columns:
            quality_rows.append({
                "Ticker": ticker,
                "최근 데이터": "없음",
                "결측치": "N/A"
            })
            continue

        s = close[ticker].dropna()

        quality_rows.append({
            "Ticker": ticker,
            "최근 데이터": (
                s.index[-1].strftime("%Y-%m-%d")
                if len(s) else "없음"
            ),
            "결측치": int(close[ticker].isna().sum())
        })

    quality_df = pd.DataFrame(quality_rows)

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "가격 데이터: Yahoo Finance. "
        "공포·탐욕 데이터: CNN의 비공식 공개 데이터 엔드포인트. "
        "데이터 지연, 누락, 제공처의 변경 가능성이 있습니다."
    )

    st.caption(
        "점수 기준은 휴리스틱입니다. 실제 매매에 사용하기 전 "
        "과거 데이터와 실시간 데이터에서 별도로 검증해야 합니다."
    )


if __name__ == "__main__":
    main()
