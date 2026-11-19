# -*- coding: utf-8 -*-
"""
장기 투자 전략 종가 알림 (카카오톡 '나에게 보내기')

- KR  : 국내 장 종료 후 TIGER 미국나스닥100 (133690.KS) 확인
- US  : 미국 장 종료 후 SCHD 확인

규칙 (test.py 의 장기 투자 전략과 동일)
- 기준 고점  : 최근 252거래일 최고 종가
- 하락 단계  : 고점 대비 -5%마다 1단계 (floor(|dd| / 5))
- 주식 목표  : 100% - 단계 x 15%p  (최소 0%)
- 매도 신호  : 하락 단계가 올라가 주식 목표가 낮아진 날
- 재매수     : 목표 비중이 올라가면 알림만 보냄 (실제 매수 한도 계산은 test.py 참고)

사용법
  python alert_long_term.py --market KR            # 국장 종료 후
  python alert_long_term.py --market US            # 미장 종료 후
  python alert_long_term.py --market KR --dry-run  # 전송 없이 메시지만 출력

"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# =========================================================
# 설정
# =========================================================

ASSETS = {
    "KR": {"name": "TIGER 미국나스닥100", "ticker": "133690.KS"},
    "US": {"name": "SCHD", "ticker": "SCHD"},
}

LOOKBACK_DAYS = 252      # 기준 고점 기간 (거래일)
DRAWDOWN_STEP = 5.0      # 하락 단계 간격 (%)
SELL_STEP = 15.0         # 단계당 축소 비중 (%p)
MAX_STAGE = 7            # -35% 이상이면 0%

STATE_FILE = Path(__file__).with_name("alert_state.json")

# =========================================================
# 데이터
# =========================================================

def fetch_close(ticker):
    """일봉 종가 (배당·분할 조정). 최근 2년."""
    data = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"{ticker} 가격 데이터를 가져오지 못했습니다.")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna().sort_index()
    close.name = ticker
    return close


# =========================================================
# 계산
# =========================================================

def stock_target(stage):
    return max(0.0, 100.0 - stage * SELL_STEP)


def analyze(close):
    """마지막 종가 기준 등락, 낙폭, 단계, 목표 비중 계산."""
    if len(close) < LOOKBACK_DAYS + 1:
        raise RuntimeError("252거래일 고점을 계산할 데이터가 부족합니다.")

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    day_change = (last / prev - 1) * 100

    recent = close.tail(LOOKBACK_DAYS)
    high = float(recent.max())
    high_date = recent.idxmax()
    drawdown = (last / high - 1) * 100

    stage = int(min(MAX_STAGE, max(0, abs(min(drawdown, 0.0)) // DRAWDOWN_STEP)))
    target = stock_target(stage)

    return {
        "date": close.index[-1],
        "last": last,
        "prev": prev,
        "day_change": day_change,
        "high": high,
        "high_date": high_date,
        "drawdown": drawdown,
        "stage": stage,
        "target": target,
    }


# =========================================================
# 상태 저장 (직전 단계 기억 → 단계가 올라갈 때만 매도 신호)
# =========================================================

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def signal_for(key, result, state):
    """
    직전 저장 단계와 비교.
    - 처음 실행: 기준만 저장하고 신호는 보내지 않음
    - 단계 상승: 매도 신호
    - 단계 하락: 회복(재매수 가능) 알림
    - 변화 없음: 신호 없음
    """
    prev_stage = state.get(key, {}).get("stage")
    if prev_stage is None:
        return "INIT", None

    cur = result["stage"]
    if cur > prev_stage:
        return "SELL", cur - prev_stage
    if cur < prev_stage:
        return "RECOVER", prev_stage - cur
    return "HOLD", 0


# =========================================================
# 메시지
# =========================================================
def to_bold(text):
    result = ""
    for char in str(text):
        code = ord(char)

        if 'A' <= char <= 'Z':
            result += chr(0x1D400 + code - ord('A'))
        elif 'a' <= char <= 'z':
            result += chr(0x1D41A + code - ord('a'))
        elif '0' <= char <= '9':
            result += chr(0x1D7CE + code - ord('0'))
        else:
            result += char

    return result

def build_message(market, name, result, signal, step_delta):
    d = result["date"].strftime("%Y-%m-%d")
    close_price = to_bold(f"{result['last']:,.2f}")

    lines = [
        f"[{market} 종가] {name}",
        f"기준일: {d}",
        f"📌 종가: {close_price} ({result['day_change']:+.2f}%, 전일 대비)",
        f"252일 고점: {result['high']:,.2f} ({result['high_date'].strftime('%Y-%m-%d')})",
        f"고점 대비: {result['drawdown']:+.2f}%",
        f"하락 단계: {result['stage']}단계 > 주식 목표 {result['target']:.0f}%",
    ]

    if signal == "SELL":
        sell_pct = SELL_STEP * step_delta
        lines.extend([
            "",
            f"🔴 매도 신호: 칸 전체 평가금액의 {sell_pct:.0f}%p 매도"
        ])
    elif signal == "RECOVER":
        lines.extend([
            "",
            "🟢 회복: 목표 비중 상승. 재매수는 주 1회(금요일)에 한도 내에서 진행"
        ])
    elif signal == "INIT":
        lines.extend([
            "",
            "ℹ️ 기준 저장 완료. 다음 실행부터 단계 변화에 따라 신호를 보냅니다."
        ])
    else:
        lines.extend([
            "",
            "신호 없음 (단계 변화 없음)"
        ])

    return "\n".join(lines)

def send_discord_message(message):
    """Discord Webhook으로 투자 알림 전송."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    if not webhook_url:
        print("[Discord] Webhook URL이 없습니다.")
        return

    try:
        response = requests.post(
            webhook_url,
            json={
                "content": message,
                "allowed_mentions": {
                    "parse": []
                }
            },
            timeout=15,
        )

        response.raise_for_status()

    except requests.RequestException as e:
        print(f"[Discord 오류] {e}")


# =========================================================
# 메인
# =========================================================

def run(market, dry_run=False):
    cfg = ASSETS[market]
    close = fetch_close(cfg["ticker"])
    result = analyze(close)

    state = load_state()
    key = f"{market}:{cfg['ticker']}"
    signal, step_delta = signal_for(key, result, state)

    msg = build_message(market, cfg["name"], result, signal, step_delta)
    print(msg)
    print("-" * 40)

    if dry_run:
        print("(dry-run: 카카오 전송 안 함, 상태 저장 안 함)")
        return

    # 실제 신호일 때만 전송 (INIT 은 전송)
    if signal in ("SELL", "RECOVER", "INIT", "HOLD"):
        send_discord_message(msg)
        print("디스코드 전송 완료")

    state[key] = {
        "stage": result["stage"],
        "date": result["date"].strftime("%Y-%m-%d"),
    }
    save_state(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["KR", "US"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        run(args.market, dry_run=args.dry_run)
    except Exception as e:
        print(f"[오류] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


# =========================================================
# 스케줄 (macOS crontab, 한국시간 기준)
#   40 15 * * 1-5  cd /Users/lotte/Documents/test_rec_model && python alert_long_term.py --market KR
#   30 6  * * 2-6  cd /Users/lotte/Documents/test_rec_model && python alert_long_term.py --market US
#   (미국 장은 서머타임에 따라 KST 05:00 또는 06:00 마감 → 06:30 실행)
# =========================================================
