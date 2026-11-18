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

카카오 설정
  환경변수 KAKAO_REST_API_KEY, KAKAO_REFRESH_TOKEN 이 필요합니다.
  (토큰 발급 방법은 파일 하단 주석 참고)
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
TOKEN_FILE = Path(__file__).with_name("kakao_tokens.json")

KAKAO_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
KAKAO_SEND_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"


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


# =========================================================
# 카카오톡 나에게 보내기
# =========================================================

def load_tokens():
    env_refresh_token = os.environ.get("KAKAO_REFRESH_TOKEN", "")

    if os.environ.get("GITHUB_ACTIONS") == "true":
        return {
            "refresh_token": env_refresh_token,
            "access_token": "",
        }

    if TOKEN_FILE.exists():
        return json.loads(TOKEN_FILE.read_text(encoding="utf-8"))

    return {
        "refresh_token": env_refresh_token,
        "access_token": "",
    }


def save_tokens(tokens):
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return

    TOKEN_FILE.write_text(
        json.dumps(tokens, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def refresh_access_token():
    """Refresh Token으로 Access Token 재발급 및 GitHub Secret 동기화."""
    api_key = os.environ.get("KAKAO_REST_API_KEY", "")
    client_secret = os.environ.get("KAKAO_CLIENT_SECRET", "")
    tokens = load_tokens()

    refresh_token = (
        tokens.get("refresh_token")
        or os.environ.get("KAKAO_REFRESH_TOKEN", "")
    )

    if not api_key or not refresh_token:
        raise RuntimeError(
            "KAKAO_REST_API_KEY 또는 KAKAO_REFRESH_TOKEN이 없습니다."
        )

    res = requests.post(
        KAKAO_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": api_key,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        },
        timeout=15,
    )
    body = res.json()

    if not res.ok:
        raise RuntimeError(
            f"Kakao token refresh failed: "
            f"HTTP {res.status_code}, "
            f"error={body.get('error')}, "
            f"description={body.get('error_description')}, "
            f"error_code={body.get('error_code')}"
        )

    tokens["access_token"] = body["access_token"]

    new_refresh_token = body.get("refresh_token")

    if new_refresh_token:
        tokens["refresh_token"] = new_refresh_token
        save_tokens(tokens)

        print("[토큰] 새로운 Refresh Token 감지")

        # GitHub Actions에서만 GitHub Secret 갱신
        if os.environ.get("GITHUB_ACTIONS") == "true":
            update_github_secret(new_refresh_token)
    else:
        save_tokens(tokens)
        print("[토큰] 새로운 Refresh Token 없음. 기존 토큰 유지")

    return tokens["access_token"]


def update_github_secret(new_refresh_token):
    """GitHub Repository Secret의 Refresh Token 갱신."""
    import base64
    import nacl.public

    github_token = os.environ.get("GH_TOKEN", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")

    if not github_token or not repository:
        raise RuntimeError("GitHub Secret 갱신에 필요한 환경변수가 없습니다.")

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    base_url = f"https://api.github.com/repos/{repository}/actions/secrets"

    # Repository 공개키 조회
    key_res = requests.get(
        f"{base_url}/public-key",
        headers=headers,
        timeout=15,
    )
    key_res.raise_for_status()
    key_data = key_res.json()

    # GitHub 공개키로 Secret 암호화
    public_key = nacl.public.PublicKey(
        base64.b64decode(key_data["key"])
    )
    sealed_box = nacl.public.SealedBox(public_key)
    encrypted_value = base64.b64encode(
        sealed_box.encrypt(new_refresh_token.encode("utf-8"))
    ).decode("utf-8")

    # Repository Secret 갱신
    update_res = requests.put(
        f"{base_url}/KAKAO_REFRESH_TOKEN",
        headers=headers,
        json={
            "encrypted_value": encrypted_value,
            "key_id": key_data["key_id"],
        },
        timeout=15,
    )
    update_res.raise_for_status()

    print("GitHub KAKAO_REFRESH_TOKEN Secret 업데이트 완료")


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
        print("[Discord] 메시지 전송 완료")

    except requests.RequestException as e:
        print(f"[Discord 오류] {e}")


def send_kakao_memo(text):
    access_token = refresh_access_token()

    template = {
        "object_type": "text",
        "text": text[:1900],  # 카카오 텍스트 길이 제한 여유
        "link": {
            "web_url": "https://finance.yahoo.com",
            "mobile_web_url": "https://finance.yahoo.com",
        },
        "button_title": "확인",
    }

    res = requests.post(
        KAKAO_SEND_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        data={"template_object": json.dumps(template, ensure_ascii=False)},
        timeout=15,
    )
    res.raise_for_status()
    return res.json()


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
        send_kakao_memo(msg)
        print("카카오톡 전송 완료")

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
# 카카오 토큰 발급 (최초 1회)
# 1) https://developers.kakao.com 에서 앱 생성 → REST API 키 확인
# 2) 카카오 로그인 활성화, Redirect URI 등록 (예: https://localhost)
# 3) 동의항목에서 "talk_message"(카카오톡 메시지 전송) 추가
# 4) 브라우저에서 아래 주소로 인가 코드 발급
#    https://kauth.kakao.com/oauth/authorize?client_id=REST_API_KEY&redirect_uri=https://localhost&response_type=code&scope=talk_message
# 5) 받은 code 로 토큰 발급
#    curl -X POST https://kauth.kakao.com/oauth/token \
#      -d grant_type=authorization_code -d client_id=REST_API_KEY \
#      -d redirect_uri=https://localhost -d code=CODE
# 6) 응답의 refresh_token 을 환경변수 KAKAO_REFRESH_TOKEN 에 넣기
#    export KAKAO_REST_API_KEY=... ; export KAKAO_REFRESH_TOKEN=...
#
# 스케줄 (macOS crontab, 한국시간 기준)
#   40 15 * * 1-5  cd /Users/lotte/Documents/test_rec_model && python alert_long_term.py --market KR
#   30 6  * * 2-6  cd /Users/lotte/Documents/test_rec_model && python alert_long_term.py --market US
#   (미국 장은 서머타임에 따라 KST 05:00 또는 06:00 마감 → 06:30 실행)
# =========================================================
