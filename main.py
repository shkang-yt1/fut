#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import logging
from datetime import datetime, timezone
import json
import ccxt
import pandas as pd

# ============== 설정값 ============== #
API_KEY = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_API_SECRET", "")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "")

SYMBOLS = [
    "AVAX/USDT:USDT",
    "OKB/USDT:USDT",
    "SOL/USDT:USDT",
]

TIMEFRAME = "1h"

RISK_PER_TRADE = 0.02
MAX_LEVERAGE   = 13
LOOP_INTERVAL  = 3

CCI_PERIOD = 14
BB_PERIOD  = 20
BB_K       = 2.0

SL_OFFSET  = 0.01  # 1%: 스톱로스 여유폭
TP_OFFSET  = 0.004 # 0.4%: 익절가 여유폭

R_THRESHOLD = 1.2  # R >= 1.0 인 경우에만 진입
MIN_DELTA   = 16.0

# 포지션당 사용할 증거금 비율: 전체 계좌를 3.5등분
MARGIN_DIVISOR = 3.5

# ===== BE SL(수수료권 SL 상향) 설정 =====
BE_R_THRESHOLD = 0.6
FEE_PCT = 0.001 * 10   # 수수료(0.1%)
BE_PCT  = 0.0011  # 수수료권(0.11%)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============== JSON 직렬화 ============== #
def _serialize_pos_state(pos_state: dict):
    out = {}
    for sym, s in pos_state.items():
        if s is None:
            out[sym] = None
            continue
        d = dict(s)
        if isinstance(d.get("entry_time"), datetime):
            d["entry_time"] = d["entry_time"].isoformat()
        out[sym] = d
    return out


def save_state(pos_state, entry_restrict, last_signal, equity, ohlcv):
    state = {
        "pos_state": _serialize_pos_state(pos_state),
        "entry_restrict": entry_restrict,
        "last_signal": last_signal,
        "equity": equity,
        "ohlcv": ohlcv,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open("/app/bot_state.json", "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============== OKX 초기화 ============== #
def init_exchange():
    exchange = ccxt.okx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "password": API_PASSPHRASE,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultSettle": "usdt"},
    })
    # exchange.set_sandbox_mode(True)
    exchange.load_markets()

    try:
        exchange.set_position_mode(hedged=False)
    except Exception:
        pass

    for sym in SYMBOLS:
        try:
            exchange.set_leverage(MAX_LEVERAGE, sym, params={"mgnMode": "cross"})
        except Exception:
            pass

    return exchange


# ============== 유틸 ============== #
def fetch_ohlcv_df(exchange, symbol, timeframe, limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("dt", inplace=True)
    return df


def calculate_indicators(df):
    # ----- CCI (OKX TradingView 스타일) -----
    tp = (df["high"] + df["low"] + df["close"]) / 3  # hlc3
    sma_tp = tp.rolling(CCI_PERIOD).mean()

    # mean deviation = sum(|x - mean(x)|)/length
    def _mean_dev(window):
        m = window.mean()
        return (window.sub(m).abs().sum()) / len(window)

    md = tp.rolling(CCI_PERIOD).apply(_mean_dev, raw=False)
    df["cci"] = (tp - sma_tp) / (0.015 * md)

    # Bollinger Bands
    ma = df["close"].rolling(BB_PERIOD).mean()
    std = df["close"].rolling(BB_PERIOD).std(ddof=0)
    df["bb_mid"]   = ma
    df["bb_upper"] = ma + BB_K * std
    df["bb_lower"] = ma - BB_K * std
    return df


def fetch_futures_equity(exchange):
    bal = exchange.fetch_balance()
    usdt = bal.get("USDT", {})
    return float(usdt.get("free", 0)), float(usdt.get("total", 0))


def compute_order_size_equal_margin_and_risk(exchange, symbol, entry_price, equity_total, stop_pct):
    """
    - 포지션당 증거금: equity_total / MARGIN_DIVISOR (3.5분할)
    - 손절 도달 시 손실: equity_total * RISK_PER_TRADE
    - 손절 거리(stop_pct)에 맞춰 레버리지를 동적으로 조정
    - 계약 단위(정수)로 내림한 뒤, 실제 노션 기준으로 레버리지를 다시 맞춰
      실제 증거금이 margin_per_pos에 최대한 근접하도록 조정
    """
    if entry_price <= 0 or stop_pct <= 0 or equity_total <= 0:
        return 0, 0.0, 0.0

    # 포지션당 사용할 목표 증거금
    margin_per_pos = equity_total / MARGIN_DIVISOR

    # 손절 시 목표 손실 금액
    risk_per_pos = equity_total * RISK_PER_TRADE

    # 손절 거리까지 갔을 때 risk_per_pos 만큼 잃으려면 필요한 노션
    target_notional_for_risk = risk_per_pos / stop_pct  # N_target

    # 레버리지 제한 고려한 노션 범위
    min_notional = margin_per_pos * 1.0
    max_notional = margin_per_pos * MAX_LEVERAGE

    # 노션을 제한 범위 안으로 클램프
    notional = max(min(target_notional_for_risk, max_notional), min_notional)

    # 심볼별 계약 단위
    market = exchange.market(symbol)
    contract_size = market.get("contractSize") or float(market["info"].get("ctVal", 1))
    notional_per_contract = entry_price * contract_size

    # 계약 수량: 정수 계약 단위로 내림
    amount = math.floor(notional / notional_per_contract)
    if amount <= 0:
        return 0, 0.0, 0.0

    # 실제 체결 기준 노션
    actual_notional = amount * notional_per_contract

    # 이 노션과 margin_per_pos로부터 레버리지 재계산
    actual_leverage = actual_notional / margin_per_pos
    if actual_leverage < 1.0:
        actual_leverage = 1.0
    if actual_leverage > MAX_LEVERAGE:
        actual_leverage = float(MAX_LEVERAGE)

    # 전체 계좌 대비 실제 레버리지
    effective_leverage = actual_notional / equity_total

    return amount, actual_leverage, effective_leverage


def sync_positions(exchange, symbols):
    """
    OKX 포지션에서:
      - size (contracts)
      - entry_price
      - leverage
      - margin
      - notional (entry_price * size * contract_size)
    까지 뽑아서 대시보드로 넘길 수 있게 구성
    """
    result = {
        sym: {
            "has_position": False,
            "side": None,
            "size": 0,
            "entry_price": None,
            "leverage": None,
            "margin": None,
            "notional": None,
        }
        for sym in symbols
    }

    try:
        positions = exchange.fetch_positions()
    except Exception:
        return result

    for p in positions:
        sym = p.get("symbol")
        if sym not in symbols:
            continue

        contracts = float(p.get("contracts") or p.get("positionAmt") or 0)
        if contracts == 0:
            continue

        side = p.get("side") or ("long" if contracts > 0 else "short")
        entry_price = float(p.get("entryPrice") or p.get("avgPrice") or 0)

        # 레버리지
        lev_raw = p.get("leverage") or (p.get("info") or {}).get("lever") or None
        try:
            leverage = float(lev_raw) if lev_raw is not None else None
        except Exception:
            leverage = None

        # 증거금
        margin_raw = (
            p.get("initialMargin")
            or p.get("margin")
            or (p.get("info") or {}).get("margin")
            or None
        )
        try:
            margin = float(margin_raw) if margin_raw is not None else None
        except Exception:
            margin = None

        # 노션 = entry_price * size * contract_size
        notional = None
        try:
            market = exchange.market(sym)
            contract_size = market.get("contractSize") or float(
                market["info"].get("ctVal", 1)
            )
            if entry_price > 0:
                notional = abs(contracts) * contract_size * entry_price
        except Exception:
            notional = None

        result[sym] = {
            "has_position": True,
            "side": side,
            "size": abs(contracts),
            "entry_price": entry_price,
            "leverage": leverage,
            "margin": margin,
            "notional": notional,
        }
    return result


# ============== CCI 신호 ============== #
def detect_cci_signal(df):
    if df is None or len(df) < CCI_PERIOD + 3:
        return None

    curr  = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]

    cci_curr  = float(curr.get("cci", float("nan")))
    cci_prev1 = float(prev1.get("cci", float("nan")))
    cci_prev2 = float(prev2.get("cci", float("nan")))

    if any(math.isnan(x) for x in [cci_curr, cci_prev1, cci_prev2]):
        return None

    entry_price = float(curr["close"])
    if entry_price <= 0:
        return None

    side = None
    stop_price = None

    if (cci_prev1 < -100) and (cci_curr > cci_prev1) and (cci_curr - cci_prev1 >= MIN_DELTA):
        side = "long"
        stop_price = float(prev1["low"]) * (1 - SL_OFFSET)

    elif (cci_prev1 > 100) and (cci_curr < cci_prev1) and (cci_prev1 - cci_curr >= MIN_DELTA):
        side = "short"
        stop_price = float(prev1["high"]) * (1 + SL_OFFSET)

    if side is None or stop_price <= 0:
        return None

    return {
        "side": side,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "signal_ts": int(curr["ts"]),
    }


def _safe_float(v):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


# ============== 메인 루프 ============== #
def main():
    exchange = init_exchange()
    logging.info("CCI + Bollinger 자동매매 (동적 TP + BB/CCI 대시보드 + 균등 증거금/리스크) 시작")

    pos_state = {
        sym: {
            "side": None,
            "size": 0,
            "entry_price": None,
            "stop_price": None,
            "tp_price": None,
            "entry_candle_ts": None,
            "stop_order_id": None,
            "entry_time": None,
            "leverage": None,
            "margin": None,
            "notional": None,
            "be_sl_applied": False,  # 추가: BE SL 적용 여부
        }
        for sym in SYMBOLS
    }

    entry_restrict = {sym: None for sym in SYMBOLS}
    last_signal_candle_ts = {}

    while True:
        try:
            # --- OHLCV & 지표 --- #
            data = {}
            for sym in SYMBOLS:
                df = fetch_ohlcv_df(exchange, sym, TIMEFRAME, 200)
                if df is None:
                    continue
                df = calculate_indicators(df)
                if len(df) < BB_PERIOD + 3:
                    continue
                data[sym] = (df, df.iloc[-3], df.iloc[-1])

            if not data:
                time.sleep(LOOP_INTERVAL)
                continue

            # --- 포지션 동기화 --- #
            exch_positions = sync_positions(exchange, SYMBOLS)

            for sym in SYMBOLS:
                if not exch_positions[sym]["has_position"]:
                    if pos_state[sym]["side"] == "long":
                        entry_restrict[sym] = None
                    elif pos_state[sym]["side"] == "short":
                        entry_restrict[sym] = None

                    pos_state[sym] = {
                        "side": None,
                        "size": 0,
                        "entry_price": None,
                        "stop_price": None,
                        "tp_price": None,
                        "entry_candle_ts": None,
                        "stop_order_id": None,
                        "entry_time": None,
                        "leverage": None,
                        "margin": None,
                        "notional": None,
                        "be_sl_applied": False,
                    }
                else:
                    pos_state[sym]["side"] = exch_positions[sym]["side"]
                    pos_state[sym]["size"] = exch_positions[sym]["size"]
                    pos_state[sym]["entry_price"] = exch_positions[sym]["entry_price"]
                    pos_state[sym]["leverage"] = exch_positions[sym]["leverage"]
                    pos_state[sym]["margin"] = exch_positions[sym]["margin"]
                    pos_state[sym]["notional"] = exch_positions[sym]["notional"]

            # --- TP 관리 (동적) --- #
            for sym in SYMBOLS:
                if sym not in data:
                    continue
                df, prev, curr = data[sym]

                side = pos_state[sym]["side"]
                size = pos_state[sym]["size"]
                if side is None or size <= 0:
                    pos_state[sym]["tp_price"] = None
                    continue

                bb_upper = float(curr["bb_upper"])
                bb_lower = float(curr["bb_lower"])
                curr_price = float(curr["close"])

                pos_state[sym]["tp_price"] = (
                    bb_upper * (1 - TP_OFFSET) if side == "long" else bb_lower * (1 + TP_OFFSET)
                )

                # ====== (추가) 조건부 BE SL: 기대손익비<=0.6 & 가격이 수수료(0.1%)~TP 사이 ====== #
                try:
                    if not pos_state[sym].get("be_sl_applied", False):
                        entry_price = pos_state[sym].get("entry_price")
                        stop_price = pos_state[sym].get("stop_price")
                        tp_price = pos_state[sym].get("tp_price")

                        if entry_price and stop_price and tp_price:
                            stop_diff = abs(entry_price - stop_price)
                            tp_diff = abs(tp_price - entry_price)
                            if stop_diff > 0:
                                R_now = tp_diff / stop_diff

                                if R_now <= BE_R_THRESHOLD:
                                    if side == "long":
                                        fee_line = entry_price * (1 + FEE_PCT)   # +0.1%
                                        be_sl    = entry_price * (1 + BE_PCT)    # +0.11%
                                        in_range = (fee_line <= curr_price <= tp_price)
                                        sl_side = "sell"
                                        better_than_current = (stop_price < be_sl)
                                    else:
                                        fee_line = entry_price * (1 - FEE_PCT)   # -0.1%
                                        be_sl    = entry_price * (1 - BE_PCT)    # -0.11%
                                        in_range = (tp_price <= curr_price <= fee_line)
                                        sl_side = "buy"
                                        better_than_current = (stop_price > be_sl)

                                    if in_range and better_than_current:
                                        if pos_state[sym]["stop_order_id"]:
                                            try:
                                                exchange.cancel_order(pos_state[sym]["stop_order_id"], sym)
                                            except Exception:
                                                pass

                                        try:
                                            sl_order = exchange.create_order(
                                                sym,
                                                "market",
                                                sl_side,
                                                size,
                                                params={
                                                    "tdMode": "cross",
                                                    "reduceOnly": True,
                                                    "stopLossPrice": be_sl,
                                                },
                                            )
                                            pos_state[sym]["stop_order_id"] = sl_order.get("id")
                                            pos_state[sym]["stop_price"] = be_sl
                                            pos_state[sym]["be_sl_applied"] = True
                                        except Exception:
                                            pass
                except Exception:
                    pass
                # ============================================================================== #

                if side == "long" and curr_price >= pos_state[sym]["tp_price"]:
                    if pos_state[sym]["stop_order_id"]:
                        try:
                            exchange.cancel_order(pos_state[sym]["stop_order_id"], sym)
                        except Exception:
                            pass

                    exch_now = sync_positions(exchange, SYMBOLS)[sym]
                    if exch_now["has_position"]:
                        exchange.create_order(
                            sym, "market", "sell", exch_now["size"], params={"tdMode": "cross"}
                        )

                    entry_restrict[sym] = None
                    pos_state[sym] = {
                        "side": None,
                        "size": 0,
                        "entry_price": None,
                        "stop_price": None,
                        "tp_price": None,
                        "entry_candle_ts": None,
                        "stop_order_id": None,
                        "entry_time": None,
                        "leverage": None,
                        "margin": None,
                        "notional": None,
                        "be_sl_applied": False,
                    }

                elif side == "short" and curr_price <= pos_state[sym]["tp_price"]:
                    if pos_state[sym]["stop_order_id"]:
                        try:
                            exchange.cancel_order(pos_state[sym]["stop_order_id"], sym)
                        except Exception:
                            pass

                    exch_now = sync_positions(exchange, SYMBOLS)[sym]
                    if exch_now["has_position"]:
                        exchange.create_order(
                            sym, "market", "buy", exch_now["size"], params={"tdMode": "cross"}
                        )

                    entry_restrict[sym] = None
                    pos_state[sym] = {
                        "side": None,
                        "size": 0,
                        "entry_price": None,
                        "stop_price": None,
                        "tp_price": None,
                        "entry_candle_ts": None,
                        "stop_order_id": None,
                        "entry_time": None,
                        "leverage": None,
                        "margin": None,
                        "notional": None,
                        "be_sl_applied": False,
                    }

            # --- 신규 진입 --- #
            for sym in SYMBOLS:
                if sym not in data:
                    continue

                df, prev, curr = data[sym]
                curr_ts = int(curr["ts"])
                if last_signal_candle_ts.get(sym) == curr_ts:
                    continue

                if pos_state[sym]["side"] is not None:
                    continue

                signal = detect_cci_signal(df)
                if not signal:
                    continue

                side_signal = signal["side"]
                entry_price = signal["entry_price"]
                stop_price = signal["stop_price"]

                if side_signal == "long":
                    if stop_price >= entry_price:
                        continue
                elif side_signal == "short":
                    if stop_price <= entry_price:
                        continue

                if entry_restrict[sym] == "long_only" and side_signal != "long":
                    continue
                if entry_restrict[sym] == "short_only" and side_signal != "short":
                    continue

                bb_upper = float(curr["bb_upper"])
                bb_lower = float(curr["bb_lower"])
                tp_price = (
                    bb_upper * (1 - TP_OFFSET)
                    if side_signal == "long"
                    else bb_lower * (1 + TP_OFFSET)
                )

                stop_diff = abs(entry_price - stop_price)
                tp_diff = abs(entry_price - tp_price)
                if stop_diff <= 0:
                    continue

                R = tp_diff / stop_diff
                if R < R_THRESHOLD:
                    continue

                free, total = fetch_futures_equity(exchange)
                if total <= 0:
                    continue

                stop_pct = stop_diff / entry_price

                amount, leverage, eff_lev = compute_order_size_equal_margin_and_risk(
                    exchange, sym, entry_price, total, stop_pct
                )
                if amount <= 0:
                    continue

                order_side = "buy" if side_signal == "long" else "sell"
                sl_side = "sell" if side_signal == "long" else "buy"

                lev_float = max(1.0, min(round(float(leverage), 2), float(MAX_LEVERAGE)))
                try:
                    exchange.set_leverage(lev_float, sym, params={"mgnMode": "cross"})
                except Exception:
                    pass

                exchange.create_order(
                    sym, "market", order_side, amount, params={"tdMode": "cross"}
                )

                time.sleep(0.3)
                after = sync_positions(exchange, SYMBOLS)[sym]
                actual_entry = after["entry_price"] or entry_price
                actual_size = after["size"]

                pos_state[sym]["side"] = side_signal
                pos_state[sym]["size"] = actual_size
                pos_state[sym]["entry_price"] = actual_entry
                pos_state[sym]["stop_price"] = stop_price
                pos_state[sym]["entry_time"] = datetime.now(timezone.utc)
                pos_state[sym]["entry_candle_ts"] = curr_ts
                pos_state[sym]["leverage"] = after.get("leverage")
                pos_state[sym]["margin"] = after.get("margin")
                pos_state[sym]["notional"] = after.get("notional")
                pos_state[sym]["be_sl_applied"] = False

                try:
                    sl_order = exchange.create_order(
                        sym,
                        "market",
                        sl_side,
                        actual_size,
                        params={
                            "tdMode": "cross",
                            "reduceOnly": True,
                            "stopLossPrice": stop_price,
                        },
                    )
                    pos_state[sym]["stop_order_id"] = sl_order.get("id")
                except Exception:
                    pos_state[sym]["stop_order_id"] = None

                last_signal_candle_ts[sym] = curr_ts
                entry_restrict[sym] = None

            # --- 대시보드용 OHLCV + 인디케이터 저장 --- #
            ohlcv_state = {}
            for sym in SYMBOLS:
                if sym not in data:
                    continue
                df, _, _ = data[sym]
                tail = df.tail(100)
                candles = []
                for row in tail.itertuples():
                    candles.append({
                        "time": int(row.ts // 1000),
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                        "bb_upper": _safe_float(getattr(row, "bb_upper", None)),
                        "bb_lower": _safe_float(getattr(row, "bb_lower", None)),
                        "bb_mid": _safe_float(getattr(row, "bb_mid", None)),
                        "cci": _safe_float(getattr(row, "cci", None)),
                    })
                ohlcv_state[sym] = candles

            _, total = fetch_futures_equity(exchange)
            save_state(pos_state, entry_restrict, last_signal_candle_ts, total, ohlcv_state)

            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            logging.warning(f"메인 루프 에러: {e}")
            time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
