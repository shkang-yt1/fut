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

BASE_RISK = 0.02
REDUCED_RISK = 0.01

RISK_PER_TRADE = BASE_RISK
MAX_LEVERAGE   = 13
LOOP_INTERVAL  = 3

CCI_PERIOD = 14
BB_PERIOD  = 20
BB_K       = 2.0

SL_OFFSET  = 0.01  # 1%: 스톱로스 여유폭
TP_OFFSET  = 0.004 # 0.4%: 익절가 여유폭

R_THRESHOLD = 1.2  # R >= 1.0 인 경우에만 진입
MIN_DELTA   = 20.0

# 포지션당 사용할 증거금 비율: 전체 계좌를 3.5등분
MARGIN_DIVISOR = 3.5

# ===== BE SL(수수료권 SL 상향) 설정 =====
BE_R_THRESHOLD = 0.6
FEE_PCT = 0.001 * 10   # 수수료(0.1%)
BE_PCT  = 0.0011  # 수수료권(0.11%)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===== 상태 파일(스냅샷) =====
STATE_PATH = "/app/state/bot_state.json"

# ===== 포지션 히스토리 저장(JSONL) =====
POSITION_HISTORY_PATH = "/app/state/position_history.jsonl"
POSITION_HISTORY_LIMIT = 10


# =========================
# 재시작 동기화 유틸
# =========================
def _parse_dt(s):
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    try:
        dt = datetime.fromisoformat(str(s))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _default_pos_state():
    """현재 코드의 pos_state 기본 구조를 그대로 반환."""
    return {
        sym: {
            "side": None,
            "size": 0,
            "entry_price": None,
            "stop_price": None,
            "init_stop_price": None,
            "tp_price": None,
            "entry_candle_ts": None,
            "stop_order_id": None,
            "entry_time": None,
            "leverage": None,
            "margin": None,
            "notional": None,
            "be_sl_applied": False,  # BE SL 적용 여부
        }
        for sym in SYMBOLS
    }


def _default_symbol_risk():
    """심볼별 리스크 기본값 (모두 BASE_RISK로 초기화)."""
    return {sym: BASE_RISK for sym in SYMBOLS}


def _hydrate_pos_state(saved_pos_state: dict):
    """bot_state.json에 저장된 pos_state(직렬화된 dict)를 런타임 구조로 복원."""
    pos_state = _default_pos_state()
    if not isinstance(saved_pos_state, dict):
        return pos_state

    for sym in SYMBOLS:
        s = saved_pos_state.get(sym)
        if not isinstance(s, dict):
            continue

        for k in list(pos_state[sym].keys()):
            if k in s:
                pos_state[sym][k] = s.get(k)

        pos_state[sym]["entry_time"] = _parse_dt(pos_state[sym].get("entry_time"))

        try:
            if pos_state[sym]["size"] is None:
                pos_state[sym]["size"] = 0
            else:
                pos_state[sym]["size"] = float(pos_state[sym]["size"])
        except Exception:
            pos_state[sym]["size"] = 0

    return pos_state


def _hydrate_symbol_risk(saved: dict):
    """bot_state.json에 저장된 symbol_risk를 런타임 구조로 복원."""
    risk = _default_symbol_risk()
    if not isinstance(saved, dict):
        return risk
    for sym in SYMBOLS:
        try:
            v = saved.get(sym)
            risk[sym] = float(v) if v is not None else BASE_RISK
        except Exception:
            risk[sym] = BASE_RISK
    return risk


def load_boot_state():
    """
    재시작 시 bot_state.json에서 다음 필드를 복구:
      - pos_state
      - position_history (최근 10개)
      - last_signal (last_signal_candle_ts)
      - symbol_risk (심볼별 리스크)
    """
    try:
        if not os.path.exists(STATE_PATH):
            return None
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return None
        state = json.loads(content)

        boot = {}

        boot["pos_state"] = _hydrate_pos_state(state.get("pos_state", {}))

        ph = state.get("position_history", [])
        boot["position_history"] = ph[:POSITION_HISTORY_LIMIT] if isinstance(ph, list) else []

        ls = state.get("last_signal", {})
        if isinstance(ls, dict):
            out = {}
            for sym in SYMBOLS:
                v = ls.get(sym)
                try:
                    out[sym] = int(v) if v is not None else None
                except Exception:
                    out[sym] = None
            boot["last_signal"] = out
        else:
            boot["last_signal"] = {}

        boot["symbol_risk"] = _hydrate_symbol_risk(state.get("symbol_risk", {}))

        return boot
    except Exception:
        return None


def _tail_lines(path: str, n: int = 10, block_size: int = 8192):
    """파일 끝에서 n줄을 효율적으로 읽는다(JSONL tail 용)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end == 0:
                return []
            data = b""
            pos = end
            lines = []
            while pos > 0 and len(lines) <= n:
                read_size = min(block_size, pos)
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size) + data
                lines = data.splitlines()
            return [ln.decode("utf-8", errors="ignore") for ln in lines[-n:]]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def load_position_history_cache(path: str, limit: int = POSITION_HISTORY_LIMIT):
    """재시작 내성: 히스토리 파일의 마지막 limit줄만 읽어서 캐시에 로드."""
    cache = []
    for line in _tail_lines(path, n=limit):
        line = (line or "").strip()
        if not line:
            continue
        try:
            cache.append(json.loads(line))
        except Exception:
            continue
    try:
        cache.sort(key=lambda r: r.get("entry_time") or "", reverse=True)
    except Exception:
        pass
    return cache[:limit]


def append_position_history(cache: list, record: dict, path: str = POSITION_HISTORY_PATH, limit: int = POSITION_HISTORY_LIMIT):
    """파일에 1줄 append + 메모리 캐시에도 반영(최근 limit개 유지)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

    line = json.dumps(record, ensure_ascii=False)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

    cache.append(record)
    try:
        cache.sort(key=lambda r: r.get("entry_time") or "", reverse=True)
    except Exception:
        pass
    del cache[limit:]
    return cache


def _get_contract_size(exchange, symbol: str) -> float:
    try:
        market = exchange.market(symbol)
        return float(market.get("contractSize") or market["info"].get("ctVal", 1))
    except Exception:
        return 1.0


def _iso(dt):
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _infer_exit_reason(p: dict) -> str:
    """TP/SL/BE/MANUAL/UNKNOWN 중 추정."""
    try:
        if p.get("be_sl_applied"):
            ep = p.get("entry_price")
            sp = p.get("stop_price")
            if ep and sp and ep > 0:
                if abs(float(sp) - float(ep)) / float(ep) <= (BE_PCT * 1.5):
                    return "BE"
        if p.get("stop_order_id"):
            return "SL"
        return "MANUAL"
    except Exception:
        return "UNKNOWN"


def _build_history_record(exchange, symbol: str, p: dict, close_price, close_time: datetime, exit_reason: str):
    side = p.get("side")
    entry_price = p.get("entry_price")
    size = p.get("size") or 0
    leverage = p.get("leverage")
    entry_time = p.get("entry_time")
    init_stop_price = p.get("init_stop_price") if p.get("init_stop_price") is not None else p.get("stop_price")

    cs = _get_contract_size(exchange, symbol)
    pnl_usdt = None
    final_rr = None
    try:
        if close_price is not None and entry_price is not None and size and side in ("long", "short"):
            entry_price_f = float(entry_price)
            close_price_f = float(close_price)
            size_f = float(size)
            if side == "long":
                pnl_usdt = (close_price_f - entry_price_f) * size_f * cs
            else:
                pnl_usdt = (entry_price_f - close_price_f) * size_f * cs

            if init_stop_price is not None:
                risk = abs(entry_price_f - float(init_stop_price)) * size_f * cs
                if risk > 0:
                    final_rr = pnl_usdt / risk
    except Exception:
        pnl_usdt = None
        final_rr = None

    return {
        "entry_time": _iso(entry_time),
        "close_time": _iso(close_time),
        "symbol": symbol,
        "leverage": leverage,
        "side": side,
        "entry_price": entry_price,
        "close_price": close_price,
        "final_rr": final_rr,
        "pnl_usdt": pnl_usdt,
        "exit_reason": exit_reason,
    }


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


def save_state(pos_state, last_signal, equity, ohlcv, position_history=None, symbol_risk=None):
    state = {
        "pos_state": _serialize_pos_state(pos_state),
        "last_signal": last_signal,
        "equity": equity,
        "ohlcv": ohlcv,
        "position_history": position_history or [],
        "symbol_risk": symbol_risk or _default_symbol_risk(),
        "timestamp": datetime.utcnow().isoformat(),
    }
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
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

    def _mean_dev(window):
        m = window.mean()
        return (window.sub(m).abs().sum()) / len(window)

    md = tp.rolling(CCI_PERIOD).apply(_mean_dev, raw=False)
    df["cci"] = (tp - sma_tp) / (0.015 * md)

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


def compute_order_size_equal_margin_and_risk(exchange, symbol, entry_price, equity_total, stop_pct, risk_pct=None):
    if entry_price <= 0 or stop_pct <= 0 or equity_total <= 0:
        return 0, 0.0, 0.0

    if risk_pct is None:
        risk_pct = RISK_PER_TRADE

    margin_per_pos = equity_total / MARGIN_DIVISOR
    risk_per_pos = equity_total * risk_pct
    target_notional_for_risk = risk_per_pos / stop_pct

    min_notional = margin_per_pos * 1.0
    max_notional = margin_per_pos * MAX_LEVERAGE
    notional = max(min(target_notional_for_risk, max_notional), min_notional)

    market = exchange.market(symbol)
    contract_size = market.get("contractSize") or float(market["info"].get("ctVal", 1))
    notional_per_contract = entry_price * contract_size

    amount = math.floor(notional / notional_per_contract)
    if amount <= 0:
        return 0, 0.0, 0.0

    actual_notional = amount * notional_per_contract
    actual_leverage = actual_notional / margin_per_pos
    if actual_leverage < 1.0:
        actual_leverage = 1.0
    if actual_leverage > MAX_LEVERAGE:
        actual_leverage = float(MAX_LEVERAGE)

    effective_leverage = actual_notional / equity_total
    return amount, actual_leverage, effective_leverage


def sync_positions(exchange, symbols):
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

        lev_raw = p.get("leverage") or (p.get("info") or {}).get("lever") or None
        try:
            leverage = float(lev_raw) if lev_raw is not None else None
        except Exception:
            leverage = None

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

        notional = None
        try:
            market = exchange.market(sym)
            contract_size = market.get("contractSize") or float(market["info"].get("ctVal", 1))
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

    return {"side": side, "entry_price": entry_price, "stop_price": stop_price, "signal_ts": int(curr["ts"])}


def _safe_float(v):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def main():
    exchange = init_exchange()
    logging.info("CCI + Bollinger 자동매매 (동적 TP + BB/CCI 대시보드 + 균등 증거금/리스크) 시작")

    pos_state = _default_pos_state()
    last_signal_candle_ts = {}
    position_history_cache = load_position_history_cache(POSITION_HISTORY_PATH, POSITION_HISTORY_LIMIT)

    symbol_risk = _default_symbol_risk()

    boot = load_boot_state()
    if boot:
        try:
            pos_state = boot.get("pos_state", pos_state)
            last_signal_candle_ts = boot.get("last_signal", last_signal_candle_ts) or {}
            ph = boot.get("position_history", [])
            if isinstance(ph, list) and ph:
                position_history_cache = ph[:POSITION_HISTORY_LIMIT]

            sr = boot.get("symbol_risk")
            if isinstance(sr, dict):
                symbol_risk = sr

            logging.info("재시작 동기화 완료: pos_state/last_signal/position_history/symbol_risk 복구")
        except Exception:
            pass

    while True:
        try:
            data = {}
            for sym in SYMBOLS:
                df = fetch_ohlcv_df(exchange, sym, TIMEFRAME, 200)
                if df is None:
                    continue
                df = calculate_indicators(df)
                if len(df) < BB_PERIOD + 3:
                    continue
                data[sym] = (df, df.iloc[-2], df.iloc[-1])

            if not data:
                time.sleep(LOOP_INTERVAL)
                continue

            exch_positions = sync_positions(exchange, SYMBOLS)

            for sym in SYMBOLS:
                if not exch_positions[sym]["has_position"]:
                    # 포지션이 사라진 경우(주로 SL/BE 또는 수동 청산) 히스토리 기록 + 리스크 조정
                    if pos_state[sym].get("side") is not None and (pos_state[sym].get("size") or 0) > 0:
                        try:
                            close_price = None
                            if sym in data:
                                close_price = float(data[sym][2]["close"])
                            close_time = datetime.now(timezone.utc)
                            exit_reason = _infer_exit_reason(pos_state[sym])

                            # 손절이면 리스크를 REDUCED_RISK로 변경, 익절이면 BASE_RISK로 복원
                            if exit_reason == "SL":
                                symbol_risk[sym] = REDUCED_RISK
                            elif exit_reason == "TP":
                                symbol_risk[sym] = BASE_RISK

                            rec = _build_history_record(exchange, sym, pos_state[sym], close_price, close_time, exit_reason)
                            append_position_history(position_history_cache, rec)
                        except Exception:
                            pass

                    pos_state[sym] = {
                        "side": None,
                        "size": 0,
                        "entry_price": None,
                        "stop_price": None,
                        "init_stop_price": None,
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
                curr_cci = float(curr["cci"])
                prev_cci = float(prev["cci"])

                pos_state[sym]["tp_price"] = (bb_upper * (1 - TP_OFFSET) if side == "long" else bb_lower * (1 + TP_OFFSET))

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
                                        fee_line = entry_price * (1 + FEE_PCT)
                                        be_sl    = entry_price * (1 + BE_PCT)
                                        in_range = (fee_line <= curr_price <= tp_price)
                                        sl_side = "sell"
                                        better_than_current = (stop_price < be_sl)
                                    else:
                                        fee_line = entry_price * (1 - FEE_PCT)
                                        be_sl    = entry_price * (1 - BE_PCT)
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
                                                sym, "market", sl_side, size,
                                                params={"tdMode": "cross", "reduceOnly": True, "stopLossPrice": be_sl},
                                            )
                                            pos_state[sym]["stop_order_id"] = sl_order.get("id")
                                            pos_state[sym]["stop_price"] = be_sl
                                            pos_state[sym]["be_sl_applied"] = True
                                        except Exception:
                                            pass
                except Exception:
                    pass

                if (side == "long" and curr_price >= pos_state[sym]["tp_price"]) or (side == "long" and (prev_cci >= 110) and (prev_cci - curr_cci >= 15)):
                    if pos_state[sym]["stop_order_id"]:
                        try:
                            exchange.cancel_order(pos_state[sym]["stop_order_id"], sym)
                        except Exception:
                            pass

                    exch_now = sync_positions(exchange, SYMBOLS)[sym]
                    close_order = {}
                    if exch_now["has_position"]:
                        close_order = exchange.create_order(sym, "market", "sell", exch_now["size"], params={"tdMode": "cross"})

                    try:
                        close_time = datetime.now(timezone.utc)
                        _cp = None
                        try:
                            _cp = close_order.get("average") or close_order.get("price")
                        except Exception:
                            _cp = None
                        close_price = float(_cp) if _cp is not None else curr_price

                        rec = _build_history_record(exchange, sym, pos_state[sym], close_price, close_time, "TP")
                        append_position_history(position_history_cache, rec)
                    except Exception:
                        pass

                    symbol_risk[sym] = BASE_RISK
                    pos_state[sym] = _default_pos_state()[sym]

                elif (side == "short" and curr_price <= pos_state[sym]["tp_price"]) or (side == "short" and (prev_cci <= -110) and (curr_cci - prev_cci >= 15)):
                    if pos_state[sym]["stop_order_id"]:
                        try:
                            exchange.cancel_order(pos_state[sym]["stop_order_id"], sym)
                        except Exception:
                            pass

                    exch_now = sync_positions(exchange, SYMBOLS)[sym]
                    close_order = {}
                    if exch_now["has_position"]:
                        close_order = exchange.create_order(sym, "market", "buy", exch_now["size"], params={"tdMode": "cross"})

                    try:
                        close_time = datetime.now(timezone.utc)
                        _cp = None
                        try:
                            _cp = close_order.get("average") or close_order.get("price")
                        except Exception:
                            _cp = None
                        close_price = float(_cp) if _cp is not None else curr_price

                        rec = _build_history_record(exchange, sym, pos_state[sym], close_price, close_time, "TP")
                        append_position_history(position_history_cache, rec)
                    except Exception:
                        pass

                    symbol_risk[sym] = BASE_RISK
                    pos_state[sym] = _default_pos_state()[sym]

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

                if side_signal == "long" and stop_price >= entry_price:
                    continue
                if side_signal == "short" and stop_price <= entry_price:
                    continue

                bb_upper = float(curr["bb_upper"])
                bb_lower = float(curr["bb_lower"])
                tp_price = (bb_upper * (1 - TP_OFFSET) if side_signal == "long" else bb_lower * (1 + TP_OFFSET))

                stop_diff = abs(entry_price - stop_price)
                tp_diff = abs(entry_price - tp_price)
                if stop_diff <= 0:
                    continue

                R = tp_diff / stop_diff
                if R < R_THRESHOLD:
                    continue

                _, total = fetch_futures_equity(exchange)
                if total <= 0:
                    continue

                stop_pct = stop_diff / entry_price

                amount, leverage, eff_lev = compute_order_size_equal_margin_and_risk(exchange, sym, entry_price, total, stop_pct, symbol_risk[sym])
                if amount <= 0:
                    continue

                order_side = "buy" if side_signal == "long" else "sell"
                sl_side = "sell" if side_signal == "long" else "buy"

                lev_float = max(1.0, min(round(float(leverage), 2), float(MAX_LEVERAGE)))
                try:
                    exchange.set_leverage(lev_float, sym, params={"mgnMode": "cross"})
                except Exception:
                    pass

                exchange.create_order(sym, "market", order_side, amount, params={"tdMode": "cross"})

                time.sleep(0.3)
                after = sync_positions(exchange, SYMBOLS)[sym]
                actual_entry = after["entry_price"] or entry_price
                actual_size = after["size"]

                pos_state[sym]["side"] = side_signal
                pos_state[sym]["size"] = actual_size
                pos_state[sym]["entry_price"] = actual_entry
                pos_state[sym]["stop_price"] = stop_price
                pos_state[sym]["init_stop_price"] = stop_price
                pos_state[sym]["entry_time"] = datetime.now(timezone.utc)
                pos_state[sym]["entry_candle_ts"] = curr_ts
                pos_state[sym]["leverage"] = after.get("leverage")
                pos_state[sym]["margin"] = after.get("margin")
                pos_state[sym]["notional"] = after.get("notional")
                pos_state[sym]["be_sl_applied"] = False

                try:
                    sl_order = exchange.create_order(
                        sym, "market", sl_side, actual_size,
                        params={"tdMode": "cross", "reduceOnly": True, "stopLossPrice": stop_price},
                    )
                    pos_state[sym]["stop_order_id"] = sl_order.get("id")
                except Exception:
                    pos_state[sym]["stop_order_id"] = None

                last_signal_candle_ts[sym] = curr_ts

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
            save_state(pos_state, last_signal_candle_ts, total, ohlcv_state, position_history_cache, symbol_risk)

            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            logging.warning(f"메인 루프 에러: {e}")
            time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
