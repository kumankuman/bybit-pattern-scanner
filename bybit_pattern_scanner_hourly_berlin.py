# -*- coding: utf-8 -*-
# bybit_pattern_scanner_hourly_berlin.py
# Python 3.10+
# pip install -r requirements.txt
# Поведение:
# - При первом запуске: стартовый опрос ПОСЛЕДНИХ ЗАКРЫТЫХ свечей (1H/4H/1D).
# - Далее: раз в ЧАС на начале часа (в обычном режиме).
# - Время в выводе/логе — Europe/Berlin.
# - Для КАЖДОГО symbol|TF анализируем ПОСЛЕДНЮЮ ЗАКРЫТУЮ свечу:
#   1) Сначала определяем, есть ли паттерн(ы).
#   2) Если есть — проверяем фильтры и показываем причины отклонения или СИГНАЛ.
#   3) Если нет — пишем "нет паттернов: [ожидаемые]".
# - Сигналы пишутся в signals_log.csv ТОЛЬКО если эта свеча ещё не логировалась.
# - Вывод отсортирован: символы по алфавиту, TF — 1H, 4H, 1D.

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
import time

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from requests.exceptions import RequestException  # оставлено на случай расширений
import pytz

# -------------------------
# Конфигурация
# -------------------------

CATEGORY = "linear"
SYMBOLS = [
    "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "LTCUSDT", "POLUSDT", "SOLUSDT", "XRPUSDT"
]  # алфавит

TF_MAP = {"1H": "60", "4H": "240", "1D": "D"}
TF_SORT = ["1H", "4H", "1D"]

# Стратегии по согласованным правилам
# (pattern_name, side, tpR, kATR)
STRATEGIES: Dict[str, Dict[str, List[Tuple[str, str, float, float]]]] = {
    "BTCUSDT": {"1D": [("bullish_engulfing", "long", 1.0, 0.1)],
                "4H": [("bearish_engulfing", "short", 1.0, 0.1)],
                "1H": [("hammer", "long", 1.0, 0.1)]},
    "ETHUSDT": {"4H": [("bearish_engulfing", "short", 1.0, 0.0),
                       ("hammer", "long", 2.0, 0.0)],
                "1H": [("bullish_engulfing", "long", 1.0, 0.0)]},
    "SOLUSDT": {"4H": [("shooting_star", "short", 2.0, 0.0),
                       ("evening_star", "short", 1.0, 0.1),
                       ("morning_star", "long", 1.0, 0.1)]},
    "XRPUSDT": {"4H": [("evening_star", "short", 1.0, 0.0),
                       ("morning_star", "long", 1.0, 0.0)]},
    "DOGEUSDT": {"4H": [("shooting_star", "short", 3.0, 0.0)],
                 "1D": [("bullish_engulfing", "long", 2.0, 0.0)]},
    "BNBUSDT": {"4H": [("evening_star", "short", 1.0, 0.1),
                       ("shooting_star", "short", 2.0, 0.0)],
                "1D": [("bullish_engulfing", "long", 1.0, 0.0)]},
    "ADAUSDT": {"4H": [("evening_star", "short", 1.0, 0.1),
                       ("morning_star", "long", 1.0, 0.0)],
                "1D": [("bullish_engulfing", "long", 1.0, 0.0)]},
    "AVAXUSDT": {"4H": [("evening_star", "short", 1.0, 0.1),
                        ("hammer", "long", 2.0, 0.0)]},
    "POLUSDT": {"1D": [("bearish_engulfing", "short", 2.0, 0.0)]},  # 1H исключён
    "LTCUSDT": {"4H": [("bullish_engulfing", "long", 1.0, 0.0)],
                "1H": [("morning_star", "long", 1.0, 0.1)],
                "1D": [("bearish_engulfing", "short", 1.0, 0.0)]},
}

# Фильтры
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_MULT_LIMIT = 2.0  # ATR <= 2 × медианы
RVOL_LEN = 20
RVOL_MIN = {"default": 1.2, ("LTCUSDT", "1H"): 1.2, ("BTCUSDT", "1H"): 1.2, ("POLUSDT", "1D"): 1.2}

# Ночной фильтр для LTC 1H (UTC 00:00–06:00)
AVOID_UTC_HOURS = {("LTCUSDT", "1H"): set(range(0, 6))}

# Файлы состояния/логов
LOG_FILE = "signals_log.csv"
STATE_FILE = "last_processed.json"

# Пул потоков
MAX_WORKERS = min(16, len(SYMBOLS) * 3)

# Временные зоны
BERLIN = pytz.timezone("Europe/Berlin")

# -------------------------
# Время/утилиты
# -------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def to_berlin(dt_utc: datetime) -> datetime:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(BERLIN)

def fmt_berlin(dt_utc: datetime) -> str:
    return to_berlin(dt_utc).strftime("%Y-%m-%d %H:%M %Z")

def tf_timedelta(tf: str) -> timedelta:
    if tf == "1H":
        return timedelta(hours=1)
    if tf == "4H":
        return timedelta(hours=4)
    if tf == "1D":
        return timedelta(days=1)
    return timedelta(minutes=1)

def sleep_until_next_top_of_hour():
    now = datetime.now()
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    sleep_s = (next_hour - now).total_seconds()
    if sleep_s > 0:
        time.sleep(sleep_s)

# -------------------------
# Данные/индикаторы
# -------------------------

def to_df(klines) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame(columns=["start", "open", "high", "low", "close", "volume", "turnover"])
    first = klines[0]
    if isinstance(first, dict) and {"open", "high", "low", "close"}.issubset(first.keys()):
        df = pd.DataFrame(klines)
        if "turnover" not in df.columns and "turnoverUsd" in df.columns:
            df = df.rename(columns={"turnoverUsd": "turnover"})
    elif isinstance(first, dict) and {"openPrice", "highPrice", "lowPrice", "closePrice"}.issubset(first.keys()):
        df = pd.DataFrame(klines).rename(columns={
            "openPrice": "open", "highPrice": "high", "lowPrice": "low", "closePrice": "close"
        })
        if "turnover" not in df.columns and "turnoverUsd" in df.columns:
            df = df.rename(columns={"turnoverUsd": "turnover"})
    elif isinstance(first, (list, tuple)):
        cols = ["start", "open", "high", "low", "close", "volume", "turnover"]
        norm = []
        for row in klines:
            row = list(row)
            if len(row) < 7:
                row = row + [None] * (7 - len(row))
            norm.append(row[:7])
        df = pd.DataFrame(norm, columns=cols)
    else:
        return pd.DataFrame(columns=["start", "open", "high", "low", "close", "volume", "turnover"])
    if "start" in df.columns:
        df["start"] = pd.to_datetime(pd.to_numeric(df["start"], errors="coerce"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "start" in df.columns:
        df = df.sort_values("start").reset_index(drop=True)
    for col in ["start", "open", "high", "low", "close", "volume", "turnover"]:
        if col not in df.columns:
            df[col] = np.nan
    return df[["start", "open", "high", "low", "close", "volume", "turnover"]]

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(window=period, min_periods=period).mean()

def volume_rvol(df: pd.DataFrame, length: int) -> pd.Series:
    v = df["turnover"].astype(float)
    if v.isna().all() or v.sum() == 0:
        v = df["volume"].astype(float)
    avg = v.rolling(length).mean()
    return v / avg

def body(df, i): return (df.loc[i, "close"] - df.loc[i, "open"])
def is_bull(df, i): return df.loc[i, "close"] > df.loc[i, "open"]
def is_bear(df, i): return df.loc[i, "close"] < df.loc[i, "open"]

# -------------------------
# Паттерны
# -------------------------

def bullish_engulfing(df, i) -> bool:
    if i < 1: return False
    return (is_bear(df, i-1) and is_bull(df, i)
            and df.loc[i, "close"] >= df.loc[i-1, "open"]
            and df.loc[i, "open"] <= df.loc[i-1, "close"])

def bearish_engulfing(df, i) -> bool:
    if i < 1: return False
    return (is_bull(df, i-1) and is_bear(df, i)
            and df.loc[i, "close"] <= df.loc[i-1, "open"]
            and df.loc[i, "open"] >= df.loc[i-1, "close"])

def hammer(df, i) -> bool:
    o, h, l, c = df.loc[i, ["open", "high", "low", "close"]]
    real_body = abs(c - o)
    if (h - l) == 0: return False
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    return (real_body > 0 and lower_shadow >= 2 * real_body and upper_shadow <= real_body)

def shooting_star(df, i) -> bool:
    o, h, l, c = df.loc[i, ["open", "high", "low", "close"]]
    real_body = abs(c - o)
    if (h - l) == 0: return False
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    return (real_body > 0 and upper_shadow >= 2 * real_body and lower_shadow <= real_body)

def morning_star(df, i) -> bool:
    if i < 2: return False
    c1 = is_bear(df, i-2)
    small2 = abs(body(df, i-1)) < abs(body(df, i-2)) * 0.5
    c3 = is_bull(df, i)
    cond3 = df.loc[i, "close"] >= (df.loc[i-2, "open"] + df.loc[i-2, "close"]) / 2
    return c1 and small2 and c3 and cond3

def evening_star(df, i) -> bool:
    if i < 2: return False
    c1 = is_bull(df, i-2)
    small2 = abs(body(df, i-1)) < abs(body[df, i-2]) * 0.5 if False else abs(body(df, i-1)) < abs(body(df, i-2)) * 0.5
    # (выше трюк для подсветки; фактически то же условие:)
    small2 = abs(body(df, i-1)) < abs(body(df, i-2)) * 0.5
    c3 = is_bear(df, i)
    cond3 = df.loc[i, "close"] <= (df.loc[i-2, "open"] + df.loc[i-2, "close"]) / 2
    return c1 and small2 and c3 and cond3

PATTERN_FUNCS = {
    "bullish_engulfing": bullish_engulfing,
    "bearish_engulfing": bearish_engulfing,
    "hammer": hammer,
    "shooting_star": shooting_star,
    "morning_star": morning_star,
    "evening_star": evening_star,
}

# -------------------------
# Фильтры
# -------------------------

def trend_ok(df: pd.DataFrame, i: int, side: str) -> bool:
    ema50 = ema(df["close"], EMA_PERIOD)
    if pd.isna(ema50.iloc[i]): return False
    if side == "long":
        return df.loc[i, "close"] > ema50.iloc[i] and ema50.iloc[i] >= ema50.iloc[i-1]
    else:
        return df.loc[i, "close"] < ema50.iloc[i] and ema50.iloc[i] <= ema50.iloc[i-1]

def atr_ok(df: pd.DataFrame, i: int) -> bool:
    a = atr(df, ATR_PERIOD)
    if pd.isna(a.iloc[i]): return False
    med = a.rolling(100).median().iloc[i] if i >= 99 else a.iloc[:i+1].median()
    return a.iloc[i] <= ATR_MULT_LIMIT * med

def rvol_ok(df: pd.DataFrame, i: int, thr: float) -> bool:
    r = volume_rvol(df, RVOL_LEN)
    if pd.isna(r.iloc[i]): return False
    return r.iloc[i] >= thr

def confirmation_level(df: pd.DataFrame, i: int, side: str) -> float:
    return float(df.loc[i, "high"] if side == "long" else df.loc[i, "low"])

def session_ok(symbol: str, tf: str, ts: pd.Timestamp) -> bool:
    key = (symbol, tf)
    if key not in AVOID_UTC_HOURS:
        return True
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.hour not in AVOID_UTC_HOURS[key]

def get_rvol_threshold(symbol: str, tf: str) -> float:
    return RVOL_MIN.get((symbol, tf), RVOL_MIN["default"])

# -------------------------
# Лог/состояние
# -------------------------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state: dict):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)

def append_log(row: dict):
    file_exists = os.path.exists(LOG_FILE)
    pd.DataFrame([row]).to_csv(
        LOG_FILE, mode="a", header=not file_exists, index=False, encoding="utf-8"
    )

# -------------------------
# Клиент Bybit
# -------------------------

class BybitClient:
    def __init__(self, max_retries=5, backoff=1.5):
        self.client = HTTP(testnet=False)  # публичные kline ключей не требуют
        self.max_retries = max_retries
        self.backoff = backoff

    def get_klines(self, symbol: str, interval: str, limit: int = 500):
        params = dict(category=CATEGORY, symbol=symbol, interval=interval, limit=min(limit, 1000))
        retry = 0
        while True:
            try:
                data = self.client.get_kline(**params)

                # Проверка retCode/retMsg
                if isinstance(data, dict) and data.get("retCode") not in (0, None):
                    rc = data.get("retCode")
                    msg = data.get("retMsg")
                    raise RuntimeError(f"Bybit retCode={rc}, retMsg={msg}, params={params}")

                lst = (data.get("result", {}) or {}).get("list", [])
                if lst is None:
                    lst = []
                return list(lst)

            except Exception as e:
                retry += 1
                print(f"[get_klines] {symbol} {interval} attempt={retry} error={type(e).__name__}: {e}", flush=True)
                if getattr(e, "args", None):
                    msg0 = str(e.args[0])
                    if len(msg0) < 1000:
                        print(f"[get_klines] args0={msg0}", flush=True)
            if retry > self.max_retries:
                raise
            time.sleep(self.backoff ** retry)

# -------------------------
# Обработка symbol|TF: последняя ЗАКРЫТАЯ свеча
# -------------------------

def process_symbol_tf(client: BybitClient, symbol: str, tf: str, last_processed_ts_ms: Optional[int]):
    interval = TF_MAP[tf]
    klines = client.get_klines(symbol, interval, limit=500)
    df = to_df(klines)
    min_len = max(EMA_PERIOD + 5, ATR_PERIOD + 5, RVOL_LEN + 5, 60)
    if df.empty or len(df) < min_len:
        return {
            "symbol": symbol, "tf": tf, "status": "мало данных", "reason": "недостаточно истории",
            "debug": None, "signals": [], "already_logged": True
        }

    # найдём ПОСЛЕДНЮЮ ЗАКРЫТУЮ свечу
    now = now_utc()
    delta = tf_timedelta(tf)
    i = None
    for j in range(len(df) - 1, -1, -1):
        st = df.loc[j, "start"]
        if pd.isna(st):
            continue
        if (st + delta) <= now:
            i = j
            break
    if i is None:
        return {
            "symbol": symbol, "tf": tf, "status": "нет закрытых свечей на момент опроса",
            "reason": "ожидаем закрытие TF", "debug": None, "signals": [], "already_logged": True
        }

    ts_utc = df.loc[i, "start"]
    ts_close_utc = ts_utc + delta
    ts_ms = int(ts_utc.value // 10**6)
    already_logged = (last_processed_ts_ms is not None and ts_ms <= last_processed_ts_ms)

    # индикаторы
    ema50_series = ema(df["close"], EMA_PERIOD)
    atr14_series = atr(df, ATR_PERIOD)
    rvol20_series = volume_rvol(df, RVOL_LEN)
    ema50_val = float(ema50_series.iloc[i]) if not pd.isna(ema50_series.iloc[i]) else float("nan")
    atr14_val = float(atr14_series.iloc[i]) if not pd.isna(atr14_series.iloc[i]) else float("nan")
    rvol20_val = float(rvol20_series.iloc[i]) if not pd.isna(rvol20_series.iloc[i]) else float("nan")

    debug_row = {
        "start_berlin": fmt_berlin(ts_utc),
        "close_berlin": fmt_berlin(ts_close_utc),
        "utc_start": ts_utc.isoformat(),
        "open": float(df.loc[i, "open"]),
        "high": float(df.loc[i, "high"]),
        "low": float(df.loc[i, "low"]),
        "close": float(df.loc[i, "close"]),
        "volume": float(df.loc[i, "volume"]) if not pd.isna(df.loc[i, "volume"]) else float("nan"),
        "turnover": float(df.loc[i, "turnover"]) if not pd.isna(df.loc[i, "turnover"]) else float("nan"),
        "ema50": ema50_val,
        "atr14": atr14_val,
        "rvol20": rvol20_val,
    }

    # 1) сначала — какие паттерны есть (без фильтров)
    cfg = STRATEGIES.get(symbol, {}).get(tf, [])
    expected_names = [p for (p, _, _, _) in cfg]
    matched = []
    for (pname, side, tpR, kATR) in cfg:
        f = PATTERN_FUNCS[pname]
        if f(df, i):
            matched.append((pname, side, tpR, kATR))

    if not matched:
        return {
            "symbol": symbol, "tf": tf,
            "status": f"последняя закрытая свеча @ {fmt_berlin(ts_utc)}",
            "reason": "нет паттернов: " + (", ".join(expected_names) if expected_names else "—"),
            "debug": debug_row, "signals": [],
            "already_logged": already_logged, "ts_ms": ts_ms
        }

    # 2) если паттерны есть — проверяем фильтры
    signals = []
    rejections = []
    rvol_thr = get_rvol_threshold(symbol, tf)
    session_ok_flag = session_ok(symbol, tf, ts_utc)
    atr_ok_flag = atr_ok(df, i)
    for (pname, side, tpR, kATR) in matched:
        reasons = []
        if not session_ok_flag:
            reasons.append("session")
        if not atr_ok_flag:
            reasons.append("ATR")
        if not rvol_ok(df, i, rvol_thr):
            reasons.append(f"RVOL<thr({rvol_thr})")
        if not trend_ok(df, i, side):
            reasons.append("trend")

        if reasons:
            rejections.append(f"{pname} {side}: отклонён — " + ", ".join(reasons))
            continue

        confirm = confirmation_level(df, i, side)
        row = {
            "start_berlin": debug_row["start_berlin"],
            "utc_start": ts_utc.isoformat(),
            "symbol": symbol,
            "tf": tf,
            "pattern": pname,
            "side": side,
            "open": debug_row["open"],
            "high": debug_row["high"],
            "low": debug_row["low"],
            "close": debug_row["close"],
            "confirm_level": float(confirm),
            "ema50": ema50_val,
            "atr14": atr14_val,
            "rvol20": rvol20_val,
            "tpR": tpR,
            "kATR": kATR,
            "ts_ms": ts_ms,
        }
        signals.append(row)

    if signals:
        return {
            "symbol": symbol, "tf": tf,
            "status": f"последняя закрытая свеча @ {fmt_berlin(ts_utc)}",
            "reason": "сигнал(ы) найден",
            "debug": debug_row, "signals": signals,
            "already_logged": already_logged, "ts_ms": ts_ms
        }
    else:
        return {
            "symbol": symbol, "tf": tf,
            "status": f"последняя закрытая свеча @ {fmt_berlin(ts_utc)}",
            "reason": "; ".join(rejections) if rejections else "паттерн есть, но причины не указаны",
            "debug": debug_row, "signals": [],
            "already_logged": already_logged, "ts_ms": ts_ms
        }

# -------------------------
# Печать баннера
# -------------------------

def print_start_banner():
    print("Онлайн-сканер паттернов — старт")
    print(f"Сейчас (Берлин): {fmt_berlin(now_utc())}")
    print(f"Категория: {CATEGORY}")
    print(f"Журнал: {LOG_FILE}")
    print(f"Кэш: {STATE_FILE}")
    print("Стартовый опрос последних закрытых свечей; далее — строго раз в час на начале часа.")
    print("Берём только ЗАКРЫТЫЕ свечи 1H/4H/1D.")
    print("Порядок обхода:")
    for s in SYMBOLS:
        tfs = [tf for tf in TF_SORT if tf in STRATEGIES.get(s, {})]
        if tfs:
            print(f"  {s}: {', '.join(tfs)}")
    print("-" * 80)

def print_cycle_header(cycle_idx: int, prefix: str = ""):
    tag = f"{prefix} " if prefix else ""
    print(f"\n[{fmt_berlin(now_utc())}] {tag}Цикл #{cycle_idx}: проверка последних закрытых свечей…")

# -------------------------
# Один проход сканирования (с сортированным выводом)
# -------------------------

def run_scan_cycle(client: BybitClient, state: dict, cycle_idx: int, prefix: str = "") -> int:
    print_cycle_header(cycle_idx, prefix)

    tasks = [(s, tf) for s in SYMBOLS for tf in TF_SORT if tf in STRATEGIES.get(s, {})]

    results: Dict[Tuple[str, str], dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futmap = {
            ex.submit(process_symbol_tf, client, s, tf, state.get(f"{s}|{tf}")): (s, tf)
            for (s, tf) in tasks
        }
        for fut in as_completed(futmap):
            s, tf = futmap[fut]
            try:
                results[(s, tf)] = fut.result()
            except Exception as e:
                results[(s, tf)] = {
                    "symbol": s, "tf": tf,
                    "status": "ошибка запроса/обработки",
                    "reason": f"{type(e).__name__}: {e}",
                    "debug": None, "signals": [], "already_logged": True
                }

    found_cnt = 0
    for (s, tf) in tasks:
        res = results[(s, tf)]
        status = res["status"]
        reason = res["reason"]
        dbg = res["debug"]
        sigs = res["signals"]
        already_logged = res.get("already_logged", True)
        ts_ms = res.get("ts_ms", None)

        print(f"  {s} {tf}: {status}")
        print(f"    причина: {reason}")

        if sigs:
            for row in sigs:
                found_cnt += 1
                print(
                    f"    ? СИГНАЛ {row['symbol']} {row['tf']} {row['pattern']} {row['side'].upper()} "
                    f"close={row['close']:.8f} confirm={row['confirm_level']:.8f} "
                    f"EMA50={row['ema50']:.8f} ATR14={row['atr14']:.8f} RVOL20={row['rvol20']:.2f} "
                    f"TP={row['tpR']}R k={row['kATR']} ({row['start_berlin']})"
                )
            key = f"{s}|{tf}"
            if ts_ms is not None and not already_logged:
                for row in sigs:
                    append_log(row)
                state[key] = ts_ms

    return found_cnt

# -------------------------
# Основной цикл
# -------------------------

def main_loop():
    client = BybitClient()
    state = load_state()
    # инициализация ключей состояния
    for s in SYMBOLS:
        for tf in STRATEGIES.get(s, {}).keys():
            state.setdefault(f"{s}|{tf}", None)
    save_state(state)

    print_start_banner()

    # стартовый опрос последних закрытых свечей
    found_first = run_scan_cycle(client, state, cycle_idx=0, prefix="Стартовый")
    save_state(state)
    print(f"Итого по стартовому циклу: новых сигналов — {found_first}.")

    # далее — строго по началу часа
    sleep_until_next_top_of_hour()

    cycle_idx = 0
    while True:
        cycle_idx += 1
        found_cnt = run_scan_cycle(client, state, cycle_idx)
        save_state(state)
        print(f"Итого по циклу: новых сигналов — {found_cnt}. Следующая проверка на начале следующего часа.")
        sleep_until_next_top_of_hour()

# -------------------------
# Точка входа (режимы: обычный / --once)
# -------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Выполнить один проход и выйти")
    args = parser.parse_args()

    client = BybitClient()
    state = load_state()
    for s in SYMBOLS:
        for tf in STRATEGIES.get(s, {}).keys():
            state.setdefault(f"{s}|{tf}", None)
    save_state(state)

    print_start_banner()

    if args.once or os.getenv("RUN_ONCE") == "1":
        found_cnt = run_scan_cycle(client, state, cycle_idx=0, prefix="Запуск по расписанию")
        save_state(state)
        print(f"Итого по циклу: новых сигналов — {found_cnt}. Завершение.")
    else:
        found_first = run_scan_cycle(client, state, cycle_idx=0, prefix="Стартовый")
        save_state(state)
        print(f"Итого по стартовому циклу: новых сигналов — {found_first}.")
        sleep_until_next_top_of_hour()
        cycle_idx = 0
        while True:
            cycle_idx += 1
            found_cnt = run_scan_cycle(client, state, cycle_idx)
            save_state(state)
            print(f"Итого по циклу: новых сигналов — {found_cnt}. Следующая проверка на начале следующего часа.")
            sleep_until_next_top_of_hour()
