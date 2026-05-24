import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import numpy as np

from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="專業當沖輔助系統", layout="centered")

FUGLE_BASE_URL = "https://api.fugle.tw/marketdata/v1.0/stock"
FINMIND_BASE_URL = "https://api.finmindtrade.com/api/v4/data"
TW_TZ = timezone(timedelta(hours=8))


# =========================
# 基礎工具
# =========================
def tw_now():
    return datetime.now(TW_TZ)

def now_hhmm():
    return tw_now().strftime("%H:%M")

def fmt_num(x, digits=2):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"

def safe_float(x, default=0.0):
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def volume_to_human(v):
    try:
        v = float(v)
    except Exception:
        return "-"
    if v >= 100000000:
        return f"{v/100000000:.2f}億"
    if v >= 10000:
        return f"{v/10000:.2f}萬"
    return f"{int(v)}"

def is_us_symbol(symbol: str) -> bool:
    s = (symbol or "").strip().upper()
    if not s:
        return False
    return s.isalpha()

def market_label(symbol: str) -> str:
    return "美股" if is_us_symbol(symbol) else "台股"

def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return default


# =========================
# 補強模組一：選股條件（高低點墊高 / 量能 / 流動性 / 非末端）
# =========================
def check_higher_highs_lows(df, lookback=10):
    if df is None or len(df) < lookback + 1:
        return {"trend_valid": False, "hh_count": 0, "hl_count": 0,
                "description": "資料不足，無法判斷趨勢結構"}
    recent = df.iloc[-(lookback + 1):].reset_index(drop=True)
    hh_count = 0
    hl_count = 0
    for i in range(1, len(recent)):
        if recent.iloc[i]["max"] > recent.iloc[i - 1]["max"]:
            hh_count += 1
        if recent.iloc[i]["min"] > recent.iloc[i - 1]["min"]:
            hl_count += 1
    threshold = int(lookback * 0.6)
    trend_valid = (hh_count + hl_count) >= threshold
    if trend_valid:
        description = f"近{lookback}根K棒中，高點墊高{hh_count}次、低點墊高{hl_count}次，趨勢結構符合選股條件。"
    else:
        description = f"近{lookback}根K棒中，高點墊高{hh_count}次、低點墊高{hl_count}次，趨勢結構尚未符合墊高條件，建議觀望。"
    return {"trend_valid": trend_valid, "hh_count": hh_count, "hl_count": hl_count, "description": description}


def check_dead_water(df, lookback=5):
    if df is None or len(df) < lookback + 1:
        return {"liquid": False, "description": "資料不足"}
    recent_vol = df["Trading_Volume"].iloc[-lookback:]
    avg_vol = recent_vol.mean()
    prev_avg = df["Trading_Volume"].iloc[-(lookback * 2):-lookback].mean()
    if prev_avg == 0:
        return {"liquid": False, "description": "前期量能為零，無法比較"}
    vol_ratio = avg_vol / prev_avg
    liquid = vol_ratio >= 0.6
    description = (
        f"近{lookback}日平均量能為前期的{vol_ratio:.2f}倍，"
        f"{'量能充足' if liquid else '量能明顯萎縮，屬死水行情，不符選股條件'}。"
    )
    return {"liquid": liquid, "vol_ratio": round(vol_ratio, 2), "description": description}


def screen_stock_conditions(df, stock_id, is_us=False):
    conditions = []
    trend = check_higher_highs_lows(df)
    conditions.append({"name": "高低點墊高趨勢", "pass": trend["trend_valid"], "detail": trend["description"]})

    liq = check_dead_water(df)
    conditions.append({"name": "近5日量能充足", "pass": liq["liquid"], "detail": liq["description"]})

    if df is not None and len(df) >= 5:
        avg_vol_5 = df["Trading_Volume"].iloc[-5:].mean()
        if is_us:
            liquid_enough = avg_vol_5 >= 500000
        else:
            liquid_enough = (avg_vol_5 / 1000) >= 500
        conditions.append({
            "name": "流動性充足",
            "pass": liquid_enough,
            "detail": f"近5日均量{'充足' if liquid_enough else '不足，屬冷門股，不符選股條件'}。"
        })
    else:
        conditions.append({"name": "流動性充足", "pass": False, "detail": "資料不足，無法判斷流動性。"})

    end_of_run = False
    if df is not None and len(df) >= 4:
        last3 = df.iloc[-3:]
        long_red_count = 0
        for _, row in last3.iterrows():
            body = (row["close"] - row["open"]) / row["open"] * 100 if row["open"] != 0 else 0
            if body >= 3.0:
                long_red_count += 1
        end_of_run = long_red_count >= 2
    conditions.append({
        "name": "非末端噴出",
        "pass": not end_of_run,
        "detail": ("近3根K棒中出現2根以上長紅，疑似末段噴出，不符選股條件。"
                   if end_of_run else "K棒結構未出現連續長紅末端，符合選股條件。")
    })

    pass_all = all(c["pass"] for c in conditions)
    fail_items = [c["name"] for c in conditions if not c["pass"]]
    summary = ("四項選股條件全部通過，可進入門號判斷。"
               if pass_all else f"以下條件未通過，建議直接略過此標的：{'、'.join(fail_items)}。")
    return {"pass_all": pass_all, "conditions": conditions, "summary": summary}


# =========================
# 補強模組二：5號門 / 6號門 逐項布林確認
# =========================
def check_market_not_crashing(us_block):
    if us_block is None or not us_block.get("indices"):
        return True, "無法取得大盤資料，請手動確認大盤狀況。"
    crash_flags = [idx.get("change_pct", 0) < -1.5 for idx in us_block["indices"]]
    if any(crash_flags):
        return False, "對應大盤指數出現急殺（跌幅逾1.5%），6號門條件不成立，禁止進場。"
    return True, "大盤未出現急殺，大盤條件通過。"


def check_breakout_prev_high(df, lookback=20):
    if df is None or len(df) < lookback + 1:
        return False, "資料不足，無法確認是否突破前高。"
    today = df.iloc[-1]
    prev_window = df.iloc[-(lookback + 1):-1]
    prev_high = prev_window["max"].max()
    breakout = today["close"] >= prev_high
    return breakout, (
        f"今日收盤{today['close']:.2f}，近{lookback}日前高{prev_high:.2f}，"
        f"{'已突破前高，條件通過。' if breakout else '尚未突破前高，6號門條件不成立。'}"
    )


def check_close_above_pressure(today_row, pressure):
    close = float(today_row["close"])
    high = float(today_row["max"])
    close_above = close >= pressure
    only_intraday = (high >= pressure) and (close < pressure)
    if close_above:
        return True, f"收盤{close:.2f}站穩壓力{pressure:.2f}以上，收盤確認通過。"
    elif only_intraday:
        return False, f"盤中高點{high:.2f}觸及壓力{pressure:.2f}，但收盤{close:.2f}未站穩，屬盤中假突破，6號門不成立。"
    else:
        return False, f"收盤{close:.2f}低於壓力{pressure:.2f}，尚未突破。"


def check_red_candle(today_row):
    close = float(today_row["close"])
    open_p = float(today_row["open"])
    body_pct = (close - open_p) / open_p * 100 if open_p != 0 else 0
    is_red = body_pct >= 0.3
    return is_red, (
        f"今日為紅K（漲幅{body_pct:.2f}%），條件通過。"
        if is_red else f"今日非紅K（漲幅{body_pct:.2f}%），屬十字或黑K，6號門不成立。"
    )


def check_volume_expansion(df):
    if df is None or len(df) < 6:
        return False, "資料不足，無法確認量能。"
    today_vol = float(df.iloc[-1]["Trading_Volume"])
    avg5 = df["Trading_Volume"].iloc[-6:-1].mean()
    expanded = today_vol > avg5
    return expanded, (
        f"今日量{today_vol/1000:.0f}張，近5日均量{avg5/1000:.0f}張，"
        f"{'量能放大，條件通過。' if expanded else '量能未放大，6號門不成立。'}"
    )


def check_false_signal(df, us_block):
    triggers = []
    if df is None or len(df) < 2:
        return False, [], "資料不足，無法判斷假訊號。"
    today = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(today["close"])
    open_p = float(today["open"])
    high = float(today["max"])
    vol_today = float(today["Trading_Volume"])
    vol_prev = float(prev["Trading_Volume"])

    upper_shadow = (high - max(close, open_p)) / close * 100 if close != 0 else 0
    vol_shrink = vol_today < vol_prev * 0.9
    if upper_shadow > 2.0 and vol_shrink:
        triggers.append("長上影 + 量縮（誘多拉高出貨訊號）")

    prev_high_20 = df["max"].iloc[-21:-1].max() if len(df) >= 21 else df["max"].iloc[:-1].max()
    if high >= prev_high_20 and close < prev_high_20:
        triggers.append("突破不收（盤中碰壓力但未站穩，假突破）")

    body_pct = (close - open_p) / open_p * 100 if open_p != 0 else 0
    if body_pct <= -2.0 and vol_today > vol_prev * 1.1:
        triggers.append("放量長黑（主力出貨或強力賣壓訊號）")

    if len(df) >= 21:
        pressure_zone = df["max"].iloc[-21:-1].max()
        if close < pressure_zone * 0.99:
            triggers.append("跌回壓力區內（突破失敗，結構轉弱）")

    market_ok, _ = check_market_not_crashing(us_block)
    if not market_ok:
        triggers.append("大盤急殺（外部系統性風險，禁止進場）")

    is_false = len(triggers) > 0
    detail = ("偵測到以下假訊號，無條件退出，不解釋、不凹單：\n" + "\n".join(f"・{t}" for t in triggers)
              if is_false else "未偵測到假訊號，結構相對乾淨。")
    return is_false, triggers, detail


def gate6_confirm(df, us_block, pressure=None):
    results = []
    breakout, breakout_msg = check_breakout_prev_high(df)
    results.append({"name": "突破前高", "pass": breakout, "detail": breakout_msg})

    if pressure is None and df is not None and len(df) >= 21:
        pressure = df["max"].iloc[-21:-1].max()
    elif pressure is None and df is not None:
        pressure = float(df.iloc[-1]["close"])

    today = df.iloc[-1] if df is not None and len(df) >= 1 else None
    if today is not None:
        close_ok, close_msg = check_close_above_pressure(today, pressure)
    else:
        close_ok, close_msg = False, "無資料"
    results.append({"name": "收盤站穩（非盤中假突破）", "pass": close_ok, "detail": close_msg})

    if today is not None:
        red_ok, red_msg = check_red_candle(today)
    else:
        red_ok, red_msg = False, "無資料"
    results.append({"name": "紅K確認", "pass": red_ok, "detail": red_msg})

    vol_ok, vol_msg = check_volume_expansion(df)
    results.append({"name": "量能放大", "pass": vol_ok, "detail": vol_msg})

    market_ok, market_msg = check_market_not_crashing(us_block)
    results.append({"name": "大盤未急殺", "pass": market_ok, "detail": market_msg})

    missing = [r["name"] for r in results if not r["pass"]]
    is_gate6 = len(missing) == 0

    if today is not None:
        stop_by_pct = round(float(pressure) * 0.995, 2)
        stop_by_low = round(float(today["min"]), 2)
        stop_loss = max(stop_by_pct, stop_by_low)
    else:
        stop_loss = None

    if is_gate6:
        summary = "五項條件全部成立，確認為6號門。建議等下一根回測不破突破價後再進場，停損設於突破價下方較緊位置。"
    elif len(missing) == 1:
        summary = f"尚差一項條件（{missing[0]}），目前為5號門門口，可小量卡位但禁止主攻。"
    else:
        summary = f"缺少{len(missing)}項條件（{'、'.join(missing)}），尚未達5號門標準，建議觀望。"

    return {
        "is_gate6": is_gate6, "conditions": results, "missing": missing,
        "summary": summary, "stop_loss": stop_loss,
        "pressure_used": round(float(pressure), 2) if pressure else None
    }


def gate5_confirm(df, us_block, pressure=None):
    if df is None or len(df) < 6:
        return {"is_gate5": False, "summary": "資料不足"}
    today = df.iloc[-1]
    close = float(today["close"])
    if pressure is None and len(df) >= 21:
        pressure = df["max"].iloc[-21:-1].max()
    elif pressure is None:
        pressure = close
    proximity_pct = (pressure - close) / close * 100 if close != 0 else 999
    conditions = []
    near_pressure = 0 <= proximity_pct <= 3.0
    conditions.append({"name": "接近壓力區（3%以內）", "pass": near_pressure,
                        "detail": f"距壓力{proximity_pct:.2f}%，{'符合' if near_pressure else '距離過遠或已突破'}。"})
    not_broken = close < pressure
    conditions.append({"name": "尚未突破壓力", "pass": not_broken,
                        "detail": f"收盤{close:.2f}{'低於' if not_broken else '已超過'}壓力{pressure:.2f}。"})
    is_false, triggers, false_detail = check_false_signal(df, us_block)
    conditions.append({"name": "無假訊號", "pass": not is_false, "detail": false_detail})
    trend = check_higher_highs_lows(df)
    conditions.append({"name": "趨勢結構有效", "pass": trend["trend_valid"], "detail": trend["description"]})

    market_ok, market_msg = check_market_not_crashing(us_block)
    if not market_ok:
        return {"is_gate5": False, "conditions": conditions,
                "summary": f"大盤急殺，5號門亦不建議操作。{market_msg}",
                "stop_loss": None, "proximity_pct": round(proximity_pct, 2)}

    missing = [c["name"] for c in conditions if not c["pass"]]
    is_gate5 = len(missing) == 0
    stop_loss = round(min(float(today["min"]), close * 0.985), 2)
    if is_gate5:
        summary = (f"符合5號門卡位條件，距壓力{proximity_pct:.2f}%。"
                   "僅可小量試單，進場前先設停損，跌破支撐立刻出場，禁止加碼，反彈先減。")
    else:
        summary = f"不符合5號門條件（{'、'.join(missing)}），建議繼續觀望。"
    return {"is_gate5": is_gate5, "conditions": conditions, "missing": missing,
            "summary": summary, "stop_loss": stop_loss, "proximity_pct": round(proximity_pct, 2)}


# =========================
# 補強模組三：資金與部位管理
# =========================
def calc_position_size(total_capital, entry_price, stop_loss_price, gate="6", risk_pct=1.5, is_tw_stock=True):
    if entry_price <= 0 or stop_loss_price <= 0:
        return {"error": "進場價或停損價不合理"}
    if entry_price == stop_loss_price:
        return {"error": "進場價與停損價相同，無法計算部位"}
    risk_per_unit = abs(entry_price - stop_loss_price)
    max_loss_amount = total_capital * (risk_pct / 100)
    max_units = max_loss_amount / risk_per_unit
    if is_tw_stock:
        lots = max(1, int(max_units / 1000))
        if gate == "5":
            lots = max(1, int(lots * 0.5))
        position_value = lots * 1000 * entry_price
        actual_risk = lots * 1000 * risk_per_unit
        return {
            "gate": gate, "total_capital": total_capital, "entry_price": entry_price,
            "stop_loss_price": stop_loss_price, "risk_per_unit": round(risk_per_unit, 2),
            "max_loss_budget": round(max_loss_amount, 0), "suggested_lots": lots,
            "position_value": round(position_value, 0), "actual_risk_amount": round(actual_risk, 0),
            "actual_risk_pct": round(actual_risk / total_capital * 100, 2), "unit": "張（台股）",
            "note": (f"{'5號門試單減半，' if gate == '5' else ''}建議買進{lots}張，"
                     f"最大虧損約{round(actual_risk, 0):,.0f}元（占資金{round(actual_risk / total_capital * 100, 2)}%）。")
        }
    else:
        shares = max(1, int(max_units))
        if gate == "5":
            shares = max(1, int(shares * 0.5))
        position_value = shares * entry_price
        actual_risk = shares * risk_per_unit
        return {
            "gate": gate, "total_capital": total_capital, "entry_price": entry_price,
            "stop_loss_price": stop_loss_price, "risk_per_unit": round(risk_per_unit, 2),
            "max_loss_budget": round(max_loss_amount, 0), "suggested_shares": shares,
            "position_value": round(position_value, 0), "actual_risk_amount": round(actual_risk, 0),
            "actual_risk_pct": round(actual_risk / total_capital * 100, 2), "unit": "股（美股）",
            "note": (f"{'5號門試單減半，' if gate == '5' else ''}建議買進{shares}股，"
                     f"最大虧損約${round(actual_risk, 0):,.0f}（占資金{round(actual_risk / total_capital * 100, 2)}%）。")
        }


def check_no_averaging_down(avg_cost, current_price, direction="多"):
    if direction == "多" and current_price < avg_cost:
        return {"warning": True, "message": (
            f"現價{current_price:.2f}低於持倉成本{avg_cost:.2f}，屬虧損狀態加碼，系統禁止此操作。請依停損紀律執行出場。")}
    if direction == "空" and current_price > avg_cost:
        return {"warning": True, "message": (
            f"現價{current_price:.2f}高於空單成本{avg_cost:.2f}，屬虧損狀態加碼，系統禁止此操作。請依停損紀律執行回補。")}
    return {"warning": False, "message": "現價未違反攤平規則，部位管理正常。"}


# =========================
# 補強模組四：出場條件系統性觸發
# =========================
def check_exit_conditions(df, stop_loss_price, breakout_price, direction="多", us_block=None):
    if df is None or len(df) < 3:
        return {"should_exit": False, "triggers": [], "priority": "持續觀察", "detail": "資料不足，無法判斷出場條件。"}
    today = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(today["close"])
    open_p = float(today["open"])
    high = float(today["max"])
    vol_today = float(today["Trading_Volume"])
    vol_prev = float(prev["Trading_Volume"])
    triggers = []
    priority_level = 0

    if direction == "多" and close <= stop_loss_price:
        triggers.append({"condition": "跌破停損價", "detail": f"收盤{close:.2f}已跌破停損{stop_loss_price:.2f}，無條件出場。", "urgency": "立即出場"})
        priority_level = max(priority_level, 3)
    elif direction == "空" and close >= stop_loss_price:
        triggers.append({"condition": "突破停損價（空單）", "detail": f"收盤{close:.2f}已突破空單停損{stop_loss_price:.2f}，無條件回補。", "urgency": "立即出場"})
        priority_level = max(priority_level, 3)

    body_pct = (close - open_p) / open_p * 100 if open_p != 0 else 0
    if body_pct <= -2.0 and vol_today > vol_prev * 1.1:
        triggers.append({"condition": "放量長黑", "detail": f"今日下跌{abs(body_pct):.2f}%且量能放大，主力賣壓明確，立即出場。", "urgency": "立即出場"})
        priority_level = max(priority_level, 3)

    if direction == "多" and close < breakout_price:
        triggers.append({"condition": "跌破突破價", "detail": f"收盤{close:.2f}已跌回突破價{breakout_price:.2f}以下，突破失敗，出場。", "urgency": "立即出場"})
        priority_level = max(priority_level, 3)

    price_flat = abs(body_pct) < 0.5
    vol_shrink = vol_today < vol_prev * 0.8
    if price_flat and vol_shrink:
        triggers.append({"condition": "價不動、量縮", "detail": f"今日漲跌幅{body_pct:.2f}%，量能萎縮，動能耗盡，逢高減碼。", "urgency": "逢高減碼"})
        priority_level = max(priority_level, 1)

    if len(df) >= 4:
        last2_vol = df["Trading_Volume"].iloc[-3:-1]
        last2_body = [abs((df.iloc[i]["close"] - df.iloc[i]["open"]) / df.iloc[i]["open"] * 100) for i in range(-3, -1)]
        high_area = close > df["close"].iloc[-20:].quantile(0.8) if len(df) >= 20 else False
        if high_area and all(v < vol_prev * 0.85 for v in last2_vol) and all(b < 1.0 for b in last2_body):
            triggers.append({"condition": "高檔量縮盤整", "detail": "股價位於高檔，連續量縮且漲跌幅收斂，主升動能減弱，逢高減碼。", "urgency": "逢高減碼"})
            priority_level = max(priority_level, 2)

    if len(df) >= 6:
        recent6 = df.iloc[-6:].reset_index(drop=True)
        if recent6.iloc[-1]["max"] < recent6.iloc[-3]["max"] and recent6.iloc[-1]["min"] < recent6.iloc[-3]["min"]:
            triggers.append({"condition": "趨勢結構破壞", "detail": "近期高低點雙雙下移，多頭結構已破，即使未跌停損亦應出場。", "urgency": "立即出場"})
            priority_level = max(priority_level, 3)

    if us_block and us_block.get("indices"):
        avg_pct = sum(i.get("change_pct", 0) for i in us_block["indices"]) / len(us_block["indices"])
        if avg_pct < -1.5:
            triggers.append({"condition": "大盤轉弱", "detail": f"對應大盤指數平均跌幅{avg_pct:.2f}%，系統性風險升高，不主動留倉。", "urgency": "逢高減碼"})
            priority_level = max(priority_level, 2)

    should_exit = priority_level >= 1
    if priority_level >= 3:
        priority = "立即出場"
    elif priority_level >= 2:
        priority = "積極減碼"
    elif priority_level >= 1:
        priority = "逢高減碼"
    else:
        priority = "持續觀察"

    detail = (f"觸發{len(triggers)}項出場訊號（{priority}）：\n" + "\n".join(f"・{t['condition']}：{t['detail']}" for t in triggers)
              if triggers else "目前未觸發任何出場條件，依停損紀律繼續持有。")
    return {"should_exit": should_exit, "triggers": triggers, "priority": priority, "detail": detail}


# =========================
# 側邊欄：專業作戰時程表
# =========================
def render_battle_timetable():
    st.sidebar.markdown(
        """
        <div style="border:2px solid #d0d7de;border-radius:12px;padding:14px;background-color:#f8fafc;margin-bottom:12px;">
            <div style="font-size:1.1rem;font-weight:700;margin-bottom:10px;">🛡️ 專業作戰時程指引</div>
            <div style="margin-bottom:12px;">
                <div style="font-weight:700;color:#0f9d58;">• 🟢 凌晨 05:00 - 08:30【最佳盤後分析期】</div>
                <div>• 任務：執行盤後分析，確認美股收盤漲跌與高檔結構標的。</div>
                <div>• 核心：依美股動能、籌碼與隔日劇本，先定好當日計畫。</div>
            </div>
            <div style="margin-bottom:12px;">
                <div style="font-weight:700;color:#d97706;">• 🔥 早上 09:00 - 11:00【黃金實戰扣動期】</div>
                <div>• 任務：切換盤中判斷，監控即時價位。</div>
                <div>• 核心：只依壓力 / 支撐與停損紀律執行，不憑感覺追單。</div>
            </div>
            <div>
                <div style="font-weight:700;color:#6b7280;">• 💤 早上 11:00 之後【收盤與身心休養】</div>
                <div>• 任務：停止新開倉動作。</div>
                <div>• 核心：讓獲利奔跑或交由停損單處理，回歸正常作息。</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# 頂部時間紅綠燈
# =========================
def render_time_banner():
    now = tw_now()
    hhmm = now.hour * 100 + now.minute
    if 500 <= hhmm <= 850:
        msg = "🔥 美股已收盤，數據最精準，適合晨間作戰規劃"
        color, bg = "#0f9d58", "#e8f5e9"
    elif 900 <= hhmm <= 1335:
        msg = "⚠️ 台股交易中，美股為昨日數據，請留意盤中波動"
        color, bg = "#b26a00", "#fff8e1"
    else:
        msg = "💤 盤後結算中，請等待傍晚籌碼更新"
        color, bg = "#666666", "#f2f2f2"
    st.markdown(
        f"""<div style="background:{bg};color:{color};padding:10px 14px;border-radius:10px;font-weight:700;margin-bottom:10px;">
            目前系統時間：{now.strftime("%Y-%m-%d %H:%M:%S")}　{msg}</div>""",
        unsafe_allow_html=True
    )


def render_after_usage_note():
    st.markdown(
        """<div style="border:1px solid #2e7d32;background:#e8f5e9;color:#1b5e20;padding:12px 14px;border-radius:10px;font-weight:700;margin-bottom:12px;">
            盤後分析使用時段：建議於凌晨 05:00 - 08:00 閱讀<br>建議用途：確認美股收盤、籌碼方向與隔日主要劇本。</div>""",
        unsafe_allow_html=True
    )

def render_intra_usage_note():
    st.markdown(
        """<div style="border:1px solid #ef6c00;background:#fff3e0;color:#e65100;padding:12px 14px;border-radius:10px;font-weight:700;margin-bottom:12px;">
            盤中判斷使用時段：建議於早上 09:00 - 11:00 使用<br>建議用途：依壓力 / 支撐與即時價位執行進場、停損與分批賣出。</div>""",
        unsafe_allow_html=True
    )


# =========================
# FinMind API
# =========================
@st.cache_data(ttl=3600)
def fetch_finmind_dataset(dataset, data_id="", start_date="2023-01-01"):
    params = {"dataset": dataset, "start_date": start_date}
    if data_id:
        params["data_id"] = data_id
    try:
        res = requests.get(FINMIND_BASE_URL, params=params, timeout=20)
        res.raise_for_status()
        payload = res.json()
    except Exception:
        return None
    if "data" not in payload:
        return None
    return pd.DataFrame(payload["data"])


# =========================
# 台股 / 美股 基本資料
# =========================
@st.cache_data(ttl=3600)
def fetch_stock_info(stock_id):
    if is_us_symbol(stock_id):
        return {"stock_id": stock_id.upper(), "stock_name": stock_id.upper(), "industry_category": "美股", "type": "US"}
    df = fetch_finmind_dataset("TaiwanStockInfo", start_date="2000-01-01")
    if df is None or df.empty or "stock_id" not in df.columns:
        return None
    target = df[df["stock_id"].astype(str) == str(stock_id)]
    if target.empty:
        return None
    row = target.iloc[0]
    return {"stock_id": str(row.get("stock_id", "")), "stock_name": str(row.get("stock_name", "")),
            "industry_category": str(row.get("industry_category", "")), "type": str(row.get("type", ""))}


@st.cache_data(ttl=3600)
def fetch_month_revenue(stock_id):
    if is_us_symbol(stock_id):
        return None
    df = fetch_finmind_dataset("TaiwanStockMonthRevenue", data_id=stock_id, start_date="2023-01-01")
    if df is None or df.empty:
        return None
    sort_cols = [c for c in ["revenue_year", "revenue_month"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    latest = df.iloc[-1].to_dict()
    prev = df.iloc[-2].to_dict() if len(df) >= 2 else None
    return latest, prev


# =========================
# yfinance 日線整理
# =========================
def _normalize_yf_frame(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        out = pd.DataFrame(index=df.index)
        mapping = {"Open": "open", "Close": "close", "High": "max", "Low": "min", "Volume": "Trading_Volume"}
        for src, dst in mapping.items():
            try:
                obj = df[src]
                out[dst] = pd.to_numeric(obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj, errors="coerce")
            except Exception:
                return None
        out["date"] = df.index
        return out.reset_index(drop=True)
    mapping = {"Open": "open", "Close": "close", "High": "max", "Low": "min", "Volume": "Trading_Volume"}
    for c in mapping:
        if c not in df.columns:
            return None
    out = df.rename(columns=mapping).reset_index()
    if "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})
    return out


@st.cache_data(ttl=1800)
def fetch_us_daily_data(symbol, period="6mo"):
    try:
        df = yf.download(symbol.upper(), period=period, interval="1d", progress=False, auto_adjust=False, group_by="column")
    except Exception:
        return None
    out = _normalize_yf_frame(df)
    if out is None:
        return None
    for c in ["open", "close", "max", "min", "Trading_Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "close", "max", "min", "Trading_Volume"]).reset_index(drop=True)
    if len(out) < 20:
        return None
    return out


@st.cache_data(ttl=3600)
def fetch_daily_data(stock_id, start_date="2023-01-01"):
    if is_us_symbol(stock_id):
        return fetch_us_daily_data(stock_id)
    df = fetch_finmind_dataset("TaiwanStockPrice", data_id=stock_id, start_date=start_date)
    if df is None or df.empty:
        return None
    required = ["date", "open", "close", "max", "min", "Trading_Volume"]
    for c in required:
        if c not in df.columns:
            return None
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["open", "close", "max", "min", "Trading_Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close", "max", "min", "Trading_Volume"]).reset_index(drop=True)
    if len(df) < 10:
        return None
    return df


# =========================
# 基本面摘要
# =========================
def build_auto_fundamental_summary(stock_id):
    if is_us_symbol(stock_id):
        try:
            ticker = yf.Ticker(stock_id.upper())
            info = ticker.info
            name = info.get("longName", stock_id.upper())
            sector = info.get("sector", "")
            pe = info.get("trailingPE", None)
            rev_growth = info.get("revenueGrowth", None)
            parts = [f"{name}"]
            if sector:
                parts.append(f"所屬產業為{sector}")
            if pe:
                parts.append(f"本益比約{pe:.1f}倍")
            if rev_growth:
                parts.append(f"營收年增率約{rev_growth*100:.1f}%")
            return "；".join(parts) + f"。(更新時間: {now_hhmm()})"
        except Exception:
            return f"{stock_id.upper()}為美股，基本面摘要建議搭配公司財報與產業趨勢另行交叉比對。(更新時間: {now_hhmm()})"

    info = fetch_stock_info(stock_id)
    rev_data = fetch_month_revenue(stock_id)
    parts = []
    if info:
        name = info.get("stock_name", "")
        industry = info.get("industry_category", "")
        if name and industry:
            parts.append(f"{name}所屬產業類別為{industry}")
        elif name:
            parts.append(f"{name}為目前查得之公司名稱")
    if rev_data:
        latest, prev = rev_data
        y = latest.get("revenue_year", "")
        m = latest.get("revenue_month", "")
        revenue = latest.get("revenue", None)
        if y and m and revenue is not None:
            parts.append(f"最新月營收資料為{y}年{m}月，單月營收約{volume_to_human(revenue)}元")
        if prev and latest.get("revenue") is not None and prev.get("revenue") is not None:
            try:
                mom = (float(latest["revenue"]) - float(prev["revenue"])) / float(prev["revenue"]) * 100
                if mom > 0:
                    parts.append(f"月營收較前一期增加約{mom:.2f}%")
                elif mom < 0:
                    parts.append(f"月營收較前一期減少約{abs(mom):.2f}%")
                else:
                    parts.append("月營收與前一期大致持平")
            except Exception:
                pass
    if not parts:
        return f"未成功抓取最新公開基本面資料。(更新時間: {now_hhmm()})"
    return "；".join(parts) + f"。(更新時間: {now_hhmm()})"


# =========================
# 美股連動
# =========================
def map_us_indices(industry_category):
    text = (industry_category or "").strip()
    if any(k in text for k in ["半導體", "電子", "通訊"]):
        return ["^SOX", "^IXIC"], "費半 / 那指"
    if any(k in text for k in ["金融", "鋼鐵", "塑膠", "水泥"]):
        return ["^DJI"], "道瓊"
    if any(k in text for k in ["AI", "雲端", "軟體", "生技", "醫療", "網通"]):
        return ["^IXIC"], "那指"
    return ["^IXIC"], "那指"


@st.cache_data(ttl=3600)
def fetch_us_index_change(ticker):
    try:
        df = yf.download(ticker, period="7d", interval="1d", progress=False, auto_adjust=False, group_by="column")
    except Exception:
        return None
    if df is None or df.empty:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            close_obj = df["Close"]
            close = pd.to_numeric(close_obj.iloc[:, 0] if isinstance(close_obj, pd.DataFrame) else close_obj, errors="coerce").dropna()
        else:
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    except Exception:
        return None
    if len(close) < 2:
        return None
    latest = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    if prev == 0:
        return None
    pct = (latest - prev) / prev * 100
    return {"ticker": ticker, "latest_close": latest, "prev_close": prev, "change_pct": round(pct, 2)}


def summarize_us_bias(indices):
    if not indices:
        return "中性"
    avg_pct = float(np.mean([i["change_pct"] for i in indices]))
    if avg_pct > 0.8:
        return "偏多"
    if avg_pct < -0.8:
        return "偏空"
    return "中性"


def build_us_correlation_block(stock_id):
    info = fetch_stock_info(stock_id)
    industry = info["industry_category"] if info else ""
    if is_us_symbol(stock_id):
        tickers, label = ["^IXIC", "^GSPC"], "Nasdaq / S&P 500"
    else:
        tickers, label = map_us_indices(industry)
    data = [item for t in tickers if (item := fetch_us_index_change(t))]
    return {"industry": industry, "mapping_label": label, "indices": data, "bias": summarize_us_bias(data)}


# =========================
# 籌碼：400張以上大戶持股比例
# =========================
@st.cache_data(ttl=3600)
def fetch_400_holder_ratio(stock_id):
    if is_us_symbol(stock_id):
        return None
    df = fetch_finmind_dataset("TaiwanStockHoldingSharesPer", data_id=stock_id, start_date="2024-01-01")
    if df is None or df.empty:
        return None
    cols = set(df.columns)
    if "date" not in cols:
        return None
    level_col = next((c for c in ["HoldingSharesLevel", "holding_shares_level", "shares_level"] if c in cols), None)
    percent_col = next((c for c in ["percent", "Percent", "percentage"] if c in cols), None)
    if level_col is None or percent_col is None:
        return None
    df = df.copy()
    df[level_col] = df[level_col].astype(str)
    df[percent_col] = pd.to_numeric(df[percent_col], errors="coerce")
    df = df.dropna(subset=[percent_col])
    df_400 = df[df[level_col].str.contains("400|600|800|1000", na=False)].copy()
    if df_400.empty:
        return None
    grouped = df_400.groupby("date", as_index=False)[percent_col].sum().sort_values("date").reset_index(drop=True)
    if len(grouped) < 2:
        return None
    return grouped.rename(columns={percent_col: "holder_400_ratio"})


def check_400_holder_change(stock_id):
    if is_us_symbol(stock_id):
        return {"available": False, "latest_ratio": None, "prev_ratio": None, "change": None,
                "warning": False, "severity": "none", "message": f"美股不適用400張以上大戶持股統計。(更新時間: {now_hhmm()})"}
    df = fetch_400_holder_ratio(stock_id)
    if df is None or len(df) < 2:
        return {"available": False, "latest_ratio": None, "prev_ratio": None, "change": None,
                "warning": False, "severity": "none", "message": f"無足夠400張以上大戶持股資料。(更新時間: {now_hhmm()})"}
    latest_ratio = float(df.iloc[-1]["holder_400_ratio"])
    prev_ratio = float(df.iloc[-2]["holder_400_ratio"])
    change = latest_ratio - prev_ratio
    warning = change < -0.5
    if change <= -1.0:
        severity, message = "strong", f"400張以上大戶持股比例較上一期下降 {abs(change):.2f}% ，屬強警戒區。(更新時間: {now_hhmm()})"
    elif change < -0.5:
        severity, message = "mild", f"400張以上大戶持股比例較上一期下降 {abs(change):.2f}% ，屬輕度警戒區。(更新時間: {now_hhmm()})"
    elif change > 0:
        severity, message = "positive", f"400張以上大戶持股比例較上一期增加 {change:.2f}% (更新時間: {now_hhmm()})"
    else:
        severity, message = "neutral", f"400張以上大戶持股比例與上一期大致持平 (更新時間: {now_hhmm()})"
    return {"available": True, "latest_ratio": round(latest_ratio, 2), "prev_ratio": round(prev_ratio, 2),
            "change": round(change, 2), "warning": warning, "severity": severity, "message": message}


# =========================
# 技術模型
# =========================
def valid_row(row):
    for c in ["open", "close", "max", "min", "Trading_Volume"]:
        if c not in row or pd.isna(row[c]):
            return False
    return row["close"] > 0 and row["max"] > row["min"]


def score_today(df):
    t = df.iloc[-1]
    p = df.iloc[-2]
    if not valid_row(t) or not valid_row(p):
        return None
    score = 0
    close_pos = (t["close"] - t["min"]) / (t["max"] - t["min"] + 1e-5)
    if close_pos > 0.7:
        score += 20
    if t["close"] > t["open"]:
        score += 15
    if (t["max"] - t["min"]) / t["close"] > 0.02:
        score += 15
    if p["close"] > 0 and (t["close"] - p["close"]) / p["close"] < 0.05:
        score += 10
    if t["Trading_Volume"] > p["Trading_Volume"]:
        score += 20
    recent_5d_high = df["max"].iloc[-6:-1].max()
    if t["close"] >= recent_5d_high:
        score += 20
    return int(score)


def backtest_strategy(df):
    returns = []
    for i in range(10, len(df) - 1):
        t = df.iloc[i]
        p = df.iloc[i - 1]
        n = df.iloc[i + 1]
        if not valid_row(t) or not valid_row(p) or not valid_row(n):
            continue
        close_pos = (t["close"] - t["min"]) / (t["max"] - t["min"] + 1e-5)
        signal = (
            close_pos > 0.7 and t["close"] > t["open"] and
            (t["max"] - t["min"]) / t["close"] > 0.02 and
            (t["close"] - p["close"]) / p["close"] < 0.05 and
            t["Trading_Volume"] > p["Trading_Volume"]
        )
        if signal:
            returns.append((n["close"] - t["max"]) / t["max"])
    if not returns:
        return {"win_rate": 0.0, "avg_return": 0.0, "trades": 0}
    return {
        "win_rate": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 2),
        "avg_return": round(sum(returns) / len(returns) * 100, 2),
        "trades": len(returns)
    }


def build_after_levels(close_price, high_price, low_price, direction="多", stage="③"):
    range_val = max(high_price - low_price, 0.01)
    if direction == "多":
        pressure1 = round(high_price, 2)
        if stage in ["③", "④"]:
            pressure2 = round(high_price + range_val * 0.8, 2)
        elif stage == "⑤":
            pressure2 = round(high_price + range_val * 0.4, 2)
        else:
            pressure2 = round(high_price + range_val * 0.25, 2)
        support1 = round(low_price, 2)
        stop_loss = round(close_price * 0.99, 2) if stage == "⑥" else round(min(low_price, close_price * 0.985), 2)
    else:
        pressure1 = round(high_price, 2)
        pressure2 = round(high_price * 1.02, 2)
        support1 = round(low_price, 2)
        stop_loss = round(high_price * 1.02, 2)
    return {"pressure1": pressure1, "pressure2": pressure2, "support1": support1, "stop_loss": stop_loss}


def detect_stage_advanced(df, stock_id):
    if df is None or len(df) < 20:
        return {"stage": "①", "stage_name": "資料不足", "stage_desc": "無法判斷階段。",
                "chip_warning": False, "chip_message": "", "fake_break": False, "upper_shadow_pct": 0}
    t = df.iloc[-1]
    close = float(t["close"])
    open_p = float(t["open"])
    high = float(t["max"])
    low = float(t["min"])
    vol = float(t["Trading_Volume"])
    vol_ma5 = df["Trading_Volume"].rolling(5).mean().iloc[-1]
    vol_ratio = vol / vol_ma5 if pd.notna(vol_ma5) and vol_ma5 != 0 else 1
    ma20 = df["close"].rolling(20).mean().iloc[-1]
    pos20 = close / ma20 if pd.notna(ma20) and ma20 != 0 else 1
    ret5 = (close - df["close"].iloc[-6]) / df["close"].iloc[-6] * 100 if len(df) >= 6 else 0
    upper_shadow_pct = (high - max(close, open_p)) / close * 100 if close != 0 else 0
    prev_high_20 = df["max"].iloc[-21:-1].max() if len(df) >= 21 else df["max"].iloc[:-1].max()
    fake_break = high >= prev_high_20 and close < open_p and vol_ratio < 1

    if pos20 < 0.97 and ret5 < 3:
        stage, stage_name, desc = "①", "底部築底", "價格仍在均線下方盤整，尚未出現明確起漲訊號。"
    elif pos20 < 1.0 and ret5 >= 3:
        stage, stage_name, desc = "②", "起漲", "突破均線區域，進入起漲初段，可持續觀察。"
    elif pos20 >= 1.0 and vol_ratio > 1.1 and ret5 < 10:
        stage, stage_name, desc = "③", "攻擊", "站穩均線之上且量能同步放大，進入主攻段。"
    elif pos20 >= 1.02 and vol_ratio > 1.2:
        stage, stage_name, desc = "④", "軋空", "量價同步加速，進入強勢推升區。"
    elif pos20 < 0.92 and ret5 > 6:
        stage, stage_name, desc = "⑤", "末段攻擊", "股價已接近高檔，但仍維持末段攻擊結構。"
    else:
        stage, stage_name, desc = "⑥", "噴出", "短線已進入噴出區，追價風險顯著提高。"

    if fake_break:
        desc += " 目前出現假突破跡象。"
    if upper_shadow_pct > 2.0:
        desc += " 上影較長，需防沖高回落。"

    chip = check_400_holder_change(stock_id)
    return {
        "stage": stage, "stage_name": stage_name,
        "stage_desc": f"{desc} (更新時間: {now_hhmm()})",
        "chip_warning": chip["warning"], "chip_message": chip["message"],
        "fake_break": fake_break, "upper_shadow_pct": round(upper_shadow_pct, 2)
    }


# =========================
# AI 權重預測：四情境
# =========================
def build_next_day_scenarios(df, stage, chip_change=0.0, us_bias="中性"):
    scenarios = {"開高走高": 25, "開高走低": 25, "開低走高": 25, "開低走低": 25}
    if df is None or len(df) < 30:
        main_name = max(scenarios, key=scenarios.get)
        return {"scenarios": scenarios, "main_scenario": main_name,
                "response_note": f"資料不足，先以保守觀察為主。(更新時間: {now_hhmm()})"}
    d = df.copy().reset_index(drop=True)
    t = d.iloc[-1]
    p = d.iloc[-2]
    close_now = float(t["close"])
    open_now = float(t["open"])
    high_now = float(t["max"])
    low_now = float(t["min"])
    prev_close = float(p["close"])
    daily_change = (close_now - prev_close) / prev_close * 100 if prev_close != 0 else 0
    intraday_body = (close_now - open_now) / open_now * 100 if open_now != 0 else 0
    upper_shadow = (high_now - max(close_now, open_now)) / close_now * 100 if close_now != 0 else 0
    vol_ma5 = d["Trading_Volume"].rolling(5).mean().iloc[-1]
    vol_ratio = float(t["Trading_Volume"]) / float(vol_ma5) if pd.notna(vol_ma5) and vol_ma5 != 0 else 1
    close_pos = (close_now - low_now) / (high_now - low_now + 1e-6)

    base_map = {
        "①": {"開高走高": 10, "開高走低": 20, "開低走高": 30, "開低走低": 40},
        "②": {"開高走高": 25, "開高走低": 20, "開低走高": 35, "開低走低": 20},
        "③": {"開高走高": 35, "開高走低": 20, "開低走高": 30, "開低走低": 15},
        "④": {"開高走高": 38, "開高走低": 27, "開低走高": 22, "開低走低": 13},
        "⑤": {"開高走高": 20, "開高走低": 30, "開低走高": 20, "開低走低": 30},
        "⑥": {"開高走高": 12, "開高走低": 38, "開低走高": 10, "開低走低": 40},
    }
    scenarios = base_map.get(stage, scenarios)

    if close_pos > 0.7:
        scenarios["開高走高"] += 6; scenarios["開低走高"] += 3; scenarios["開低走低"] -= 5
    elif close_pos < 0.3:
        scenarios["開低走低"] += 8; scenarios["開高走低"] += 4; scenarios["開高走高"] -= 5
    if daily_change > 2 and intraday_body > 1 and vol_ratio > 1.2:
        scenarios["開高走高"] += 6; scenarios["開低走高"] += 4; scenarios["開高走低"] -= 5; scenarios["開低走低"] -= 5
    if upper_shadow > 2:
        scenarios["開高走低"] += 8; scenarios["開低走低"] += 4; scenarios["開高走高"] -= 6; scenarios["開低走高"] -= 6
    if daily_change < -1 and vol_ratio > 1.1:
        scenarios["開低走低"] += 8; scenarios["開高走低"] += 4; scenarios["開高走高"] -= 6; scenarios["開低走高"] -= 6
    if chip_change < -1.0:
        scenarios["開高走低"] += 8; scenarios["開低走低"] += 10; scenarios["開高走高"] -= 8; scenarios["開低走高"] -= 6
    elif chip_change < -0.5:
        scenarios["開高走低"] += 5; scenarios["開低走低"] += 6; scenarios["開高走高"] -= 5
    elif chip_change > 0.5:
        scenarios["開高走高"] += 5; scenarios["開低走高"] += 4; scenarios["開低走低"] -= 4
    if us_bias == "偏多":
        scenarios["開高走高"] += 5; scenarios["開低走高"] += 3; scenarios["開低走低"] -= 5
    elif us_bias == "偏空":
        scenarios["開高走低"] += 4; scenarios["開低走低"] += 6; scenarios["開高走高"] -= 5

    for k in scenarios:
        scenarios[k] = max(5, scenarios[k])
    total = sum(scenarios.values())
    scenarios = {k: round(v / total * 100) for k, v in scenarios.items()}
    diff = 100 - sum(scenarios.values())
    if diff != 0:
        scenarios[max(scenarios, key=scenarios.get)] += diff

    main_name = max(scenarios, key=scenarios.get)
    response_map = {
        "開高走高": f"主要劇本為開高走高，盤前應對以順勢突破追蹤為主。(更新時間: {now_hhmm()})",
        "開高走低": f"主要劇本為開高走低，盤前應對應避免開盤追價，慎防沖高回落。(更新時間: {now_hhmm()})",
        "開低走高": f"主要劇本為開低走高，盤前應對宜等回穩後再介入，不宜急單。(更新時間: {now_hhmm()})",
        "開低走低": f"主要劇本為開低走低，盤前應對以防守為主，不主動追多。(更新時間: {now_hhmm()})"
    }
    return {"scenarios": scenarios, "main_scenario": main_name, "response_note": response_map[main_name]}


# =========================
# 交易員結論 / 有利區 / 勝率 AI
# =========================
def build_trader_comment(stage, chip_warning, main_scenario, us_bias="中性"):
    if chip_warning and stage in ["⑤", "⑥"]:
        return f"高檔籌碼轉弱，優先防守 (更新時間: {now_hhmm()})"
    if stage == "⑥":
        return f"噴出過熱，嚴禁追價 (更新時間: {now_hhmm()})"
    if stage == "⑤":
        if main_scenario in ["開高走低", "開低走低"]:
            return f"末段攻擊仍在，但隔日拉回風險偏高 (更新時間: {now_hhmm()})"
        return f"末段攻擊延續，但仍須控管追價風險 (更新時間: {now_hhmm()})"
    if stage in ["③", "④"] and us_bias == "偏空":
        return f"主升段結構仍在，但外部環境偏空 (更新時間: {now_hhmm()})"
    if stage in ["③", "④"]:
        return f"主升段延續，結構健康 (更新時間: {now_hhmm()})"
    if stage == "②":
        return f"起漲初期，可持續觀察 (更新時間: {now_hhmm()})"
    if stage == "①":
        return f"仍在築底，暫不介入 (更新時間: {now_hhmm()})"
    return f"結構不明，建議觀望 (更新時間: {now_hhmm()})"


def build_favorable_zone(stage, main_scenario, chip_warning):
    if chip_warning and stage in ["⑤", "⑥"]:
        return "🔴 警戒：主力出貨中"
    if stage in ["③", "④"]:
        return "是（主升段）"
    if stage == "⑤":
        return "否（高檔拉回風險升高）" if main_scenario in ["開高走低", "開低走低"] else "有限有利（末段攻擊）"
    if stage == "⑥":
        return "❌ 否（風險極高）"
    if stage == "②":
        return "觀察"
    return "否"


def ai_winrate_model(df, stage, chip_warning, us_block):
    score = 50
    if df is None or len(df) < 20:
        return 50, "資料不足"
    d = df.copy().reset_index(drop=True)
    t = d.iloc[-1]
    p = d.iloc[-2]
    close = float(t["close"]); open_p = float(t["open"]); high = float(t["max"]); low = float(t["min"]); prev_close = float(p["close"])
    score += 5 if close > open_p else -5
    pos = (close - low) / (high - low + 1e-6)
    score += 8 if pos > 0.7 else (-8 if pos < 0.3 else 0)
    change = (close - prev_close) / prev_close * 100 if prev_close != 0 else 0
    score += 6 if change > 2 else (-6 if change < -2 else 0)
    vol_ma5 = d["Trading_Volume"].rolling(5).mean().iloc[-1]
    if pd.notna(vol_ma5) and vol_ma5 != 0:
        vol_ratio = float(t["Trading_Volume"]) / float(vol_ma5)
        score += 6 if vol_ratio > 1.2 else (-4 if vol_ratio < 0.8 else 0)
    stage_score = {"③": 10, "④": 10, "⑤": -5, "⑥": -12, "①": -8, "②": 0}
    score += stage_score.get(stage, 0)
    score += -15 if chip_warning else 5
    if us_block and us_block["indices"]:
        for idx in us_block["indices"]:
            pct = idx.get("change_pct", 0)
            score += 5 if pct > 1 else (-5 if pct < -1 else 0)
    score = max(5, min(95, int(score)))
    if score >= 70: risk = "低風險（偏多）"
    elif score >= 55: risk = "中性偏多"
    elif score >= 45: risk = "震盪"
    elif score >= 30: risk = "偏空風險"
    else: risk = "高風險（不建議交易）"
    return score, risk


def ai_final_decision(winrate, stage, main_scenario, chip_warning):
    if chip_warning and stage in ["⑤", "⑥"]:
        return "❌ 主力出貨中，不可進場"
    if winrate >= 70: return "✅ 可積極操作（順勢）"
    if winrate >= 55: return "⚠️ 可小倉操作"
    if winrate >= 45: return "⚖️ 觀望為主"
    return "🚫 不建議進場"


# =========================
# 盤後分析主函式
# =========================
def analyze_after_stock(stock_id, direction="多"):
    df = fetch_daily_data(stock_id)
    if df is None:
        return {"stock": stock_id, "error": "無資料、資料不足或抓取失敗"}
    score_value = score_today(df)
    if score_value is None:
        return {"stock": stock_id, "error": "資料異常"}
    backtest_result = backtest_strategy(df)
    t = df.iloc[-1]
    close_price = round(float(t["close"]), 2)
    high_price = round(float(t["max"]), 2)
    low_price = round(float(t["min"]), 2)
    stage_info = detect_stage_advanced(df, stock_id)
    us_block = build_us_correlation_block(stock_id)
    chip_info = check_400_holder_change(stock_id)
    levels = build_after_levels(close_price, high_price, low_price, direction, stage=stage_info["stage"])
    scenario_info = build_next_day_scenarios(df, stage_info["stage"],
                                              chip_change=chip_info["change"] if chip_info["change"] is not None else 0.0,
                                              us_bias=us_block["bias"])
    favorable_zone = build_favorable_zone(stage_info["stage"], scenario_info["main_scenario"], stage_info["chip_warning"])
    comment = build_trader_comment(stage_info["stage"], stage_info["chip_warning"], scenario_info["main_scenario"], us_bias=us_block["bias"])
    winrate_ai, risk_ai = ai_winrate_model(df, stage_info["stage"], stage_info["chip_warning"], us_block)
    final_ai_decision = ai_final_decision(winrate_ai, stage_info["stage"], scenario_info["main_scenario"], stage_info["chip_warning"])
    risk_level = ("高" if favorable_zone in ["🔴 警戒：主力出貨中", "❌ 否（風險極高）", "否（高檔拉回風險升高）"]
                  else "中" if favorable_zone in ["有限有利（末段攻擊）", "觀察"] else "低")
    return {
        "stock": stock_id, "close": close_price, "high": high_price, "low": low_price,
        "score": score_value, "backtest": backtest_result,
        "pressure1": levels["pressure1"], "pressure2": levels["pressure2"],
        "support1": levels["support1"], "stop_loss": levels["stop_loss"],
        "stage": stage_info["stage"], "stage_name": stage_info["stage_name"],
        "stage_desc": stage_info["stage_desc"], "chip_warning": stage_info["chip_warning"],
        "chip_message": chip_info["message"], "chip_severity": chip_info["severity"],
        "favorable_zone": favorable_zone, "comment": comment, "scenario_info": scenario_info,
        "us_block": us_block, "risk_level": risk_level, "market_label": market_label(stock_id),
        "winrate_ai": winrate_ai, "risk_ai": risk_ai, "final_ai_decision": final_ai_decision
    }


# =========================
# 盤中：台股 / 美股雙市場
# =========================
def fugle_quote(symbol, api_key):
    headers = {"X-API-KEY": api_key}
    url = f"{FUGLE_BASE_URL}/intraday/quote/{symbol}"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_us_intraday_quote(symbol):
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period="2d", interval="1m")
        if hist is None or hist.empty:
            hist = ticker.history(period="5d", interval="1d")
            if hist is None or hist.empty:
                return None
            last = hist.iloc[-1]
            prev_close = hist["Close"].iloc[-2] if len(hist) >= 2 else last["Close"]
            last_price = float(last["Close"])
            ref_price = float(prev_close)
            change_pct = (last_price - ref_price) / ref_price * 100 if ref_price != 0 else 0
            return {"lastPrice": last_price, "openPrice": float(last["Open"]),
                    "highPrice": float(last["High"]), "lowPrice": float(last["Low"]),
                    "referencePrice": ref_price, "changePercent": round(change_pct, 2)}
        last_row = hist.iloc[-1]
        open_price = float(hist["Open"].iloc[0])
        high_price = float(hist["High"].max())
        low_price = float(hist["Low"].min())
        last_price = float(last_row["Close"])
        daily = ticker.history(period="5d", interval="1d")
        ref_price = float(daily["Close"].iloc[-2]) if daily is not None and len(daily) >= 2 else open_price
        change_pct = (last_price - ref_price) / ref_price * 100 if ref_price != 0 else 0
        return {"lastPrice": last_price, "openPrice": open_price, "highPrice": high_price,
                "lowPrice": low_price, "referencePrice": ref_price, "changePercent": round(change_pct, 2)}
    except Exception:
        return None


def fetch_live_quote(symbol):
    if is_us_symbol(symbol):
        return fetch_us_intraday_quote(symbol)
    api_key = get_secret("FUGLE_API_KEY", "")
    if not api_key:
        return None
    return fugle_quote(symbol, api_key)


# =========================
# 盤中獨立模式
# =========================
def build_intraday_independent_levels(price, open_price, high, low, ref_price, direction="多"):
    price = safe_float(price); open_price = safe_float(open_price)
    high = safe_float(high); low = safe_float(low); ref_price = safe_float(ref_price)
    range_now = max(high - low, 0.01)
    pivot = round((high + low + price) / 3, 2)
    if direction == "多":
        support1 = round(max(low, min(open_price, price)), 2)
        support2 = round(low, 2)
        pressure1 = round(max(pivot, price + range_now * 0.25), 2)
        pressure2 = round(max(high, pressure1 + range_now * 0.35), 2)
        strong_pressure = round(max(high, pressure2 + range_now * 0.25), 2)
        entry_price = round(max(price, support1 + range_now * 0.3), 2)
        stop_loss = round(support2, 2)
        tp1, tp2 = pressure1, pressure2
        if high > ref_price and price > open_price and price >= support1:
            status = f"盤中獨立模式：屬強勢震盪結構，但須等重新站穩確認位後再偏多操作。(更新時間: {now_hhmm()})"
        elif price < open_price and price <= support1:
            status = f"盤中獨立模式：早盤轉弱，先以防守為主。(更新時間: {now_hhmm()})"
        else:
            status = f"盤中獨立模式：高檔震盪中，暫不宜追價。(更新時間: {now_hhmm()})"
    else:
        pressure1 = round(max(high, price), 2)
        pressure2 = round(pressure1 + range_now * 0.3, 2)
        strong_pressure = pressure2
        support1 = round(min(low, price - range_now * 0.25), 2)
        support2 = round(low, 2)
        entry_price = support1
        stop_loss = round(pressure1, 2)
        tp1 = round(support1 - range_now * 0.3, 2)
        tp2 = round(support1 - range_now * 0.6, 2)
        if price < open_price and price < ref_price:
            status = f"盤中獨立模式：空方偏強，可留意跌破支撐後的續弱訊號。(更新時間: {now_hhmm()})"
        else:
            status = f"盤中獨立模式：尚未形成明確空方結構。(更新時間: {now_hhmm()})"
    return {"mode": "盤中獨立模式", "pressure1": pressure1, "pressure2": pressure2,
            "strong_pressure": strong_pressure, "support1": support1, "support2": support2,
            "entry_price": entry_price, "stop_loss": stop_loss, "tp1": tp1, "tp2": tp2, "status": status}


# =========================
# 盤中承接模式
# =========================
def build_intraday_plan(price, open_price, high, low, ref_price, pressure, support, direction="多"):
    if direction == "多":
        entry_price = round(pressure, 2)
        stop_loss = round(max(support, pressure * 0.995), 2)
        tp1 = round(pressure + (pressure - support) * 0.5, 2)
        tp2 = round(pressure + (pressure - support) * 1.0, 2)
        chase_gap = ((price - entry_price) / entry_price * 100) if entry_price != 0 else 0
        if price > pressure and price >= open_price and price >= ref_price and chase_gap <= 1.5:
            status = f"盤後承接模式：已突破壓力，可依紀律追蹤。(更新時間: {now_hhmm()})"
        elif price > pressure and chase_gap > 1.5:
            status = f"盤後承接模式：已突破但乖離偏大，不宜追價。(更新時間: {now_hhmm()})"
        elif high > pressure and price < pressure:
            status = f"盤後承接模式：盤中假突破，需防回落。(更新時間: {now_hhmm()})"
        else:
            status = f"盤後承接模式：尚未突破壓力。(更新時間: {now_hhmm()})"
    else:
        entry_price = round(support, 2)
        stop_loss = round(pressure * 1.015, 2)
        tp1 = round(support - (pressure - support) * 0.5, 2)
        tp2 = round(support - (pressure - support) * 1.0, 2)
        if price < support and price <= open_price and price <= ref_price:
            status = f"盤後承接模式：已跌破支撐，空方延續。(更新時間: {now_hhmm()})"
        elif low < support and price > support:
            status = f"盤後承接模式：盤中假跌破，空方追擊風險高。(更新時間: {now_hhmm()})"
        else:
            status = f"盤後承接模式：尚未跌破支撐。(更新時間: {now_hhmm()})"
    return {"mode": "盤後承接模式", "entry_price": entry_price, "stop_loss": stop_loss, "tp1": tp1, "tp2": tp2, "status": status}


# =========================
# 盤中 AI 決策
# =========================
def dynamic_intraday_levels(data, base_pressure, base_support):
    price = safe_float(data.get("lastPrice"))
    high = safe_float(data.get("highPrice"))
    low = safe_float(data.get("lowPrice"))
    return {
        "pressure": round(max(base_pressure, high), 2),
        "support": round(min(base_support, low), 2),
        "mid": round((max(base_pressure, high) + min(base_support, low)) / 2, 2)
    }


def intraday_ai_decision(data, pressure, support):
    price = safe_float(data.get("lastPrice"))
    open_p = safe_float(data.get("openPrice"))
    high = safe_float(data.get("highPrice"))
    low = safe_float(data.get("lowPrice"))
    ref = safe_float(data.get("referencePrice"))
    if price > pressure:
        if price > open_p and price > ref:
            return {"decision": "有效突破", "strength": "多方強勢", "action": "可順勢做多"}
        return {"decision": "假突破", "strength": "誘多", "action": "禁止追價"}
    elif price < support:
        return {"decision": "跌破支撐", "strength": "空方轉強", "action": "避免做多"}
    elif high >= pressure * 0.98:
        return {"decision": "高檔震盪", "strength": "動能不足", "action": "不追 / 逢高減碼"}
    elif low <= support:
        return {"decision": "支撐反彈", "strength": "短線止跌", "action": "可小倉試單"}
    return {"decision": "盤整", "strength": "方向不明", "action": "觀望"}


def generate_trade_plan(pressure, support, direction="多"):
    if direction == "多":
        entry = pressure; stop = support
        tp1 = round(pressure + (pressure - support) * 0.5, 2)
        tp2 = round(pressure + (pressure - support) * 1.0, 2)
    else:
        entry = support; stop = pressure
        tp1 = round(support - (pressure - support) * 0.5, 2)
        tp2 = round(support - (pressure - support) * 1.0, 2)
    return {"entry": round(entry, 2), "stop": round(stop, 2), "tp1": tp1, "tp2": tp2}


# =========================
# Session
# =========================
if "after_result" not in st.session_state:
    st.session_state.after_result = None
if "total_capital" not in st.session_state:
    st.session_state.total_capital = 1000000.0


# =========================
# UI
# =========================
render_battle_timetable()
render_time_banner()

tab1, tab2 = st.tabs(["盤後分析", "盤中判斷"])

with tab1:
    render_after_usage_note()
    st.title("盤後分析")

    with st.form("after_form"):
        stock_id = st.text_input("股票代號（台股輸入數字，美股輸入英文）", value="4906").strip()
        stock_name = st.text_input("股票名稱", value="").strip()
        market = st.text_input("市場別", value="").strip()
        direction = st.selectbox("操作方向", ["多", "空"])
        cost = st.text_input("買進或放空成本", value="")
        capital_str = st.text_input("總資金（元，用於部位計算）", value="1000000")
        submitted_after = st.form_submit_button("開始分析")

    if submitted_after:
        try:
            st.session_state.total_capital = float(capital_str.replace(",", ""))
        except Exception:
            st.session_state.total_capital = 1000000.0

        with st.spinner("分析中，請稍候..."):
            res = analyze_after_stock(stock_id, direction)

        if "error" in res:
            st.error(res["error"])
        else:
            df = fetch_daily_data(stock_id)
            auto_fundamental = build_auto_fundamental_summary(stock_id)
            us_block = res["us_block"]
            is_us = is_us_symbol(stock_id)

            st.session_state.after_result = {
                "stock": stock_id, "stock_name": stock_name,
                "market": market if market else res["market_label"],
                "cost": cost, "direction": direction,
                "auto_fundamental": auto_fundamental,
                "pressure1": res["pressure1"], "pressure2": res["pressure2"],
                "support1": res["support1"], "stop_loss": res["stop_loss"],
                "stage": res["stage"], "stage_name": res["stage_name"],
                "stage_desc": res["stage_desc"], "favorable_zone": res["favorable_zone"],
                "chip_warning": res["chip_warning"], "chip_message": res["chip_message"],
                "comment": res["comment"], "scenario_info": res["scenario_info"],
                "us_block": us_block, "risk_level": res["risk_level"],
                "market_label": res["market_label"], "winrate_ai": res["winrate_ai"],
                "risk_ai": res["risk_ai"], "final_ai_decision": res["final_ai_decision"]
            }

            # ---- AI 勝率 ----
            st.subheader("AI勝率分析")
            c1, c2 = st.columns(2)
            c1.metric("今日勝率", f"{res['winrate_ai']}%")
            c2.metric("風險等級", res["risk_ai"])

            st.subheader("AI最終決策")
            st.markdown(f"### {res['final_ai_decision']}")

            st.subheader("交易員判讀結論")
            st.markdown(f"### {res['comment']}")

            st.subheader("市場別")
            st.write(f"{res['market_label']} (更新時間: {now_hhmm()})")

            # ---- 風險燈號 ----
            st.subheader("風險燈號")
            risk_color = "#16a34a" if res["risk_level"] == "低" else "#d97706" if res["risk_level"] == "中" else "#dc2626"
            st.markdown(f"<div style='font-weight:700;color:{risk_color};'>風險等級：{res['risk_level']} (更新時間: {now_hhmm()})</div>", unsafe_allow_html=True)

            # ---- 基本面 ----
            st.subheader("基本面摘要")
            st.write(auto_fundamental)

            # ---- 美股連動 ----
            st.subheader("美股連動觀測")
            st.write(f"對位邏輯：{us_block['mapping_label']} (更新時間: {now_hhmm()})")
            st.write(f"美股連動偏向：{us_block['bias']} (更新時間: {now_hhmm()})")
            if us_block["indices"]:
                cols = st.columns(len(us_block["indices"]))
                for i, item in enumerate(us_block["indices"]):
                    cols[i].metric(item["ticker"], f"{item['change_pct']}%", f"收盤 {fmt_num(item['latest_close'])}")
            else:
                st.write(f"無法取得對應美股指數資料 (更新時間: {now_hhmm()})")

            # ---- 階段 ----
            st.subheader("目前階段")
            c1, c2 = st.columns([1, 2])
            c1.metric("階段", f"{res['stage']} {res['stage_name']}")
            fz = res["favorable_zone"]
            if fz in ["🔴 警戒：主力出貨中", "❌ 否（風險極高）", "否（高檔拉回風險升高）"]:
                c2.markdown(f"<div style='font-size:1.02rem;font-weight:700;color:#ff4b4b;padding-top:0.4rem;'>{fz} (更新時間: {now_hhmm()})</div>", unsafe_allow_html=True)
            elif fz in ["有限有利（末段攻擊）", "觀察"]:
                c2.markdown(f"<div style='font-size:1.02rem;font-weight:700;color:#d97706;padding-top:0.4rem;'>{fz} (更新時間: {now_hhmm()})</div>", unsafe_allow_html=True)
            else:
                c2.markdown(f"<div style='font-size:1.02rem;font-weight:700;color:#16a34a;padding-top:0.4rem;'>{fz} (更新時間: {now_hhmm()})</div>", unsafe_allow_html=True)
            st.write(res["stage_desc"])

            # ---- 大戶籌碼 ----
            st.subheader("大戶籌碼增減狀況")
            if res["chip_warning"]:
                st.markdown(f"<span style='color:#ff4b4b;font-weight:700;'>{res['chip_message']}</span>", unsafe_allow_html=True)
            else:
                st.write(res["chip_message"])

            # ---- 明日走勢 ----
            st.subheader("明日走勢推演（盤前規劃用）")
            s = res["scenario_info"]["scenarios"]
            g1, g2 = st.columns(2); g3, g4 = st.columns(2)
            g1.metric("開高走高", f"{s['開高走高']}%"); g2.metric("開高走低", f"{s['開高走低']}%")
            g3.metric("開低走高", f"{s['開低走高']}%"); g4.metric("開低走低", f"{s['開低走低']}%")
            st.caption(f"更新時間: {now_hhmm()}")
            st.subheader("主要劇本")
            st.write(f"{res['scenario_info']['main_scenario']} (更新時間: {now_hhmm()})")
            st.subheader("盤前應對")
            st.write(res["scenario_info"]["response_note"])

            # ---- 關鍵價位 ----
            st.subheader("關鍵價位")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("壓力一", fmt_num(res["pressure1"])); d2.metric("壓力二", fmt_num(res["pressure2"]))
            d3.metric("支撐", fmt_num(res["support1"])); d4.metric("停損", fmt_num(res["stop_loss"]))
            st.caption(f"更新時間: {now_hhmm()}")

            st.divider()

            # ============================================================
            # 補強區塊一：選股條件四項全面檢查
            # ============================================================
            st.subheader("選股條件檢查（系統選股四項）")
            if df is not None:
                screen = screen_stock_conditions(df, stock_id, is_us=is_us)
                if screen["pass_all"]:
                    st.success(screen["summary"])
                else:
                    st.error(screen["summary"])
                for cond in screen["conditions"]:
                    icon = "✅" if cond["pass"] else "❌"
                    st.markdown(f"{icon} **{cond['name']}**：{cond['detail']}")
            else:
                st.warning("無法取得日線資料進行選股條件檢查。")

            st.divider()

            # ============================================================
            # 補強區塊二：假訊號偵測
            # ============================================================
            st.subheader("假訊號偵測（五項判死條件）")
            if df is not None:
                is_false, triggers, false_detail = check_false_signal(df, us_block)
                if is_false:
                    st.error(false_detail)
                else:
                    st.success(false_detail)
            else:
                st.warning("無法取得資料進行假訊號偵測。")

            st.divider()

            # ============================================================
            # 補強區塊三：5號門 / 6號門 精確判斷
            # ============================================================
            st.subheader("門號精確判斷（5號門 / 6號門）")
            if df is not None:
                pressure_val = res.get("pressure1")
                g6 = gate6_confirm(df, us_block, pressure=pressure_val)
                g5 = gate5_confirm(df, us_block, pressure=pressure_val)

                if g6["is_gate6"]:
                    st.success(f"**6號門確認** — {g6['summary']}")
                    gate_label = "6"
                elif g5["is_gate5"]:
                    st.warning(f"**5號門卡位** — {g5['summary']}")
                    gate_label = "5"
                else:
                    st.info(f"**尚未達門號標準** — {g5['summary']}")
                    gate_label = "觀望"

                with st.expander("查看6號門五項條件逐項明細"):
                    for cond in g6["conditions"]:
                        icon = "✅" if cond["pass"] else "❌"
                        st.markdown(f"{icon} **{cond['name']}**：{cond['detail']}")
            else:
                gate_label = "觀望"
                g6 = {"stop_loss": res.get("stop_loss")}
                g5 = {"stop_loss": res.get("stop_loss")}
                st.warning("無法取得資料進行門號判斷。")

            st.divider()

            # ============================================================
            # 補強區塊四：資金與部位管理
            # ============================================================
            st.subheader("資金與部位管理")
            total_capital = st.session_state.total_capital
            stop_loss_price = g6.get("stop_loss") or g5.get("stop_loss") or res.get("stop_loss")
            entry_price_val = res.get("pressure1", 0)

            if total_capital > 0 and stop_loss_price and entry_price_val:
                pos = calc_position_size(
                    total_capital=total_capital,
                    entry_price=float(entry_price_val),
                    stop_loss_price=float(stop_loss_price),
                    gate=gate_label if gate_label in ["5", "6"] else "6",
                    risk_pct=1.5,
                    is_tw_stock=not is_us
                )
                if "error" not in pos:
                    unit_key = "suggested_lots" if not is_us else "suggested_shares"
                    unit_label = "建議張數" if not is_us else "建議股數"
                    cp1, cp2, cp3 = st.columns(3)
                    cp1.metric(unit_label, pos.get(unit_key, "-"))
                    cp2.metric("預估最大虧損", f"{pos['actual_risk_amount']:,.0f}")
                    cp3.metric("占總資金比", f"{pos['actual_risk_pct']}%")
                    if pos["actual_risk_pct"] > 2.0:
                        st.error(f"警告：預估虧損占比{pos['actual_risk_pct']}%，超過2%上限，請縮減部位。")
                    else:
                        st.success(pos["note"])
                else:
                    st.warning(pos["error"])
            else:
                st.info("請確認總資金、壓力價與停損價是否已正確設定。")

            # 攤平防護
            try:
                cost_val = float(cost.replace(",", "")) if cost else None
            except Exception:
                cost_val = None
            if cost_val and df is not None:
                current_price = float(df.iloc[-1]["close"])
                avg_check = check_no_averaging_down(cost_val, current_price, direction)
                if avg_check["warning"]:
                    st.error(f"攤平警告：{avg_check['message']}")

            st.divider()

            # ============================================================
            # 補強區塊五：出場條件系統性觸發
            # ============================================================
            st.subheader("出場條件監測（七項系統觸發）")
            if df is not None:
                breakout_price = float(entry_price_val) if entry_price_val else 0.0
                sl_price = float(stop_loss_price) if stop_loss_price else 0.0
                exit_result = check_exit_conditions(
                    df=df, stop_loss_price=sl_price, breakout_price=breakout_price,
                    direction=direction, us_block=us_block
                )
                if exit_result["should_exit"]:
                    if exit_result["priority"] == "立即出場":
                        st.error(exit_result["detail"])
                    else:
                        st.warning(exit_result["detail"])
                else:
                    st.success(exit_result["detail"])
            else:
                st.warning("無法取得資料進行出場條件監測。")

            st.divider()

            # ============================================================
            # 補強區塊六：盤前三問自我檢核
            # ============================================================
            st.subheader("盤前三問自我檢核")
            q1 = gate_label if gate_label in ["5", "6"] else "尚未確認"
            q2 = f"{float(stop_loss_price):.2f}" if stop_loss_price else "尚未設定"
            q3 = "可以立刻走" if stop_loss_price else "停損未設定，禁止下單"
            col1, col2, col3 = st.columns(3)
            col1.metric("現在是幾號門？", q1)
            col2.metric("停損在哪？", q2)
            col3.metric("錯了能立刻走嗎？", q3)
            if gate_label == "觀望" or not stop_loss_price:
                st.error("三問中有項目未確認，系統建議今日不下單。")
            else:
                st.success("三問均已確認，可依紀律執行操作。")


with tab2:
    render_intra_usage_note()
    st.title("盤中判斷")

    after_data = st.session_state.after_result
    default_stock = after_data["stock"] if after_data else "4906"
    default_name = after_data["stock_name"] if after_data else ""
    default_market = after_data["market"] if after_data else market_label("4906")
    default_cost = after_data["cost"] if after_data else ""
    default_direction = after_data["direction"] if after_data else "多"
    default_pressure = float(after_data["pressure1"]) if after_data else None
    default_support = float(after_data["support1"]) if after_data else None
    default_fundamental = after_data["auto_fundamental"] if after_data else build_auto_fundamental_summary(default_stock)
    default_stage = after_data["stage"] if after_data else "-"
    default_stage_name = after_data["stage_name"] if after_data else "未判斷"
    default_stage_desc = after_data["stage_desc"] if after_data else ""

    if after_data:
        st.info(f"已載入盤後分析資料：{default_stock}（{default_market}）｜方向：{default_direction}｜壓力：{default_pressure}｜支撐：{default_support}")

    with st.form("intra_form"):
        intra_stock = st.text_input("股票代號", value=default_stock).strip()
        intra_direction = st.selectbox("操作方向", ["多", "空"], index=0 if default_direction == "多" else 1)
        intra_pressure = st.number_input("壓力價（盤後帶入或手動輸入）", value=default_pressure or 0.0, step=0.1)
        intra_support = st.number_input("支撐價（盤後帶入或手動輸入）", value=default_support or 0.0, step=0.1)
        mode = st.radio("盤中模式", ["盤後承接模式（沿用昨日壓力支撐）", "盤中獨立模式（純盤中即時計算）"])
        submitted_intra = st.form_submit_button("取得即時報價並判斷")

    if submitted_intra:
        with st.spinner("抓取即時報價中..."):
            quote = fetch_live_quote(intra_stock)

        if quote is None:
            st.error("無法取得即時報價，請確認代號或 API Key 是否正確。")
        else:
            price = safe_float(quote.get("lastPrice"))
            open_price = safe_float(quote.get("openPrice"))
            high = safe_float(quote.get("highPrice"))
            low = safe_float(quote.get("lowPrice"))
            ref_price = safe_float(quote.get("referencePrice"))
            change_pct = safe_float(quote.get("changePercent"))

            st.subheader("即時報價")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("現價", fmt_num(price), f"{change_pct:+.2f}%")
            q2.metric("開盤", fmt_num(open_price))
            q3.metric("今高", fmt_num(high))
            q4.metric("今低", fmt_num(low))
            st.caption(f"更新時間: {now_hhmm()}")

            st.subheader("基本面摘要")
            st.write(default_fundamental)

            st.subheader(f"目前階段：{default_stage} {default_stage_name}")
            st.write(default_stage_desc)

            dyn = dynamic_intraday_levels(quote, intra_pressure, intra_support)
            ai_dec = intraday_ai_decision(quote, dyn["pressure"], dyn["support"])
            trade_plan = generate_trade_plan(dyn["pressure"], dyn["support"], intra_direction)

            st.subheader("動態壓力支撐（盤中更新）")
            lv1, lv2, lv3 = st.columns(3)
            lv1.metric("壓力", fmt_num(dyn["pressure"]))
            lv2.metric("中間軸", fmt_num(dyn["mid"]))
            lv3.metric("支撐", fmt_num(dyn["support"]))

            st.subheader("AI 盤中決策")
            da1, da2, da3 = st.columns(3)
            da1.metric("訊號", ai_dec["decision"])
            da2.metric("強度", ai_dec["strength"])
            da3.metric("建議動作", ai_dec["action"])

            st.subheader("交易計畫")
            tp1, tp2, tp3, tp4 = st.columns(4)
            tp1.metric("進場價", fmt_num(trade_plan["entry"]))
            tp2.metric("停損", fmt_num(trade_plan["stop"]))
            tp3.metric("目標一", fmt_num(trade_plan["tp1"]))
            tp4.metric("目標二", fmt_num(trade_plan["tp2"]))

            if "盤後承接" in mode and intra_pressure and intra_support:
                plan = build_intraday_plan(price, open_price, high, low, ref_price, intra_pressure, intra_support, intra_direction)
            else:
                plan = build_intraday_independent_levels(price, open_price, high, low, ref_price, intra_direction)

            st.subheader(f"模式：{plan['mode']}")
            st.write(plan["status"])
            mp1, mp2, mp3, mp4 = st.columns(4)
            mp1.metric("進場價", fmt_num(plan["entry_price"]))
            mp2.metric("停損", fmt_num(plan["stop_loss"]))
            mp3.metric("目標一", fmt_num(plan["tp1"]))
            mp4.metric("目標二", fmt_num(plan["tp2"]))

            # ---- 盤中出場條件監測 ----
            st.subheader("盤中出場條件監測")
            df_intra = fetch_daily_data(intra_stock)
            if df_intra is not None:
                us_block_intra = build_us_correlation_block(intra_stock)
                exit_result_intra = check_exit_conditions(
                    df=df_intra,
                    stop_loss_price=float(plan["stop_loss"]),
                    breakout_price=float(dyn["pressure"]),
                    direction=intra_direction,
                    us_block=us_block_intra
                )
                if exit_result_intra["should_exit"]:
                    if exit_result_intra["priority"] == "立即出場":
                        st.error(exit_result_intra["detail"])
                    else:
                        st.warning(exit_result_intra["detail"])
                else:
                    st.success(exit_result_intra["detail"])
            else:
                st.info("無法取得歷史資料進行出場條件監測。")

            st.caption(f"更新時間: {now_hhmm()}")
