#!/usr/bin/env python3
"""
A股云端扫描脚本 — GitHub Actions 用
仅做 Layer 1 (量化初筛) + Layer 2 (K线技术过滤)
发现候选股时通过钉钉推送，建议本地深度分析
"""
import os, sys, json, time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np

# ============================================================
# 配置
# ============================================================
DINGTALK_URL = os.getenv("DINGTALK_WEBHOOK", "")
CST = timezone(timedelta(hours=8))
UA = "Mozilla/5.0 (compatible; StockAlertCI/1.0)"
EM_HEADERS = {"User-Agent": UA, "Referer": "https://data.eastmoney.com/"}

# ============================================================
# 工具函数
# ============================================================

def em_get(url, params=None):
    """东方财富 API 请求"""
    r = requests.get(url, params=params, headers=EM_HEADERS, timeout=15)
    return r


def send_dingtalk(title, text):
    """钉钉推送"""
    if not DINGTALK_URL:
        return False
    # 确保含关键词 "k线"
    if "k线" not in text.lower() and "K线" not in text:
        text = "k线 " + text
    try:
        r = requests.post(DINGTALK_URL, json={
            "msgtype": "markdown",
            "markdown": {"title": title[:64], "text": text}
        }, headers={"Content-Type": "application/json"}, timeout=10)
        return r.json().get("errcode") == 0
    except Exception as e:
        print(f"  DingTalk error: {e}")
        return False


# ============================================================
# Layer 1: 量化初筛
# ============================================================

def get_market_data():
    """获取全市场行情 (成交额 Top 100)"""
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1", "pz": "100", "po": "1", "np": "1",
        "fltt": "2", "invt": "2", "fid": "f8",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields": "f2,f3,f5,f6,f9,f12,f14,f20,f23,f62",
    }
    r = em_get(url, params)
    data = r.json()
    stocks = []
    if data and data.get("data") and data["data"].get("diff"):
        for item in data["data"]["diff"]:
            stocks.append({
                "code": item.get("f12", ""),
                "name": item.get("f14", ""),
                "price": item.get("f2", 0) or 0,
                "change_pct": item.get("f3", 0) or 0,
                "volume": item.get("f5", 0) or 0,
                "amount": item.get("f6", 0) or 0,
                "pe": item.get("f9", 0) or 0,
                "market_cap": item.get("f20", 0) or 0,
                "main_inflow": item.get("f62", 0) or 0,
            })
    return stocks


def layer1_filter(stocks):
    """
    严格保守初筛:
    - 非ST | 涨幅 2-18% (区分主板/创业板) | 成交额 > 1亿
    - PE 10-50 | 市值 > 50亿 | 主力净流入 > 0
    """
    passed = []
    for s in stocks:
        name = s["name"]
        code = s["code"]
        chg = s["change_pct"]
        pe = s["pe"]
        cap = s["market_cap"]
        amt = s["amount"]
        inflow = s["main_inflow"]

        if "ST" in name or "*ST" in name:
            continue
        if pe <= 0 or pe > 50:
            continue
        if cap < 50e8:
            continue
        if amt < 1e8:
            continue
        if inflow <= 0:
            continue

        # 区分涨跌幅限制
        is_20pct = code.startswith("30") or code.startswith("688")
        max_chg = 18 if is_20pct else 9
        if chg < 2 or chg > max_chg:
            continue

        passed.append(s)

    passed.sort(key=lambda x: x.get("main_inflow", 0), reverse=True)
    return passed


# ============================================================
# Layer 2: K线技术过滤
# ============================================================

def fetch_kline_sina(code, days=60):
    """从新浪 HTTP API 获取K线"""
    if code.startswith("6") or code.startswith("9"):
        symbol = f"sh{code}"
    else:
        symbol = f"sz{code}"

    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {"symbol": symbol, "scale": "240", "ma": "no", "datalen": str(days + 5)}
    try:
        r = requests.get(url, params=params, timeout=15,
                        headers={"User-Agent": UA, "Referer": "https://finance.sina.com.cn/"})
        data = r.json()
        if not data or len(data) < 25:
            return None
        df = pd.DataFrame(data)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["close"])
        return df.tail(days) if len(df) >= 25 else None
    except Exception:
        return None


def calc_indicators(df):
    """计算 MACD / RSI / 均线 / 量比"""
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macdh = macd_line - signal

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 均线
    sma50 = close.rolling(50).mean().iloc[-1]

    # 量比
    vol5 = vol.tail(5).mean()
    vol20 = vol.tail(20).mean()

    # 近期涨跌幅
    p = close.iloc[-1]
    chg5 = (p / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
    chg20 = (p / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0

    return {
        "price": round(p, 2),
        "sma50": round(sma50, 2),
        "macd_line": round(macd_line.iloc[-1], 3),
        "macdh": round(macdh.iloc[-1], 3),
        "rsi": round(rsi.iloc[-1], 1),
        "vol_ratio": round(vol5 / vol20, 2) if vol20 > 0 else 0,
        "chg_5d": round(chg5, 1),
        "chg_20d": round(chg20, 1),
    }


def layer2_filter(candidates):
    """
    K线技术过滤 (严格保守):
    MACD主线>0 & MACD柱>0 | RSI 40-70 | 价格>50SMA | 量比>0.8
    """
    passed = []
    for s in candidates:
        code = s["code"]
        df = fetch_kline_sina(code)
        time.sleep(0.3)

        if df is None:
            continue

        ind = calc_indicators(df)
        if ind is None:
            continue

        # 技术面检查
        if not (ind["macd_line"] > 0 and ind["macdh"] > 0):
            continue
        if not (40 <= ind["rsi"] <= 70):
            continue
        if not (ind["price"] > ind["sma50"]):
            continue
        if not (ind["vol_ratio"] > 0.8):
            continue

        s["indicators"] = ind
        passed.append(s)

    return passed


# ============================================================
# 报告 & 推送
# ============================================================

def generate_dingtalk(l1_count, l2_candidates):
    """生成钉钉推送内容"""
    now = datetime.now(CST)
    hour = now.hour
    scan_name = "盘前" if hour < 10 else ("开盘" if hour < 12 else "午后")

    if not l2_candidates:
        text = f"## 📋 A股k线扫描 — {now.strftime('%m-%d %H:%M')} ({scan_name})\n\n"
        text += f"> 严格保守过滤 | 初筛 {l1_count}只 → **k线技术通过: 0只**\n\n"
        text += "⚠️ 今日暂无符合严格买入条件的股票，建议观望。\n\n"
        text += f"---\n📅 {now.strftime('%Y-%m-%d %H:%M')} | 下次扫描见"
        return "A股k线扫描无信号", text

    # 有候选股！
    text = f"## 🔍 A股k线扫描 — {now.strftime('%m-%d %H:%M')} ({scan_name})\n\n"
    text += f"> 严格保守过滤 | 初筛 {l1_count}只 → **k线技术通过: {len(l2_candidates)}只**\n\n"
    text += "### 📊 技术面全通过的候选股\n\n"
    text += "| 代码 | 名称 | 价格 | 涨跌 | PE | MACDh | RSI | >50SMA |\n"
    text += "|------|------|------|------|----|------|-----|--------|\n"

    for s in l2_candidates:
        ind = s.get("indicators", {})
        code = s["code"]
        name = s["name"]
        text += f"| {code} | {name} | {s['price']:.2f} | "
        text += f"+{s['change_pct']:.1f}% | {s['pe']:.0f} | "
        text += f"{ind.get('macdh', 0):+.3f} | {ind.get('rsi', 0)} | ✅ |\n"

    text += "\n---\n"
    text += "> 💡 以上股票已通过量化+K线双重筛选，建议进行AI深度分析确认\n"
    text += f"> 📅 {now.strftime('%Y-%m-%d %H:%M:%S')}"
    return f"🔔 k线候选 {len(l2_candidates)}只", text


def main():
    now = datetime.now(CST)
    print(f"A股云端扫描 — {now.strftime('%Y-%m-%d %H:%M')} CST")
    print(f"模式: Layer 1+2 (量化 + K线技术)")

    # Layer 1
    print("[Layer 1] 获取全市场数据...")
    all_stocks = get_market_data()
    l1 = layer1_filter(all_stocks)
    print(f"  初筛通过: {len(l1)}只")
    for s in l1[:8]:
        print(f"  {s['code']} {s['name']} ¥{s['price']:.2f} +{s['change_pct']:.1f}% PE{s['pe']:.0f}")

    if not l1:
        title, text = generate_dingtalk(0, [])
        send_dingtalk(title, text)
        print("无候选股，扫描结束")
        return

    # Layer 2
    print(f"\n[Layer 2] K线技术过滤 ({len(l1)}只)...")
    l2 = layer2_filter(l1)
    print(f"  技术通过: {len(l2)}只")
    for s in l2:
        ind = s["indicators"]
        print(f"  ✅ {s['code']} {s['name']} MACDh={ind['macdh']:+.3f} RSI={ind['rsi']} "
              f">50SMA({ind['sma50']}) 量比={ind['vol_ratio']}x")

    # 推送
    title, text = generate_dingtalk(len(l1), l2)
    ok = send_dingtalk(title, text)
    print(f"\n钉钉推送: {'✅' if ok else '❌'}")
    print(f"扫描完成: L1={len(l1)} → L2={len(l2)}")


if __name__ == "__main__":
    main()
