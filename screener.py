#!/usr/bin/env python3
"""
NSE Momentum Screener  --  Banking Knowledge with Anky

Filters (sab edit kar sakte ho neeche CONFIG me):
  1. RS Rating            >= 80   (IBD-style, 1-99, poore NSE universe me percentile)
  2. Market Cap           >  500 crore
  3. Price                >  200 DMA
  4. 30-day avg turnover  >  7 crore   (close x volume ka 30 session average)
  5. Price                >  10 INR

Install:   pip install yfinance pandas requests
Run:       python screener.py
Output:    screener_results.csv  +  screener_results.html (sortable table)

Optional:  python screener.py --symbols my_list.txt   (ek line me ek NSE symbol)
"""
import argparse
import io
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# ----------------------------- CONFIG ---------------------------------------
CONFIG = dict(
    min_rs=80,
    min_mcap_cr=500,
    dma_len=200,
    min_turnover_cr=7,
    turnover_days=30,
    min_price=10,
    batch_size=150,        # kitne tickers ek baar me download hon
    history="16mo",        # 252 trading days + 200 DMA ke liye kaafi
)
CR = 1e7  # 1 crore = 1,00,00,000
NSE_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
# ----------------------------------------------------------------------------


def load_universe(path=None):
    """NSE ke saare EQ-series symbols."""
    if path:
        syms = [s.strip().upper() for s in open(path) if s.strip()]
        return syms
    import os
    import requests
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(3):
        try:
            r = requests.get(NSE_LIST_URL, headers=headers, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            df = df[df["SERIES"].str.strip() == "EQ"]
            syms = sorted(df["SYMBOL"].str.strip().unique())
            open("symbols.txt", "w").write("\n".join(syms))  # backup copy
            return syms
        except Exception as e:
            print(f"  NSE list attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    if os.path.exists("symbols.txt"):  # pichli baar ki saved list
        print("  NSE list nahi mili, pichli saved list use ho rahi hai")
        return [s.strip() for s in open("symbols.txt") if s.strip()]
    raise SystemExit("NSE symbol list nahi mili aur backup symbols.txt bhi nahi hai.")


def download_prices(symbols, cfg):
    """Returns (close_df, volume_df) -- columns = symbol."""
    import yfinance as yf
    closes, vols = {}, {}
    n = len(symbols)
    for i in range(0, n, cfg["batch_size"]):
        chunk = symbols[i:i + cfg["batch_size"]]
        tickers = [s + ".NS" for s in chunk]
        print(f"  downloading {i + 1}-{min(i + len(chunk), n)} / {n}", flush=True)
        try:
            data = yf.download(tickers, period=cfg["history"], interval="1d",
                               auto_adjust=True, group_by="ticker",
                               threads=True, progress=False)
        except Exception as e:  # network hiccup -> skip batch
            print("   batch failed:", e)
            continue
        for s, t in zip(chunk, tickers):
            try:
                d = data[t] if len(tickers) > 1 else data
                d = d.dropna(subset=["Close"])
                if len(d):
                    closes[s] = d["Close"]
                    vols[s] = d["Volume"]
            except KeyError:
                pass
        time.sleep(1)
    return pd.DataFrame(closes), pd.DataFrame(vols)


def rs_rating(close):
    """
    IBD-style RS: weighted 12-month performance, last quarter double weight.
    score = 0.4*ROC(63) + 0.2*ROC(126) + 0.2*ROC(189) + 0.2*ROC(252)
    phir poore universe me percentile -> 1..99.
    Jis stock ki 252 din ki history nahi (naye IPO), uska RS NaN rahega.
    """
    def roc(n):
        return close.iloc[-1] / close.iloc[-1 - n] - 1 if len(close) > n else pd.Series(np.nan, index=close.columns)

    score = 0.4 * roc(63) + 0.2 * roc(126) + 0.2 * roc(189) + 0.2 * roc(252)
    rating = score.rank(pct=True) * 98 + 1
    return rating.round(0)


def compute_metrics(close, volume, cfg):
    last = close.ffill().iloc[-1]
    dma = close.rolling(cfg["dma_len"]).mean().iloc[-1]
    turnover = (close * volume).tail(cfg["turnover_days"]).mean()
    m = pd.DataFrame({
        "Price": last.round(2),
        "DMA200": dma.round(2),
        "Above200DMA_%": ((last / dma - 1) * 100).round(1),
        "AvgTurnover_Cr": (turnover / CR).round(2),
        "RS": rs_rating(close.ffill()),
    })
    m["Pct_from_52wH"] = ((last / close.tail(252).max() - 1) * 100).round(1)
    return m


def apply_filters(m, cfg):
    f = (
        (m["RS"] >= cfg["min_rs"]) &
        (m["Price"] > m["DMA200"]) &
        (m["Price"] > cfg["min_price"]) &
        (m["AvgTurnover_Cr"] > cfg["min_turnover_cr"])
    )
    return m[f].copy()


def add_market_cap(df, cfg):
    """Market cap sirf bache hue stocks ke liye (slow call, isliye last me)."""
    import yfinance as yf
    caps = []
    for s in df.index:
        try:
            mc = yf.Ticker(s + ".NS").fast_info["market_cap"]
        except Exception:
            mc = np.nan
        caps.append(mc / CR if mc else np.nan)
    df["MCap_Cr"] = np.round(caps, 0)
    return df[df["MCap_Cr"] > cfg["min_mcap_cr"]]


def save_html(df, path):
    html = df.reset_index().rename(columns={"index": "Symbol"}).to_html(
        index=False, table_id="t", border=0, na_rep="-")
    page = f"""<!doctype html><meta charset=utf-8><title>NSE Screener</title>
<style>body{{font:14px system-ui;margin:24px}}table{{border-collapse:collapse}}
th,td{{padding:6px 12px;border-bottom:1px solid #ddd;text-align:right}}
th{{cursor:pointer;background:#f4f4f4}}td:first-child,th:first-child{{text-align:left}}</style>
<h2>NSE Momentum Screener <small>({datetime.now():%d %b %Y})</small></h2>
<p>RS &ge; 80 &middot; MCap &gt; 500 Cr &middot; Price &gt; 200 DMA &middot; 30D avg turnover &gt; 7 Cr &middot; Price &gt; 10</p>
{html}
<script>
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{{
 const tb=document.querySelector('#t tbody'),rows=[...tb.rows],d=th.d=!th.d;
 rows.sort((a,b)=>{{const x=a.cells[i].innerText,y=b.cells[i].innerText,n=parseFloat(x)-parseFloat(y);
  return (isNaN(n)?x.localeCompare(y):n)*(d?1:-1)}});rows.forEach(r=>tb.appendChild(r))}});
</script>"""
    open(path, "w", encoding="utf-8").write(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="text file, ek line me ek NSE symbol")
    a, _ = ap.parse_known_args()  # Colab/Jupyter ke extra arguments ignore karo
    cfg = CONFIG

    print("1/4 Universe load ho raha hai...")
    syms = load_universe(a.symbols)
    print(f"    {len(syms)} symbols")

    print("2/4 Price history download...")
    close, vol = download_prices(syms, cfg)
    if close.empty:
        sys.exit("Koi data nahi mila -- internet / yfinance check karo.")

    print("3/4 RS, 200 DMA, turnover calculate...")
    m = compute_metrics(close, vol, cfg)
    short = apply_filters(m, cfg)
    print(f"    {len(short)} stocks market-cap check se pehle")

    print("4/4 Market cap check...")
    res = add_market_cap(short, cfg).sort_values("RS", ascending=False)

    cols = ["MCap_Cr", "Price", "RS", "DMA200", "Above200DMA_%",
            "AvgTurnover_Cr", "Pct_from_52wH"]
    res = res[cols]
    res.to_csv("screener_results.csv")
    save_html(res, "screener_results.html")
    print(f"\n{len(res)} stocks pass hue. Saved: screener_results.csv / .html\n")
    print(res.head(25).to_string())


if __name__ == "__main__":
    main()
