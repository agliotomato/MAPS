"""
Market Agent — 시장 기술적 분석 모듈

notebook(agentic/market_agent.ipynb)에서 추출한 Python 모듈.
추세, 모멘텀, 수급, 상대강도, 리스크를 분석하고
Timing Readiness(0-100)를 산출합니다.

사용법:
    from market_agent.market_agent import MarketAgent

    agent = MarketAgent(use_gpt=False)
    result = agent.run(collect_data=True)
"""

from datetime import datetime
from pathlib import Path
from io import StringIO
import time
import re
import json
import os

import numpy as np
import pandas as pd
import requests

# ── 경로 설정 (모듈 기준 상대 경로) ──
_MODULE_DIR = Path(__file__).resolve().parent
OUT_DIR = _MODULE_DIR / "market_agent_data"
KEY_FILE = _MODULE_DIR.parent / "OpenAI_key.txt"

PRICE_FILE = OUT_DIR / "market_price_data.csv"
FLOW_FILE = OUT_DIR / "market_flow_data.csv"
FEATURE_FILE = OUT_DIR / "market_feature_data.csv"
REPORT_JSON_FILE = OUT_DIR / "market_agent_report.json"
REPORT_TABLE_FILE = OUT_DIR / "market_agent_report_table.csv"
FINAL_TXT_FILE = OUT_DIR / "market_agent_final_report.txt"
LLM_REPORT_FILE = OUT_DIR / "market_agent_llm_report.txt"
LLM_JSON_FILE = OUT_DIR / "market_agent_llm_report.json"
INTEGRATION_PAYLOAD_FILE = OUT_DIR / "market_agent_integration_payload.json"

START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

KR_STOCKS = {
    "005930": "Samsung Electronics",
    "000660": "SK Hynix",
    "042700": "Hanmi Semiconductor",
}

KR_INDEXES = {
    "KS11": "KOSPI",
    "KQ11": "KOSDAQ",
}

GLOBAL_ASSETS = {
    "SOXX": ("iShares Semiconductor ETF", "US_ETF"),
    "SMH": ("VanEck Semiconductor ETF", "US_ETF"),
    "^IXIC": ("NASDAQ Composite", "US_INDEX"),
    "^GSPC": ("S&P 500", "US_INDEX"),
    "KRW=X": ("USD/KRW", "FX"),
}


def ensure_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_openai_key(key_file=KEY_FILE):
    key_file = Path(key_file)
    if not key_file.exists():
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            return api_key
        raise FileNotFoundError(f"OpenAI API key not found: {key_file.resolve()}")
    key = key_file.read_text(encoding="utf-8").strip()
    if "=" in key and "OPENAI_API_KEY" in key:
        key = key.split("=", 1)[1].strip()
    return key.strip().strip('"').strip("'")


# ── Utility 함수 ──

def clean_number(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace(",", "").replace("%", "").replace("+", "")
    x = x.replace("−", "-").replace("▲", "").replace("▼", "-")
    x = re.sub(r"[^0-9.\-]", "", x)
    if x in ["", "-", "."]:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x).strip() for x in col if str(x) != "nan"])
            for col in df.columns
        ]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(columns, must_include=None, exclude=None):
    must_include = must_include or []
    exclude = exclude or []
    for col in columns:
        col_str = str(col).strip()
        if all(key in col_str for key in must_include) and not any(
            key in col_str for key in exclude
        ):
            return col
    return None


def safe_value(row, col):
    val = row.get(col, np.nan)
    if pd.isna(val):
        return None
    return val


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def fmt_num(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:,.0f}"


def to_long_price_df(df, asset_code, asset_name, market, source):
    columns = [
        "Date", "AssetCode", "AssetName", "Market", "Source",
        "Open", "High", "Low", "Close", "Volume", "Value",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    out = df.copy().reset_index()
    if "Date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Date"})
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in out.columns:
            out[col] = np.nan
    if "Value" not in out.columns:
        out["Value"] = np.nan

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["AssetCode"] = asset_code
    out["AssetName"] = asset_name
    out["Market"] = market
    out["Source"] = source
    out = out[columns].copy()
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def timing_zone(score):
    score = int(max(0, min(100, score)))
    if score >= 80:
        return "Strong Entry Zone"
    if score >= 60:
        return "Entry Candidate"
    if score >= 40:
        return "Neutral"
    if score >= 20:
        return "Defensive"
    return "Avoid"


# ── Data Collector ──

class MarketDataCollector:
    def __init__(self, start_date, end_date, kr_stocks, kr_indexes, global_assets):
        self.start_date = start_date
        self.end_date = end_date
        self.kr_stocks = kr_stocks
        self.kr_indexes = kr_indexes
        self.global_assets = global_assets

    def fetch_fdr_kr_stock_prices(self):
        import FinanceDataReader as fdr

        frames = []
        for ticker, asset_name in self.kr_stocks.items():
            try:
                raw = fdr.DataReader(ticker, self.start_date, self.end_date)
                price_df = to_long_price_df(raw, ticker, asset_name, "KR_STOCK", "FinanceDataReader")
                if not price_df.empty:
                    frames.append(price_df)
                    print(f"[OK] FDR stock: {ticker}, rows={len(price_df)}")
            except Exception as e:
                print(f"[ERROR] FDR stock failed: {ticker} / {e}")
            time.sleep(0.2)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_fdr_kr_index_prices(self):
        import FinanceDataReader as fdr

        frames = []
        for code, name in self.kr_indexes.items():
            try:
                raw = fdr.DataReader(code, self.start_date, self.end_date)
                price_df = to_long_price_df(raw, code, name, "KR_INDEX", "FinanceDataReader")
                if not price_df.empty:
                    frames.append(price_df)
                    print(f"[OK] FDR index: {code}, rows={len(price_df)}")
            except Exception as e:
                print(f"[ERROR] FDR index failed: {code} / {e}")
            time.sleep(0.2)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_yfinance_global_prices(self):
        import yfinance as yf

        frames = []
        for ticker, (asset_name, market) in self.global_assets.items():
            try:
                raw = yf.download(
                    ticker, start=self.start_date, end=self.end_date,
                    auto_adjust=False, progress=False,
                )
                if raw is None or raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
                raw = raw[keep_cols].copy()
                price_df = to_long_price_df(raw, ticker, asset_name, market, "yfinance")
                if not price_df.empty:
                    frames.append(price_df)
                    print(f"[OK] yfinance: {ticker}, rows={len(price_df)}")
            except Exception as e:
                print(f"[ERROR] yfinance failed: {ticker} / {e}")
            time.sleep(0.2)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_naver_flow_one_stock(self, ticker, asset_name, max_pages=250):
        columns = [
            "Date", "AssetCode", "AssetName", "Market", "Source",
            "InvestorType", "NetBuyVolume",
        ]
        frames = []
        headers = {"User-Agent": "Mozilla/5.0"}

        for page in range(1, max_pages + 1):
            url = f"https://finance.naver.com/item/frgn.naver?code={ticker}&page={page}"
            try:
                res = requests.get(url, headers=headers, timeout=10)
                res.raise_for_status()
                tables = pd.read_html(StringIO(res.text), encoding="cp949")

                target = None
                date_col = foreign_col = institution_col = None

                for table in tables:
                    table = flatten_columns(table.copy()).dropna(how="all")
                    if table.empty:
                        continue
                    cols = list(table.columns)
                    date_candidate = find_col(cols, must_include=["날짜"])
                    foreign_candidate = find_col(cols, must_include=["외국인"], exclude=["보유", "비율", "소진"])
                    institution_candidate = find_col(cols, must_include=["기관"], exclude=["보유", "비율", "소진"])
                    if date_candidate and (foreign_candidate or institution_candidate):
                        target, date_col = table.copy(), date_candidate
                        foreign_col, institution_col = foreign_candidate, institution_candidate
                        break

                if target is None or target.empty:
                    continue

                target["Date"] = pd.to_datetime(target[date_col], errors="coerce")
                target = target.dropna(subset=["Date"])
                if target.empty:
                    continue

                oldest_on_page = target["Date"].min()
                target = target[
                    (target["Date"] >= pd.to_datetime(self.start_date))
                    & (target["Date"] <= pd.to_datetime(self.end_date))
                ]
                if target.empty:
                    if oldest_on_page < pd.to_datetime(self.start_date):
                        break
                    continue

                page_rows = []
                if foreign_col is not None:
                    temp = target[["Date", foreign_col]].copy()
                    temp["InvestorType"] = "Foreign"
                    temp["NetBuyVolume"] = temp[foreign_col].apply(clean_number)
                    page_rows.append(temp[["Date", "InvestorType", "NetBuyVolume"]])
                if institution_col is not None:
                    temp = target[["Date", institution_col]].copy()
                    temp["InvestorType"] = "Institution"
                    temp["NetBuyVolume"] = temp[institution_col].apply(clean_number)
                    page_rows.append(temp[["Date", "InvestorType", "NetBuyVolume"]])

                if page_rows:
                    page_df = pd.concat(page_rows, ignore_index=True)
                    page_df["AssetCode"] = ticker
                    page_df["AssetName"] = asset_name
                    page_df["Market"] = "KR_STOCK"
                    page_df["Source"] = "NaverFinance"
                    frames.append(page_df[columns])
            except Exception as e:
                print(f"[WARN] Naver flow page failed: {ticker}, page={page}, error={e}")
            time.sleep(0.15)

        if not frames:
            return pd.DataFrame(columns=columns)
        result = pd.concat(frames, ignore_index=True)
        result = result.dropna(subset=["Date", "NetBuyVolume"])
        result = result.drop_duplicates(subset=["Date", "AssetCode", "InvestorType"])
        result = result.sort_values(["Date", "AssetCode", "InvestorType"]).reset_index(drop=True)
        print(f"[OK] Naver flow: {ticker}, rows={len(result)}")
        return result

    def fetch_naver_flows(self, max_pages=250):
        frames = []
        for ticker, asset_name in self.kr_stocks.items():
            flow_df = self.fetch_naver_flow_one_stock(ticker, asset_name, max_pages)
            if not flow_df.empty:
                frames.append(flow_df)
        if not frames:
            return pd.DataFrame(columns=[
                "Date", "AssetCode", "AssetName", "Market", "Source",
                "InvestorType", "NetBuyVolume",
            ])
        return pd.concat(frames, ignore_index=True)

    def collect(self, max_pages=250):
        ensure_output_dir()
        print("[START] Data collection")

        kr_stock = self.fetch_fdr_kr_stock_prices()
        kr_index = self.fetch_fdr_kr_index_prices()
        global_p = self.fetch_yfinance_global_prices()
        flow_df = self.fetch_naver_flows(max_pages=max_pages)

        price_frames = [df for df in [kr_stock, kr_index, global_p] if df is not None and not df.empty]
        price_df = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()

        if not price_df.empty:
            price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
            price_df = price_df.dropna(subset=["Date"]).sort_values(["Date", "AssetCode"]).drop_duplicates().reset_index(drop=True)

        if not flow_df.empty:
            flow_df["Date"] = pd.to_datetime(flow_df["Date"], errors="coerce")
            flow_df = flow_df.dropna(subset=["Date"]).sort_values(["Date", "AssetCode", "InvestorType"]).drop_duplicates().reset_index(drop=True)

        ensure_output_dir()
        price_df.to_csv(PRICE_FILE, index=False, encoding="utf-8-sig")
        flow_df.to_csv(FLOW_FILE, index=False, encoding="utf-8-sig")
        print(f"[DONE] Data collection  price={price_df.shape}  flow={flow_df.shape}")
        return price_df, flow_df


# ── Feature Builder ──

class MarketFeatureBuilder:
    def add_price_features(self, group):
        group = group.sort_values("Date").copy()
        close = group["Close"]
        volume = group["Volume"]

        for n in [1, 5, 20, 60, 120, 240]:
            group[f"Return_{n}D"] = close.pct_change(n)

        for w in [30, 60, 120]:
            group[f"MA{w}"] = close.rolling(w).mean()

        group["MA30_Above_MA60"] = (group["MA30"] > group["MA60"]).astype(int)
        group["MA60_Above_MA120"] = (group["MA60"] > group["MA120"]).astype(int)
        group["MA_Alignment"] = ((group["MA30"] > group["MA60"]) & (group["MA60"] > group["MA120"])).astype(int)

        for w in [30, 60, 120]:
            group[f"Close_to_MA{w}"] = close / group[f"MA{w}"] - 1

        group["High_52W"] = close.rolling(240).max()
        group["Drawdown_52W"] = close / group["High_52W"] - 1
        group["Volatility_20D"] = group["Return_1D"].rolling(20).std()
        group["Volatility_60D"] = group["Return_1D"].rolling(60).std()
        group["RSI14"] = compute_rsi(close, window=14)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        group["MACD"] = ema12 - ema26
        group["MACD_Signal"] = group["MACD"].ewm(span=9, adjust=False).mean()
        group["MACD_Hist"] = group["MACD"] - group["MACD_Signal"]
        group["MACD_GoldenCross"] = (
            (group["MACD"] > group["MACD_Signal"]) & (group["MACD"].shift(1) <= group["MACD_Signal"].shift(1))
        ).astype(int)
        group["ROC60"] = close / close.shift(60) - 1
        group["Volume_MA20"] = volume.rolling(20).mean()
        group["Volume_Ratio_20D"] = volume / group["Volume_MA20"]
        return group

    def add_flow_features(self, group):
        group = group.sort_values("Date").copy()
        for col_base in ["Foreign", "Institution"]:
            group[f"{col_base}_5D"] = group[col_base].rolling(5).sum()
            group[f"{col_base}_20D"] = group[col_base].rolling(20).sum()
        group["TotalFlow"] = group["Foreign"] + group["Institution"]
        group["TotalFlow_5D"] = group["TotalFlow"].rolling(5).sum()
        group["TotalFlow_20D"] = group["TotalFlow"].rolling(20).sum()
        return group

    def build_relative_strength_features(self, price_df, feature_df):
        close_wide = price_df.pivot_table(
            index="Date", columns="AssetCode", values="Close", aggfunc="last"
        ).sort_index()

        pairs = [
            ("005930", "KS11", "RS_vs_KOSPI"), ("000660", "KS11", "RS_vs_KOSPI"),
            ("042700", "KS11", "RS_vs_KOSPI"), ("SOXX", "^IXIC", "RS_vs_NASDAQ"),
            ("SMH", "^IXIC", "RS_vs_NASDAQ"), ("SOXX", "^GSPC", "RS_vs_SP500"),
            ("SMH", "^GSPC", "RS_vs_SP500"),
        ]
        rows = []
        for asset, bench, rs_col in pairs:
            if asset not in close_wide.columns or bench not in close_wide.columns:
                continue
            rs = close_wide[asset] / close_wide[bench]
            temp = pd.DataFrame({
                "Date": close_wide.index, "AssetCode": asset,
                rs_col: rs.values,
                f"{rs_col}_Return_20D": rs.pct_change(20).values,
                f"{rs_col}_Return_60D": rs.pct_change(60).values,
            })
            rows.append(temp.reset_index(drop=True))

        if rows:
            rel_df = pd.concat(rows, ignore_index=True)
            rel_df = rel_df.groupby(["Date", "AssetCode"], as_index=False).first()
            feature_df = feature_df.merge(rel_df, on=["Date", "AssetCode"], how="left")
        return feature_df

    def build(self, price_df=None, flow_df=None):
        ensure_output_dir()
        if price_df is None:
            price_df = pd.read_csv(PRICE_FILE, dtype={"AssetCode": str})
        if flow_df is None:
            flow_df = pd.read_csv(FLOW_FILE, dtype={"AssetCode": str})

        price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
        flow_df["Date"] = pd.to_datetime(flow_df["Date"], errors="coerce")
        price_df = price_df.dropna(subset=["Date"]).sort_values(["AssetCode", "Date"]).reset_index(drop=True)
        flow_df = flow_df.dropna(subset=["Date"]).sort_values(["AssetCode", "InvestorType", "Date"]).reset_index(drop=True)

        if price_df.empty:
            raise ValueError("price_df is empty")

        # pandas 3.x: groupby().apply()가 그룹 키 컬럼을 제거하므로 보존 처리
        asset_codes = price_df[["AssetCode"]].copy()
        price_feat = price_df.groupby("AssetCode", group_keys=False).apply(self.add_price_features).reset_index(drop=True)
        if "AssetCode" not in price_feat.columns:
            price_feat["AssetCode"] = asset_codes["AssetCode"].values

        if flow_df.empty:
            flow_feat = pd.DataFrame()
        else:
            flow_wide = flow_df.pivot_table(
                index=["Date", "AssetCode", "AssetName", "Market", "Source"],
                columns="InvestorType", values="NetBuyVolume", aggfunc="sum",
            ).reset_index()
            flow_wide.columns.name = None
            for c in ["Foreign", "Institution"]:
                if c not in flow_wide.columns:
                    flow_wide[c] = 0.0
            flow_wide = flow_wide.sort_values(["AssetCode", "Date"]).reset_index(drop=True)
            flow_asset_codes = flow_wide[["AssetCode"]].copy()
            flow_feat = flow_wide.groupby("AssetCode", group_keys=False).apply(self.add_flow_features).reset_index(drop=True)
            flow_feat = flow_feat.rename(columns={"Source": "FlowSource"})
            if "AssetCode" not in flow_feat.columns:
                flow_feat["AssetCode"] = flow_asset_codes["AssetCode"].values

        flow_cols = [
            "Date", "AssetCode", "FlowSource", "Foreign", "Institution",
            "Foreign_5D", "Foreign_20D", "Institution_5D", "Institution_20D",
            "TotalFlow", "TotalFlow_5D", "TotalFlow_20D",
        ]
        if not flow_feat.empty:
            feature_df = price_feat.merge(flow_feat[flow_cols], on=["Date", "AssetCode"], how="left")
        else:
            feature_df = price_feat

        feature_df = self.build_relative_strength_features(price_df, feature_df)
        feature_df = feature_df.sort_values(["Date", "AssetCode"]).reset_index(drop=True)

        ensure_output_dir()
        feature_df.to_csv(FEATURE_FILE, index=False, encoding="utf-8-sig")
        print(f"[DONE] Feature building  shape={feature_df.shape}")
        return feature_df


# ── Rule-based Reporter ──

class MarketReporter:
    def judge_trend(self, row):
        evidence, bullish, bearish = [], 0, 0
        ma = safe_value(row, "MA_Alignment")
        c60 = safe_value(row, "Close_to_MA60")
        c120 = safe_value(row, "Close_to_MA120")
        r60 = safe_value(row, "Return_60D")
        r120 = safe_value(row, "Return_120D")
        dd = safe_value(row, "Drawdown_52W")

        if ma == 1:
            bullish += 1; evidence.append("MA30>MA60>MA120 정배열")
        elif ma == 0:
            bearish += 1; evidence.append("이동평균 정배열 미형성")
        if c60 is not None:
            if c60 > 0: bullish += 1; evidence.append(f"종가 MA60 대비 {fmt_pct(c60)} 상회")
            else: bearish += 1; evidence.append(f"종가 MA60 대비 {fmt_pct(c60)} 하회")
        if c120 is not None:
            if c120 > 0: bullish += 1; evidence.append(f"종가 MA120 대비 {fmt_pct(c120)} 상회")
            else: bearish += 1; evidence.append(f"종가 MA120 대비 {fmt_pct(c120)} 하회")
        if r60 is not None:
            if r60 > 0: bullish += 1; evidence.append(f"60일 수익률 {fmt_pct(r60)} 양수")
            else: bearish += 1; evidence.append(f"60일 수익률 {fmt_pct(r60)} 음수")
        if r120 is not None:
            if r120 > 0: bullish += 1; evidence.append(f"120일 수익률 {fmt_pct(r120)} 양수")
            else: bearish += 1; evidence.append(f"120일 수익률 {fmt_pct(r120)} 음수")
        if dd is not None and dd < -0.25:
            bearish += 1; evidence.append(f"52주 고점 대비 낙폭 {fmt_pct(dd)}")

        if bullish >= 5 and bearish == 0: view = "Bullish"
        elif bullish >= 3 and bearish <= 1: view = "MildBullish"
        elif bearish >= 4: view = "Bearish"
        elif bearish >= 2: view = "MildBearish"
        else: view = "Neutral"
        return {"view": view, "bullish_evidence_count": bullish, "bearish_evidence_count": bearish, "evidence": evidence, "limitations": []}

    def judge_momentum(self, row):
        evidence, risk_flags, pos, neg = [], [], 0, 0
        rsi = safe_value(row, "RSI14")
        mh = safe_value(row, "MACD_Hist")
        ms = safe_value(row, "MACD_Signal")
        macd = safe_value(row, "MACD")
        roc = safe_value(row, "ROC60")

        if rsi is not None:
            if 45 <= rsi <= 65: pos += 1; evidence.append(f"RSI14 {rsi:.1f} 안정 구간")
            elif 65 < rsi <= 75: pos += 1; risk_flags.append(f"RSI {rsi:.1f} 과열 접근")
            elif rsi > 75: neg += 1; risk_flags.append(f"RSI {rsi:.1f} 과열")
            elif rsi < 30: neg += 1; risk_flags.append(f"RSI {rsi:.1f} 과매도")
        if mh is not None:
            if mh > 0: pos += 1; evidence.append("MACD histogram 양수")
            else: neg += 1; evidence.append("MACD histogram 음수")
        if macd is not None and ms is not None:
            if macd > ms: pos += 1; evidence.append("MACD > signal")
            else: neg += 1; evidence.append("MACD < signal")
        if roc is not None:
            if roc > 0: pos += 1; evidence.append(f"ROC60 {fmt_pct(roc)} 양수")
            else: neg += 1; evidence.append(f"ROC60 {fmt_pct(roc)} 음수")

        if pos >= 3 and neg <= 1: view = "Positive"
        elif neg >= 3: view = "Negative"
        else: view = "Mixed"
        return {"view": view, "positive_evidence_count": pos, "negative_evidence_count": neg, "evidence": evidence, "risk_flags": risk_flags, "limitations": []}

    def judge_flow(self, row):
        evidence, limitations = [], []
        f20 = safe_value(row, "Foreign_20D")
        i20 = safe_value(row, "Institution_20D")
        t20 = safe_value(row, "TotalFlow_20D")

        if f20 is None and i20 is None and t20 is None:
            return {"view": "NotAvailable", "foreign_20d": None, "institution_20d": None, "total_flow_20d": None,
                    "evidence": [], "risk_flags": [], "limitations": ["수급 데이터 없음"]}

        if f20 is not None:
            evidence.append(f"외국인 20일 순매수 {fmt_num(f20)}주 {'양수' if f20 > 0 else '음수'}")
        if i20 is not None:
            evidence.append(f"기관 20일 순매수 {fmt_num(i20)}주 {'양수' if i20 > 0 else '음수'}")
        if t20 is not None:
            evidence.append(f"합산 20일 수급 {fmt_num(t20)}주 {'양수' if t20 > 0 else '음수'}")

        fp = f20 is not None and f20 > 0
        fn = f20 is not None and f20 < 0
        ip = i20 is not None and i20 > 0
        tp = t20 is not None and t20 > 0
        tn = t20 is not None and t20 < 0

        if fp and ip and tp: view = "StrongAccumulation"
        elif fp and tp: view = "Accumulation"
        elif fn and ip and tp: view = "MixedAccumulation"
        elif fn and (i20 is not None and i20 < 0) and tn: view = "StrongDistribution"
        elif fn and tn: view = "Distribution"
        elif fp and (i20 is not None and i20 < 0) and tn: view = "MixedDistribution"
        else: view = "Mixed"

        return {"view": view, "foreign_20d": f20, "institution_20d": i20, "total_flow_20d": t20,
                "evidence": evidence, "risk_flags": [], "limitations": limitations}

    def judge_relative_strength(self, row):
        evidence, vals = [], []
        for bench, label in [("RS_vs_KOSPI_Return_20D", "KOSPI"), ("RS_vs_NASDAQ_Return_20D", "NASDAQ"), ("RS_vs_SP500_Return_20D", "S&P500")]:
            v = safe_value(row, bench)
            if v is not None:
                vals.append(v)
                evidence.append(f"{label} 대비 20일 상대강도 {fmt_pct(v)} {'개선' if v > 0 else '약화'}")

        if not vals:
            return {"view": "NotAvailable", "average_relative_strength_20d": None,
                    "evidence": [], "risk_flags": [], "limitations": ["상대강도 benchmark 없음"]}

        avg = float(np.mean(vals))
        if avg > 0.03: view = "Outperforming"
        elif avg < -0.03: view = "Underperforming"
        else: view = "Neutral"
        return {"view": view, "average_relative_strength_20d": avg, "evidence": evidence, "risk_flags": [], "limitations": []}

    def judge_risk(self, row):
        evidence, risk_flags, hi, mod = [], [], 0, 0
        dd = safe_value(row, "Drawdown_52W")
        v20 = safe_value(row, "Volatility_20D")
        rsi = safe_value(row, "RSI14")

        if dd is not None:
            if dd < -0.30: hi += 1; risk_flags.append(f"52주 낙폭 {fmt_pct(dd)}")
            elif dd < -0.15: mod += 1; risk_flags.append(f"52주 낙폭 {fmt_pct(dd)} 조정")
            else: evidence.append(f"52주 낙폭 {fmt_pct(dd)} 제한적")
        if v20 is not None:
            if v20 > 0.035: hi += 1; risk_flags.append(f"20일 변동성 {fmt_pct(v20)} 높음")
            elif v20 > 0.02: mod += 1; risk_flags.append(f"20일 변동성 {fmt_pct(v20)} 중간")
            else: evidence.append(f"20일 변동성 {fmt_pct(v20)} 낮음")
        if rsi is not None:
            if rsi > 75: hi += 1; risk_flags.append(f"RSI {rsi:.1f} 과열")
            elif rsi > 70: mod += 1; risk_flags.append(f"RSI {rsi:.1f} 과열 접근")
            elif rsi < 30: mod += 1; risk_flags.append(f"RSI {rsi:.1f} 과매도")

        if hi >= 2: view = "High"
        elif hi == 1 or mod >= 2: view = "Moderate"
        else: view = "Low"
        return {"view": view, "high_risk_count": hi, "moderate_risk_count": mod,
                "evidence": evidence, "risk_flags": risk_flags, "limitations": []}

    def judge_timing_readiness(self, row, trend, momentum, flow, relative_strength, risk):
        score = 50
        sup, cau = [], []

        ma = safe_value(row, "MA_Alignment")
        macd = safe_value(row, "MACD")
        ms = safe_value(row, "MACD_Signal")
        mh = safe_value(row, "MACD_Hist")
        rsi = safe_value(row, "RSI14")
        roc = safe_value(row, "ROC60")
        vr = safe_value(row, "Volume_Ratio_20D")
        dd = safe_value(row, "Drawdown_52W")
        v20 = safe_value(row, "Volatility_20D")
        f20 = safe_value(row, "Foreign_20D")
        i20 = safe_value(row, "Institution_20D")
        t20 = safe_value(row, "TotalFlow_20D")

        rs_vals = [safe_value(row, c) for c in ["RS_vs_KOSPI_Return_20D", "RS_vs_NASDAQ_Return_20D", "RS_vs_SP500_Return_20D"]]
        rs_vals = [x for x in rs_vals if x is not None]

        if ma == 1: score += 12; sup.append("이동평균 정배열")
        else: score -= 5; cau.append("이동평균 정배열 부재")

        if macd is not None and ms is not None:
            if macd > ms: score += 8; sup.append("MACD > signal")
            else: score -= 6; cau.append("MACD < signal")
        if mh is not None:
            if mh > 0: score += 6; sup.append("MACD histogram 양수")
            else: score -= 5; cau.append("MACD histogram 음수")
        if rsi is not None:
            if 45 <= rsi <= 70: score += 10; sup.append("RSI 안정 상승 구간")
            elif 70 < rsi <= 80: score += 2; cau.append("RSI 과열 접근")
            elif rsi > 80: score -= 15; cau.append("RSI 과열")
            elif rsi < 30: score -= 10; cau.append("RSI 과매도")
        if roc is not None:
            if roc > 0: score += 8; sup.append("ROC60 양수")
            else: score -= 8; cau.append("ROC60 음수")
        if t20 is not None:
            if t20 > 0: score += 8; sup.append("합산 수급 양수")
            else: score -= 8; cau.append("합산 수급 음수")
        else:
            cau.append("수급 데이터 부재")
        if f20 is not None:
            if f20 > 0: score += 4; sup.append("외국인 순매수")
            else: score -= 4; cau.append("외국인 순매도")
        if i20 is not None:
            if i20 > 0: score += 4; sup.append("기관 순매수")
            else: score -= 4; cau.append("기관 순매도")
        if rs_vals:
            avg_rs = float(np.mean(rs_vals))
            if avg_rs > 0.03: score += 10; sup.append("상대강도 개선")
            elif avg_rs < -0.03: score -= 10; cau.append("상대강도 약화")
        else:
            cau.append("상대강도 benchmark 부재")
        if vr is not None:
            if 1.0 <= vr <= 2.5: score += 4; sup.append("거래량 평균 이상")
            elif vr > 3.0: score -= 3; cau.append("거래량 급증")
        if dd is not None:
            if dd < -0.30: score -= 12; cau.append("52주 낙폭 과대")
            elif dd < -0.15: score -= 5; cau.append("52주 조정 구간")
        if v20 is not None:
            if v20 > 0.035: score -= 10; cau.append("변동성 높음")
            elif v20 > 0.02: score -= 4; cau.append("변동성 중간")
            else: score += 3; sup.append("변동성 낮음")
        if risk["view"] == "High": score -= 15; cau.append("risk High")
        elif risk["view"] == "Moderate": score -= 5; cau.append("risk Moderate")

        score = int(max(0, min(100, round(score))))
        return {"score": score, "zone": timing_zone(score), "supporting_signals": sup[:8], "caution_signals": cau[:8]}

    def synthesize_stance(self, trend, momentum, flow, relative_strength, risk):
        pos, neg, mix = [], [], []
        if trend["view"] in ["Bullish", "MildBullish"]: pos.append("trend")
        elif trend["view"] in ["Bearish", "MildBearish"]: neg.append("trend")
        else: mix.append("trend")
        if momentum["view"] == "Positive": pos.append("momentum")
        elif momentum["view"] == "Negative": neg.append("momentum")
        else: mix.append("momentum")
        if flow["view"] in ["StrongAccumulation", "Accumulation", "MixedAccumulation"]: pos.append("flow")
        elif flow["view"] in ["StrongDistribution", "Distribution", "MixedDistribution"]: neg.append("flow")
        elif flow["view"] != "NotAvailable": mix.append("flow")
        if relative_strength["view"] == "Outperforming": pos.append("relative_strength")
        elif relative_strength["view"] == "Underperforming": neg.append("relative_strength")
        elif relative_strength["view"] != "NotAvailable": mix.append("relative_strength")
        if risk["view"] == "High": neg.append("risk")
        elif risk["view"] == "Moderate": mix.append("risk")

        pn, nn = len(pos), len(neg)
        if pn >= 3 and nn <= 1 and risk["view"] != "High": stance = "Positive"
        elif nn >= 3 or (nn >= 2 and risk["view"] == "High"): stance = "Cautious"
        else: stance = "Neutral"

        total = pn + nn + len(mix)
        dom = max(pn, nn)
        if total == 0: conf = "Low"
        elif dom / total >= 0.70: conf = "High"
        elif dom / total >= 0.45: conf = "Medium"
        else: conf = "Low"
        return stance, conf, {"positive_blocks": pos, "negative_blocks": neg, "mixed_blocks": mix}

    def generate_report_for_asset(self, row):
        trend = self.judge_trend(row)
        momentum = self.judge_momentum(row)
        flow = self.judge_flow(row)
        rs = self.judge_relative_strength(row)
        risk = self.judge_risk(row)
        timing = self.judge_timing_readiness(row, trend, momentum, flow, rs, risk)
        stance, conf, basis = self.synthesize_stance(trend, momentum, flow, rs, risk)

        evidence, limitations, risk_flags = [], [], []
        for b in [trend, momentum, flow, rs, risk]:
            evidence.extend(b.get("evidence", []))
            limitations.extend(b.get("limitations", []))
            risk_flags.extend(b.get("risk_flags", []))

        name = row["AssetName"]
        rationale = (
            f"{name}: {stance} (신뢰도 {conf}). "
            f"우호 블록={basis['positive_blocks']}, 경계={basis['negative_blocks']}. "
            f"Timing {timing['score']}점({timing['zone']})."
        )

        as_of_date = str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"])
        return {
            "agent": "Market Agent",
            "as_of_date": as_of_date,
            "data_sources": [
                {"kind": "price", "provider": safe_value(row, "Source"), "as_of_date": as_of_date},
                {"kind": "flow", "provider": safe_value(row, "FlowSource"),
                 "as_of_date": as_of_date if flow["view"] != "NotAvailable" else None},
            ],
            "target": {"asset_code": row["AssetCode"], "asset_name": row["AssetName"], "market": row["Market"]},
            "stance": stance, "confidence": conf, "decision_basis": basis,
            "technical_view": trend, "momentum_view": momentum, "flow_view": flow,
            "relative_strength_view": rs, "risk_view": risk, "timing_readiness": timing,
            "evidence": evidence[:8], "limitations": limitations[:6], "risk_flags": risk_flags[:6],
            "rationale": rationale,
        }

    def build_reports(self, feature_df):
        latest = feature_df.sort_values(["AssetCode", "Date"]).groupby("AssetCode").tail(1).reset_index(drop=True)
        reports = [self.generate_report_for_asset(row) for _, row in latest.iterrows()]

        table_rows = []
        for r in reports:
            table_rows.append({
                "Date": r["as_of_date"], "AssetCode": r["target"]["asset_code"],
                "AssetName": r["target"]["asset_name"], "Stance": r["stance"],
                "Confidence": r["confidence"], "TimingScore": r["timing_readiness"]["score"],
                "TimingZone": r["timing_readiness"]["zone"],
                "Trend": r["technical_view"]["view"], "Momentum": r["momentum_view"]["view"],
                "Flow": r["flow_view"]["view"], "Risk": r["risk_view"]["view"],
            })
        table_df = pd.DataFrame(table_rows)

        output = {
            "agent": "Market Agent",
            "generated_at": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            "n_assets": len(reports), "reports": reports,
        }
        ensure_output_dir()
        with open(REPORT_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        table_df.to_csv(REPORT_TABLE_FILE, index=False, encoding="utf-8-sig")
        print(f"[DONE] Rule-based reporting  assets={len(reports)}")
        return output, table_df


# ── GPT Market Interpreter ──

class GPTMarketInterpreter:
    def __init__(self, api_key=None, model="gpt-4.1-mini"):
        from openai import OpenAI
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate_llm_report(self, report_json):
        compact = []
        for r in report_json["reports"]:
            compact.append({
                "asset_code": r["target"]["asset_code"], "asset_name": r["target"]["asset_name"],
                "as_of_date": r.get("as_of_date"), "data_sources": r.get("data_sources", []),
                "limitations": r.get("limitations", []),
                "stance": r["stance"], "confidence": r["confidence"],
                "timing": r["timing_readiness"],
                "trend": r["technical_view"]["view"], "momentum": r["momentum_view"]["view"],
                "flow": r["flow_view"]["view"], "risk": r["risk_view"]["view"],
                "evidence": r["evidence"], "risk_flags": r["risk_flags"],
            })

        prompt = (
            "너는 Market Agent다. 아래 rule-based 분석을 참고하되 독립적으로 재판단하라.\n"
            "JSON만 출력. 최종 투자 추천 금지.\n\n"
            + json.dumps(compact, ensure_ascii=False, indent=2)
        )

        response = self.client.responses.create(model=self.model, input=prompt, temperature=0.2)
        text = response.output_text.strip()
        try:
            parsed = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {"raw": text}

        output = {"agent": "Market Agent", "model": self.model, "llm_judgment": parsed}
        ensure_output_dir()
        with open(LLM_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return output


# ── MarketAgent (메인 클래스) ──

class MarketAgent:
    def __init__(self, start_date=START_DATE, end_date=END_DATE,
                 kr_stocks=KR_STOCKS, kr_indexes=KR_INDEXES, global_assets=GLOBAL_ASSETS,
                 use_gpt=False, openai_api_key=None, gpt_model="gpt-4.1-mini"):
        self.collector = MarketDataCollector(start_date, end_date, kr_stocks, kr_indexes, global_assets)
        self.feature_builder = MarketFeatureBuilder()
        self.reporter = MarketReporter()
        self.use_gpt = use_gpt
        self.gpt_interpreter = None
        if use_gpt:
            self.gpt_interpreter = GPTMarketInterpreter(api_key=openai_api_key, model=gpt_model)

    def collect(self, max_pages=250):
        return self.collector.collect(max_pages=max_pages)

    def build_features(self, price_df=None, flow_df=None):
        return self.feature_builder.build(price_df=price_df, flow_df=flow_df)

    def generate_report(self, feature_df=None):
        if feature_df is None:
            feature_df = pd.read_csv(FEATURE_FILE, dtype={"AssetCode": str})
            feature_df["Date"] = pd.to_datetime(feature_df["Date"], errors="coerce")
        report_json, table_df = self.reporter.build_reports(feature_df)
        llm_report = None
        if self.use_gpt:
            llm_report = self.gpt_interpreter.generate_llm_report(report_json)
        return {"json": report_json, "table": table_df, "llm_report": llm_report}

    def run(self, max_pages=250, collect_data=True):
        ensure_output_dir()
        if collect_data:
            price_df, flow_df = self.collect(max_pages=max_pages)
        else:
            price_df = pd.read_csv(PRICE_FILE, dtype={"AssetCode": str})
            flow_df = pd.read_csv(FLOW_FILE, dtype={"AssetCode": str})
        feature_df = self.build_features(price_df=price_df, flow_df=flow_df)
        report = self.generate_report(feature_df=feature_df)
        return {"price_df": price_df, "flow_df": flow_df, "feature_df": feature_df, "report": report}


if __name__ == "__main__":
    agent = MarketAgent(use_gpt=False)
    result = agent.run(collect_data=True, max_pages=50)
    print(json.dumps(result["report"]["json"], ensure_ascii=False, indent=2)[:3000])
