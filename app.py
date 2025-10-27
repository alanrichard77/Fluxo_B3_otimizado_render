import os, io, json, math, base64, logging, time
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from flask import Flask, render_template, redirect, url_for

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

import requests
from bs4 import BeautifulSoup

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("fluxo-b3")

DATA_DIR  = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

B3_DELAY_DAYS = int(os.getenv("B3_DELAY_DAYS", "2"))
TZ            = os.getenv("TZ", "America/Sao_Paulo")
FLUXO_CSV_URL = os.getenv("FLUXO_CSV_URL", "").strip()

CATEGORIAS = ["Estrangeiro", "Institucional", "Pessoa Física", "Inst. Financeira", "Outros"]

# =============================================================================
# CACHE
# =============================================================================
def _cache_path(key: str) -> str:
    safe = key.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe}.json")

def cache_get(key: str):
    fp = _cache_path(key)
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def cache_set(key: str, data):
    if isinstance(data, dict):
        data = {str(k): v for k, v in data.items()}
    with open(_cache_path(key), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def cache_clear(prefix: Optional[str] = None):
    if not os.path.isdir(CACHE_DIR):
        return
    for fname in os.listdir(CACHE_DIR):
        if prefix and not fname.startswith(prefix):
            continue
        try:
            os.remove(os.path.join(CACHE_DIR, fname))
        except Exception:
            pass

def _df_to_cache_payload(df: pd.DataFrame) -> dict:
    payload = df.copy()
    if "data" in payload.columns:
        payload["data"] = pd.to_datetime(payload["data"]).dt.strftime("%Y-%m-%d")
    dct = payload.to_dict(orient="list")
    return {str(k): v for k, v in dct.items()}

# =============================================================================
# NORMALIZAÇÃO
# =============================================================================
def _rename_fluxo_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("data", "date"):
            rename_map[c] = "data"
        elif "estrange" in cl:
            rename_map[c] = "Estrangeiro"
        elif "institucional" in cl and "inst." not in cl:
            rename_map[c] = "Institucional"
        elif "pessoa" in cl:
            rename_map[c] = "Pessoa Física"
        elif ("inst" in cl and "financ" in cl) or "financeira" in cl:
            rename_map[c] = "Inst. Financeira"
        elif "outros" in cl:
            rename_map[c] = "Outros"
    return df.rename(columns=rename_map)

def _normalize_fluxo_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_fluxo_cols(df)

    cols = ["data"] + [c for c in CATEGORIAS if c in df.columns]
    df = df[cols].copy()

    if "data" not in df.columns or df.empty:
        return pd.DataFrame(columns=["data"] + CATEGORIAS)

    # datas e números
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    for c in CATEGORIAS:
        if c in df.columns:
            df[c] = pd.to_numeric(str(df[c].dtype).startswith("float") and df[c] or
                                  df[c].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                                  errors="coerce")

    # heurística de escala (se vier em milhões → /1000)
    for c in CATEGORIAS:
        if c in df.columns and df[c].notna().any():
            med = df[c].abs().median()
            if med is not None and med > 20:   # ~R$ 20 mi
                df[c] = df[c] / 1000.0

    df = df.dropna(subset=["data"]).sort_values("data").reset_index(drop=True)
    return df

def _detect_is_accumulated(df: pd.DataFrame) -> bool:
    """
    Detecta se a série por player já está acumulada.
    Heurística: diferença média em módulo << valor médio absoluto,
    e série quase sempre monótona (poucas inversões de sinal no delta).
    """
    inv = 0
    total = 0
    for c in CATEGORIAS:
        if c in df.columns:
            s = df[c].dropna()
            if len(s) >= 3:
                d = s.diff().dropna()
                inv += (np.sign(d) != 0).sum()
                total += len(d)
    if total == 0:
        return False
    ratio = inv / total
    # se a maioria dos dias tem delta pequeno e mantém o mesmo sinal, tende a acumulado
    return ratio < 0.25

# =============================================================================
# FONTES — Fluxo
# =============================================================================
DDM_URL = "https://www.dadosdemercado.com.br/fluxo"
CF_URL  = "https://carteirafundos.com.br/fluxo"
BHM_URL = "https://www.bhmoptions.com.br/participantes"

def fetch_fluxo_ddm() -> Optional[pd.DataFrame]:
    try:
        logger.info("Coletando fluxo: Dados de Mercado…")
        html = requests.get(DDM_URL, timeout=25, headers={"User-Agent":"Mozilla/5.0"}).text
        tables = pd.read_html(html, thousands=".", decimal=",")
        candidates = [t for t in tables if any(str(c).strip().lower() in ("data","date") for c in t.columns)]
        df = candidates[0] if candidates else None
        if df is None or df.empty:  # fallback: primeira tabela
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table")
            if table:
                df = pd.read_html(str(table), thousands=".", decimal=",")[0]
        if df is None or df.empty:
            return None
        return _normalize_fluxo_df(df)
    except Exception as e:
        logger.warning(f"Falha DDM: {e}")
        return None

def fetch_fluxo_carteirafundos() -> Optional[pd.DataFrame]:
    try:
        logger.info("Coletando fluxo: CarteiraFundos…")
        html = requests.get(CF_URL, timeout=25, headers={"User-Agent":"Mozilla/5.0"}).text
        tables = pd.read_html(html, thousands=".", decimal=",")
        # nesta página costuma haver uma tabela grande com Data e colunas por player
        tables.sort(key=lambda d: d.shape[0], reverse=True)
        for t in tables:
            if any(str(c).strip().lower() in ("data","date") for c in t.columns):
                df = _normalize_fluxo_df(t)
                if not df.empty:
                    return df
        return None
    except Exception as e:
        logger.warning(f"Falha CarteiraFundos: {e}")
        return None

def fetch_fluxo_bhmoptions() -> Optional[pd.DataFrame]:
    try:
        logger.info("Coletando fluxo: BHM Options…")
        html = requests.get(BHM_URL, timeout=25, headers={"User-Agent":"Mozilla/5.0"}).text
        tables = pd.read_html(html, thousands=".", decimal=",")
        tables.sort(key=lambda d: d.shape[0], reverse=True)
        for t in tables:
            if any(str(c).strip().lower() in ("data","date") for c in t.columns):
                df = _normalize_fluxo_df(t)
                if not df.empty:
                    return df
        return None
    except Exception as e:
        logger.warning(f"Falha BHM Options: {e}")
        return None

def fetch_fluxo_csv() -> Optional[pd.DataFrame]:
    if not FLUXO_CSV_URL:
        return None
    try:
        logger.info(f"Coletando CSV externo: {FLUXO_CSV_URL}")
        df = pd.read_csv(FLUXO_CSV_URL)
        df = _normalize_fluxo_df(df)
        return df
    except Exception as e:
        logger.warning(f"Falha CSV externo: {e}")
        return None

def fetch_fluxo_synthetic(start: date, end: date) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D")
    np.random.seed(42)
    base = pd.DataFrame({"data": idx})
    base["Estrangeiro"]      = np.random.normal(0.06, 0.28, len(idx))
    base["Institucional"]    = np.random.normal(-0.03, 0.16, len(idx))
    base["Pessoa Física"]    = np.random.normal(0.01, 0.05, len(idx))
    base["Inst. Financeira"] = np.random.normal(0.006, 0.04, len(idx))
    base["Outros"]           = np.random.normal(0.00, 0.03, len(idx))
    return base

# =============================================================================
# IBOVESPA – yfinance + Yahoo Chart API
# =============================================================================
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in t if x is not None]).strip() for t in df.columns.to_list()]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def fetch_ibov_history_yfinance(start: date, end: date) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        data = yf.download("^BVSP", start=start, end=end + timedelta(days=1),
                           progress=False, auto_adjust=False, threads=False)
        if data is None or data.empty:
            return None
        data = _flatten_columns(data)
        close_col = None
        for c in data.columns:
            if c.lower().endswith("close"):
                if c.lower() == "close":
                    close_col = c; break
                close_col = c
        if close_col is None: return None
        df = data.reset_index().rename(columns={"Date":"data", close_col:"Ibovespa"})[["data","Ibovespa"]]
        df["data"] = pd.to_datetime(df["data"]); df["Ibovespa"] = pd.to_numeric(df["Ibovespa"], errors="coerce")
        df = df.dropna().sort_values("data").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Falha yfinance: {e}")
        return None

def fetch_ibov_history_yahoo_chart(years: int = 2) -> Optional[pd.DataFrame]:
    try:
        end = int(time.time()); start = end - 60*60*24*365*years
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP?period1={start}&period2={end}&interval=1d"
        j = requests.get(url, timeout=25).json()
        res = j["chart"]["result"][0]
        ts  = res["timestamp"]; closes = res["indicators"]["quote"][0]["close"]
        df = pd.DataFrame({"data": pd.to_datetime(ts, unit="s"), "Ibovespa": closes})
        df = df.dropna().sort_values("data").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Falha Yahoo Chart: {e}")
        return None

# =============================================================================
# PIPELINE (com cache) — retorna (df, fonte_primaria)
# =============================================================================
def get_last_possible_date() -> date:
    return (pd.Timestamp.today().normalize() - BDay(B3_DELAY_DAYS)).date()

def load_fluxo_raw() -> Tuple[pd.DataFrame, str]:
    cached = cache_get("fluxo_raw")
    cached_src = cache_get("fluxo_raw_src")
    if cached is not None:
        df = pd.DataFrame(cached); df["data"] = pd.to_datetime(df["data"])
        return df, (cached_src or "cache")

    # cadeia de fontes
    for src_name, fn in [
        ("Dados de Mercado",    fetch_fluxo_ddm),
        ("CarteiraFundos",      fetch_fluxo_carteirafundos),
        ("BHM Options",         fetch_fluxo_bhmoptions),
        ("CSV Externo",         fetch_fluxo_csv),
    ]:
        df = fn()
        if df is not None and not df.empty:
            cache_set("fluxo_raw", _df_to_cache_payload(df))
            cache_set("fluxo_raw_src", {"src": src_name})
            return df, src_name

    # fallback sintético
    start = date.today().replace(month=1, day=1) - timedelta(days=180)
    df = fetch_fluxo_synthetic(start, date.today())
    cache_set("fluxo_raw", _df_to_cache_payload(df))
    cache_set("fluxo_raw_src", {"src": "sintético"})
    return df, "sintético"

def load_ibov_history(start: date, end: date) -> pd.DataFrame:
    key = f"ibov_hist_{start}_{end}"
    cached = cache_get(key)
    if cached is not None:
        df = pd.DataFrame(cached); df["data"] = pd.to_datetime(df["data"]); return df

    df = fetch_ibov_history_yfinance(start, end)
    if df is None:
        df = fetch_ibov_history_yahoo_chart(2)
        if df is not None:
            df = df[(df["data"].dt.date >= start) & (df["data"].dt.date <= end)].
