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

B3_DELAY_DAYS = int(os.getenv("B3_DELAY_DAYS", "2"))   # atraso de divulgação (dias úteis)
TZ             = os.getenv("TZ", "America/Sao_Paulo")
FLUXO_CSV_URL  = os.getenv("FLUXO_CSV_URL", "").strip() # fallback opcional

CATEGORIAS = ["Estrangeiro", "Institucional", "Pessoa Física", "Inst. Financeira", "Outros"]

# =============================================================================
# CACHE DISCO
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

# =============================================================================
# FONTE — Dados de Mercado (scraping)
# =============================================================================
DDM_URL = "https://www.dadosdemercado.com.br/fluxo"

def _normalize_fluxo_df(df: pd.DataFrame) -> pd.DataFrame:
    # Mapear nomes de colunas (variam na página)
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "data":
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
    df = df.rename(columns=rename_map)

    cols = ["data"] + [c for c in CATEGORIAS if c in df.columns]
    df = df[cols].copy()

    if "data" not in df.columns or df.empty:
        return pd.DataFrame(columns=["data"] + CATEGORIAS)

    # Converte data (geralmente DD/MM/YYYY)
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")

    # Números: a página já usa "." como milhar e "," decimal; read_html com decimal="," e thousands="."
    # mas garantimos numeric:
    for c in CATEGORIAS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Heurística de escala (muitas vezes vem em milhões); se mediana |x|>20, assumimos milhões e dividimos por 1000 → bilhões
    for c in CATEGORIAS:
        if c in df.columns and df[c].notna().any():
            med = df[c].abs().median()
            if med is not None and med > 20:   # ~R$ 20 mi
                df[c] = df[c] / 1000.0

    df = df.dropna(subset=["data"]).sort_values("data").reset_index(drop=True)
    return df

def fetch_fluxo_ddm() -> Optional[pd.DataFrame]:
    """
    Tenta capturar a tabela principal do Dados de Mercado (/fluxo).
    Retorna DIÁRIO por player (em R$ bilhões).
    """
    try:
        logger.info("Coletando fluxo em Dados de Mercado…")
        html = requests.get(DDM_URL, timeout=25).text

        # 1) Primeiro tenta via read_html (mais robusto)
        tables = pd.read_html(html, thousands=".", decimal=",")
        candidates = [t for t in tables if any(str(c).strip().lower() == "data" for c in t.columns)]
        df = None
        if candidates:
            candidates.sort(key=lambda d: d.shape[0], reverse=True)
            df = candidates[0]
        else:
            # 2) Fallback: primeira tabela do HTML
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table")
            if table:
                df = pd.read_html(str(table), thousands=".", decimal=",")[0]

        if df is None or df.empty:
            return None

        return _normalize_fluxo_df(df)

    except Exception as e:
        logger.warning(f"Falha ao coletar Dados de Mercado: {e}")
        return None

# =============================================================================
# FONTE — CSV externo opcional (caso queira usar a sua planilha)
# =============================================================================
def fetch_fluxo_csv() -> Optional[pd.DataFrame]:
    if not FLUXO_CSV_URL:
        return None
    try:
        logger.info(f"Coletando CSV externo: {FLUXO_CSV_URL}")
        df = pd.read_csv(FLUXO_CSV_URL)
        rename = {
            "data": "data",
            "Estrangeiro": "Estrangeiro",
            "Institucional": "Institucional",
            "Pessoa Física": "Pessoa Física",
            "Inst. Financeira": "Inst. Financeira",
            "Outros": "Outros",
        }
        df = df.rename(columns=rename)
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
        for c in CATEGORIAS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["data"]).sort_values("data").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Falha ao coletar CSV externo: {e}")
        return None

# =============================================================================
# Fallback sintético (apenas para não quebrar em dev)
# =============================================================================
def fetch_fluxo_synthetic(start: date, end: date) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D")
    np.random.seed(42)
    base = pd.DataFrame({"data": idx})
    base["Estrangeiro"] = np.random.normal(0.06, 0.28, len(idx))
    base["Institucional"] = np.random.normal(-0.03, 0.16, len(idx))
    base["Pessoa Física"] = np.random.normal(0.01, 0.05, len(idx))
    base["Inst. Financeira"] = np.random.normal(0.006, 0.04, len(idx))
    base["Outros"] = np.random.normal(0.00, 0.03, len(idx))
    return base

# =============================================================================
# IBOVESPA: yfinance (principal) + Yahoo Chart API (fallback)
# =============================================================================
def fetch_ibov_history_yfinance(start: date, end: date) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        data = yf.download("^BVSP", start=start, end=end + timedelta(days=1), progress=False, auto_adjust=False, threads=False)
        if data is None or data.empty:
            return None
        df = data.reset_index().rename(columns={"Date": "data", "Close": "Ibovespa"})[["data", "Ibovespa"]]
        df["data"] = pd.to_datetime(df["data"])
        df = df.dropna().sort_values("data").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Falha yfinance: {e}")
        return None

def fetch_ibov_history_yahoo_chart(years: int = 2) -> Optional[pd.DataFrame]:
    try:
        end = int(time.time())
        start = end - 60 * 60 * 24 * 365 * years
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP?period1={start}&period2={end}&interval=1d"
        r = requests.get(url, timeout=25)
        j = r.json()
        res = j["chart"]["result"][0]
        ts = res["timestamp"]
        closes = res["indicators"]["quote"][0]["close"]
        df = pd.DataFrame({"data": pd.to_datetime(ts, unit="s"), "Ibovespa": closes})
        df = df.dropna().sort_values("data").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Falha Yahoo Chart API: {e}")
        return None

# =============================================================================
# PIPELINE DE DADOS (com cache)
# =============================================================================
def get_last_possible_date() -> date:
    return (pd.Timestamp.today().normalize() - BDay(B3_DELAY_DAYS)).date()

def load_fluxo_raw() -> pd.DataFrame:
    cached = cache_get("fluxo_raw")
    if cached is not None:
        df = pd.DataFrame(cached)
        df["data"] = pd.to_datetime(df["data"])
        return df

    df = fetch_fluxo_ddm()
    if df is None:
        df = fetch_fluxo_csv()
    if df is None:
        start = date.today().replace(month=1, day=1) - timedelta(days=180)
        df = fetch_fluxo_synthetic(start, date.today())

    cache_set("fluxo_raw", df.assign(data=df["data"].dt.strftime("%Y-%m-%d")).to_dict(orient="list"))
    return df

def load_ibov_history(start: date, end: date) -> pd.DataFrame:
    key = f"ibov_hist_{start}_{end}"
    cached = cache_get(key)
    if cached is not None:
        df = pd.DataFrame(cached); df["data"] = pd.to_datetime(df["data"]); return df

    df = fetch_ibov_history_yfinance(start, end)
    if df is None:
        df = fetch_ibov_history_yahoo_chart(2)
        if df is not None:
            df = df[(df["data"].dt.date >= start) & (df["data"].dt.date <= end)].copy()

    if df is None:
        # fallback sintético
        idx = pd.date_range(start, end, freq="D")
        serie = 120_000 + np.cumsum(np.random.normal(0, 120, len(idx)))
        df = pd.DataFrame({"data": idx, "Ibovespa": serie})

    cache_set(key, df.assign(data=df["data"].dt.strftime("%Y-%m-%d")).to_dict(orient="list"))
    return df

# =============================================================================
# TRANSFORMAÇÕES
# =============================================================================
def compute_ytd(df_daily: pd.DataFrame, end_date: date) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Diário → Acumulado YTD + resumos (cards, dia, mês anterior)."""
    start_ytd = date(end_date.year, 1, 1)
    dfd = df_daily[(df_daily["data"].dt.date >= start_ytd) & (df_daily["data"].dt.date <= end_date)].copy()
    dfd = dfd.sort_values("data").reset_index(drop=True)

    # acumulado YTD (cumsum)
    for c in CATEGORIAS:
        if c in dfd.columns:
            dfd[c] = dfd[c].fillna(0).cumsum()

    resumo_cards = {c: float(dfd[c].dropna().iloc[-1]) if c in dfd.columns and len(dfd) else 0.0 for c in CATEGORIAS}

    # movimentos do dia (delta diário) usando o diário original
    dl = df_daily[df_daily["data"].dt.date <= end_date].copy().sort_values("data")
    mov_dia = {c: 0.0 for c in CATEGORIAS}
    if len(dl) >= 2:
        last = dl.iloc[-1][CATEGORIAS]
        prev = dl.iloc[-2][CATEGORIAS]
        mov_dia = {c: float(last[c] - prev[c]) for c in CATEGORIAS}

    # mês anterior
    mov_mes = {c: 0.0 for c in CATEGORIAS}
    if not dl.empty:
        last_date = dl["data"].dt.date.max()
        ref = (last_date.replace(day=1) - timedelta(days=1))
        month_start = ref.replace(day=1)
        m = dl[(dl["data"].dt.date >= month_start) & (dl["data"].dt.date <= ref)].copy()
        if len(m) >= 2:
            delta = m.iloc[-1][CATEGORIAS] - m.iloc[0][CATEGORIAS]
            mov_mes = {c: float(delta.get(c, 0.0)) for c in CATEGORIAS}

    return dfd, resumo_cards, mov_dia, mov_mes

# =============================================================================
# FORMATAÇÃO / PLOTS
# =============================================================================
def thousand_dot(x, pos=None):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)

def plot_linhas_ytd(df_ytd: pd.DataFrame, df_ibov: pd.DataFrame) -> str:
    fig, ax1 = plt.subplots(figsize=(14, 6.7), dpi=150)
    fig.patch.set_facecolor("#0b1220")
    ax1.set_facecolor("#0b1220")
    ax2 = ax1.twinx()

    colors = {
        "Estrangeiro": "#50a7ff",
        "Institucional": "#ff7f0e",
        "Pessoa Física": "#22c55e",
        "Inst. Financeira": "#ec4899",
        "Outros": "#8b5cf6",
    }
    for col in CATEGORIAS:
        if col in df_ytd.columns:
            ax1.plot(df_ytd["data"], df_ytd[col], label=col, linewidth=2, color=colors.get(col))

    if not df_ibov.empty:
        ax2.plot(df_ibov["data"], df_ibov["Ibovespa"], linestyle=":", linewidth=2.2, color="white", label="Ibovespa (pontilhado)")

    days = (df_ytd["data"].max() - df_ytd["data"].min()).days
    if days <= 185:
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    else:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))

    # Y esquerda
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax1.set_ylabel("Acumulado (R$ bilhões)", color="white", labelpad=8)

    # Y direita (Ibov) - passo 2.500
    if not df_ibov.empty:
        mn, mx = float(df_ibov["Ibovespa"].min()), float(df_ibov["Ibovespa"].max())
        lo = 2500 * math.floor(mn / 2500) - 2500
        hi = 2500 * math.ceil (mx / 2500) + 2500
        ax2.set_ylim(lo, hi)
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(2500))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(thousand_dot))
        ax2.set_ylabel("Ibovespa (pts)", color="white", labelpad=8)

    for sp in ["top", "right", "bottom", "left"]:
        ax1.spines[sp].set_color("#233148")
        ax2.spines[sp].set_color("#233148")
    ax1.tick_params(axis="x", colors="white"); ax1.tick_params(axis="y", colors="white")
    ax2.tick_params(axis="y", colors="white")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color("white")

    fig.text(0.5, 0.5, "@alan_richard", fontsize=28, color="gray", alpha=0.06, ha="center", va="center")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def plot_estrangeiro_30dias(df_daily: pd.DataFrame, end_date: date) -> str:
    """Barras VERTICAIS últimos 30 dias + MM(28) e rótulo de valor nas barras."""
    d = df_daily[df_daily["data"].dt.date <= end_date].copy().sort_values("data")
    d["estrangeiro"] = d["Estrangeiro"].fillna(0.0)
    d["mm28"] = d["estrangeiro"].rolling(28, min_periods=1).mean()
    d = d.tail(30).copy()

    fig, ax = plt.subplots(figsize=(14, 6.0), dpi=150)
    fig.patch.set_facecolor("#0b1220"); ax.set_facecolor("#0b1220")

    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in d["estrangeiro"]]
    ax.bar(d["data"], d["estrangeiro"], color=colors, width=0.85, zorder=2)

    # média móvel (usa a janela exibida para coerência visual)
    ax.plot(d["data"], d["mm28"], color="#f59e0b", linewidth=2.0, label="Média 28", zorder=3)

    # rótulos nas barras
    for x, y in zip(d["data"], d["estrangeiro"]):
        va = "bottom" if y >= 0 else "top"
        off = 0.03 if y >= 0 else -0.03
        ax.text(x, y + off, f"{y:+.2f}", ha="center", va=va, fontsize=8, color="white")

    ax.axhline(0, color="#233148", linewidth=1)
    ax.set_ylabel("R$ bi (diário)", color="white")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.tick_params(axis="x", colors="white"); ax.tick_params(axis="y", colors="white")
    for sp in ["top", "right", "bottom", "left"]:
        ax.spines[sp].set_color("#233148")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor="white")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

# =============================================================================
# ROTAS
# =============================================================================
@app.route("/refresh")
def refresh():
    cache_clear()
    return redirect(url_for("home"))

@app.route("/", methods=["GET"])
def home():
    last_possible = get_last_possible_date()

    # FLUXO diário
    df_daily = load_fluxo_raw()

    # IBOV
    start_for_ibov = date(last_possible.year, 1, 1) - timedelta(days=10)  # pequena margem
    df_ibov_hist = load_ibov_history(start_for_ibov, last_possible)
    df_ibov = df_ibov_hist[df_ibov_hist["data"].dt.date <= last_possible].copy().sort_values("data").reset_index(drop=True)

    # Transformações → YTD e resumos
    df_ytd, resumo_cards, mov_dia, mov_mes = compute_ytd(df_daily, last_possible)

    # Gráficos
    img_linhas = plot_linhas_ytd(df_ytd, df_ibov)
    img_barras = plot_estrangeiro_30dias(df_daily, last_possible)

    # Texto do último dia
    ld = pd.Series(mov_dia)
    comprador = ld.idxmax(); vcomp = ld.max()
    vendedor  = ld.idxmin(); vvend = ld.min()
    resumo_dia_txt = (
        f"Maior comprador: {comprador} ({'+' if vcomp>=0 else '–'} R$ {abs(vcomp):.1f} bi) • "
        f"Maior vendedor: {vendedor} ({'+' if vvend>=0 else '–'} R$ {abs(vvend):.1f} bi)"
    )

    # Ibovespa do dia
    ibov_close = df_ibov["Ibovespa"].iloc[-1] if len(df_ibov) else np.nan
    ibov_prev  = df_ibov["Ibovespa"].iloc[-2] if len(df_ibov) > 1 else ibov_close
    var = (ibov_close - ibov_prev) if pd.notna(ibov_close) and pd.notna(ibov_prev) else 0.0
    ibov_txt = (
        f"Ibovespa: {int(ibov_close):,}".replace(",", ".")
        + f" ({'+' if var>=0 else '–'}{abs(int(var))} pts no dia)"
        if pd.notna(ibov_close) else "-"
    )

    last_date_str = last_possible.strftime("%d/%m/%Y")

    return render_template(
        "home.html",
        imagem=img_linhas,
        imagem_bar=img_barras,
        resumo=resumo_cards,
        categorias=CATEGORIAS,
        movimentos_dia=mov_dia,
        movimentos_mes=mov_mes,
        resumo_dia_txt=resumo_dia_txt,
        ibov_txt=ibov_txt,
        last_date=last_date_str,
    )

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
