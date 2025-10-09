
import os, io, json, math, base64, logging
from datetime import datetime, timedelta, date
from typing import Dict
import pandas as pd, numpy as np
from pandas.tseries.offsets import BDay
from flask import Flask, render_template
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# ---------- Setup ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("fluxo-b3")

DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

B3_DELAY_DAYS = int(os.getenv("B3_DELAY_DAYS", "2"))
TZ = os.getenv("TZ", "America/Sao_Paulo")

CATEGORIAS = ["Estrangeiro", "Institucional", "Pessoa Física", "Inst. Financeira", "Outros"]

# ---------- Cache helpers ----------
def _cache_path(key: str) -> str: return os.path.join(CACHE_DIR, f"{key}.json")
def _cache_get(key: str):
    fp = _cache_path(key)
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: return None
    return None
def _cache_set(key: str, data): 
    with open(_cache_path(key), "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False)

# ---------- Data sources (trocar por B3 real no deploy) ----------
def fetch_fluxo_b3(start: str, end: str) -> pd.DataFrame:
    """
    Esperado: série diária acumulada por categoria (R$ bi). 
    PROD: troque por coleta real (B3/Dados de Mercado, StatusInvest etc.).
    """
    key = f"fluxo_{start}_{end}"; cached = _cache_get(key)
    if cached is not None:
        df = pd.DataFrame(cached); df["data"] = pd.to_datetime(df["data"]); return df

    # fallback sintético
    d0, d1 = pd.to_datetime(start), pd.to_datetime(end)
    idx = pd.date_range(d0, d1, freq="D", tz=TZ)
    np.random.seed(42)
    base = pd.DataFrame({"data": idx.tz_localize(None)})
    base["Estrangeiro"]      = np.cumsum(np.random.normal(0.06, 0.28, len(idx))) + 5
    base["Institucional"]    = np.cumsum(np.random.normal(-0.03, 0.16, len(idx))) - 10
    base["Pessoa Física"]    = np.cumsum(np.random.normal(0.01, 0.05, len(idx))) + 2
    base["Inst. Financeira"] = np.cumsum(np.random.normal(0.006, 0.04, len(idx))) + 1
    base["Outros"]           = np.cumsum(np.random.normal(0.00, 0.03, len(idx))) + 0.5
    _cache_set(key, base.assign(data=base["data"].dt.strftime("%Y-%m-%d")).to_dict(orient="list"))
    return base

def fetch_ibov_close(start: str, end: str) -> pd.DataFrame:
    key = f"ibov_{start}_{end}"; cached = _cache_get(key)
    if cached is not None:
        df = pd.DataFrame(cached); df["data"] = pd.to_datetime(df["data"]); return df
    d0, d1 = pd.to_datetime(start), pd.to_datetime(end)
    idx = pd.date_range(d0, d1, freq="D", tz=TZ)
    serie = 120000 + np.cumsum(np.random.normal(0, 120, len(idx)))
    df = pd.DataFrame({"data": idx.tz_localize(None), "Ibovespa": serie})
    _cache_set(key, df.assign(data=df["data"].dt.strftime("%Y-%m-%d")).to_dict(orient="list"))
    return df

# ---------- Helpers ----------
def thousand_dot(x, pos=None):
    try: return f"{int(x):,}".replace(",", ".")
    except Exception: return str(x)

def rebase_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    first = df.iloc[0][CATEGORIAS]
    for c in CATEGORIAS: df[c] = df[c] - float(first[c])
    return df

def last_day_movements(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) < 2: return {c: 0.0 for c in CATEGORIAS}
    last, prev = df.iloc[-1][CATEGORIAS], df.iloc[-2][CATEGORIAS]
    return {c: float(last[c] - prev[c]) for c in CATEGORIAS}

def last_month_movements(df_full: pd.DataFrame) -> Dict[str, float]:
    if df_full.empty: return {c: 0.0 for c in CATEGORIAS}
    last_date = df_full["data"].dt.date.max()
    ref = (last_date.replace(day=1) - timedelta(days=1))
    month_start = ref.replace(day=1)
    m = df_full[(df_full["data"].dt.date >= month_start) & (df_full["data"].dt.date <= ref)]
    if m.empty: return {c: 0.0 for c in CATEGORIAS}
    deltas = m.iloc[-1][CATEGORIAS] - m.iloc[0][CATEGORIAS]
    return {c: float(deltas[c]) for c in CATEGORIAS}

# ---------- Plots ----------
def plot_ytd(df_fluxo: pd.DataFrame, df_ibov: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(13, 6.5), dpi=150)
    fig.patch.set_facecolor("#0b1220"); ax1.set_facecolor("#0b1220")
    ax2 = ax1.twinx()

    for col in CATEGORIAS:
        ax1.plot(df_fluxo["data"], df_fluxo[col], label=col, linewidth=2)

    ax2.plot(df_ibov["data"], df_ibov["Ibovespa"], linestyle=":", linewidth=2.2, color="white", label="Ibovespa (pontilhado)")

    days = (df_fluxo["data"].max() - df_fluxo["data"].min()).days
    if days <= 185:
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    else:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))

    ax1.yaxis.set_major_locator(mticker.MultipleLocator(5))
    mn, mx = float(df_ibov["Ibovespa"].min()), float(df_ibov["Ibovespa"].max())
    lo = 2500 * math.floor(mn / 2500) - 2500
    hi = 2500 * math.ceil (mx / 2500) + 2500
    ax2.set_ylim(lo, hi)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(2500))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(thousand_dot))

    for sp in ["top","right","bottom","left"]:
        ax1.spines[sp].set_color("#233148"); ax2.spines[sp].set_color("#233148")
    ax1.tick_params(axis="x", colors="white"); ax1.tick_params(axis="y", colors="white")
    ax2.tick_params(axis="y", colors="white")

    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(l1+l2, lb1+lb2, loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color("white")

    fig.text(0.5, 0.5, "@alan_richard", fontsize=30, color="gray", alpha=0.06, ha="center", va="center")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight"); buf.seek(0)
    enc = base64.b64encode(buf.read()).decode("utf-8"); plt.close(fig)
    return enc

def plot_estrangeiro_30d(df_fluxo_ytd: pd.DataFrame):
    """Barra horizontal com os últimos 30 DIAS (deltas do estrangeiro)."""
    df = df_fluxo_ytd[["data","Estrangeiro"]].copy()
    df["delta"] = df["Estrangeiro"].diff()
    df = df.dropna().tail(30)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor("#0b1220"); ax.set_facecolor("#0b1220")
    # cores: verde/positivo, vermelho/negativo
    colors = ["#16a34a" if v >= 0 else "#ef4444" for v in df["delta"]]
    ax.barh(df["data"].dt.strftime("%d/%m"), df["delta"], color=colors)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.set_xlabel("R$ bi (diário)", color="white")
    ax.tick_params(axis="x", colors="white"); ax.tick_params(axis="y", colors="white")
    for sp in ["top","right","bottom","left"]:
        ax.spines[sp].set_color("#233148")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight"); buf.seek(0)
    enc = base64.b64encode(buf.read()).decode("utf-8"); plt.close(fig)
    return enc

# ---------- Route ----------
@app.route("/", methods=["GET"])
def home():
    # 1) Último dia “divulgável” = hoje - delay (dias ÚTEIS)
    last_possible = (pd.Timestamp.today().normalize() - BDay(B3_DELAY_DAYS)).date()

    # 2) Buscar desde jan do ano anterior (para consolidar mês anterior corretamente)
    start_fetch = date(last_possible.year - 1, 1, 1).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")
    df_fluxo_all = fetch_fluxo_b3(start_fetch, today)
    df_ibov_all  = fetch_ibov_close(start_fetch, today)

    # 3) Fim = min(último registro real, last_possible)
    last_flux_date = df_fluxo_all["data"].max().date()
    end_date = min(last_flux_date, last_possible)
    start_ytd = date(end_date.year, 1, 1)

    # 4) Recorte YTD + rebase
    df_fluxo = df_fluxo_all[(df_fluxo_all["data"].dt.date >= start_ytd) & (df_fluxo_all["data"].dt.date <= end_date)].reset_index(drop=True)
    df_ibov  = df_ibov_all [(df_ibov_all ["data"].dt.date >= start_ytd) & (df_ibov_all ["data"].dt.date <= end_date)].reset_index(drop=True)
    if not df_fluxo.empty: df_fluxo = rebase_period(df_fluxo)

    imagem = imagem_bar = None
    resumo_cards = {c: 0.0 for c in CATEGORIAS}
    mov_dia = {c: 0.0 for c in CATEGORIAS}
    mov_mes = {c: 0.0 for c in CATEGORIAS}
    last_date_str = "-"
    diario_texto = "-"
    ibov_texto = "-"

    if not df_fluxo.empty and not df_ibov.empty:
        imagem = plot_ytd(df_fluxo, df_ibov)
        imagem_bar = plot_estrangeiro_30d(df_fluxo)
        resumo_cards = {c: float(df_fluxo[c].dropna().iloc[-1]) for c in CATEGORIAS}
        mov_dia = last_day_movements(df_fluxo)
        mov_mes = last_month_movements(df_fluxo_all)
        last_date_str = end_date.strftime("%d/%m/%Y")
        # resumo escrito do último dia
        # quem comprou/vendeu = maiores positivos/negativos do dia
        ld = pd.Series(mov_dia)
        comprador = ld.idxmax(); vcomp = ld.max()
        vendedor  = ld.idxmin(); vvend = ld.min()
        diario_texto = f"Maior comprador: {comprador} ({'+' if vcomp>=0 else '–'} R$ {abs(vcomp):.1f} bi) • Maior vendedor: {vendedor} ({'+' if vvend>=0 else '–'} R$ {abs(vvend):.1f} bi)"
        ibov_close = df_ibov["Ibovespa"].iloc[-1]
        ibov_prev  = df_ibov["Ibovespa"].iloc[-2] if len(df_ibov)>1 else ibov_close
        var = ibov_close - ibov_prev
        ibov_texto = f"Ibovespa: {int(ibov_close):,}".replace(",", ".") + f" ({'+' if var>=0 else '–'}{abs(var):.0f} pts no dia)"

    return render_template("home.html",
        imagem=imagem, imagem_bar=imagem_bar,
        resumo=resumo_cards, last_date=last_date_str,
        categorias=CATEGORIAS, movimentos_dia=mov_dia, movimentos_mes=mov_mes,
        diario_texto=diario_texto, ibov_texto=ibov_texto
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
