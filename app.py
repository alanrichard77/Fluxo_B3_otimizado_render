import os
import io
import json
import math
import base64
import logging
from datetime import datetime, timedelta, date
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from flask import Flask, render_template, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# --- Flask com caminhos explícitos ---
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

DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Delay padrão de divulgação da B3 (pode ajustar via variável de ambiente)
B3_DELAY_DAYS = int(os.getenv("B3_DELAY_DAYS", "2"))

CATEGORIAS = ["Estrangeiro", "Institucional", "Pessoa Física", "Inst. Financeira", "Outros"]

# ---------------- Cache helpers ----------------
def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")

def _cache_get(key: str):
    fp = _cache_path(key)
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _cache_set(key: str, data):
    with open(_cache_path(key), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

# --------------- Coleta de dados (fallback offline) ---------------
def fetch_fluxo_b3(start: str, end: str) -> pd.DataFrame:
    """
    Em produção, troque por coleta real da B3/Dados de Mercado.
    Fallback offline: gera série acumulada coerente por categoria em R$ bi.
    (Sem suavização para não distorcer saldos.)
    """
    key = f"fluxo_{start}_{end}"
    cached = _cache_get(key)
    if cached is not None:
        df = pd.DataFrame(cached)
        df["data"] = pd.to_datetime(df["data"])
        return df

    try:
        d0, d1 = datetime.fromisoformat(start).date(), datetime.fromisoformat(end).date()
    except Exception:
        d0, d1 = date.today() - timedelta(days=365), date.today()

    idx = pd.date_range(d0, d1, freq="D")
    np.random.seed(42)
    base = pd.DataFrame({"data": idx})
    base["Estrangeiro"]      = np.cumsum(np.random.normal(0.06, 0.28, len(idx))) + 5
    base["Institucional"]    = np.cumsum(np.random.normal(-0.03, 0.16, len(idx))) - 10
    base["Pessoa Física"]    = np.cumsum(np.random.normal(0.01, 0.05, len(idx))) + 2
    base["Inst. Financeira"] = np.cumsum(np.random.normal(0.006, 0.04, len(idx))) + 1
    base["Outros"]           = np.cumsum(np.random.normal(0.00, 0.03, len(idx))) + 0.5

    _cache_set(key, base.assign(data=base["data"].dt.strftime("%Y-%m-%d")).to_dict(orient="list"))
    return base

def fetch_ibov_close(start: str, end: str) -> pd.DataFrame:
    key = f"ibov_{start}_{end}"
    cached = _cache_get(key)
    if cached is not None:
        df = pd.DataFrame(cached)
        df["data"] = pd.to_datetime(df["data"])
        return df

    try:
        d0, d1 = datetime.fromisoformat(start).date(), datetime.fromisoformat(end).date()
    except Exception:
        d0, d1 = date.today() - timedelta(days=365), date.today()

    idx = pd.date_range(d0, d1, freq="D")
    serie = 120000 + np.cumsum(np.random.normal(0, 120, len(idx)))
    df = pd.DataFrame({"data": idx, "Ibovespa": serie})
    _cache_set(key, df.assign(data=df["data"].dt.strftime("%Y-%m-%d")).to_dict(orient="list"))
    return df

# ----------------- Helpers de formatação -----------------
def fmt_bi(x: float) -> str:
    s = f"{x:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    sinal = "+" if x >= 0 else "–"
    return f"{sinal} R$ {s.replace('-', '')} bi"

def thousand_dot(x, pos=None):
    try:
        s = f"{int(x):,}".replace(",", ".")
    except Exception:
        s = str(x)
    return s

# ----------------- Regras de negócio -----------------
def rebase_period(df: pd.DataFrame) -> pd.DataFrame:
    """Zera as séries no 1º dia do filtro para mostrar APENAS o acumulado do período (corrige YTD/2025)."""
    df = df.copy()
    first = df.iloc[0][CATEGORIAS]
    for c in CATEGORIAS:
        if c in df.columns:
            df[c] = df[c] - float(first[c])
    return df

def last_day_movements(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) < 2:
        return {c: 0.0 for c in CATEGORIAS}
    last = df.iloc[-1][CATEGORIAS]
    prev = df.iloc[-2][CATEGORIAS]
    return {c: float(last[c] - prev[c]) for c in CATEGORIAS}

def last_month_movements(df_full: pd.DataFrame) -> Dict[str, float]:
    """Consolidado do mês ANTERIOR, com base na série completa (não no filtro)."""
    if df_full.empty:
        return {c: 0.0 for c in CATEGORIAS}
    last_date = df_full["data"].dt.date.max()
    ref = (last_date.replace(day=1) - timedelta(days=1))  # último dia do mês anterior
    month_start = ref.replace(day=1)
    m = df_full[(df_full["data"].dt.date >= month_start) & (df_full["data"].dt.date <= ref)]
    if m.empty:
        return {c: 0.0 for c in CATEGORIAS}
    deltas = m.iloc[-1][CATEGORIAS] - m.iloc[0][CATEGORIAS]
    return {c: float(deltas[c]) for c in CATEGORIAS}

# ----------------- Plot -----------------
def plot_fluxo(df_fluxo: pd.DataFrame, df_ibov: pd.DataFrame) -> Tuple[str, Dict[str, float], str]:
    fig, ax1 = plt.subplots(figsize=(13, 6.5), dpi=150)
    fig.patch.set_facecolor("#0b1220")
    ax1.set_facecolor("#0b1220")
    ax2 = ax1.twinx()

    # Séries de fluxo
    for col in CATEGORIAS:
        if col in df_fluxo.columns:
            ax1.plot(df_fluxo["data"], df_fluxo[col], label=col, linewidth=2)

    # Ibovespa: branco e pontilhado
    if "Ibovespa" in df_ibov.columns:
        ax2.plot(
            df_ibov["data"], df_ibov["Ibovespa"],
            linestyle=":", linewidth=2.2, color="white", label="Ibovespa (pontilhado)"
        )

    # Eixo X legível: semanal (<= 6 meses) ou mensal (> 6 meses)
    days = (df_fluxo["data"].max() - df_fluxo["data"].min()).days
    if days <= 185:
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    else:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))

    # Y esquerda (R$ bi): passo de 5 em 5
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(5))

    # Y direita (Ibov): passo 2.500 + margem de um passo pra não “comer” a curva
    if "Ibovespa" in df_ibov.columns:
        mn, mx = float(df_ibov["Ibovespa"].min()), float(df_ibov["Ibovespa"].max())
        lo = 2500 * math.floor(mn / 2500) - 2500
        hi = 2500 * math.ceil (mx / 2500) + 2500
        ax2.set_ylim(lo, hi)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(2500))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(thousand_dot))

    # Visual: sem grades (fundo liso) e spines discretos
    for sp in ["top", "right", "bottom", "left"]:
        ax1.spines[sp].set_color("#233148")
        ax2.spines[sp].set_color("#233148")
    ax1.tick_params(axis="x", colors="white")
    ax1.tick_params(axis="y", colors="white")
    ax2.tick_params(axis="y", colors="white")

    # Legenda
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("white")

    # Marca d'água menor
    fig.text(0.5, 0.5, "@alan_richard", fontsize=30, color="gray", alpha=0.06, ha="center", va="center")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    resumo = {c: float(df_fluxo[c].dropna().iloc[-1]) for c in CATEGORIAS}
    last_date = df_fluxo["data"].dropna().iloc[-1].strftime("%d/%m/%Y")
    return encoded, resumo, last_date

# ----------------- Rota -----------------
@app.route("/", methods=["GET"])
def home():
    # Usuário define apenas a data de início; a final é sempre o último dado disponível (considerando delay da B3)
    start_req = request.args.get("start_date")
    start = start_req or date.today().replace(month=1, day=1).strftime("%Y-%m-%d")

    # Busca ampla
    wide_start = (datetime.fromisoformat(start) - timedelta(days=3*365)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")
    df_fluxo_all = fetch_fluxo_b3(wide_start, today)
    df_ibov_all = fetch_ibov_close(wide_start, today)

    # Data final = min(último registro na base, hoje - B3_DELAY_DAYS dias ÚTEIS)
    last_possible = (pd.Timestamp.today().normalize() - BDay(B3_DELAY_DAYS)).date()
    last_flux_date = df_fluxo_all["data"].max().date()
    end_date = min(last_flux_date, last_possible).strftime("%Y-%m-%d")

    # Recorte + rebase (acumulado do período)
    df_fluxo = df_fluxo_all[(df_fluxo_all["data"] >= pd.to_datetime(start)) & (df_fluxo_all["data"] <= pd.to_datetime(end_date))].reset_index(drop=True)
    df_ibov  = df_ibov_all [(df_ibov_all ["data"] >= pd.to_datetime(start)) & (df_ibov_all ["data"] <= pd.to_datetime(end_date))].reset_index(drop=True)
    if not df_fluxo.empty:
        df_fluxo = rebase_period(df_fluxo)

    imagem, resumo_cards, last_date = (None, {c: 0.0 for c in CATEGORIAS}, "-")
    mov_dia = {c: 0.0 for c in CATEGORIAS}
    mov_mes = {c: 0.0 for c in CATEGORIAS}

    if not df_fluxo.empty and not df_ibov.empty:
        imagem, resumo_cards, last_date = plot_fluxo(df_fluxo, df_ibov)
        mov_dia = last_day_movements(df_fluxo)
        mov_mes = last_month_movements(df_fluxo_all)

    return render_template(
        "home.html",
        imagem=imagem,
        resumo=resumo_cards,
        last_date=last_date,
        start_date=start,
        end_date=end_date,
        categorias=CATEGORIAS,
        movimentos_dia=mov_dia,
        movimentos_mes=mov_mes
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
