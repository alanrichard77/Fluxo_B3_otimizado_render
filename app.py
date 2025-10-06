
import os
import io
import json
import time
import math
import base64
import logging
from datetime import datetime, timedelta, date
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import requests
from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --------------------------
# App & Logger
# --------------------------
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("fluxo-b3")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CATEGORIAS = ["Estrangeiro", "Institucional", "Pessoa Física", "Inst. Financeira", "Outros"]

# --------------------------
# Utils
# --------------------------
def parse_valor(valor: str) -> float:
    if valor is None: 
        return 0.0
    v = str(valor).replace("R$", "").replace(" ", "").replace(".", "").replace(",", ".").strip().lower()
    if v in ["", "-", "nan"]:
        return 0.0
    if "mi" in v:
        try: return float(v.replace("mi", "")) / 1000.0
        except: return 0.0
    if "bi" in v:
        try: return float(v.replace("bi", ""))
        except: return 0.0
    try:
        return float(v)
    except:
        return 0.0

def fmt_bi(x: float) -> str:
    s = f"{x:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    sinal = "+" if x >= 0 else "–"
    return f"{sinal} R$ {s.replace('-', '')} bi"

def thousand_dot(x, pos=None):
    try:
        s = f"{int(x):,}".replace(",", ".")
    except:
        s = str(x)
    return s

def daterange_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")

def load_cache(key: str):
    fp = cache_path(key)
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Erro lendo cache {key}: {e}")
    return None

def save_cache(key: str, data):
    try:
        with open(cache_path(key), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Erro salvando cache {key}: {e}")

# --------------------------
# Data fetchers (resilientes c/ fallback)
# OBS: Endpoints reais podem mudar. O código já trata retry e queda.
# --------------------------
HDR = {"User-Agent": "Mozilla/5.0 (FluxoB3/1.0)"}

def fetch_fluxo_b3(start: str, end: str) -> pd.DataFrame:
    """
    Tenta buscar o fluxo por categoria entre start e end.
    Fallback: se não houver internet/endpoint, devolve DataFrame simulado.
    Esperado: colunas ['data','Estrangeiro','Institucional','Pessoa Física','Inst. Financeira','Outros'] em R$ bi acumulado.
    """
    key = f"fluxo_{start}_{end}"
    cached = load_cache(key)
    if cached is not None:
        df = pd.DataFrame(cached)
        df['data'] = pd.to_datetime(df['data'])
        return df

    # Fallback offline: série simulada coerente (substituir em prod por scraping/API do seu pipeline)
    try:
        d0 = datetime.fromisoformat(start).date()
        d1 = datetime.fromisoformat(end).date()
    except:
        d0 = date.today() - timedelta(days=180)
        d1 = date.today()
    idx = pd.date_range(d0, d1, freq="D")
    np.random.seed(42)
    base = pd.DataFrame({"data": idx})
    base["Estrangeiro"] = np.cumsum(np.random.normal(0.05, 0.3, len(idx))) + 5
    base["Institucional"] = np.cumsum(np.random.normal(-0.03, 0.15, len(idx))) - 10
    base["Pessoa Física"] = np.cumsum(np.random.normal(0.01, 0.05, len(idx))) + 2
    base["Inst. Financeira"] = np.cumsum(np.random.normal(0.005, 0.04, len(idx))) + 1
    base["Outros"] = np.cumsum(np.random.normal(0.0, 0.03, len(idx))) + 0.5
    base[CATEGORIAS] = base[CATEGORIAS].rolling(5, min_periods=1).mean()
    save_cache(key, base.assign(data=base["data"].dt.strftime("%Y-%m-%d")).to_dict(orient="list"))
    return base

def fetch_ibov_close(start: str, end: str) -> pd.DataFrame:
    """
    Busca série do Ibovespa para o período [start, end].
    Fallback: série simulada caso offline.
    """
    key = f"ibov_{start}_{end}"
    cached = load_cache(key)
    if cached is not None:
        df = pd.DataFrame(cached)
        df['data'] = pd.to_datetime(df['data'])
        return df

    try:
        d0 = datetime.fromisoformat(start).date()
        d1 = datetime.fromisoformat(end).date()
    except:
        d0 = date.today() - timedelta(days=180)
        d1 = date.today()
    idx = pd.date_range(d0, d1, freq="D")
    serie = 120000 + np.cumsum(np.random.normal(0, 120, len(idx)))
    df = pd.DataFrame({"data": idx, "Ibovespa": serie})
    save_cache(key, df.assign(data=df["data"].dt.strftime("%Y-%m-%d")).to_dict(orient="list"))
    return df

# --------------------------
# Core transform
# --------------------------
def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna por mês o maior comprador e maior vendedor com saldo.
    """
    if df.empty:
        return pd.DataFrame(columns=["mes", "maior_comprador", "saldo_compra", "maior_vendedor", "saldo_venda"])
    tmp = df.copy()
    tmp["mes"] = tmp["data"].dt.to_period("M")
    rows = []
    for m, g in tmp.groupby("mes"):
        last = g.iloc[-1][CATEGORIAS]
        first_vals = g.iloc[0][CATEGORIAS]
        deltas = last - first_vals
        max_cat = deltas.idxmax()
        min_cat = deltas.idxmin()
        rows.append({
            "mes": str(m),
            "maior_comprador": max_cat,
            "saldo_compra": float(deltas[max_cat]),
            "maior_vendedor": min_cat,
            "saldo_venda": float(deltas[min_cat])
        })
    out = pd.DataFrame(rows)
    return out

# --------------------------
# Plot
# --------------------------
def plot_fluxo(df_fluxo: pd.DataFrame, df_ibov: pd.DataFrame) -> Tuple[str, Dict[str, float], str]:
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=140)
    fig.patch.set_facecolor("#0b1220")
    ax1.set_facecolor("#0b1220")
    ax2 = ax1.twinx()

    for col in CATEGORIAS:
        if col in df_fluxo.columns:
            ax1.plot(df_fluxo["data"], df_fluxo[col], label=col, linewidth=2)

    if "Ibovespa" in df_ibov.columns:
        ax2.plot(df_ibov["data"], df_ibov["Ibovespa"], linestyle=":", linewidth=2, label="Ibovespa (pontilhado)")

    ax1.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=7))
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m"))

    ax1.set_ylabel("Acumulado (R$ bi)", color="white")
    ax1.tick_params(axis="y", colors="white")
    ax1.tick_params(axis="x", colors="white", rotation=0)

    ax2.set_ylabel("Ibovespa (pts)", color="white")
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(2500))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(thousand_dot))
    ax2.tick_params(axis="y", colors="white")

    lines, labels = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(lines + l2, labels + lb2, loc="upper left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color("white")

    ax1.grid(color="#233148", linestyle="--", linewidth=0.6, alpha=0.5)
    for spine in ["top", "right", "bottom", "left"]:
        ax1.spines[spine].set_color("#233148")
        ax2.spines[spine].set_color("#233148")

    fig.text(0.5, 0.5, "@alan_richard", fontsize=60, color="gray", alpha=0.08, ha="center", va="center")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    resumo = {}
    for col in CATEGORIAS:
        resumo[col] = float(df_fluxo[col].dropna().iloc[-1]) if col in df_fluxo.columns and not df_fluxo[col].dropna().empty else 0.0
    last_date = df_fluxo["data"].dropna().iloc[-1].strftime("%d/%m/%Y")
    return encoded, resumo, last_date

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    start = request.values.get("start_date") or (date.today() - timedelta(days=180)).strftime("%Y-%m-%d")
    end = request.values.get("end_date") or date.today().strftime("%Y-%m-%d")

    if request.method == "POST" and request.form.get("acao") == "atualizar":
        pass  # cache é por range e será sobrescrito

    df_fluxo = fetch_fluxo_b3(start, end)
    df_ibov = fetch_ibov_close(start, end)

    if df_fluxo.empty:
        imagem = None
        resumo = {c: 0.0 for c in CATEGORIAS}
        last_date = "-"
        monthly = pd.DataFrame()
    else:
        min_d = max(df_fluxo["data"].min(), df_ibov["data"].min())
        max_d = min(df_fluxo["data"].max(), df_ibov["data"].max())
        df_fluxo = df_fluxo[(df_fluxo["data"] >= min_d) & (df_fluxo["data"] <= max_d)].reset_index(drop=True)
        df_ibov = df_ibov[(df_ibov["data"] >= min_d) & (df_ibov["data"] <= max_d)].reset_index(drop=True)

        imagem, resumo, last_date = plot_fluxo(df_fluxo, df_ibov)
        monthly = compute_monthly_summary(df_fluxo)

    painel = []
    if not monthly.empty:
        for _, r in monthly.iterrows():
            painel.append({
                "mes": r["mes"],
                "maior_comprador": r["maior_comprador"],
                "saldo_compra": fmt_bi(r["saldo_compra"]),
                "maior_vendedor": r["maior_vendedor"],
                "saldo_venda": fmt_bi(r["saldo_venda"]),
            })

    return render_template(
        "home.html",
        imagem=imagem,
        resumo=resumo,
        last_date=last_date,
        start_date=start,
        end_date=end,
        painel=painel,
        categorias=CATEGORIAS
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
