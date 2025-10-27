import os
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger("fluxo_b3")

__all__ = [
    "get_daily_flows_cards",
    "get_series_by_player",
    "get_ibov_series",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
    )
}
TIMEOUT = 20

# ===== Snapshot local em CSV (sem dependências extras) =====
SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "players_snapshot.csv")


def _save_snapshot(df: pd.DataFrame) -> None:
    try:
        if df is not None and not df.empty:
            df.to_csv(SNAPSHOT_PATH, index=False)
    except Exception as e:
        logger.warning("Falha ao salvar snapshot: %s", e)


def _load_snapshot() -> pd.DataFrame | None:
    try:
        if os.path.exists(SNAPSHOT_PATH):
            df = pd.read_csv(SNAPSHOT_PATH)
            if {"date", "player", "net"}.issubset(df.columns):
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df[["date", "player", "net"]]
    except Exception as e:
        logger.warning("Falha ao carregar snapshot: %s", e)
    return None


# ===== Normalização de players =====
PLAYER_MAP = {
    "Estrangeiro": "Estrangeiro",
    "Investidor Estrangeiro": "Estrangeiro",
    "Institucional": "Institucional",
    "Investidor Institucional": "Institucional",
    "Pessoa Física": "Pessoa Física",
    "Pessoa Fisica": "Pessoa Física",
    "Instituição Financeira": "Inst. Financeira",
    "Instituicao Financeira": "Inst. Financeira",
    "Financeira": "Inst. Financeira",
    "Outros": "Outros",
}


def _normalize_player(name):
    if pd.isna(name):
        return None
    name = str(name).strip()
    return PLAYER_MAP.get(name, name)


def _fmt_brl_bi(x):
    try:
        v = abs(float(x))
    except Exception:
        return "R$ 0,0bi"
    sign = "-" if float(x) < 0 else ""
    return f"{sign}R$ {v:,.1f}bi".replace(",", "X").replace(".", ",").replace("X", ".")


# ===== HTML reader sem lxml =====
def _read_html_tables(url: str):
    """Lê tabelas via BeautifulSoup+html5lib com tolerância."""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            tables = pd.read_html(r.text, flavor="bs4", thousands=".", decimal=",")
            if tables:
                return tables
        except Exception as e:
            logger.warning("Falha ao ler %s, tentativa %d, erro=%s", url, attempt + 1, e)
            time.sleep(1 + attempt)
    raise RuntimeError(f"Não foi possível extrair tabelas de {url}")


# ===== Fontes =====
def _get_players_from_carteirafundos() -> pd.DataFrame | None:
    url = "https://carteirafundos.com.br/fluxo"
    try:
        tabs = _read_html_tables(url)
        best = None
        for t in tabs:
            cols = list(t.columns)
            if any("Data" in str(c) for c in cols) and len(cols) >= 4:
                best = t
                break
        if best is None:
            return None

        best.columns = [str(c).strip() for c in best.columns]
        date_col = [c for c in best.columns if "Data" in c][0]
        df = best.rename(columns={date_col: "date"})

        player_cols = [c for c in df.columns if c != "date"]
        melted = df.melt(
            id_vars=["date"], value_vars=player_cols,
            var_name="player_raw", value_name="net_raw"
        )
        melted["player"] = melted["player_raw"].apply(_normalize_player)

        if not np.issubdtype(melted["net_raw"].dtype, np.number):
            melted["net_raw"] = (
                melted["net_raw"].astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
        melted["net"] = pd.to_numeric(melted["net_raw"], errors="coerce") / 1e9
        melted["date"] = pd.to_datetime(melted["date"], dayfirst=True, errors="coerce")
        melted = melted.dropna(subset=["date", "player", "net"])
        melted["date"] = melted["date"].dt.date
        return melted[["date", "player", "net"]]
    except Exception as e:
        logger.warning("Falha carteirafundos: %s", e)
        return None


def _get_players_from_bhm() -> pd.DataFrame | None:
    url = "https://www.bhmoptions.com.br/participantes"
    try:
        tabs = _read_html_tables(url)
        best = max(tabs, key=lambda x: x.shape[0] * x.shape[1])

        best.columns = [str(c).strip() for c in best.columns]
        date_col = None
        for c in best.columns:
            if any(k in str(c) for k in ("Data", "Dia", "Date")):
                date_col = c
                break
        if date_col is None:
            return None

        df = best.rename(columns={date_col: "date"})
        player_cols = [c for c in df.columns if c != "date"]
        melted = df.melt(
            id_vars=["date"], value_vars=player_cols,
            var_name="player_raw", value_name="net_raw"
        )
        melted["player"] = melted["player_raw"].apply(_normalize_player)

        melted["net"] = (
            melted["net_raw"].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        melted["net"] = pd.to_numeric(melted["net"], errors="coerce") / 1e9
        melted["date"] = pd.to_datetime(melted["date"], dayfirst=True, errors="coerce")
        melted = melted.dropna(subset=["date", "player", "net"])
        melted["date"] = melted["date"].dt.date
        return melted[["date", "player", "net"]]
    except Exception as e:
        logger.warning("Falha BHM: %s", e)
        return None


# ===== API pública =====
def get_daily_flows_cards():
    """
    Cards do acumulado no ano por player.
    Fontes -> snapshot -> zeros. Nunca lança exceção.
    """
    df = _get_players_from_carteirafundos()
    if df is None or df.empty:
        df = _get_players_from_bhm()
    if df is None or df.empty:
        df = _load_snapshot()

    if df is None or df.empty:
        zero = {"valor": 0.0, "texto": "Sem dados recentes", "formatado": "R$ 0,0bi"}
        return {"ok": True, "cards": {
            "Estrangeiro": zero, "Institucional": zero, "Pessoa Física": zero,
            "Inst. Financeira": zero, "Outros": zero
        }}

    df["player"] = df["player"].apply(_normalize_player)
    df = df.groupby(["date", "player"], as_index=False)["net"].sum()

    today = datetime.utcnow().date()
    year_start = today.replace(month=1, day=1)
    df = df[df["date"] >= year_start]

    pivot = df.pivot_table(index="date", columns="player", values="net", aggfunc="sum").fillna(0).cumsum()
    last = pivot.tail(1).T.reset_index()
    last.columns = ["player", "acumulado_ano"]

    cards = {}
    for pl in ["Estrangeiro", "Institucional", "Pessoa Física", "Inst. Financeira", "Outros"]:
        v = float(last.loc[last["player"] == pl, "acumulado_ano"].values[0]) if pl in last["player"].values else 0.0
        cards[pl] = {"valor": v, "texto": "Entrada líquida" if v >= 0 else "Saída líquida", "formatado": _fmt_brl_bi(v)}

    try:
        _save_snapshot(df[["date", "player", "net"]])
    except Exception:
        pass

    return {"ok": True, "cards": cards}


def get_series_by_player() -> pd.DataFrame:
    """
    Série (date, player, net). Tolerante: fontes -> snapshot -> vazio.
    """
    df1 = _get_players_from_carteirafundos()
    df2 = _get_players_from_bhm()
    base = df1 if (df1 is not None and not df1.empty) else df2
    if base is None or base.empty:
        base = _load_snapshot()

    if base is None or base.empty:
        return pd.DataFrame(columns=["date", "player", "net"])

    base["player"] = base["player"].apply(_normalize_player)
    base = base.groupby(["date", "player"], as_index=False)["net"].sum()

    try:
        _save_snapshot(base[["date", "player", "net"]])
    except Exception:
        pass

    return base


def get_ibov_series(period_days: int = 365 * 2):
    """Fechamentos do Ibovespa (yfinance). Pode retornar None."""
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=period_days + 10)
        data = yf.download("^BVSP", start=start, end=end, progress=False, auto_adjust=False)
        if data is None or data.empty:
            return None
        s = data["Close"].reset_index()
        s.columns = ["date", "close"]
        s["date"] = s["date"].dt.date
        return s
    except Exception as e:
        logger.warning("Falha ao carregar IBOV: %s", e)
        return None
