import time
import logging
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
import yfinance as yf
import os
SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "players_snapshot.parquet")

def _save_snapshot(df: pd.DataFrame):
    try:
        if df is not None and not df.empty:
            df.to_parquet(SNAPSHOT_PATH, index=False)
    except Exception as e:
        logger.warning("Falha ao salvar snapshot: %s", e)

def _load_snapshot() -> pd.DataFrame | None:
    try:
        if os.path.exists(SNAPSHOT_PATH):
            return pd.read_parquet(SNAPSHOT_PATH)
    except Exception as e:
        logger.warning("Falha ao carregar snapshot: %s", e)
    return None

logger = logging.getLogger("fluxo_b3")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11, Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
}
TIMEOUT = 20

# Normalização de nomes
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

def _read_html_tables(url):
    """Lê todas as tabelas da página com BeautifulSoup+html5lib (sem lxml) e com tolerância a mudanças."""
    import pandas as pd
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            # Usa o parser 'html5lib' via BeautifulSoup (flavor='bs4')
            tables = pd.read_html(
                r.text,
                flavor="bs4",          # força BeautifulSoup
                thousands=".",         # separador de milhar BR
                decimal=",",           # decimal BR
            )
            if tables:
                return tables
        except Exception as e:
            logger.warning("Falha ao ler %s, tentativa %d, erro=%s", url, attempt + 1, e)
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Não foi possível extrair tabelas de {url}")

def _normalize_player(name):
    if pd.isna(name):
        return None
    name = str(name).strip()
    return PLAYER_MAP.get(name, name)

def get_daily_flows_cards():
    """
    Constrói cards de acumulado do ano por player, com indicação de entrada/saída.
    Fonte primária: CarteiraFundos, fallback: BHM.
    """
    df = _get_players_from_carteirafundos()
    if df is None or df.empty:
        df = _get_players_from_bhm()
    if df is None or df.empty:
        raise RuntimeError("Falha em obter dados de players.")
    # Último valor acumulado por player
    last = (
        df.sort_values("date")
          .groupby("player", as_index=False)["net"].last()
          .rename(columns={"net": "acumulado_ano"})
    )
    last["sinal"] = np.where(last["acumulado_ano"] >= 0, "Entrada líquida", "Saída líquida")
    # Formatação
    last["valor_str"] = last["acumulado_ano"].apply(_fmt_brl_bi)
    # cards esperados:
    cards = {}
    for pl in ["Estrangeiro", "Institucional", "Pessoa Física", "Inst. Financeira", "Outros"]:
        row = last[last["player"] == pl]
        if row.empty:
            continue
        v = float(row["acumulado_ano"].values[0])
        cards[pl] = {
            "valor": v,
            "texto": "Entrada líquida" if v >= 0 else "Saída líquida",
            "formatado": _fmt_brl_bi(v),
        }
    return {"ok": True, "cards": cards}

def get_series_by_player():
    """
    Retorna DataFrame com colunas: date, player, net (R$ bilhões)
    unindo fontes e normalizando dados. Priorizamos CarteiraFundos.
    """
    df1 = _get_players_from_carteirafundos()
    df2 = _get_players_from_bhm()
    if df1 is None and df2 is None:
        raise RuntimeError("Sem séries de players disponíveis.")
    if df1 is not None and not df1.empty:
        base = df1
    else:
        base = df2

    # Ajuste de tipos
    base["date"] = pd.to_datetime(base["date"]).dt.date
    base["player"] = base["player"].apply(_normalize_player)
    base = base.dropna(subset=["player"])
    # agregado por dia, garantindo consistência
    base = base.groupby(["date", "player"], as_index=False)["net"].sum()
    return base

def _get_players_from_carteirafundos():
    """
    Lê https://carteirafundos.com.br/fluxo e extrai tabela por participante.
    Site costuma ter uma tabela com datas e colunas por player.
    """
    url = "https://carteirafundos.com.br/fluxo"
    try:
        tables = _read_html_tables(url)
        # Heurística: escolher a tabela com mais colunas e uma coluna de datas
        best = None
        for t in tables:
            cols = [c for c in t.columns]
            if any("Data" in str(c) for c in cols) and len(cols) >= 4:
                best = t
                break
        if best is None:
            return None

        # Normalização
        best.columns = [str(c).strip() for c in best.columns]
        date_col = [c for c in best.columns if "Data" in c][0]
        df = best.rename(columns={date_col: "date"})
        # Tenta identificar colunas de players
        player_cols = [c for c in df.columns if c != "date"]
        melted = df.melt(id_vars=["date"], value_vars=player_cols,
                         var_name="player_raw", value_name="net_raw")
        melted["player"] = melted["player_raw"].apply(_normalize_player)
        # Converter valores para bilhões
        melted["net"] = pd.to_numeric(melted["net_raw"], errors="coerce") / 1e9
        melted = melted.dropna(subset=["net"])
        melted["date"] = pd.to_datetime(melted["date"], dayfirst=True, errors="coerce")
        melted = melted.dropna(subset=["date"])
        return melted[["date", "player", "net"]]
    except Exception as e:
        logger.warning("Falha carteirafundos: %s", e)
        return None

def _get_players_from_bhm():
    """
    Lê https://www.bhmoptions.com.br/participantes
    A página possui uma tabela por participante com histórico diário.
    """
    url = "https://www.bhmoptions.com.br/participantes"
    try:
        tables = _read_html_tables(url)
        # tentar a maior tabela com colunas similares
        best = max(tables, key=lambda x: x.shape[0] * x.shape[1])
        best.columns = [str(c).strip() for c in best.columns]
        # Procurar a coluna de data
        date_col = None
        for c in best.columns:
            if "Data" in c or "Dia" in c or "Date" in c:
                date_col = c
                break
        if date_col is None:
            return None

        df = best.rename(columns={date_col: "date"})
        player_cols = [c for c in df.columns if c != "date"]
        melted = df.melt(id_vars=["date"], value_vars=player_cols,
                         var_name="player_raw", value_name="net_raw")
        melted["player"] = melted["player_raw"].apply(_normalize_player)
        # converter valores, aceitar "." milhar e "," decimal
        melted["net"] = (
            melted["net_raw"].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        melted["net"] = pd.to_numeric(melted["net"], errors="coerce") / 1e9
        melted = melted.dropna(subset=["net"])
        melted["date"] = pd.to_datetime(melted["date"], dayfirst=True, errors="coerce")
        melted = melted.dropna(subset=["date"])
        return melted[["date", "player", "net"]]
    except Exception as e:
        logger.warning("Falha BHM: %s", e)
        return None

def get_ibov_series(period_days: int = 365*2):
    """
    Série de fechamento do Ibovespa para overlay no gráfico principal.
    Usa yfinance como fonte acessível.
    """
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=period_days+10)
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

def _fmt_brl_bi(x):
    try:
        s = abs(float(x))
    except:
        return "R$ 0,0 bi"
    sign = "-" if x < 0 else ""
    return f"{sign}R$ {s:,.1f}bi".replace(",", "X").replace(".", ",").replace("X", ".")

