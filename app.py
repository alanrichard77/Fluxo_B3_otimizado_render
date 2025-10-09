diff --git a/app.py b/app.py
index 8df269a40955615033a2ccedd87c9bd362495270..28058ac3de1a9e01b2ee5e0c3696aa2f353fc87a 100644
--- a/app.py
+++ b/app.py
@@ -1,509 +1,664 @@
-import os, io, json, math, base64, logging, time
-from datetime import datetime, timedelta, date
-from typing import Dict, List, Optional, Tuple
+import os, io, json, math, base64, logging, time, zipfile
+from datetime import timedelta, date
+from typing import Dict, Optional, Tuple
 
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
+from requests import Session
 from bs4 import BeautifulSoup
 
 # ============================================================================
 # CONFIG
 # ============================================================================
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
 
 # Dias ÚTEIS de atraso para a B3 divulgar
 B3_DELAY_DAYS = int(os.getenv("B3_DELAY_DAYS", "2"))
 TZ = os.getenv("TZ", "America/Sao_Paulo")
 
-# CSV externo opcional, como fallback de dados
-FLUXO_CSV_URL = os.getenv("FLUXO_CSV_URL", "").strip()
-
 CATEGORIAS = ["Estrangeiro", "Institucional", "Pessoa Física", "Inst. Financeira", "Outros"]
 
 
 # ============================================================================
 # CACHE BÁSICO EM DISCO
 # ============================================================================
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
 
 
 # ============================================================================
 # FONTE 1 — Dados de Mercado (scraping)
 #   - Página: https://www.dadosdemercado.com.br/fluxo
 #   - Captura a tabela principal (data e saldos diários por player)
 #   - Campos esperados (variam em maiúsc./acentos): Data, Estrangeiro, Institucional,
 #     Pessoa Física, Inst. Financeira, Outros
 #   - Valores em bilhões (R$)
 # ============================================================================
+# Endpoints e headers para dados externos
 DDM_URL = "https://www.dadosdemercado.com.br/fluxo"
+DDM_EXPORT_URL = "https://www.dadosdemercado.com.br/fluxo/export"
+B3_FLUXO_API = "https://arquivos.b3.com.br/api/download/"
+DEFAULT_HEADERS = {
+    "User-Agent": (
+        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
+        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
+    ),
+    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
+}
+
+
+def _make_session() -> Session:
+    """Cria uma sessão HTTP resiliente (honra proxy quando disponível)."""
+    sess = requests.Session()
+    sess.headers.update(DEFAULT_HEADERS)
+    sess.trust_env = True
+    return sess
+
+def _normaliza_fluxo(df: pd.DataFrame) -> pd.DataFrame:
+    """Padroniza colunas numéricas para bilhões de reais."""
+    df = df.copy()
+    for col in CATEGORIAS:
+        if col in df.columns:
+            df[col] = pd.to_numeric(df[col], errors="coerce")
+            max_abs = df[col].abs().max()
+            if pd.notna(max_abs) and max_abs > 1_000:
+                df[col] = df[col] / 1_000
+    return df
+
 
 def fetch_fluxo_ddm() -> Optional[pd.DataFrame]:
     try:
         logger.info("Coletando fluxo em Dados de Mercado...")
-        html = requests.get(DDM_URL, timeout=20).text
-        # Primeiro tenta via read_html (mais robusto)
-        tables = pd.read_html(html, thousands=".", decimal=",")
+        sess = _make_session()
+
+        # Tenta endpoint de exportação (CSV) para dados limpos
+        params = {"formato": "csv"}
+        html = None
+        try:
+            r_csv = sess.get(DDM_EXPORT_URL, params=params, timeout=20)
+            if r_csv.ok and "data;" in r_csv.text.lower():
+                df = pd.read_csv(io.StringIO(r_csv.text), sep=";", decimal=",", thousands=".")
+            else:
+                raise ValueError("CSV indisponível")
+        except Exception:
+            html = sess.get(DDM_URL, timeout=20).text
+            df = None
+        if html is not None and df is None:
+            # Primeiro tenta via read_html (mais robusto)
+            tables = pd.read_html(html, thousands=".", decimal=",")
         # Escolhe a maior tabela com coluna "Data"
-        candidates = [t for t in tables if any(c.lower() == "data" for c in t.columns)]
-        if not candidates:
-            # Fallback parse manual com BeautifulSoup
-            soup = BeautifulSoup(html, "html.parser")
-            table = soup.find("table")
-            df = pd.read_html(str(table), thousands=".", decimal=",")[0]
-        else:
-            # geralmente a primeira maior é a correta
-            candidates.sort(key=lambda d: d.shape[0], reverse=True)
-            df = candidates[0]
+        if df is None:
+            candidates = [t for t in tables if any(str(c).strip().lower() == "data" for c in t.columns)]
+            if not candidates:
+                soup = BeautifulSoup(html, "html.parser")
+                table = soup.find("table")
+                if table is None:
+                    raise ValueError("Tabela de fluxo não encontrada")
+                df = pd.read_html(str(table), thousands=".", decimal=",")[0]
+            else:
+                candidates.sort(key=lambda d: d.shape[0], reverse=True)
+                df = candidates[0]
 
         # Normaliza colunas
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
             elif "inst." in cl or "financeir" in cl:
                 rename_map[c] = "Inst. Financeira"
             elif "outros" in cl:
                 rename_map[c] = "Outros"
         df = df.rename(columns=rename_map)
 
         cols_needed = ["data"] + CATEGORIAS
         df = df[[c for c in cols_needed if c in df.columns]].copy()
         if "data" not in df.columns or len(df) == 0:
             return None
 
         # Converte datas e números (já veio com decimal=, thousands=.)
         df["data"] = pd.to_datetime(df["data"], dayfirst=True)
 
         # Se os números estiverem em milhões, converte p/ bilhões (heurística)
-        for col in CATEGORIAS:
-            if col in df.columns:
-                df[col] = pd.to_numeric(df[col], errors="coerce")
+        df = _normaliza_fluxo(df)
         # Remove linhas totalmente vazias
         df = df.dropna(subset=["data"]).reset_index(drop=True)
 
         # Ordena por data ascendente
         df = df.sort_values("data").reset_index(drop=True)
 
         # Alguns sites trazem dados diários líquidos (não acumulados).
         # Manteremos como "diário" aqui; acumulado aplicamos depois (rebase YTD).
         return df
     except Exception as e:
         logger.warning(f"Falha ao coletar Dados de Mercado: {e}")
         return None
 
 
-# ============================================================================
-# FONTE 2 — CSV externo (se informado em FLUXO_CSV_URL)
-#   Espera colunas: date, estrangeiro, institucional, pessoa_fisica, inst_financeira, outros
-#   em valores DIÁRIOS (R$ bi). Data em yyyy-mm-dd.
-# ============================================================================
-def fetch_fluxo_csv() -> Optional[pd.DataFrame]:
-    if not FLUXO_CSV_URL:
+def fetch_fluxo_b3(start: date, end: date) -> Optional[pd.DataFrame]:
+    """Consulta os arquivos oficiais da B3 para fluxo de investidores."""
+    sess = _make_session()
+    registros = []
+    atual = pd.Timestamp(start)
+    limite = pd.Timestamp(end)
+    while atual <= limite:
+        caminho = f"FluxoInvestidor/{atual.year}/{atual.month:02d}/Fluxo_Investidor_{atual:%Y%m%d}.csv"
+        try:
+            resp = sess.get(B3_FLUXO_API, params={"fileName": caminho}, timeout=20)
+            if resp.status_code != 200 or not resp.content:
+                atual += BDay(1)
+                continue
+            conteudo = resp.content
+            if conteudo[:2] == b"PK":
+                with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
+                    csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
+                    if not csv_name:
+                        atual += BDay(1)
+                        continue
+                    conteudo = zf.read(csv_name)
+            csv_buffer = io.StringIO(conteudo.decode("latin-1"))
+            df_day = pd.read_csv(csv_buffer, sep=";")
+            df_day.columns = [str(c).strip() for c in df_day.columns]
+            if "Data" not in df_day.columns:
+                atual += BDay(1)
+                continue
+            df_day["Data"] = pd.to_datetime(df_day["Data"], dayfirst=True, errors="coerce")
+            df_day = df_day.dropna(subset=["Data"])
+
+            saldo_cols = [c for c in df_day.columns if "saldo" in c.lower()]
+            if not saldo_cols:
+                atual += BDay(1)
+                continue
+            saldo_col = saldo_cols[0]
+
+            categoria_col = None
+            for possible in ["Categoria", "Investidor", "Tipo Investidor"]:
+                if possible in df_day.columns:
+                    categoria_col = possible
+                    break
+            if categoria_col is None:
+                atual += BDay(1)
+                continue
+
+            df_pivot = (
+                df_day[["Data", categoria_col, saldo_col]]
+                .rename(columns={"Data": "data", categoria_col: "categoria", saldo_col: "valor"})
+            )
+            df_pivot["valor"] = pd.to_numeric(df_pivot["valor"], errors="coerce")
+            if df_pivot["valor"].abs().max() > 1_000_000:
+                df_pivot["valor"] = df_pivot["valor"] / 1_000_000
+            elif df_pivot["valor"].abs().max() > 1_000:
+                df_pivot["valor"] = df_pivot["valor"] / 1_000
+
+            pivot = df_pivot.pivot_table(index="data", columns="categoria", values="valor", aggfunc="sum")
+            pivot = pivot.reset_index()
+            registros.append(pivot)
+        except Exception as exc:
+            logger.warning(f"Erro ao coletar fluxo na B3 para {atual.date():%Y-%m-%d}: {exc}")
+        finally:
+            atual += BDay(1)
+
+    if not registros:
         return None
-    try:
-        logger.info(f"Coletando CSV externo: {FLUXO_CSV_URL}")
-        df = pd.read_csv(FLUXO_CSV_URL)
-        rename = {
-            "date": "data",
-            "estrangeiro": "Estrangeiro",
-            "institucional": "Institucional",
-            "pessoa_fisica": "Pessoa Física",
-            "inst_financeira": "Inst. Financeira",
-            "outros": "Outros",
-        }
-        df = df.rename(columns=rename)
-        df["data"] = pd.to_datetime(df["data"])
-        for c in CATEGORIAS:
-            if c in df.columns:
-                df[c] = pd.to_numeric(df[c], errors="coerce")
-        df = df.dropna(subset=["data"]).sort_values("data").reset_index(drop=True)
-        return df
-    except Exception as e:
-        logger.warning(f"Falha ao coletar CSV externo: {e}")
+
+    combinado = pd.concat(registros, ignore_index=True)
+    combinado = combinado.rename(columns={
+        "Investidor Estrangeiro": "Estrangeiro",
+        "Estrangeiro": "Estrangeiro",
+        "Invest. Estrangeiro": "Estrangeiro",
+        "Pessoa Física": "Pessoa Física",
+        "Pessoa Fisica": "Pessoa Física",
+        "Institucional": "Institucional",
+        "Instituicao Financeira": "Inst. Financeira",
+        "Instituição Financeira": "Inst. Financeira",
+        "Outros": "Outros",
+    })
+    colunas = [c for c in CATEGORIAS if c in combinado.columns]
+    if not colunas:
         return None
+    combinado = combinado[["data"] + colunas]
+    combinado = combinado.sort_values("data").reset_index(drop=True)
+    combinado = _normaliza_fluxo(combinado)
+    return combinado
 
 
-# ============================================================================
-# FONTE 3 — Fallback sintético (apenas para não quebrar em dev)
 # ============================================================================
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
 
 
 # ============================================================================
 # IBOVESPA — Yahoo Finance (primário) e brapi (fallback)
 # ============================================================================
 def fetch_ibov_history_yahoo(years: int = 2) -> Optional[pd.DataFrame]:
     try:
         # RANGE por UNIX time (Yahoo chart API)
         end = int(time.time())
         start = end - 60 * 60 * 24 * 365 * years
+        sess = _make_session()
         url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP?period1={start}&period2={end}&interval=1d"
-        r = requests.get(url, timeout=20)
+        r = sess.get(url, timeout=20)
         j = r.json()
         ts = j["chart"]["result"][0]["timestamp"]
         closes = j["chart"]["result"][0]["indicators"]["quote"][0]["close"]
         df = pd.DataFrame({"data": pd.to_datetime(ts, unit="s"), "Ibovespa": closes})
         df = df.dropna().sort_values("data").reset_index(drop=True)
         return df
     except Exception as e:
         logger.warning(f"Falha Yahoo: {e}")
         return None
 
 def fetch_ibov_history_brapi() -> Optional[pd.DataFrame]:
     try:
+        sess = _make_session()
         url = "https://brapi.dev/api/quote/%5EBVSP?range=2y&interval=1d"
-        j = requests.get(url, timeout=20).json()
+        j = sess.get(url, timeout=20).json()
         # brapi muitas vezes devolve no campo 'results'[0]['historicalDataPrice']
         hist = j.get("results", [{}])[0].get("historicalDataPrice", [])
         if not hist:
             return None
         df = pd.DataFrame(hist)
         df["data"] = pd.to_datetime(df["date"], unit="s")
         df = df.rename(columns={"close": "Ibovespa"})[["data", "Ibovespa"]]
         df = df.dropna().sort_values("data").reset_index(drop=True)
         return df
     except Exception as e:
         logger.warning(f"Falha brapi: {e}")
         return None
 
 
-# ============================================================================
-# PIPELINE DE DADOS (com cache)
-# ============================================================================
 def get_last_possible_date() -> date:
     return (pd.Timestamp.today().normalize() - BDay(B3_DELAY_DAYS)).date()
 
 def load_fluxo_raw() -> pd.DataFrame:
     """Dados DIÁRIOS (não acumulados), um por player."""
     # cache
     cached = cache_get("fluxo_raw")
     if cached is not None:
         df = pd.DataFrame(cached)
         df["data"] = pd.to_datetime(df["data"])
-        return df
+        if not df.empty and df["data"].dt.date.max() >= get_last_possible_date():
+            return df
 
     # cadeia de fontes
     df = fetch_fluxo_ddm()
     if df is None:
-        df = fetch_fluxo_csv()
+        start = date.today().replace(month=1, day=1) - timedelta(days=365)
+        df = fetch_fluxo_b3(start, get_last_possible_date())
     if df is None:
         # fallback sintético
         start = date.today().replace(month=1, day=1) - timedelta(days=180)
         df = fetch_fluxo_synthetic(start, date.today())
 
     # guarda cache
     cache_set("fluxo_raw", df.assign(data=df["data"].dt.strftime("%Y-%m-%d")).to_dict(orient="list"))
     return df
 
 def load_ibov_history() -> pd.DataFrame:
     cached = cache_get("ibov_hist")
     if cached is not None:
         df = pd.DataFrame(cached)
         df["data"] = pd.to_datetime(df["data"])
-        return df
+        if not df.empty and df["data"].dt.date.max() >= get_last_possible_date():
+            return df
 
     df = fetch_ibov_history_yahoo(2)
     if df is None:
         df = fetch_ibov_history_brapi()
     if df is None:
         # fallback sintético
         idx = pd.date_range(date.today() - timedelta(days=365), date.today(), freq="D")
         serie = 120_000 + np.cumsum(np.random.normal(0, 120, len(idx)))
         df = pd.DataFrame({"data": idx, "Ibovespa": serie})
 
     cache_set("ibov_hist", df.assign(data=df["data"].dt.strftime("%Y-%m-%d")).to_dict(orient="list"))
     return df
 
 def compute_ytd(df_daily: pd.DataFrame, end_date: date) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
     """Transforma DIÁRIO em ACUMULADO YTD + resumos."""
     start_ytd = date(end_date.year, 1, 1)
 
     # recorte do ano corrente
     dfd = df_daily[(df_daily["data"].dt.date >= start_ytd) & (df_daily["data"].dt.date <= end_date)].copy()
 
     # acumula (cumsum) por coluna de player
     for c in CATEGORIAS:
         if c in dfd.columns:
             dfd[c] = dfd[c].fillna(0).cumsum()
 
     dfd = dfd.sort_values("data").reset_index(drop=True)
 
-    resumo_cards = {c: float(dfd[c].dropna().iloc[-1]) if c in dfd.columns and len(dfd) else 0.0 for c in CATEGORIAS}
+    resumo_cards = {}
+    for c in CATEGORIAS:
+        if c in dfd.columns and len(dfd):
+            serie = dfd[c].dropna()
+            resumo_cards[c] = round(float(serie.iloc[-1]), 2) if not serie.empty else 0.0
+        else:
+            resumo_cards[c] = 0.0
 
-    # movimentação do último dia (delta diário — usa df_daily)
+    # movimentação do último dia (valor líquido diário)
     dl = df_daily[(df_daily["data"].dt.date <= end_date)].copy().sort_values("data")
     mov_dia = {c: 0.0 for c in CATEGORIAS}
-    if len(dl) >= 2:
-        last = dl.iloc[-1][CATEGORIAS]
-        prev = dl.iloc[-2][CATEGORIAS]
-        mov_dia = {c: float(last[c] - prev[c]) for c in CATEGORIAS}
+    if len(dl) >= 1:
+        last = dl.iloc[-1]
+        mov_dia = {c: round(float(last.get(c, 0.0)), 2) for c in CATEGORIAS}
 
-    # mês anterior (a partir do diário)
+    # mês anterior (soma do mês calendário imediatamente anterior ao end_date)
     mov_mes = {c: 0.0 for c in CATEGORIAS}
     if len(dl) > 0:
         last_date = dl["data"].dt.date.max()
         ref = (last_date.replace(day=1) - timedelta(days=1))
         month_start = ref.replace(day=1)
-        m = dl[(dl["data"].dt.date >= month_start) & (dl["data"].dt.date <= ref)]
-        if len(m) >= 2:
-            delta = m.iloc[-1][CATEGORIAS] - m.iloc[0][CATEGORIAS]
-            mov_mes = {c: float(delta[c]) for c in CATEGORIAS}
+        mask_prev_month = dl["data"].dt.date.between(month_start, ref)
+        m = dl[mask_prev_month]
+        if len(m) > 0:
+            mov_mes = {c: round(float(m[c].fillna(0.0).sum()), 2) for c in CATEGORIAS if c in m.columns}
+            for c in CATEGORIAS:
+                mov_mes.setdefault(c, 0.0)
 
     return dfd, resumo_cards, mov_dia, mov_mes
 
 
 # ============================================================================
 # FORMATAÇÃO
 # ============================================================================
 def thousand_dot(x, pos=None):
     try:
         return f"{int(x):,}".replace(",", ".")
     except Exception:
         return str(x)
 
 
+def format_number_br(value: float, decimals: int = 2) -> str:
+    formatted = f"{value:,.{decimals}f}"
+    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")
+
+
+def format_currency_br(value: float, decimals: int = 1) -> str:
+    sinal = "-" if value < 0 else ""
+    return f"{sinal}R$ {format_number_br(abs(value), decimals)}"
+
+
 # ============================================================================
 # PLOTS
 # ============================================================================
 def plot_linhas_ytd(df_ytd: pd.DataFrame, df_ibov: pd.DataFrame) -> str:
+    if df_ytd.empty or df_ibov.empty:
+        return ""
+
     fig, ax1 = plt.subplots(figsize=(14, 6.7), dpi=150)
     fig.patch.set_facecolor("#0b1220")
     ax1.set_facecolor("#0b1220")
     ax2 = ax1.twinx()
 
     # séries de fluxo (acumulado YTD)
     for col in CATEGORIAS:
         if col in df_ytd.columns:
             ax1.plot(df_ytd["data"], df_ytd[col], label=col, linewidth=2)
 
     # ibovespa
     ax2.plot(df_ibov["data"], df_ibov["Ibovespa"], linestyle=":", linewidth=2.2, color="white", label="Ibovespa (pontilhado)")
 
     # eixo X — semanal até 6m, senão mensal
-    days = (df_ytd["data"].max() - df_ytd["data"].min()).days
+    days = (df_ytd["data"].max() - df_ytd["data"].min()).days if len(df_ytd) else 0
     if days <= 185:
         ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
         ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
     else:
         ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
         ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
 
     # Y esquerda: acumulado (R$ bi), passo 5
     ax1.yaxis.set_major_locator(mticker.MultipleLocator(5))
     ax1.set_ylabel("Acumulado (R$ bilhões)", color="white", labelpad=8)
 
     # Y direita: Ibovespa (pts), passo 2.500, com margem
     mn, mx = float(df_ibov["Ibovespa"].min()), float(df_ibov["Ibovespa"].max())
     lo = 2500 * math.floor(mn / 2500) - 2500
+    lo = max(0, lo)
     hi = 2500 * math.ceil (mx / 2500) + 2500
     ax2.set_ylim(lo, hi)
     ax2.yaxis.set_major_locator(mticker.MultipleLocator(2500))
     ax2.yaxis.set_major_formatter(mticker.FuncFormatter(thousand_dot))
     ax2.set_ylabel("Ibovespa (pts)", color="white", labelpad=8)
 
     # estilo
     for sp in ["top", "right", "bottom", "left"]:
         ax1.spines[sp].set_color("#233148")
         ax2.spines[sp].set_color("#233148")
     ax1.tick_params(axis="x", colors="white")
     ax1.tick_params(axis="y", colors="white")
     ax2.tick_params(axis="y", colors="white")
 
     # legenda
     l1, lb1 = ax1.get_legend_handles_labels()
     l2, lb2 = ax2.get_legend_handles_labels()
     leg = ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", frameon=False, fontsize=9)
     for t in leg.get_texts():
         t.set_color("white")
 
     # marca d'água
-    fig.text(0.5, 0.5, "@alan_richard", fontsize=28, color="gray", alpha=0.06, ha="center", va="center")
+    fig.text(0.5, 0.5, "@alan_richard", fontsize=28, color="gray", alpha=0.08, ha="center", va="center")
 
     plt.tight_layout()
     buf = io.BytesIO()
     plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
     buf.seek(0)
     encoded = base64.b64encode(buf.read()).decode("utf-8")
     plt.close(fig)
     return encoded
 
 
 def plot_estrangeiro_30dias(df_daily: pd.DataFrame, end_date: date) -> str:
-    """Barras centradas no zero (últimos 30 dias), com MM(28)."""
+    """Barras verticais com MM curta (últimos 30 pregões)."""
     d = df_daily[df_daily["data"].dt.date <= end_date].copy().sort_values("data")
+    d = d.tail(60)  # garante janela maior antes do recorte final
     d["estrangeiro"] = d["Estrangeiro"].fillna(0.0)
-    d["mm28"] = d["estrangeiro"].rolling(28).mean()
-    d = d.tail(60)  # pega 60 e depois corta 30 úteis mais recentes
+    d["mm10"] = d["estrangeiro"].rolling(10, min_periods=1).mean()
     d = d.tail(30)
 
+    if d.empty:
+        return ""
+
     fig, ax = plt.subplots(figsize=(14, 6.0), dpi=150)
     fig.patch.set_facecolor("#0b1220")
     ax.set_facecolor("#0b1220")
 
-    # barras horizontais com cores por sinal
+    idx = np.arange(len(d))
     colors = ["#22c55e" if v >= 0 else "#ef4444" for v in d["estrangeiro"]]
-    ax.barh(d["data"].dt.strftime("%d/%m"), d["estrangeiro"], color=colors)
-    ax.invert_yaxis()
+    bars = ax.bar(idx, d["estrangeiro"], color=colors, width=0.65, label="Fluxo diário")
+
+    ax.plot(idx, d["mm10"], color="#f59e0b", linewidth=2.0, marker="o", markersize=3, label="Média móvel 10 pregões")
+
+    # Rótulos nas barras
+    for rect, val in zip(bars, d["estrangeiro"]):
+        offset = 6 if val >= 0 else -10
+        va = "bottom" if val >= 0 else "top"
+        value_str = format_number_br(val, 2)
+        if val > 0:
+            value_str = f"+{value_str}"
+        ax.annotate(
+            value_str,
+            xy=(rect.get_x() + rect.get_width() / 2, val),
+            xytext=(0, offset),
+            textcoords="offset points",
+            ha="center",
+            va=va,
+            fontsize=8,
+            color="#e2e8f0",
+        )
 
-    # média móvel
-    ax2 = ax.twiny()
-    ax2.plot(d["mm28"], d["data"].dt.strftime("%d/%m"), color="#f59e0b", linewidth=2.0)
-
-    # eixo X centralizado no zero
     max_abs = max(abs(d["estrangeiro"].min()), abs(d["estrangeiro"].max()), 0.5)
-    ax.set_xlim(-max_abs, max_abs)
-    ax.xaxis.set_major_locator(mticker.MultipleLocator(max(0.25, round(max_abs/6, 2))))
-    ax.set_xlabel("R$ bi (diário)", color="white")
-
-    for a in (ax, ax2):
-        a.tick_params(colors="white")
-        for sp in ["top", "right", "bottom", "left"]:
-            try:
-                a.spines[sp].set_color("#233148")
-            except Exception:
-                pass
+    ax.set_ylim(-max_abs * 1.15, max_abs * 1.2)
+
+    ax.set_xticks(idx)
+    ax.set_xticklabels(d["data"].dt.strftime("%d/%m"), rotation=45, ha="right")
+    step = max(0.1, round(max_abs / 5, 2))
+    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
+
+    ax.tick_params(colors="white")
+    ax.set_ylabel("R$ bilhões", color="white")
+    ax.set_xlabel("Data", color="white")
+    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: format_number_br(v, 2)))
+
+    for spine in ["top", "right", "bottom", "left"]:
+        ax.spines[spine].set_color("#233148")
+
+    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
+    for text in legend.get_texts():
+        text.set_color("white")
+
+    fig.text(0.5, 0.5, "@alan_richard", fontsize=26, color="gray", alpha=0.08, ha="center", va="center")
 
     plt.tight_layout()
     buf = io.BytesIO()
     plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
     buf.seek(0)
     encoded = base64.b64encode(buf.read()).decode("utf-8")
     plt.close(fig)
     return encoded
 
 
 # ============================================================================
 # ROTAS
 # ============================================================================
 @app.route("/refresh")
 def refresh():
     # limpa caches e redireciona
     cache_clear()
     return redirect(url_for("home"))
 
 @app.route("/", methods=["GET"])
 def home():
-    last_possible = get_last_possible_date()
-
     # carrega fontes
     df_daily = load_fluxo_raw()               # DIÁRIO por player
     df_ibov_hist = load_ibov_history()        # histórico IBOV
 
+    reference_date = get_last_possible_date()
+    if not df_daily.empty:
+        reference_date = df_daily["data"].dt.date.max()
+
     # garante recorte do ibov até last_possible
-    df_ibov = df_ibov_hist[df_ibov_hist["data"].dt.date <= last_possible].copy()
+    df_ibov = df_ibov_hist[df_ibov_hist["data"].dt.date <= reference_date].copy()
     if df_ibov.empty:
         df_ibov = df_ibov_hist.copy()
     df_ibov = df_ibov.sort_values("data").reset_index(drop=True)
+    if not df_ibov.empty:
+        start_ytd = date(reference_date.year, 1, 1)
+        df_ibov = df_ibov[df_ibov["data"].dt.date >= start_ytd].reset_index(drop=True)
 
     # transforma diário -> acumulado YTD + resumos
-    df_ytd, resumo_cards, mov_dia, mov_mes = compute_ytd(df_daily, last_possible)
+    df_ytd, resumo_cards, mov_dia, mov_mes = compute_ytd(df_daily, reference_date)
 
     # gráficos
     img_linhas = plot_linhas_ytd(df_ytd, df_ibov)
-    img_barras = plot_estrangeiro_30dias(df_daily, last_possible)
+    img_barras = plot_estrangeiro_30dias(df_daily, reference_date)
 
     # resumo textual
     ld = pd.Series(mov_dia)
     comprador = ld.idxmax(); vcomp = ld.max()
     vendedor  = ld.idxmin(); vvend = ld.min()
     resumo_dia_txt = (
-        f"Maior comprador: {comprador} ({'+' if vcomp>=0 else '–'} R$ {abs(vcomp):.1f} bi) • "
-        f"Maior vendedor: {vendedor} ({'+' if vvend>=0 else '–'} R$ {abs(vvend):.1f} bi)"
+        f"Maior comprador: {comprador} ({'+' if vcomp>=0 else '–'}{format_currency_br(abs(vcomp), 2)} bi) • "
+        f"Maior vendedor: {vendedor} ({'+' if vvend>=0 else '–'}{format_currency_br(abs(vvend), 2)} bi)"
     )
 
     # ibov do dia
     ibov_close = df_ibov["Ibovespa"].iloc[-1] if len(df_ibov) else np.nan
     ibov_prev = df_ibov["Ibovespa"].iloc[-2] if len(df_ibov) > 1 else ibov_close
     var = (ibov_close - ibov_prev) if pd.notna(ibov_close) and pd.notna(ibov_prev) else 0.0
-    ibov_txt = f"Ibovespa: {int(ibov_close):,}".replace(",", ".") + f" ({'+' if var>=0 else '–'}{abs(var):.0f} pts no dia)" if pd.notna(ibov_close) else "-"
+    if pd.notna(ibov_close):
+        ibov_txt = (
+            f"Ibovespa: {format_number_br(ibov_close, 2)} pts "
+            f"({'+' if var>=0 else '–'}{format_number_br(abs(var), 2)} pts no dia)"
+        )
+    else:
+        ibov_txt = "-"
 
     # última data conhecida
-    last_date_str = last_possible.strftime("%d/%m/%Y")
+    last_date_str = reference_date.strftime("%d/%m/%Y") if df_daily.size else "-"
 
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
 
 
 # ============================================================================
 # MAIN (local)
 # ============================================================================
 if __name__ == "__main__":
     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
