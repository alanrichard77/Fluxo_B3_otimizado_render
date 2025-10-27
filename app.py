import os
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request

from cache_utils import cache_get, cache_set
import data_sources as ds  # import do módulo inteiro para evitar ImportError
from processors import (
    build_accumulated_df,
    build_daily_table,
    build_monthly_leaders,
    build_heatmap_df,
)
from charts import (
    fig_main_accumulated_with_ibov,
    fig_daily_bars_by_player,
    fig_monthly_leaders,
    fig_daily_heatmap,
)
from utils import to_plotly_json, parse_date

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fluxo_b3")

# ===== Defaults =====
DEFAULT_DAYS = 180  # janela padrão
CACHE_TTL = 60 * 60  # 1 hora


# ===== Wrappers seguros (não quebram a app caso data_sources falhe) =====
def _safe_cards():
    try:
        if hasattr(ds, "get_daily_flows_cards"):
            return ds.get_daily_flows_cards()
    except Exception:
        logger.exception("safe_cards falhou")
    zero = {"valor": 0.0, "texto": "Sem dados recentes", "formatado": "R$ 0,0bi"}
    return {"ok": True, "cards": {
        "Estrangeiro": zero, "Institucional": zero, "Pessoa Física": zero,
        "Inst. Financeira": zero, "Outros": zero
    }}


def _safe_series_by_player():
    import pandas as pd
    try:
        if hasattr(ds, "get_series_by_player"):
            return ds.get_series_by_player()
    except Exception:
        logger.exception("safe_series_by_player falhou")
    return pd.DataFrame(columns=["date", "player", "net"])


def _safe_ibov():
    try:
        if hasattr(ds, "get_ibov_series"):
            return ds.get_ibov_series()
    except Exception as e:
        logger.warning("safe_ibov falhou: %s", e)
    return None


# ===== Rotas =====
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/refresh", methods=["POST"])
def refresh():
    cache_set("last_refresh", datetime.utcnow().isoformat(), ttl=5)
    return jsonify({"ok": True, "msg": "Atualização solicitada com sucesso."})


@app.route("/api/cards")
def api_cards():
    try:
        cache_key = "cards_v1"
        cached = cache_get(cache_key)
        if cached:
            return jsonify(cached)
        cards = _safe_cards()
        cache_set(cache_key, cards, ttl=CACHE_TTL)
        return jsonify(cards)
    except Exception:
        logger.exception("Erro em /api/cards")
        return jsonify(_safe_cards())


@app.route("/api/series")
def api_series():
    try:
        start = parse_date(request.args.get("start"))
        end = parse_date(request.args.get("end"))
        days = int(request.args.get("days", DEFAULT_DAYS))
        if not end:
            end = datetime.utcnow().date()
        if not start:
            start = end - timedelta(days=days)

        cache_key = f"series_v1_{start}_{end}"
        cached = cache_get(cache_key)
        if cached:
            return jsonify(cached)

        players_df = _safe_series_by_player()
        ibov = _safe_ibov()

        # Sem dados: devolve estrutura neutra, sem 500
        if players_df is None or players_df.empty:
            resp = {
                "ok": True, "start": str(start), "end": str(end),
                "fig_main": {"data": [], "layout": {}},
                "fig_daily": {"data": [], "layout": {}},
                "fig_leaders": {"data": [], "layout": {}},
                "fig_heat": {"data": [], "layout": {}},
                "tables": {"daily": [], "leaders": []},
                "message": "Sem dados recentes para o intervalo selecionado."
            }
            cache_set(cache_key, resp, ttl=300)
            return jsonify(resp)

        # Processamento
        acc_df = build_accumulated_df(players_df, start, end)
        daily_df = build_daily_table(players_df, start, end)
        hm_df = build_heatmap_df(daily_df)
        leaders_df = build_monthly_leaders(daily_df)

        # Gráficos
        fig_main = fig_main_accumulated_with_ibov(acc_df, ibov)
        fig_daily = fig_daily_bars_by_player(daily_df)
        fig_leaders = fig_monthly_leaders(leaders_df)
        fig_heat = fig_daily_heatmap(hm_df)

        resp = {
            "ok": True,
            "start": str(start),
            "end": str(end),
            "fig_main": to_plotly_json(fig_main),
            "fig_daily": to_plotly_json(fig_daily),
            "fig_leaders": to_plotly_json(fig_leaders),
            "fig_heat": to_plotly_json(fig_heat),
            "tables": {
                "daily": daily_df.reset_index(drop=True).to_dict(orient="records"),
                "leaders": leaders_df.reset_index(drop=True).to_dict(orient="records"),
            },
        }
        cache_set(cache_key, resp, ttl=CACHE_TTL)
        return jsonify(resp)
    except Exception:
        logger.exception("Erro em /api/series")
        return jsonify({
            "ok": True,
            "fig_main": {"data": [], "layout": {}},
            "fig_daily": {"data": [], "layout": {}},
            "fig_leaders": {"data": [], "layout": {}},
            "fig_heat": {"data": [], "layout": {}},
            "tables": {"daily": [], "leaders": []},
            "message": "Falha temporária ao coletar dados. Tente novamente mais tarde."
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
