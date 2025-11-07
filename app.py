import logging
import os
from datetime import datetime, timedelta

from flask import Flask, render_template, jsonify, request

from cache_utils import cache_get, cache_set, cache_clear
from data_sources import (
    get_daily_flows_cards,
    get_series_by_player,
    get_ibov_series,
)
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

# ===== Flask =====
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fluxo_b3")

# ===== Defaults =====
DEFAULT_DAYS = 180  # janela padrão
CACHE_TTL = 60 * 60  # 1 hora


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cards")
def api_cards():
    try:
        cache_key = "cards"
        cached = cache_get(cache_key)
        if cached is not None:
            return jsonify(cached)

        cards = get_daily_flows_cards()
        # resposta já vem pronta
        cache_set(cache_key, cards, TTL=CACHE_TTL)
        return jsonify(cards)
    except Exception as e:
        logger.exception("Erro em /api/cards")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/series")
def api_series():
    try:
        # datas
        end_str = request.args.get("end")
        start_str = request.args.get("start")
        if not end_str:
            end = datetime.utcnow().date()
        else:
            end = parse_date(end_str).date()

        if not start_str:
            start = end - timedelta(days=DEFAULT_DAYS)
        else:
            start = parse_date(start_str).date()

        cache_key = f"series:{start.isoformat()}:{end.isoformat()}"
        cached = cache_get(cache_key)
        if cached is not None:
            return jsonify(cached)

        # fontes
        players = get_series_by_player()  # date, player, net
        players = players[(players["date"] >= start) & (players["date"] <= end)]

        ibov = get_ibov_series(period_days=(end - start).days + 10)

        # processamentos
        acc_df = build_accumulated_df(players, ibov)
        daily_df = build_daily_table(players)
        leaders_df = build_monthly_leaders(players)
        heat_df = build_heatmap_df(players)

        # gráficos
        f1 = to_plotly_json(fig_main_accumulated_with_ibov(acc_df))
        f2 = to_plotly_json(fig_daily_bars_by_player(daily_df))
        f3 = to_plotly_json(fig_monthly_leaders(leaders_df))
        f4 = to_plotly_json(fig_daily_heatmap(heat_df))

        payload = {
            "ok": True,
            "fig_main": f1,
            "fig_daily": f2,
            "fig_leaders": f3,
            "fig_heat": f4,
        }
        cache_set(cache_key, payload, TTL=CACHE_TTL)
        return jsonify(payload)
    except Exception as e:
        logger.exception("Erro em /api/series")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    try:
        cache_clear()
        return jsonify({"ok": True, "message": "cache limpo"})
    except Exception as e:
        logger.exception("Erro em /api/refresh")
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
