import pandas as pd
import numpy as np
from datetime import datetime

def _ensure_df(df, cols):
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    return df.copy()

def build_accumulated_df(players_df: pd.DataFrame, ibov_df: pd.DataFrame | None):
    """
    Retorna:
      - acc_players: index=date, col=player, valores acumulados (R$ bi) por player
      - ibov_norm:   index=date, col=value, Ibov normalizado base 100 na 1ª data disponível
    """
    players_df = _ensure_df(players_df, ["date", "player", "net"])
    if not players_df.empty:
        acc = (
            players_df.pivot_table(
                index="date", columns="player", values="net", aggfunc="sum"
            )
            .fillna(0)
            .cumsum()
        )
    else:
        acc = pd.DataFrame()

    ibov_norm = pd.DataFrame()
    if ibov_df is not None and not ibov_df.empty:
        s = ibov_df[["date", "close"]].dropna().copy()
        s = s.sort_values("date")
        s["value"] = s["close"] / float(s["close"].iloc[0]) * 100.0
        ibov_norm = s[["date", "value"]].set_index("date")

    return acc, ibov_norm

def build_daily_table(players_df: pd.DataFrame):
    """
    Tabela diária por player para barras.
    index=date, col=player, valor=net (R$ bi)
    """
    players_df = _ensure_df(players_df, ["date", "player", "net"])
    if players_df.empty:
        return pd.DataFrame()
    tbl = players_df.pivot_table(
        index="date", columns="player", values="net", aggfunc="sum"
    ).fillna(0)
    return tbl

def build_monthly_leaders(players_df: pd.DataFrame):
    """
    Retorna df com colunas:
      month, top_buyer_player, top_buyer_value, top_seller_player, top_seller_value
    """
    players_df = _ensure_df(players_df, ["date", "player", "net"])
    if players_df.empty:
        return pd.DataFrame(columns=[
            "month", "top_buyer_player", "top_buyer_value",
            "top_seller_player", "top_seller_value"
        ])

    temp = players_df.copy()
    temp["month"] = pd.to_datetime(temp["date"]).dt.to_period("M")
    agg = temp.groupby(["month", "player"])["net"].sum().reset_index()

    rows = []
    for m, g in agg.groupby("month"):
        g = g.sort_values("net")
        top_seller = g.iloc[0]
        top_buyer = g.iloc[-1]
        rows.append({
            "month": m.to_timestamp(),
            "top_buyer_player": top_buyer["player"],
            "top_buyer_value": top_buyer["net"],
            "top_seller_player": top_seller["player"],
            "top_seller_value": top_seller["net"],
        })
    return pd.DataFrame(rows)

def build_heatmap_df(players_df: pd.DataFrame):
    """
    Heatmap: linhas = datas, colunas = players, valores = net (R$ bi)
    """
    return build_daily_table(players_df)
