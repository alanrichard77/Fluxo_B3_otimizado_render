import pandas as pd
import numpy as np
from datetime import date

PLAYERS_ORDER = ["Estrangeiro", "Institucional", "Pessoa Física", "Inst. Financeira", "Outros"]

def _clip_range(df, start, end, date_col="date"):
    df = df.copy()
    df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    return df

def build_accumulated_df(players_df, start, end):
    df = _clip_range(players_df, start, end)
    # pivô: linhas data, colunas player, valores net, acumulado por coluna
    pivot = df.pivot_table(index="date", columns="player", values="net", aggfunc="sum").fillna(0)
    acc = pivot.cumsum()
    # ordenar colunas
    cols = [c for c in PLAYERS_ORDER if c in acc.columns] + [c for c in acc.columns if c not in PLAYERS_ORDER]
    acc = acc[cols]
    acc = acc.reset_index()
    return acc

def build_daily_table(players_df, start, end):
    df = _clip_range(players_df, start, end)
    # tabela diária, por player
    tbl = df.groupby(["date", "player"], as_index=False)["net"].sum()
    # garante presença de todos os players por dia com 0
    all_days = pd.DataFrame({"date": sorted(tbl["date"].unique())})
    frames = []
    for pl in sorted(tbl["player"].unique()):
        s = all_days.merge(tbl[tbl["player"] == pl], on="date", how="left")
        s["player"].fillna(pl, inplace=True)
        s["net"].fillna(0.0, inplace=True)
        frames.append(s)
    full = pd.concat(frames, ignore_index=True)
    return full

def build_monthly_leaders(daily_df):
    d = daily_df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    ag = d.groupby(["month", "player"], as_index=False)["net"].sum()
    # maior comprador e maior vendedor por mês
    winners = ag.loc[ag.groupby("month")["net"].idxmax()].rename(columns={"player": "maior_comprador", "net": "saldo_maior_compra"})
    losers = ag.loc[ag.groupby("month")["net"].idxmin()].rename(columns={"player": "maior_vendedor", "net": "saldo_maior_venda"})
    res = winners.merge(losers[["month", "maior_vendedor", "saldo_maior_venda"]], on="month", how="left")
    return res

def build_heatmap_df(daily_df):
    d = daily_df.copy()
    # Heatmap: linhas player, colunas data, valores net
    hm = d.pivot_table(index="player", columns="date", values="net", aggfunc="sum").fillna(0)
    # ordenação dos players
    rows = [p for p in PLAYERS_ORDER if p in hm.index] + [i for i in hm.index if i not in PLAYERS_ORDER]
    hm = hm.loc[rows]
    return hm

