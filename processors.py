import pandas as pd
import numpy as np

PLAYERS_ORDER = ["Estrangeiro", "Institucional", "Pessoa Física", "Inst. Financeira", "Outros"]


def _clip_range(df, start, end, date_col="date"):
    df = df.copy()
    if df.empty:
        return df
    return df[(df[date_col] >= start) & (df[date_col] <= end)]


def build_accumulated_df(players_df, start, end):
    df = _clip_range(players_df, start, end)
    pivot = df.pivot_table(index="date", columns="player", values="net", aggfunc="sum").fillna(0)
    acc = pivot.cumsum()
    cols = [c for c in PLAYERS_ORDER if c in acc.columns] + [c for c in acc.columns if c not in PLAYERS_ORDER]
    return acc[cols].reset_index()


def build_daily_table(players_df, start, end):
    df = _clip_range(players_df, start, end)
    tbl = df.groupby(["date", "player"], as_index=False)["net"].sum()
    all_days = pd.DataFrame({"date": sorted(tbl["date"].unique())}) if not tbl.empty else pd.DataFrame(columns=["date"])
    frames = []
    for pl in sorted(tbl["player"].unique()):
        s = all_days.merge(tbl[tbl["player"] == pl], on="date", how="left")
        s["player"].fillna(pl, inplace=True)
        s["net"].fillna(0.0, inplace=True)
        frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else tbl


def build_monthly_leaders(daily_df):
    d = daily_df.copy()
    if d.empty:
        return d
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    ag = d.groupby(["month", "player"], as_index=False)["net"].sum()
    winners = ag.loc[ag.groupby("month")["net"].idxmax()].rename(columns={"player": "maior_comprador", "net": "saldo_maior_compra"})
    losers = ag.loc[ag.groupby("month")["net"].idxmin()].rename(columns={"player": "maior_vendedor", "net": "saldo_maior_venda"})
    return winners.merge(losers[["month", "maior_vendedor", "saldo_maior_venda"]], on="month", how="left")


def build_heatmap_df(daily_df):
    d = daily_df.copy()
    if d.empty:
        return d
    hm = d.pivot_table(index="player", columns="date", values="net", aggfunc="sum").fillna(0)
    rows = [p for p in PLAYERS_ORDER if p in hm.index] + [i for i in hm.index if i not in PLAYERS_ORDER]
    return hm.loc[rows]
