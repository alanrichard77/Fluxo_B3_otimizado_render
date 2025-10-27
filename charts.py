import plotly.graph_objects as go
import numpy as np
import pandas as pd

WM = "@alan_richard"

def _watermark(fig):
    fig.add_annotation(
        text=WM,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=28),
        opacity=0.12
    )
    return fig

def _layout_base(fig, title=""):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0f1629",
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        font=dict(size=12)
    )
    return fig

def fig_main_accumulated_with_ibov(acc_df, ibov_df):
    fig = go.Figure()
    # linhas acumuladas por player
    for col in acc_df.columns:
        if col == "date":
            continue
        fig.add_trace(go.Scatter(
            x=acc_df["date"], y=acc_df[col], mode="lines",
            name=col
        ))
    # eixo secundário para IBOV normalizado
    if ibov_df is not None and not ibov_df.empty:
        merged = pd.DataFrame({"date": acc_df["date"]}).merge(ibov_df, on="date", how="left").ffill()
        ref = merged["close"].dropna()
        if len(ref) > 0:
            base = ref.iloc[0]
            ibov_norm = (merged["close"] / base) * (acc_df.select_dtypes(include=[float, int]).to_numpy().max() * 0.9)
            fig.add_trace(go.Scatter(
                x=acc_df["date"], y=ibov_norm, mode="lines",
                name="Ibovespa, eixo normalizado", line=dict(dash="dot")
            ))
        fig.update_yaxes(title_text="Acumulado, R$ bi", secondary_y=False)

    _layout_base(fig)
    _watermark(fig)
    return fig

def fig_daily_bars_by_player(daily_df):
    # barras empilhadas por player
    piv = daily_df.pivot_table(index="date", columns="player", values="net", aggfunc="sum").fillna(0)
    fig = go.Figure()
    for col in piv.columns:
        fig.add_trace(go.Bar(x=piv.index, y=piv[col], name=col))
    _layout_base(fig, title="")
    fig.update_layout(barmode="relative")
    _watermark(fig)
    return fig

def fig_monthly_leaders(leaders_df):
    if leaders_df is None or leaders_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=leaders_df["month"], y=leaders_df["saldo_maior_compra"], name="Maior comprador"))
    fig.add_trace(go.Bar(x=leaders_df["month"], y=leaders_df["saldo_maior_venda"], name="Maior vendedor"))
    _layout_base(fig, title="")
    fig.update_layout(barmode="group")
    _watermark(fig)
    return fig

def fig_daily_heatmap(hm_df):
    if hm_df is None or hm_df.empty:
        return go.Figure()
    z = hm_df.values
    fig = go.Figure(data=go.Heatmap(
        z=z, x=hm_df.columns, y=hm_df.index,
        colorbar=dict(title="R$ bi")
    ))
    _layout_base(fig, title="")
    _watermark(fig)
    return fig

