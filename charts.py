import plotly.graph_objects as go
import pandas as pd

def _empty_fig():
    fig = go.Figure()
    # layout transparente, o JS já injeta anotação "Sem dados recentes" se vier vazio
    fig.update_layout(
        template={},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7ff"),
        margin=dict(l=50, r=10, t=10, b=40),
    )
    return fig

def fig_main_accumulated_with_ibov(acc_tuple):
    acc_players, ibov_norm = acc_tuple
    fig = go.Figure()

    if acc_players is not None and not acc_players.empty:
        for col in acc_players.columns:
            fig.add_trace(
                go.Scatter(
                    x=acc_players.index,
                    y=acc_players[col],
                    mode="lines",
                    name=col,
                )
            )

    if ibov_norm is not None and not ibov_norm.empty:
        fig.add_trace(
            go.Scatter(
                x=ibov_norm.index,
                y=ibov_norm["value"],
                mode="lines",
                name="Ibovespa (normalizado)",
                line=dict(dash="dot"),
                yaxis="y2",
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7ff"),
        margin=dict(l=50, r=60, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Acumulado (R$ bi)"),
        yaxis2=dict(title="Ibov (base 100)", overlaying="y", side="right", showgrid=False),
    )

    return fig if (fig.data and len(fig.data) > 0) else _empty_fig()

def fig_daily_bars_by_player(daily_df: pd.DataFrame):
    if daily_df is None or daily_df.empty:
        return _empty_fig()

    fig = go.Figure()
    for col in daily_df.columns:
        fig.add_trace(
            go.Bar(
                x=daily_df.index,
                y=daily_df[col],
                name=col
            )
        )

    fig.update_layout(
        barmode="relative",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7ff"),
        margin=dict(l=50, r=10, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Diário (R$ bi)"),
    )
    return fig

def fig_monthly_leaders(leaders_df: pd.DataFrame):
    if leaders_df is None or leaders_df.empty:
        return _empty_fig()

    leaders_df = leaders_df.sort_values("month")
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=leaders_df["month"],
            y=leaders_df["top_buyer_value"],
            name="Maior comprador",
        )
    )
    fig.add_trace(
        go.Bar(
            x=leaders_df["month"],
            y=leaders_df["top_seller_value"],
            name="Maior vendedor",
        )
    )

    # rótulos com o nome do player no hover
    fig.update_traces(
        hovertemplate="<b>%{x|%b/%Y}</b><br>%{y:.2f} bi"
    )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7ff"),
        margin=dict(l=50, r=10, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Saldo mensal (R$ bi)"),
    )
    return fig

def fig_daily_heatmap(heat_df: pd.DataFrame):
    if heat_df is None or heat_df.empty:
        return _empty_fig()

    z = heat_df.T.values  # players x dias
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=heat_df.index,
            y=list(heat_df.columns),
            colorbar=dict(title="R$ bi"),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7ff"),
        margin=dict(l=60, r=20, t=10, b=40),
    )
    return fig
