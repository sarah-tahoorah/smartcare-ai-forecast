import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=DM+Sans:wght@400;600&display=swap');

        :root {
            --bg: #0b0f17;
            --panel: #111827;
            --panel-2: #0f172a;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #22d3ee;
            --accent-2: #4ade80;
            --danger: #fb7185;
            --warning: #f59e0b;
            --border: #1f2937;
        }

        html, body, [class*="css"]  {
            font-family: 'DM Sans', sans-serif;
            color: var(--text);
            background-color: var(--bg);
        }

        .stApp {
            background: radial-gradient(1200px 600px at 20% -10%, #0ea5e9 0%, transparent 50%),
                        radial-gradient(900px 600px at 120% 10%, #22d3ee 0%, transparent 45%),
                        linear-gradient(180deg, #0b0f17 0%, #0b0f17 100%);
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.2px;
        }

        .sidebar .sidebar-content {
            background: var(--panel-2);
        }

        .kpi-card {
            background: linear-gradient(135deg, rgba(34,211,238,0.08), rgba(15,23,42,0.9));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .kpi-title {
            color: var(--muted);
            font-size: 12px;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .kpi-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--text);
        }

        .kpi-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: rgba(34,211,238,0.15);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--accent);
            font-weight: 700;
            font-size: 14px;
            margin-right: 10px;
        }

        .section-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        }

        .alert-box {
            border-radius: 14px;
            padding: 14px 16px;
            border: 1px solid var(--border);
            margin-bottom: 10px;
            background: rgba(251, 113, 133, 0.08);
        }

        .alert-title {
            color: var(--danger);
            font-weight: 700;
            margin-bottom: 4px;
        }

        .muted {
            color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title: str, value: str, icon_text: str):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div style="display:flex; align-items:center;">
            <div class="kpi-icon">{icon_text}</div>
            <div>
              <div class="kpi-title">{title}</div>
              <div class="kpi-value">{value}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def indicator_gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"color": "#e5e7eb"}},
            gauge={
                "axis": {"range": [None, value * 1.4]},
                "bar": {"color": "#22d3ee"},
                "bgcolor": "#0f172a",
                "borderwidth": 1,
                "bordercolor": "#1f2937",
            },
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=260, paper_bgcolor="#0f172a")
    return fig


def line_chart(df: pd.DataFrame, x: str, y: list, title: str) -> go.Figure:
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_traces(line=dict(width=2.6))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def heatmap_calendar(pivot: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Teal",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=360,
        title=title,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=y, title=title, color_continuous_scale="teal")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def forecast_chart(history: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["children_in_care"],
            name="Historical",
            mode="lines",
            line=dict(color="#60a5fa", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["pred"],
            name="Forecast",
            mode="lines",
            line=dict(color="#22d3ee", width=2.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["hi"],
            name="Upper",
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["lo"],
            name="Confidence",
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            fillcolor="rgba(34,211,238,0.18)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
