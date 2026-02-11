"""
Visualization utilities for the fire risk dashboard.

Provides consistent styling, color scales, and reusable chart components
built on Plotly and Folium.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# ─── Color Palette ──────────────────────────────────────────────────────
COLORS = {
    "bg": "#0B0E11",
    "surface": "#131820",
    "surface_light": "#1A2233",
    "border": "#2A3548",
    "text": "#E8ECF1",
    "text_muted": "#7B8DA4",
    "accent": "#FF6B35",
    "accent_light": "#FF8C5E",
    "danger": "#FF3B4E",
    "warning": "#FFAA2B",
    "safe": "#2ECC71",
    "blue": "#3498DB",
    "purple": "#9B59B6",
}

RISK_COLORS = ["#2ECC71", "#FF6B35", "#FFAA2B", "#FF3B4E"]
RISK_LABELS = ["Low", "Moderate", "High", "Critical"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["surface"],
    font=dict(family="JetBrains Mono, Fira Code, monospace", color=COLORS["text"], size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
)


def risk_color(score: float) -> str:
    """Return hex color for a risk score (0–1)."""
    if score >= 0.75:
        return COLORS["danger"]
    elif score >= 0.50:
        return COLORS["warning"]
    elif score >= 0.25:
        return COLORS["accent"]
    return COLORS["safe"]


def risk_label(score: float) -> str:
    if score >= 0.75:
        return "Critical"
    elif score >= 0.50:
        return "High"
    elif score >= 0.25:
        return "Moderate"
    return "Low"


def make_feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of feature importance."""
    df = importance_df.head(top_n).iloc[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["importance_permutation"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["importance_permutation"],
            colorscale=[[0, COLORS["accent"]], [1, COLORS["danger"]]],
        ),
        error_x=dict(type="data", array=df["importance_perm_std"], color=COLORS["text_muted"]),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Feature Importance (Permutation)",
        xaxis_title="Importance (mean decrease in R²)",
        yaxis_title="",
        height=350,
        showlegend=False,
    )
    return fig


def make_monthly_chart(monthly_data: list | np.ndarray, title: str = "Monthly Distribution") -> go.Figure:
    """Bar chart of monthly incident distribution."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    values = list(monthly_data)[:12]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months,
        y=values,
        marker=dict(
            color=values,
            colorscale=[[0, COLORS["blue"]], [0.5, COLORS["accent"]], [1, COLORS["danger"]]],
        ),
        hovertemplate="<b>%{x}</b>: %{y} incidents<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title="",
        yaxis_title="Incidents",
        height=280,
        showlegend=False,
    )
    return fig


def make_actual_vs_predicted_chart(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Scatter plot of actual vs predicted values."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual, y=predicted,
        mode="markers",
        marker=dict(
            color=predicted, colorscale=[[0, COLORS["safe"]], [0.5, COLORS["warning"]], [1, COLORS["danger"]]],
            size=7, opacity=0.7, line=dict(width=0.5, color=COLORS["border"]),
        ),
        hovertemplate="Actual: %{x}<br>Predicted: %{y:.1f}<extra></extra>",
    ))

    # Perfect prediction line
    max_val = max(actual.max(), predicted.max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color=COLORS["text_muted"], dash="dash", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Actual vs. Predicted Structural Fires",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=380,
        showlegend=False,
    )
    return fig


def make_risk_distribution_chart(risk_scores: np.ndarray) -> go.Figure:
    """Histogram of risk score distribution."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=risk_scores,
        nbinsx=30,
        marker=dict(
            color=COLORS["accent"],
            line=dict(color=COLORS["border"], width=0.5),
        ),
        hovertemplate="Risk: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    # Add risk threshold lines
    for threshold, label, color in [
        (0.25, "Moderate", COLORS["accent"]),
        (0.50, "High", COLORS["warning"]),
        (0.75, "Critical", COLORS["danger"]),
    ]:
        fig.add_vline(x=threshold, line_dash="dash", line_color=color,
                       annotation_text=label, annotation_font_color=color)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Count",
        height=300,
        showlegend=False,
    )
    return fig


def make_borough_comparison_chart(boro_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing boroughs."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Structural Fires",
        x=boro_df["borough"],
        y=boro_df["structural_fires"],
        marker_color=COLORS["danger"],
    ))
    fig.add_trace(go.Bar(
        name="Total Incidents",
        x=boro_df["borough"],
        y=boro_df["total_incidents"],
        marker_color=COLORS["blue"],
        opacity=0.5,
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Borough Comparison",
        barmode="group",
        height=350,
        legend=dict(font=dict(size=10)),
    )
    return fig


def create_folium_map(features_df, risk_scores, label_col, center=None, zoom=11):
    """Create a Folium map with risk-colored circle markers."""
    import folium
    from branca.colormap import LinearColormap

    if center is None:
        center = [40.72, -73.95]

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
    )

    colormap = LinearColormap(
        colors=["#2ECC71", "#FF6B35", "#FFAA2B", "#FF3B4E"],
        vmin=0, vmax=1,
        caption="Fire Risk Score",
    )

    for idx, row in features_df.iterrows():
        lat = row.get("latitude")
        lng = row.get("longitude")
        if pd.isna(lat) or pd.isna(lng):
            continue

        risk = risk_scores[idx] if idx < len(risk_scores) else 0
        label = row.get(label_col, str(idx))
        fires = row.get("structural_fires", 0)
        total = row.get("total_incidents", 0)

        radius = 5 + risk * 20

        popup_html = f"""
        <div style="font-family:monospace;font-size:12px;min-width:150px">
            <b>{label}</b><br>
            Risk: <span style="color:{risk_color(risk)}">{risk_label(risk)} ({risk:.2f})</span><br>
            Structural Fires: {fires}<br>
            Total Incidents: {total}<br>
            Fire Rate: {row.get('structural_fire_rate', 0):.1%}
        </div>
        """

        folium.CircleMarker(
            location=[lat, lng],
            radius=radius,
            color=risk_color(risk),
            fill=True,
            fill_color=risk_color(risk),
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{label} — {risk_label(risk)} Risk",
        ).add_to(m)

    colormap.add_to(m)
    return m
