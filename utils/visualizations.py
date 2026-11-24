"""
Interactive visualization module for E-Commerce Intelligence Dashboard.
Built with Plotly for dynamic, responsive, and publication-quality charts.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Professional Color Palette (Blue, Soft Purple, Slate, Teal)
PRIMARY_BLUE = "#2563EB"
SECONDARY_PURPLE = "#8B5CF6"
DARK_NAVY = "#1E293B"
SOFT_PURPLE = "#DDD6FE"
LIGHT_BG = "#F8FAFC"
ACCENT_TEAL = "#0D9488"
ACCENT_AMBER = "#F59E0B"
ACCENT_ROSE = "#F43F5E"

COLOR_SEQUENCE = [
    "#2563EB", "#7C3AED", "#0D9488", "#F59E0B", "#EC4899",
    "#06B6D4", "#84CC16", "#6366F1", "#10B981", "#F97316"
]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", size=13, color="#1E293B"),
    margin=dict(l=40, r=30, t=50, b=40),
    hoverlabel=dict(bgcolor="#1E293B", font_size=13, font_color="#FFFFFF")
)


def format_currency(val):
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    elif val >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.2f}"


def plot_sales_by_country(df, top_n=None):
    """Generates an interactive bar chart of sales by country."""
    country_sales = (
        df.groupby("Country", as_index=False)
        .agg({"Total Sales": "sum", "Order ID": "count"})
        .rename(columns={"Order ID": "Orders"})
        .sort_values(by="Total Sales", ascending=False)
    )
    if top_n:
        country_sales = country_sales.head(top_n)

    fig = px.bar(
        country_sales,
        x="Country",
        y="Total Sales",
        text="Total Sales",
        color="Total Sales",
        color_continuous_scale=["#93C5FD", "#3B82F6", "#1D4ED8", "#1E1B4B"],
        title="Total Sales by Country ($ USD)",
        hover_data={"Orders": True, "Total Sales": ":$,.2f"}
    )
    fig.update_traces(
        texttemplate="%{text:$,.0f}",
        textposition="outside",
        marker_line_color="#1E3A8A",
        marker_line_width=1
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        coloraxis_showscale=False,
        xaxis_tickangle=-35,
        yaxis=dict(gridcolor="#E2E8F0", title="Total Revenue ($)")
    )
    return fig


def plot_sales_by_category(df):
    """Generates a horizontal bar chart of sales by product category."""
    cat_sales = (
        df.groupby("Product Category", as_index=False)
        .agg({"Total Sales": "sum", "Quantity Sold": "sum"})
        .sort_values(by="Total Sales", ascending=True)
    )

    fig = go.Figure(
        go.Bar(
            y=cat_sales["Product Category"],
            x=cat_sales["Total Sales"],
            orientation="h",
            marker=dict(
                color=cat_sales["Total Sales"],
                colorscale=["#DDD6FE", "#8B5CF6", "#6D28D9"],
                line=dict(color="#5B21B6", width=1)
            ),
            text=cat_sales["Total Sales"].apply(format_currency),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Total Sales: $%{x:,.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Revenue by Product Category ($ USD)",
        xaxis=dict(gridcolor="#E2E8F0", title="Revenue ($)"),
        yaxis=dict(title="")
    )
    return fig


def plot_monthly_sales_trend(df):
    """Generates a smooth line/area chart for monthly revenue trends."""
    df_sorted = df.dropna(subset=["Order Date"]).copy()
    try:
        monthly = (
            df_sorted.set_index("Order Date")
            .resample("ME")
            .agg({"Total Sales": "sum", "Order ID": "count"})
            .reset_index()
        )
    except Exception:
        monthly = (
            df_sorted.set_index("Order Date")
            .resample("M")
            .agg({"Total Sales": "sum", "Order ID": "count"})
            .reset_index()
        )
    monthly["Month_Label"] = monthly["Order Date"].dt.strftime("%b %Y")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["Month_Label"],
            y=monthly["Total Sales"],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color=PRIMARY_BLUE, width=3, shape="spline"),
            marker=dict(size=7, color=SECONDARY_PURPLE, line=dict(color="#FFFFFF", width=1.5)),
            fillcolor="rgba(37, 99, 235, 0.12)",
            hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<br>Orders: %{text}<extra></extra>",
            text=monthly["Order ID"]
        )
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Monthly Sales & Revenue Growth Trend",
        xaxis=dict(gridcolor="#E2E8F0", title="Month", tickangle=-30),
        yaxis=dict(gridcolor="#E2E8F0", title="Monthly Revenue ($)")
    )
    return fig


def plot_yearly_sales(df):
    """Generates a bar chart of sales aggregated by year."""
    yearly = (
        df.groupby("Year", as_index=False)
        .agg({"Total Sales": "sum", "Order ID": "count"})
        .rename(columns={"Order ID": "Orders"})
        .sort_values(by="Year")
    )
    yearly["Year"] = yearly["Year"].astype(str)

    fig = px.bar(
        yearly,
        x="Year",
        y="Total Sales",
        text="Total Sales",
        color="Year",
        color_discrete_sequence=[PRIMARY_BLUE, SECONDARY_PURPLE, ACCENT_TEAL],
        title="Annual Revenue Distribution by Year",
        hover_data={"Orders": True, "Total Sales": ":$,.2f"}
    )
    fig.update_traces(
        texttemplate="%{text:$,.0f}",
        textposition="outside"
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        showlegend=False,
        yaxis=dict(gridcolor="#E2E8F0", title="Total Sales ($)")
    )
    return fig


def plot_quantity_by_category(df):
    """Generates a bar chart showing units sold across categories."""
    qty_data = (
        df.groupby("Product Category", as_index=False)["Quantity Sold"]
        .sum()
        .sort_values(by="Quantity Sold", ascending=False)
    )

    fig = px.bar(
        qty_data,
        x="Product Category",
        y="Quantity Sold",
        text="Quantity Sold",
        color="Quantity Sold",
        color_continuous_scale="Purples",
        title="Total Units Sold by Product Category"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        coloraxis_showscale=False,
        xaxis_tickangle=-35,
        yaxis=dict(gridcolor="#E2E8F0", title="Units Sold")
    )
    return fig


def plot_order_status_distribution(df):
    """Generates an attractive donut chart showing order status distribution."""
    status_counts = df["Order Status"].value_counts().reset_index()
    status_counts.columns = ["Order Status", "Count"]

    color_map = {
        "Completed": "#10B981",
        "Pending": "#F59E0B",
        "Processing": "#3B82F6",
        "Cancelled": "#EF4444",
        "Refunded": "#8B5CF6",
        "Unknown": "#94A3B8"
    }
    colors = [color_map.get(s, "#CBD5E1") for s in status_counts["Order Status"]]

    fig = go.Figure(
        go.Pie(
            labels=status_counts["Order Status"],
            values=status_counts["Count"],
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Orders: %{value:,} (%{percent})<extra></extra>"
        )
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Order Fulfillment Status Breakdown",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


def plot_actual_vs_predicted(actual, predicted):
    """
    Generates a scatter plot comparing Actual Total Sales vs Model Predicted Sales
    with a reference 45-degree line.
    """
    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            name="Test Predictions",
            marker=dict(
                size=6,
                color="#6366F1",
                opacity=0.7,
                line=dict(color="#312E81", width=0.5)
            ),
            hovertemplate="<b>Actual:</b> $%{x:,.2f}<br><b>Predicted:</b> $%{y:,.2f}<extra></extra>"
        )
    )

    # Reference 45-degree ideal fit line
    max_val = max(float(np.max(actual)), float(np.max(predicted)))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Ideal Fit (y = x)",
            line=dict(color="#EF4444", dash="dash", width=2)
        )
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Actual vs Predicted Total Sales (Test Set)",
        xaxis=dict(gridcolor="#E2E8F0", title="Actual Total Sales ($)"),
        yaxis=dict(gridcolor="#E2E8F0", title="Predicted Total Sales ($)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_feature_importance(top_features_list):
    """Generates a horizontal bar chart of Random Forest feature importances."""
    feat_df = pd.DataFrame(top_features_list).sort_values(by="Importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            y=feat_df["Feature"],
            x=feat_df["Importance"],
            orientation="h",
            marker=dict(
                color=feat_df["Importance"],
                colorscale=["#93C5FD", "#2563EB", "#1D4ED8"],
                line=dict(color="#1E3A8A", width=1)
            ),
            text=feat_df["Importance"].apply(lambda v: f"{v*100:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance Weight: %{x:.4f} (%{text})<extra></extra>"
        )
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Top 10 Feature Importances (Random Forest)",
        xaxis=dict(gridcolor="#E2E8F0", title="Importance Score"),
        yaxis=dict(title="")
    )
    return fig
