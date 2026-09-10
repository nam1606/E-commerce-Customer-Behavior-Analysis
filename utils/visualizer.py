"""
Visualization factory for the E-commerce Behavior Analysis dashboard.

Every public function returns a Plotly ``Figure`` object ready for
``st.plotly_chart()``.  Colour palettes and templates are pulled from
``config.py`` so the look-and-feel is consistent across charts.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config as cfg

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. Funnel Chart
# ═══════════════════════════════════════════════════════════════

def plot_funnel_chart(funnel_df: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal funnel chart showing user counts at each stage.

    Args:
        funnel_df: Output of ``analyzer.compute_funnel()`` with columns
                   ``label``, ``users``, ``percentage``.

    Returns:
        go.Figure: Plotly funnel figure.

    Example:
        >>> fig = plot_funnel_chart(funnel_df)
        >>> fig.show()
    """
    try:
        fig = go.Figure(
            go.Funnel(
                y=funnel_df["label"],
                x=funnel_df["users"],
                textinfo="value+percent initial+percent previous",
                marker=dict(
                    color=cfg.FUNNEL_COLORS[: len(funnel_df)],
                    line=dict(width=2, color="white"),
                ),
                connector=dict(line=dict(color="royalblue", dash="dot", width=2)),
            )
        )
        fig.update_layout(
            title="Customer Conversion Funnel",
            template=cfg.PLOTLY_TEMPLATE,
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        logger.info("Funnel chart created")
        return fig

    except Exception as exc:
        logger.error("Funnel chart creation failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 2. Cohort Retention Heatmap
# ═══════════════════════════════════════════════════════════════

def plot_cohort_heatmap(retention_df: pd.DataFrame) -> go.Figure:
    """
    Plot a cohort retention heatmap.

    Rows are cohort months, columns are period offsets (M0, M1, …).
    Cell colour intensity encodes retention percentage.

    Args:
        retention_df: Retention percentage matrix from
                      ``analyzer.compute_cohort_retention()``.

    Returns:
        go.Figure: Annotated heatmap figure.

    Example:
        >>> fig = plot_cohort_heatmap(retention_df)
        >>> fig.show()
    """
    try:
        # Build annotation text matrix (show "—" for NaN/zero)
        z_values = retention_df.values
        text_values = np.where(
            np.isnan(z_values) | (z_values == 0),
            "",
            np.char.add(np.round(z_values, 1).astype(str), "%"),
        )

        fig = go.Figure(
            go.Heatmap(
                z=z_values,
                x=retention_df.columns.tolist(),
                y=retention_df.index.tolist(),
                text=text_values,
                texttemplate="%{text}",
                colorscale=cfg.HEATMAP_COLORSCALE,
                colorbar=dict(title="Retention %"),
                hovertemplate=(
                    "Cohort: %{y}<br>Period: %{x}<br>"
                    "Retention: %{z:.1f}%<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title="Monthly Cohort Retention",
            xaxis_title="Months Since First Activity",
            yaxis_title="Cohort (First Activity Month)",
            template=cfg.PLOTLY_TEMPLATE,
            height=max(350, 50 * len(retention_df)),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        logger.info("Cohort heatmap created")
        return fig

    except Exception as exc:
        logger.error("Cohort heatmap creation failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 3. Segment Distribution
# ═══════════════════════════════════════════════════════════════

def plot_segment_distribution(segment_df: pd.DataFrame) -> go.Figure:
    """
    Show the proportion of users in each behaviour segment as a
    donut chart and a grouped bar chart of average metrics.

    Args:
        segment_df: Output of ``analyzer.segment_users()`` with a
                    ``segment`` column.

    Returns:
        go.Figure: Donut chart of segment counts.

    Example:
        >>> fig = plot_segment_distribution(segment_df)
        >>> fig.show()
    """
    try:
        counts = segment_df["segment"].value_counts().reset_index()
        counts.columns = ["segment", "users"]

        # Colour mapping for consistency
        color_map = {
            cfg.SEGMENT_PURCHASER: cfg.COLOR_PALETTE[2],      # green
            cfg.SEGMENT_CART_ABANDONER: cfg.COLOR_PALETTE[1],  # red-orange
            cfg.SEGMENT_BROWSER: cfg.COLOR_PALETTE[0],         # blue
        }

        fig = go.Figure(
            go.Pie(
                labels=counts["segment"],
                values=counts["users"],
                hole=0.45,
                marker=dict(
                    colors=[color_map.get(s, "#999") for s in counts["segment"]]
                ),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} users (%{percent})<extra></extra>",
            )
        )
        fig.update_layout(
            title="User Behaviour Segments",
            template=cfg.PLOTLY_TEMPLATE,
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        logger.info("Segment distribution chart created")
        return fig

    except Exception as exc:
        logger.error("Segment chart creation failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 4. Time-of-Day / Day-of-Week Heatmap
# ═══════════════════════════════════════════════════════════════

def plot_time_heatmap(heatmap_df: pd.DataFrame) -> go.Figure:
    """
    Plot a purchase-count heatmap with hour-of-day on the y-axis and
    day-of-week on the x-axis.

    Args:
        heatmap_df: Pivot table (hour × weekday) from
                    ``analyzer.compute_time_patterns()["heatmap"]``.

    Returns:
        go.Figure: Heatmap figure.

    Example:
        >>> fig = plot_time_heatmap(heatmap_df)
        >>> fig.show()
    """
    try:
        if heatmap_df.empty:
            # Return an empty placeholder if no purchase data exists
            fig = go.Figure()
            fig.add_annotation(
                text="No purchase data available for time analysis",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )
            fig.update_layout(
                title="Purchase Heatmap (Hour × Day)",
                template=cfg.PLOTLY_TEMPLATE,
                height=400,
            )
            return fig

        fig = go.Figure(
            go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns.tolist(),
                y=[f"{h:02d}:00" for h in heatmap_df.index],
                colorscale="Blues",
                colorbar=dict(title="Purchases"),
                hovertemplate=(
                    "Day: %{x}<br>Hour: %{y}<br>"
                    "Purchases: %{z}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title="Purchase Heatmap (Hour × Day of Week)",
            xaxis_title="Day of Week",
            yaxis_title="Hour of Day",
            template=cfg.PLOTLY_TEMPLATE,
            height=500,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        logger.info("Time heatmap created")
        return fig

    except Exception as exc:
        logger.error("Time heatmap creation failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 5. Brand Comparison
# ═══════════════════════════════════════════════════════════════

def plot_brand_comparison(brand_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing views, carts, and purchases by brand.

    Args:
        brand_df: Output of ``analyzer.compute_brand_analysis()``.

    Returns:
        go.Figure: Grouped bar chart.

    Example:
        >>> fig = plot_brand_comparison(brand_df)
        >>> fig.show()
    """
    try:
        if brand_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No brand data available",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )
            fig.update_layout(title="Brand Comparison", template=cfg.PLOTLY_TEMPLATE)
            return fig

        brand_df = brand_df.sort_values("total_events", ascending=True)

        fig = go.Figure()
        # Stack bars: views → carts → purchases
        for metric, colour, name in [
            ("views", cfg.COLOR_PALETTE[0], "Views"),
            ("carts", cfg.COLOR_PALETTE[4], "Add to Cart"),
            ("purchases", cfg.COLOR_PALETTE[2], "Purchases"),
        ]:
            fig.add_trace(
                go.Bar(
                    y=brand_df[cfg.BRAND_COL],
                    x=brand_df[metric],
                    name=name,
                    orientation="h",
                    marker_color=colour,
                )
            )

        fig.update_layout(
            title="Top Brands: Views vs Carts vs Purchases",
            xaxis_title="Event Count",
            yaxis_title="Brand",
            barmode="group",
            template=cfg.PLOTLY_TEMPLATE,
            height=max(400, 35 * len(brand_df)),
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=20, r=20, t=70, b=20),
        )
        logger.info("Brand comparison chart created")
        return fig

    except Exception as exc:
        logger.error("Brand comparison chart failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 6. Cluster Scatter Plot
# ═══════════════════════════════════════════════════════════════

def plot_cluster_scatter(clustered_df: pd.DataFrame) -> go.Figure:
    """
    2-D scatter plot coloured by K-Means cluster label.

    Uses the ``pc1`` and ``pc2`` columns produced by
    ``analyzer.cluster_users()``.

    Args:
        clustered_df: DataFrame with ``pc1``, ``pc2``, ``cluster`` columns.

    Returns:
        go.Figure: Scatter plot figure.

    Example:
        >>> fig = plot_cluster_scatter(clustered_df)
        >>> fig.show()
    """
    try:
        fig = px.scatter(
            clustered_df,
            x="pc1",
            y="pc2",
            color="cluster",
            color_continuous_scale="Viridis",
            hover_data=["total_events", "total_purchases", "segment"],
            title="User Behaviour Clusters (K-Means)",
            labels={"pc1": "Component 1", "pc2": "Component 2", "cluster": "Cluster"},
            template=cfg.PLOTLY_TEMPLATE,
        )
        # Treat cluster as discrete
        fig.update_traces(
            marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color="white"))
        )
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_colorbar=dict(title="Cluster"),
        )
        logger.info("Cluster scatter plot created")
        return fig

    except Exception as exc:
        logger.error("Cluster scatter plot failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 7. Event Distribution Pie
# ═══════════════════════════════════════════════════════════════

def plot_event_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Pie chart showing the proportion of each event type.

    Args:
        df: Event DataFrame with ``event_type`` column.

    Returns:
        go.Figure: Pie chart figure.

    Example:
        >>> fig = plot_event_distribution(df)
        >>> fig.show()
    """
    try:
        counts = (
            df[cfg.EVENT_TYPE_COL]
            .value_counts()
            .reset_index()
        )
        counts.columns = ["event_type", "count"]

        fig = px.pie(
            counts,
            values="count",
            names="event_type",
            title="Event Type Distribution",
            color_discrete_sequence=cfg.COLOR_PALETTE,
            template=cfg.PLOTLY_TEMPLATE,
            hole=0.35,
        )
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        logger.info("Event distribution chart created")
        return fig

    except Exception as exc:
        logger.error("Event distribution chart failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 8. Daily Events Timeline
# ═══════════════════════════════════════════════════════════════

def plot_daily_events(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of daily event counts, coloured by event type.

    Args:
        df: Event DataFrame with ``event_time`` and ``event_type``.

    Returns:
        go.Figure: Multi-line time-series figure.

    Example:
        >>> fig = plot_daily_events(df)
        >>> fig.show()
    """
    try:
        df = df.copy()
        df["_date"] = pd.to_datetime(df[cfg.EVENT_TIME_COL]).dt.date

        daily = (
            df.groupby(["_date", cfg.EVENT_TYPE_COL])
            .size()
            .reset_index(name="count")
        )

        fig = px.line(
            daily,
            x="_date",
            y="count",
            color=cfg.EVENT_TYPE_COL,
            title="Daily Event Volume",
            labels={"_date": "Date", "count": "Events", cfg.EVENT_TYPE_COL: "Event"},
            template=cfg.PLOTLY_TEMPLATE,
            color_discrete_sequence=cfg.COLOR_PALETTE,
        )
        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", y=1.12),
        )
        logger.info("Daily events chart created")
        return fig

    except Exception as exc:
        logger.error("Daily events chart failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 9. Session Length Distribution
# ═══════════════════════════════════════════════════════════════

def plot_session_length_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of session durations (seconds → minutes).

    Args:
        df: Session-annotated DataFrame with ``session_duration_sec``.

    Returns:
        go.Figure: Histogram figure.

    Example:
        >>> fig = plot_session_length_distribution(df)
        >>> fig.show()
    """
    try:
        if "session_duration_sec" not in df.columns or "session_id" not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Session duration data not available",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )
            fig.update_layout(title="Session Duration", template=cfg.PLOTLY_TEMPLATE)
            return fig

        # One row per unique session
        session_dur = (
            df.groupby("session_id")["session_duration_sec"]
            .first()
            .reset_index()
        )
        # Convert seconds to minutes for readability
        session_dur["duration_min"] = session_dur["session_duration_sec"] / 60

        # Cap at 99th percentile to avoid extreme outliers stretching the axis
        cap = session_dur["duration_min"].quantile(0.99)
        plot_data = session_dur[session_dur["duration_min"] <= cap]

        fig = px.histogram(
            plot_data,
            x="duration_min",
            nbins=50,
            title="Session Duration Distribution",
            labels={"duration_min": "Duration (minutes)"},
            template=cfg.PLOTLY_TEMPLATE,
            color_discrete_sequence=[cfg.COLOR_PALETTE[0]],
        )
        fig.update_layout(
            yaxis_title="Number of Sessions",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        logger.info("Session length histogram created")
        return fig

    except Exception as exc:
        logger.error("Session length histogram failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 10. Segment Metrics Comparison Bar
# ═══════════════════════════════════════════════════════════════

def plot_segment_metrics(segment_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing average metrics across segments.

    Args:
        segment_df: Output of ``analyzer.segment_users()``.

    Returns:
        go.Figure: Grouped bar chart of avg_price, session_count, etc.

    Example:
        >>> fig = plot_segment_metrics(segment_df)
        >>> fig.show()
    """
    try:
        metrics = (
            segment_df.groupby("segment")
            .agg(
                avg_events=("total_events", "mean"),
                avg_views=("total_views", "mean"),
                avg_price=("avg_price", "mean"),
                avg_sessions=("session_count", "mean"),
            )
            .round(1)
            .reset_index()
        )

        fig = go.Figure()
        for col, colour, label in [
            ("avg_events", cfg.COLOR_PALETTE[0], "Avg Events"),
            ("avg_views", cfg.COLOR_PALETTE[3], "Avg Views"),
            ("avg_sessions", cfg.COLOR_PALETTE[2], "Avg Sessions"),
        ]:
            fig.add_trace(
                go.Bar(
                    x=metrics["segment"],
                    y=metrics[col],
                    name=label,
                    marker_color=colour,
                )
            )

        fig.update_layout(
            title="Average Metrics by Segment",
            xaxis_title="Segment",
            yaxis_title="Average Value",
            barmode="group",
            template=cfg.PLOTLY_TEMPLATE,
            height=400,
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=20, r=20, t=70, b=20),
        )
        logger.info("Segment metrics chart created")
        return fig

    except Exception as exc:
        logger.error("Segment metrics chart failed: %s", exc)
        raise

