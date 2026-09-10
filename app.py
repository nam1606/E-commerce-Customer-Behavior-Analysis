"""
E-commerce Customer Behavior Analysis — Streamlit Dashboard.

Entry point for the application.  Run with:
    streamlit run app.py

Features:
    * Upload CSV or generate sample data with one click
    * Date-range slider and brand multi-select filters
    * Six interactive tabs: Overview · Funnel · Cohort · Segments · Time · Brands
    * Download filtered dataset as CSV
"""

import logging
import sys
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st

# ── Ensure project root is importable ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from utils.data_loader import load_csv, generate_sample_data
from utils.session_builder import remove_bots, build_sessions, deduplicate_events
from utils.analyzer import (
    compute_funnel,
    compute_cohort_retention,
    segment_users,
    compute_time_patterns,
    compute_brand_analysis,
    compute_kpis,
    cluster_users,
)
from utils.visualizer import (
    plot_funnel_chart,
    plot_cohort_heatmap,
    plot_segment_distribution,
    plot_time_heatmap,
    plot_brand_comparison,
    plot_cluster_scatter,
    plot_event_distribution,
    plot_daily_events,
    plot_session_length_distribution,
    plot_segment_metrics,
)
from utils.validator import validate_dataframe

# ── Logging ──────────────────────────────────────────────────
cfg.setup_logging()
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=cfg.APP_TITLE,
    page_icon=cfg.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
# Helper: Cache-Friendly Pipeline
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_pipeline(raw_df: pd.DataFrame) -> dict:
    """
    Execute the full analysis pipeline on raw event data.

    Steps: validate → remove bots → deduplicate → build sessions →
    compute all analytics.  Results are cached by Streamlit so the
    heavy computation only runs once per unique dataset.

    Args:
        raw_df: Raw event DataFrame (from CSV or generator).

    Returns:
        dict: Keys for every analysis result (funnel, cohort, segments,
              time_patterns, brands, kpis, clustered, sessioned_df)
              or ``error`` if validation fails.

    Example:
        >>> results = run_pipeline(df)
        >>> results["kpis"]["total_events"]
        50000
    """
    # ── Validate ─────────────────────────────────────────────
    validation = validate_dataframe(raw_df)
    if not validation.is_valid:
        return {"error": validation.summary(), "validation": validation}

    # ── Clean ────────────────────────────────────────────────
    df = remove_bots(raw_df)
    df = deduplicate_events(df)

    # ── Sessions ─────────────────────────────────────────────
    df = build_sessions(df, use_existing=True)

    # ── Analytics ────────────────────────────────────────────
    funnel = compute_funnel(df)
    retention, cohort_sizes = compute_cohort_retention(df)
    segments = segment_users(df)
    time_patterns = compute_time_patterns(df)
    brands = compute_brand_analysis(df)
    kpis = compute_kpis(df)
    clustered = cluster_users(segments)

    return {
        "sessioned_df": df,
        "validation": validation,
        "funnel": funnel,
        "retention": retention,
        "cohort_sizes": cohort_sizes,
        "segments": segments,
        "time_patterns": time_patterns,
        "brands": brands,
        "kpis": kpis,
        "clustered": clustered,
    }


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """
    Render the sidebar with data upload, sample generation, and filters.

    Manages ``st.session_state["raw_df"]`` to hold the loaded data.

    Returns:
        None

    Example:
        >>> render_sidebar()  # populates session_state
    """
    with st.sidebar:
        st.title(f"{cfg.APP_ICON} {cfg.APP_TITLE}")
        st.markdown("---")

        # ── Data Source ──────────────────────────────────────
        st.subheader("📂 Data Source")

        uploaded = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help=f"Max {cfg.MAX_UPLOAD_SIZE_MB} MB. Required columns: "
                 f"{', '.join(cfg.REQUIRED_COLUMNS)}",
        )

        if uploaded is not None:
            try:
                with st.spinner("Loading CSV…"):
                    df = load_csv(uploaded, parse_dates=True)
                    st.session_state["raw_df"] = df
                    st.success(f"✅ Loaded {len(df):,} rows")
            except Exception as exc:
                st.error(f"❌ Failed to load CSV: {exc}")

        st.markdown("**— or —**")

        if st.button("🎲 Generate Sample Data", use_container_width=True):
            with st.spinner(f"Generating {cfg.SAMPLE_DATA_ROWS:,} synthetic events…"):
                df = generate_sample_data(n_rows=cfg.SAMPLE_DATA_ROWS)
                st.session_state["raw_df"] = df
                st.success(f"✅ Generated {len(df):,} events")

        st.markdown("---")

        # ── Filters (shown only when data is loaded) ─────────
        if "raw_df" in st.session_state:
            df = st.session_state["raw_df"]
            st.subheader("🔍 Filters")

            # Date range
            if cfg.EVENT_TIME_COL in df.columns:
                try:
                    dates = pd.to_datetime(df[cfg.EVENT_TIME_COL])
                    min_date = dates.min().date()
                    max_date = dates.max().date()

                    date_range = st.date_input(
                        "Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                    )
                    st.session_state["date_range"] = date_range
                except Exception:
                    st.session_state["date_range"] = None

            # Brand filter
            if cfg.BRAND_COL in df.columns:
                all_brands = sorted(df[cfg.BRAND_COL].dropna().unique().tolist())
                selected_brands = st.multiselect(
                    "Brands",
                    options=all_brands,
                    default=all_brands,
                    help="Select brands to include in analysis",
                )
                st.session_state["selected_brands"] = selected_brands

        st.markdown("---")
        st.caption("Built with Streamlit • Python • Plotly")


# ═══════════════════════════════════════════════════════════════
# Filter Helper
# ═══════════════════════════════════════════════════════════════

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sidebar filters (date range, brands) to the raw DataFrame.

    Args:
        df: Raw event DataFrame.

    Returns:
        pd.DataFrame: Filtered copy.

    Example:
        >>> filtered = apply_filters(raw_df)
        >>> len(filtered) <= len(raw_df)
        True
    """
    filtered = df.copy()

    # ── Date filter ──────────────────────────────────────────
    date_range = st.session_state.get("date_range")
    if date_range and len(date_range) == 2 and cfg.EVENT_TIME_COL in filtered.columns:
        try:
            dates = pd.to_datetime(filtered[cfg.EVENT_TIME_COL])
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            # Make end date inclusive by going to end of day
            end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            # Handle timezone-aware datetimes
            if dates.dt.tz is not None:
                start = start.tz_localize(dates.dt.tz)
                end = end.tz_localize(dates.dt.tz)
            filtered = filtered[(dates >= start) & (dates <= end)]
        except Exception as exc:
            logger.warning("Date filter failed: %s", exc)

    # ── Brand filter ─────────────────────────────────────────
    selected_brands = st.session_state.get("selected_brands")
    if selected_brands and cfg.BRAND_COL in filtered.columns:
        filtered = filtered[filtered[cfg.BRAND_COL].isin(selected_brands)]

    return filtered


# ═══════════════════════════════════════════════════════════════
# KPI Row
# ═══════════════════════════════════════════════════════════════

def render_kpis(kpis: dict) -> None:
    """
    Display a row of metric cards at the top of the dashboard.

    Args:
        kpis: Dictionary from ``analyzer.compute_kpis()``.

    Returns:
        None

    Example:
        >>> render_kpis({"total_events": 50000, ...})
    """
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Events", f"{kpis['total_events']:,}")
    c2.metric("Unique Users", f"{kpis['unique_users']:,}")
    c3.metric("Unique Products", f"{kpis['unique_products']:,}")
    c4.metric("Total Revenue", f"${kpis['total_revenue']:,.2f}")
    c5.metric("Conversion Rate", f"{kpis['overall_conversion_rate']}%")
    c6.metric("Cart Abandon Rate", f"{kpis['cart_abandonment_rate']}%")


# ═══════════════════════════════════════════════════════════════
# Tab Renderers
# ═══════════════════════════════════════════════════════════════

def render_overview_tab(results: dict) -> None:
    """
    Render the Overview tab with event distribution and daily timeline.

    Args:
        results: Pipeline output dictionary.

    Returns:
        None

    Example:
        >>> render_overview_tab(results)
    """
    df = results["sessioned_df"]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_event_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_daily_events(df), use_container_width=True)

    # Session length distribution
    st.plotly_chart(plot_session_length_distribution(df), use_container_width=True)

    # Quick stats table
    st.subheader("📋 Quick Stats")
    kpis = results["kpis"]
    stats_data = {
        "Metric": [
            "Avg Order Value",
            "Overall Conversion Rate",
            "Cart Abandonment Rate",
            "Total Revenue",
        ],
        "Value": [
            f"${kpis['avg_order_value']:.2f}",
            f"{kpis['overall_conversion_rate']}%",
            f"{kpis['cart_abandonment_rate']}%",
            f"${kpis['total_revenue']:,.2f}",
        ],
    }
    st.table(pd.DataFrame(stats_data))


def render_funnel_tab(results: dict) -> None:
    """
    Render the Funnel Analysis tab.

    Args:
        results: Pipeline output dictionary.

    Returns:
        None

    Example:
        >>> render_funnel_tab(results)
    """
    funnel = results["funnel"]

    st.plotly_chart(plot_funnel_chart(funnel), use_container_width=True)

    # Stage-by-stage breakdown table
    st.subheader("Stage-by-Stage Breakdown")

    display_df = funnel[["label", "users", "percentage", "drop_off_pct"]].copy()
    display_df.columns = ["Stage", "Users", "% of Total", "Drop-off %"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Interpretation
    st.info(
        f"**Key Insight:** The largest drop-off occurs at "
        f"**{funnel.loc[funnel['drop_off_pct'].idxmax(), 'label']}** "
        f"({funnel['drop_off_pct'].max():.1f}% drop-off). "
        f"This is the primary area for conversion rate optimisation."
    )


def render_cohort_tab(results: dict) -> None:
    """
    Render the Cohort Retention tab.

    Args:
        results: Pipeline output dictionary.

    Returns:
        None

    Example:
        >>> render_cohort_tab(results)
    """
    retention = results["retention"]
    cohort_sizes = results["cohort_sizes"]

    st.plotly_chart(plot_cohort_heatmap(retention), use_container_width=True)

    # Cohort sizes
    st.subheader("Cohort Sizes")
    st.dataframe(cohort_sizes.T, use_container_width=True)

    # Interpretation
    if "M1" in retention.columns:
        avg_m1 = retention["M1"].mean()
        st.info(
            f"**Average M1 Retention:** {avg_m1:.1f}% — "
            f"this indicates {'healthy' if avg_m1 > 20 else 'low'} "
            f"repeat engagement."
        )


def render_segments_tab(results: dict) -> None:
    """
    Render the User Segments tab with distribution and clustering.

    Args:
        results: Pipeline output dictionary.

    Returns:
        None

    Example:
        >>> render_segments_tab(results)
    """
    segments = results["segments"]
    clustered = results["clustered"]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_segment_distribution(segments), use_container_width=True)
    with col2:
        st.plotly_chart(plot_segment_metrics(segments), use_container_width=True)

    # Segment summary table
    st.subheader("Segment Summary")
    summary = (
        segments.groupby("segment")
        .agg(
            users=("segment", "count"),
            avg_events=("total_events", "mean"),
            avg_views=("total_views", "mean"),
            avg_carts=("total_carts", "mean"),
            avg_purchases=("total_purchases", "mean"),
            avg_price=("avg_price", "mean"),
            avg_session_duration=("avg_session_duration_sec", "mean"),
        )
        .round(1)
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # K-Means clustering
    st.subheader("🔬 K-Means Behaviour Clusters")
    st.plotly_chart(plot_cluster_scatter(clustered), use_container_width=True)

    # Cluster summary
    cluster_summary = (
        clustered.groupby("cluster")
        .agg(
            users=("cluster", "count"),
            avg_events=("total_events", "mean"),
            avg_purchases=("total_purchases", "mean"),
            dominant_segment=("segment", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A"),
        )
        .round(1)
        .reset_index()
    )
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)


def render_time_tab(results: dict) -> None:
    """
    Render the Time Analysis tab.

    Args:
        results: Pipeline output dictionary.

    Returns:
        None

    Example:
        >>> render_time_tab(results)
    """
    patterns = results["time_patterns"]

    # Purchase heatmap (hour × day)
    st.plotly_chart(
        plot_time_heatmap(patterns["heatmap"]), use_container_width=True
    )

    # Hourly event distribution
    st.subheader("Hourly Event Distribution")
    hourly = patterns["hourly"]
    if not hourly.empty:
        import plotly.express as px

        fig = px.bar(
            hourly[hourly[cfg.EVENT_TYPE_COL] == "purchase"],
            x="hour",
            y="count",
            title="Purchases by Hour of Day",
            labels={"hour": "Hour (0-23)", "count": "Purchase Count"},
            template=cfg.PLOTLY_TEMPLATE,
            color_discrete_sequence=[cfg.COLOR_PALETTE[2]],
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Peak hours insight
    if not patterns["heatmap"].empty:
        heatmap = patterns["heatmap"]
        peak_hour = heatmap.sum(axis=1).idxmax()
        peak_day = heatmap.sum(axis=0).idxmax()
        st.info(
            f"**Peak Purchase Time:** {peak_hour}:00 on **{peak_day}**. "
            f"Consider scheduling promotions and email campaigns around this window."
        )


def render_brands_tab(results: dict) -> None:
    """
    Render the Brand Analysis tab.

    Args:
        results: Pipeline output dictionary.

    Returns:
        None

    Example:
        >>> render_brands_tab(results)
    """
    brands = results["brands"]

    if brands.empty:
        st.warning("No brand data available (brand column may be missing or below event threshold).")
        return

    st.plotly_chart(plot_brand_comparison(brands), use_container_width=True)

    # Brand conversion table
    st.subheader("Brand Conversion Metrics")
    display_cols = [
        cfg.BRAND_COL, "views", "carts", "purchases",
        "view_to_cart_rate", "cart_to_purchase_rate",
        "overall_conversion_rate", "avg_price",
    ]
    available_cols = [c for c in display_cols if c in brands.columns]
    st.dataframe(
        brands[available_cols].sort_values("overall_conversion_rate", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # Top converting brand
    if not brands.empty:
        top = brands.sort_values("overall_conversion_rate", ascending=False).iloc[0]
        st.info(
            f"**Top Converting Brand:** {top[cfg.BRAND_COL]} "
            f"with {top['overall_conversion_rate']}% overall conversion rate "
            f"({int(top['purchases'])} purchases from {int(top['views'])} views)."
        )


# ═══════════════════════════════════════════════════════════════
# Download Button
# ═══════════════════════════════════════════════════════════════

def render_download(df: pd.DataFrame) -> None:
    """
    Provide a CSV download button for the processed dataset.

    Args:
        df: Processed/filtered DataFrame.

    Returns:
        None

    Example:
        >>> render_download(sessioned_df)
    """
    try:
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Processed Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name="ecommerce_analysis_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Download failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """
    Application entry point — orchestrates sidebar, pipeline, and tabs.

    Returns:
        None

    Example:
        >>> main()
    """
    render_sidebar()

    # ── Guard: no data loaded yet ────────────────────────────
    if "raw_df" not in st.session_state:
        st.markdown(
            """
            ## 👋 Welcome to the E-commerce Customer Behavior Analysis Dashboard

            **Get started by:**
            1. **Uploading** a CSV file via the sidebar, or
            2. Clicking **"🎲 Generate Sample Data"** to explore with synthetic data.

            ### Required CSV Columns
            | Column | Description |
            |---|---|
            | `event_time` | Event timestamp |
            | `event_type` | `view` / `cart` / `remove_from_cart` / `purchase` |
            | `product_id` | Product identifier |
            | `user_id` | User identifier |
            | `price` | Product price |

            ### Optional Columns
            `brand`, `category_code`, `user_session`
            """
        )
        return

    # ── Apply sidebar filters ────────────────────────────────
    raw_df = st.session_state["raw_df"]
    filtered_df = apply_filters(raw_df)

    if filtered_df.empty:
        st.error("No data remaining after filters — please widen your selection.")
        return

    # ── Run pipeline (cached) ────────────────────────────────
    with st.spinner("🔄 Running analysis pipeline…"):
        results = run_pipeline(filtered_df)

    # ── Handle validation errors ─────────────────────────────
    if "error" in results:
        st.error(results["error"])
        return

    # ── KPI Header ───────────────────────────────────────────
    render_kpis(results["kpis"])
    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────
    tab_overview, tab_funnel, tab_cohort, tab_segments, tab_time, tab_brands = st.tabs(
        ["📊 Overview", "🔻 Funnel", "👥 Cohort", "🧩 Segments", "⏰ Time", "🏷️ Brands"]
    )

    with tab_overview:
        render_overview_tab(results)

    with tab_funnel:
        render_funnel_tab(results)

    with tab_cohort:
        render_cohort_tab(results)

    with tab_segments:
        render_segments_tab(results)

    with tab_time:
        render_time_tab(results)

    with tab_brands:
        render_brands_tab(results)

    # ── Download ─────────────────────────────────────────────
    st.markdown("---")
    render_download(results["sessioned_df"])


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

