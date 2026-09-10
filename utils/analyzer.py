"""
Core analytics engine for e-commerce customer behavior.

Provides:
* Funnel conversion-rate computation
* Monthly cohort retention analysis
* Behaviour segmentation (Browser / Cart Abandoner / Purchaser)
* Hourly & daily time-pattern analysis
* Brand-level performance metrics
* K-Means behaviour clustering
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config as cfg

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. Funnel Analysis
# ═══════════════════════════════════════════════════════════════

def compute_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate how many unique users reached each funnel stage.

    Each user is assigned the **highest** stage they ever reached
    (view < cart < purchase). The result is an aggregated DataFrame
    suitable for plotting a funnel chart.

    Args:
        df: Event DataFrame with ``event_type`` and ``user_id`` columns.

    Returns:
        pd.DataFrame: Columns ``stage``, ``label``, ``users``,
        ``percentage``, ``drop_off_pct``.

    Example:
        >>> funnel = compute_funnel(df)
        >>> funnel["stage"].tolist()
        ['view', 'cart', 'purchase']
    """
    try:
        # Map each event to its numeric stage rank
        df = df.copy()
        df["_stage_rank"] = df[cfg.EVENT_TYPE_COL].map(cfg.FUNNEL_STAGE_ORDER)

        # Keep only events whose type maps to a known stage
        df = df.dropna(subset=["_stage_rank"])

        # Highest stage each user ever reached
        user_max = df.groupby(cfg.USER_ID_COL)["_stage_rank"].max()

        rows = []
        total_users = len(user_max)

        for stage in cfg.FUNNEL_STAGES:
            rank = cfg.FUNNEL_STAGE_ORDER[stage]
            # Users who reached *at least* this stage
            count = int((user_max >= rank).sum())
            pct = round(count / total_users * 100, 2) if total_users else 0.0
            rows.append({
                "stage": stage,
                "label": cfg.FUNNEL_STAGE_LABELS.get(stage, stage),
                "users": count,
                "percentage": pct,
            })

        result = pd.DataFrame(rows)

        # Compute stage-over-stage drop-off
        result["drop_off_pct"] = 0.0
        for i in range(1, len(result)):
            prev = result.loc[i - 1, "users"]
            curr = result.loc[i, "users"]
            if prev > 0:
                result.loc[i, "drop_off_pct"] = round(
                    (1 - curr / prev) * 100, 2
                )

        logger.info("Funnel computed — %d total users", total_users)
        return result

    except KeyError as exc:
        logger.error("Funnel computation failed — missing column: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 2. Cohort Retention Analysis
# ═══════════════════════════════════════════════════════════════

def compute_cohort_retention(
    df: pd.DataFrame,
    max_periods: int = cfg.COHORT_PERIODS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a monthly cohort retention matrix.

    Each user is placed into the cohort of their **first event** month.
    For every subsequent month we count how many users from that cohort
    were still active (had at least one event).

    Args:
        df: Event DataFrame with ``user_id`` and ``event_time``.
        max_periods: Maximum number of months to track.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - retention_pct: Retention percentages (cohort × period).
            - cohort_sizes: Series of cohort sizes indexed by cohort month.

    Example:
        >>> retention, sizes = compute_cohort_retention(df)
        >>> retention.iloc[0, 0]  # M0 retention is always 100 %
        100.0
    """
    try:
        df = df.copy()
        df["_event_month"] = pd.to_datetime(
            df[cfg.EVENT_TIME_COL]
        ).dt.to_period("M")

        # Cohort = month of first activity
        cohort_map = (
            df.groupby(cfg.USER_ID_COL)["_event_month"]
            .min()
            .rename("cohort")
        )
        df = df.merge(cohort_map, on=cfg.USER_ID_COL)

        # Period index = months since cohort month
        df["_period"] = (
            df["_event_month"].astype(int) - df["cohort"].astype(int)
        )

        # Limit to max_periods
        df = df[df["_period"] <= max_periods]

        # Unique active users per cohort × period
        cohort_data = (
            df.groupby(["cohort", "_period"])[cfg.USER_ID_COL]
            .nunique()
            .reset_index(name="active_users")
        )

        # Pivot into matrix
        cohort_pivot = cohort_data.pivot(
            index="cohort", columns="_period", values="active_users"
        ).fillna(0)

        # Cohort sizes (period 0 column)
        cohort_sizes = cohort_pivot[0].copy() if 0 in cohort_pivot.columns else cohort_pivot.iloc[:, 0].copy()

        # Convert to retention percentages
        retention_pct = cohort_pivot.div(cohort_sizes, axis=0) * 100
        retention_pct = retention_pct.round(1)

        # Friendly index labels
        retention_pct.index = retention_pct.index.astype(str)
        cohort_sizes.index = cohort_sizes.index.astype(str)
        retention_pct.columns = [f"M{int(c)}" for c in retention_pct.columns]

        logger.info(
            "Cohort retention matrix: %d cohorts × %d periods",
            retention_pct.shape[0],
            retention_pct.shape[1],
        )
        return retention_pct, cohort_sizes.to_frame("cohort_size")

    except Exception as exc:
        logger.error("Cohort analysis failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 3. Behaviour Segmentation
# ═══════════════════════════════════════════════════════════════

def segment_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify every user into a behaviour segment.

    Segments:
        * **Purchaser** — completed at least one purchase.
        * **Cart Abandoner** — added to cart but never purchased.
        * **Browser** — only viewed products.

    Also computes per-user aggregated metrics (total events, views,
    average price, session count, average session duration).

    Args:
        df: Session-annotated event DataFrame.

    Returns:
        pd.DataFrame: One row per user with columns ``segment``,
        ``total_events``, ``total_views``, ``total_carts``,
        ``total_purchases``, ``avg_price``, ``session_count``,
        ``avg_session_duration_sec``.

    Example:
        >>> segments = segment_users(df)
        >>> segments["segment"].unique()
        array(['Browser', 'Cart Abandoner', 'Purchaser'], dtype=object)
    """
    try:
        # Aggregate per-user metrics
        user_agg = df.groupby(cfg.USER_ID_COL).agg(
            total_events=(cfg.EVENT_TYPE_COL, "count"),
            total_views=(cfg.EVENT_TYPE_COL, lambda x: (x == "view").sum()),
            total_carts=(cfg.EVENT_TYPE_COL, lambda x: (x == "cart").sum()),
            total_purchases=(cfg.EVENT_TYPE_COL, lambda x: (x == "purchase").sum()),
            avg_price=(cfg.PRICE_COL, "mean"),
            session_count=("session_id", "nunique") if "session_id" in df.columns else (cfg.EVENT_TYPE_COL, "count"),
        ).reset_index()

        # Session duration (average across user's sessions)
        if "session_duration_sec" in df.columns and "session_id" in df.columns:
            sess_dur = (
                df.groupby([cfg.USER_ID_COL, "session_id"])["session_duration_sec"]
                .first()
                .groupby(level=0)
                .mean()
                .rename("avg_session_duration_sec")
            )
            user_agg = user_agg.merge(
                sess_dur, on=cfg.USER_ID_COL, how="left"
            )
        else:
            user_agg["avg_session_duration_sec"] = 0.0

        # Assign segment label based on highest stage reached
        def _assign_segment(row: pd.Series) -> str:
            """Determine segment from purchase/cart counts."""
            if row["total_purchases"] > 0:
                return cfg.SEGMENT_PURCHASER
            if row["total_carts"] > 0:
                return cfg.SEGMENT_CART_ABANDONER
            return cfg.SEGMENT_BROWSER

        user_agg["segment"] = user_agg.apply(_assign_segment, axis=1)
        user_agg["avg_price"] = user_agg["avg_price"].round(2)
        user_agg["avg_session_duration_sec"] = user_agg["avg_session_duration_sec"].round(1)

        logger.info(
            "Segmentation complete — %s",
            user_agg["segment"].value_counts().to_dict(),
        )
        return user_agg

    except Exception as exc:
        logger.error("User segmentation failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 4. Time-Pattern Analysis
# ═══════════════════════════════════════════════════════════════

def compute_time_patterns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyse purchase distribution by hour-of-day and day-of-week.

    Args:
        df: Event DataFrame with ``event_time`` and ``event_type``.

    Returns:
        Dict with keys:
            * ``hourly``  — event counts per hour (0-23) per event type.
            * ``daily``   — event counts per weekday (0=Mon … 6=Sun).
            * ``heatmap`` — pivot (hour × weekday) of purchase counts.

    Example:
        >>> patterns = compute_time_patterns(df)
        >>> patterns["hourly"].columns.tolist()
        ['hour', 'event_type', 'count']
    """
    try:
        df = df.copy()
        dt_col = pd.to_datetime(df[cfg.EVENT_TIME_COL])
        df["_hour"] = dt_col.dt.hour
        df["_weekday"] = dt_col.dt.dayofweek  # 0=Monday
        df["_weekday_name"] = dt_col.dt.day_name()

        # Hourly distribution by event type
        hourly = (
            df.groupby(["_hour", cfg.EVENT_TYPE_COL])
            .size()
            .reset_index(name="count")
            .rename(columns={"_hour": "hour"})
        )

        # Daily distribution by event type
        daily = (
            df.groupby(["_weekday", "_weekday_name", cfg.EVENT_TYPE_COL])
            .size()
            .reset_index(name="count")
            .rename(columns={"_weekday": "weekday", "_weekday_name": "weekday_name"})
        )

        # Heatmap pivot: purchases only, hour vs weekday
        purchases = df[df[cfg.EVENT_TYPE_COL] == "purchase"]
        if not purchases.empty:
            heatmap = (
                purchases.groupby(["_hour", "_weekday"])
                .size()
                .reset_index(name="purchases")
                .pivot(index="_hour", columns="_weekday", values="purchases")
                .fillna(0)
                .astype(int)
            )
            heatmap.index.name = "hour"
            heatmap.columns = [
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][c]
                for c in heatmap.columns
            ]
        else:
            heatmap = pd.DataFrame()

        logger.info("Time pattern analysis complete")
        return {"hourly": hourly, "daily": daily, "heatmap": heatmap}

    except Exception as exc:
        logger.error("Time pattern analysis failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 5. Brand Analysis
# ═══════════════════════════════════════════════════════════════

def compute_brand_analysis(
    df: pd.DataFrame,
    top_n: int = cfg.TOP_N_BRANDS,
    min_events: int = cfg.MIN_BRAND_EVENTS,
) -> pd.DataFrame:
    """
    Compute per-brand metrics: views, carts, purchases, conversion rate,
    and average price.

    Args:
        df: Event DataFrame containing ``brand`` and ``event_type``.
        top_n: Number of top brands to return (by total events).
        min_events: Minimum total events for a brand to be included.

    Returns:
        pd.DataFrame: One row per brand with performance metrics.

    Example:
        >>> brands = compute_brand_analysis(df, top_n=10)
        >>> "view_to_cart_rate" in brands.columns
        True
    """
    try:
        if cfg.BRAND_COL not in df.columns:
            logger.warning("No '%s' column — skipping brand analysis", cfg.BRAND_COL)
            return pd.DataFrame()

        # Drop rows with missing brand
        branded = df.dropna(subset=[cfg.BRAND_COL])

        brand_stats = (
            branded.groupby(cfg.BRAND_COL)
            .agg(
                total_events=(cfg.EVENT_TYPE_COL, "count"),
                views=(cfg.EVENT_TYPE_COL, lambda x: (x == "view").sum()),
                carts=(cfg.EVENT_TYPE_COL, lambda x: (x == "cart").sum()),
                purchases=(cfg.EVENT_TYPE_COL, lambda x: (x == "purchase").sum()),
                avg_price=(cfg.PRICE_COL, "mean"),
                unique_products=(cfg.PRODUCT_ID_COL, "nunique"),
            )
            .reset_index()
        )

        # Filter by minimum event count to avoid noisy small brands
        brand_stats = brand_stats[brand_stats["total_events"] >= min_events]

        # Conversion rates
        brand_stats["view_to_cart_rate"] = np.where(
            brand_stats["views"] > 0,
            (brand_stats["carts"] / brand_stats["views"] * 100).round(2),
            0.0,
        )
        brand_stats["cart_to_purchase_rate"] = np.where(
            brand_stats["carts"] > 0,
            (brand_stats["purchases"] / brand_stats["carts"] * 100).round(2),
            0.0,
        )
        brand_stats["overall_conversion_rate"] = np.where(
            brand_stats["views"] > 0,
            (brand_stats["purchases"] / brand_stats["views"] * 100).round(2),
            0.0,
        )
        brand_stats["avg_price"] = brand_stats["avg_price"].round(2)

        # Return top N by total events
        brand_stats = brand_stats.sort_values("total_events", ascending=False).head(top_n)

        logger.info("Brand analysis: %d brands analysed", len(brand_stats))
        return brand_stats.reset_index(drop=True)

    except Exception as exc:
        logger.error("Brand analysis failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 6. K-Means Behaviour Clustering
# ═══════════════════════════════════════════════════════════════

def cluster_users(
    user_features: pd.DataFrame,
    n_clusters: int = cfg.N_CLUSTERS,
) -> pd.DataFrame:
    """
    Apply K-Means clustering on user-level feature vectors.

    Features are standardised before clustering so that columns with
    large magnitudes do not dominate.

    Args:
        user_features: DataFrame from ``segment_users()`` with numeric
                       behaviour columns.
        n_clusters: Number of clusters for K-Means.

    Returns:
        pd.DataFrame: Input DataFrame with an added ``cluster`` column
        (integer label) and two PCA-like columns (``pc1``, ``pc2``) for
        scatter-plot visualisation.

    Example:
        >>> clustered = cluster_users(segments, n_clusters=4)
        >>> clustered["cluster"].nunique()
        4
    """
    try:
        feature_cols = [
            "total_events", "total_views", "total_carts",
            "total_purchases", "avg_price", "session_count",
        ]
        # Use only columns that exist
        available = [c for c in feature_cols if c in user_features.columns]
        if len(available) < 2:
            logger.warning("Not enough features for clustering — need ≥ 2")
            user_features["cluster"] = 0
            user_features["pc1"] = 0.0
            user_features["pc2"] = 0.0
            return user_features

        X = user_features[available].fillna(0).values

        # Standardise to zero-mean, unit-variance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit K-Means
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        user_features = user_features.copy()
        user_features["cluster"] = labels

        # Simple 2-D projection for visualisation: use first two scaled dims
        # (avoids adding PCA as a dependency)
        user_features["pc1"] = X_scaled[:, 0]
        user_features["pc2"] = X_scaled[:, 1] if X_scaled.shape[1] > 1 else 0.0

        logger.info("K-Means clustering: %d clusters on %d users", n_clusters, len(user_features))
        return user_features

    except Exception as exc:
        logger.error("Clustering failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# 7. Chi-Square Test (Device / Referral Conversion Comparison)
# ═══════════════════════════════════════════════════════════════

def chi_square_conversion_test(
    df: pd.DataFrame,
    group_col: str,
) -> Dict[str, Any]:
    """
    Run a chi-square test comparing conversion rates across groups.

    Useful for testing whether conversion differs significantly
    between device types, referral sources, etc.

    Args:
        df: Event DataFrame containing ``event_type`` and the grouping
            column.
        group_col: Column name to group by (e.g., ``device_type``).

    Returns:
        Dict with keys ``chi2``, ``p_value``, ``dof``, ``significant``
        (at α = 0.05), and ``contingency_table``.

    Example:
        >>> result = chi_square_conversion_test(df, "device_type")
        >>> result["significant"]
        True
    """
    try:
        if group_col not in df.columns:
            logger.warning("Column '%s' not in DataFrame — skipping χ² test", group_col)
            return {"chi2": None, "p_value": None, "dof": None,
                    "significant": None, "contingency_table": pd.DataFrame()}

        # Build binary purchase flag per user
        user_group = df.groupby(cfg.USER_ID_COL).agg(
            group=(group_col, "first"),
            purchased=(cfg.EVENT_TYPE_COL, lambda x: int("purchase" in x.values)),
        )

        # Contingency table: group × purchased (0/1)
        ct = pd.crosstab(user_group["group"], user_group["purchased"])

        chi2, p_value, dof, _ = chi2_contingency(ct)

        result = {
            "chi2": round(chi2, 4),
            "p_value": round(p_value, 6),
            "dof": dof,
            "significant": p_value < 0.05,
            "contingency_table": ct,
        }
        logger.info("χ² test on '%s': χ²=%.4f, p=%.6f", group_col, chi2, p_value)
        return result

    except Exception as exc:
        logger.error("Chi-square test failed: %s", exc)
        return {"chi2": None, "p_value": None, "dof": None,
                "significant": None, "contingency_table": pd.DataFrame()}


# ═══════════════════════════════════════════════════════════════
# 8. Summary KPIs
# ═══════════════════════════════════════════════════════════════

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute high-level KPI metrics for the dashboard header.

    Args:
        df: Event DataFrame.

    Returns:
        Dict with keys ``total_events``, ``unique_users``,
        ``unique_products``, ``total_revenue``, ``avg_order_value``,
        ``overall_conversion_rate``, ``cart_abandonment_rate``.

    Example:
        >>> kpis = compute_kpis(df)
        >>> kpis["total_events"]
        50000
    """
    try:
        total_events = len(df)
        unique_users = df[cfg.USER_ID_COL].nunique()
        unique_products = df[cfg.PRODUCT_ID_COL].nunique()

        purchases = df[df[cfg.EVENT_TYPE_COL] == "purchase"]
        total_revenue = round(purchases[cfg.PRICE_COL].sum(), 2)
        avg_order_value = round(purchases[cfg.PRICE_COL].mean(), 2) if not purchases.empty else 0.0

        # Users who viewed anything
        viewers = df[df[cfg.EVENT_TYPE_COL] == "view"][cfg.USER_ID_COL].nunique()
        purchasers = purchases[cfg.USER_ID_COL].nunique()
        overall_conv = round(purchasers / viewers * 100, 2) if viewers > 0 else 0.0

        # Cart abandonment rate: users who carted but never purchased
        carters = set(df[df[cfg.EVENT_TYPE_COL] == "cart"][cfg.USER_ID_COL])
        buyers = set(purchases[cfg.USER_ID_COL])
        abandoners = carters - buyers
        cart_abandon_rate = (
            round(len(abandoners) / len(carters) * 100, 2)
            if carters else 0.0
        )

        kpis = {
            "total_events": total_events,
            "unique_users": unique_users,
            "unique_products": unique_products,
            "total_revenue": total_revenue,
            "avg_order_value": avg_order_value,
            "overall_conversion_rate": overall_conv,
            "cart_abandonment_rate": cart_abandon_rate,
        }
        logger.info("KPIs computed: %s", kpis)
        return kpis

    except Exception as exc:
        logger.error("KPI computation failed: %s", exc)
        raise

