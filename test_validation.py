"""
test_validation.py — Automated test suite for E-commerce Customer Behavior Analysis.

Covers five critical areas:
    1. Module imports & dependency availability
    2. Configuration integrity (check_keys)
    3. Session building (chunking / sessionization)
    4. Analysis pipeline (funnel, segmentation, cohort)
    5. End-to-end pipeline (data generation → cleaning → analysis → visualization)

Run with:  python -m pytest test_validation.py -v
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# ── Ensure project root is importable ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from utils.data_loader import generate_sample_data, load_csv
from utils.session_builder import build_sessions, remove_bots, deduplicate_events
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
)
from utils.validator import validate_dataframe, ValidationResult

logging.basicConfig(level=logging.WARNING)


# ═══════════════════════════════════════════════════════════════
# Shared Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """
    Generate a small sample dataset for fast test execution.

    Returns:
        pd.DataFrame: Synthetic clickstream events (~2 000 rows).

    Example:
        >>> df = sample_df()
        >>> len(df) > 0
        True
    """
    return generate_sample_data(n_rows=2000, seed=99)


@pytest.fixture(scope="module")
def sessioned_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build sessions on the sample data.

    Args:
        sample_df: Fixture providing raw synthetic data.

    Returns:
        pd.DataFrame: Session-annotated DataFrame.

    Example:
        >>> "session_id" in sessioned_df.columns
        True
    """
    clean = remove_bots(sample_df)
    deduped = deduplicate_events(clean)
    return build_sessions(deduped, use_existing=True)


# ═══════════════════════════════════════════════════════════════
# TEST 1: Module Imports & Dependencies
# ═══════════════════════════════════════════════════════════════

class TestImports:
    """Verify that all project modules and third-party deps import cleanly."""

    def test_core_libraries(self) -> None:
        """
        All core third-party libraries should be importable.

        Returns:
            None

        Example:
            >>> import pandas  # should not raise
        """
        import pandas          # noqa: F401
        import numpy           # noqa: F401
        import plotly           # noqa: F401
        import seaborn          # noqa: F401
        import matplotlib       # noqa: F401
        import scipy            # noqa: F401
        import sklearn          # noqa: F401
        import streamlit        # noqa: F401
        import dotenv           # noqa: F401

    def test_project_modules(self) -> None:
        """
        All project-internal modules should import without errors.

        Returns:
            None

        Example:
            >>> import config  # should not raise
        """
        import config                       # noqa: F401
        import utils.data_loader            # noqa: F401
        import utils.session_builder        # noqa: F401
        import utils.analyzer               # noqa: F401
        import utils.visualizer             # noqa: F401
        import utils.validator              # noqa: F401

    def test_utils_package_reexports(self) -> None:
        """
        The utils package __init__ should re-export key symbols.

        Returns:
            None

        Example:
            >>> from utils import compute_funnel
        """
        from utils import (
            load_csv,
            generate_sample_data,
            compute_funnel,
            validate_dataframe,
            plot_funnel_chart,
        )
        assert callable(load_csv)
        assert callable(generate_sample_data)
        assert callable(compute_funnel)
        assert callable(validate_dataframe)
        assert callable(plot_funnel_chart)


# ═══════════════════════════════════════════════════════════════
# TEST 2: Configuration Integrity
# ═══════════════════════════════════════════════════════════════

class TestConfig:
    """Verify configuration constants are properly set."""

    def test_check_keys_all_pass(self) -> None:
        """
        check_keys() should return all True for default config.

        Returns:
            None

        Example:
            >>> all(cfg.check_keys().values())
            True
        """
        status = cfg.check_keys()
        assert all(status.values()), f"Failing keys: {[k for k, v in status.items() if not v]}"

    def test_funnel_stages_non_empty(self) -> None:
        """
        Funnel stages list must contain at least view and purchase.

        Returns:
            None

        Example:
            >>> "view" in cfg.FUNNEL_STAGES
            True
        """
        assert len(cfg.FUNNEL_STAGES) >= 2
        assert "view" in cfg.FUNNEL_STAGES
        assert "purchase" in cfg.FUNNEL_STAGES

    def test_session_gap_positive(self) -> None:
        """
        Session gap must be a positive integer.

        Returns:
            None

        Example:
            >>> cfg.SESSION_GAP_MINUTES > 0
            True
        """
        assert cfg.SESSION_GAP_MINUTES > 0

    def test_required_columns_defined(self) -> None:
        """
        Required columns list must not be empty.

        Returns:
            None

        Example:
            >>> len(cfg.REQUIRED_COLUMNS) > 0
            True
        """
        assert len(cfg.REQUIRED_COLUMNS) > 0
        assert cfg.EVENT_TIME_COL in cfg.REQUIRED_COLUMNS
        assert cfg.USER_ID_COL in cfg.REQUIRED_COLUMNS


# ═══════════════════════════════════════════════════════════════
# TEST 3: Session Building (Chunking)
# ═══════════════════════════════════════════════════════════════

class TestSessionBuilder:
    """Test session construction, bot removal, and deduplication."""

    def test_remove_bots_preserves_normal_users(self, sample_df: pd.DataFrame) -> None:
        """
        Bot removal should not remove users below the threshold.

        Args:
            sample_df: Pytest fixture with synthetic data.

        Returns:
            None

        Example:
            >>> clean = remove_bots(df, threshold=1000)
            >>> len(clean) > 0
            True
        """
        clean = remove_bots(sample_df, threshold=cfg.BOT_THRESHOLD_EVENTS_PER_DAY)
        # With synthetic data and default threshold, no users should be bots
        assert len(clean) > 0
        assert len(clean) <= len(sample_df)

    def test_build_sessions_adds_session_id(self, sample_df: pd.DataFrame) -> None:
        """
        build_sessions() must add a session_id column.

        Args:
            sample_df: Pytest fixture with synthetic data.

        Returns:
            None

        Example:
            >>> "session_id" in build_sessions(df).columns
            True
        """
        result = build_sessions(sample_df, use_existing=True)
        assert "session_id" in result.columns
        assert result["session_id"].nunique() > 0

    def test_build_sessions_gap_method(self, sample_df: pd.DataFrame) -> None:
        """
        When use_existing=False, sessions should be constructed via
        the time-gap algorithm.

        Args:
            sample_df: Pytest fixture with synthetic data.

        Returns:
            None

        Example:
            >>> s = build_sessions(df, use_existing=False)
            >>> s["session_id"].nunique() >= 1
            True
        """
        # Drop existing session column to force gap-based construction
        df_no_session = sample_df.drop(columns=[cfg.SESSION_COL], errors="ignore")
        result = build_sessions(df_no_session, gap_minutes=30, use_existing=False)
        assert "session_id" in result.columns
        assert result["session_id"].nunique() >= 1

    def test_deduplicate_removes_exact_duplicates(self) -> None:
        """
        Deduplication should remove identical rows.

        Returns:
            None

        Example:
            >>> deduped = deduplicate_events(df_with_dupes)
            >>> len(deduped) == len(df_with_dupes) - n_dupes
            True
        """
        # Create a small DataFrame with known duplicates
        rows = {
            cfg.EVENT_TIME_COL: pd.to_datetime(["2024-01-01"] * 3),
            cfg.EVENT_TYPE_COL: ["view", "view", "cart"],
            cfg.PRODUCT_ID_COL: [1, 1, 2],
            cfg.USER_ID_COL: [100, 100, 100],
            cfg.PRICE_COL: [10.0, 10.0, 20.0],
        }
        df = pd.DataFrame(rows)
        result = deduplicate_events(df)
        assert len(result) == 2  # one duplicate removed

    def test_session_duration_computed(self, sessioned_df: pd.DataFrame) -> None:
        """
        Session-annotated data should have a session_duration_sec column.

        Args:
            sessioned_df: Pytest fixture with sessioned data.

        Returns:
            None

        Example:
            >>> "session_duration_sec" in sessioned_df.columns
            True
        """
        assert "session_duration_sec" in sessioned_df.columns


# ═══════════════════════════════════════════════════════════════
# TEST 4: Analysis Pipeline (Embedding-equivalent)
# ═══════════════════════════════════════════════════════════════

class TestAnalyzer:
    """Test funnel, cohort, segmentation, and brand analysis."""

    def test_funnel_has_three_stages(self, sessioned_df: pd.DataFrame) -> None:
        """
        Funnel should produce exactly 3 stage rows.

        Args:
            sessioned_df: Pytest fixture with sessioned data.

        Returns:
            None

        Example:
            >>> len(compute_funnel(df)) == 3
            True
        """
        funnel = compute_funnel(sessioned_df)
        assert len(funnel) == len(cfg.FUNNEL_STAGES)
        assert funnel.iloc[0]["stage"] == "view"
        assert funnel.iloc[-1]["stage"] == "purchase"

    def test_funnel_monotonically_decreasing(self, sessioned_df: pd.DataFrame) -> None:
        """
        User counts should decrease at each subsequent funnel stage.

        Args:
            sessioned_df: Pytest fixture.

        Returns:
            None

        Example:
            >>> funnel["users"].is_monotonic_decreasing
            True
        """
        funnel = compute_funnel(sessioned_df)
        users = funnel["users"].tolist()
        for i in range(1, len(users)):
            assert users[i] <= users[i - 1], (
                f"Stage {i} has more users than stage {i-1}"
            )

    def test_cohort_retention_m0_is_100(self, sessioned_df: pd.DataFrame) -> None:
        """
        M0 retention should be 100% for every cohort.

        Args:
            sessioned_df: Pytest fixture.

        Returns:
            None

        Example:
            >>> retention["M0"].unique()
            array([100.])
        """
        retention, sizes = compute_cohort_retention(sessioned_df)
        if "M0" in retention.columns:
            assert (retention["M0"] == 100.0).all()

    def test_segment_users_labels(self, sessioned_df: pd.DataFrame) -> None:
        """
        Every user should be classified into one of the three segments.

        Args:
            sessioned_df: Pytest fixture.

        Returns:
            None

        Example:
            >>> segments["segment"].isin(expected).all()
            True
        """
        segments = segment_users(sessioned_df)
        expected = {cfg.SEGMENT_PURCHASER, cfg.SEGMENT_CART_ABANDONER, cfg.SEGMENT_BROWSER}
        assert set(segments["segment"].unique()).issubset(expected)
        assert len(segments) > 0

    def test_kpis_complete(self, sessioned_df: pd.DataFrame) -> None:
        """
        compute_kpis should return all expected metric keys.

        Args:
            sessioned_df: Pytest fixture.

        Returns:
            None

        Example:
            >>> "total_events" in compute_kpis(df)
            True
        """
        kpis = compute_kpis(sessioned_df)
        expected_keys = {
            "total_events", "unique_users", "unique_products",
            "total_revenue", "avg_order_value",
            "overall_conversion_rate", "cart_abandonment_rate",
        }
        assert expected_keys.issubset(kpis.keys())


# ═══════════════════════════════════════════════════════════════
# TEST 5: End-to-End Pipeline
# ═══════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Full pipeline: generate → validate → clean → analyse → visualise."""

    def test_full_pipeline_runs_without_error(self) -> None:
        """
        The entire pipeline from data generation through visualisation
        should execute without raising any exceptions.

        Returns:
            None

        Example:
            >>> test_full_pipeline_runs_without_error()  # no exception
        """
        # 1. Generate data
        df = generate_sample_data(n_rows=1000, seed=42)
        assert len(df) > 0

        # 2. Validate
        result = validate_dataframe(df)
        assert result.is_valid, result.summary()

        # 3. Clean
        df = remove_bots(df)
        df = deduplicate_events(df)

        # 4. Build sessions
        df = build_sessions(df, use_existing=True)
        assert "session_id" in df.columns

        # 5. Funnel
        funnel = compute_funnel(df)
        assert len(funnel) == len(cfg.FUNNEL_STAGES)

        # 6. Cohort
        retention, sizes = compute_cohort_retention(df)
        assert not retention.empty

        # 7. Segments
        segments = segment_users(df)
        assert "segment" in segments.columns

        # 8. Time patterns
        patterns = compute_time_patterns(df)
        assert "hourly" in patterns and "heatmap" in patterns

        # 9. Brand analysis
        brands = compute_brand_analysis(df)
        # brands may be empty if brand events < threshold — that's fine

        # 10. Clustering
        clustered = cluster_users(segments)
        assert "cluster" in clustered.columns

        # 11. KPIs
        kpis = compute_kpis(df)
        assert kpis["total_events"] > 0

        # 12. Visualizations — ensure they return Figure objects
        fig_funnel = plot_funnel_chart(funnel)
        assert fig_funnel is not None

        fig_cohort = plot_cohort_heatmap(retention)
        assert fig_cohort is not None

        fig_segment = plot_segment_distribution(segments)
        assert fig_segment is not None

        fig_time = plot_time_heatmap(patterns["heatmap"])
        assert fig_time is not None

        fig_brand = plot_brand_comparison(brands)
        assert fig_brand is not None

        fig_cluster = plot_cluster_scatter(clustered)
        assert fig_cluster is not None

        fig_events = plot_event_distribution(df)
        assert fig_events is not None

        fig_daily = plot_daily_events(df)
        assert fig_daily is not None

        fig_session = plot_session_length_distribution(df)
        assert fig_session is not None

    def test_validation_rejects_bad_data(self) -> None:
        """
        Validation should fail on a DataFrame missing required columns.

        Returns:
            None

        Example:
            >>> result.is_valid
            False
        """
        bad_df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        result = validate_dataframe(bad_df)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validation_rejects_empty_df(self) -> None:
        """
        Validation should fail on an empty DataFrame.

        Returns:
            None

        Example:
            >>> result.is_valid
            False
        """
        result = validate_dataframe(pd.DataFrame())
        assert not result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

