"""
Utility package for E-commerce Customer Behavior Analysis.

Sub-modules
-----------
data_loader      Load and generate e-commerce event data.
session_builder  Construct sessions, remove bots, deduplicate events.
analyzer         Funnel, cohort, segmentation, and time-pattern analysis.
visualizer       Build Plotly / Seaborn charts for the dashboard.
validator        Validate DataFrame schema and data quality.
"""

from utils.data_loader import load_csv, generate_sample_data
from utils.session_builder import remove_bots, build_sessions, deduplicate_events
from utils.analyzer import (
    compute_funnel,
    compute_cohort_retention,
    segment_users,
    compute_time_patterns,
    compute_brand_analysis,
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

__all__ = [
    # data_loader
    "load_csv",
    "generate_sample_data",
    # session_builder
    "remove_bots",
    "build_sessions",
    "deduplicate_events",
    # analyzer
    "compute_funnel",
    "compute_cohort_retention",
    "segment_users",
    "compute_time_patterns",
    "compute_brand_analysis",
    "cluster_users",
    # visualizer
    "plot_funnel_chart",
    "plot_cohort_heatmap",
    "plot_segment_distribution",
    "plot_time_heatmap",
    "plot_brand_comparison",
    "plot_cluster_scatter",
    "plot_event_distribution",
    "plot_daily_events",
    "plot_session_length_distribution",
    # validator
    "validate_dataframe",
    "ValidationResult",
]

