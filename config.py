"""
Centralized configuration for E-commerce Customer Behavior Analysis.

Loads every tuneable parameter from environment variables (.env file)
so that nothing is hard-coded in the business-logic modules.
Call ``check_keys()`` at startup to verify the configuration is sane.
"""

import os
import logging
from typing import Dict, List

from dotenv import load_dotenv

# ── Load .env once at import time ────────────────────────────
load_dotenv()

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Application Settings
# ═══════════════════════════════════════════════════════════════
APP_TITLE: str = os.getenv("APP_TITLE", "E-commerce Customer Behavior Analysis")
APP_ICON: str = os.getenv("APP_ICON", "🛒")
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# ═══════════════════════════════════════════════════════════════
# Data Configuration
# ═══════════════════════════════════════════════════════════════
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
SAMPLE_DATA_ROWS: int = int(os.getenv("SAMPLE_DATA_ROWS", "50000"))


# ═══════════════════════════════════════════════════════════════
# Column Names — single source of truth for DataFrame columns
# ═══════════════════════════════════════════════════════════════
EVENT_TIME_COL: str = "event_time"
EVENT_TYPE_COL: str = "event_type"
PRODUCT_ID_COL: str = "product_id"
USER_ID_COL: str = "user_id"
PRICE_COL: str = "price"
BRAND_COL: str = "brand"
CATEGORY_COL: str = "category_code"
SESSION_COL: str = "user_session"

REQUIRED_COLUMNS: List[str] = [
    EVENT_TIME_COL,
    EVENT_TYPE_COL,
    PRODUCT_ID_COL,
    USER_ID_COL,
    PRICE_COL,
]

OPTIONAL_COLUMNS: List[str] = [
    "category_id",
    CATEGORY_COL,
    BRAND_COL,
    SESSION_COL,
]


# ═══════════════════════════════════════════════════════════════
# Event Types & Funnel Stages
# ═══════════════════════════════════════════════════════════════
VALID_EVENT_TYPES: List[str] = ["view", "cart", "remove_from_cart", "purchase"]

# Ordered funnel stages used for conversion-rate computation
FUNNEL_STAGES: List[str] = ["view", "cart", "purchase"]

FUNNEL_STAGE_LABELS: Dict[str, str] = {
    "view": "Product View",
    "cart": "Add to Cart",
    "purchase": "Purchase",
}

# Numeric ordering for determining "max stage reached" per session
FUNNEL_STAGE_ORDER: Dict[str, int] = {
    "view": 1,
    "cart": 2,
    "remove_from_cart": 2,  # treated as same level as cart
    "purchase": 3,
}


# ═══════════════════════════════════════════════════════════════
# Session Construction Parameters
# ═══════════════════════════════════════════════════════════════
SESSION_GAP_MINUTES: int = int(os.getenv("SESSION_GAP_MINUTES", "30"))
BOT_THRESHOLD_EVENTS_PER_DAY: int = int(os.getenv("BOT_THRESHOLD", "1000"))


# ═══════════════════════════════════════════════════════════════
# Analysis Configuration
# ═══════════════════════════════════════════════════════════════
COHORT_PERIODS: int = int(os.getenv("COHORT_PERIODS", "12"))
N_CLUSTERS: int = int(os.getenv("N_CLUSTERS", "4"))
TOP_N_BRANDS: int = int(os.getenv("TOP_N_BRANDS", "15"))
MIN_BRAND_EVENTS: int = int(os.getenv("MIN_BRAND_EVENTS", "100"))


# ═══════════════════════════════════════════════════════════════
# Visualization Settings
# ═══════════════════════════════════════════════════════════════
PLOTLY_TEMPLATE: str = os.getenv("PLOTLY_TEMPLATE", "plotly_white")
COLOR_PALETTE: List[str] = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
]
HEATMAP_COLORSCALE: str = os.getenv("HEATMAP_COLORSCALE", "YlOrRd")
FUNNEL_COLORS: List[str] = ["#636EFA", "#FFA15A", "#00CC96"]


# ═══════════════════════════════════════════════════════════════
# Behaviour-Segment Labels
# ═══════════════════════════════════════════════════════════════
SEGMENT_PURCHASER: str = "Purchaser"
SEGMENT_CART_ABANDONER: str = "Cart Abandoner"
SEGMENT_BROWSER: str = "Browser"


# ═══════════════════════════════════════════════════════════════
# Logging Helpers
# ═══════════════════════════════════════════════════════════════
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """
    Configure root logger using settings from the environment.

    Returns:
        None

    Example:
        >>> setup_logging()
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )


def check_keys() -> Dict[str, bool]:
    """
    Validate that every critical configuration value is set and sensible.

    Returns:
        Dict[str, bool]: Mapping of config key → validity flag.

    Example:
        >>> status = check_keys()
        >>> assert all(status.values()), "Some config keys are invalid"
    """
    checks: Dict[str, bool] = {
        "APP_TITLE": bool(APP_TITLE),
        "REQUIRED_COLUMNS": len(REQUIRED_COLUMNS) > 0,
        "FUNNEL_STAGES": len(FUNNEL_STAGES) > 0,
        "SESSION_GAP_MINUTES": SESSION_GAP_MINUTES > 0,
        "BOT_THRESHOLD": BOT_THRESHOLD_EVENTS_PER_DAY > 0,
        "SAMPLE_DATA_ROWS": SAMPLE_DATA_ROWS > 0,
        "N_CLUSTERS": N_CLUSTERS >= 2,
        "MAX_UPLOAD_SIZE_MB": MAX_UPLOAD_SIZE_MB > 0,
    }

    for key, valid in checks.items():
        if not valid:
            logger.warning("Configuration check FAILED for: %s", key)
        else:
            logger.debug("Configuration check passed for: %s", key)

    return checks

