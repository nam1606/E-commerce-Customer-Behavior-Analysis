"""
Data loading and synthetic-data generation for e-commerce event analysis.

Supports:
* Loading CSV files (uploaded or local path).
* Generating realistic sample data so the project works out-of-the-box
  without downloading an external dataset.
"""

import logging
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CSV Loader
# ═══════════════════════════════════════════════════════════════

def load_csv(
    source: Union[str, Path, StringIO, "streamlit.runtime.uploaded_file_manager.UploadedFile"],
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load a CSV file from disk, a file-like object, or a Streamlit UploadedFile.

    Automatically parses the event_time column into datetime and
    lowercases the event_type column for consistency.

    Args:
        source: File path (str / Path) or file-like object (StringIO /
                Streamlit UploadedFile).
        parse_dates: If True, convert the event_time column to datetime.

    Returns:
        pd.DataFrame: Raw event DataFrame.

    Raises:
        FileNotFoundError: If a file path is given but does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        ValueError: If required columns are missing after loading.

    Example:
        >>> df = load_csv("data/events.csv")
        >>> df.columns.tolist()
        ['event_time', 'event_type', 'product_id', 'user_id', 'price', ...]
    """
    try:
        # Read CSV — handle both paths and file-like objects
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {path}")
            logger.info("Loading CSV from path: %s", path)
            df = pd.read_csv(path, low_memory=False)
        else:
            # Streamlit UploadedFile or StringIO
            logger.info("Loading CSV from file-like object")
            df = pd.read_csv(source, low_memory=False)

    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        raise
    except UnicodeDecodeError as exc:
        logger.error("Encoding error while reading CSV: %s", exc)
        raise ValueError(f"Could not decode CSV — try saving as UTF-8: {exc}") from exc

    # Standardise column names: strip whitespace, lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Normalise event_type values to lowercase for consistent matching
    if cfg.EVENT_TYPE_COL in df.columns:
        df[cfg.EVENT_TYPE_COL] = df[cfg.EVENT_TYPE_COL].str.strip().str.lower()

    # Parse timestamps if requested
    if parse_dates and cfg.EVENT_TIME_COL in df.columns:
        try:
            df[cfg.EVENT_TIME_COL] = pd.to_datetime(
                df[cfg.EVENT_TIME_COL], utc=True, format="mixed"
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Date parsing with UTC failed, retrying without UTC: %s", exc)
            try:
                df[cfg.EVENT_TIME_COL] = pd.to_datetime(
                    df[cfg.EVENT_TIME_COL], format="mixed"
                )
            except Exception as inner_exc:
                logger.error("Could not parse event_time column: %s", inner_exc)
                raise ValueError(
                    f"Cannot parse '{cfg.EVENT_TIME_COL}' as datetime"
                ) from inner_exc

    # Ensure price is numeric (some datasets store it as string)
    if cfg.PRICE_COL in df.columns:
        df[cfg.PRICE_COL] = pd.to_numeric(df[cfg.PRICE_COL], errors="coerce")

    logger.info(
        "Loaded %d rows × %d columns", df.shape[0], df.shape[1]
    )
    return df


# ═══════════════════════════════════════════════════════════════
# Synthetic Sample-Data Generator
# ═══════════════════════════════════════════════════════════════

def generate_sample_data(
    n_rows: int = cfg.SAMPLE_DATA_ROWS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic e-commerce clickstream dataset.

    The generator simulates users browsing a cosmetics shop: each user
    runs through one or more sessions, viewing products, optionally
    adding them to cart, and sometimes completing a purchase. This
    produces natural funnel drop-off patterns suitable for analysis.

    Args:
        n_rows: Approximate number of events to generate.
        seed:   Random seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic events with columns matching the schema
        expected by the rest of the pipeline.

    Example:
        >>> df = generate_sample_data(n_rows=1000)
        >>> set(df["event_type"].unique()) <= {"view", "cart", "remove_from_cart", "purchase"}
        True
    """
    rng = np.random.RandomState(seed)
    logger.info("Generating synthetic sample data (~%d rows)…", n_rows)

    # ── Product catalogue ────────────────────────────────────
    brands: list[str] = [
        "Maybelline", "L'Oreal", "Garnier", "Nivea", "Dove",
        "Clinique", "MAC", "NYX", "Revlon", "Estee Lauder",
        "Bobbi Brown", "Urban Decay", "Too Faced", "Neutrogena", "Olay",
    ]
    categories: list[str] = [
        "cosmetics.face.foundation", "cosmetics.lip.lipstick",
        "cosmetics.eye.mascara", "cosmetics.eye.eyeshadow",
        "skincare.face.cream", "skincare.body.lotion",
        "haircare.shampoo", "haircare.conditioner",
        "fragrance.women", "fragrance.men",
    ]

    n_products: int = 500
    product_ids = np.arange(1_000_000, 1_000_000 + n_products)
    product_brands = rng.choice(brands, size=n_products)
    product_categories = rng.choice(categories, size=n_products)
    # Log-normal price distribution mimics real-world skew
    product_prices = np.round(rng.lognormal(mean=2.5, sigma=0.8, size=n_products), 2)

    # ── User simulation ──────────────────────────────────────
    n_users: int = max(n_rows // 10, 200)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 4, 30)
    date_range_seconds = int((end_date - start_date).total_seconds())

    events: list[dict] = []

    for user_id in range(1, n_users + 1):
        # Geometric distribution gives some heavy and some light users
        n_sessions = min(int(rng.geometric(p=0.15)), 20)

        for sess_idx in range(n_sessions):
            session_start = start_date + timedelta(
                seconds=int(rng.randint(0, date_range_seconds))
            )
            session_id = f"{user_id}_{sess_idx}"

            # Number of product views per session (geometric → right-skewed)
            n_views = min(int(rng.geometric(p=0.2)), 15)
            viewed_indices = rng.choice(n_products, size=n_views, replace=True)

            current_time = session_start

            # ── Views ────────────────────────────────────────
            for prod_idx in viewed_indices:
                events.append({
                    cfg.EVENT_TIME_COL: current_time,
                    cfg.EVENT_TYPE_COL: "view",
                    cfg.PRODUCT_ID_COL: int(product_ids[prod_idx]),
                    cfg.USER_ID_COL: user_id,
                    cfg.PRICE_COL: float(product_prices[prod_idx]),
                    cfg.BRAND_COL: product_brands[prod_idx],
                    cfg.CATEGORY_COL: product_categories[prod_idx],
                    cfg.SESSION_COL: session_id,
                })
                # Random dwell time between page views
                current_time += timedelta(seconds=int(rng.randint(10, 300)))

            # ── Cart add (~25 % of sessions) ─────────────────
            if rng.random() < 0.25 and n_views > 0:
                cart_idx = int(rng.choice(viewed_indices))
                current_time += timedelta(seconds=int(rng.randint(5, 60)))
                events.append({
                    cfg.EVENT_TIME_COL: current_time,
                    cfg.EVENT_TYPE_COL: "cart",
                    cfg.PRODUCT_ID_COL: int(product_ids[cart_idx]),
                    cfg.USER_ID_COL: user_id,
                    cfg.PRICE_COL: float(product_prices[cart_idx]),
                    cfg.BRAND_COL: product_brands[cart_idx],
                    cfg.CATEGORY_COL: product_categories[cart_idx],
                    cfg.SESSION_COL: session_id,
                })

                # ── Purchase (~40 % of cart adds) ────────────
                if rng.random() < 0.40:
                    current_time += timedelta(seconds=int(rng.randint(30, 300)))
                    events.append({
                        cfg.EVENT_TIME_COL: current_time,
                        cfg.EVENT_TYPE_COL: "purchase",
                        cfg.PRODUCT_ID_COL: int(product_ids[cart_idx]),
                        cfg.USER_ID_COL: user_id,
                        cfg.PRICE_COL: float(product_prices[cart_idx]),
                        cfg.BRAND_COL: product_brands[cart_idx],
                        cfg.CATEGORY_COL: product_categories[cart_idx],
                        cfg.SESSION_COL: session_id,
                    })
                else:
                    # ── Remove from cart (~50 % of abandonments) ──
                    if rng.random() < 0.5:
                        current_time += timedelta(seconds=int(rng.randint(30, 180)))
                        events.append({
                            cfg.EVENT_TIME_COL: current_time,
                            cfg.EVENT_TYPE_COL: "remove_from_cart",
                            cfg.PRODUCT_ID_COL: int(product_ids[cart_idx]),
                            cfg.USER_ID_COL: user_id,
                            cfg.PRICE_COL: float(product_prices[cart_idx]),
                            cfg.BRAND_COL: product_brands[cart_idx],
                            cfg.CATEGORY_COL: product_categories[cart_idx],
                            cfg.SESSION_COL: session_id,
                        })

            # Stop early if we've generated enough rows
            if len(events) >= n_rows:
                break
        if len(events) >= n_rows:
            break

    df = pd.DataFrame(events[:n_rows])

    # Sort chronologically for downstream processing
    df.sort_values(by=[cfg.USER_ID_COL, cfg.EVENT_TIME_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Generated %d synthetic events for %d users", len(df), n_users)
    return df

