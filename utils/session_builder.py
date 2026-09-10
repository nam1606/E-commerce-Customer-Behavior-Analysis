"""
Session construction, bot removal, and event deduplication.

Transforms raw clickstream events into clean, session-annotated data
ready for funnel and cohort analysis.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Bot Removal
# ═══════════════════════════════════════════════════════════════

def remove_bots(
    df: pd.DataFrame,
    threshold: int = cfg.BOT_THRESHOLD_EVENTS_PER_DAY,
) -> pd.DataFrame:
    """
    Remove users whose daily event count exceeds a threshold.

    Automated scrapers and bots typically generate far more events than
    any human user.  Filtering them out prevents skewed metrics.

    Args:
        df: DataFrame with at least ``user_id`` and ``event_time`` columns.
        threshold: Maximum allowed events per user per day.

    Returns:
        pd.DataFrame: Filtered DataFrame with bot users excluded.

    Example:
        >>> clean = remove_bots(df, threshold=1000)
        >>> clean.shape[0] <= df.shape[0]
        True
    """
    try:
        # Derive date from event_time for daily aggregation
        df = df.copy()
        df["_date"] = pd.to_datetime(df[cfg.EVENT_TIME_COL]).dt.date

        # Count events per user per day
        daily_counts = (
            df.groupby([cfg.USER_ID_COL, "_date"])
            .size()
            .reset_index(name="_count")
        )

        # Identify bot user IDs (any day exceeding threshold flags the user)
        bot_ids = set(
            daily_counts.loc[
                daily_counts["_count"] > threshold, cfg.USER_ID_COL
            ]
        )

        n_bots = len(bot_ids)
        if n_bots > 0:
            logger.info(
                "Removed %d bot user(s) (>%d events/day)", n_bots, threshold
            )
        else:
            logger.info("No bot users detected (threshold=%d)", threshold)

        # Drop temporary column and bot rows in one step
        result = df.loc[~df[cfg.USER_ID_COL].isin(bot_ids)].drop(
            columns=["_date"]
        )
        return result.reset_index(drop=True)

    except KeyError as exc:
        logger.error("Missing column for bot removal: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════
# Session Construction
# ═══════════════════════════════════════════════════════════════

def build_sessions(
    df: pd.DataFrame,
    gap_minutes: int = cfg.SESSION_GAP_MINUTES,
    use_existing: bool = True,
) -> pd.DataFrame:
    """
    Assign a session identifier to every event.

    If the dataset already contains a ``user_session`` column **and**
    ``use_existing`` is True, that column is kept as-is. Otherwise a
    new session is constructed: any gap of more than ``gap_minutes``
    between consecutive events of the same user starts a new session.

    Args:
        df: DataFrame sorted by ``user_id`` and ``event_time``.
        gap_minutes: Inactivity threshold (minutes) that triggers a new
                     session.
        use_existing: When True and ``user_session`` already exists, skip
                      reconstruction.

    Returns:
        pd.DataFrame: Copy of the input with ``session_id`` and
        ``session_duration_sec`` columns appended.

    Example:
        >>> sessioned = build_sessions(df, gap_minutes=30)
        >>> "session_id" in sessioned.columns
        True
    """
    df = df.copy()

    # ── Decide whether to rebuild or reuse existing session IDs ──
    if use_existing and cfg.SESSION_COL in df.columns:
        logger.info("Using existing '%s' column for sessions", cfg.SESSION_COL)
        df["session_id"] = df[cfg.SESSION_COL].astype(str)
    else:
        logger.info(
            "Building sessions with %d-min inactivity gap", gap_minutes
        )
        try:
            # Sort to guarantee chronological order within each user
            df.sort_values(
                by=[cfg.USER_ID_COL, cfg.EVENT_TIME_COL], inplace=True
            )
            df.reset_index(drop=True, inplace=True)

            # Time difference to previous event for the same user
            df["_time_diff"] = df.groupby(cfg.USER_ID_COL)[
                cfg.EVENT_TIME_COL
            ].diff()

            # A new session starts whenever the gap exceeds the threshold
            gap_td = pd.Timedelta(minutes=gap_minutes)
            df["_new_session"] = (
                df["_time_diff"].isna() | (df["_time_diff"] > gap_td)
            )

            # Cumulative sum of new-session flags gives a monotonic session id
            df["session_id"] = df["_new_session"].cumsum().astype(str)

            # Housekeeping — drop helper columns
            df.drop(columns=["_time_diff", "_new_session"], inplace=True)

        except KeyError as exc:
            logger.error("Missing column for session building: %s", exc)
            raise

    # ── Compute per-session duration (seconds) for later analysis ──
    try:
        session_bounds = df.groupby("session_id")[cfg.EVENT_TIME_COL].agg(
            ["min", "max"]
        )
        session_bounds["session_duration_sec"] = (
            (session_bounds["max"] - session_bounds["min"]).dt.total_seconds()
        )
        duration_map = session_bounds["session_duration_sec"].to_dict()
        df["session_duration_sec"] = df["session_id"].map(duration_map)
    except Exception as exc:
        # Duration is nice-to-have; don't fail the whole pipeline
        logger.warning("Could not compute session duration: %s", exc)
        df["session_duration_sec"] = 0.0

    n_sessions = df["session_id"].nunique()
    logger.info("Total sessions constructed: %d", n_sessions)
    return df


# ═══════════════════════════════════════════════════════════════
# De-duplication
# ═══════════════════════════════════════════════════════════════

def deduplicate_events(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Remove exact duplicate events.

    Duplicates can arise from double-firing tracking pixels or
    replay-based data pipelines.

    Args:
        df: Event DataFrame.
        subset: Columns to consider when identifying duplicates.
                Defaults to [user_id, event_type, product_id, event_time].

    Returns:
        pd.DataFrame: De-duplicated DataFrame.

    Example:
        >>> deduped = deduplicate_events(df)
        >>> deduped.duplicated(subset=["user_id","event_type","product_id","event_time"]).sum()
        0
    """
    if subset is None:
        subset = [
            cfg.USER_ID_COL,
            cfg.EVENT_TYPE_COL,
            cfg.PRODUCT_ID_COL,
            cfg.EVENT_TIME_COL,
        ]

    # Keep only columns that actually exist (defensive against schemas
    # missing optional columns)
    available_subset = [c for c in subset if c in df.columns]
    if not available_subset:
        logger.warning("No columns available for deduplication — returning as-is")
        return df

    before = len(df)
    df = df.drop_duplicates(subset=available_subset, keep="first")
    after = len(df)

    removed = before - after
    if removed > 0:
        logger.info("Removed %d duplicate events (%.1f%%)", removed, removed / before * 100)
    else:
        logger.info("No duplicate events found")

    return df.reset_index(drop=True)

