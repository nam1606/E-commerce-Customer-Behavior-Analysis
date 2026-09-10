"""
Data-quality validation for e-commerce event DataFrames.

Checks schema conformance, value ranges, and data integrity before
the data enters the analysis pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Validation Result Container
# ═══════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """
    Holds the outcome of a DataFrame validation run.

    Attributes:
        is_valid: True if all critical checks passed.
        errors: List of critical errors that block analysis.
        warnings: List of non-critical issues the user should know about.
        row_count: Total rows in the validated DataFrame.
        column_count: Total columns.

    Example:
        >>> result = validate_dataframe(df)
        >>> if result.is_valid:
        ...     print("Good to go!")
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0

    def summary(self) -> str:
        """
        Return a human-readable summary string.

        Returns:
            str: Multi-line summary of validation outcome.

        Example:
            >>> print(result.summary())
            ✅ Validation PASSED (50000 rows × 8 cols)
        """
        status = "✅ PASSED" if self.is_valid else "❌ FAILED"
        lines = [
            f"Validation {status} ({self.row_count:,} rows × {self.column_count} cols)"
        ]
        for err in self.errors:
            lines.append(f"  ❌ {err}")
        for warn in self.warnings:
            lines.append(f"  ⚠️ {warn}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Main Validation Function
# ═══════════════════════════════════════════════════════════════

def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Run all data-quality checks on a raw event DataFrame.

    Checks performed:
        1. DataFrame is not empty.
        2. All required columns exist.
        3. ``event_type`` values are recognised.
        4. ``price`` column is numeric and non-negative.
        5. No excessive null rates in critical columns.
        6. ``event_time`` is parseable as datetime.

    Args:
        df: Raw event DataFrame to validate.

    Returns:
        ValidationResult: Container with pass/fail status and messages.

    Example:
        >>> result = validate_dataframe(df)
        >>> result.is_valid
        True
    """
    result = ValidationResult()

    try:
        result.row_count = len(df)
        result.column_count = len(df.columns)

        # ── Check 1: Non-empty ───────────────────────────────
        if df.empty:
            result.is_valid = False
            result.errors.append("DataFrame is empty — no rows to analyse.")
            logger.error("Validation failed: DataFrame is empty")
            return result

        # ── Check 2: Required columns ────────────────────────
        missing_cols = [
            col for col in cfg.REQUIRED_COLUMNS if col not in df.columns
        ]
        if missing_cols:
            result.is_valid = False
            result.errors.append(
                f"Missing required columns: {missing_cols}"
            )
            logger.error("Missing columns: %s", missing_cols)

        # Warn about missing optional columns (non-blocking)
        missing_optional = [
            col for col in cfg.OPTIONAL_COLUMNS if col not in df.columns
        ]
        if missing_optional:
            result.warnings.append(
                f"Missing optional columns (some features may be limited): {missing_optional}"
            )

        # ── Check 3: Event-type values ───────────────────────
        if cfg.EVENT_TYPE_COL in df.columns:
            unknown_types = set(df[cfg.EVENT_TYPE_COL].dropna().unique()) - set(
                cfg.VALID_EVENT_TYPES
            )
            if unknown_types:
                result.warnings.append(
                    f"Unknown event types found (will be ignored): {unknown_types}"
                )
                logger.warning("Unknown event types: %s", unknown_types)

            # Critical: at least "view" events must exist
            if "view" not in df[cfg.EVENT_TYPE_COL].values:
                result.is_valid = False
                result.errors.append(
                    "No 'view' events found — cannot build funnel."
                )

        # ── Check 4: Price column ────────────────────────────
        if cfg.PRICE_COL in df.columns:
            price_series = pd.to_numeric(df[cfg.PRICE_COL], errors="coerce")
            null_pct = price_series.isna().mean() * 100
            if null_pct > 50:
                result.warnings.append(
                    f"Price column has {null_pct:.1f}% missing/non-numeric values."
                )
            neg_count = (price_series < 0).sum()
            if neg_count > 0:
                result.warnings.append(
                    f"Price column has {neg_count:,} negative values."
                )

        # ── Check 5: Null rates in critical columns ──────────
        for col in cfg.REQUIRED_COLUMNS:
            if col in df.columns:
                null_pct = df[col].isna().mean() * 100
                if null_pct > 30:
                    result.warnings.append(
                        f"Column '{col}' has {null_pct:.1f}% null values."
                    )
                if null_pct > 80:
                    result.is_valid = False
                    result.errors.append(
                        f"Column '{col}' is >80% null — data is unusable."
                    )

        # ── Check 6: Datetime parsing ────────────────────────
        if cfg.EVENT_TIME_COL in df.columns:
            try:
                sample = df[cfg.EVENT_TIME_COL].head(100)
                pd.to_datetime(sample, format="mixed")
            except (ValueError, TypeError):
                result.is_valid = False
                result.errors.append(
                    f"Cannot parse '{cfg.EVENT_TIME_COL}' as datetime."
                )

        # ── Final log ────────────────────────────────────────
        if result.is_valid:
            logger.info("Validation PASSED (%d rows)", result.row_count)
        else:
            logger.warning(
                "Validation FAILED — %d error(s)", len(result.errors)
            )

    except Exception as exc:
        result.is_valid = False
        result.errors.append(f"Unexpected validation error: {exc}")
        logger.error("Validation crashed: %s", exc)

    return result


# ═══════════════════════════════════════════════════════════════
# Column Coercion Helper
# ═══════════════════════════════════════════════════════════════

def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort column type coercion and cleanup.

    * Strips whitespace from string columns.
    * Converts ``price`` to float.
    * Converts ``user_id`` and ``product_id`` to int-safe strings.

    Args:
        df: Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame (copy).

    Example:
        >>> clean = coerce_columns(raw_df)
        >>> clean[cfg.PRICE_COL].dtype
        dtype('float64')
    """
    df = df.copy()

    try:
        # Strip whitespace from object columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        # Numeric coercion for price
        if cfg.PRICE_COL in df.columns:
            df[cfg.PRICE_COL] = pd.to_numeric(
                df[cfg.PRICE_COL], errors="coerce"
            )

        # Ensure user_id and product_id are consistent types
        for col in [cfg.USER_ID_COL, cfg.PRODUCT_ID_COL]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        logger.info("Column coercion complete")

    except Exception as exc:
        logger.warning("Column coercion partially failed: %s", exc)

    return df

