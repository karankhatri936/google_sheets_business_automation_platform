"""Business data schema.

This module is the single source of truth for *what* the incoming business data
must look like: column names, their expected types, the allowed categorical
values and the header aliases that are accepted from Google Sheets.

Keeping this separate from the validation/cleaning code means the rules are
declarative, testable and easy to extend without touching engine logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColumnKind = Literal["date", "text", "integer", "numeric", "category"]

# Statuses that do not represent realised revenue. Used by the analytics layer
# for the "net revenue" metric - see docs/BUSINESS_RULES.md.
NON_REVENUE_STATUSES: frozenset[str] = frozenset({"Cancelled", "Refunded"})

# Explicitly supported date formats, tried in order. Ambiguous formats
# (for example 03/04/2025) are intentionally NOT accepted - see README
# "Limitations".
DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%d %b %Y",
    "%d.%m.%Y",
    "%Y-%m-%dT%H:%M:%S",
)


@dataclass(frozen=True)
class ColumnSpec:
    """Declarative description of one business column."""

    name: str
    kind: ColumnKind
    required: bool = True
    aliases: tuple[str, ...] = ()
    allowed_values: frozenset[str] | None = None
    minimum: float | None = None
    maximum: float | None = None
    # Rule for missing values: None means "record is invalid when missing".
    default_on_missing: str | None = None
    # Columns whose value is recalculated deterministically during cleaning.
    derived: bool = False
    description: str = ""

    @property
    def accepted_headers(self) -> tuple[str, ...]:
        """Header spellings accepted for this column (canonical name first)."""
        return (self.name, *self.aliases)


# Canonical categorical values. Anything not listed (after alias/case
# normalisation) makes the record invalid, so bad values surface in the
# Data_Quality sheet instead of silently polluting the analytics.
REGION_VALUES: frozenset[str] = frozenset({"North", "South", "East", "West"})
STATUS_VALUES: frozenset[str] = frozenset({"Completed", "Pending", "Cancelled", "Refunded"})
CATEGORY_VALUES: frozenset[str] = frozenset(
    {"Electronics", "Home & Kitchen", "Office Supplies", "Outdoors", "Apparel"}
)

# Value-level aliases used for categorical normalisation (case-insensitive).
CATEGORY_ALIASES: dict[str, str] = {
    "electronics": "Electronics",
    "electronic": "Electronics",
    "home and kitchen": "Home & Kitchen",
    "home & kitchen": "Home & Kitchen",
    "home-kitchen": "Home & Kitchen",
    "kitchen": "Home & Kitchen",
    "office supplies": "Office Supplies",
    "office-supplies": "Office Supplies",
    "officesupplies": "Office Supplies",
    "outdoor": "Outdoors",
    "outdoors": "Outdoors",
    "apparel": "Apparel",
    "clothing": "Apparel",
}

REGION_ALIASES: dict[str, str] = {
    "north": "North",
    "n": "North",
    "south": "South",
    "s": "South",
    "east": "East",
    "e": "East",
    "west": "West",
    "w": "West",
}

STATUS_ALIASES: dict[str, str] = {
    "completed": "Completed",
    "complete": "Completed",
    "closed": "Completed",
    "pending": "Pending",
    "in progress": "Pending",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
    "refunded": "Refunded",
    "refund": "Refunded",
}

CATEGORY_ALIAS_MAP: dict[str, dict[str, str]] = {
    "category": CATEGORY_ALIASES,
    "region": REGION_ALIASES,
    "status": STATUS_ALIASES,
}

BUSINESS_SCHEMA: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        name="date",
        kind="date",
        required=True,
        aliases=("order_date", "date_of_order", "order date"),
        description="Date the order was placed (used for period analysis).",
    ),
    ColumnSpec(
        name="order_id",
        kind="text",
        required=True,
        aliases=("order", "order_no", "order_number", "order id"),
        description="Business key of the order. Duplicates are reported and de-duplicated.",
    ),
    ColumnSpec(
        name="customer",
        kind="text",
        required=True,
        aliases=("customer_name", "client", "customer name"),
        default_on_missing="Unknown Customer",
        description="Customer name. Missing values are replaced with the documented default.",
    ),
    ColumnSpec(
        name="product",
        kind="text",
        required=True,
        aliases=("product_name", "item", "sku_name"),
        description="Product name. Missing values make the record invalid.",
    ),
    ColumnSpec(
        name="category",
        kind="category",
        required=True,
        aliases=("product_category", "segment"),
        allowed_values=CATEGORY_VALUES,
        description="Product category (canonical values only, aliases normalised).",
    ),
    ColumnSpec(
        name="quantity",
        kind="integer",
        required=True,
        aliases=("qty", "units", "units_sold"),
        minimum=1,
        description="Units sold. Must be a whole number >= 1.",
    ),
    ColumnSpec(
        name="unit_price",
        kind="numeric",
        required=True,
        aliases=("price", "price_per_unit", "unit cost"),
        minimum=0,
        description="Price per unit. Must be numeric and >= 0.",
    ),
    ColumnSpec(
        name="revenue",
        kind="numeric",
        required=False,
        aliases=("sales", "total", "amount"),
        minimum=0,
        derived=True,
        description="Recalculated as quantity * unit_price; discrepancies are reported.",
    ),
    ColumnSpec(
        name="region",
        kind="category",
        required=True,
        aliases=("sales_region", "territory", "area"),
        allowed_values=REGION_VALUES,
        description="Sales region (canonical values only, aliases normalised).",
    ),
    ColumnSpec(
        name="status",
        kind="category",
        required=True,
        aliases=("order_status", "state"),
        allowed_values=STATUS_VALUES,
        description="Order lifecycle status.",
    ),
    ColumnSpec(
        name="feedback",
        kind="text",
        required=False,
        aliases=("comment", "comments", "customer_feedback", "notes"),
        description="Optional free-text customer feedback (used only by the AI classifier).",
    ),
)

SCHEMA_BY_NAME: dict[str, ColumnSpec] = {spec.name: spec for spec in BUSINESS_SCHEMA}

REQUIRED_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in BUSINESS_SCHEMA if spec.required)
OPTIONAL_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in BUSINESS_SCHEMA if not spec.required)
ALL_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in BUSINESS_SCHEMA)
