import logging
from datetime import datetime
from typing import Any, Optional

def normalize_date(value: str) -> Optional[str]:
    """
    Parses a date string and normalizes it to YYYY-MM-DD format.

    Args:
        value: The raw date string extracted.

    Returns:
        The normalized date string or None if parsing fails.
    """
    if not value:
        return None
    try:
        # Attempt to parse common date formats
        # This can be expanded with more formats if needed
        dt_object = datetime.strptime(value, '%B %d, %Y')
        return dt_object.strftime('%Y-%m-%d')
    except ValueError:
        logging.warning(f"Could not parse date: {value}. Returning raw value.")
        return value

def normalize_currency(value: str) -> Optional[float]:
    """
    Converts a currency string (e.g., "$5,000.00") to a float.

    Args:
        value: The raw currency string.

    Returns:
        The normalized float value or None if parsing fails.
    """
    if not value:
        return None
    try:
        # Remove currency symbols and commas
        cleaned_value = value.replace('$', '').replace(',', '').strip()
        return float(cleaned_value)
    except (ValueError, TypeError):
        logging.warning(f"Could not parse currency: {value}. Returning raw value.")
        return value