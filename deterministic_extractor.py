import re
from typing import List, Tuple, Optional

# Confidence levels
HIGH_CONFIDENCE = "high"
LOW_CONFIDENCE = "low"
AMBIGUOUS = "ambiguous"
NOT_FOUND = "not_found"

def extract_entity(full_text: str, entity_type: str) -> Tuple[Optional[str], str]:
    """
    Extracts a named entity (like Tenant or Landlord) from the text.

    It prioritizes clear, anchored patterns like "Tenant: [Name]". If multiple
    such patterns are found, it flags the result as ambiguous. As a fallback,
    it looks for "between [Party A] and [Party B]" patterns, which are treated
    as low confidence since the roles aren't specified.

    Args:
        full_text: The entire text of the document.
        entity_type: The type of entity to search for (e.g., "Tenant", "Landlord").

    Returns:
        A tuple containing the extracted value (or None) and a confidence level.
    """
    # Pattern for specific, anchored definitions like "Landlord: [Name]"
    # The (?i) flag makes it case-insensitive. It captures the rest of the line.
    specific_pattern = re.compile(rf"(?i)^\s*{entity_type}:\s*([^\n]+)", re.MULTILINE)
    matches = specific_pattern.findall(full_text)

    if len(matches) == 1:
        return (matches[0].strip(), HIGH_CONFIDENCE)
    if len(matches) > 1:
        # If we find "Tenant: A" and later "Tenant: B", it's ambiguous.
        return ("; ".join(m.strip() for m in matches), AMBIGUOUS)

    # Fallback pattern for "This lease is made... between [PARTY 1] and [PARTY 2]"
    # This is less reliable because it doesn't assign roles.
    between_pattern = re.compile(r"(?i)between\s+(.+?)\s+and\s+(.+?)(?:,?\s+hereinafter|\n)")
    between_matches = between_pattern.findall(full_text)

    if len(between_matches) == 1:
        # We found the parties but don't know who is who. Return as low confidence.
        parties = [p.strip().replace("\n", " ") for p in between_matches[0][:2]]
        return (f"{parties[0]} or {parties[1]}", LOW_CONFIDENCE)
    if len(between_matches) > 1:
        return ("Multiple 'between' clauses found", AMBIGUOUS)

    return (None, NOT_FOUND)


def extract_rent(full_text: str) -> Tuple[Optional[str], str]:
    """
    Extracts rent amount using regex.

    Searches for phrases like "Base Rent of $5,000.00" or "monthly rent shall be...".
    It collects all unique values found. If only one is found, confidence is high.
    If multiple different values are found, it's ambiguous.

    Args:
        full_text: The entire text of the document.

    Returns:
        A tuple containing the extracted value (or None) and a confidence level.
    """
    # This pattern looks for different rent-related keywords followed by a dollar amount.
    pattern = re.compile(
        r"(?i)(?:base|annual|monthly)\s+rent(?:|s)\s+(?:of|is|shall\s+be)\s+([\$€£]\s*[\d,]+\.?\d*)"
    )
    matches = pattern.findall(full_text)

    # Use a set to find unique rent amounts mentioned.
    unique_matches = list(set(matches))

    if len(unique_matches) == 1:
        return (unique_matches[0], HIGH_CONFIDENCE)
    if len(unique_matches) > 1:
        return ("; ".join(unique_matches), AMBIGUOUS)

    return (None, NOT_FOUND)


def extract_lease_term(full_text: str) -> Tuple[Optional[str], str]:
    """
    Extracts the lease term (e.g., "ten (10) years").

    Args:
        full_text: The entire text of the document.

    Returns:
        A tuple containing the extracted value (or None) and a confidence level.
    """
    # Looks for "Term of X years/months"
    pattern = re.compile(r"(?i)term\s+of\s+([\w\s\(\)]+)\s+(year|month)")
    matches = pattern.findall(full_text)

    unique_matches = list(set([f"{m[0].strip()} {m[1]}" for m in matches]))

    if len(unique_matches) == 1:
        return (unique_matches[0], HIGH_CONFIDENCE)
    if len(unique_matches) > 1:
        return ("; ".join(unique_matches), AMBIGUOUS)

    return (None, NOT_FOUND)


def extract_security_deposit(full_text: str) -> Tuple[Optional[str], str]:
    """
    Extracts the security deposit amount using regex.
    Looks for dollar amounts near "Security Deposit".

    Args:
        full_text: The entire text of the document.

    Returns:
        A tuple containing the extracted value (or None) and a confidence level.
    """
    # This pattern looks for "security deposit" followed by a dollar amount.
    pattern = re.compile(
        r"(?i)security\s+deposit\s+(?:of|is|shall\s+be)\s+([\$€£]\s*[\d,]+\.?\d*)"
    )
    matches = pattern.findall(full_text)

    # Use a set to find unique deposit amounts mentioned.
    unique_matches = list(set(matches))

    if len(unique_matches) == 1:
        return (unique_matches[0], HIGH_CONFIDENCE)
    if len(unique_matches) > 1:
        return ("; ".join(unique_matches), AMBIGUOUS)

    return (None, NOT_FOUND)