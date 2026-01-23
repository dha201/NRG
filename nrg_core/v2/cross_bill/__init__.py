"""
Cross-Bill Reference Handling.

Detects and resolves references to external statutes, bills, and legal codes
within legislative text.

Components:
- ReferenceDetector: Pattern-based detection of statutory references (Tier 1)
- USCODELookup: API-based resolution of U.S. Code references (Tier 2)
- BillReference: Data model for detected references
- ResolvedDefinition: Data model for resolved statute text

Two-tier architecture:
- Tier 1 (Detection): Fast regex-based pattern matching, no API calls
- Tier 2 (Resolution): On-demand USCODE API lookup for full text

Why two tiers:
- Most bills don't need full resolution (save API calls)
- Detection is cheap, resolution is expensive
- Enables selective resolution based on analysis needs
"""
from nrg_core.v2.cross_bill.reference_detector import (
    ReferenceDetector,
    BillReference
)
from nrg_core.v2.cross_bill.uscode_lookup import (
    USCODELookup,
    ResolvedDefinition
)

__all__ = [
    "ReferenceDetector",
    "BillReference",
    "USCODELookup",
    "ResolvedDefinition"
]
