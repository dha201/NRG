"""
Tests for ReferenceDetector.

Validates:
- U.S. Code citation detection (e.g., "26 U.S.C. 48")
- "amends" pattern detection
- "as defined in" pattern detection
- Section location tracking
- Deduplication of repeated references
"""
import pytest
from nrg_core.v2.cross_bill.reference_detector import ReferenceDetector, BillReference


def test_detect_usc_citations():
    """Should detect U.S. Code citations."""
    detector = ReferenceDetector()
    
    bill_text = """
    Section 2.1: This Act amends section 48 of the Internal Revenue Code 
    of 1986 (26 U.S.C. 48) to extend renewable energy tax credits.
    
    Section 3.1: As defined in 42 U.S.C. 7401, clean air standards apply.
    """
    
    refs = detector.detect(bill_text)
    
    assert len(refs) >= 2
    
    # Check for 26 U.S.C. 48
    usc_refs = [r for r in refs if r.citation == "26 U.S.C. 48"]
    assert len(usc_refs) == 1
    assert usc_refs[0].reference_type == "statutory_amendment"
    assert usc_refs[0].location == "Section 2.1:"


def test_detect_amends_pattern():
    """Should detect 'amends' pattern."""
    detector = ReferenceDetector()
    
    bill_text = "This Act amends the Clean Air Act to add new emission standards."
    
    refs = detector.detect(bill_text)
    
    assert any(r.citation == "the Clean Air Act" for r in refs)
    assert any(r.reference_type == "statutory_amendment" for r in refs)


def test_detect_as_defined_in():
    """Should detect 'as defined in' pattern."""
    detector = ReferenceDetector()
    
    bill_text = "The term 'renewable energy' as defined in 26 U.S.C. 45 applies."
    
    refs = detector.detect(bill_text)
    
    # Should detect both the "as defined in" and the USC citation
    defined_refs = [r for r in refs if r.reference_type == "definition_by_reference"]
    assert len(defined_refs) >= 1


def test_detect_public_law():
    """Should detect Public Law citations."""
    detector = ReferenceDetector()
    
    bill_text = "Pursuant to Public Law 117-169, the following provisions apply."
    
    refs = detector.detect(bill_text)
    
    pl_refs = [r for r in refs if "117-169" in r.citation]
    assert len(pl_refs) >= 1
    assert pl_refs[0].reference_type == "precedent_citation"


def test_deduplicates_repeated_references():
    """Should deduplicate repeated references to same citation."""
    detector = ReferenceDetector()
    
    bill_text = """
    Section 1: Under 26 U.S.C. 48, credits apply.
    Section 2: The provisions of 26 U.S.C. 48 are modified.
    Section 3: See 26 U.S.C. 48 for definitions.
    """
    
    refs = detector.detect(bill_text)
    
    # Should only have one reference to 26 U.S.C. 48
    usc_48_refs = [r for r in refs if r.citation == "26 U.S.C. 48"]
    assert len(usc_48_refs) == 1


def test_extracts_context():
    """Should extract surrounding context for each reference."""
    detector = ReferenceDetector()
    
    bill_text = "The investment tax credit under 26 U.S.C. 48 provides incentives for renewable energy."
    
    refs = detector.detect(bill_text)
    
    assert len(refs) >= 1
    assert len(refs[0].context) > 0
    assert "investment tax credit" in refs[0].context.lower() or "renewable" in refs[0].context.lower()
