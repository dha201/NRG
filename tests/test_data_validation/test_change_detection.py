"""
Change detection tests.

Tests validate hash-based change detection accuracy between cached and current bills.
Critical for ensuring we detect all changes and avoid false positives.
"""
import pytest
import json

from nrg_core.analysis.changes import (
    detect_bill_changes,
    compute_text_diff,
    CHANGE_TYPE_TEXT,
    CHANGE_TYPE_STATUS,
    CHANGE_TYPE_AMENDMENTS,
)
from nrg_core.db.cache import compute_bill_hash


class TestHashBasedChangeDetection:
    """Tests for hash-based text change detection."""
    
    def test_text_change_detected_by_hash(self, sample_cached_bill):
        """Different text should produce different hash and trigger change detection."""
        current_bill = {
            "number": "HB 1234",
            "source": "OpenStates",
            "title": "Energy Market Reform Act",
            "status": "Introduced",
            "summary": "Updated bill summary with new provisions."
        }
        
        result = detect_bill_changes(sample_cached_bill, current_bill)
        
        assert result["has_changes"] is True
        assert result["is_new"] is False
        assert result["change_type"] == "modified"
        
        text_changes = [c for c in result["changes"] if c["type"] == CHANGE_TYPE_TEXT]
        assert len(text_changes) == 1
    
    def test_identical_text_no_change_detected(self, sample_cached_bill):
        """Identical text should produce same hash and no change detected."""
        cached_data = json.loads(sample_cached_bill["full_data_json"])
        current_bill = {
            "number": "HB 1234",
            "source": "OpenStates",
            "title": "Energy Market Reform Act",
            "status": "Introduced",
            "summary": cached_data["summary"]
        }
        sample_cached_bill["text_hash"] = compute_bill_hash(cached_data["summary"])
        
        result = detect_bill_changes(sample_cached_bill, current_bill)
        
        assert result["has_changes"] is False
        assert result["change_type"] == "unchanged"
    
    def test_new_bill_no_changes_reported(self):
        """New bill (no cache) should report is_new=True, has_changes=False."""
        current_bill = {
            "number": "HB 9999",
            "source": "OpenStates",
            "title": "New Bill",
            "status": "Introduced",
            "summary": "Brand new bill text."
        }
        
        result = detect_bill_changes(None, current_bill)
        
        assert result["has_changes"] is False
        assert result["is_new"] is True
        assert result["change_type"] == "new_bill"
        assert result["changes"] == []


class TestStatusChangeDetection:
    """Tests for bill status change detection."""
    
    def test_status_change_detected(self, sample_cached_bill):
        """Status change from Introduced to Engrossed should be detected."""
        cached_data = json.loads(sample_cached_bill["full_data_json"])
        current_bill = {
            "number": "HB 1234",
            "source": "OpenStates",
            "title": "Energy Market Reform Act",
            "status": "Engrossed",
            "summary": cached_data["summary"]
        }
        sample_cached_bill["text_hash"] = compute_bill_hash(cached_data["summary"])
        
        result = detect_bill_changes(sample_cached_bill, current_bill)
        
        assert result["has_changes"] is True
        status_changes = [c for c in result["changes"] if c["type"] == CHANGE_TYPE_STATUS]
        assert len(status_changes) == 1
        assert status_changes[0]["old_value"] == "Introduced"
        assert status_changes[0]["new_value"] == "Engrossed"
    
    def test_same_status_no_change(self, sample_cached_bill):
        """Same status should not trigger status change."""
        cached_data = json.loads(sample_cached_bill["full_data_json"])
        current_bill = {
            "number": "HB 1234",
            "source": "OpenStates",
            "title": "Energy Market Reform Act",
            "status": "Introduced",
            "summary": cached_data["summary"]
        }
        sample_cached_bill["text_hash"] = compute_bill_hash(cached_data["summary"])
        
        result = detect_bill_changes(sample_cached_bill, current_bill)
        
        status_changes = [c for c in result["changes"] if c["type"] == CHANGE_TYPE_STATUS]
        assert len(status_changes) == 0


class TestAmendmentDetection:
    """Tests for new amendment detection."""
    
    def test_new_amendments_detected(self, sample_cached_bill):
        """New amendments should be detected."""
        cached_data = json.loads(sample_cached_bill["full_data_json"])
        current_bill = {
            "number": "HB 1234",
            "source": "OpenStates",
            "title": "Energy Market Reform Act",
            "status": "Introduced",
            "summary": cached_data["summary"],
            "amendments": [
                {"id": "amend1", "description": "First amendment"},
                {"id": "amend2", "description": "Second amendment"}
            ]
        }
        sample_cached_bill["text_hash"] = compute_bill_hash(cached_data["summary"])
        
        result = detect_bill_changes(sample_cached_bill, current_bill)
        
        assert result["has_changes"] is True
        amend_changes = [c for c in result["changes"] if c["type"] == CHANGE_TYPE_AMENDMENTS]
        assert len(amend_changes) == 1
        assert amend_changes[0]["count"] == 2
    
    def test_additional_amendments_detected(self, sample_cached_bill):
        """Adding amendments to existing should detect only new ones."""
        sample_cached_bill["full_data_json"] = json.dumps({
            "summary": "Original bill summary text.",
            "amendments": [
                {"id": "amend1", "description": "First amendment"}
            ]
        })
        cached_data = json.loads(sample_cached_bill["full_data_json"])
        
        current_bill = {
            "number": "HB 1234",
            "source": "OpenStates",
            "title": "Energy Market Reform Act",
            "status": "Introduced",
            "summary": cached_data["summary"],
            "amendments": [
                {"id": "amend1", "description": "First amendment"},
                {"id": "amend2", "description": "Second amendment"},
                {"id": "amend3", "description": "Third amendment"}
            ]
        }
        sample_cached_bill["text_hash"] = compute_bill_hash(cached_data["summary"])
        
        result = detect_bill_changes(sample_cached_bill, current_bill)
        
        amend_changes = [c for c in result["changes"] if c["type"] == CHANGE_TYPE_AMENDMENTS]
        assert len(amend_changes) == 1
        assert amend_changes[0]["count"] == 2
    
    def test_no_new_amendments_when_same_count(self, sample_cached_bill):
        """Same number of amendments should not trigger change."""
        sample_cached_bill["full_data_json"] = json.dumps({
            "summary": "Original bill summary text.",
            "amendments": [
                {"id": "amend1", "description": "First amendment"}
            ]
        })
        cached_data = json.loads(sample_cached_bill["full_data_json"])
        
        current_bill = {
            "number": "HB 1234",
            "source": "OpenStates",
            "title": "Energy Market Reform Act",
            "status": "Introduced",
            "summary": cached_data["summary"],
            "amendments": [
                {"id": "amend1", "description": "First amendment"}
            ]
        }
        sample_cached_bill["text_hash"] = compute_bill_hash(cached_data["summary"])
        
        result = detect_bill_changes(sample_cached_bill, current_bill)
        
        amend_changes = [c for c in result["changes"] if c["type"] == CHANGE_TYPE_AMENDMENTS]
        assert len(amend_changes) == 0


class TestMultipleChangeTypes:
    """Tests for detecting multiple change types simultaneously."""
    
    def test_multiple_change_types_detected(self, sample_cached_bill):
        """Text, status, and amendment changes can all be detected together."""
        current_bill = {
            "number": "HB 1234",
            "source": "OpenStates",
            "title": "Energy Market Reform Act",
            "status": "Engrossed",
            "summary": "Completely new bill text with major revisions.",
            "amendments": [
                {"id": "amend1", "description": "New amendment"}
            ]
        }
        
        result = detect_bill_changes(sample_cached_bill, current_bill)
        
        assert result["has_changes"] is True
        change_types = [c["type"] for c in result["changes"]]
        
        assert CHANGE_TYPE_TEXT in change_types
        assert CHANGE_TYPE_STATUS in change_types
        assert CHANGE_TYPE_AMENDMENTS in change_types


class TestComputeTextDiff:
    """Tests for unified diff generation."""
    
    def test_compute_diff_shows_changes(self):
        """Diff should show added and removed lines."""
        old_text = "Section 1. Short title.\nSection 2. Original text."
        new_text = "Section 1. Short title.\nSection 2. Modified text."
        
        diff = compute_text_diff(old_text, new_text)
        
        assert "-Section 2. Original text." in diff
        assert "+Section 2. Modified text." in diff
    
    def test_compute_diff_missing_old_text(self):
        """Empty old_text should return error message."""
        diff = compute_text_diff("", "New text here.")
        assert "not available" in diff.lower()
    
    def test_compute_diff_missing_new_text(self):
        """Empty new_text should return error message."""
        diff = compute_text_diff("Old text here.", "")
        assert "not available" in diff.lower()
    
    def test_compute_diff_identical_text(self):
        """Identical text should produce empty diff (no changes)."""
        text = "Section 1. Same text.\nSection 2. More same text."
        diff = compute_text_diff(text, text)
        assert "-" not in diff or "---" in diff
        assert "+" not in diff or "+++" in diff


class TestComputeBillHash:
    """Tests for bill hash computation."""
    
    def test_same_text_same_hash(self):
        """Identical text should produce identical hash."""
        text = "Section 1. Short title."
        hash1 = compute_bill_hash(text)
        hash2 = compute_bill_hash(text)
        assert hash1 == hash2
    
    def test_different_text_different_hash(self):
        """Different text should produce different hash."""
        hash1 = compute_bill_hash("Original text")
        hash2 = compute_bill_hash("Modified text")
        assert hash1 != hash2
    
    def test_empty_text_returns_empty_hash(self):
        """Empty text should return empty string."""
        assert compute_bill_hash("") == ""
        assert compute_bill_hash(None) == ""
    
    def test_hash_is_sha256(self):
        """Hash should be SHA-256 (64 hex characters)."""
        text = "Test bill text."
        hash_value = compute_bill_hash(text)
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)
