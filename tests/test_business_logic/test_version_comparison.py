"""
Version comparison tests.

Tests verify consecutive version comparison logic and diff generation.
"""
import pytest

from nrg_core.analysis.changes import compare_consecutive_versions
from nrg_core.db.cache import compute_bill_hash


class TestCompareConsecutiveVersions:
    """Tests for compare_consecutive_versions() function."""
    
    def test_versions_with_text_changes(self):
        """Changed text should be detected between versions."""
        old_version = {
            "version_type": "Introduced",
            "full_text": "Section 1. Short title.\nSection 2. Original definitions.",
            "word_count": 20
        }
        new_version = {
            "version_type": "Engrossed",
            "full_text": "Section 1. Short title.\nSection 2. Amended definitions.\nSection 3. New section.",
            "word_count": 30
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["changed"] is True
        assert result["from_version"] == "Introduced"
        assert result["to_version"] == "Engrossed"
        assert result["lines_added"] > 0
        assert result["word_count_change"] == 10
    
    def test_versions_with_no_changes(self):
        """Identical text should show no changes."""
        text = "Section 1. Short title.\nSection 2. Definitions."
        old_version = {
            "version_type": "Introduced",
            "full_text": text,
            "word_count": 15
        }
        new_version = {
            "version_type": "Committee Report",
            "full_text": text,
            "word_count": 15
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["changed"] is False
        assert result["from_version"] == "Introduced"
        assert result["to_version"] == "Committee Report"
    
    def test_versions_missing_old_text(self):
        """Missing old_text should return error."""
        old_version = {
            "version_type": "Introduced",
            "full_text": "",
            "word_count": 0
        }
        new_version = {
            "version_type": "Engrossed",
            "full_text": "New text here.",
            "word_count": 10
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["changed"] is False
        assert "error" in result
    
    def test_versions_missing_new_text(self):
        """Missing new_text should return error."""
        old_version = {
            "version_type": "Introduced",
            "full_text": "Old text here.",
            "word_count": 10
        }
        new_version = {
            "version_type": "Engrossed",
            "full_text": "",
            "word_count": 0
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["changed"] is False
        assert "error" in result


class TestDiffStatistics:
    """Tests for diff statistics calculation."""
    
    def test_lines_added_counted(self):
        """Lines added should be counted correctly."""
        old_version = {
            "version_type": "v1",
            "full_text": "Line 1\nLine 2",
            "word_count": 4
        }
        new_version = {
            "version_type": "v2",
            "full_text": "Line 1\nLine 2\nLine 3\nLine 4",
            "word_count": 8
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["changed"] is True
        assert result["lines_added"] >= 2
    
    def test_lines_removed_counted(self):
        """Lines removed should be counted correctly."""
        old_version = {
            "version_type": "v1",
            "full_text": "Line 1\nLine 2\nLine 3\nLine 4",
            "word_count": 8
        }
        new_version = {
            "version_type": "v2",
            "full_text": "Line 1\nLine 2",
            "word_count": 4
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["changed"] is True
        assert result["lines_removed"] >= 2
    
    def test_word_count_change_positive(self):
        """Positive word count change when text added."""
        old_version = {
            "version_type": "v1",
            "full_text": "Short text.",
            "word_count": 50
        }
        new_version = {
            "version_type": "v2",
            "full_text": "Short text with more words added.",
            "word_count": 100
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["word_count_change"] == 50
    
    def test_word_count_change_negative(self):
        """Negative word count change when text removed."""
        old_version = {
            "version_type": "v1",
            "full_text": "Long text with many words.",
            "word_count": 200
        }
        new_version = {
            "version_type": "v2",
            "full_text": "Short text.",
            "word_count": 75
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["word_count_change"] == -125


class TestDiffPreview:
    """Tests for diff preview/summary generation."""
    
    def test_diff_summary_included(self):
        """Changed versions should include diff summary."""
        old_version = {
            "version_type": "v1",
            "full_text": "Original text content.",
            "word_count": 10
        }
        new_version = {
            "version_type": "v2",
            "full_text": "Modified text content.",
            "word_count": 10
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["changed"] is True
        assert "diff_summary" in result
        assert len(result["diff_summary"]) > 0
    
    def test_key_changes_preview_truncated(self):
        """Key changes preview should be truncated for long diffs."""
        old_text = "\n".join([f"Line {i}" for i in range(100)])
        new_text = "\n".join([f"Modified Line {i}" for i in range(100)])
        
        old_version = {
            "version_type": "v1",
            "full_text": old_text,
            "word_count": 200
        }
        new_version = {
            "version_type": "v2",
            "full_text": new_text,
            "word_count": 300
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert "key_changes_preview" in result
        assert len(result["key_changes_preview"]) <= 500


class TestHashBasedComparison:
    """Tests for hash-based comparison in version changes."""
    
    def test_same_hash_no_change(self):
        """Versions with same text hash should report no change."""
        text = "Identical text content."
        hash_val = compute_bill_hash(text)
        
        old_version = {
            "version_type": "v1",
            "full_text": text,
            "word_count": 10
        }
        new_version = {
            "version_type": "v2",
            "full_text": text,
            "word_count": 10
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["changed"] is False
    
    def test_different_hash_triggers_change(self):
        """Versions with different text should have different hashes."""
        old_text = "Original text."
        new_text = "Modified text."
        
        old_hash = compute_bill_hash(old_text)
        new_hash = compute_bill_hash(new_text)
        
        assert old_hash != new_hash
        
        old_version = {
            "version_type": "v1",
            "full_text": old_text,
            "word_count": 5
        }
        new_version = {
            "version_type": "v2",
            "full_text": new_text,
            "word_count": 5
        }
        
        result = compare_consecutive_versions(old_version, new_version)
        
        assert result["changed"] is True
