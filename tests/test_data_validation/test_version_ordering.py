"""
Version ordering tests.

Tests ensure versions are correctly ordered and latest version identified.
Critical for ensuring LLM analyzes correct version pairs.
"""
import pytest

from nrg_core.models import Bill, BillVersion


class TestVersionSorting:
    """Tests for version sorting by version_number."""
    
    def test_versions_sorted_by_version_number(self, sample_bill_with_versions):
        """Versions should be sorted by version_number ascending."""
        bill = Bill.from_dict(sample_bill_with_versions)
        version_numbers = [v.version_number for v in bill.versions]
        assert version_numbers == [1, 2, 3]
    
    def test_out_of_order_versions_remain_in_original_order(self):
        """Bill.from_dict preserves original order - sorting is application responsibility."""
        bill_dict = {
            "source": "OpenStates",
            "type": "State Bill",
            "number": "HB 999",
            "title": "Test Bill",
            "status": "Introduced",
            "url": "",
            "summary": "",
            "versions": [
                {"version_number": 3, "version_type": "Engrossed", "version_date": "", "full_text": "", "text_hash": "", "word_count": 0},
                {"version_number": 1, "version_type": "Introduced", "version_date": "", "full_text": "", "text_hash": "", "word_count": 0},
                {"version_number": 2, "version_type": "Committee Report", "version_date": "", "full_text": "", "text_hash": "", "word_count": 0},
            ]
        }
        bill = Bill.from_dict(bill_dict)
        version_numbers = [v.version_number for v in bill.versions]
        assert version_numbers == [3, 1, 2]
    
    def test_sort_versions_by_number(self):
        """Versions can be sorted by version_number when needed."""
        bill_dict = {
            "source": "OpenStates",
            "type": "State Bill",
            "number": "HB 999",
            "title": "Test Bill",
            "status": "Introduced",
            "url": "",
            "summary": "",
            "versions": [
                {"version_number": 3, "version_type": "Engrossed", "version_date": "", "full_text": "", "text_hash": "", "word_count": 0},
                {"version_number": 1, "version_type": "Introduced", "version_date": "", "full_text": "", "text_hash": "", "word_count": 0},
                {"version_number": 2, "version_type": "Committee Report", "version_date": "", "full_text": "", "text_hash": "", "word_count": 0},
            ]
        }
        bill = Bill.from_dict(bill_dict)
        sorted_versions = sorted(bill.versions, key=lambda v: v.version_number)
        version_numbers = [v.version_number for v in sorted_versions]
        assert version_numbers == [1, 2, 3]


class TestLatestVersionIdentification:
    """Tests for identifying the latest version."""
    
    def test_latest_version_is_highest_number(self, sample_bill_with_versions):
        """Latest version should be the one with highest version_number."""
        bill = Bill.from_dict(sample_bill_with_versions)
        latest = max(bill.versions, key=lambda v: v.version_number)
        assert latest.version_number == 3
        assert latest.version_type_raw == "Engrossed"
    
    def test_single_version_is_latest(self, sample_bill_dict):
        """Single version should be identified as latest."""
        sample_bill_dict["versions"] = [
            {"version_number": 1, "version_type": "Introduced", "version_date": "", "full_text": "text", "text_hash": "abc", "word_count": 10}
        ]
        bill = Bill.from_dict(sample_bill_dict)
        assert len(bill.versions) == 1
        latest = max(bill.versions, key=lambda v: v.version_number)
        assert latest.version_number == 1
    
    def test_no_versions_handled(self, sample_bill_dict):
        """Bill with no versions should not error."""
        sample_bill_dict["versions"] = []
        bill = Bill.from_dict(sample_bill_dict)
        assert len(bill.versions) == 0


class TestConsecutiveVersionPairing:
    """Tests for pairing consecutive versions for comparison."""
    
    def test_consecutive_pairs_from_sorted_versions(self, sample_bill_with_versions):
        """Generate consecutive pairs: (v1,v2), (v2,v3), not (v1,v3)."""
        bill = Bill.from_dict(sample_bill_with_versions)
        sorted_versions = sorted(bill.versions, key=lambda v: v.version_number)
        
        pairs = list(zip(sorted_versions[:-1], sorted_versions[1:]))
        
        assert len(pairs) == 2
        assert pairs[0][0].version_number == 1
        assert pairs[0][1].version_number == 2
        assert pairs[1][0].version_number == 2
        assert pairs[1][1].version_number == 3
    
    def test_single_version_no_pairs(self, sample_bill_dict):
        """Single version should produce no pairs."""
        sample_bill_dict["versions"] = [
            {"version_number": 1, "version_type": "Introduced", "version_date": "", "full_text": "", "text_hash": "", "word_count": 0}
        ]
        bill = Bill.from_dict(sample_bill_dict)
        sorted_versions = sorted(bill.versions, key=lambda v: v.version_number)
        pairs = list(zip(sorted_versions[:-1], sorted_versions[1:]))
        assert len(pairs) == 0
    
    def test_two_versions_one_pair(self):
        """Two versions should produce one pair."""
        bill_dict = {
            "source": "OpenStates",
            "type": "State Bill",
            "number": "HB 888",
            "title": "Test Bill",
            "status": "Engrossed",
            "url": "",
            "summary": "",
            "versions": [
                {"version_number": 1, "version_type": "Introduced", "version_date": "", "full_text": "v1 text", "text_hash": "a", "word_count": 10},
                {"version_number": 2, "version_type": "Engrossed", "version_date": "", "full_text": "v2 text", "text_hash": "b", "word_count": 20},
            ]
        }
        bill = Bill.from_dict(bill_dict)
        sorted_versions = sorted(bill.versions, key=lambda v: v.version_number)
        pairs = list(zip(sorted_versions[:-1], sorted_versions[1:]))
        
        assert len(pairs) == 1
        assert pairs[0][0].version_number == 1
        assert pairs[0][1].version_number == 2


class TestVersionGapDetection:
    """Tests for detecting missing versions in sequence."""
    
    def test_detect_missing_version_in_sequence(self):
        """Detect when version numbers have gaps (e.g., 1, 3 missing 2)."""
        bill_dict = {
            "source": "OpenStates",
            "type": "State Bill",
            "number": "HB 777",
            "title": "Test Bill",
            "status": "Engrossed",
            "url": "",
            "summary": "",
            "versions": [
                {"version_number": 1, "version_type": "Introduced", "version_date": "", "full_text": "", "text_hash": "", "word_count": 0},
                {"version_number": 3, "version_type": "Engrossed", "version_date": "", "full_text": "", "text_hash": "", "word_count": 0},
                {"version_number": 5, "version_type": "Enrolled", "version_date": "", "full_text": "", "text_hash": "", "word_count": 0},
            ]
        }
        bill = Bill.from_dict(bill_dict)
        sorted_versions = sorted(bill.versions, key=lambda v: v.version_number)
        version_numbers = [v.version_number for v in sorted_versions]
        
        expected_sequence = list(range(1, max(version_numbers) + 1))
        missing = set(expected_sequence) - set(version_numbers)
        
        assert missing == {2, 4}
    
    def test_no_gaps_in_complete_sequence(self, sample_bill_with_versions):
        """Complete sequence (1, 2, 3) should have no gaps."""
        bill = Bill.from_dict(sample_bill_with_versions)
        version_numbers = [v.version_number for v in bill.versions]
        
        expected_sequence = list(range(1, max(version_numbers) + 1))
        missing = set(expected_sequence) - set(version_numbers)
        
        assert missing == set()
