"""
Cross-Bill Reference Detection (Tier 1).

Extracts references to external statutes, bills, and legal codes using
regex pattern matching. Fast, no API calls required.

Design rationale:
- Regex-based detection is fast and deterministic
- Multiple patterns cover common legislative citation formats
- Section tracking enables precise reference location
- Deduplication prevents redundant processing

Patterns detected:
1. U.S. Code citations: "26 U.S.C. 48", "42 U.S.C. § 7401"
2. "amends [statute]" patterns
3. "as defined in [citation]" patterns
4. Public Law citations: "Public Law 117-169"

Why regex over NLP:
- Legislative citations follow strict formatting
- Regex is orders of magnitude faster than NLP
- High precision for well-formatted citations
- Easy to extend with new patterns

Usage:
    detector = ReferenceDetector()
    refs = detector.detect(bill_text)
    for ref in refs:
        print(f"{ref.citation} ({ref.reference_type}) in {ref.location}")
"""
import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class BillReference:
    """
    Detected reference to external statute or bill.
    
    Attributes:
        reference_type: Category of reference
            - "statutory_amendment": Modifies existing law
            - "definition_by_reference": Uses external definition
            - "precedent_citation": Cites prior legislation
        citation: The citation text (e.g., "26 U.S.C. 48")
        context: Surrounding text for understanding
        location: Section where reference appears
    
    Design: Separates type from citation to enable different
    handling strategies (amendments need resolution, definitions
    need lookup, precedents are informational).
    """
    reference_type: str
    citation: str
    context: str
    location: str


# Regex patterns for legislative citation detection
# Each tuple: (compiled_pattern, reference_type)

# U.S. Code: "26 U.S.C. 48" or "26 U.S.C. § 48"
USC_PATTERN = re.compile(
    r'(\d+)\s+U\.S\.C\.?\s*§?\s*(\d+[a-z]?)',
    re.IGNORECASE
)

# Amends pattern: "amends the Clean Air Act" or "amends section 48"
AMENDS_PATTERN = re.compile(
    r'amends?\s+(.+?)(?:\s+to\s|\s+by\s|\.|\,)',
    re.IGNORECASE
)

# As defined in: "as defined in 26 U.S.C. 45"
AS_DEFINED_PATTERN = re.compile(
    r'as\s+defined\s+in\s+(.+?)(?:\s+applies|\s+shall|,|\.)',
    re.IGNORECASE
)

# Public Law: "Public Law 117-169"
PUBLIC_LAW_PATTERN = re.compile(
    r'Public\s+Law\s+(\d+-\d+)',
    re.IGNORECASE
)


class ReferenceDetector:
    """
    Detect cross-bill references using pattern matching.
    
    Tier 1 of two-tier reference system:
    - Tier 1 (this): Fast detection via regex, no API calls
    - Tier 2: On-demand resolution via USCODE API
    
    Detection strategy:
    1. Split bill into sections for location tracking
    2. Apply each pattern to each section
    3. Extract citation and surrounding context
    4. Deduplicate by (type, citation) pair
    
    Why section tracking:
    - Enables precise reference location in audit trail
    - Supports section-specific analysis
    - Helps identify which provisions have external dependencies
    """
    
    def __init__(self):
        """
        Initialize detector with pattern registry.
        
        Patterns are ordered by specificity (more specific first)
        to avoid partial matches.
        """
        self.patterns = [
            (USC_PATTERN, "statutory_amendment"),
            (AMENDS_PATTERN, "statutory_amendment"),
            (AS_DEFINED_PATTERN, "definition_by_reference"),
            (PUBLIC_LAW_PATTERN, "precedent_citation")
        ]
    
    def detect(self, bill_text: str) -> List[BillReference]:
        """
        Detect all references in bill text.
        
        Process:
        1. Split text into sections
        2. Apply all patterns to each section
        3. Extract citation and context
        4. Deduplicate results
        
        Args:
            bill_text: Full bill text
        
        Returns:
            List of BillReference objects, deduplicated
        """
        references = []
        
        # Split into sections for location tracking
        sections = self._split_sections(bill_text)
        
        for section_name, section_text in sections.items():
            # Try each pattern
            for pattern, ref_type in self.patterns:
                matches = pattern.finditer(section_text)
                
                for match in matches:
                    citation = self._extract_citation(match, ref_type)
                    context = self._extract_context(
                        section_text,
                        match.start(),
                        match.end()
                    )
                    
                    ref = BillReference(
                        reference_type=ref_type,
                        citation=citation,
                        context=context,
                        location=section_name
                    )
                    references.append(ref)
        
        # Deduplicate by (type, citation)
        return self._deduplicate(references)
    
    def _split_sections(self, bill_text: str) -> Dict[str, str]:
        """
        Split bill text into sections.
        
        Recognizes patterns like "Section 2.1:" or "SECTION 3:"
        Falls back to "Preamble" for text before first section.
        
        Args:
            bill_text: Full bill text
        
        Returns:
            Dict mapping section name to section text
        """
        sections = {}
        
        # Pattern: "Section X.Y:" at start of line or after newline
        # Requires colon or end-of-line to distinguish from "section 48 of..."
        section_pattern = r'(Section\s+\d+\.?\d*[a-z]?\s*:)'
        parts = re.split(section_pattern, bill_text, flags=re.IGNORECASE)
        
        current_section = "Preamble"
        sections[current_section] = ""
        
        for i, part in enumerate(parts):
            if re.match(section_pattern, part, re.IGNORECASE):
                current_section = part.strip()
                if current_section not in sections:
                    sections[current_section] = ""
            else:
                sections[current_section] += part
        
        return sections
    
    def _extract_citation(self, match: re.Match, ref_type: str) -> str:
        """
        Extract clean citation from regex match.
        
        Handles different match group structures:
        - USC: groups (title, section) -> "title U.S.C. section"
        - Others: group(1) or group(0)
        
        Args:
            match: Regex match object
            ref_type: Reference type for format selection
        
        Returns:
            Cleaned citation string
        """
        # USC citations have two groups (title, section)
        if ref_type == "statutory_amendment" and match.lastindex and match.lastindex >= 2:
            try:
                title = match.group(1)
                section = match.group(2)
                # Check if this looks like a USC citation
                if title.isdigit() and "U.S.C" in match.group(0).upper():
                    return f"{title} U.S.C. {section}"
            except (IndexError, AttributeError):
                pass
        
        # For other patterns, use first group or full match
        if match.lastindex and match.lastindex >= 1:
            return match.group(1).strip()
        return match.group(0).strip()
    
    def _extract_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 100
    ) -> str:
        """
        Extract surrounding context for a match.
        
        Args:
            text: Full section text
            start: Match start position
            end: Match end position
            window: Characters to include on each side
        
        Returns:
            Context string with match highlighted
        """
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _deduplicate(self, references: List[BillReference]) -> List[BillReference]:
        """
        Remove duplicate references by (type, citation) pair.
        
        Keeps first occurrence (preserves location from first mention).
        
        Args:
            references: List with potential duplicates
        
        Returns:
            Deduplicated list
        """
        seen = set()
        unique = []
        
        for ref in references:
            key = (ref.reference_type, ref.citation)
            if key not in seen:
                seen.add(key)
                unique.append(ref)
        
        return unique
