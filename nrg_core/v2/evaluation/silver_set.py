"""
Silver Set Management

Infrastructure for managing 50-100 expert-labeled bills used for regression testing
and continuous validation of the legislative analysis system.

Design:
- JSON-based storage for expert-labeled bills
- Validation ensures all bills have required expert labels
- Supports loading individual bills or entire silver set
- Helper scripts for adding new bills to the set

Why:
- Expert-labeled data provides ground truth for evaluation
- Silver set enables regression testing without golden dataset
- Structured storage supports versioning and reproducibility
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SilverBill:
    """
    Single bill in silver set with expert labels.
    
    Attributes:
        bill_id: Unique bill identifier (e.g., "HB123")
        bill_text: Full text of the bill
        expert_labels: Expert-provided labels including findings and rubric scores
        nrg_context: Optional NRG business context for the bill
    """
    bill_id: str
    bill_text: str
    expert_labels: Dict[str, Any]
    nrg_context: Optional[str] = None
    
    def __post_init__(self):
        """
        Validate that expert labels are provided.
        
        Expert labels are required for silver set bills to ensure
        ground truth is available for evaluation.
        """
        if self.expert_labels is None:
            raise ValueError("Silver bills must have expert_labels")


class SilverSet:
    """
    Manage silver set of expert-labeled bills for evaluation.
    
    Silver set structure:
    data/silver_set/
      HB123.json
      HB456.json
      ...
    
    Each JSON contains:
    {
      "bill_id": "HB123",
      "bill_text": "...",
      "expert_labels": {
        "findings": [...],
        "rubric_scores": {...}
      },
      "nrg_context": "..."
    }
    
    Why this structure:
    - JSON format is human-readable and version-controllable
    - One file per bill enables granular updates
    - Standardized structure supports automated evaluation
    """
    
    def __init__(self, data_dir: str = "data/silver_set"):
        """
        Initialize silver set with data directory.
        
        Args:
            data_dir: Directory containing silver set JSON files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> List[SilverBill]:
        """
        Load all bills from silver set.
        
        Returns:
            List of SilverBill objects with expert labels
        """
        bills = []
        
        for json_file in self.data_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            bill = SilverBill(
                bill_id=data["bill_id"],
                bill_text=data["bill_text"],
                expert_labels=data["expert_labels"],
                nrg_context=data.get("nrg_context")
            )
            bills.append(bill)
        
        return bills
    
    def add_bill(self, bill: SilverBill) -> None:
        """
        Add bill to silver set.
        
        Args:
            bill: SilverBill object with expert labels
        """
        file_path = self.data_dir / f"{bill.bill_id}.json"
        
        data = {
            "bill_id": bill.bill_id,
            "bill_text": bill.bill_text,
            "expert_labels": bill.expert_labels,
            "nrg_context": bill.nrg_context
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_bill(self, bill_id: str) -> Optional[SilverBill]:
        """
        Get specific bill by ID.
        
        Args:
            bill_id: Bill identifier to retrieve
        
        Returns:
            SilverBill if found, None otherwise
        """
        file_path = self.data_dir / f"{bill_id}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return SilverBill(
            bill_id=data["bill_id"],
            bill_text=data["bill_text"],
            expert_labels=data["expert_labels"],
            nrg_context=data.get("nrg_context")
        )
