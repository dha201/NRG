#!/usr/bin/env python3
"""
Helper script to add expert-labeled bills to silver set.

This script provides a command-line interface for adding bills with expert
labels to the silver set used for evaluation and regression testing.

Usage:
    python scripts/add_to_silver_set.py --bill-id HB123 --bill-text-file bill.txt --labels-file labels.json
"""
import argparse
import json
from pathlib import Path
from nrg_core.v2.evaluation.silver_set import SilverSet, SilverBill


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description="Add bill to silver set")
    parser.add_argument("--bill-id", required=True, help="Bill identifier (e.g., HB123)")
    parser.add_argument("--bill-text-file", required=True, help="File containing bill text")
    parser.add_argument("--labels-file", required=True, help="JSON file with expert labels")
    parser.add_argument("--nrg-context-file", help="Optional file with NRG business context")
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.bill_text_file).exists():
        raise FileNotFoundError(f"Bill text file not found: {args.bill_text_file}")
    
    if not Path(args.labels_file).exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_file}")
    
    # Load bill text
    with open(args.bill_text_file, 'r') as f:
        bill_text = f.read()
    
    # Load expert labels
    with open(args.labels_file, 'r') as f:
        expert_labels = json.load(f)
    
    # Load NRG context if provided
    nrg_context = None
    if args.nrg_context_file:
        if not Path(args.nrg_context_file).exists():
            raise FileNotFoundError(f"NRG context file not found: {args.nrg_context_file}")
        with open(args.nrg_context_file, 'r') as f:
            nrg_context = f.read()
    
    # Create silver bill
    bill = SilverBill(
        bill_id=args.bill_id,
        bill_text=bill_text,
        expert_labels=expert_labels,
        nrg_context=nrg_context
    )
    
    # Add to silver set
    silver_set = SilverSet()
    silver_set.add_bill(bill)
    
    print(f"✓ Added {args.bill_id} to silver set")
    print(f"  - Findings: {len(expert_labels.get('findings', []))}")
    print(f"  - Rubric scores: {len(expert_labels.get('rubric_scores', {}))}")


if __name__ == "__main__":
    main()
