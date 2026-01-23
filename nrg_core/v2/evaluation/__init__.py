"""
Evaluation and calibration infrastructure.

This module provides tools for evaluating system performance without a golden dataset,
using silver sets, LLM-judge ensembles, and spot-checking for continuous validation.
"""
from nrg_core.v2.evaluation.silver_set import SilverSet, SilverBill

__all__ = ["SilverSet", "SilverBill"]
