"""TermMap-X+ : improvements layered on top of vanilla TermMap-X without modifying it.

Vanilla X uses the `subconcept` column in two places (cross-encoder query text, stacker feature
tag_compat). The judges' file has no such column but is diagnosis-only, so X+ feeds the constant
"DiagnosisCode" (Step 0 leak fix). That requires X's legacy code path, enabled here before termmap loads.
"""
import os

os.environ["TERMMAP_USE_SUBCONCEPT"] = "1"
