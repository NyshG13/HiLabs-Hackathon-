"""Subconcept-free predict: an input with subconcept=ProcedureCode (plus other stray columns) must give
byte-identical output to the same rows without those columns.

  python tests/test_sf_guard.py   (or pytest; GPU recommended)
"""
import filecmp
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _rows(n=8) -> pd.DataFrame:
    from termmap.data import load_split
    d = load_split("dev")
    d = d[d.subconcept == "DiagnosisCode"].drop_duplicates(["entity", "chunk_text"]).head(n)
    return d[["entity", "chunk_text"]].rename(columns={"entity": "clinical_entity"}).reset_index(drop=True)


def test_guard_drops_columns():
    from termmap_plus.predict_sf import guard
    a = _rows(3)
    b = a.assign(subconcept="ProcedureCode", coding_system="ICD10CM", code="Z00.00")
    ga, gb = guard(a), guard(b)
    assert list(ga.columns) == ["entity", "chunk_text", "chunk_id"]
    assert ga.equals(gb)


def test_subconcept_column_byte_identical(tmp_path):
    from termmap_plus import predict_sf as PS
    rows = _rows()
    plain, dirty = tmp_path / "plain.csv", tmp_path / "dirty.csv"
    rows.to_csv(plain, index=False)
    rows.assign(subconcept="ProcedureCode", coding_system="SNOMEDCT_US").to_csv(dirty, index=False)
    o1, o2 = tmp_path / "out_plain.csv", tmp_path / "out_dirty.csv"
    PS.main(["--input", str(plain), "--output", str(o1)])
    PS.main(["--input", str(dirty), "--output", str(o2)])
    assert filecmp.cmp(o1, o2, shallow=False), "subconcept column changed the output"
    out = pd.read_csv(o1, dtype=str)
    assert len(out) == len(rows) and "subconcept" not in out.columns


if __name__ == "__main__":  # runnable without pytest: python tests/test_sf_guard.py [--quick]
    import tempfile
    test_guard_drops_columns()
    print("guard drops columns: OK")
    if "--quick" not in sys.argv:
        with tempfile.TemporaryDirectory() as d:
            test_subconcept_column_byte_identical(Path(d))
        print("subconcept=ProcedureCode -> byte-identical output: OK")
