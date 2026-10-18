"""Unit tests for termmap.context. Run: python -m unittest discover -s tests"""
import unittest

from termmap.context import annotate, co_entities, find_section, get_clause, locate


class TestContext(unittest.TestCase):
    def test_family_history(self):
        chunk = "PMH: asthma. Family History: mother with breast cancer, father had MI at 50."
        s, e = locate("breast cancer", chunk)
        self.assertEqual(find_section(chunk, s), "family")
        self.assertEqual(get_clause(chunk, s, e), "mother with breast cancer")
        s2, _ = locate("asthma", chunk)
        self.assertEqual(find_section(chunk, s2), "history")

    def test_comma_list_pmh(self):
        chunk = "past medical history: htn, hld, dm2, ckd stage 3; meds: lisinopril"
        s, e = locate("hld", chunk)
        self.assertEqual(find_section(chunk, s), "history")
        clause = get_clause(chunk, s, e)
        self.assertIn("hld", clause)
        # 'hld' is one word -> extended by one neighbouring piece on each side
        self.assertEqual(clause, "htn, hld, dm2")
        s3, e3 = locate("ckd stage 3", chunk)
        self.assertEqual(get_clause(chunk, s3, e3), "ckd stage 3")
        s4, _ = locate("lisinopril", chunk)
        self.assertEqual(find_section(chunk, s4), "meds_allergies")

    def test_entity_missing(self):
        chunk = "assessment: chest pain, unclear etiology"
        self.assertIsNone(locate("pulmonary embolism", chunk))
        self.assertEqual(find_section(chunk, None), "none")
        self.assertEqual(get_clause(chunk, None, None, "pulmonary embolism"), "pulmonary embolism")

    def test_mention_spanning_delimiter_is_not_cut(self):
        chunk = "a/p: copd, frequent exacerbations (hcc); continue inhalers"
        s, e = locate("copd, frequent exacerbations", chunk)
        self.assertIn("copd, frequent exacerbations", get_clause(chunk, s, e))
        self.assertEqual(find_section(chunk, s), "assessment_plan")

    def test_no_false_header_inside_words(self):
        chunk = "shoulder pain since the plane trip; pe negative"  # 'sh', 'plan', 'pe' inside words / no colon
        s, _ = locate("shoulder pain", chunk)
        self.assertEqual(find_section(chunk, s), "none")

    def test_co_entities_and_annotate(self):
        import pandas as pd
        self.assertEqual(co_entities("HTN", ["htn", "HLD", "dm2", "hld"]), ["HLD", "dm2"])
        df = pd.DataFrame({"entity": ["htn", "hld", "rash"], "chunk_text": ["pmh: htn, hld", "pmh: htn, hld", "rash on arm"],
                           "group": ["g1", "g1", "g2"]})
        a = annotate(df)
        self.assertEqual(a.coents.tolist(), [["hld"], ["htn"], []])
        self.assertEqual(a.section.tolist(), ["history", "history", "none"])


if __name__ == "__main__":
    unittest.main()
