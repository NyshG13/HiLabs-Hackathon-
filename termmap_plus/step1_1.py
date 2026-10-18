"""Step 1.1: X's LLM gate, decided per system (keep it for a system only if it helps that system)."""
import termmap_plus  # noqa: F401
import pandas as pd

from termmap_plus import core as K


def main():
    P_on = pd.read_parquet(K.ART / "P_step0.parquet")
    P_off = pd.read_parquet(K.ART / "P_step0_nogate.parquet")
    m_on, m_off = K.metrics(P_on), K.metrics(P_off)
    keep = {}
    for s, tag in (("ICD10CM", "ICD"), ("SNOMEDCT_US", "SNOMED")):
        gain = m_on[tag] - m_off[tag]
        unseen_drop = m_off[f"unseen_{tag}"] - m_on[f"unseen_{tag}"]
        keep[s] = bool(gain >= 0.005 and unseen_drop <= 0.01)
        print(f"{tag}: gate on {m_on[tag]:.3f} vs off {m_off[tag]:.3f} (unseen {m_on[f'unseen_{tag}']:.3f} vs {m_off[f'unseen_{tag}']:.3f}) -> gate {'ON' if keep[s] else 'OFF'}")
    P = pd.concat([P_on[P_on.system == s] if keep[s] else P_off[P_off.system == s] for s in keep], ignore_index=True)
    P.to_parquet(K.ART / "P_step1_1.parquet")
    changed = keep != {"ICD10CM": True, "SNOMEDCT_US": True}
    m = K.ledger("1.1", f"LLM gate per system (ICD {'on' if keep['ICD10CM'] else 'off'}, SNOMED {'on' if keep['SNOMEDCT_US'] else 'off'})",
                 P, P_on, m_on, "KEPT" if changed else "DROPPED", "LLM gate hurting one system",
                 note="kept per system: rule = system accuracy +0.5pt and unseen not worse than 1pt")
    K.log_decision(f"Step 1.1 LLM gate per system -> {keep} (min {m['min']:.3f}, ICD {m['ICD']:.3f}, SNOMED {m['SNOMED']:.3f})")
    (K.ART / "state_llm_gate.json").write_text(pd.Series(keep).to_json(), encoding="utf8")


if __name__ == "__main__":
    main()
