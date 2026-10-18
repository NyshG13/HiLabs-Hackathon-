"""TermMap-X command line.

  python -m termmap reproduce                      # everything: download -> build -> train -> experiment -> predict
  python -m termmap predict --input X.csv --output Y.csv
  python -m termmap demo                           # hard cases, with confidence and alternatives
"""
from __future__ import annotations

import argparse
import json
import time

from . import config as C


def cmd_download(_):
    from .data import download_raw
    from .resources import download_cdc
    download_raw()
    download_cdc()


def cmd_build(_):
    from .data import make_split
    from .resources import build_all
    make_split()
    build_all()


def cmd_train(a):
    from . import train
    stages = a.stages.split(",")
    if "dense" in stages:
        from .data import load_split
        from .dense import Encoder, build_dense_index, fine_tune
        from .resources import load_aliases
        fine_tune(load_split("train"))
        enc = Encoder()
        for s in C.SYSTEMS:
            build_dense_index(s, load_aliases(s), enc)
        del enc
        train._free()
    if "cands" in stages:
        train.stage_cands()
    if "cross" in stages:
        train.stage_cross(context=True)
    if "comparator" in stages:
        train.stage_cross(context=False)
    if "score" in stages:
        train.stage_score()
    if "subconcept" in stages:
        from .pipeline import train_subconcept_model
        train_subconcept_model()


def cmd_experiment(a):
    from .experiment import run
    r = run(use_llm=not a.no_llm)
    print(json.dumps({k: r[k] for k in ("ablations_test",)}, indent=1))
    print("test headline (min exact):", r["test"]["headline_min_exact"])


def cmd_predict(a):
    from .pipeline import predict_file
    mem = tuple(a.memory.split(","))
    t = predict_file(a.input, a.output, memory_splits=mem, use_llm=not a.no_llm)
    if a.timing:
        (C.OUTPUTS / "latency.json").write_text(json.dumps(t, indent=1))


def cmd_demo(a):
    import pandas as pd
    from .pipeline import predict_frame
    cases = [
        ("HTN", "pmh: htn, hld, dm2. bp 150/90 today, continue lisinopril.", "DiagnosisCode"),
        ("pulmonary hypertension", "echo: rvsp 55 mmhg consistent with pulmonary hypertension.", "DiagnosisCode"),
        ("MS", "neuro f/u: ms diagnosed 2015 on ocrevus, new optic neuritis flare.", "DiagnosisCode"),
        ("MS", "echo: rheumatic mitral valve with moderate ms, mean gradient 8 mmhg.", "DiagnosisCode"),
        ("RA", "rheumatology: ra on methotrexate, morning stiffness 1 hr, synovitis of mcps.", "DiagnosisCode"),
        ("chronickidney disease stag 3", "assessment: chronickidney disease stag 3, egfr 45.", "DiagnosisCode"),
        ("diabetes", "a/p: diabetes - continue metformin, a1c 7.1.", "DiagnosisCode"),
        ("type 2 diabetes with diabetic neuropathy", "a/p: type 2 diabetes with diabetic neuropathy, on gabapentin.", "DiagnosisCode"),
        ("fx of right distal radius", "ed: fall on outstretched hand, fx of right distal radius, splinted.", "DiagnosisCode"),
        ("1-2+ ai", "- 5) aov sclerosis with 1-2+ ai .", "DiagnosisCode"),
        ("sob", "pt c/o sob on exertion x 2 weeks.", "DiagnosisCode"),
    ]
    if a.text:
        cases = [(a.text, a.context or "", a.subconcept or "")]
    df = pd.DataFrame(cases, columns=["entity", "chunk_text", "subconcept"])
    res, timing = predict_frame(df, use_llm=not a.no_llm, verbose=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    for _, r in res.iterrows():
        print(f"\n> {r.entity!r}   context: {r.chunk_text[:90]!r}")
        for tag, name in (("icd10cm", "ICD-10-CM"), ("snomed", "SNOMED CT")):
            flag = "  ABSTAIN (low confidence)" if r[f"{tag}_abstain"] else ""
            llm = "  [LLM adjudicated]" if r[f"{tag}_llm_adjudicated"] else ""
            print(f"   {name:9s} {r[f'{tag}_code']:<18} {r[f'{tag}_desc'][:60]:<60} conf={r[f'{tag}_confidence']:.2f}{flag}{llm}")
            print(f"   {'':9s} alternatives: {r[f'{tag}_alternatives']}")
        print(f"   joint agreement ICD<->SNOMED: {r.joint_agree}")
    print("\ntiming:", {k: round(v, 2) if isinstance(v, float) else v for k, v in timing.items()})


def cmd_reproduce(a):
    t = time.time()
    cmd_download(a)
    cmd_build(a)
    a.stages = "dense,cands,cross,comparator,score,subconcept"
    cmd_train(a)
    cmd_experiment(a)
    a.input, a.output, a.memory, a.timing = str(C.SPLITS / "test_input.csv"), str(C.OUTPUTS / "test_input_predictions.csv"), "train", True
    cmd_predict(a)
    print(f"reproduce finished in {(time.time() - t) / 60:.1f} min")


def main():
    p = argparse.ArgumentParser(prog="termmap")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("download").set_defaults(f=cmd_download)
    sp.add_parser("build").set_defaults(f=cmd_build)
    t = sp.add_parser("train")
    t.add_argument("--stages", default="dense,cands,cross,comparator,score,subconcept")
    t.set_defaults(f=cmd_train)
    e = sp.add_parser("experiment")
    e.add_argument("--no-llm", action="store_true")
    e.set_defaults(f=cmd_experiment)
    pr = sp.add_parser("predict")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--memory", default="train", help="labelled splits used as gold memory, e.g. train,dev,test")
    pr.add_argument("--no-llm", action="store_true")
    pr.add_argument("--timing", action="store_true")
    pr.set_defaults(f=cmd_predict)
    d = sp.add_parser("demo")
    d.add_argument("--text")
    d.add_argument("--context")
    d.add_argument("--subconcept")
    d.add_argument("--no-llm", action="store_true")
    d.set_defaults(f=cmd_demo)
    r = sp.add_parser("reproduce")
    r.add_argument("--no-llm", action="store_true")
    r.set_defaults(f=cmd_reproduce)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
