"""Build per-system concept and alias tables from the supplied releases (plan section 3, step 1).

Rebuilt from whatever release files are present, so a new ICD-10-CM (October) or SNOMED CT
(biannual) drop is picked up by re-running `python -m termmap build`; nothing is hard-coded.
"""
from __future__ import annotations

import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

import pandas as pd

from . import config as C
from .normalize import AbbreviationTable, clean, tokens

_TAG = re.compile(r"\s*\(([^()]+)\)\s*$")


# ----------------------------------------------------------------------------------------------
# ICD-10-CM
# ----------------------------------------------------------------------------------------------
def download_cdc() -> None:
    """CDC/NCHS ICD-10-CM code descriptions and Alphabetic Index (public domain)."""
    C.CDC_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "codes.zip": f"{C.CDC_BASE}/{C.CDC_YEAR}/icd10cm-Code%20Descriptions-{C.CDC_YEAR}.zip",
        "tabindex.zip": f"{C.CDC_BASE}/{C.CDC_YEAR}/icd10cm-table%20and%20index-{C.CDC_YEAR}.zip",
    }
    for name, url in files.items():
        z = C.CDC_DIR.parent / f"{name[:-4]}-{C.CDC_YEAR}.zip"
        if not z.exists():
            print("downloading", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            z.write_bytes(urllib.request.urlopen(req).read())
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.endswith((f"order-{C.CDC_YEAR}.txt", f"index-{C.CDC_YEAR}.xml")) and "eindex" not in m:
                    if not (C.CDC_DIR / m).exists():
                        zf.extract(m, C.CDC_DIR)


def _cdc_order() -> dict[str, tuple[bool, str]]:
    p = C.CDC_DIR / f"icd10cm-order-{C.CDC_YEAR}.txt"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf8").splitlines():
        code = line[6:13].strip()
        if not code:
            continue
        dotted = code if len(code) <= 3 else code[:3] + "." + code[3:]
        out[dotted] = (line[14] == "1", line[77:].strip())
    return out


def _index_aliases(valid: set[str]) -> list[tuple[str, str]]:
    """Alphabetic Index paths such as 'Hypertension > essential' -> I10, as flat alias strings."""
    p = C.CDC_DIR / f"icd10cm-index-{C.CDC_YEAR}.xml"
    if not p.exists():
        return []
    root = ET.parse(p).getroot()
    out: list[tuple[str, str]] = []

    def title(el) -> str:
        t = el.find("title")
        if t is None:
            return ""
        return " ".join(((t.text or "") + " ".join((c.tail or "") for c in t)).split())

    def walk(el, path):
        name = title(el)
        path = path + ([name] if name else [])
        code = el.find("code")
        if code is not None and code.text:
            c = code.text.strip().rstrip("-").rstrip(".")
            if c in valid and path:
                main = path[0].split(",")[0]
                out.append((" ".join([main] + path[1:]), c))
                if len(path) > 1:  # natural order: "essential hypertension"
                    out.append((" ".join(path[1:] + [main]), c))
        for t in el.findall("term"):
            walk(t, path)

    for mt in root.iter("mainTerm"):
        walk(mt, [])
    return out


def build_icd() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(C.ICD_XLSX, dtype=str).dropna()
    raw["Code"] = raw.Code.str.strip()
    ranges = raw[raw.Code.str.contains("-")]
    codes = raw[~raw.Code.str.contains("-")]
    order = _cdc_order()
    all_codes = sorted(set(codes.Code))
    code_set = set(all_codes)
    has_child = {c: False for c in all_codes}
    for c in all_codes:
        p = parent_icd(c)
        while p and p not in code_set:
            p = parent_icd(p)
        if p:
            has_child[p] = True
    rng = [(r.Code.split("-")[0], r.Code.split("-")[1], r.Description) for r in ranges.itertuples()]

    def blocks(c):
        cat = c[:3]
        hits = sorted([(e[0], e[1], e[2]) for e in rng if e[0] <= cat <= e[1]], key=lambda e: (e[0], e[1]))
        if not hits:
            return "", ""
        mn = min(h[0] for h in hits)
        widest = max((h for h in hits if h[0] == mn), key=lambda h: h[1])
        narrow = max(hits, key=lambda h: (h[0], [-ord(x) for x in h[1]]))
        return widest[0] + "-" + widest[1], narrow[2]

    first_desc = codes.groupby("Code", sort=False).Description.first()
    rows = []
    for c in all_codes:
        billable, long_desc = order.get(c, (not has_child[c], ""))
        chap, block_desc = blocks(c)
        cat_desc = first_desc.get(c[:3], "")
        rows.append(dict(code=c, preferred=long_desc or first_desc[c], billable=bool(billable),
                         depth=len(c.replace(".", "")), category=c[:3], category_desc=cat_desc,
                         parent=parent_icd(c) or "", chapter=chap, block_desc=block_desc,
                         unspecified="unspecified" in (long_desc or first_desc[c]).lower()))
    concepts = pd.DataFrame(rows)
    al = [(r.Description, r.Code, "dict") for r in codes.itertuples()]
    al += [(d, c, "pref") for c, d in zip(concepts.code, concepts.preferred)]
    idx = _index_aliases(code_set)
    al += [(a, c, "cdc_index") for a, c in idx]
    aliases = _dedupe_aliases(pd.DataFrame(al, columns=["alias", "code", "source"]))
    print(f"ICD: {len(concepts)} codes ({concepts.billable.sum()} billable), {len(aliases)} aliases "
          f"({len(idx)} from CDC index)")
    return concepts, aliases


def parent_icd(c: str) -> str | None:
    s = c.replace(".", "")
    if len(s) <= 3:
        return None
    s = s[:-1]
    return s if len(s) <= 3 else s[:3] + "." + s[3:]


# ----------------------------------------------------------------------------------------------
# SNOMED CT
# ----------------------------------------------------------------------------------------------
def snomed_raw() -> pd.DataFrame:
    cache = C.PROCESSED / "snomed_raw.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raw = pd.read_excel(C.SNOMED_XLSX, dtype={"snomed_code": str, "description": str})
    raw = raw.dropna(subset=["description"]).reset_index(drop=True)
    raw.to_parquet(cache)
    return raw


def build_snomed(gold_codes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = snomed_raw()
    raw["snomed_code"] = raw.snomed_code.str.strip()
    raw = raw.sort_values(["snomed_code", "description_order"], kind="stable")
    tag = raw.description.str.extract(_TAG)[0]
    tag_counts = tag.value_counts()
    valid_tags = set(tag_counts[tag_counts >= 20].index)
    is_fsn = tag.isin(valid_tags)
    concept_tag = tag[is_fsn].groupby(raw.snomed_code[is_fsn]).last()
    pref = raw.groupby("snomed_code", sort=False).description.first()
    concepts = pd.DataFrame({"code": pref.index, "preferred": pref.values})
    concepts["tag"] = concepts.code.map(concept_tag).fillna("")
    gold_tags = set(concepts.loc[concepts.code.isin(gold_codes), "tag"])
    excluded = C.SNOMED_EXCLUDED_TAGS - gold_tags
    keep = ~concepts.tag.isin(excluded) | concepts.code.isin(gold_codes)
    concepts = concepts[keep].reset_index(drop=True)
    kept = set(concepts.code)
    desc = raw[raw.snomed_code.isin(kept)].copy()
    desc["alias"] = desc.description.where(~is_fsn[desc.index], desc.description.str.replace(_TAG, "", regex=True))
    aliases = desc.rename(columns={"snomed_code": "code"})[["alias", "code"]].assign(source="dict")
    aliases = _dedupe_aliases(aliases)
    print(f"SNOMED: {len(concepts)} concepts kept (excluded tags: {sorted(excluded)[:8]}...), {len(aliases)} aliases")
    return concepts, aliases


def _dedupe_aliases(a: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    a["alias_clean"] = a.alias.map(clean)
    a = a[a.alias_clean.str.len() > 0]
    a = a.drop_duplicates(["alias_clean", "code"]).reset_index(drop=True)
    return a


# ----------------------------------------------------------------------------------------------
def build_all() -> None:
    from .data import load_split
    download_cdc()
    train = load_split("train")
    gold_sn = set(train.loc[train.coding_system == C.SNOMED, "code"])
    icd_c, icd_a = build_icd()
    sn_c, sn_a = build_snomed(gold_sn)
    icd_c.to_parquet(C.PROCESSED / "icd_concepts.parquet")
    icd_a.to_parquet(C.PROCESSED / "icd_aliases.parquet")
    sn_c.to_parquet(C.PROCESSED / "snomed_concepts.parquet")
    sn_a.to_parquet(C.PROCESSED / "snomed_aliases.parquet")
    build_traindep(train, icd_a, sn_a)


def build_traindep(train=None, icd_a=None, sn_a=None) -> None:
    """Train-dependent front-end resources (typo vocabulary, abbreviation table) -> C.TRAINDEP."""
    from .data import load_split
    train = load_split("train") if train is None else train
    icd_a = load_aliases(C.ICD) if icd_a is None else icd_a
    sn_a = load_aliases(C.SNOMED) if sn_a is None else sn_a

    # vocabulary for the typo gate: dictionary aliases + training mentions
    vocab = Counter()
    for s in pd.concat([icd_a.alias_clean, sn_a.alias_clean]):
        vocab.update(s.split())
    for s in train.entity:
        vocab.update(tokens(s))
    (C.TRAINDEP / "vocab.json").write_text(json.dumps(vocab), encoding="utf8")

    # abbreviation table: dictionary patterns + training gold (train split only)
    ab = AbbreviationTable()
    ab.mine_descriptions(snomed_raw().description)
    ab.mine_descriptions(icd_a.alias)
    ab.mine_gold(train.entity, train.get("display", train.entity))
    (C.TRAINDEP / "abbreviations.json").write_text(json.dumps(ab.to_json()), encoding="utf8")
    print(f"abbreviations: {len(ab.table)} short forms; vocab {len(vocab)} words")


def load_concepts(system: str) -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / ("icd_concepts.parquet" if system == C.ICD else "snomed_concepts.parquet"))


def load_aliases(system: str) -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / ("icd_aliases.parquet" if system == C.ICD else "snomed_aliases.parquet"))
