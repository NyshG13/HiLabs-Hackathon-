"""Candidate pools for diagnosis dev rows, rebuilt with X's own retriever + X+ options; every candidate is
scored by X's own cross-encoders (cached by exact query/candidate text). Non-diagnosis dev rows keep X's
vanilla pools (they are only stacker-training rows). Reads vanilla artefacts, never writes them."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from termmap import config as C
from termmap_plus import core as K

CE_CACHE = K.ART / "ce_cache.parquet"


# ----------------------------------------------------------------------------------------------
# SNOMED semantic-tag scope (1.3)
# ----------------------------------------------------------------------------------------------
def allowed_tags(coverage: float = 0.995) -> list[str]:
    con = K.concepts()[C.SNOMED]
    t = K.diag_train()
    tags = t[t.coding_system == C.SNOMED].code.map(con.tag).value_counts()
    cum = tags.cumsum() / tags.sum()
    keep = list(cum.index[: int((cum < coverage).sum()) + 1])
    return keep


# ----------------------------------------------------------------------------------------------
# ICD tree (1.9)
# ----------------------------------------------------------------------------------------------
def icd_parent(code: str, codes: set) -> str | None:
    s = code.replace(".", "")
    while len(s) > 3:
        s = s[:-1]
        cand = s if len(s) <= 3 else s[:3] + "." + s[3:]
        if s.endswith("X"):  # placeholder level: skip
            continue
        if cand in codes:
            return cand
    return None


def build_icd_tree() -> pd.DataFrame:
    from termmap.normalize import content_tokens
    p = K.SHARED / "icd_tree.parquet"
    if p.exists():
        return pd.read_parquet(p)
    con = K.concepts()[C.ICD]
    codes = set(con.index)
    par = {c: icd_parent(c, codes) for c in con.index}
    rows = []
    for c in con.index:
        chain, x = [], par[c]
        while x:
            chain.append(x)
            x = par[x]
        own = set(content_tokens(con.at[c, "preferred"]))
        anc = set()
        for a in chain:
            anc |= set(content_tokens(con.at[a, "preferred"]))
        rows.append(dict(code=c, parent=par[c] or "", depth=len(chain), category=c[:3],
                         path_text=" > ".join(con.at[a, "preferred"] for a in list(reversed(chain)) + [c]),
                         distinguishing_words=" ".join(sorted(own - anc))))
    tree = pd.DataFrame(rows)
    kids = tree.groupby("parent").code.agg(list)
    tree["is_leaf"] = ~tree.code.isin(kids.index)
    tree["n_siblings"] = tree.parent.map(kids.map(len)).fillna(1).astype(int) - 1
    tree["is_unspecified"] = tree.code.map(con.unspecified).astype(bool)
    tree.to_parquet(p)
    return tree


_LAT = {"right": "left", "left": "right"}


def icd_expansions(top_codes: list[str], tree: pd.DataFrame, cap_leaves: int = 15) -> list[str]:
    """unspecified sibling, other leaves of the category (cap), laterality variants, 7th-character A/D/S."""
    t = tree.set_index("code")
    con = K.concepts()[C.ICD]
    by_parent = tree.groupby("parent").code.agg(list).to_dict()
    leaves_by_cat = tree[tree.is_leaf].groupby("category").code.agg(list).to_dict()
    out = []
    for c in top_codes:
        if c not in t.index:
            continue
        par = t.at[c, "parent"]
        sib = by_parent.get(par, []) if par else []
        out += [x for x in sib if t.at[x, "is_unspecified"]]
        out += leaves_by_cat.get(c[:3], [])[:cap_leaves]
        pref = con.at[c, "preferred"].lower()
        for a, b in _LAT.items():
            if a in pref:
                for x in sib:
                    if b in con.at[x, "preferred"].lower() and con.at[x, "preferred"].lower().replace(b, a) == pref:
                        out.append(x)
        if len(c.replace(".", "")) == 7 and c[-1] in "ADS":
            for e in "ADS":
                v = c[:-1] + e
                if v in t.index and v != c:
                    out.append(v)
    return list(dict.fromkeys(x for x in out if x not in top_codes))


# ----------------------------------------------------------------------------------------------
# Pool construction
# ----------------------------------------------------------------------------------------------
def _seed_ce_cache():
    if CE_CACHE.exists():
        return pd.read_parquet(CE_CACHE)
    from termmap.reranker import cand_text, query_text
    con, rows = K.concepts(), []
    for s, cand in K.vanilla_pools().items():
        d = K.dev_frames()[s]
        for kind, col in (("ctx", "ce_ctx"), ("noctx", "ce_noctx")):
            q = [query_text(s, d.entity.iat[m], d.chunk_text.iat[m], d.sub_in.iat[m], kind == "ctx") for m in cand.mention]
            c = [cand_text(s, con[s].loc[k], a) for k, a in zip(cand.code, cand.best_alias)]
            rows.append(pd.DataFrame({"kind": kind, "q": q, "c": c, "score": cand[col].to_numpy()}))
    df = pd.concat(rows, ignore_index=True).drop_duplicates(["kind", "q", "c"])
    df.to_parquet(CE_CACHE)
    return df


def score_pools(pools: dict) -> float:
    """Fill ce_ctx / ce_noctx using the cache; score only unseen pairs with X's models. Returns GPU seconds."""
    from termmap.reranker import CrossEncoder, cand_text, query_text
    cache = _seed_ce_cache()
    lut = {(k, q, c): v for k, q, c, v in zip(cache.kind, cache.q, cache.c, cache.score)}
    con, todo, t0 = K.concepts(), {}, time.perf_counter()
    keys = {}
    for s, cand in pools.items():
        d = K.dev_frames()[s]
        cand_txt = [cand_text(s, con[s].loc[k], a) for k, a in zip(cand.code, cand.best_alias)]
        for kind in ("ctx", "noctx"):
            q = [query_text(s, d.entity.iat[m], d.chunk_text.iat[m], d.sub_in.iat[m], kind == "ctx") for m in cand.mention]
            ks = list(zip([kind] * len(q), q, cand_txt))
            keys[(s, kind)] = ks
            for k in ks:
                if k not in lut:
                    todo.setdefault(kind, set()).add(k)
    new = []
    for kind, ks in todo.items():
        ks = sorted(ks)
        ce = CrossEncoder(str(C.MODELS / ("cross-ctx" if kind == "ctx" else "cross-noctx")),
                          max_len=C.CROSS_MAX_LEN if kind == "ctx" else C.COMPARATOR_MAX_LEN)
        sc = ce.score([k[1] for k in ks], [k[2] for k in ks])
        del ce
        for k, v in zip(ks, sc):
            lut[k] = float(v)
        new.append(pd.DataFrame({"kind": kind, "q": [k[1] for k in ks], "c": [k[2] for k in ks], "score": sc}))
        print(f"  scored {len(ks)} new {kind} pairs", flush=True)
    if new:
        pd.concat([cache] + new, ignore_index=True).to_parquet(CE_CACHE)
    for s, cand in pools.items():
        cand["ce_ctx"] = [lut[k] for k in keys[(s, "ctx")]]
        cand["ce_noctx"] = [lut[k] for k in keys[(s, "noctx")]]
    return time.perf_counter() - t0


def build_pools(opts: dict) -> tuple[dict, tuple]:
    """opts: snomed_tag_scope (bool), icd_siblings (bool), pool_size (int).
    Returns pools for ALL dev rows (diagnosis rows rebuilt) and (p50, p95) ms/term for the rebuilt path."""
    from termmap.dense import Encoder
    from termmap.retrieval import GoldMemory, Mention, SystemRetriever, load_frontend
    from termmap.data import load_split
    C.seed_everything()
    van = K.vanilla_pools()
    ab, tg = load_frontend()
    enc = Encoder(str(C.MODELS / "sapbert-ft"))
    train = pd.concat([load_split(s) for s in opts.get("memory_splits", ["train"])], ignore_index=True)
    mem_all = GoldMemory(train)
    mem_all.attach_dense(enc)
    mem_diag = None
    if opts.get("snomed_tag_scope"):
        mem_diag = GoldMemory(K.diag_train())  # no non-diagnosis training entities as aliases
        mem_diag.attach_dense(enc)
    tags = set(allowed_tags()) if opts.get("snomed_tag_scope") else None
    tree = build_icd_tree() if opts.get("icd_siblings") else None
    size = opts.get("pool_size", 64)
    pools, times = {}, []
    for s in K.SYSTEMS:
        d = K.dev_frames()[s]
        R = SystemRetriever(s)
        idx = np.where(d.is_diag)[0]
        ms = [Mention(d.entity.iat[m], d.chunk_text.iat[m], d.group.iat[m]) for m in idx]
        scoped = s == C.SNOMED and tags is not None
        rows = []
        for b in range(0, len(ms), 8):  # small batches -> per-term latency distribution
            t0 = time.perf_counter()
            if scoped:
                cand = R.retrieve(ms[b:b + 8], ab, tg, encoder=enc, memory=mem_diag, topk=300, fused_k=200, max_cand=400)
                tag = K.concepts()[s].tag.reindex(cand.code).to_numpy()
                cand = cand[np.isin(tag, list(tags))]
                cand = cand.groupby("mention", sort=False).head(size)
            else:
                cand = R.retrieve(ms[b:b + 8], ab, tg, encoder=enc, memory=mem_all, max_cand=size)
            if tree is not None and s == C.ICD:
                extra_rows = []
                for mi, g in cand.groupby("mention", sort=False):
                    top = g.sort_values("fused_rank").code.head(20).tolist()
                    have = set(g.code)
                    for code in icd_expansions(top, tree):
                        if code not in have and code in R.code2idx:
                            ci = R.code2idx[code]
                            extra_rows.append(dict(mention=mi, cidx=ci, code=code, origin=1, rrf=0.0, tfidf=0.0, tfidf_rank=999,
                                                   bm25=0.0, bm25_rank=999, dense=0.0, dense_rank=999,
                                                   best_alias=R.concepts.preferred.iat[ci], best_alias_src="expand",
                                                   best_kind="none", mem_sim=0.0, mem_exact=0, mem_dsim=0.0))
                if extra_rows:
                    cand = pd.concat([cand, pd.DataFrame(extra_rows)], ignore_index=True)
            cand = cand.copy()
            cand["mention"] = idx[b + cand.mention.to_numpy()]
            rows.append(cand)
            times.append((time.perf_counter() - t0) * 1000 / max(1, len(ms[b:b + 8])))
        new = pd.concat(rows, ignore_index=True)
        new["fused_rank"] = new.groupby("mention").cumcount()
        pools[s] = pd.concat([van[s][~van[s].mention.isin(idx)], new], ignore_index=True).sort_values(
            ["mention", "fused_rank"], kind="stable").reset_index(drop=True)
        del R
    del enc
    K._cache.pop("adj", None)
    gpu = score_pools(pools)
    n = sum(int(K.dev_frames()[s].is_diag.sum()) for s in K.SYSTEMS)
    lat = (float(np.percentile(times, 50)) + 1000 * gpu / n, float(np.percentile(times, 95)) + 1000 * gpu / n)
    return pools, lat
