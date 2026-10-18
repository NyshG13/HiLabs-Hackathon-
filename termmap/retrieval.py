"""Candidate generation per system (plan section 3, steps 3-4).

char n-gram TF-IDF + word BM25 + dense SapBERT, each searched with every query variant, aggregated
to concepts by best alias, fused with reciprocal rank, then augmented with gold-memory candidates and
hierarchy expansion (ICD: siblings, children and the 'unspecified' sibling; SNOMED: lexical family as a
licence-free stand-in for IS-A neighbours).
"""
from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import config as C
from .normalize import AbbreviationTable, TypoGate, clean, content_tokens, variants
from .resources import load_aliases, load_concepts


# ----------------------------------------------------------------------------------------------
def _topk_sparse(S: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = S.shape[0]
    I = np.full((n, k), -1, np.int64)
    V = np.zeros((n, k), np.float32)
    for r in range(n):
        a, b = S.indptr[r], S.indptr[r + 1]
        if a == b:
            continue
        d, ix = S.data[a:b], S.indices[a:b]
        if len(d) > k:
            sel = np.argpartition(-d, k - 1)[:k]
            d, ix = d[sel], ix[sel]
        o = np.lexsort((ix, -d))  # deterministic tie-break on alias index
        I[r, :len(o)], V[r, :len(o)] = ix[o], d[o]
    return V, I


class LexicalIndex:
    def __init__(self):
        self.tfidf: TfidfVectorizer | None = None
        self.A_T = None
        self.bm25_vec: CountVectorizer | None = None
        self.W_T = None

    def fit(self, texts: list[str], k1: float = 1.2, b: float = 0.75) -> "LexicalIndex":
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True,
                                     dtype=np.float32)
        A = self.tfidf.fit_transform(texts)
        self.A_T = A.T.tocsr()
        self.bm25_vec = CountVectorizer(token_pattern=r"[a-z0-9+']+", dtype=np.float32)
        tf = self.bm25_vec.fit_transform(texts).tocsr()
        dl = np.asarray(tf.sum(1)).ravel()
        avg = dl.mean()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (tf.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
        rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        d = tf.data
        tf.data = (idf[tf.indices] * d * (k1 + 1) / (d + k1 * (1 - b + b * dl[rows] / avg))).astype(np.float32)
        self.W_T = tf.T.tocsr()
        return self

    def search(self, queries: list[str], k: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out = {"tfidf": [], "bm25": []}
        for i in range(0, len(queries), 32):
            q = queries[i:i + 32]
            out["tfidf"].append(_topk_sparse((self.tfidf.transform(q) @ self.A_T).tocsr(), k))
            Q = self.bm25_vec.transform(q)
            Q.data[:] = 1.0
            out["bm25"].append(_topk_sparse((Q @ self.W_T).tocsr(), k))
        return {key: (np.concatenate([v[0] for v in val]), np.concatenate([v[1] for v in val]))
                for key, val in out.items()}


# ----------------------------------------------------------------------------------------------
class GoldMemory:
    """Normalised training mention -> code, per system. Shortcut when exact, prior feature otherwise."""

    def __init__(self, df: pd.DataFrame):
        self.by_sys = {}
        for system in C.SYSTEMS:
            d = df[df.coding_system == system]
            ents = [clean(e) for e in d.entity]
            if not ents:
                continue
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
            M = vec.fit_transform(ents)
            self.by_sys[system] = dict(vec=vec, M_T=M.T.tocsr(), ents=ents, codes=d.code.tolist(),
                                       groups=d.group.tolist(), exact=self._exact(ents, d.code.tolist()))
        self.prior = {s: Counter(df.loc[df.coding_system == s, "code"]) for s in C.SYSTEMS}
        self.cat_prior = Counter(c[:3] for c in df.loc[df.coding_system == C.ICD, "code"])

    def attach_dense(self, encoder) -> None:
        """Embed the memory mentions so paraphrases ('dry heaved' ~ 'vomiting') hit the memory too."""
        self.encoder = encoder
        for m in self.by_sys.values():
            m["emb"] = encoder.encode(m["ents"])

    @staticmethod
    def _exact(ents, codes):
        m = defaultdict(Counter)
        for e, c in zip(ents, codes):
            m[e][c] += 1
        return m

    def lookup(self, system: str, mentions: list[str], groups: list[str] | None = None, k: int = 10):
        """Return per mention {code: (char_sim, exact, dense_sim)} excluding rows from the same group."""
        res = [dict() for _ in mentions]
        if system not in self.by_sys:
            return res
        m = self.by_sys[system]
        q = [clean(x) for x in mentions]
        hits = [_topk_sparse((m["vec"].transform(q) @ m["M_T"]).tocsr(), k + 5)]
        if "emb" in m:
            qe = self.encoder.encode(q).astype(np.float32)
            S = qe @ m["emb"].astype(np.float32).T
            I = np.argsort(-S, axis=1, kind="stable")[:, :k + 5]
            hits.append((np.take_along_axis(S, I, 1), I))
        for h, (V, I) in enumerate(hits):
            for r in range(len(q)):
                g = groups[r] if groups is not None else None
                n = 0
                for v, i in zip(V[r], I[r]):
                    if i < 0 or n >= k:
                        break
                    if g is not None and m["groups"][i] == g:
                        continue
                    n += 1
                    code = m["codes"][i]
                    old = res[r].get(code, (0.0, False, 0.0))
                    if h == 0:
                        res[r][code] = (max(old[0], float(v)), old[1] or m["ents"][i] == q[r], old[2])
                    else:
                        res[r][code] = (old[0], old[1], max(old[2], float(v)))
        return res


# ----------------------------------------------------------------------------------------------
@dataclass
class Mention:
    entity: str
    context: str = ""
    group: str = ""
    clause: str = ""  # context_v2: clause around the mention (used only when span_extension is on)


class SystemRetriever:
    def __init__(self, system: str, load_dense: bool = True, dense_path=None):
        self.system = system
        self.concepts = load_concepts(system)
        self.aliases = load_aliases(system)
        self.code2idx = {c: i for i, c in enumerate(self.concepts.code)}
        self.alias_cidx = self.aliases.code.map(self.code2idx).to_numpy()
        self.alias_text = self.aliases.alias.to_numpy()
        self.alias_src = self.aliases.source.to_numpy()
        p = C.PROCESSED / f"lexical_{system}.pkl"
        if p.exists():
            self.lex = pickle.loads(p.read_bytes())
        else:
            self.lex = LexicalIndex().fit(self.aliases.alias_clean.tolist())
            p.write_bytes(pickle.dumps(self.lex, protocol=4))
        self.dense = None
        dense_path = dense_path or (C.TRAINDEP / f"dense_{system}.npy")
        if load_dense and dense_path.exists():
            import numpy as _np
            from .dense import DenseIndex
            self.dense = DenseIndex(_np.load(dense_path))
        self._build_hierarchy()

    # ---- hierarchy -------------------------------------------------------------------------
    def _build_hierarchy(self):
        con = self.concepts
        if self.system == C.ICD:
            self.children = defaultdict(list)
            for c, p in zip(con.code, con.parent):
                if p:
                    self.children[p].append(c)
            self.billable = dict(zip(con.code, con.billable))
            self.unspec = dict(zip(con.code, con.unspecified))
        else:
            self.tokset = {}
            self.inv = defaultdict(list)
            for i, t in enumerate(con.preferred):
                ts = frozenset(content_tokens(t))
                if not ts or len(ts) > 8:
                    continue
                self.tokset.setdefault(ts, []).append(i)
                for w in ts:
                    self.inv[w].append(i)
            self.inv = {w: np.asarray(v, np.int32) for w, v in self.inv.items()}
            self.pref_toks = [frozenset(content_tokens(t)) for t in con.preferred]

    def expand(self, cidx: int, limit: int = 8) -> list[int]:
        code = self.concepts.code.iat[cidx]
        out: list[str] = []
        if self.system == C.ICD:
            par = self.concepts.parent.iat[cidx]
            sib = self.children.get(par, []) if par else []
            kids = self.children.get(code, [])
            out += [c for c in sib if self.unspec.get(c)]          # the 'unspecified' sibling first
            out += [c for c in kids if self.billable.get(c)][:4]  # more specific children
            if not self.billable.get(code):
                out += [c for c in kids if self.unspec.get(c)]
            out += sib[:6]
            res = [self.code2idx[c] for c in dict.fromkeys(out) if c != code]
        else:
            ts = self.pref_toks[cidx]
            if not ts:
                return []
            res = []
            for n in range(len(ts) - 1, 0, -1):  # parents-like: strict token subsets
                from itertools import combinations
                for sub in combinations(sorted(ts), n):
                    res += self.tokset.get(frozenset(sub), [])[:2]
                if len(res) >= 3:
                    break
            rare = min(ts, key=lambda w: len(self.inv.get(w, ())))
            post = self.inv.get(rare, np.zeros(0, np.int32))
            if len(post) <= 20000:
                sup = [i for i in post.tolist() if i != cidx and ts < self.pref_toks[i] and len(self.pref_toks[i]) <= len(ts) + 2]
                sup.sort(key=lambda i: (len(self.pref_toks[i]), i))
                res += sup[:limit]
            res = [i for i in dict.fromkeys(res) if i != cidx]
        return res[:limit]

    # ---- retrieval -------------------------------------------------------------------------
    def retrieve(self, mentions: list[Mention], abbr: AbbreviationTable, typo: TypoGate | None,
                 encoder=None, memory: GoldMemory | None = None, use_memory_groups: bool = False,
                 topk: int = C.TOPK_PER_RETRIEVER, fused_k: int = C.TOPK_FUSED,
                 max_cand: int = C.MAX_CANDIDATES, expand: bool = True,
                 retrievers=("tfidf", "bm25", "dense"), weights: dict | None = None,
                 span_extension: bool | None = None) -> pd.DataFrame:
        weights = weights or C.RRF_WEIGHTS
        span_extension = C.SPAN_EXTENSION if span_extension is None else span_extension
        qs, owner, kinds = [], [], []
        for mi, m in enumerate(mentions):
            mine = set()
            for kind, q in variants(m.entity, abbr, typo, m.context):
                qs.append(clean(q) or q.lower())
                owner.append(mi)
                kinds.append(kind)
                mine.add(qs[-1])
            # context_v2 span extension: the clause as ONE extra variant when it is longer than the entity
            if span_extension and m.clause and len(m.clause.split()) > len(m.entity.split()):
                q = clean(m.clause) or m.clause.lower()
                if q not in mine:
                    qs.append(q)
                    owner.append(mi)
                    kinds.append("span")
        res = self.lex.search(qs, topk) if {"tfidf", "bm25"} & set(retrievers) else {}
        if "dense" in retrievers and self.dense is not None and encoder is not None:
            res["dense"] = self.dense.search(encoder.encode(qs), topk)
        res = {k: v for k, v in res.items() if k in retrievers}

        # per mention, per concept, per retriever: best score / best rank / alias
        agg = [defaultdict(dict) for _ in mentions]
        found_by = [defaultdict(set) for _ in mentions]  # which variant kinds reached each concept
        for r, (V, I) in res.items():
            for qi in range(len(qs)):
                mi, seen, rank = owner[qi], set(), 0
                for v, ai in zip(V[qi], I[qi]):
                    if ai < 0:
                        break
                    ci = self.alias_cidx[ai]
                    if ci in seen:
                        continue
                    seen.add(ci)
                    rank += 1
                    found_by[mi][ci].add(kinds[qi])
                    cur = agg[mi][ci].get(r)
                    if cur is None or v > cur[0]:
                        agg[mi][ci][r] = (float(v), min(rank, cur[1]) if cur else rank, int(ai), kinds[qi])
                    elif rank < cur[1]:
                        agg[mi][ci][r] = (cur[0], rank, cur[2], cur[3])

        mem = None
        if memory is not None:
            mem = memory.lookup(self.system, [m.entity for m in mentions],
                                [m.group for m in mentions] if use_memory_groups else None)
        rows = []
        for mi, d in enumerate(agg):
            scored = []
            for ci, rr in d.items():
                rrf = sum(weights.get(r, 1.0) / (C.RRF_K + v[1]) for r, v in rr.items())
                scored.append((rrf, ci))
            scored.sort(key=lambda x: (-x[0], x[1]))
            cand = [ci for _, ci in scored[:fused_k]]
            origin = {ci: 0 for ci in cand}
            head, tail = cand[:10], cand[10:]
            cand = head
            if mem is not None:
                for code, (s, ex, ds) in sorted(mem[mi].items(), key=lambda kv: -max(kv[1][0], kv[1][2]))[:6]:
                    ci = self.code2idx.get(code)
                    if ci is not None and (s >= 0.5 or ds >= 0.75) and ci not in origin:
                        cand.append(ci)
                        origin[ci] = 2
            cand += tail
            if expand:
                extra = []
                for ci in [c for c in cand if origin[c] != 1][:5]:
                    for e in self.expand(ci):
                        if e not in origin:
                            extra.append(e)
                            origin[e] = 1
                # expansions take the slots just after the first 32 fused candidates
                cand = cand[:C.RERANK_TOPN - 12] + extra[:12] + cand[C.RERANK_TOPN - 12:] + extra[12:]
            cand = cand[:max_cand]
            rrf_of = dict((ci, s) for s, ci in scored)
            for rank, ci in enumerate(cand):
                rr = d.get(ci, {})
                best = max(rr.items(), key=lambda kv: kv[1][0] if kv[0] != "bm25" else -1, default=None)
                ai = best[1][2] if best else -1
                ms = mem[mi].get(self.concepts.code.iat[ci], (0.0, False, 0.0)) if mem is not None else (0.0, False, 0.0)
                rows.append(dict(
                    mention=mi, cidx=ci, code=self.concepts.code.iat[ci], origin=origin[ci], fused_rank=rank,
                    rrf=rrf_of.get(ci, 0.0),
                    tfidf=rr.get("tfidf", (0, 999))[0], tfidf_rank=rr.get("tfidf", (0, 999))[1],
                    bm25=rr.get("bm25", (0, 999))[0], bm25_rank=rr.get("bm25", (0, 999))[1],
                    dense=rr.get("dense", (0, 999))[0], dense_rank=rr.get("dense", (0, 999))[1],
                    best_alias=self.alias_text[ai] if ai >= 0 else self.concepts.preferred.iat[ci],
                    best_alias_src=self.alias_src[ai] if ai >= 0 else "expand",
                    best_kind=best[1][3] if best else "none",
                    mem_sim=ms[0], mem_exact=int(ms[1]), mem_dsim=ms[2],
                    via_span_ext=int(found_by[mi].get(ci) == {"span"}),
                ))
        return pd.DataFrame(rows)


def load_frontend(abbr_path=None) -> tuple[AbbreviationTable, TypoGate]:
    abbr_path = abbr_path or (C.TRAINDEP / "abbreviations.json")
    ab = AbbreviationTable.from_json(json.loads(abbr_path.read_text(encoding="utf8")))
    tg = TypoGate(Counter(json.loads((C.TRAINDEP / "vocab.json").read_text(encoding="utf8"))))
    return ab, tg
