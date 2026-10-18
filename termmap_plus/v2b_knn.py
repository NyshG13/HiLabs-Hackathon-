"""S2: convention kNN over diagnosis-train entities (two views: X's SapBERT on entity + window, char TF-IDF on
the entity). Neighbours come only from train and stacker rows are pooled dev+test, so no row is ever its own
neighbour (no leave-self-out needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from termmap import config as C
from termmap.normalize import clean
from termmap_plus import core as K
from termmap_plus.features_plus import window

KNN_FEATS = ["knn_vote", "knn_rank", "knn_top1", "knn_group_vote"]


def knn_votes(frames: dict, k: int = 10) -> dict:
    """{system: {mention: (Counter code->vote, Counter group->vote)}} for every row of `frames`."""
    from termmap.dense import Encoder
    enc = Encoder(str(C.MODELS / "sapbert-ft"))
    con = K.concepts()
    t = K.diag_train()
    out = {}
    for s in K.SYSTEMS:
        tr = t[t.coding_system == s].reset_index(drop=True)
        tr_text = [clean(e) + " " + window(e, c) for e, c in zip(tr.entity, tr.chunk_text)]
        Etr = enc.encode(tr_text).astype(np.float32)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(tr.entity.map(clean))
        Ttr = vec.transform(tr.entity.map(clean))
        d = frames[s]
        q_text = [clean(e) + " " + window(e, c) for e, c in zip(d.entity, d.chunk_text)]
        Eq = enc.encode(q_text).astype(np.float32)
        Tq = vec.transform(d.entity.map(clean))
        S = 0.5 * (Eq @ Etr.T) + 0.5 * (Tq @ Ttr.T).toarray()
        idx = np.argsort(-S, axis=1, kind="stable")[:, :k]
        grp = (lambda c: con[s].tag.get(c, "")) if s == C.SNOMED else (lambda c: c[:3])
        res = {}
        for m in range(len(d)):
            cv, gv = {}, {}
            for j in idx[m]:
                w = max(0.0, float(S[m, j]))
                c = tr.code.iat[j]
                cv[c] = cv.get(c, 0.0) + w
                g = grp(c)
                gv[g] = gv.get(g, 0.0) + w
            tot = sum(cv.values()) or 1.0
            res[m] = ({c: v / tot for c, v in cv.items()}, {g: v / tot for g, v in gv.items()})
        out[s] = res
    del enc
    return out


def add_knn_candidates(pools: dict, votes: dict, top: int = 3) -> dict:
    """Append the top-voted codes missing from each diagnosis row's pool (origin=2, no retrieval scores)."""
    con = K.concepts()
    new = {}
    for s, cand in pools.items():
        d = K.dev_frames()[s]
        have = cand.groupby("mention").code.agg(set).to_dict()
        rows = []
        for m in np.where(d.is_diag)[0]:
            cv = votes[s][m][0]
            for c, _ in sorted(cv.items(), key=lambda kv: -kv[1])[:top]:
                if c in con[s].index and c not in have.get(m, set()):
                    rows.append(dict(mention=m, cidx=-1, code=c, origin=2, rrf=0.0, tfidf=0.0, tfidf_rank=999, bm25=0.0,
                                     bm25_rank=999, dense=0.0, dense_rank=999, best_alias=con[s].preferred[c],
                                     best_alias_src="expand", best_kind="none", mem_sim=0.0, mem_exact=0, mem_dsim=0.0))
        add = pd.DataFrame(rows)
        c2 = pd.concat([cand, add], ignore_index=True) if len(add) else cand
        c2 = c2.sort_values(["mention", "fused_rank"], na_position="last", kind="stable").reset_index(drop=True)
        c2["fused_rank"] = c2.groupby("mention").cumcount()
        new[s] = c2
        print(f"  kNN added {len(add)} candidates for {s}", flush=True)
    return new


def knn_features(X: pd.DataFrame, votes: dict) -> pd.DataFrame:
    X = X.copy()
    con = K.concepts()
    sysm = X.qid.str.split(":", n=1, expand=True)
    vote, gvote = np.zeros(len(X)), np.zeros(len(X))
    for i, (s, m, c) in enumerate(zip(sysm[0].to_numpy(), sysm[1].astype(int).to_numpy(), X.code.to_numpy())):
        cv, gv = votes[s][m]
        vote[i] = cv.get(c, 0.0)
        gvote[i] = gv.get(con[s].tag.get(c, "") if s == C.SNOMED else c[:3], 0.0)
    X["knn_vote"], X["knn_group_vote"] = vote, gvote
    X["knn_rank"] = np.where(vote > 0, X.assign(v=vote).groupby("qid").v.rank(ascending=False, method="min"), 99)
    X["knn_top1"] = ((X.knn_rank == 1) & (vote > 0)).astype(int)
    return X
