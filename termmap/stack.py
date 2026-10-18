"""Learned fusion (step 6), joint consistency (step 7) and selective LLM routing (step 8).

Protocol: the cross-encoders were trained on `train`; the stacker is trained on `dev` with grouped
5-fold cross-fitting (out-of-fold scores are used for every dev-tuned choice: temperature, joint weight,
LLM gate, isotonic calibration); a final stacker fitted on all of `dev` is applied to the untouched `test`.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from . import config as C
from .features import FEATURES

LGB_PARAMS = dict(objective="lambdarank", n_estimators=400, learning_rate=0.03, num_leaves=15,
                  min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=C.SEED, deterministic=True, force_row_wise=True,
                  n_jobs=4, verbose=-1, lambdarank_truncation_level=10, label_gain=[0, 1])


class Stacker:
    def __init__(self, features=FEATURES):
        self.features = list(features)
        self.model = None

    def fit(self, X: pd.DataFrame) -> "Stacker":
        X = X.sort_values(["qid", "fused_rank"], kind="stable")
        sizes = X.groupby("qid", sort=False).size().to_numpy()
        self.model = lgb.LGBMRanker(**LGB_PARAMS)
        self.model.fit(X[self.features], X.label.astype(int), group=sizes)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features])

    def save(self, path):
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path) -> "Stacker":
        return pickle.loads(path.read_bytes())


def crossfit(X: pd.DataFrame, features=FEATURES, k: int = 5) -> np.ndarray:
    oof = np.zeros(len(X))
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(X, groups=X.group):
        st = Stacker(features).fit(X.iloc[tr])
        oof[te] = st.score(X.iloc[te])
    return oof


def mention_probs(X: pd.DataFrame, col: str, T: float = 1.0) -> np.ndarray:
    s = X[col].to_numpy() / T
    mx = X.groupby("qid")[col].transform("max").to_numpy() / T
    e = np.exp(s - mx)
    return e / pd.Series(e).groupby(X.qid.to_numpy()).transform("sum").to_numpy()


def fit_temperature(X: pd.DataFrame, col: str) -> float:
    best, bt = 1e9, 1.0
    has = X.groupby("qid").label.transform("max").to_numpy() > 0
    for T in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0):
        p = mention_probs(X, col, T)
        nll = -np.log(np.clip(p[(X.label.to_numpy() == 1) & has], 1e-9, 1)).mean()
        if nll < best:
            best, bt = nll, T
    return bt


def top1(X: pd.DataFrame, col: str) -> pd.DataFrame:
    """Best candidate per qid by `col` (ties broken by fused rank)."""
    i = X.sort_values(["qid", col, "fused_rank"], ascending=[True, False, True], kind="stable").groupby("qid").head(1)
    return i.set_index("qid")


# ----------------------------------------------------------------------------------------------
# Joint consistency (licence-free agreement signal)
# ----------------------------------------------------------------------------------------------
class Bridge:
    """SNOMED concept -> nearest ICD-10-CM codes by fine-tuned SapBERT similarity of the SNOMED preferred
    term to ICD aliases (substitute for the NLM refset 6011000124106, which needs a UTS licence), plus
    (ICD, SNOMED) pairs observed together in the training gold."""

    def __init__(self, train: pd.DataFrame, concepts: dict):
        self.con = concepts
        pairs = train.pivot_table(index=["entity", "chunk_text"], columns="coding_system", values="code",
                                  aggfunc="first").dropna()
        self.pair = Counter(zip(pairs[C.ICD], pairs[C.SNOMED]))
        self.pair_cat = Counter((i[:3], s) for i, s in self.pair)
        self.near: dict[str, list[str]] = {}
        self._enc = None

    def fill(self, sn_codes) -> None:
        todo = sorted(set(sn_codes) - set(self.near))
        if not todo:
            return
        from .dense import DenseIndex, Encoder, load_dense_index
        from .normalize import clean
        from .resources import load_aliases
        if self._enc is None:
            self._enc = Encoder()
            self._idx = DenseIndex(load_dense_index(C.ICD))
            self._alias_code = load_aliases(C.ICD).code.to_numpy()
        pref = self.con[C.SNOMED].preferred
        q = self._enc.encode([clean(pref.get(c, "")) for c in todo])
        V, I = self._idx.search(q, 40)
        for c, row in zip(todo, I):
            out = []
            for ai in row:
                k = self._alias_code[ai]
                if k not in out:
                    out.append(k)
                if len(out) == 3:
                    break
            self.near[c] = out

    def agree(self, icd: str, sn: str) -> float:
        if self.pair.get((icd, sn)) or icd in self.near.get(sn, [])[:3]:
            return 1.0
        if self.pair_cat.get((icd[:3], sn)) or icd[:3] in {x[:3] for x in self.near.get(sn, [])}:
            return 0.5
        return 0.0

    def crosswalk(self, sn: str) -> str | None:
        """ICD derived from a SNOMED prediction (the crosswalk baseline arm)."""
        best = [i for (i, s), n in self.pair.most_common() if s == sn]
        if best:
            return best[0]
        near = self.near.get(sn, [])
        return near[0] if near else None


def pair_index(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(icd_qid, snomed_qid) for mentions present in both systems."""
    key = df.entity + "\x1f" + df.chunk_text
    out = []
    for _, g in df.assign(key=key).groupby("key"):
        if set(g.coding_system) >= set(C.SYSTEMS):
            i = g[g.coding_system == C.ICD].qid.iat[0]
            s = g[g.coding_system == C.SNOMED].qid.iat[0]
            out.append((i, s))
    return out


def joint_marginals(X: pd.DataFrame, pairs, bridge: Bridge, w: float, prob_col: str = "p", k: int = 5):
    """Re-weight each paired mention's top-k by p(ICD) p(SNOMED) exp(w * agree), marginalised per system.
    Returns {qid: {code: new_p}} and {qid: agree_of_argmax_pair}."""
    top = {q: g.nlargest(k, prob_col)[["code", prob_col]].to_numpy() for q, g in X.groupby("qid")}
    newp, agree_flag = {}, {}
    for qi, qs in pairs:
        if qi not in top or qs not in top:
            continue
        ti, ts = top[qi], top[qs]
        bridge.fill([c for c, _ in ts])
        M = np.zeros((len(ti), len(ts)))
        A = np.zeros_like(M)
        for a, (ci, pi) in enumerate(ti):
            for b, (cs, ps) in enumerate(ts):
                A[a, b] = bridge.agree(ci, cs)
                M[a, b] = float(pi) * float(ps) * math.exp(w * A[a, b])
        M /= M.sum()
        mi, ms = M.sum(1), M.sum(0)
        newp[qi] = {c: float(v) for (c, _), v in zip(ti, mi)}
        newp[qs] = {c: float(v) for (c, _), v in zip(ts, ms)}
        a, b = np.unravel_index(np.argmax(M), M.shape)
        agree_flag[qi] = agree_flag[qs] = float(A[a, b])
    return newp, agree_flag


def apply_newp(X: pd.DataFrame, newp: dict, col_in: str = "p", col_out: str = "pj") -> None:
    """Paired mentions get their joint marginal on the top-k (others keep their mass scaled to 0)."""
    vals = X[col_in].to_numpy().copy()
    for i, (q, c) in enumerate(zip(X.qid.to_numpy(), X.code.to_numpy())):
        d = newp.get(q)
        if d is not None:
            vals[i] = d.get(c, 0.0) * (1 - 1e-6) + 1e-6 * vals[i]
    X[col_out] = vals
