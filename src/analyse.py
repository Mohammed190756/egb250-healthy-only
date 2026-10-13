"""Shared evaluation helpers: load the nested-LORO score files and compute detection, specificity and related metrics."""
import numpy as np, pandas as pd, glob, json, re, os
from scipy.stats import friedmanchisquare, shapiro, rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from models import MODELS, NAMES, THR_ORDER, STOCHASTIC

F = pd.read_parquet('data/features.parquet')[['rec', 'win']]
REC = F.rec.values
P99 = THR_ORDER.index('p99')
SEEDS = [0, 1, 2, 3, 4, 42]
TAXON = {}
for p in range(2, 26):
    pos = ['2S_DV', '2S_IV', '1S_DV', '1S_IVb', '2S_IV+2S_DV', '1S_IVb+2S_DV'][(p-2)//4]
    TAXON[f'Valve_P{p}'] = ('Valve', pos, ['Seat wear', 'Plate corrosion', 'Plate fracture', 'Spring break'][(p-2) % 4])
COND = list(TAXON)                                       # 24 valve-fault conditions, paper order
CAMPAIGN = {c: 'Valve' for c in COND}                    # all conditions belong to the valve sub-dataset
def cond_mask(c): return np.array([r.startswith(c + '_R') for r in REC])
CMASK = {c: cond_mask(c) for c in COND}
HEALTHY = {b: sorted({r for r in REC if r.startswith(f'{b}_P1_')}, key=lambda r: int(r.split('_R')[1])) for b in ['Valve']}

def load():
    D = {}
    for f in glob.glob('runs/*.npy'):
        b, s = re.match(r'runs/(\w+?)_s(\d+)_', f).groups()
        for m, v in np.load(f, allow_pickle=True).item().items():
            D[(b, int(s), m)] = v
    return D

def seeds_for(D, b, m):
    return [s for s in SEEDS if (b, s, m) in D]

# ------------------------------------------------------------------ per-fold statistics (nested design)
def fold_stats(v, b, thr_idx=P99, conds=None):
    """For each outer fold o: healthy held-out scores, fault flags per condition, U statistic per condition."""
    conds = conds or [c for c in COND if CAMPAIGN[c] == b]
    out = []
    for o, ro in enumerate(HEALTHY[b]):
        s = v['outer_scores'][o]; t = v['outer_thr'][o, thr_idx]
        h = s[REC == ro]; h = h[~np.isnan(h)]
        row = dict(h_n=len(h), h_fp=int((h > t).sum()), cond={})
        for c in conds:
            f = s[CMASK[c]]; f = f[~np.isnan(f)]
            # Mann-Whitney U / (n_h n_f): probability a fault window outscores a held-out healthy window
            u = (np.sum(f[:, None] > h[None, :]) + 0.5*np.sum(f[:, None] == h[None, :])) / (len(f)*len(h))
            row['cond'][c] = dict(n=len(f), tp=int((f > t).sum()), u=u, scores=f, thr=t)
        row['h'] = h; out.append(row)
    return out

def summarise(stats_by_seed, conds):
    """Average per-fold statistics over seeds -> headline metrics."""
    det, spec, auc, prauc, bacc, prec, f1 = [], [], [], [], [], [], []
    per_cond = {c: [] for c in conds}
    for st in stats_by_seed:
        H = sum(r['h_n'] for r in st); FP = sum(r['h_fp'] for r in st)
        sp = 1 - FP/H
        d_f, a_f, p_f = [], [], []
        for r in st:
            tp = sum(r['cond'][c]['tp'] for c in conds); n = sum(r['cond'][c]['n'] for c in conds)
            d_f.append(tp/n)
            fs = np.concatenate([r['cond'][c]['scores'] for c in conds]); h = r['h']
            y = np.r_[np.zeros(len(h)), np.ones(len(fs))]; sc = np.r_[h, fs]
            a_f.append(roc_auc_score(y, sc)); p_f.append(average_precision_score(y, sc))
        for c in conds:
            per_cond[c].append(np.mean([r['cond'][c]['tp']/r['cond'][c]['n'] for r in st]))
        dr = np.mean(d_f); det.append(dr); spec.append(sp); auc.append(np.mean(a_f)); prauc.append(np.mean(p_f))
        bacc.append((dr+sp)/2)
        nf = np.mean([sum(r['cond'][c]['n'] for c in conds) for r in st])
        tp_tot = dr*nf                      # expected true positives per fold-model
        fp_rate = FP/H                      # held-out healthy exceedance
        # precision at the constructed class ratio of one fold (48 healthy vs nf fault windows)
        hpf = H/len(st); p = tp_tot/(tp_tot + fp_rate*hpf) if tp_tot + fp_rate*hpf > 0 else np.nan
        prec.append(p); f1.append(2*p*dr/(p+dr) if p+dr > 0 else np.nan)
    agg = lambda x: (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)
    return dict(det=agg(det), spec=agg(spec), far=agg([1-x for x in spec]), auc=agg(auc), prauc=agg(prauc),
                bacc=agg(bacc), prec=agg(prec), f1=agg(f1), n_seeds=len(stats_by_seed),
                per_cond={c: float(np.mean(v)) for c, v in per_cond.items()},
                H=sum(r['h_n'] for r in stats_by_seed[0]), FP=float(np.mean([sum(r['h_fp'] for r in st) for st in stats_by_seed])),
                NF=sum(stats_by_seed[0][0]['cond'][c]['n'] for c in conds))

def bootstrap(stats_by_seed, conds, B=2000, seed=42):
    """Recording-level bootstrap: resample fault recordings (conditions) and held-out healthy recordings (folds)."""
    rng = np.random.default_rng(seed)
    K = len(stats_by_seed[0]); C = len(conds)
    TP = np.mean([[[r['cond'][c]['tp'] for c in conds] for r in st] for st in stats_by_seed], 0)  # K x C
    N = np.array([[r['cond'][c]['n'] for c in conds] for r in stats_by_seed[0]])
    U = np.mean([[[r['cond'][c]['u'] for c in conds] for r in st] for st in stats_by_seed], 0)
    FP = np.mean([[r['h_fp'] for r in st] for st in stats_by_seed], 0); HN = np.array([r['h_n'] for r in stats_by_seed[0]])
    det, spec, auc = [], [], []
    for _ in range(B):
        ko = rng.integers(0, K, K); kc = rng.integers(0, C, C)
        det.append(np.mean(TP[np.ix_(ko, kc)].sum(1)/N[np.ix_(ko, kc)].sum(1)))
        spec.append(1 - FP[ko].sum()/HN[ko].sum())
        auc.append(np.mean((U[np.ix_(ko, kc)]*N[np.ix_(ko, kc)]).sum(1)/N[np.ix_(ko, kc)].sum(1)))
    q = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return dict(det=q(det), spec=q(spec), auc=q(auc))

def nemenyi_cd(k, n, q=3.219):
    return q*np.sqrt(k*(k+1)/(6*n))
