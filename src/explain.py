"""Attribution on held-out data.

For outer folds of the nested design, each detector is fitted on 14 healthy recordings and
evaluated on the held-out healthy recording versus the valve-fault recordings, so no
evaluation window was used for fitting.
  * Permutation importance: each of the 156 raw features is permuted across the evaluation
    windows (5 repeats); importance = drop in ROC-AUC.  Computed for all conditions and,
    separately, for conditions that do / do not involve the 2S_DV valve (composition effect).
  * Kernel SHAP (shap.KernelExplainer) in the retained PCA space, 8-centroid k-means background,
    24 held-out healthy + 24 fault instances per draw, three independent draws to assess
    sampling stability.  The projection onto raw features (|loading|-weighted) is a heuristic.
"""
import numpy as np, pandas as pd, sys, time, warnings
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from extract import FEATURES
from models import Rep, Detector
warnings.filterwarnings('ignore')
NONSEQ = ['IF', 'OCSVM', 'AE', 'VAE', 'LOF', 'KMeans', 'DBSCAN', 'Z', 'MD', 'EE']
f = pd.read_parquet('data/features.parquet'); X = f[FEATURES].values; rec = f.rec.values; win = f.win.values
B = sys.argv[1]; FOLDS = [int(x) for x in sys.argv[2].split(',')]; MODE = sys.argv[3]
base = sorted({r for r in rec if r.startswith(f'{B}_P1_')}, key=lambda r: int(r.split('_R')[1]))
faults = [f'Valve_P{p}' for p in range(2, 26)]
s2dv = {f'Valve_P{p}' for p in list(range(2, 6)) + list(range(18, 26))}
cond_of = np.array(['_'.join(r.split('_')[:2]) for r in rec])
rows, shap_rows = [], []
for o in FOLDS:
    ro = base[o]; tr = np.isin(rec, [r for r in base if r != ro])
    ev = (rec == ro) | np.isin(cond_of, faults)
    y = np.isin(cond_of[ev], faults).astype(int)
    sub = {'all': np.ones(ev.sum(), bool), '2S_DV': (y == 0) | np.isin(cond_of[ev], list(s2dv)),
           'non-2S_DV': (y == 0) | (np.isin(cond_of[ev], faults) & ~np.isin(cond_of[ev], list(s2dv)))}
    rep = Rep().fit(X[tr]); Ztr = rep.transform(X[tr]); Xe = X[ev]
    for m in NONSEQ:
        t0 = time.time()
        D = Detector(m, 42).fit(Ztr)
        if MODE == 'perm':
            base_s = D.score(rep.transform(Xe))
            base_auc = {k: roc_auc_score(y[s], base_s[s]) for k, s in sub.items()}
            rng = np.random.default_rng(101 + o)
            for j, fn in enumerate(FEATURES):
                for rpt in range(5):
                    Xp = Xe.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); sp = D.score(rep.transform(Xp))
                    for k, s in sub.items():
                        rows.append(dict(fold=o, model=m, feature=fn, rep=rpt, subset=k,
                                         d_auc=base_auc[k] - roc_auc_score(y[s], sp[s])))
        else:
            import shap
            bg = KMeans(8, n_init=10, random_state=0).fit(Ztr).cluster_centers_
            Ze = rep.transform(Xe)
            for draw in range(3):
                rng = np.random.default_rng(1000*draw + o)
                ih = rng.choice(np.flatnonzero(y == 0), 24, replace=False)   # held-out healthy (48 available)
                iff = rng.choice(np.flatnonzero(y == 1), 24, replace=False)
                idx = np.r_[ih, iff]
                ex = shap.KernelExplainer(D.score, bg, silent=True)
                sv = ex.shap_values(Ze[idx], nsamples=2*Ze.shape[1]+512, silent=True)
                for i, ii in enumerate(idx):
                    for c in range(Ze.shape[1]):
                        shap_rows.append(dict(fold=o, model=m, draw=draw, inst=int(ii), y=int(y[ii]), pc=c+1,
                                              shap=float(sv[i, c]), val=float(Ze[ii, c])))
            # loadings of this fold's PCA for the heuristic projection
            L = np.abs(rep.pca.components_)                # d x kept
            pd.DataFrame(L, columns=np.array(FEATURES)[rep.keep]).assign(fold=o, pc=np.arange(1, L.shape[0]+1)) \
              .to_csv(f'runs/loadings_{B}_f{o}.csv', index=False)
        print(B, MODE, o, m, f'{time.time()-t0:.0f}s', flush=True)
out = pd.DataFrame(rows if MODE == 'perm' else shap_rows)
out.to_parquet(f'runs/{MODE}_{B}_{"-".join(map(str, FOLDS))}.parquet')
