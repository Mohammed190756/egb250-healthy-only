"""Microphone-free configuration, healthy baseline from another session, and PCA-retention sensitivity.

Reads the nested-design outputs in runs/ (all channels), runs_nomic/ (all channels except the
microphones) and, if present, runs_p95/, runs_p99/ and runs_nopca/. For every outer fold the
model and its 99th-percentile threshold score
  * the held-out Valve/P1 recording (specificity),
  * the 24 valve-fault recordings (detection rate, ROC-AUC), and
  * the 15 healthy recordings of the bearing sub-dataset (Bearing/P1), a healthy baseline of the
    same machine recorded in another acquisition session: share of windows flagged.
Metrics are averaged over the fifteen outer folds and then over seeds.
"""
import numpy as np, pandas as pd, glob, os
from sklearn.metrics import roc_auc_score
from analyse import REC, HEALTHY, CMASK, COND, P99, NAMES, MODELS

FAULT = np.zeros(len(REC), bool)
for c in COND: FAULT |= CMASK[c]
OTHER = np.array([r.startswith('Bearing_P1_') for r in REC])

def fold_metrics(v):
    out = dict(det=[], spec=[], auc=[], other_flag=[])
    for o, ro in enumerate(HEALTHY['Valve']):
        s = v['outer_scores'][o]; t = v['outer_thr'][o, P99]
        h = s[REC == ro]; h = h[~np.isnan(h)]; f = s[FAULT]; f = f[~np.isnan(f)]
        out['det'].append((f > t).mean()); out['spec'].append((h <= t).mean())
        out['auc'].append(roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(f))], np.r_[h, f]))
        if OTHER.any():
            x = s[OTHER]; x = x[~np.isnan(x)]; out['other_flag'].append((x > t).mean())
    return {k: float(np.mean(x)) for k, x in out.items() if x}

def collect(folder):
    R = {}
    for fn in sorted(glob.glob(f'{folder}/Valve*.npy')):
        for m, v in np.load(fn, allow_pickle=True).item().items():
            R.setdefault(m, []).append(fold_metrics(v))
    rows = []
    for m in MODELS:
        if m not in R: continue
        L = R[m]; g = lambda k: [x[k] for x in L]
        rows.append(dict(Model=NAMES[m], Seeds=len(L),
                         det=100*np.mean(g('det')), det_sd=100*np.std(g('det'), ddof=1) if len(L) > 1 else 0.0,
                         spec=100*np.mean(g('spec')), auc=np.mean(g('auc')),
                         other_flag=100*np.mean(g('other_flag')) if 'other_flag' in L[0] else np.nan,
                         other_flag_min=100*np.min(g('other_flag')) if 'other_flag' in L[0] else np.nan,
                         other_flag_max=100*np.max(g('other_flag')) if 'other_flag' in L[0] else np.nan))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    os.makedirs('tables', exist_ok=True)
    for name, folder in [('all', 'runs'), ('nomic', 'runs_nomic'), ('p95', 'runs_p95'), ('p99', 'runs_p99'), ('nopca', 'runs_nopca')]:
        if not glob.glob(f'{folder}/*.npy'): continue
        df = collect(folder); df.round(4).to_csv(f'tables/T_{name}_session.csv', index=False)
        print('==', name); print(df.round(3).to_string(index=False))
