"""Nested leave-one-recording-out design, plus the single-level protocol as a secondary analysis.

For the healthy baseline (15 recordings) and each outer fold o:
  inner LORO over the 14 remaining recordings -> pooled out-of-fold healthy scores -> thresholds;
  fit representation + detector on the 14 -> score every window not used for fitting
  (held-out healthy recording o and all fault recordings).
Usage: python run_nested.py Valve <seed> <comma-separated models> [sensor group]
Output folder: environment variable OUTDIR (default runs/ for all channels, runs_groups/ for a sensor group).
"""
import sys, numpy as np, pandas as pd, time, os
from extract import FEATURES
from models import Rep, Detector, thresholds, THR_ORDER

GROUPS = {'all': lambda c: True, 'accel': lambda c: c.startswith('A'), 'pressure': lambda c: c.startswith('PR'),
          'nomic': lambda c: not c.startswith('M'), 'mic': lambda c: c.startswith('M'), 'current': lambda c: c.startswith('I')}

def run(baseline, seed, models, group='all'):
    f = pd.read_parquet('data/features.parquet')
    X = f[[c for c in FEATURES if GROUPS[group](c)]].values; rec = f.rec.values; win = f.win.values
    base_recs = sorted(set(r for r in rec if r.startswith(f'{baseline}_P1_')), key=lambda r: int(r.split('_R')[1]))
    out = {}
    for name in models:
        t0 = time.time()
        def fit_score(train_recs, score_mask):
            m = np.isin(rec, train_recs)
            rep = Rep().fit(X[m]); Ztr = rep.transform(X[m])
            D = Detector(name, seed).fit(Ztr, rec[m], win[m])
            s = np.full(len(X), np.nan)
            Z = rep.transform(X[score_mask]); s[score_mask] = D.score(Z, rec[score_mask], win[score_mask])
            return s, rep.d, D
        outer_scores = np.full((len(base_recs), len(X)), np.nan)
        outer_thr = np.zeros((len(base_recs), len(THR_ORDER)))
        dims = []
        for o, ro in enumerate(base_recs):
            inner = [r for r in base_recs if r != ro]
            oof = np.full(len(X), np.nan)
            for ri in inner:
                tr = [r for r in inner if r != ri]
                s, _, _ = fit_score(tr, rec == ri); oof[rec == ri] = s[rec == ri]
            th = thresholds(oof[np.isin(rec, inner)])
            outer_thr[o] = [th[k] for k in THR_ORDER]
            s, d, _ = fit_score(inner, ~np.isin(rec, inner)); outer_scores[o] = s; dims.append(d)
        # single-level protocol (secondary analysis)
        oof = np.full(len(X), np.nan)
        for ri in base_recs:
            tr = [r for r in base_recs if r != ri]
            s, _, _ = fit_score(tr, rec == ri); oof[rec == ri] = s[rec == ri]
        th = thresholds(oof[np.isin(rec, base_recs)])
        sfin, dfin, Dfin = fit_score(base_recs, ~np.isin(rec, base_recs))
        out[name] = dict(outer_scores=outer_scores.astype(np.float32), outer_thr=outer_thr, dims=np.array(dims),
                         sl_oof=oof.astype(np.float32), sl_thr=np.array([th[k] for k in THR_ORDER]),
                         sl_final=sfin.astype(np.float32), sl_dim=dfin,
                         epochs=getattr(getattr(Dfin, 'net', None), 'epochs_run', -1),
                         core_frac=getattr(Dfin, 'core_frac', np.nan), ee_fallback=getattr(Dfin, 'ee_fallback', False))
        print(baseline, seed, name, f'{time.time()-t0:.0f}s', flush=True)
    os.makedirs('runs', exist_ok=True)
    tag = '' if group == 'all' else f'_G{group}'
    os.makedirs('runs_groups', exist_ok=True)
    od = os.environ.get('OUTDIR', 'runs' if group == 'all' else 'runs_groups'); os.makedirs(od, exist_ok=True)
    np.save(f'{od}/{baseline}_s{seed}_{"-".join(models)}.npy' if group == 'all' else f'{od}/{baseline}_{group}_s{seed}.npy', out, allow_pickle=True)

if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3].split(','), sys.argv[4] if len(sys.argv) > 4 else 'all')
