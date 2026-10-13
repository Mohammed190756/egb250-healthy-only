import numpy as np, pandas as pd, json, os
from scipy.stats import friedmanchisquare, shapiro, rankdata
from sklearn.metrics import roc_auc_score
from analyse import *

os.makedirs('tables', exist_ok=True)
D = load(); K = {}
CONDS = {'Valve': list(COND)}
pct = lambda x: f'{100*x:.1f}'

def stats(b, m, thr=P99, conds=None):
    return [fold_stats(D[(b, s, m)], b, thr, conds or CONDS[b]) for s in seeds_for(D, b, m)]

# ------------------------------------------------------------ primary: nested LORO on the Valve/P1 baseline
prim, boot, percond = {}, {}, {}
for m in MODELS:
    st = stats('Valve', m); sv = summarise(st, CONDS['Valve'])
    prim[m] = sv; boot[m] = bootstrap(st, CONDS['Valve']); percond[m] = sv['per_cond']

rows = []
for m in MODELS:
    s = prim[m]
    rows.append(dict(Model=NAMES[m], Seeds=s['n_seeds'],
                     Detection=f"{pct(s['det'][0])} ± {100*s['det'][1]:.1f}",
                     Specificity=f"{pct(s['spec'][0])}", FAR=f"{pct(s['far'][0])}",
                     BalancedAcc=pct(s['bacc'][0]), ROC_AUC=f"{s['auc'][0]:.3f}", PR_AUC=f"{s['prauc'][0]:.3f}",
                     Precision=pct(s['prec'][0]), F1=pct(s['f1'][0]),
                     FP_of_H=f"{s['FP']:.1f}/{s['H']}"))
pd.DataFrame(rows).to_csv('tables/T_primary_metrics.csv', index=False)

rows = []
for m in MODELS:
    s = prim[m]; bt = boot[m]
    rows.append(dict(Model=NAMES[m],
                     Detection=f"{pct(s['det'][0])} ({pct(bt['det'][0])}–{pct(bt['det'][1])})",
                     Specificity=f"{pct(s['spec'][0])} ({pct(bt['spec'][0])}–{pct(bt['spec'][1])})",
                     ROC_AUC=f"{s['auc'][0]:.3f} ({bt['auc'][0]:.3f}–{bt['auc'][1]:.3f})"))
pd.DataFrame(rows).to_csv('tables/T_bootstrap.csv', index=False)

# seed stability
rows = []
for m in MODELS:
    s = prim[m]
    rows.append({'Model': NAMES[m], 'Detection (%)': f"{pct(s['det'][0])} ± {100*s['det'][1]:.2f}",
                 'ROC-AUC': f"{s['auc'][0]:.4f} ± {s['auc'][1]:.4f}", 'Seeds': s['n_seeds']})
pd.DataFrame(rows).to_csv('tables/T_seed_stability.csv', index=False)

# per-condition table (six-seed mean, mean over outer folds)
pc = pd.DataFrame({NAMES[m]: {c: 100*percond[m][c] for c in COND} for m in MODELS})
pc.insert(0, 'Mode', [TAXON[c][2] for c in COND]); pc.insert(0, 'Location', [TAXON[c][1] for c in COND])
pc.insert(0, 'Code', [c.split('_')[1] for c in COND])
pc['Mean'] = pc[[NAMES[m] for m in MODELS]].mean(1)
pc.round(3).to_csv('tables/T_per_condition.csv', index=False)
K['n_full_all_models'] = int((pc[[NAMES[m] for m in MODELS]] >= 100 - 1e-9).all(1).sum())
K['lowest_conditions'] = pc.sort_values('Mean').head(4)[['Code', 'Location', 'Mode', 'Mean']].round(1).values.tolist()
# by valve location
pc.groupby('Location')[[NAMES[m] for m in MODELS]].mean().T.round(1).to_csv('tables/T_location.csv')

# Friedman / Nemenyi over the 24 conditions (blocks in the statistical sense) -- average ranks with ties averaged
A = pc[[NAMES[m] for m in MODELS]].values
chi, p = friedmanchisquare(*A.T)
R = np.array([rankdata(-row) for row in A]).mean(0)
cd = nemenyi_cd(len(MODELS), len(A))
pairs = [(NAMES[MODELS[i]], NAMES[MODELS[j]], abs(R[i]-R[j])) for i in range(11) for j in range(i+1, 11) if abs(R[i]-R[j]) > cd]
n_tied = int((A.max(1) == A.min(1)).sum())
K['friedman'] = dict(chi2=float(chi), p=float(p), cd=float(cd), pairs=pairs, n_all_tied_blocks=n_tied)
Av = pc[[NAMES[m] for m in MODELS]].values
chv, pv_ = friedmanchisquare(*Av.T); Rv = np.array([rankdata(-row) for row in Av]).mean(0); cdv = nemenyi_cd(len(MODELS), len(Av))
pairs_v = [(NAMES[MODELS[i]], NAMES[MODELS[j]], abs(Rv[i]-Rv[j])) for i in range(11) for j in range(i+1, 11) if abs(Rv[i]-Rv[j]) > cdv]
K['friedman_valve'] = dict(chi2=float(chv), p=float(pv_), cd=float(cdv), pairs=pairs_v, ranks={NAMES[m]: float(r) for m, r in zip(MODELS, Rv)})
pd.DataFrame(dict(Model=[NAMES[m] for m in MODELS], AvgRank=R.round(2))).to_csv('tables/T_ranks.csv', index=False)

# ------------------------------------------------------------ secondary: single-level protocol
rows = []
for m in MODELS:
    v = D[('Valve', 42, m)]; t = v['sl_thr'][P99]
    h = v['sl_oof'][np.isin(REC, HEALTHY['Valve'])]; h = h[~np.isnan(h)]
    f = np.concatenate([v['sl_final'][CMASK[c]] for c in COND]); f = f[~np.isnan(f)]
    fv = np.concatenate([v['sl_final'][CMASK[c]] for c in CONDS['Valve']]); fv = fv[~np.isnan(fv)]
    rows.append(dict(Model=NAMES[m], **{'Detection (%)': round(100*(fv > t).mean(), 1),
                     'Specificity (%)': round(100*(h <= t).mean(), 1),
                     'FP / healthy': f'{int((h > t).sum())}/{len(h)}',
                     'ROC-AUC': round(roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(fv))], np.r_[h, fv]), 4)}))
pd.DataFrame(rows).to_csv('tables/T_single_level.csv', index=False)

# ------------------------------------------------------------ threshold sensitivity on held-out healthy data
rows = []
for k, name in enumerate(THR_ORDER):
    fars, dets = [], []
    for m in MODELS:
        sv = summarise(stats('Valve', m, thr=k), CONDS['Valve'])
        fars.append(sv['far'][0]); dets.append(sv['det'][0])
    rows.append({'Threshold': name, 'Mean FAR (%)': round(100*np.mean(fars), 2),
                 'FAR range (%)': f'{100*min(fars):.2f}–{100*max(fars):.2f}',
                 'Mean detection (%)': round(100*np.mean(dets), 2),
                 'Detection range (%)': f'{100*min(dets):.2f}–{100*max(dets):.2f}'})
pd.DataFrame(rows).to_csv('tables/T_thresholds.csv', index=False)
# Shapiro-Wilk on single-level pooled OOF healthy scores (valve)
sw = {}
for m in MODELS:
    h = D[('Valve', 42, m)]['sl_oof']; h = h[np.isin(REC, HEALTHY['Valve']) & ~np.isnan(h)]
    sw[m] = float(shapiro(h).pvalue)
K['shapiro_p_range'] = [min(sw.values()), max(sw.values())]; K['shapiro'] = sw

# headline aggregates
K['primary'] = {m: dict(det=prim[m]['det'][0], spec=prim[m]['spec'][0], auc=prim[m]['auc'][0]) for m in MODELS}
K['dims'] = {b: sorted(set(int(x) for m in MODELS for s in seeds_for(D, b, m) for x in D[(b, s, m)]['dims'])) for b in ['Valve']}
K['final_dim'] = {b: int(D[(b, 42, 'MD')]['sl_dim']) for b in ['Valve']}
K['epochs'] = {b: {m: int(D[(b, 42, m)]['epochs']) for m in ['AE', 'VAE', 'LSTMAE']} for b in ['Valve']}
K['dbscan_core'] = {b: float(D[(b, 42, 'DBSCAN')]['core_frac']) for b in ['Valve']}
K['ee_fallback'] = {b: bool(D[(b, 42, 'EE')]['ee_fallback']) for b in ['Valve']}
K['seeds'] = {m: seeds_for(D, 'Valve', m) for m in MODELS}
json.dump(K, open('tables/key_numbers.json', 'w'), indent=1, default=float)
print(json.dumps({k: K[k] for k in ['friedman', 'n_full_all_models', 'lowest_conditions', 'dims', 'final_dim', 'epochs', 'shapiro_p_range']}, indent=1, default=float))
