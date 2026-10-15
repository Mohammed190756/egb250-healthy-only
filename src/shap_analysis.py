"""Kernel SHAP analysis for the valve task: detector-level tables, agreement, stability, beeswarm figure."""
import glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyse import NAMES

p = pd.concat([pd.read_parquet(f) for f in glob.glob('runs/shap_Valve_*.parquet')])
p['abs'] = p.shap.abs()
tot = p.groupby(['fold', 'model', 'draw', 'pc'])['abs'].mean().groupby(['fold', 'model', 'draw']).sum().rename('tot')
p = p.join(tot, on=['fold', 'model', 'draw'])
p['nabs'] = p['abs'] / p['tot']; p['nshap'] = p['shap'] / p['tot']
L = {f: pd.read_csv(f'runs/loadings_Valve_f{f}.csv').set_index('pc').drop(columns='fold') for f in sorted(p.fold.unique())}
GROUP = lambda c: {'A': 'Accelerometers', 'M': 'Microphones', 'I': 'Current clamps', 'P': 'Pressure'}[c[0]]

def project(q):
    out = []
    for f, qq in q.groupby('fold'):
        ci = qq.groupby('pc')['nabs'].mean(); Lf = L[f].loc[ci.index]
        Lf = Lf.div(Lf.sum(axis=1), axis=0)          # each component's attribution distributed over raw features
        out.append(Lf.mul(ci, axis=0).sum(axis=0))
    r = pd.concat(out, axis=1).mean(axis=1); return r / r.sum()

S = {}
# consensus component ranking (fold 0) and agreement between detectors
f0 = p[p.fold == 0]
M = f0.groupby(['model', 'pc'])['nabs'].mean().unstack(0)            # pcs x models
cons = M.mean(axis=1)
R = M.rank(ascending=False); n, k = R.shape
Ssum = R.sum(axis=1); W = 12 * ((Ssum - Ssum.mean())**2).sum() / (k**2 * (n**3 - n))
S['kendall_w'] = float(W)
# fault vs healthy attribution magnitude
inst = p.groupby(['fold', 'model', 'draw', 'inst', 'y'])['nabs'].sum().reset_index()
rat = inst.groupby(['model', 'y'])['nabs'].mean().unstack(); rat['ratio'] = rat[1] / rat[0]
S['ratio_all'] = float(inst[inst.y == 1].nabs.mean() / inst[inst.y == 0].nabs.mean())
# direction: mean signed SHAP of fault windows on the top consensus components
top = cons.sort_values(ascending=False).index[:5]
S['top_pcs_fold0'] = [int(c) for c in top]
S['fault_pos_share'] = float((f0[(f0.y == 1)].groupby(['model', 'draw', 'inst']).nshap.sum() > 0).mean())
# raw projection: consensus, per fold stability, sensor-group shares
proj = project(p); S['proj_top'] = proj.sort_values(ascending=False).head(10).round(4).to_dict()
pf = {f: project(p[p.fold == f]) for f in L}
fs = sorted(pf); S['fold_rho'] = [float(spearmanr(pf[a], pf[b]).correlation) for i, a in enumerate(fs) for b in fs[i+1:]]
pfault = project(p[p.y == 1]); phealthy = project(p[p.y == 0])
S['group_share'] = proj.groupby(lambda c: GROUP(c)).sum().round(3).to_dict()
S['group_share_fault'] = pfault.groupby(lambda c: GROUP(c)).sum().round(3).to_dict()
# stability of the component ranking between instance draws
rs = []
for (f, m), q in p.groupby(['fold', 'model']):
    t = q.groupby(['draw', 'pc'])['nabs'].mean().unstack(0)
    rs += [spearmanr(t[a], t[b]).correlation for a in range(3) for b in range(a+1, 3)]
S['draw_rho_mean'], S['draw_rho_min'] = float(np.nanmean(rs)), float(np.nanmin(rs))
# per-detector table
rows = []
for m in M.columns:
    pm = project(p[p.model == m]).sort_values(ascending=False)
    rows.append({'Detector': NAMES[m],
                 'Top components (fold 1)': ', '.join(f'PC{c}' for c in M[m].sort_values(ascending=False).index[:3]),
                 'Top raw features (projected)': ', '.join(pm.index[:3]),
                 'Fault/healthy |SHAP| ratio': f"{rat.loc[m, 'ratio']:.2f}",
                 'ρ with consensus': f"{spearmanr(M[m], cons).correlation:.2f}"})
order = ['IF', 'OCSVM', 'AE', 'VAE', 'LOF', 'KMeans', 'DBSCAN', 'Z', 'MD', 'EE']
det = pd.DataFrame(rows).set_index('Detector').loc[[NAMES[m] for m in order]].reset_index()
det.to_csv('tables/shap_detectors.csv', index=False)
S['rho_cons_range'] = [float(min(spearmanr(M[m], cons).correlation for m in M)), float(max(spearmanr(M[m], cons).correlation for m in M))]
json.dump(S, open('tables/shap_stats.json', 'w'), indent=1)
print(json.dumps(S, indent=1))

# ------------------------------------------------------------- SHAP summary figure (outer fold 1), sized for 16-cm text width
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 10, 'axes.labelsize': 10.5, 'xtick.labelsize': 9.5,
                     'ytick.labelsize': 9.5, 'axes.linewidth': 0.8, 'savefig.dpi': 1000, 'savefig.bbox': 'tight'})
tops = list(cons.sort_values(ascending=False).index[:10])[::-1]          # bottom-to-top
L0 = L[0]
def lab(c):
    f = L0.loc[c].sort_values(ascending=False).index[0]
    return f'PC{c} · {f}'
fig = plt.figure(figsize=(6.3, 8.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.025], height_ratios=[1.35, 1], hspace=0.32, wspace=0.03)
ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1]); bx = fig.add_subplot(gs[1, 0])
rng = np.random.default_rng(0); cmap = plt.get_cmap('coolwarm'); FLOOR = 1e-3
# one point per explained window: normalised SHAP averaged over the ten detectors (consensus attribution)
W_ = f0.groupby(['draw', 'inst', 'y', 'pc']).agg(nshap=('nshap', 'mean'), val=('val', 'first')).reset_index()
XMIN, XMAX = FLOOR * 0.8, 3.0
AXW_PT = 6.3 * 72 * 0.80; ROW_PT = 8.2 * 72 * 0.55 / len(tops); D = 4.6     # axis width/row height/marker diameter in points
def swarm(xpt):
    placed = []; ys = np.zeros(len(xpt)); lim = ROW_PT * 0.46
    for i in np.argsort(xpt):
        for k in range(0, 60):
            for cand in ([0.0] if k == 0 else [k * D * 0.5, -k * D * 0.5]):
                if abs(cand) > lim: continue
                if all((xpt[i] - px)**2 + (cand - py)**2 >= D**2 for px, py in placed if abs(xpt[i] - px) < D):
                    ys[i] = cand; placed.append((xpt[i], cand)); break
            else: continue
            break
        else:
            ys[i] = rng.uniform(-lim, lim); placed.append((xpt[i], ys[i]))
    return ys / ROW_PT
for i, c in enumerate(tops):
    q = W_[W_.pc == c]; x = np.clip(q.nshap.values, FLOOR, None); v = q.val.values
    lo, hi = np.percentile(v, [5, 95]); col = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
    xpt = (np.log10(x) - np.log10(XMIN)) / (np.log10(XMAX) - np.log10(XMIN)) * AXW_PT
    y = i + swarm(xpt)
    ax.scatter(x, y, c=cmap(col), s=13, lw=0.25, edgecolors='white', alpha=0.95, zorder=3)
ax.set_xscale('log'); ax.set_xlim(XMIN, XMAX); ax.set_ylim(-0.6, len(tops) - 0.4)
ax.set_xticks([1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(['≤0.001', '0.01', '0.1', '1'])
ax.set_yticks(range(len(tops))); ax.set_yticklabels([lab(c) for c in tops])
ax.set_xlabel('Normalised SHAP value per window, mean of ten detectors (log scale)')
ax.set_title('(a) Contribution of each component to the anomaly score', fontsize=10.5, loc='left', fontweight='bold')
sm = plt.cm.ScalarMappable(cmap=cmap); cb = fig.colorbar(sm, cax=cax, ticks=[0, 1])
cb.ax.set_yticklabels(['Low', 'High'], fontsize=9.5); cb.set_label('Component score of the window', fontsize=10); cb.outline.set_linewidth(0.6)
# (b) mean |SHAP| healthy vs fault, with ratio annotation
m = f0[f0.pc.isin(tops)].groupby(['pc', 'y'])['nabs'].mean().unstack().reindex(tops)
yy = np.arange(len(tops)); h = 0.38
bx.barh(yy - h/2, m[0], h, color='#8c96a3', label='Healthy windows (held-out)')
bx.barh(yy + h/2, m[1], h, color='#c0392b', label='Faulty windows')
for i_, c in enumerate(tops):
    r = m.loc[c, 1] / max(m.loc[c, 0], 1e-12)
    bx.text(m.loc[c, 1] + 0.002, i_ + h/2, f'×{r:.0f}' if r >= 10 else f'×{r:.1f}', va='center', fontsize=9, color='#7a1f16')
bx.set_yticks(yy); bx.set_yticklabels([lab(c) for c in tops]); bx.set_ylim(-0.6, len(tops) - 0.4)
bx.set_xlim(0, m[1].max() * 1.18)
bx.set_xlabel('Mean |normalised SHAP|')
bx.set_title('(b) Healthy versus faulty windows (ratio annotated)', fontsize=10.5, loc='left', fontweight='bold')
bx.legend(fontsize=9.5, frameon=False, loc='lower right')
for a_ in (ax, bx):
    a_.spines[['top', 'right']].set_visible(False); a_.grid(axis='x', color='#e3e3e3', lw=0.7); a_.set_axisbelow(True)
    a_.tick_params(axis='y', length=0)
fig.savefig('figs/Fig07b_shap_beeswarm.png', dpi=1000); plt.close(fig)
