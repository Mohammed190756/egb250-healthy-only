"""Manuscript figures (PNG, 1000 dpi): Figs. 1-4, 9, 10, 11 and 13.
Usage: python figs.py pipeline variance rationale design sensor_groups embedding permutation shap
"""
import numpy as np, pandas as pd, matplotlib, json, sys, glob
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from extract import FEATURES
from models import Rep, MODELS, NAMES

BLUE, ORANGE, AQUA, YELLOW, MAG, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Liberation Sans', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'savefig.dpi': 1000,
                     'savefig.bbox': 'tight', 'legend.frameon': False})
OUT = 'figs'
import os; os.makedirs(OUT, exist_ok=True); os.makedirs('tables', exist_ok=True)
F = pd.read_parquet('data/features.parquet')

def box(ax, x, y, w, h, txt, fc='#eef4fc', ec=BLUE, fs=8, bold=False, pad=0.01):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad},rounding_size=0.008', fc=fc, ec=ec, lw=1))
    ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=fs, color=INK, wrap=True,
            fontweight='bold' if bold else 'normal')
def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1))

def fig_pipeline():
    from matplotlib.patches import Circle
    with plt.rc_context({'font.family': 'Liberation Serif'}):
        fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.set_xlim(0, 100); ax.set_ylim(5, 66); ax.axis('off')
        def bx(x, y, w, h, t):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc='white', ec='black', lw=0.8))
            ax.text(x + w/2, y + h/2, t, ha='center', va='center', fontsize=7, color='black', linespacing=1.25)
        def arr(x0, y0, x1, y1):
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='black', lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))
        def stage(n, y, title):
            ax.add_patch(Circle((1.9, y), 1.7, fc='white', ec='black', lw=0.8))
            ax.text(1.9, y, str(n), ha='center', va='center', fontsize=9)
            ax.text(4.8, y, title, ha='left', va='center', fontsize=10.5)
        H = 11; rows = [(48.5, 'Data preparation', ['Raw signals\n13 channels +\nshaft encoder,\n10 s per recording',
                                                    'Windowing\n48 × 0.2-s windows,\nnon-overlapping,\ntime-aligned',
                                                    'Feature extraction\n12 descriptors ×\n13 channels\n= 156 features',
                                                    'Standardisation\nfitted on healthy\ntraining windows',
                                                    'PCA\n39 components,\n90% cumulative\nvariance']),
                        (28.5, 'Model fitting and threshold calibration', ['Eleven healthy-only\ndetectors from\nfive families',
                                                    'Inner LORO on the\n14 training recordings\n→ pooled out-of-fold\nhealthy scores',
                                                    'Threshold\n99th percentile\n(ten alternatives\nevaluated)',
                                                    'Outer refit on the\n14 recordings → score\nheld-out healthy and\nfault windows']),
                        (8.5, 'Evaluation, stability and explainability', ['Detection metrics\ndetection rate,\nspecificity, FAR,\nROC-AUC, PR-AUC',
                                                    'Sensor-group analysis\nper sensor type and\nmicrophone-free\nconfiguration',
                                                    'Statistical stability\nbootstrap CIs, six seeds,\nFriedman–Nemenyi,\nthreshold sensitivity',
                                                    'Explainability\npermutation importance,\nPCA loadings,\nKernel SHAP'])]
        for k, (y, title, items) in enumerate(rows):
            stage(k + 1, y + H + 3.6, title)
            n = len(items); gap = 2.6; x0 = 5.6; w = (95.5 - x0 - gap*(n - 1)) / n
            xs = [x0 + i*(w + gap) for i in range(n)]
            for i, (x, t) in enumerate(zip(xs, items)):
                bx(x, y, w, H, t)
                if i < n - 1: arr(x + w, y + H/2, x + w + gap, y + H/2)
            if k < 2:
                xe = xs[-1] + w/2; ny = rows[k + 1][0] + H; xn = 4.2 + 3.4
                ax.plot([xe, xe, xn - 3.4], [y, y - 2.4, y - 2.4], color='black', lw=0.8)
                ax.plot([xn - 3.4, xn - 3.4], [y - 2.4, ny - H/2], color='black', lw=0.8)
                arr(xn - 3.4, ny - H/2, 5.6, ny - H/2)
        fig.savefig(f'{OUT}/Fig01_pipeline.png'); plt.close(fig)

def fig_rationale():
    with plt.rc_context({'font.family': 'Liberation Serif'}):
        fig, ax = plt.subplots(figsize=(7.2, 4.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
        def bx(x, y, w, h, t, it=False, fs=8):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc='white', ec='black', lw=0.8))
            ax.text(x + w/2, y + h/2, t, ha='center', va='center', fontsize=fs, style='italic' if it else 'normal', linespacing=1.2)
        def arr(x0, y0, x1, y1):
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='black', lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))
        top = [('Sensor data\n(§3.1)', 2), ('Feature extraction\n(§3.1)', 27), ('Standardisation +\nPCA (§3.2)', 52), ('Eleven unsupervised\ndetectors, five\nalgorithmic families', 76)]
        for t, x in top: bx(x, 87, 22 if x < 76 else 23, 10, t)
        for x in (24, 49, 74): arr(x, 92, x + 3, 92)
        fam = [('Tree-/isolation-\nbased', 'Random partitioning;\nno distance metric', ['Isolation Forest']),
               ('Kernel-/margin-\nbased', 'Non-linear boundary\naround healthy data', ['One-Class SVM']),
               ('Reconstruction-\nbased (deep)', 'Reproduces healthy\npatterns; flags high\nreconstruction error', ['Autoencoder', 'Variational\nAutoencoder', 'LSTM\nAutoencoder']),
               ('Distance-/density-\nbased', 'Local density or\ncluster distance', ['Local Outlier\nFactor', 'K-Means', 'DBSCAN']),
               ('Statistical', 'Gaussian or robust-\ncovariance assumption', ['Z-Score', 'Mahalanobis\nDistance', 'Elliptic\nEnvelope'])]
        w = 18.4; xs = [1 + i*(w + 1.4) for i in range(5)]
        ax.plot([xs[0] + w/2, xs[-1] + w/2], [82, 82], color='black', lw=0.8); ax.plot([87.5, 87.5], [87, 82], color='black', lw=0.8)
        for (t, d, ms), x in zip(fam, xs):
            arr(x + w/2, 82, x + w/2, 79); bx(x, 71, w, 8, t)
            arr(x + w/2, 71, x + w/2, 68); bx(x, 57, w, 11, d, it=True, fs=7.5)
            y = 57
            for m in ms:
                arr(x + w/2, y, x + w/2, y - 3); bx(x, y - 11, w, 8, m); y -= 11
            ax.plot([x + w/2, x + w/2], [y, 14], color='black', lw=0.8)
        ax.plot([xs[0] + w/2, xs[-1] + w/2], [14, 14], color='black', lw=0.8)
        arr(30, 14, 30, 11); bx(14, 1, 32, 10, 'Shared nested LORO\nthreshold calibration (§3.4)')
        arr(46, 6, 52, 6); bx(52, 1, 34, 10, 'Side-by-side anomaly-detection\nbenchmark, identical input (§4)')
        fig.savefig(f'{OUT}/Fig03_rationale.png'); plt.close(fig)

def fig_design():
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis('off')
    n = 15; w = 0.05
    for i in range(n):
        x = 0.05 + i*(w+0.012)
        c = '#fdeee7' if i == 4 else '#eef4fc'; e = ORANGE if i == 4 else BLUE
        box(ax, x, 0.66, w, 0.14, f'R{i+1}', fc=c, ec=e, fs=7, pad=0.002)
    ax.text(0.05, 0.86, 'Healthy baseline Valve/P1 (15 recordings, 48 windows each)', fontsize=8.5, color=INK)
    ax.text(0.05, 0.57, 'Outer fold o: held-out recording (orange) is never used for fitting or calibration', fontsize=8, color=INK2)
    box(ax, 0.03, 0.25, 0.29, 0.22, 'Inner LORO over the 14\ntraining recordings → pooled\nout-of-fold scores → p99', fc='#fdeee7', ec=ORANGE, fs=7.5)
    box(ax, 0.355, 0.25, 0.29, 0.22, 'Fit scaler, PCA and detector\non the 14 recordings\n(one model per fold)', fc='#eef4fc', ec=BLUE, fs=7.5)
    box(ax, 0.68, 0.25, 0.30, 0.22, 'Same model scores the\nheld-out healthy recording\nand all fault windows', fc='#e6f6ef', ec=AQUA, fs=7.5)
    arrow(ax, 0.32, 0.36, 0.355, 0.36); arrow(ax, 0.645, 0.36, 0.68, 0.36)
    ax.text(0.05, 0.08, 'Repeated for o = 1, …, 15 and for six seeds.', fontsize=8, color=INK2)
    fig.savefig(f'{OUT}/Fig04_validation_design.png'); plt.close(fig)

def fig_variance():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    out = {}
    for b, c in [('Valve', BLUE)]:
        tr = F.rec.str.startswith(f'{b}_P1_'); rep = Rep().fit(F.loc[tr, FEATURES].values)
        from sklearn.decomposition import PCA
        Z = rep.sc.transform(F.loc[tr, FEATURES].values[:, rep.keep]); ev = PCA(svd_solver='full').fit(Z).explained_variance_ratio_
        cum = np.cumsum(ev)[:60]*100
        if b == 'Valve':
            ax.plot(np.arange(1, 61), cum, color=c, lw=2, label='Valve/P1 (720 windows)'); ax.scatter([rep.d], [cum[rep.d-1]], color=c, s=30, zorder=3)
        out[b] = dict(d=rep.d, cum=float(cum[rep.d-1]), ev=(ev[:40]*100).round(2).tolist(), kept=int(rep.keep.sum()),
                      dropped=list(np.array(FEATURES)[~rep.keep]))
    ax.axhline(90, color=INK2, lw=1, ls='--')
    ax.annotate(f"{out['Valve']['d']} components, {out['Valve']['cum']:.1f}%", (out['Valve']['d'], out['Valve']['cum']), xytext=(out['Valve']['d']+4, out['Valve']['cum']-14), fontsize=7, color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    ax.set_xlabel('Retained principal components'); ax.set_ylabel('Cumulative explained variance (%)'); ax.legend(fontsize=7, loc='lower right')
    fig.savefig(f'{OUT}/Fig02_variance.png'); plt.close(fig)
    json.dump(out, open('tables/pca_final.json', 'w'), indent=1)

def fig_embedding():
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Fv = F[F.rec.str.startswith('Valve_')].reset_index(drop=True)
    X = StandardScaler().fit_transform(Fv[FEATURES].values)
    P1 = Fv.rec.str.contains('_P1_').values
    grp = np.where(P1, 'Healthy (Valve/P1)', 'Valve-fault recordings')
    order = [('Valve-fault recordings', '#f6c9b5'), ('Healthy (Valve/P1)', BLUE)]
    E1 = PCA(2).fit_transform(X); E2 = TSNE(2, perplexity=40, random_state=42, init='pca').fit_transform(X)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, E, t in [(axs[0], E1, '(a) PCA'), (axs[1], E2, '(b) t-SNE')]:
        for g, c in order:
            k = grp == g; ax.scatter(E[k, 0], E[k, 1], s=4 if 'fault' in g else 7, color=c, label=g, alpha=0.9, lw=0)
        ax.set_title(t, fontsize=9, loc='left'); ax.set_xticks([]); ax.set_yticks([])
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.legend(h, l, fontsize=7, markerscale=2.5, loc='lower center', ncol=2)
    fig.savefig(f'{OUT}/Fig10_embedding.png'); plt.close(fig)

def group_table():
    """Nested-design metrics per sensor group (8 non-neural detectors, seed 42)."""
    from analyse import REC, HEALTHY, CMASK, COND, P99
    from sklearn.metrics import roc_auc_score
    rows = []
    files = {'all': glob.glob('runs/Valve_s42_*.npy')[0]}
    for f in glob.glob('runs_groups/Valve_*_s42.npy'):
        files[os.path.basename(f).split('_')[1]] = f
    fm = np.zeros(len(REC), bool)
    for c in COND: fm |= CMASK[c]
    for g, f in files.items():
        D = np.load(f, allow_pickle=True).item()
        for m in ['OCSVM', 'LOF', 'DBSCAN', 'Z', 'MD', 'IF', 'KMeans', 'EE']:
            v = D[m]; r = dict(group=g, model=m, det=[], spec=[], auc=[])
            for o, ro in enumerate(HEALTHY['Valve']):
                s = v['outer_scores'][o]; t = v['outer_thr'][o, P99]
                h = s[REC == ro]; ff = s[fm]
                r['det'].append((ff > t).mean()); r['spec'].append((h <= t).mean())
                r['auc'].append(roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(ff))], np.r_[h, ff]))
            rows.append({k: (np.mean(v_) if isinstance(v_, list) else v_) for k, v_ in r.items()})
    df = pd.DataFrame(rows); df.to_csv('tables/T_groups_long.csv', index=False); return df

def fig_sensor_groups():
    df = group_table()
    order = ['all', 'nomic', 'accel', 'pressure', 'current', 'mic']
    lab = {'all': 'All\nchannels', 'nomic': 'All but\nmicrophones', 'accel': 'Accel-\nerometers', 'pressure': 'Pressure', 'current': 'Current\nclamps', 'mic': 'Micro-\nphones'}
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.arange(len(order)); w = 0.36
    a = df.groupby('group')[['auc', 'det']].mean().reindex(order)
    b1 = ax.bar(x - w/2 - 0.01, 100*a['det'], w, color='#b8d3f2', label='Detection (%)')
    b2 = ax.bar(x + w/2 + 0.01, 100*a['auc'], w, color=BLUE, label='ROC-AUC (× 100)')
    for bars, vals in [(b1, 100*a['det']), (b2, 100*a['auc'])]:
        for bb, v in zip(bars, vals): ax.text(bb.get_x()+bb.get_width()/2, v+1.2, f'{v:.1f}', ha='center', fontsize=5.5, color=INK2)
    ax.set_ylim(0, 115); ax.set_ylabel('Value')
    ax.set_xticks(x); ax.set_xticklabels([lab[o] for o in order], fontsize=7); ax.grid(axis='x', visible=False)
    ax.legend(fontsize=6.5, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig09_sensor_groups.png'); plt.close(fig)

def perm_tables():
    out = {}
    for b in ['Valve']:
        fs = glob.glob(f'runs/perm_{b}_*.parquet')
        if not fs: continue
        p = pd.concat([pd.read_parquet(f) for f in fs])
        # mean over models within (fold, repeat), then mean/sd over (fold, repeat)
        g = p.groupby(['subset', 'feature', 'fold', 'rep']).d_auc.mean().reset_index()
        s = g.groupby(['subset', 'feature']).d_auc.agg(['mean', 'std']).reset_index()
        out[b] = s
        s.to_csv(f'tables/perm_{b}.csv', index=False)
    return out

def fig_permutation():
    out = perm_tables(); s = out['Valve']
    top = s[s.subset == 'all'].sort_values('mean', ascending=False).head(15).feature.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(top))[::-1]; h = 0.27
    for k, (sub, c, lab) in enumerate([('all', BLUE, 'All 24 valve conditions'), ('2S_DV', ORANGE, '12 conditions with a 2S_DV fault'),
                                       ('non-2S_DV', AQUA, '12 conditions without a 2S_DV fault')]):
        v = s[s.subset == sub].set_index('feature').reindex(top)
        ax.barh(y + (1-k)*h, v['mean'], h*0.92, color=c, label=lab)
    ax.set_yticks(y); ax.set_yticklabels(top, fontsize=7); ax.set_xlabel('Mean drop in ROC-AUC when permuted')
    ax.legend(fontsize=7, loc='lower right'); ax.grid(axis='y', visible=False)
    fig.savefig(f'{OUT}/Fig11_permutation.png'); plt.close(fig)

def shap_summary():
    fs = glob.glob('runs/shap_Valve_*.parquet'); p = pd.concat([pd.read_parquet(f) for f in fs])
    p['abs'] = p.shap.abs()
    # put detectors on a common scale: share of each component in the detector's total mean |SHAP| (per fold, draw)
    tot = p.groupby(['fold', 'model', 'draw', 'pc'])['abs'].mean().groupby(['fold', 'model', 'draw']).transform('sum')
    p = p.merge(tot.rename('tot').reset_index()[['fold', 'model', 'draw', 'pc', 'tot']], on=['fold', 'model', 'draw', 'pc'])
    p['abs'] = p['abs'] / p['tot']
    comp = p[p.fold == 0].groupby('pc')['abs'].mean()
    # stability: Spearman correlation of component ranking between draws (within fold & model, averaged)
    from scipy.stats import spearmanr
    rs = []
    for (f, m), q in p.groupby(['fold', 'model']):
        t = q.groupby(['draw', 'pc'])['abs'].mean().unstack(0)
        for a in range(3):
            for b in range(a+1, 3): rs.append(spearmanr(t[a], t[b]).correlation)
    # pooled across models: rank stability of the detector-averaged ranking
    t = p.groupby(['draw', 'pc'])['abs'].mean().unstack(0)
    pooled = [spearmanr(t[a], t[b]).correlation for a in range(3) for b in range(a+1, 3)]
    # heuristic projection to raw features using each fold's loadings
    proj = []
    for f in sorted(set(p.fold)):
        L = pd.read_csv(f'runs/loadings_Valve_f{f}.csv'); L = L.set_index('pc').drop(columns='fold')
        ci = p[p.fold == f].groupby('pc')['abs'].mean()
        proj.append((L.loc[ci.index].mul(ci, axis=0)).sum(axis=0))
    proj = pd.concat(proj, axis=1).mean(axis=1).sort_values(ascending=False)
    return comp, float(np.nanmean(rs)), float(np.nanmin(rs)), float(np.mean(pooled)), proj

def fig_shap():
    comp, rmean, rmin, pooled, proj = shap_summary()
    tr = F.rec.str.startswith('Valve_P1_') & (F.rec != 'Valve_P1_R1')   # outer fold 1: trained on R2-R15
    pca = (100*Rep().fit(F.loc[tr, FEATURES].values).pca.explained_variance_ratio_).tolist()   # same fit as the SHAP values
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x = comp.index.values
    ax.bar(x, comp.values, color=BLUE, width=0.7, label='share')
    ax.set_xlabel('Principal component (outer fold 1)'); ax.set_ylabel('Share of mean |SHAP|')
    ax2 = ax.twinx(); ax2.plot(np.arange(1, len(pca)+1), pca, color=ORANGE, lw=1.6, label='explained variance (%)')
    ax2.set_ylabel('Explained variance (%)', color=ORANGE); ax2.grid(False); ax2.spines['right'].set_visible(True)
    ax.grid(axis='x', visible=False)
    ax.set_xlim(0.3, len(pca) + 0.7); ax.set_ylim(0, comp.values.max()*1.12); ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 39])
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, ['Share of mean |SHAP|', 'Explained variance (%)'], fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.savefig(f'{OUT}/Fig13_shap.png'); plt.close(fig)
    json.dump(dict(rank_rho_mean=rmean, rank_rho_min=rmin, rank_rho_pooled=pooled,
                   top_pcs=[int(i) for i in comp.sort_values(ascending=False).index[:5]],
                   top_pcs_val=[float(v) for v in comp.sort_values(ascending=False).values[:5]],
                   proj_top=proj.head(10).round(4).to_dict()), open('tables/shap.json', 'w'), indent=1)

if __name__ == '__main__':
    for name in sys.argv[1:]:
        globals()[f'fig_{name}'](); print('done', name)
