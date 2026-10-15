"""PCA tables: final fit on all 720 healthy training windows (Table 8; Supplementary Tables S2 and S11)
and the outer-fold-1 fit on recordings R2-R15 (Supplementary Table S12).
Loadings are the PCA component coefficients of the standardised features (sign arbitrary).
"""
import numpy as np, pandas as pd, os
from extract import FEATURES
from models import Rep

f = pd.read_parquet('data/features.parquet'); os.makedirs('tables', exist_ok=True)
H = f[f.rec.str.startswith('Valve_P1_')]

def fit(rows, tag):
    rep = Rep().fit(rows[FEATURES].values)
    names = [c for c, k in zip(FEATURES, rep.keep) if k]
    pcs = [f'PC{i+1}' for i in range(rep.d)]
    L = pd.DataFrame(rep.pca.components_.T, index=names, columns=pcs)
    v = 100*rep.pca.explained_variance_ratio_
    V = pd.DataFrame({'PC': pcs, 'Variance (%)': v, 'Cumulative (%)': np.cumsum(v)})
    L.round(4).to_csv(f'tables/pca_{tag}_loadings.csv', index_label='Feature')
    V.round(4).to_csv(f'tables/pca_{tag}_variance.csv', index=False)
    top = []
    for pc, var in zip(pcs, v):
        s = L[pc].reindex(L[pc].abs().sort_values(ascending=False).index)
        for r, (feat, val) in enumerate(s.head(10).items(), 1):
            top.append(dict(Component=pc, Variance=round(var, 2), Rank=r, Feature=feat, Loading=round(val, 3)))
    pd.DataFrame(top).to_csv(f'tables/pca_{tag}_top10.csv', index=False)
    print(tag, rep.d, 'components,', len(names), 'features, cumulative', round(v.sum(), 2))

fit(H, 'final')
fit(H[H.rec != 'Valve_P1_R1'], 'fold1')
