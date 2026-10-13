"""Selection of k for K-Means by the elbow method on healthy training windows only.

For each outer fold of the nested design (and for the final fit on all 15 healthy recordings),
the scaler and PCA are fitted on the training recordings, K-Means (10 initialisations) is fitted
for k = 1..KMAX and the within-cluster sum of squares is recorded. The elbow is the point of the
min-max normalised curve farthest from the straight line joining its end points.
Usage: python kmeans_k_selection.py [KMAX=20]
"""
import sys, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from extract import FEATURES
from models import Rep

KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20

def elbow(y):
    yn = (y - y.min()) / (y.max() - y.min()); xn = np.arange(len(y)) / (len(y) - 1)
    return int(np.argmax(np.abs(yn - (1 - xn))) + 1)

f = pd.read_parquet('data/features.parquet')
H = f[f.rec.str.startswith('Valve_P1_')]
base = sorted(H.rec.unique(), key=lambda r: int(r.split('_R')[1]))
rows = []
for ro in base + ['(final fit)']:
    tr = H[H.rec != ro][FEATURES].values
    Z = Rep().fit(tr).transform(tr)
    y = np.array([KMeans(k, n_init=10, random_state=42).fit(Z).inertia_ for k in range(1, KMAX + 1)])
    rows.append(dict(held_out=ro, elbow_k=elbow(y), **{f'wcss_k{k}': v for k, v in zip(range(1, KMAX + 1), y)}))
    print(ro, 'elbow k =', rows[-1]['elbow_k'], flush=True)
pd.DataFrame(rows).to_csv('tables/kmeans_elbow.csv', index=False)
