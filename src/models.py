"""Eleven healthy-only anomaly detectors on a shared standardise -> PCA representation.

All hyperparameters are fixed before evaluation (Table 3 of the paper); the K-Means k is
selected by the elbow method on healthy training data (k = 5). Scores are oriented so
that larger = more anomalous.
Environment options for the sensitivity analyses: PCA_VAR (retained variance, default
0.90), PCA_CAP (maximum number of components, default 40) and NO_PCA=1 (standardised
features without PCA).
"""
import numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.covariance import LedoitWolf, MinCovDet
from scipy.spatial.distance import pdist
torch.set_num_threads(1)

MODELS = ['IF','OCSVM','AE','VAE','LSTMAE','LOF','KMeans','DBSCAN','Z','MD','EE']
NAMES = {'IF':'Isolation Forest','OCSVM':'One-Class SVM','AE':'Autoencoder','VAE':'VAE',
         'LSTMAE':'LSTM Autoencoder','LOF':'LOF','KMeans':'K-Means','DBSCAN':'DBSCAN',
         'Z':'Z-Score','MD':'Mahalanobis Distance','EE':'Elliptic Envelope'}
STOCHASTIC = {'IF','KMeans','EE','AE','VAE','LSTMAE'}
SEQ_T = 3
NN_CFG = dict(lr=2e-3, batch=128, max_epochs=150, patience=15, val_frac=0.1, weight_decay=0.0)

# ----------------------------------------------------------------------------- representation
class Rep:
    """Standardise (training statistics), drop zero-variance columns, PCA to 90 % (cap 40)."""
    def fit(self, X):
        import os
        self.keep = X.std(0) > 1e-12
        self.sc = StandardScaler().fit(X[:, self.keep])
        Z = self.sc.transform(X[:, self.keep])
        self.nopca = os.environ.get('NO_PCA') == '1'
        if self.nopca:
            self.d = Z.shape[1]; return self
        var = float(os.environ.get('PCA_VAR', '0.90')); cap = int(os.environ.get('PCA_CAP', '40'))
        full = PCA(svd_solver='full').fit(Z)
        self.d = int(min(cap, np.searchsorted(np.cumsum(full.explained_variance_ratio_), var) + 1))
        self.pca = PCA(n_components=self.d, svd_solver='full').fit(Z)
        return self
    def transform(self, X):
        Z = self.sc.transform(X[:, self.keep])
        return Z if self.nopca else self.pca.transform(Z)

def nn_sizes(d):
    h = max(4, int(np.ceil(d / 2)))
    z = max(2, int(np.ceil(d / 4)))
    return h, z

# ----------------------------------------------------------------------------- neural models
class _AE(nn.Module):
    def __init__(s, d, h, z, var=False):
        super().__init__(); s.var = var
        s.enc = nn.Sequential(nn.Linear(d, h), nn.Tanh())
        s.mu = nn.Linear(h, z); s.lv = nn.Linear(h, z) if var else None
        s.dec = nn.Sequential(nn.Linear(z, h), nn.Tanh(), nn.Linear(h, d))
    def forward(s, x, sample=True):
        e = s.enc(x); mu = s.mu(e)
        if not s.var: return s.dec(torch.tanh(mu)), None, None
        lv = s.lv(e); zz = mu + torch.exp(0.5*lv)*torch.randn_like(mu) if sample else mu
        return s.dec(zz), mu, lv

class _LSTMAE(nn.Module):
    def __init__(s, d, h, z):
        super().__init__()
        s.enc = nn.LSTM(d, h, batch_first=True); s.to_z = nn.Linear(h, z)
        s.dec = nn.LSTM(z, h, batch_first=True); s.out = nn.Linear(h, d)
    def forward(s, x):
        _, (hn, _) = s.enc(x); z = torch.tanh(s.to_z(hn[-1]))
        y, _ = s.dec(z.unsqueeze(1).repeat(1, x.shape[1], 1)); return s.out(y), None, None

def _train(net, X, seed, beta=0.0):
    g = torch.Generator().manual_seed(seed)
    n = len(X); idx = torch.randperm(n, generator=g); nv = max(1, int(NN_CFG['val_frac']*n))
    Xv, Xt = X[idx[:nv]], X[idx[nv:]]
    opt = torch.optim.Adam(net.parameters(), lr=NN_CFG['lr'], weight_decay=NN_CFG['weight_decay'])
    best, best_state, bad, ep_run = np.inf, None, 0, 0
    for ep in range(NN_CFG['max_epochs']):
        net.train(); perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), NN_CFG['batch']):
            xb = Xt[perm[i:i+NN_CFG['batch']]]
            r, mu, lv = net(xb); loss = ((r-xb)**2).mean()
            if beta and mu is not None: loss = loss + beta*(-0.5*torch.mean(1+lv-mu**2-lv.exp()))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            r, mu, lv = net(Xv); vl = ((r-Xv)**2).mean().item()
            if beta and mu is not None: vl += beta*(-0.5*torch.mean(1+lv-mu**2-lv.exp())).item()
        ep_run = ep+1
        if vl < best - 1e-7: best, bad, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= NN_CFG['patience']: break
    net.load_state_dict(best_state); net.eval(); net.epochs_run = ep_run
    return net

def make_sequences(Z, rec, win):
    """Length-T sequences of consecutive windows from the same recording; score -> last window."""
    order = np.lexsort((win, rec)); Z, rec, win = Z[order], rec[order], win[order]
    seqs, tgt = [], []
    for i in range(SEQ_T-1, len(Z)):
        if rec[i-SEQ_T+1] == rec[i] and win[i]-win[i-SEQ_T+1] == SEQ_T-1:
            seqs.append(Z[i-SEQ_T+1:i+1]); tgt.append(order[i])
    return np.array(seqs, dtype=np.float32), np.array(tgt)

# ----------------------------------------------------------------------------- detector wrapper
class Detector:
    def __init__(self, name, seed=42): self.name, self.seed = name, seed
    def fit(self, Z, rec=None, win=None):
        n, s, d = self.name, self.seed, Z.shape[1]
        if n == 'IF': self.m = IsolationForest(n_estimators=300, max_samples='auto', random_state=s).fit(Z)
        elif n == 'OCSVM': self.m = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale').fit(Z)
        elif n == 'LOF': self.m = LocalOutlierFactor(n_neighbors=min(10, len(Z)-1), novelty=True).fit(Z)
        elif n == 'KMeans': self.m = KMeans(n_clusters=5, n_init=10, random_state=s).fit(Z)
        elif n == 'DBSCAN':
            eps = 0.75*np.median(pdist(Z)); db = DBSCAN(eps=eps, min_samples=3).fit(Z)
            core = Z[db.core_sample_indices_] if len(db.core_sample_indices_) else Z
            self.core_frac = len(db.core_sample_indices_)/len(Z)
            self.m = NearestNeighbors(n_neighbors=1).fit(core)
        elif n == 'Z': self.mu, self.sd = Z.mean(0), Z.std(0, ddof=1)
        elif n == 'MD':
            lw = LedoitWolf().fit(Z); self.mu, self.P = lw.location_, lw.precision_
        elif n == 'EE':
            try:
                mcd = MinCovDet(support_fraction=0.95, random_state=s).fit(Z)
                self.mu, self.P = mcd.location_, np.linalg.pinv(mcd.covariance_); self.ee_fallback = False
            except Exception:
                lw = LedoitWolf().fit(Z); self.mu, self.P = lw.location_, lw.precision_; self.ee_fallback = True
        elif n in ('AE', 'VAE'):
            torch.manual_seed(s); h, z = nn_sizes(d)
            self.net = _train(_AE(d, h, z, var=(n == 'VAE')), torch.tensor(Z, dtype=torch.float32), s,
                              beta=1e-3 if n == 'VAE' else 0.0)
        elif n == 'LSTMAE':
            torch.manual_seed(s); h, z = nn_sizes(d)
            S, _ = make_sequences(Z, rec, win)
            self.net = _train(_LSTMAE(d, h, z), torch.tensor(S), s)
        return self
    def score(self, Z, rec=None, win=None):
        n = self.name
        if n == 'IF': return -self.m.score_samples(Z)
        if n == 'OCSVM': return -self.m.decision_function(Z)
        if n == 'LOF': return -self.m.score_samples(Z)
        if n == 'KMeans': return self.m.transform(Z).min(1)
        if n == 'DBSCAN': return self.m.kneighbors(Z)[0][:, 0]
        if n == 'Z': return np.abs((Z-self.mu)/self.sd).max(1)
        if n in ('MD', 'EE'):
            D = Z-self.mu; return np.einsum('ij,jk,ik->i', D, self.P, D)
        with torch.no_grad():
            if n == 'AE':
                X = torch.tensor(Z, dtype=torch.float32); r, _, _ = self.net(X); return ((r-X)**2).mean(1).numpy()
            if n == 'VAE':
                X = torch.tensor(Z, dtype=torch.float32); g = torch.Generator().manual_seed(self.seed+1)
                acc = 0
                for _ in range(8):
                    e = self.net.enc(X); mu = self.net.mu(e); lv = self.net.lv(e)
                    zz = mu + torch.exp(0.5*lv)*torch.randn(mu.shape, generator=g)
                    acc = acc + ((self.net.dec(zz)-X)**2).mean(1)
                return (acc/8).numpy()
            if n == 'LSTMAE':
                S, tgt = make_sequences(Z, rec, win); out = np.full(len(Z), np.nan)
                X = torch.tensor(S); r, _, _ = self.net(X)
                out[tgt] = ((r-X)**2).mean((1, 2)).numpy(); return out

# ----------------------------------------------------------------------------- thresholds
def thresholds(s):
    from scipy.stats import genpareto
    s = s[~np.isnan(s)]
    med = np.median(s); mad = 1.4826*np.median(np.abs(s-med))
    out = {'p90': np.percentile(s, 90), 'mean+2sd': s.mean()+2*s.std(ddof=1), 'p95': np.percentile(s, 95),
           'mean+3sd': s.mean()+3*s.std(ddof=1), 'p97.5': np.percentile(s, 97.5), 'p99': np.percentile(s, 99),
           'median+3MAD': med+3*mad, 'p99.5': np.percentile(s, 99.5), 'p99.9': np.percentile(s, 99.9),
           'max': s.max()}
    u = np.percentile(s, 90); exc = s[s > u]-u
    try:
        c, loc, sc = genpareto.fit(exc, floc=0)
        out['GPD'] = u + genpareto.ppf(1-0.001/0.10, c, loc=0, scale=sc)   # P(S>t)=0.001
    except Exception:
        out['GPD'] = np.percentile(s, 99.9)
    return out
THR_ORDER = ['p90','mean+2sd','p95','mean+3sd','p97.5','p99','median+3MAD','p99.5','p99.9','max','GPD']
