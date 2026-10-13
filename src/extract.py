"""Feature extraction for the EGB-250 healthy-only study.

Sampling rates: 51,200 Hz (accelerometers, microphones, current clamps), 25,000 Hz
(pressure transducers) and 500,000 Hz (shaft encoder). Each recording is cut by duration
into 48 time-aligned, non-overlapping 0.2-s windows (first 9.6 s). Each window is
DC-removed (window mean subtracted) and described by 12 descriptors per channel
(13 channels x 12 = 156 features). The encoder is used only to measure shaft speed.

Set the environment variable EGB250_RAW to the folder that holds the GIDTEC raw files.
"""
import numpy as np, scipy.io as sio, pandas as pd, re, glob, os, sys
from scipy.signal import welch, hilbert
from scipy.stats import skew, kurtosis
from multiprocessing import Pool

FS50, FS25, FS500 = 51200.0, 25000.0, 500000.0
WIN_S, NWIN = 0.2, 48
N50, N25, N500 = int(FS50*WIN_S), int(FS25*WIN_S), int(FS500*WIN_S)   # 10240, 5000, 100000
NPS50, NPS25 = 2048, 1000                                              # 25 Hz Welch resolution
ENV_BAND = 2000.0
ROOT = os.environ.get('EGB250_RAW', 'RawData')

CHANNELS = [('A1','50','TDMSAnalog50kAcelerometer0'),('A2','50','TDMSAnalog50kAcelerometer1'),
            ('A3','50','TDMSAnalog50kAcelerometer2'),('A4','50','TDMSAnalog50kAcelerometer3'),
            ('M1','50','TDMSAnalog50kMic0'),('M2','50','TDMSAnalog50kMic1'),
            ('I1','50','TDMSAnalog50kCurrentClamp0'),('I2','50','TDMSAnalog50kCurrentClamp1'),
            ('I3','50','TDMSAnalog50kCurrentClamp2'),
            ('PR1','25','TDMSAnalog25kPressure0'),('PR2','25','TDMSAnalog25kPressure1'),
            ('PR3','25','TDMSAnalog25kPressure2'),('PR4','25','TDMSAnalog25kPressure3')]
DESC = ['std','skew','kurt','p2p','crest','shape','impulse',
        'spec_centroid','spec_kurt','env_peak_amp','env_peak_freq','env_rms']
FEATURES = [f'{c}_{d}' for c,_,_ in CHANNELS for d in DESC]

def descriptors(x, fs, nps):
    x = x - x.mean()
    sd = x.std(ddof=1); rms = np.sqrt(np.mean(x**2)); ab = np.abs(x); mab = ab.mean(); pk = ab.max()
    f, P = welch(x, fs=fs, nperseg=nps)
    p = P/ P.sum()
    cen = float((f*p).sum())
    sk = kurtosis(P, fisher=False)
    e = np.abs(hilbert(x))
    fe, Pe = welch(e, fs=fs, nperseg=len(e))       # single Hann segment (5 Hz resolution); detrend='constant' removes envelope DC
    m = (fe > 0) & (fe <= ENV_BAND)
    i = np.argmax(Pe[m])
    ec = e - e.mean()
    return [sd, skew(x), kurtosis(x, fisher=False), x.max()-x.min(), pk/rms, rms/mab, pk/mab,
            cen, sk, np.sqrt(Pe[m][i]), fe[m][i], np.sqrt(np.mean(ec**2))]

def peak_cps(x, lo, hi):
    x = x - x.mean(); n = len(x); N = 1 << int(np.ceil(np.log2(n)) + 2)
    X = np.abs(np.fft.rfft(x*np.hanning(n), N)); f = np.fft.rfftfreq(N)
    m = (f > lo) & (f < hi); return f[m][np.argmax(X[m])]

def process(path):
    rel = os.path.relpath(path, ROOT)
    sub = rel.split(os.sep)[0].replace('Raw_Data_','').replace('_Faults_DB_011V0','')
    P = rel.split(os.sep)[1]; R = re.match(r'(R\d+)P', os.path.basename(path)).group(1)
    M = sio.loadmat(path)
    d = M['data'][0,0]
    g = {'50': d['Analog50k'][0,0], '25': d['Analog25k'][0,0]}
    sig = {c: g[m][k].ravel().astype(float) for c,m,k in CHANNELS}
    enc = d['Analog500k'][0,0]['TDMSAnalog500kEncoder0'].ravel()
    edges = np.flatnonzero(np.diff((enc > 0).astype(np.int8)) == 1)
    per = np.diff(edges)
    # ---- sampling-rate consistency checks (A1) -------------------------------------
    enc_cps = 1.0/np.median(per)                                   # shaft cycles per encoder sample
    pr_cps = peak_cps(sig['PR3'], 0.8*10/FS25, 1.2*16/FS25)        # shaft cycles per pressure sample
    # 51.2k module: shaft-locked periodicity of A2 envelope (2nd-stage discharge area)
    a_cps = peak_cps(np.abs(sig['A2'][:200000]), 0.8*10/FS50, 1.2*16/FS50)
    i_cps = peak_cps(sig['I1'], 40/FS50, 80/FS50)                  # supply fundamental, cycles/sample
    meta = dict(subset=sub, P=P, R=R, rec=f'{sub}_{P}_{R}',
                n50=len(sig['A1']), n25=len(sig['PR1']), n500=len(enc),
                shaft_hz=FS500*enc_cps, edges_total=len(edges),
                ratio_enc_over_pr=pr_cps/enc_cps, ratio_enc_over_50=a_cps and (enc_cps and None),
                fs25_implied=FS500*enc_cps/pr_cps, fs50_implied=FS500*enc_cps/a_cps,
                supply_hz_at_51k2=i_cps*FS50, supply_hz_at_50k=i_cps*50000.0)
    rows = []
    for w in range(NWIN):
        r = dict(rec=meta['rec'], win=w)
        for c, m, _ in CHANNELS:
            n, fs, nps = (N50, FS50, NPS50) if m == '50' else (N25, FS25, NPS25)
            x = sig[c][w*n:(w+1)*n]
            for dn, v in zip(DESC, descriptors(x, fs, nps)):
                r[f'{c}_{dn}'] = v
        e = edges[(edges >= w*N500) & (edges < (w+1)*N500)]
        r['n_edges'] = len(e)                     # kept as metadata only (not a model input)
        r['win_shaft_hz'] = FS500/np.mean(np.diff(e)) if len(e) > 1 else np.nan
        rows.append(r)
    return meta, rows

if __name__ == '__main__':
    files = sorted(glob.glob(f'{ROOT}/Raw_Data_Valve_Faults_DB_011V0/**/*.mat', recursive=True))
    with Pool(2) as pool:
        out = pool.map(process, files)
    meta = pd.DataFrame([m for m,_ in out]).drop(columns=['ratio_enc_over_50'])
    feats = pd.DataFrame([r for _,rows in out for r in rows])
    os.makedirs('data', exist_ok=True)
    meta.to_csv('data/recordings.csv', index=False)
    feats.to_parquet('data/features.parquet')
    print(meta.describe().T.to_string()); print(feats.shape)
