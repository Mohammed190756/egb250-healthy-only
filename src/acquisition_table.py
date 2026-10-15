"""Per-recording acquisition record (Supplementary Table S16).

For every recording used: mean tank (PR4) and stage (PR1-PR3) pressures over the analysed 9.6 s,
the encoder-measured crankshaft speed (from data/recordings.csv, written by extract.py) and the
crankshaft speed estimated from the second-stage discharge pressure (PR3) spectrum. The pressure
estimate maximises the summed spectral magnitude of the first 20 harmonics over a 0.0005-Hz grid.
In valve conditions P19-P25 the encoder reads low because of a coupling slip; the pressure
spectrum gives the shaft speed there.
Usage: python acquisition_table.py out.csv
"""
import os, sys, numpy as np, scipy.io as sio, pandas as pd
ROOT = os.environ.get('EGB250_RAW', 'RawData')
FS25, N25, NFFT = 25000.0, 240000, 1 << 22   # 9.6 s at 25 kS/s for the means

def pressure_speed(x):
    x = (x - x.mean()) * np.hanning(len(x))
    X = np.abs(np.fft.rfft(x, NFFT)); fr = np.fft.rfftfreq(NFFT, 1/FS25)
    grid = np.arange(12.5, 13.3, 0.0005)
    return grid[np.argmax(sum(np.interp(h*grid, fr, X) for h in range(1, 21)))]

def record(path):
    g = sio.loadmat(path)['data'][0, 0]['Analog25k'][0, 0]
    p = [g[f'TDMSAnalog25kPressure{i}'].ravel().astype(float) for i in range(4)]
    return [x[:N25].mean() for x in p], pressure_speed(p[2])

spd = pd.read_csv('data/recordings.csv').set_index('rec').shaft_hz
sel = ([('Valve', 'P1', f'R{i}') for i in range(1, 16)] + [('Valve', f'P{p}', 'R1') for p in range(2, 26)]
       + [('Bearing', 'P1', f'R{i}') for i in range(1, 16)])
rows = []
for S, P, R in sel:
    pr, fp = record(os.path.join(ROOT, f'Raw_Data_{S}_Faults_DB_011V0', P, f'{R}{P}.mat'))
    enc = spd[f'{S}_{P}_{R}']
    rows.append(dict(Condition=f'{S}/{P}', Recording=R, PR4=pr[3], PR1=pr[0], PR2=pr[1], PR3=pr[2],
                     Speed_encoder_Hz=enc, Speed_pressure_Hz=fp, Encoder_biased=bool(enc < 0.995*fp)))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
