"""Per-recording acquisition record (Supplementary Table S16).

Mean tank (PR4) and stage (PR1-PR3) pressures over the analysed 9.6 s of each recording used,
and the encoder-measured crankshaft speed (from data/recordings.csv, written by extract.py).
Usage: python acquisition_table.py out.csv
"""
import os, sys, scipy.io as sio, pandas as pd
ROOT = os.environ.get('EGB250_RAW', 'RawData')
N25 = 240000  # 9.6 s at 25 kS/s
def means(path):
    g = sio.loadmat(path)['data'][0, 0]['Analog25k'][0, 0]
    return [g[f'TDMSAnalog25kPressure{i}'].ravel().astype(float)[:N25].mean() for i in range(4)]
spd = pd.read_csv('data/recordings.csv').set_index('rec').shaft_hz
sel = [('P1', f'R{i}') for i in range(1, 16)] + [(f'P{p}', 'R1') for p in range(2, 26)]
rows = []
for P, R in sel:
    pr = means(os.path.join(ROOT, 'Raw_Data_Valve_Faults_DB_011V0', P, f'{R}{P}.mat'))
    rows.append(dict(Condition=f'Valve/{P}', Recording=R, PR4=pr[3], PR1=pr[0], PR2=pr[1], PR3=pr[2],
                     Speed_Hz=spd[f'Valve_{P}_{R}']))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
