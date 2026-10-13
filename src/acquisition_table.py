"""Per-recording acquisition record (Supplementary Table S16).

Mean tank (PR4) and stage (PR1-PR3) pressures over the analysed 9.6 s of each recording,
with the encoder speed taken from the dataset metadata.
Usage: python acquisition_table.py <master_metadata.csv> <out.csv>
"""
import os, sys, numpy as np, scipy.io as sio, pandas as pd
ROOT = os.environ.get('EGB250_RAW', 'RawData')
N25 = 240000  # 9.6 s at 25 kS/s
def means(path):
    g = sio.loadmat(path)['data'][0, 0]['Analog25k'][0, 0]
    return [g[f'TDMSAnalog25kPressure{i}'].ravel().astype(float)[:N25].mean() for i in range(4)]
M = pd.read_csv(sys.argv[1])
sel = ([('A - Valve faults', 'Valve', 'P1', f'R{i}') for i in range(1, 16)]
       + [('A - Valve faults', 'Valve', f'P{p}', 'R1') for p in range(2, 26)])
rows = []
for ds, sub, P, R in sel:
    m = M[(M.Dataset == ds) & (M.P_ID == P) & (M.Repetition_ID == R)].iloc[0]
    pr = means(os.path.join(ROOT, f'Raw_Data_{sub}_Faults_DB_011V0', P, f'{R}{P}.mat'))
    rows.append(dict(Set=f'{sub}/{P}', Rec=R, PR1=pr[0], PR2=pr[1], PR3=pr[2], PR4=pr[3],
                     Speed=m.Measured_shaft_speed_Hz))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
