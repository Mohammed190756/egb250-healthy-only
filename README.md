# Healthy-only anomaly detection on the EGB-250 reciprocating compressor

Code for the paper *"A Healthy-Only Anomaly Detection Framework for Reciprocating Compressor Condition Monitoring: Nested Validation, Sensor-Group Analysis, and Explainability"* (M. J. Ghulam Mohiuddin, A. Deshmukh).

The code reproduces the numbers, tables and figures of the paper from the raw recordings of the valve-fault sub-dataset.

## Data
The EGB-250 dataset is the property of GIDTEC, Universidad Politécnica Salesiana (Cuenca, Ecuador). It is **not included** in this repository. Researchers who wish to use it should contact the authors of the dataset reference cited in the paper.

Place the raw `.mat` files in a folder and set `EGB250_RAW` to its path. The code reads the sub-folder `Raw_Data_Valve_Faults_DB_011V0`.

## Environment
- Python 3.13.
- Install the pinned packages with `pip install -r requirements.txt`.
- Set `OMP_NUM_THREADS=1` when running several processes in parallel.

## Pipeline (run from `src/`, in order)
| Step | Script | Output |
|---|---|---|
| 1 | `extract.py` | `data/features.parquet` (156 features per 0.2-s window), `data/recordings.csv` |
| 2 | `run_nested.py Valve <seed> <models> [group]` | nested LORO scores and thresholds in `runs/` (all channels) and `runs_groups/` (sensor groups) |
| 3 | `explain.py Valve 0,3,6,9,12 perm`, `explain.py Valve 0,5,10 shap`, `shap_analysis.py` | permutation importance and Kernel SHAP on held-out data |
| 4 | `kmeans_k_selection.py 20` | elbow-based selection of k for K-Means (healthy data only) |
| 5 | `make_tables.py` | primary metrics, bootstrap, seed stability, per-condition results, Friedman/Nemenyi, single-level protocol, threshold sensitivity (`tables/`) |
| 6 | `figs.py pipeline variance rationale design sensor_groups embedding permutation shap` | Figs. 1–4, 9, 10, 11 and 13 |
| 7 | `acquisition_table.py master_metadata.csv out.csv` | per-recording mean tank and stage pressures and shaft speed (Supplementary Table S16) |

### Notes on step 2
- **Seeds:** the paper uses seeds 0–4 and 42 for the stochastic detectors (Isolation Forest, K-Means, Elliptic Envelope and the three neural networks), and seed 42 for the deterministic ones.
- **Sensor groups:** `accel`, `mic`, `current`, `pressure` and `nomic` (all channels except the microphones). The single-group analysis uses the eight non-neural detectors at seed 42.
- **Microphone-free configuration:** run step 2 with group `nomic` for all eleven detectors, with the same seeds as above.
- **PCA-retention sensitivity:** run step 2 with `PCA_VAR=0.95 PCA_CAP=1000` or `PCA_VAR=0.99 PCA_CAP=1000`, or with `NO_PCA=1` for Isolation Forest, One-Class SVM and LOF. Set `OUTDIR` to keep these runs apart.

### Remaining figures
Figs. 5–8, 12, 14 and 15 are plots of the tables written in steps 3 and 5. Their values are listed in the supplementary tables of the paper.

## Detectors
- **Detectors** (`models.py`), all trained on healthy data only: Isolation Forest, One-Class SVM, Autoencoder, VAE, LSTM Autoencoder, LOF, K-Means (k = 5), DBSCAN, Z-Score, Mahalanobis distance and Elliptic Envelope.
- **Primary threshold:** the 99th percentile of the pooled inner out-of-fold healthy scores.
- **Sensitivity analysis:** ten further threshold candidates are evaluated.

## Licence
MIT (see `LICENSE`). If you use this code, please cite the paper (see `CITATION.cff`).
