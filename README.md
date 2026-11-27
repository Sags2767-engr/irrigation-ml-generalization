# Cross-Dataset Generalization of ML Models for IoT-Based Irrigation Prediction

Reproducibility code for evaluating cross-dataset generalization using the Arnesano precision-irrigation dataset, International Soil Moisture Network (ISMN) observations, and NASA POWER daily weather data.

## Scope and interpretation

The primary outcome is a threshold-derived irrigation-need proxy based on the future 24-hour soil-moisture condition. It is not a record of an actual irrigation decision or actuator event. Results should therefore be interpreted as prediction of the defined proxy, not as evidence of water savings or improved farm operations.

This repository currently contains the corrected cross-dataset validation script only. The separate Experiment 1 script (`irrigation_generalization.py`) and final feature-diagnostics script referenced in the dissertation are not included in this repository version.

## Included file

- `cross_dataset_validation.py` — corrected cross-dataset evaluation for P15 and P25 threshold configurations, as identified in the dissertation's reproducibility appendix.

## Data requirements

The datasets are not redistributed here. Obtain each dataset from its original provider and comply with its terms of use. Arrange the locally downloaded data in the formats expected by the script. The Arnesano processed-data directory, ISMN station files, and NASA POWER daily point-data files are passed as separate command-line arguments.

Do not commit raw sensor, ISMN, or NASA POWER data to this repository unless redistribution rights have been independently confirmed.

## Environment

Python 3.12 is the intended runtime. The reported cross-dataset results use XGBoost; install it explicitly. If XGBoost is unavailable, the script falls back to HistGradientBoosting, which is a different model and will not reproduce the reported XGBoost results. Install the required packages:

```bash
python -m pip install numpy pandas scikit-learn scipy matplotlib xgboost
```

## Run

```bash
python cross_dataset_validation.py \
  --arnesano /path/to/02_processed_data \
  --ismn /path/to/ISMN_data \
  --power /path/to/POWER_daily_data \
  --out results_cross
```

Use `--only p15` or `--only p25` to run one threshold configuration. Without `--only`, the script runs both configurations.

## Important reproducibility note

The script aligns weather by using the previous completed calendar day's daily record for each soil-moisture observation: a row dated D uses weather from D-1. Its three-day rainfall feature sums weather over D-3 through D-1. This is intentional to avoid using a same-day daily aggregate that may include conditions occurring after the observation time. Do not change this alignment without re-running the experiment and updating the manuscript. The script also falls back to HistGradientBoosting when XGBoost is unavailable; that fallback is not equivalent to the XGBoost configuration used for the reported XGBoost results. For reproduction, ensure XGBoost is installed and record the exact Python and package versions, data snapshot, script revision, and command-line arguments used.

The corrected script reports descriptive results. Do not infer statistical significance from descriptive means alone.

## Data sources

- Arnesano precision-irrigation dataset, Mendeley Data, version 2: https://doi.org/10.17632/c837v6p8ph.2
- International Soil Moisture Network: https://ismn.earth/
- NASA Prediction of Worldwide Energy Resources (POWER): https://power.larc.nasa.gov/

## License

See `LICENSE`. The repository license applies to original code in this repository only; it does not grant rights to third-party datasets or external materials.
