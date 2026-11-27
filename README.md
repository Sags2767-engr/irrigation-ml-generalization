# Cross-Dataset Generalization of ML Models for IoT-Based Irrigation Prediction

Reproducibility code for evaluating cross-dataset generalization using the Arnesano precision-irrigation dataset, International Soil Moisture Network (ISMN) observations, and NASA POWER daily weather data.

## Scope and interpretation

The primary outcome is a threshold-derived irrigation-need proxy based on the future 24-hour soil-moisture condition. It is not a record of an actual irrigation decision or actuator event. Results should therefore be interpreted as prediction of the defined proxy, not as evidence of water savings or improved farm operations.

## Included files

- `cross_dataset_validation.py` — corrected cross-dataset evaluation for P15 and P25 threshold configurations.
- `run_feature_diagnostics.py` — feature ablation and permutation-importance analyses using the corrected cross-dataset pipeline.

The separate three-class experiment script `irrigation_generalization_experiment.py` is not part of this cross-dataset publication workflow and is not included here.

## Data requirements

The datasets are not redistributed here. Obtain each dataset from its original provider and comply with its terms of use. Arrange the locally downloaded data in the formats expected by the scripts. The Arnesano processed-data directory, ISMN station files, and NASA POWER daily point-data files are passed as separate command-line arguments.

Do not commit raw sensor, ISMN, or NASA POWER data to this repository unless redistribution rights have been independently confirmed.

## Environment

Python 3.12 is the intended runtime. The reported cross-dataset results use XGBoost; install it explicitly. If XGBoost is unavailable, the validation script falls back to HistGradientBoosting, which is a different model and will not reproduce the reported XGBoost results. Install the required packages:

```bash
python -m pip install numpy pandas scikit-learn scipy matplotlib xgboost
```

## Run the cross-dataset validation

```bash
python cross_dataset_validation.py \
  --arnesano /path/to/02_processed_data \
  --ismn /path/to/ISMN_data \
  --power /path/to/POWER_daily_data \
  --out results_cross
```

Use `--only p15` or `--only p25` to run one threshold configuration. Without `--only`, the script runs both configurations.

## Run feature diagnostics

```bash
python run_feature_diagnostics.py \
  --pipeline cross_dataset_validation.py \
  --arnesano /path/to/02_processed_data \
  --ismn /path/to/ISMN_data \
  --power /path/to/POWER_daily_data \
  --out feature_diagnostics
```

The diagnostics script writes ablation and permutation-importance result files to its output directory. It does not overwrite the cross-dataset validation result files.

## Important reproducibility notes

The validation script aligns weather by using the previous completed calendar day's daily record for each soil-moisture observation: a row dated D uses weather from D-1. Its three-day rainfall feature sums weather over D-3 through D-1. This is intentional to avoid using a same-day daily aggregate that may include conditions occurring after the observation time. Do not change this alignment without re-running the experiment and updating the publication.

For reproduction of the reported XGBoost results, ensure XGBoost is installed. Record the exact Python and package versions, data snapshot, script revision, and command-line arguments used.

The diagnostics script uses ordinary row-wise permutation. Because observations are temporally dependent, permutation importance should be interpreted as descriptive feature sensitivity, not causal importance or a time-series-preserving intervention. The scripts report descriptive results; do not infer statistical significance from descriptive means alone.

## Data sources

- Arnesano precision-irrigation dataset, Mendeley Data, version 2: https://doi.org/10.17632/c837v6p8ph.2
- International Soil Moisture Network: https://ismn.earth/
- NASA Prediction of Worldwide Energy Resources (POWER): https://power.larc.nasa.gov/

## License

See `LICENSE`. The repository license applies to original code in this repository only; it does not grant rights to third-party datasets or external materials.
