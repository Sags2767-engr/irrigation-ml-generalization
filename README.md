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

Python 3.12 is the intended runtime. Install the required packages:

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

The exact script revision, data snapshot, package versions, and command-line arguments must match the analysis that generated the manuscript's reported results. The script's weather-date alignment is part of the analysis definition and must not be changed without re-running the experiment and updating the manuscript. The dissertation identifies this file as the corrected cross-dataset script; verify that the run used to generate the final manuscript tables was generated from this same script revision.

The corrected script reports descriptive results. Do not infer statistical significance from descriptive means alone.

## Data sources

- Arnesano precision-irrigation dataset, Mendeley Data, version 2: https://doi.org/10.17632/c837v6p8ph.2
- International Soil Moisture Network: https://ismn.earth/
- NASA Prediction of Worldwide Energy Resources (POWER): https://power.larc.nasa.gov/

## License

See `LICENSE`. The repository license applies to original code in this repository only; it does not grant rights to third-party datasets or external materials.
