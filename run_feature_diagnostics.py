"""Additional feature diagnostics for the corrected cross-dataset irrigation pipeline.

This script imports cross_dataset_validation.py without modifying it, then computes:
1) Feature ablation (soil-moisture-only, weather-only, combined features).
2) Permutation importance (Macro-F1 decrease after feature shuffling).

It does not overwrite the cross-dataset results_p15.csv / results_p25.csv files.
It writes diagnostic outputs to the selected output directory.

Example (Windows PowerShell):
python run_feature_diagnostics.py --pipeline "cross_dataset_validation.py" --arnesano "PATH_TO_02_processed_data" --ismn "PATH_TO_ISMN_FOLDER" --power "PATH_TO_POWER_FOLDER" --out "feature_diagnostics"
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

FEATURE_GROUPS = {
    "Soil_moisture_only": ["sm_margin_z"],
    "Weather_only": ["temp", "rh", "rain", "rain3"],
    "Combined": ["sm_margin_z", "temp", "rh", "rain", "rain3"],
}
SEEDS = [0, 1, 2]


def load_pipeline(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("corrected_irrigation_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_scenarios(arn, ism):
    """Return the five evaluation scenarios described in the publication."""
    hot_cut = ism.groupby("site")["temp"].transform(lambda x: x.quantile(2 / 3))
    return [
        ("Arnesano_temporal", arn[arn.index < "2025-08-15"],
         arn[arn.index >= "2025-08-16"]),
        ("ISMN_hot_day_regime", ism[ism["temp"] <= hot_cut],
         ism[ism["temp"] > hot_cut]),
        ("ISMN_temporal", ism[ism.index < "2022-12-25"],
         ism[ism.index >= "2023-01-01"]),
        ("Arnesano_to_ISMN", arn, ism),
        ("ISMN_to_Arnesano", ism, arn),
    ]


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="cross_dataset_validation.py",
                    help="Path to cross_dataset_validation.py")
    ap.add_argument("--arnesano", required=True)
    ap.add_argument("--ismn", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--out", default="feature_diagnostics")
    ap.add_argument("--permutation-repeats", type=int, default=30)
    args = ap.parse_args()

    pipe = load_pipeline(args.pipeline)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ablation_rows = []
    permutation_rows = []

    for tag, percentile in [("p15", 15), ("p25", 25)]:
        print(f"\\nLoading data for {tag}...")
        arn = pipe.load_arnesano(args.arnesano)
        ism = pipe.load_ismn(args.ismn, args.power, percentile)
        scenarios = build_scenarios(arn, ism)

        for scenario_name, train, test in scenarios:
            if len(train) < 80 or len(test) < 30 or train["y"].nunique() < 2:
                print(f"Skipping {tag}/{scenario_name}: insufficient rows or one training class")
                continue
            if test["y"].nunique() < 2:
                print(f"Skipping {tag}/{scenario_name}: test set has only one class")
                continue

            for seed in SEEDS:
                for group_name, features in FEATURE_GROUPS.items():
                    for model_name, (estimator, grid) in pipe.make_models(seed).items():
                        fitted = pipe.fit_tuned(estimator, grid, train[features], train["y"])
                        if fitted is None:
                            continue
                        pred = fitted.predict(test[features])
                        score = get_score(fitted, test[features])
                        metric = pipe.metrics(test["y"].to_numpy(), pred, score)
                        ablation_rows.append({
                            "threshold": tag, "scenario": scenario_name, "seed": seed,
                            "feature_group": group_name, "features": ";".join(features),
                            "model": model_name, "n_train": len(train), "n_test": len(test),
                            **metric,
                        })

                for model_name, (estimator, grid) in pipe.make_models(seed).items():
                    fitted = pipe.fit_tuned(estimator, grid, train[pipe.FEATS], train["y"])
                    if fitted is None:
                        continue
                    base_pred = fitted.predict(test[pipe.FEATS])
                    base_macro_f1 = pipe.f1_score(
                        test["y"].to_numpy(), base_pred, average="macro"
                    )
                    pi = permutation_importance(
                        fitted, test[pipe.FEATS], test["y"], scoring="f1_macro",
                        n_repeats=args.permutation_repeats, random_state=seed, n_jobs=1,
                    )
                    for feature, mean_drop, std_drop in zip(
                        pipe.FEATS, pi.importances_mean, pi.importances_std
                    ):
                        permutation_rows.append({
                            "threshold": tag, "scenario": scenario_name, "seed": seed,
                            "model": model_name, "feature": feature,
                            "baseline_macro_f1": float(base_macro_f1),
                            "permutation_macro_f1_decrease": float(mean_drop),
                            "permutation_std": float(std_drop), "n_test": len(test),
                            "repeats": args.permutation_repeats,
                        })
            print(f"Completed {tag}/{scenario_name}")

    ablation = pd.DataFrame(ablation_rows)
    permutation = pd.DataFrame(permutation_rows)
    if ablation.empty or permutation.empty:
        raise RuntimeError(
            "No diagnostic rows were produced. Check data paths, class counts, "
            "and whether the expected datasets were loaded."
        )

    ablation.to_csv(out / "ablation_results.csv", index=False)
    permutation.to_csv(out / "permutation_importance.csv", index=False)
    ablation.groupby(
        ["threshold", "scenario", "feature_group", "model"], as_index=False
    )[["macro_f1", "balanced_acc", "roc_auc", "precision_positive",
       "recall_positive", "f1_positive"]].mean().round(4).to_csv(
           out / "ablation_summary.csv", index=False
       )
    permutation.groupby(
        ["threshold", "scenario", "model", "feature"], as_index=False
    )[["permutation_macro_f1_decrease", "permutation_std"]].mean().round(4).to_csv(
        out / "permutation_importance_summary.csv", index=False
    )

    print("\\nDiagnostics completed. Outputs saved in:", out.resolve())
    print("Permutation importance uses ordinary row-wise shuffling. Because observations")
    print("are temporally dependent, interpret it as descriptive feature sensitivity, not causal importance.")


if __name__ == "__main__":
    main()
