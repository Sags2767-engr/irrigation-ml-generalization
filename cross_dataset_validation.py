"""
Cross-dataset validation of irrigation-need models.

Datasets
  A) Arnesano (Italy) precision-irrigation data, daily means (Feb-Sep 2025), 4 sectors.
  B) ISMN soil moisture (REMEDHUS Spain + SCAN USA) at ~5 cm with NASA POWER daily weather,
     5 stations, one row per day (12:00 UTC), 2021-01-20 to 2024-11-20.
  Weather in both datasets is the daily value for the calendar day of the row.

Target  need = 1 if the mean soil moisture over the next 24 h is below the site threshold
  Arnesano : the farm's own start threshold (40, 40, 80, 50 for sectors 1, 2, 4, 5)
  ISMN     : the P-th percentile of the station's daily soil moisture (default P = 15)

Features (shared by both datasets, same units)
  sm_margin_z  (soil moisture today - threshold) / station standard deviation
  temp         air temperature at 2 m, daily mean, deg C
  rh           relative humidity at 2 m, daily mean, %
  rain         daily rain, mm
  rain3        rain over today and the previous 2 days, mm
Wind and radiation are not used: Arnesano and POWER differ in units and measurement height.

Run
    python cross_dataset_validation.py --arnesano <02_processed_data> --ismn <ISMN unzip folder> \
        --power <folder with POWER csv files> --out results_cross
"""
import argparse
import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
warnings.filterwarnings("ignore")

FEATS = ["sm_margin_z", "temp", "rh", "rain", "rain3"]
ARN_TH = {1: 40, 2: 40, 4: 80, 5: 50}          # sector 3 excluded (probe failed, ~no positives)
ISMN_STATIONS = {                                # (network, station name in the ISMN file header)
    "Concejo_del_Monte": ("REMEDHUS", "Concejo_del_Monte"),
    "Walnut_Gulch_1": ("SCAN", "Walnut_Gulch_#1"),
    "Levelland": ("SCAN", "Levelland"),
    "Tuskegee": ("SCAN", "Tuskegee"),
    "Cochora_Ranch": ("SCAN", "Cochora_Ranch"),
}
SEEDS = [0, 1, 2]


# ------------------------------------------------------------------ loaders
def daily_weather(w):
    """w: daily frame (date index) with temp, rh, rain. Adds rain3 (today + previous 2 days)."""
    w = w.asfreq("D")
    w["rain3"] = w["rain"].rolling(3, min_periods=3).sum()
    return w


def finish(d, th, sd, label_series):
    """Common last step: margin feature, label, drop incomplete rows."""
    d = d.copy()
    d["sm_margin_z"] = (d["sm"] - th) / sd
    d["y"] = np.where(label_series.notna(), (label_series < th).astype(float), np.nan)
    d = d.dropna(subset=FEATS + ["y"])
    d["y"] = d["y"].astype(int)
    return d


def load_arnesano(root):
    """Hourly rows (HH:00) with the dataset's own 24 h-ahead mean moisture; daily weather from the merged layer."""
    root = Path(root)
    out = []
    for z, th in ARN_TH.items():
        p = pd.read_csv(root / "preprocessed" / f"dataset_zone_{z}_preprocessed.csv", parse_dates=["ts"]).set_index("ts").sort_index()
        sd = p["soil_moisture"].std()
        p = p[p.index.minute == 0].dropna(subset=["soil_moisture", "target_mean_24h"])
        m = pd.read_csv(root / "merged" / f"dataset_zone_{z}.csv", parse_dates=["ts"]).set_index("ts").sort_index()
        # weather is hourly (ERA5) repeated on the 10-minute grid: collapse to hours before summing rain
        h = m[["weather_temp", "weather_humidity", "weather_rain"]].resample("1h").mean()
        g = h.groupby(h.index.floor("D"))
        w = pd.DataFrame({"temp": g["weather_temp"].mean(), "rh": g["weather_humidity"].mean(),
                          "rain": g["weather_rain"].sum(), "n": g["weather_temp"].count()})
        w = daily_weather(w[w["n"] >= 20].drop(columns="n"))
        
        d = pd.DataFrame({"sm": p["soil_moisture"]})
        d = d.join(w, on=(d.index.floor("D") - pd.Timedelta(days=1)))
        d = finish(d, th, sd, p["target_mean_24h"].reindex(d.index))
        d["dataset"], d["site"] = "Arnesano", f"sector{z}"
        d.index.name = "date"
        out.append(d)
    return pd.concat(out)


def read_stm(path):
    d = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, usecols=[0, 1, 2, 3],
                    names=["d", "t", "v", "q"], dtype={"d": str, "t": str, "q": str}, encoding="latin-1")
    d["ts"] = pd.to_datetime(d["d"] + " " + d["t"], format="%Y/%m/%d %H:%M", errors="coerce")
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d[(d["q"] == "G") & d["v"].notna()].set_index("ts")["v"]


def find_station_files(ismn_root, network, station):
    sm_files, ts_files = [], []
    # ISMN exports may be arranged as network/station/*.stm or as a flat
    # station_files/*.stm directory. Search recursively to support both layouts.
    for f in glob.glob(os.path.join(ismn_root, "**", "*.stm"), recursive=True):
        with open(f, encoding="latin-1") as fh:
            h = fh.readline().split()
        if len(h) < 9 or h[2] != station or float(h[7]) > 0.101:
            continue
        (sm_files if "_sm_" in os.path.basename(f) else ts_files).append((f, float(h[3]), float(h[4])))
    return sm_files, ts_files


def load_power(power_dir):
    out = []
    for f in glob.glob(os.path.join(power_dir, "POWER_Point_Daily_*.csv")):
        lines = open(f, encoding="latin-1").read().splitlines()
        end = [i for i, l in enumerate(lines) if "END HEADER" in l][0]
        loc = [l for l in lines if l.startswith("Location")][0].split()
        lat, lon = float(loc[2]), float(loc[4])
        d = pd.read_csv(f, skiprows=end + 1)
        d["date"] = pd.to_datetime(d["YEAR"].astype(str) + d["DOY"].astype(str).str.zfill(3), format="%Y%j")
        d = d.set_index("date").rename(columns={"T2M": "temp", "RH2M": "rh", "PRECTOTCORR": "rain"})
        weather = daily_weather(d[["temp", "rh", "rain"]].where(lambda x: x > -90))
        out.append((lat, lon, weather))
    return out


def load_ismn(ismn_root, power_dir, pct):
    """One row per day (12:00 UTC). Target: mean moisture over the next 24 h below the station's pct-th percentile."""
    power = load_power(power_dir)
    frames = []
    for name, (net, st) in ISMN_STATIONS.items():
        sm_files, ts_files = find_station_files(ismn_root, net, st)
        if not sm_files:
            print("WARNING: no sm file for", name)
            continue
        best = max(sm_files, key=lambda x: len(read_stm(x[0])))
        lat, lon = best[1], best[2]
        s = read_stm(best[0]).resample("1h").mean()
        th, sd = s.quantile(pct / 100), s.std()
        fut = s.rolling(24, min_periods=12).mean().shift(-24)
        d = pd.DataFrame({"sm": s}).dropna()
        d = d[d.index.hour == 12]
        if ts_files:                                   # drop frozen days (daily mean soil temperature <= 1 C)
            t = read_stm(ts_files[0][0])
            tm = t.groupby(t.index.floor("D")).mean()
            d = d[~d.index.floor("D").isin(tm[tm <= 1].index)]
        wlat, wlon, w = min(power, key=lambda p: (p[0] - lat) ** 2 + (p[1] - lon) ** 2)
        if (wlat - lat) ** 2 + (wlon - lon) ** 2 > 0.05:
            print("WARNING: no POWER file near", name)
            continue
        d = d.join(w, on=(d.index.floor("D") - pd.Timedelta(days=1)))
        d = finish(d, th, sd, fut.reindex(d.index))
        d["dataset"], d["site"] = "ISMN", name
        d.index.name = "date"
        frames.append(d)
    return pd.concat(frames)


# ------------------------------------------------------------------ models
def make_models(seed):
    m = {
        "LogReg": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                   {"logisticregression__C": [0.01, 0.1, 1, 10]}),
        "RandomForest": (RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1),
                         {"max_depth": [5, 10, None]}),
        "SVM": (make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", gamma="scale")),
                {"svc__C": [1, 10]}),
    }
    if HAVE_XGB:
        m["XGBoost"] = (XGBClassifier(n_estimators=300, eval_metric="logloss", random_state=seed, n_jobs=-1),
                        {"learning_rate": [0.05, 0.1], "max_depth": [3, 6]})
    else:
        m["GradBoost(HGB)"] = (HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
                               {"learning_rate": [0.05, 0.1], "max_depth": [3, 6]})
    return m


def fit_tuned(est, grid, X, y):
    order = np.argsort(X.index.values, kind="stable")
    X, y = X.iloc[order], y.iloc[order]
    if y.nunique() < 2:
        return None
    gs = GridSearchCV(est, grid, cv=TimeSeriesSplit(3), scoring="f1_macro", n_jobs=1)
    try:
        gs.fit(X, y)
        return gs.best_estimator_
    except ValueError:
        est.fit(X, y)
        return est


def metrics(y, pred, score):
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "precision_positive": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall_positive": recall_score(y, pred, pos_label=1, zero_division=0),
        "f1_positive": f1_score(y, pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
    }

def block_ci(te, pred, n_boot=300, block_days=14, seed=0):
    """95% CI for macro-F1 by resampling 14-day blocks within each site."""
    rng = np.random.default_rng(seed)
    key = te["site"].astype(str) + "_" + (te.index.map(pd.Timestamp.toordinal) // block_days).astype(str)
    codes, uniq = pd.factorize(key)
    groups = [np.where(codes == k)[0] for k in range(len(uniq))]
    y = te["y"].values
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        vals.append(f1_score(y[idx], pred[idx], average="macro"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(level, split, tr, te, seed, rows):
    if len(tr) < 80 or len(te) < 30:
        return
    meta = dict(level=level, split=split, seed=seed, n_train=len(tr), n_test=len(te),
                pos_train=round(tr["y"].mean(), 3), pos_test=round(te["y"].mean(), 3))
    # baseline: persistence (is moisture already below the threshold today?)
    pers = (te["sm_margin_z"] < 0).astype(int).values
    r = metrics(te["y"].values, pers, -te["sm_margin_z"].values)
    r.update(meta, model="Persistence")
    if seed == 0:
        r["f1_lo"], r["f1_hi"] = block_ci(te, pers)
    rows.append(r)
    for name, (est, grid) in make_models(seed).items():
        m = fit_tuned(est, grid, tr[FEATS], tr["y"])
        if m is None:
            continue
        pred = m.predict(te[FEATS])
        score = m.predict_proba(te[FEATS])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(te[FEATS])
        r = metrics(te["y"].values, pred, score)
        r.update(meta, model=name)
        if seed == 0:
            r["f1_lo"], r["f1_hi"] = block_ci(te, pred)
        rows.append(r)


# ------------------------------------------------------------------ experiments
def run_config(arn, ism, tag, out, quick):
    rows = []
    for label, d in [("Arnesano", arn), ("ISMN", ism)]:
        if not quick:
            for seed in SEEDS:
                tri, tei = train_test_split(np.arange(len(d)), test_size=0.2, stratify=d["y"], random_state=seed)
                evaluate(f"{label}_L1_random", "random_80_20", d.iloc[tri], d.iloc[tei], seed, rows)
            for k, (tri, tei) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(d, d["y"])):
                evaluate(f"{label}_L2_cv", f"fold{k}", d.iloc[tri], d.iloc[tei], 0, rows)
        for s in sorted(d["site"].unique()):
            for seed in SEEDS[:2]:
                evaluate(f"{label}_L3_leave_site_out", f"leave_out_{s}", d[d["site"] != s], d[d["site"] == s], seed, rows)

    # temporal splits
    for seed in SEEDS:
        evaluate("ISMN_L3_temporal", "train<=2022_test>=2023", ism[ism.index < "2022-12-25"], ism[ism.index >= "2023-01-01"], seed, rows)
        evaluate("Arnesano_L3_temporal", "train<Aug15_test>=Aug16", arn[arn.index < "2025-08-15"], arn[arn.index >= "2025-08-16"], seed, rows)
    # hot-day regime within ISMN (per-station top third of daily temperature)
    q = ism.groupby("site")["temp"].transform(lambda x: x.quantile(2 / 3))
    for seed in SEEDS:
        evaluate("ISMN_L3_regime", "cool->hot", ism[ism["temp"] <= q], ism[ism["temp"] > q], seed, rows)

    # cross-dataset transfer
    for seed in SEEDS:
        evaluate("X_Arnesano->ISMN", "pooled", arn, ism, seed, rows)
        evaluate("X_ISMN->Arnesano", "pooled", ism, arn, seed, rows)
    for s in sorted(ism["site"].unique()):
        evaluate("X_Arnesano->ISMN", f"to_{s}", arn, ism[ism["site"] == s], 0, rows)
    for s in sorted(arn["site"].unique()):
        evaluate("X_ISMN->Arnesano", f"to_{s}", ism, arn[arn["site"] == s], 0, rows)

    res = pd.DataFrame(rows)
    res.to_csv(out / f"results_{tag}.csv", index=False)
    summ = res.groupby(["level", "model"])[["accuracy", "balanced_acc", "macro_f1", "roc_auc"]].mean().round(3)
    summ.to_csv(out / f"summary_{tag}.csv")
    print(f"\n===== {tag} =====")
    print(summ.to_string())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arnesano", required=True)
    ap.add_argument("--ismn", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--out", default="results_cross")
    ap.add_argument("--only", default="", help="run one config tag only: p15, p25")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("XGBoost installed:", HAVE_XGB)
    configs = [("p15", 15, False), ("p25", 25, True)]       # quick = skip the random and CV splits
    for tag, pct, quick in configs:
        if a.only and a.only != tag:
            continue
        arn = load_arnesano(a.arnesano)
        ism = load_ismn(a.ismn, a.power, pct)
        print(f"\n[{tag}] Arnesano rows {len(arn)} pos {arn['y'].mean():.3f} | ISMN rows {len(ism)} pos {ism['y'].mean():.3f}")
        print(arn.groupby("site")["y"].agg(["size", "mean"]).round(3).T.to_string())
        print(ism.groupby("site")["y"].agg(["size", "mean"]).round(3).T.to_string())
        run_config(arn, ism, tag, out, quick)


if __name__ == "__main__":
    main()