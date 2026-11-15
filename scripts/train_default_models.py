"""Train both targets on the bundled dataset with the default settings and save
models/mrr_model.pkl and models/roughness_model.pkl (+ results/metrics CSVs).

    python scripts/train_default_models.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_default  # noqa: E402
from src.evaluation import metrics_table  # noqa: E402
from src.model_training import ALGORITHMS, RANDOM_STATE, feature_importance, train_models  # noqa: E402
from src.prediction import bundle_from_run, save_bundle  # noqa: E402
from src.preprocessing import TARGETS, default_features, prepare  # noqa: E402

NAME = "Hannover longitudinal turning dataset (DOI 10.25835/9xffqwnc)"


def main():
    raw = load_default()
    if raw is None:
        sys.exit("data/dataset.csv missing — run scripts/build_dataset.py first")
    df, _ = prepare(raw)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    summary = {"random_state": RANDOM_STATE}
    for kind in ("MRR", "Ra"):
        feats = default_features(df, kind)
        run = train_models(df, feats, TARGETS[kind]["column"], list(ALGORITHMS), split_mode="grouped")
        p = save_bundle(bundle_from_run(run, target_kind=kind, dataset_name=NAME))
        t = metrics_table(run.results)
        t.to_csv(out_dir / f"{kind}_metrics.csv", index=False)
        imp = feature_importance(run, run.best_model_name)
        imp.to_csv(out_dir / f"{kind}_importance.csv", index=False)
        summary[kind] = {"features": feats, "best": run.best_model_name, "n_train": run.n_train, "n_test": run.n_test,
                         "split": run.split_mode, "best_params": {k: str(v) for k, v in run.best["best_params"].items()},
                         "test_metrics": run.best["test_metrics"]}
        print(f"\n=== {kind} -> {p.name}  (best: {run.best_model_name})")
        print(t[["Model", "CV RMSE (mean)", "Test R2", "Test MAE", "Test MSE", "Test RMSE"]].round(5).to_string(index=False))
        print(imp.round(4).to_string(index=False))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
