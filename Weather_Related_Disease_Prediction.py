

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode, rankdata
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# -----------------------------------------------------------------------------
# Reproducibility and experiment configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_COLUMN = "prognosis"
TEST_SIZE = 0.20
N_CLIENTS = 3
CV_FOLDS = 3
BOOTSTRAP_RESAMPLES = 2_000

# Centralized model configurations. The same configurations are used by the
# corresponding simulated clients to keep centralized/distributed comparisons
# on a consistent model specification.
GBM_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 150,
    "learning_rate": 0.10,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 1,
    "reg_lambda": 1,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}

LGBM_PARAMS: dict[str, Any] = {
    "num_leaves": 15,
    "max_depth": 5,
    "n_estimators": 400,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "min_child_samples": 30,
    "min_child_weight": 0.01,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "verbosity": -1,
    "n_jobs": 1,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("weather_disease")


# -----------------------------------------------------------------------------
# Data loading and preparation
# -----------------------------------------------------------------------------
def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, LabelEncoder]:
    """Load the CSV, remove exact duplicates, and encode the target labels."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Pass its location with --data PATH."
        )

    data = pd.read_csv(csv_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"The dataset must contain a '{TARGET_COLUMN}' column.")

    original_rows = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    LOGGER.info(
        "Loaded %d rows and %d columns; removed %d exact duplicates (remaining: %d).",
        original_rows,
        data.shape[1],
        original_rows - len(data),
        len(data),
    )

    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"The target column '{TARGET_COLUMN}' contains missing values.")

    X = data.drop(columns=[TARGET_COLUMN]).copy()
    y_text = data[TARGET_COLUMN].astype(str)

    # The supplied dataset has numeric predictors. If a future version contains
    # categorical predictors, encode them once into a numeric feature matrix.
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        LOGGER.info("One-hot encoding categorical predictors: %s", non_numeric)
        X = pd.get_dummies(X, columns=non_numeric, drop_first=True, dtype=float)

    X = X.apply(pd.to_numeric, errors="raise")
    X = X.replace([np.inf, -np.inf], np.nan)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    if X.shape[1] == 0:
        raise ValueError("No predictor columns remain after data preparation.")

    LOGGER.info("Predictors: %d | Disease classes: %d", X.shape[1], len(label_encoder.classes_))
    LOGGER.info("Class labels: %s", list(label_encoder.classes_))
    return data, X, y, label_encoder


def make_model_specs(n_classes: int) -> dict[str, Any]:
    """Return fresh centralized estimator instances."""
    xgb_params = dict(XGB_PARAMS)
    xgb_params["num_class"] = n_classes

    lgbm_params = dict(LGBM_PARAMS)
    lgbm_params.update({"objective": "multiclass", "num_class": n_classes})

    return {
        "GBM": GradientBoostingClassifier(**GBM_PARAMS),
        "XGBoost": XGBClassifier(**xgb_params),
        "LightGBM": LGBMClassifier(**lgbm_params),
    }


def make_pipeline(estimator: Any) -> Pipeline:
    """Impute missing numeric values using training data only, then fit a model."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


# -----------------------------------------------------------------------------
# Metrics and plots
# -----------------------------------------------------------------------------
def multiclass_auc(y_true: np.ndarray, y_probability: np.ndarray) -> tuple[float, float]:

    return (
        float(roc_auc_score(y_true, y_probability, multi_class="ovr", average="macro")),
        float(roc_auc_score(y_true, y_probability, multi_class="ovr", average="weighted")),
    )


def stratified_bootstrap_auc_ci(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    n_classes: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    random_state: int = RANDOM_STATE,
) -> list[tuple[float, float]]:

    if n_resamples <= 0:
        return [(float("nan"), float("nan"))] * n_classes

    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(y_true == class_id) for class_id in range(n_classes)]
    class_counts = [len(indices) for indices in class_indices]
    if any(count == 0 for count in class_counts):
        raise ValueError("Every class must be represented in the test set for OvR AUC.")

    # Resample each true class independently, then concatenate the class blocks.
    bootstrap_indices = np.concatenate(
        [
            rng.choice(indices, size=(n_resamples, len(indices)), replace=True)
            for indices in class_indices
        ],
        axis=1,
    )

    offsets = np.cumsum([0] + class_counts)
    confidence_intervals: list[tuple[float, float]] = []
    for class_id, n_positive in enumerate(class_counts):
        class_scores = y_probability[bootstrap_indices, class_id]
        start, end = offsets[class_id], offsets[class_id + 1]
        positive_scores = class_scores[:, start:end]
        negative_scores = np.concatenate(
            [class_scores[:, :start], class_scores[:, end:]], axis=1
        )
        n_negative = negative_scores.shape[1]

        scores = np.concatenate([positive_scores, negative_scores], axis=1)
        ranks = rankdata(scores, axis=1, method="average")
        rank_sum_positive = ranks[:, :n_positive].sum(axis=1)
        auc_samples = (
            rank_sum_positive - n_positive * (n_positive + 1) / 2
        ) / (n_positive * n_negative)

        lower, upper = np.percentile(auc_samples, [2.5, 97.5])
        confidence_intervals.append((float(lower), float(upper)))

    return confidence_intervals


def build_classwise_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
    class_names: list[str],
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    """Calculate per-class support, sensitivity, specificity, and OvR AUC."""
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    ci_values = stratified_bootstrap_auc_ci(
        y_true, y_probability, len(class_names), n_resamples=n_bootstrap
    )

    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        tp = int(cm[class_id, class_id])
        fn = int(cm[class_id, :].sum() - tp)
        fp = int(cm[:, class_id].sum() - tp)
        tn = total - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        class_auc = roc_auc_score(y_true == class_id, y_probability[:, class_id])
        ci_low, ci_high = ci_values[class_id]

        rows.append(
            {
                "Model": model_name,
                "Disease class": class_name,
                "Test support": int((y_true == class_id).sum()),
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "OvR AUC": float(class_auc),
                "AUC 95% CI lower": ci_low,
                "AUC 95% CI upper": ci_high,
            }
        )
    return rows


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        display_labels=class_names,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        ax=ax,
    )
    display.ax_.set_title(title)
    display.ax_.set_xlabel("Predicted class")
    display.ax_.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_ovr_roc_plot(
    model_name: str,
    y_true: np.ndarray,
    y_probability: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for class_id, class_name in enumerate(class_names):
        binary_true = (y_true == class_id).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_probability[:, class_id])
        class_auc = roc_auc_score(binary_true, y_probability[:, class_id])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={class_auc:.3f})", linewidth=1.4)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set(
        title=f"One-vs-rest ROC curves: {model_name}",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(output_path, index=True)


# -----------------------------------------------------------------------------
# Centralized model evaluation
# -----------------------------------------------------------------------------
def evaluate_centralized_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    n_bootstrap: int,
) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.DataFrame]:
    n_classes = len(class_names)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=False)
    model_specs = make_model_specs(n_classes)

    fitted_models: dict[str, Pipeline] = {}
    summary_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []

    for model_name, estimator in model_specs.items():
        LOGGER.info("Training centralized %s model...", model_name)
        # Cross-validation and final fitting use separate fresh pipelines.
        # This also keeps fold-specific imputation isolated from the final fit.
        cv_model = make_pipeline(clone(estimator))
        cv_scores = cross_val_score(
            cv_model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )
        LOGGER.info("Completed %d-fold cross-validation for %s.", CV_FOLDS, model_name)

        model = make_pipeline(clone(estimator))
        model.fit(X_train, y_train)

        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)
        test_probability = model.predict_proba(X_test)
        macro_auc, weighted_auc = multiclass_auc(y_test, test_probability)

        metrics = {
            "Model": model_name,
            "Train accuracy": accuracy_score(y_train, train_prediction),
            "Test accuracy": accuracy_score(y_test, test_prediction),
            "Precision macro": precision_score(y_test, test_prediction, average="macro", zero_division=0),
            "Precision weighted": precision_score(y_test, test_prediction, average="weighted", zero_division=0),
            "Recall macro": recall_score(y_test, test_prediction, average="macro", zero_division=0),
            "Recall weighted": recall_score(y_test, test_prediction, average="weighted", zero_division=0),
            "F1 macro": f1_score(y_test, test_prediction, average="macro", zero_division=0),
            "F1 weighted": f1_score(y_test, test_prediction, average="weighted", zero_division=0),
            "Balanced accuracy": balanced_accuracy_score(y_test, test_prediction),
            "MCC": matthews_corrcoef(y_test, test_prediction),
            "Macro OvR AUC": macro_auc,
            "Weighted OvR AUC": weighted_auc,
            "CV accuracy mean": float(np.mean(cv_scores)),
            "CV accuracy SD": float(np.std(cv_scores, ddof=1)),
        }
        summary_rows.append(metrics)

        classwise_rows.extend(
            build_classwise_metrics(
                model_name,
                y_test,
                test_prediction,
                test_probability,
                class_names,
                n_bootstrap=n_bootstrap,
            )
        )

        save_classification_report(
            y_test,
            test_prediction,
            class_names,
            output_dir / f"classification_report_{model_name.lower()}.csv",
        )
        save_confusion_matrix(
            y_test,
            test_prediction,
            class_names,
            f"{model_name} - Test Confusion Matrix",
            output_dir / f"confusion_matrix_{model_name.lower()}.png",
        )
        save_ovr_roc_plot(
            model_name,
            y_test,
            test_probability,
            class_names,
            output_dir / f"roc_ovr_{model_name.lower()}.png",
        )
        fitted_models[model_name] = model

        LOGGER.info(
            "%s | test accuracy %.4f | macro-F1 %.4f | weighted-F1 %.4f | MCC %.4f | macro OvR AUC %.4f",
            model_name,
            metrics["Test accuracy"],
            metrics["F1 macro"],
            metrics["F1 weighted"],
            metrics["MCC"],
            metrics["Macro OvR AUC"],
        )

    summary = pd.DataFrame(summary_rows).sort_values("Test accuracy", ascending=False)
    classwise = pd.DataFrame(classwise_rows)
    summary.to_csv(output_dir / "centralized_model_metrics.csv", index=False)
    classwise.to_csv(output_dir / "classwise_centralized_metrics.csv", index=False)
    return fitted_models, summary, classwise


# -----------------------------------------------------------------------------
# Three-client simulated learning with prediction-level majority voting
# -----------------------------------------------------------------------------
def fit_client_models(
    model_name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_clients: int = N_CLIENTS,
) -> tuple[list[Pipeline], list[int]]:

    client_indices = np.array_split(np.arange(len(X_train)), n_clients)
    X_parts = [X_train.iloc[indices] for indices in client_indices]
    y_parts = [y_train[indices] for indices in client_indices]
    client_models: list[Pipeline] = []
    client_sizes: list[int] = []

    expected_classes = np.unique(y_train)
    for client_id, (X_local, y_local) in enumerate(zip(X_parts, y_parts), start=1):
        if not np.array_equal(np.unique(y_local), expected_classes):
            raise ValueError(
                f"Client {client_id} does not contain all training classes. "
                "The current multiclass aggregation expects every client model "
                "to return probabilities for the same set of classes."
            )

        local_model = make_pipeline(clone(estimator))
        local_model.fit(X_local, y_local)
        client_models.append(local_model)
        client_sizes.append(len(X_local))
        LOGGER.info("%s client %d trained on %d observations.", model_name, client_id, len(X_local))

    return client_models, client_sizes


def aggregate_client_predictions(
    client_models: list[Pipeline],
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:

    client_predictions = np.vstack([model.predict(X_test) for model in client_models])
    majority_result = mode(client_predictions, axis=0, keepdims=False).mode
    majority_prediction = np.asarray(majority_result, dtype=int).reshape(-1)

    client_probabilities = np.stack(
        [model.predict_proba(X_test) for model in client_models], axis=0
    )
    mean_probability = np.mean(client_probabilities, axis=0)
    return majority_prediction, mean_probability


def evaluate_federated_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> tuple[dict[str, list[Pipeline]], pd.DataFrame]:

    n_classes = len(class_names)
    estimator_specs = make_model_specs(n_classes)
    client_model_sets: dict[str, list[Pipeline]] = {}
    summary_rows: list[dict[str, Any]] = []

    for model_name, estimator in estimator_specs.items():
        LOGGER.info("Running three-client %s simulation...", model_name)
        client_models, client_sizes = fit_client_models(
            model_name, estimator, X_train, y_train, n_clients=N_CLIENTS
        )
        global_prediction, mean_probability = aggregate_client_predictions(client_models, X_test)
        macro_auc, weighted_auc = multiclass_auc(y_test, mean_probability)

        metrics = {
            "Model": model_name,
            "Number of clients": N_CLIENTS,
            "Client training sizes": ", ".join(map(str, client_sizes)),
            "Test accuracy": accuracy_score(y_test, global_prediction),
            "Precision macro": precision_score(y_test, global_prediction, average="macro", zero_division=0),
            "Precision weighted": precision_score(y_test, global_prediction, average="weighted", zero_division=0),
            "Recall macro": recall_score(y_test, global_prediction, average="macro", zero_division=0),
            "Recall weighted": recall_score(y_test, global_prediction, average="weighted", zero_division=0),
            "F1 macro": f1_score(y_test, global_prediction, average="macro", zero_division=0),
            "F1 weighted": f1_score(y_test, global_prediction, average="weighted", zero_division=0),
            "Balanced accuracy": balanced_accuracy_score(y_test, global_prediction),
            "MCC": matthews_corrcoef(y_test, global_prediction),
            "Macro OvR AUC (mean client probabilities)": macro_auc,
            "Weighted OvR AUC (mean client probabilities)": weighted_auc,
        }
        summary_rows.append(metrics)

        save_classification_report(
            y_test,
            global_prediction,
            class_names,
            output_dir / f"federated_classification_report_{model_name.lower()}.csv",
        )
        save_confusion_matrix(
            y_test,
            global_prediction,
            class_names,
            f"Three-client {model_name} - Majority-vote Confusion Matrix",
            output_dir / f"federated_confusion_matrix_{model_name.lower()}.png",
        )

        prediction_frame = pd.DataFrame(
            {
                "true_class": [class_names[int(label)] for label in y_test],
                "majority_vote_prediction": [class_names[int(label)] for label in global_prediction],
            }
        )
        prediction_frame.to_csv(
            output_dir / f"federated_predictions_{model_name.lower()}.csv", index=False
        )
        client_model_sets[model_name] = client_models

        LOGGER.info(
            "Federated %s | test accuracy %.4f | macro-F1 %.4f | MCC %.4f",
            model_name,
            metrics["Test accuracy"],
            metrics["F1 macro"],
            metrics["MCC"],
        )

    summary = pd.DataFrame(summary_rows).sort_values("Test accuracy", ascending=False)
    summary.to_csv(output_dir / "federated_model_metrics.csv", index=False)
    return client_model_sets, summary


# -----------------------------------------------------------------------------
# Optional SHAP and LIME explanations
# -----------------------------------------------------------------------------
def run_xai(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    class_names: list[str],
    federated_xgb_clients: list[Pipeline],
    output_dir: Path,
) -> None:
   
    try:
        import shap
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError(
            "Optional explainability packages are missing. Install them with: pip install shap lime"
        ) from exc

    LOGGER.info("Running SHAP for the first local XGBoost client model...")
    first_client = federated_xgb_clients[0]
    tree_model = first_client.named_steps["model"]
    imputer = first_client.named_steps["imputer"]
    X_test_transformed = imputer.transform(X_test)

    shap_explainer = shap.TreeExplainer(tree_model)
    shap_values = shap_explainer.shap_values(X_test_transformed)
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        show=False,
    )
    plt.gcf().savefig(
        output_dir / "shap_summary_xgboost_client1.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

    LOGGER.info("Running LIME for the first held-out observation...")
    X_train_transformed = imputer.transform(X_train)
    lime_explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train_transformed),
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode="classification",
    )

    def mean_client_probabilities(input_array: np.ndarray) -> np.ndarray:
        input_frame = pd.DataFrame(input_array, columns=X_train.columns)
        all_probabilities = [model.predict_proba(input_frame) for model in federated_xgb_clients]
        return np.mean(all_probabilities, axis=0)

    local_explanation = lime_explainer.explain_instance(
        data_row=np.asarray(X_test_transformed[0]),
        predict_fn=mean_client_probabilities,
        num_features=min(10, X_test.shape[1]),
    )
    local_explanation.save_to_file(str(output_dir / "lime_xgboost_first_test_observation.html"))
    LOGGER.info("LIME explanation: %s", local_explanation.as_list())


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Centralized and three-client disease classification pipeline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Data Set Weather-related disease prediction.csv"),
        help="Path to the input CSV file (default: Data Set Weather-related disease prediction.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weather_disease_results"),
        help="Directory in which metrics and figures will be saved.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
        help="Number of stratified bootstrap resamples for class-specific AUC CIs; use 0 to skip CIs.",
    )
    parser.add_argument(
        "--skip-federated",
        action="store_true",
        help="Run centralized models only.",
    )
    parser.add_argument(
        "--run-xai",
        action="store_true",
        help="Run optional SHAP and LIME explanations (requires shap and lime).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples < 0:
        raise ValueError("--bootstrap-resamples must be zero or a positive integer.")

    np.random.seed(RANDOM_STATE)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, X, y, label_encoder = load_dataset(args.data)
    class_names = [str(name) for name in label_encoder.classes_]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    LOGGER.info(
        "Final split | train: %d | test: %d | classes: %d",
        len(X_train),
        len(X_test),
        len(class_names),
    )
    LOGGER.info("Training class counts: %s", dict(zip(*np.unique(y_train, return_counts=True))))
    LOGGER.info("Test class counts: %s", dict(zip(*np.unique(y_test, return_counts=True))))

    # Save a concise record of the data and reproducibility settings.
    run_info = pd.DataFrame(
        [
            {"Item": "Input file", "Value": str(args.data.resolve())},
            {"Item": "Rows after exact duplicate removal", "Value": len(data)},
            {"Item": "Number of predictors", "Value": X.shape[1]},
            {"Item": "Number of classes", "Value": len(class_names)},
            {"Item": "Training observations", "Value": len(X_train)},
            {"Item": "Test observations", "Value": len(X_test)},
            {"Item": "Test fraction", "Value": TEST_SIZE},
            {"Item": "Random state", "Value": RANDOM_STATE},
            {"Item": "Simulated clients", "Value": N_CLIENTS},
            {"Item": "Cross-validation folds", "Value": CV_FOLDS},
            {"Item": "Bootstrap resamples", "Value": args.bootstrap_resamples},
        ]
    )
    run_info.to_csv(args.output_dir / "run_information.csv", index=False)

    centralized_models, centralized_summary, classwise_summary = evaluate_centralized_models(
        X_train,
        X_test,
        y_train,
        y_test,
        class_names,
        args.output_dir,
        n_bootstrap=args.bootstrap_resamples,
    )

    print("\nCENTRALIZED MODEL RESULTS")
    print(centralized_summary.round(4).to_string(index=False))
    print("\nCLASS-SPECIFIC RESULTS (TABLE-READY)")
    table_ready = classwise_summary.copy()
    table_ready["OvR AUC (95% CI)"] = table_ready.apply(
        lambda row: (
            f"{row['OvR AUC']:.4f} "
            f"({row['AUC 95% CI lower']:.4f}–{row['AUC 95% CI upper']:.4f})"
        ),
        axis=1,
    )
    print(
        table_ready[
            ["Model", "Disease class", "Test support", "Sensitivity", "Specificity", "OvR AUC (95% CI)"]
        ].round(4).to_string(index=False)
    )

    if not args.skip_federated:
        federated_clients, federated_summary = evaluate_federated_models(
            X_train,
            X_test,
            y_train,
            y_test,
            class_names,
            args.output_dir,
        )
        print("\nTHREE-CLIENT MAJORITY-VOTE RESULTS")
        print(federated_summary.round(4).to_string(index=False))

        if args.run_xai:
            run_xai(
                X_train,
                X_test,
                class_names,
                federated_clients["XGBoost"],
                args.output_dir,
            )
    elif args.run_xai:
        raise ValueError("--run-xai requires federated clients; do not combine it with --skip-federated.")

    LOGGER.info("Completed. Outputs saved in: %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
