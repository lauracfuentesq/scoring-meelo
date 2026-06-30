"""
modelisation.py
===============
Utilitaires partagés pour la phase de modélisation du scoring d'impayé.

Architecture
------------
Trois responsabilités, séparées proprement :

1. Calibration & métriques : find_optimal_threshold cale le seuil sur
   val sous contrainte recall >= RECALL_MIN, compute_metrics produit
   AP, Gini, FP/FN/TP/TN et précision/recall classe impayée à seuil fixé.


2. Visualisation : plot_pr_curve reproduit la courbe PR avec la même
   esthétique que la baseline Random Forest (baseline pointillée + étoile rouge
   au seuil optimal).

3. Tracking MLflow : setup_mlflow configure l'expérience, log_run
   loggue params/metrics/figure/modèle. evaluate_and_log orchestre les
   trois étapes pour qu'un modèle entraîné soit évalué + tracké en une ligne.

"""

import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# 1. Constantes — contraintes métier + esthétique
# ---------------------------------------------------------------------------

#: Nom par défaut de l'expérience MLflow.
MLFLOW_EXPERIMENT_NAME = "scoring_impaye"

#: Contrainte métier : on exige au moins 30 % de recall sur la classe impayé
#: pour qu'un seuil soit considéré comme acceptable.
DEFAULT_RECALL_MIN = 0.30

#: Couleur dédiée à chaque modèle dans les courbes PR, alignée avec la palette
#: Plotly D3 utilisée par eda_viz pour rester visuellement cohérent.
MODEL_COLORS: dict[str, str] = {
    "Random Forest": "#4C78A8",
    "LightGBM":      "#F58518",
    "CatBoost":      "#54A24B",
    "LightGBM-opt":  "#B279A2",
}


# ---------------------------------------------------------------------------
# 2. Métriques de base
# ---------------------------------------------------------------------------

def gini(y_true, y_score) -> float:
    """Gini = 2·AUC − 1 """
    return 2 * roc_auc_score(y_true, y_score) - 1


def find_optimal_threshold(
    y_true,
    y_proba,
    recall_min: float = DEFAULT_RECALL_MIN,
) -> dict[str, Any]:
    """
    Cherche le seuil de probabilité qui maximise la précision sous
    contrainte recall >= recall_min sur la classe impayée.

    Si aucun point de la courbe PR ne satisfait la contrainte, retourne
    le seuil par défaut 0.5 avec satisfied=False.

    Retour
    ------
    dict : {threshold, precision, recall, satisfied,
            precision_curve, recall_curve, thresholds}
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve retourne len(thresholds) = len(precision) - 1.
    # On aligne en tronquant le dernier point (recall=0, précision=1 ou inverse).
    mask = recall[:-1] >= recall_min

    if mask.any():
        idx_opt = int(np.argmax(precision[:-1][mask]))
        threshold = float(thresholds[mask][idx_opt])
        prec_opt = float(precision[:-1][mask][idx_opt])
        rec_opt = float(recall[:-1][mask][idx_opt])
        satisfied = True
    else:
        threshold = 0.5
        prec_opt = float("nan")
        rec_opt = float("nan")
        satisfied = False

    return {
        "threshold": threshold,
        "precision": prec_opt,
        "recall": rec_opt,
        "satisfied": satisfied,
        "precision_curve": precision,
        "recall_curve": recall,
        "thresholds": thresholds,
    }


def compute_metrics(y_true, y_proba, threshold: float) -> dict[str, float]:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    prec_unpaid = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_unpaid  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_unpaid   = (
        2 * prec_unpaid * rec_unpaid / (prec_unpaid + rec_unpaid)
        if (prec_unpaid + rec_unpaid) > 0 else 0.0
    )
    roc_auc = float(roc_auc_score(y_true, y_proba))

    return {
        "ap":               float(average_precision_score(y_true, y_proba)),
        "gini":             float(2 * roc_auc - 1),
        "roc_auc":          roc_auc,                   
        "threshold":        float(threshold),
        "tn":               int(tn),
        "fp":               int(fp),
        "fn":               int(fn),
        "tp":               int(tp),
        "precision_unpaid": float(prec_unpaid),
        "recall_unpaid":    float(rec_unpaid),
        "f1_unpaid":        float(f1_unpaid),
    }

def print_evaluation(y_true, y_proba, threshold, model_name="", split_name="") -> dict[str, float]:
    metrics = compute_metrics(y_true, y_proba, threshold)
    y_pred  = (np.asarray(y_proba) >= threshold).astype(int)

    label = " — ".join(s for s in [model_name, split_name] if s)
    if label:
        print(f"── {label} ──")
    print(f"  AP (PR-AUC)   : {metrics['ap']:.4f}")
    print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")   
    print(f"  Gini          : {metrics['gini']:.4f}")
    print(f"  Seuil         : {threshold:.3f}")
    print(f"  Faux Positifs : {metrics['fp']}  ← métrique cible")
    print(classification_report(y_true, y_pred, target_names=["payé", "impayé"], digits=3))
    return metrics


# ---------------------------------------------------------------------------
# 3. Visualisation — courbe Precision-Recall
# ---------------------------------------------------------------------------

def plot_pr_curve(
    y_true,
    y_proba,
    model_name: str,
    threshold: float | None = None,
    color: str | None = None,
    split_name: str = "val",
) -> go.Figure:
    """
    Courbe Precision-Recall avec :
      - baseline pointillée = taux d'impayé réel (classificateur naïf)
      - étoile rouge au seuil optimal si fourni
      - tooltip qui affiche le seuil sur chaque point
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    color = color or MODEL_COLORS.get(model_name, "#4C78A8")

    fig = go.Figure()

    # Baseline = taux de positifs (impayés) dans le split évalué
    baseline = float(np.mean(y_true))
    fig.add_hline(
        y=baseline, line_dash="dash", line_color="grey",
        annotation_text=f"Baseline ({baseline:.2f})",
        annotation_position="bottom right",
    )

    # Point optimal (si seuil fourni)
    if threshold is not None:
        # On retrouve le point le plus proche du seuil sur la courbe
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        fig.add_trace(go.Scatter(
            x=[recall[idx]], y=[precision[idx]],
            mode="markers",
            marker=dict(size=12, color="red", symbol="star"),
            name=f"Seuil optimal ({threshold:.2f})",
        ))

    fig.add_trace(go.Scatter(
        x=recall[:-1], y=precision[:-1],
        mode="lines",
        customdata=thresholds,
        hovertemplate=(
            "Recall: %{x:.3f}<br>"
            "Précision: %{y:.3f}<br>"
            "Seuil: %{customdata:.3f}"
            "<extra></extra>"
        ),
        name=f"{model_name} (AP={ap:.3f})",
        line=dict(color=color, width=2),
    ))

    fig.update_layout(
        title=f"Courbe Precision-Recall — {model_name} ({split_name})",
        xaxis=dict(title="Recall", range=[0, 1]),
        yaxis=dict(title="Précision", range=[0, 1]),
        width=700, height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


# ---------------------------------------------------------------------------
# 4. MLflow — setup, logging, comparaison
# ---------------------------------------------------------------------------

def setup_mlflow(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> str:
    """
    Initialise MLflow : tracking URI (défaut : ./mlruns) et expérience.
    Retourne le nom de l'expérience (pour vérification).
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return experiment_name


def _log_model_artifact(model: Any, flavor: str | None) -> None:
    """Logge le modèle dans le flavor MLflow correspondant (best-effort)."""
    if model is None or flavor is None:
        return
    try:
        if flavor == "sklearn":
            import mlflow.sklearn
            mlflow.sklearn.log_model(model, artifact_path="model")

        elif flavor == "lightgbm":
            import mlflow.lightgbm
            mlflow.lightgbm.log_model(model, artifact_path="model")

        elif flavor == "catboost":
            import mlflow.catboost
            mlflow.catboost.log_model(model, artifact_path="model")
        else:

            print(f" Flavor MLflow inconnu : {flavor!r}, modèle non loggué.")

    except Exception as exc:  # noqa: BLE001

        print(f" Échec du log du modèle ({flavor}) : {exc}")


def log_run(
    run_name: str,
    params: dict,
    metrics_val: dict,
    metrics_train: dict | None = None,  
    figure: go.Figure | None = None,
    model: Any = None,
    model_flavor: str | None = None,
    tags: dict | None = None,
) -> str:
    
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)

        mlflow.log_params({
            k: str(v) if not isinstance(v, (int, float, str, bool)) else v
            for k, v in params.items()
        })

        for k, v in metrics_val.items():
            mlflow.log_metric(f"val_{k}", float(v))

        if metrics_train:
            for k, v in metrics_train.items():
                mlflow.log_metric(f"train_{k}", float(v))  

        if figure is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "pr_curve.html"
                figure.write_html(str(path), include_plotlyjs="cdn")
                mlflow.log_artifact(str(path))

        _log_model_artifact(model, model_flavor)

        return run.info.run_id


def evaluate_and_log(
    model_name: str,
    run_name: str,
    params: dict,
    y_val,
    proba_val,
    y_train=None,       
    proba_train=None,   
    *,
    recall_min: float = DEFAULT_RECALL_MIN,
    model: Any = None,
    model_flavor: str | None = None,
    show_figure: bool = True,
    tags: dict | None = None,
) -> dict[str, Any]:
    
    thr_info  = find_optimal_threshold(y_val, proba_val, recall_min)
    threshold = thr_info["threshold"]
    if not thr_info["satisfied"]:
        print(f" Aucun seuil ne satisfait recall ≥ {recall_min:.2f} "
              f" seuil par défaut 0.5 utilisé.")

    # Métriques val, puis train (même seuil pour comparaison directe)
    metrics_val   = print_evaluation(y_val,   proba_val,   threshold,
                                     model_name=model_name, split_name="val")
    metrics_train = None
    if y_train is not None and proba_train is not None:
        metrics_train = print_evaluation(y_train, proba_train, threshold,
                                         model_name=model_name, split_name="train")

    #  Courbe PR sur val
    fig = plot_pr_curve(y_val, proba_val, model_name, threshold=threshold)
    if show_figure:
        fig.show()

    #  MLflow
    run_id = log_run(
        run_name=run_name,
        params=params,
        metrics_val=metrics_val,
        metrics_train=metrics_train, 
        figure=fig,
        model=model,
        model_flavor=model_flavor,
        tags=tags,
    )
    return {
        "threshold":     threshold,
        "metrics_val":   metrics_val,
        "metrics_train": metrics_train,  
        "figure":        fig,
        "run_id":        run_id,
    }

# ---------------------------------------------------------------------------
# 5. Comparaison des runs
# ---------------------------------------------------------------------------
def compare_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric_for_sort: str = "val_ap",
) -> pd.DataFrame:
    
    exp = mlflow.get_experiment_by_name(experiment_name)
    
    if exp is None:
        print(f" Expérience MLflow '{experiment_name}' introuvable.")
        return pd.DataFrame()

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        return runs

    rename_map = {
        "tags.mlflow.runName":           "run_name",
        # val
        "metrics.val_ap":                "val_ap",
        "metrics.val_roc_auc":           "val_roc_auc",      
        "metrics.val_gini":              "val_gini",
        "metrics.val_fp":                "val_fp",
        "metrics.val_precision_unpaid":  "val_precision",
        "metrics.val_recall_unpaid":     "val_recall",
        "metrics.val_f1_unpaid":         "val_f1",
        "metrics.val_threshold":         "val_threshold",
        # train
        "metrics.train_ap":              "train_ap",          
        "metrics.train_roc_auc":         "train_roc_auc",    
        "metrics.train_gini":            "train_gini",       
        "metrics.train_fp":              "train_fp",         
        "metrics.train_precision_unpaid":"train_precision",  
        "metrics.train_recall_unpaid":   "train_recall",     
    }

    cols_present = [c for c in rename_map if c in runs.columns]
    out = runs[cols_present].rename(columns=rename_map)

    if metric_for_sort in out.columns:
        out = out.sort_values(metric_for_sort, ascending=False)

    return out.reset_index(drop=True)
