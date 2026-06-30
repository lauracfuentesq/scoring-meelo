# scoring-meelo

Scoring d'impayé (`top_unpaid`) sur des données de souscription : de l'analyse exploratoire au modèle, avec suivi des expériences via MLflow.

## Structure

| Fichier | Rôle |
|---------|------|
| `eda.ipynb` | Analyse exploratoire : distributions, relation avec la cible, colinéarité (V de Cramér), Information Value. |
| `eda.html` | Export HTML autonome de `eda.ipynb` : ouvre-le dans un navigateur pour visualiser les graphiques Plotly interactifs sans rien exécuter. |
| `scoring.ipynb` | Pipeline complet : feature engineering → modélisation (RF, LightGBM, CatBoost, Optuna) → interprétabilité (SHAP, Precision/Recall@K). |
| `scoring.html` | Export HTML autonome de `scoring.ipynb` : ouvre-le dans un navigateur pour visualiser les graphiques Plotly interactifs sans rien exécuter. |
| `feature_engineering.py` | Transformations fit-sur-train / apply-sur-val-test (split temporel, imputation, encodings, troncature, etc.). |
| `eda_viz.py` | Fonctions de visualisation utilisées par `eda.ipynb`. |
| `modelisation.py` | Helpers d'évaluation et de logging MLflow. |
| `data/data.csv` | Jeu de données d'entrée. |
| `mlruns/` | Expériences enregistrées par MLflow. |

## Utilisation

```bash
uv sync                 # installe les dépendances (Python 3.12)
uv run jupyter lab      # ouvre les notebooks
mlflow ui               # explore les expériences sur http://localhost:5000
```

Ordre recommandé : `eda.ipynb` → `scoring.ipynb`.

Pour consulter les résultats et les graphiques sans installer ni exécuter quoi que ce soit, ouvre directement `eda.html` ou `scoring.html` dans un navigateur (fichiers autonomes, fonctionnent hors ligne).

## Principes

- **Split temporel** (non aléatoire) : on entraîne sur le passé, on évalue sur le futur.
- **Zéro fuite** : toute statistique est fittée sur le train et appliquée telle quelle à val/test.
