"""
feature_eng.py
==============
Pipeline de feature engineering pour le scoring d'impayé.

Architecture
------------
Toutes les transformations sont **fittées sur le train** et **appliquées**
au validation/test via des classes ``Fit/Transform`` (style scikit-learn).
Aucune statistique du test ne fuite dans le train.

Modules logiques (correspondant aux phases du notebook) :
  - Phase 1 : ``temporal_split``        — split temporel 70/15/15
  - Phase 2 : ``MissingHandler``        — flags + imputation
  - Phase 3 : ``RareCategoryGrouper``   — regroupement par fréquence
  - Phase 4 : ``WoEEncoder``            — encoding WoE (catégorielles)
              ``TargetEncoder``         — encoding par taux moyen (zip)
  - Phase 5 : ``add_derived_features``  — tenure_months
  - Phase 6 : helpers de vérification (IV, corrélation)
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Phase 1 — Split temporel
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    date_col: str,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """
    Découpe le DataFrame par ordre chronologique du ``date_col``.

    train_frac + val_frac doivent être < 1 ; le reste va au test.
    Le split aléatoire est interdit en scoring : on entraîne sur le passé
    et on évalue sur le futur, comme en production.

    """
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac doivent laisser de la place au test")

    ordered = df.sort_values(date_col).reset_index(drop=True)
    n = len(ordered)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    train = ordered.iloc[:i_train].copy()
    val = ordered.iloc[i_train:i_val].copy()
    test = ordered.iloc[i_val:].copy()

    return train, val, test


def split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    date_col: str,
) -> pd.DataFrame:
    
    """Résume volume, période et taux d'impayé de chaque sous-ensemble."""
    rows = []
    for name, sub in [("train", train), ("val", val), ("test", test)]:
        rows.append({
            "split": name,
            "n": len(sub),
            "période_début": sub[date_col].min().strftime("%Y-%m-%d"),
            "période_fin": sub[date_col].max().strftime("%Y-%m-%d"),
            "taux_impayé_%": round(sub[target_col].mean() * 100, 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 2 — Valeurs manquantes
# ---------------------------------------------------------------------------

@dataclass
class MissingHandler:
    """
    Impute les valeurs manquantes :
      - "MISSING" pour les catégorielles
      - médiane (figée sur train) pour les numériques

    """
    cat_cols: Sequence[str]
    num_cols: Sequence[str]
    medians_: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "MissingHandler":
        self.medians_ = {c: float(df[c].median()) for c in self.num_cols}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.cat_cols:
            out[col] = out[col].fillna("MISSING")
        for col in self.num_cols:
            out[col] = out[col].fillna(self.medians_[col])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

# ---------------------------------------------------------------------------
# Phase 3 — Regroupement des modalités rares (par fréquence)
# ---------------------------------------------------------------------------

@dataclass
class RareCategoryGrouper:
    """
    Pour chaque colonne catégorielle : toute modalité représentant moins
    de ``min_freq`` du volume **en train** est remplacée par ``"OTHER"``.

    Calculé sur train uniquement -> les modalités fréquentes en train
    restent telles quelles, les rares (et toute modalité jamais vue)
    deviennent "OTHER" dans val/test.

    ``protected`` : labels toujours conservés quelle que soit leur fréquence
    (par défaut ``{"MISSING"}`` pour cohérence avec ``MissingHandler``).
    """
    cols: Sequence[str]
    min_freq: float = 0.01           # 1% du volume par défaut
    other_label: str = "OTHER"
    protected: frozenset = frozenset({"MISSING"})
    kept_: dict[str, set] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "RareCategoryGrouper":
        for col in self.cols:
            freq = df[col].value_counts(normalize=True)
            frequent = set(freq[freq >= self.min_freq].index)
            present = set(df[col].dropna().unique())
            self.kept_[col] = frequent | (self.protected & present)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.cols:
            kept = self.kept_[col]
            out[col] = out[col].where(out[col].isin(kept), self.other_label)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def summary(self) -> pd.DataFrame:
            """
            Récapitulatif par variable : nombre et liste des modalités conservées.
            Les modalités sont triées alphabétiquement pour faciliter la lecture.
            """
            return pd.DataFrame(
                [
                    {
                        "variable": c,
                        "n_modalités_gardées": len(v),
                        "modalités_gardées": sorted(map(str, v)),
                    }
                    for c, v in self.kept_.items()
                ]
            )


# ---------------------------------------------------------------------------
# Phase 4 — Encodings
# ---------------------------------------------------------------------------

@dataclass
class WoEEncoder:
    """
    WoE encoding pour variables catégorielles : chaque modalité est
    remplacée par son Weight of Evidence calculé sur train.

    Détails :
      - epsilon ajouté pour éviter log(0) sur les classes vides
      - modalités inconnues au transform -> WoE = 0 (neutre)
    """
    cols: Sequence[str]
    eps: float = 0.5
    woe_maps_: dict[str, dict] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, target_col: str) -> "WoEEncoder":
        total_bad = df[target_col].sum()
        total_good = len(df) - total_bad

        for col in self.cols:
            grouped = df.groupby(col, observed=True)[target_col].agg(["sum", "count"])
            bad = grouped["sum"]
            good = grouped["count"] - bad
            pct_bad = (bad + self.eps) / (total_bad + self.eps)
            pct_good = (good + self.eps) / (total_good + self.eps)
            self.woe_maps_[col] = np.log(pct_bad / pct_good).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.cols:
            out[f"{col}_woe"] = out[col].map(self.woe_maps_[col]).fillna(0.0)
        return out

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        return self.fit(df, target_col).transform(df)


@dataclass
class TargetEncoder:
    """
    Target encoding avec lissage bayésien pour variables à forte cardinalité
    (typiquement ``zip_code_prefix``) :

        encoding(c) = (n_c * taux_c + smoothing * taux_global) / (n_c + smoothing)

    Le smoothing tire les modalités à faible effectif vers la moyenne globale,
    ce qui évite les estimations bruitées sur les zones peu peuplées.
    """
    cols: Sequence[str]
    smoothing: float = 30.0
    global_rate_: float = 0.0
    maps_: dict[str, dict] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, target_col: str) -> "TargetEncoder":
        self.global_rate_ = float(df[target_col].mean())
        for col in self.cols:
            grouped = df.groupby(col, observed=True)[target_col].agg(["mean", "count"])
            n = grouped["count"]
            rate = grouped["mean"]
            encoded = (n * rate + self.smoothing * self.global_rate_) / (n + self.smoothing)
            self.maps_[col] = encoded.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.cols:
            out[f"{col}_target_enc"] = (
                out[col].map(self.maps_[col]).fillna(self.global_rate_)
            )
        return out

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        return self.fit(df, target_col).transform(df)




# ---------------------------------------------------------------------------
# Phase 5 — Features dérivées
# ---------------------------------------------------------------------------

def add_tenure_months(
    df: pd.DataFrame,
    date_col: str,
    *,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Ajoute ``tenure_months`` (ancienneté en mois) calculée par rapport à
    une ``reference_date``. Si non fournie, on prend la date max du DataFrame.

    Important : pour val/test on doit passer la **même** ``reference_date``
    que celle utilisée sur le train, sinon la définition de l'ancienneté
    change entre splits.
    """
    out = df.copy()
    ref = reference_date if reference_date is not None else out[date_col].max()
    out["tenure_months"] = ((ref - out[date_col]).dt.days / 30.44).round(2)
    return out


def truncate_at_percentile(
    df: pd.DataFrame,
    col: str,
    *,
    percentile: float = 99,
    cap: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Tronque (winsorise) ``col`` à un plafond haut pour limiter l'influence
    des valeurs extrêmes.

    Le plafond est le ``percentile``-ème centile **calculé sur le train**.
    Pour val/test, on passe le ``cap`` retourné par l'appel sur le train,
    afin de ne pas réestimer le seuil sur d'autres splits (pas de fuite).

    Retourne le DataFrame transformé **et** le plafond utilisé, de sorte
    que l'appelant puisse le réinjecter pour val/test :

        train, cap = truncate_at_percentile(train, "monthly_amount")
        val,  _ = truncate_at_percentile(val,  "monthly_amount", cap=cap)
        test, _ = truncate_at_percentile(test, "monthly_amount", cap=cap)
    """
    out = df.copy()
    threshold = cap if cap is not None else float(np.nanpercentile(out[col], percentile))
    out[col] = out[col].clip(upper=threshold)
    return out, threshold


# ---------------------------------------------------------------------------
# Encodage géographique : zip_code_prefix -> densité urbaine
# ---------------------------------------------------------------------------

# Catégories urbaines selon la densité (seuils INSEE simplifiés)
_URBAN_CATEGORIES = [
    (5000, "métropole_dense"),    # Paris, petite couronne
    (500,  "urbain_dense"),       # Grandes villes
    (150,  "urbain"),             # Villes moyennes
    (50,   "périurbain"),         # Périphéries, semi-rural
    (0,    "rural"),              # Faible densité
]


def build_dept_density_table(
    csv_url: str = "https://www.data.gouv.fr/api/1/datasets/r/262afe2d-1c35-40da-ace0-a2eb595eaced",
) -> dict[str, float]:
    """
    Construit la table {préfixe code postal (2 chiffres) : densité hab/km²}
    à partir du dataset officiel 'Communes et villes de France' (data.gouv.fr).

    On agrège par les 2 premiers chiffres du code postal
    pour matcher exactement la définition de zip_code_prefix dans nos données.

    """
    communes = pd.read_csv(
        csv_url,
        dtype={"code_postal": str},
        compression="gzip",
        usecols=["code_postal", "population", "superficie_km2"],
    )
    communes = communes.dropna(subset=["code_postal", "population", "superficie_km2"])

    # Préfixe = 2 premiers caractères du code postal (en string, avec zéros à gauche)
    communes["zip_prefix"] = communes["code_postal"].str.zfill(5).str[:2]

    agg = (
        communes.groupby("zip_prefix")
        .agg(pop=("population", "sum"), surface=("superficie_km2", "sum"))
    )
    agg["density"] = agg["pop"] / agg["surface"]
    
    return {k: round(float(v), 1) for k, v in agg["density"].items()}


def encode_zip_urban_density(
    df: pd.DataFrame,
    density_table: dict[int, float],
    *,
    zip_col: str = "zip_code_prefix",
    out_col: str = "urban_density",
    unknown_label: str = "unknown",
) -> pd.DataFrame:
    """
    Transforme ``zip_code_prefix`` en catégorie d'urbanité basée sur la
    densité de population du département.

    Catégories produites :
      - métropole_dense (> 5000 hab/km²)
      - urbain_dense   (500 - 5000)
      - urbain         (150 - 500)
      - périurbain     (50 - 150)
      - rural          (< 50)
      - unknown        (département absent du référentiel)

    Avantages vs target encoding :
      - aucune fuite (la table est statique, indépendante du target)
      - interprétable
      - robuste aux modalités à faible effectif
    """
    out = df.copy()
    prefix = (
        pd.to_numeric(out[zip_col], errors="coerce")  
        .astype("Int64")                              
        .astype(str)                                  
        .str.zfill(2)                                 
    )
    density = prefix.map(density_table)

    def to_category(d: float) -> str:
        if pd.isna(d):
            return unknown_label
        for threshold, label in _URBAN_CATEGORIES:
            if d >= threshold:
                return label
        return unknown_label

    out[out_col] = density.map(to_category)
    return out

# ---------------------------------------------------------------------------
# Regroupements sémantiques (avant le RareCategoryGrouper)
# ---------------------------------------------------------------------------

# Taxonomie métier des banques françaises.
# Toute banque absente du mapping tombera dans "OTHER" via apply_manual_groups.
BANK_GROUPS: dict[str, str] = {
    # Néobanques / fintechs
    "REVOLUT": "neobanque",
    "N26": "neobanque",
    "LYDIA SOLUTIONS": "neobanque",
    "NICKEL": "neobanque",
    "MA FRENCH BANK": "neobanque",
    # Banques en ligne
    "BOURSORAMA": "banque_en_ligne",
    "FORTUNEO": "banque_en_ligne",
    # Banques traditionnelles
    "BNP PARIBAS": "banque_traditionnelle",
    "CRÉDIT AGRICOLE": "banque_traditionnelle",
    "SOCIÉTÉ GÉNÉRALE": "banque_traditionnelle",
    "LCL": "banque_traditionnelle",
    "CIC": "banque_traditionnelle",
    "CRÉDIT MUTUEL": "banque_traditionnelle",
    "CRÉDIT INDUSTRIEL DE L'OUEST": "banque_traditionnelle",
    "CAISSE D'ÉPARGNE": "banque_traditionnelle",
    "BANQUE POPULAIRE": "banque_traditionnelle",
    # Banque postale (profil socio-éco distinct)
    "LA BANQUE POSTALE": "banque_postale",
}


def apply_manual_groups(
    df: pd.DataFrame,
    mapping: dict[str, dict[str, str]],
    *,
    other_label: str = "OTHER",
    missing_label: str = "MISSING",
) -> pd.DataFrame:
    """
    Applique des regroupements sémantiques manuels aux variables catégorielles.

    Paramètres
    ----------
    mapping : {nom_colonne: {valeur_originale: groupe_cible}}
        Les valeurs absentes du mapping tombent dans ``other_label``.
        Les NaN sont préservés en ``missing_label`` (cohérent avec
        MissingHandler).

    À utiliser AVANT le RareCategoryGrouper : le regroupement consolide
    les modalités, ce qui permet à des groupes (et non des modalités
    individuelles) de franchir le seuil de fréquence.
    """
    out = df.copy()
    for col, col_map in mapping.items():
        # On préserve l'information de manque avant le mapping
        mask_missing = out[col].isna()
        out[col] = out[col].map(col_map).fillna(other_label)
        out.loc[mask_missing, col] = missing_label
    return out


def derive_email_features(
    df: pd.DataFrame,
    *,
    col: str = "email_domain",
) -> pd.DataFrame:
    """
    Dérive des features sémantiques du domaine email.

    Avantage vs un mapping manuel : ces features s'appliquent à n'importe
    quel domaine, y compris ceux jamais vus en train (proton.me, dominios
    professionnels, étrangers...).

    Features créées :
      - email_tld           : extension (.fr, .com, .net, ...)
      - email_is_french     : domaine français (TLD .fr ou FAI fr connu)
      - email_is_webmail    : webmail grand public (gmail, yahoo, hotmail...)
      - email_domain_length : longueur du domaine (atypique si très court/long)
    """
    out = df.copy()
    domain = out[col].astype(str).str.lower()

    out["email_tld"] = domain.str.extract(r"\.([a-z]+)$")[0].fillna("unknown")

    fai_fr = ["orange", "wanadoo", "free", "sfr", "bbox", "laposte", "numericable"]
    is_fr_provider = domain.str.contains("|".join(fai_fr), regex=True, na=False)
    out["email_is_french"] = (domain.str.endswith(".fr") | is_fr_provider).astype(int)

    webmails = ["gmail", "yahoo", "hotmail", "outlook", "live", "icloud", "msn", "aol"]
    out["email_is_webmail"] = domain.str.contains(
        "|".join(webmails), regex=True, na=False
    ).astype(int)

    out["email_domain_length"] = domain.str.len()
    return out

# ---------------------------------------------------------------------------
# Phase 6 — Vérifications
# ---------------------------------------------------------------------------

def feature_correlation(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Matrice de corrélation de Pearson pour les features finales (numériques)."""
    return df[list(features)].corr().round(3)


def drop_highly_correlated(
    corr: pd.DataFrame,
    *,
    threshold: float = 0.85,
) -> list[str]:
    """
    Retourne la liste des features à candidates au retrait pour cause
    de corrélation supérieure à ``threshold``. Garde la première de chaque
    paire, retire la seconde.
    """
    to_drop = set()
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            if abs(corr.loc[c1, c2]) >= threshold:
                to_drop.add(c2)
    return sorted(to_drop)
