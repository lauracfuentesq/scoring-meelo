"""
eda_viz.py
==========
Fonctions standardisées de visualisation pour l'analyse exploratoire
d'un modèle de scoring d'impayé.

Principes de conception
-----------------------
- Une seule source de vérité pour l'esthétique.
- Pas de répétition.
- Rien n'est codé en dur : hauteurs, marges et troncatures sont calculées
  dynamiquement à partir du nombre de modalités et de la longueur des labels.
  Ainsi une variable binaire et une variable à 20 modalités gardent la même
  épaisseur de barre et un aspect cohérent.

Usage
-----
>>> import eda_viz as viz
>>> viz.plot_categorical_distribution(df, "phone_carrier").show()
>>> viz.plot_rate_by_category(df, "channel", "top_unpaid").show()
"""

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 1. Constantes esthétiques — réglées une seule fois, réutilisées partout
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#2563EB",   # bleu — fréquences / volumes
    "warning": "#EA580C",   # orange — valeurs manquantes
    "paid": "#16A34A",      # vert — payé (0)
    "unpaid": "#DC2626",    # rouge — impayé (1)
    "muted": "#94A3B8",     # gris — éléments secondaires
}

# Bordure fine appliquée à toutes les barres : garantit qu'un remplissage
# clair reste visible sur fond blanc, quelle que soit la couleur.
_BAR_LINE = dict(color="rgba(15,23,42,0.45)", width=0.8)


SCALE_RISK = "RdYlGn_r"     # Échelle de couleur réservée au TAUX d'impayé 
SCALE_NEUTRAL = "Blues"     # uniquement pour la heatmap de Cramér (imshow)

FONT_FAMILY = "Inter, Segoe UI, Helvetica, Arial, sans-serif"

# Géométrie des barres horizontales — garantit une épaisseur CONSTANTE
_ROW_HEIGHT = 34       
_CHROME_HEIGHT = 130   
_BARGAP = 0.30         

# Marge gauche dynamique en fonction de la longueur des labels
_CHAR_WIDTH = 7        
_LABEL_MAX_LEN = 34    
_MARGIN_LEFT_MAX = 320 

# Nombre de modalités affichées par défaut. Le titre indique "top N"
_DEFAULT_TOP_N = 20


# ---------------------------------------------------------------------------
# 2. Helpers privés — mutualisés par toutes les fonctions publiques
# ---------------------------------------------------------------------------

def _apply_theme(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Applique l'identité visuelle commune à n'importe quelle figure."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=13, color="#1E293B"),
        title=dict(font=dict(size=17, color="#0F172A"), x=0.02, xanchor="left"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(t=70, b=55, l=70, r=85),
        bargap=_BARGAP,
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def _truncate(label: object, max_len: int = _LABEL_MAX_LEN) -> str:
    """Tronque un label trop long en gardant le texte complet pour le survol."""
    text = str(label)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _dynamic_height(n_bars: int) -> int:
    """Hauteur proportionnelle au nombre de barres -> épaisseur constante."""
    return int(_CHROME_HEIGHT + n_bars * _ROW_HEIGHT)


def _dynamic_left_margin(labels: Iterable[object]) -> int:
    """Marge gauche calculée d'après le label affiché le plus long."""
    longest = max((len(_truncate(lab)) for lab in labels), default=8)
    return int(min(longest * _CHAR_WIDTH + 24, _MARGIN_LEFT_MAX))


def compute_rate(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    *,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Agrège le taux d'événement positif (impayé) par modalité.

    Retourne un DataFrame : group_col, total, positifs, taux (%).
    ``min_count`` écarte les modalités à faible effectif (taux bruités).
    Cette fonction est l'unique endroit où le taux est calculé.
    """
    stats = (
        df.groupby(group_col, observed=True)[target_col]
        .agg(total="count", positifs="sum")
        .assign(taux=lambda d: (d["positifs"] / d["total"] * 100).round(1))
        .reset_index()
    )
    if min_count > 1:
        stats = stats[stats["total"] >= min_count]
    return stats


def _resample_rate(
    df: pd.DataFrame,
    date_col: str,
    target_col: str | None,
    freq: str,
) -> pd.DataFrame:
    """Agrège volume (et taux si target fourni) sur une fréquence temporelle."""
    grouper = df.set_index(date_col).resample(freq)
    if target_col is None:
        out = grouper.size().reset_index(name="total")
    else:
        out = (
            grouper[target_col]
            .agg(total="count", positifs="sum")
            .assign(taux=lambda d: (d["positifs"] / d["total"] * 100).round(1))
            .reset_index()
        )
    out["periode"] = out[date_col].dt.strftime("%b %Y")
    return out


def _horizontal_bar(
    data: pd.DataFrame,
    *,
    value_col: str,
    label_col: str,
    title: str,
    x_title: str,
    solid_color: str | None = None,
    color_scale: str | None = None,
    hover_cols: Sequence[str] | None = None,
    order: Sequence[object] | None = None,
) -> go.Figure:
    """
    Builder unique de toutes les barres horizontales (distribution, taux,
    manquants). Gère la troncature des labels, la hauteur et la marge
    dynamiques, l'ordre des catégories et la coloration.

    Coloration (exclusive) :
      - ``solid_color`` -> remplissage uni (distributions, manquants : le
        degradé n'encodait rien et masquait les petites barres).
      - ``color_scale``  -> degradé par valeur (taux d'impayé : rouge = risque).
    Une bordure fine (``_BAR_LINE``) est appliquée dans tous les cas pour
    que même un remplissage clair reste lisible sur fond blanc.

    ``order=None``  -> tri par valeur croissante (plus grande barre en haut).
    ``order=[...]`` -> ordre imposé (variables ordinales, ex. ancienneté).
    """
    data = data.copy()
    data["_label"] = data[label_col].map(_truncate)

    bar_kwargs = dict(
        x=value_col,
        y="_label",
        orientation="h",
        text=value_col,
        title=title,
        labels={value_col: x_title, "_label": ""},
        custom_data=[label_col, *(hover_cols or [])],
    )
    if color_scale is not None:
        bar_kwargs.update(color=value_col, color_continuous_scale=color_scale)
    else:
        bar_kwargs["color_discrete_sequence"] = [solid_color or COLORS["primary"]]

    fig = px.bar(data, bar_kwargs)

    # Étiquettes de barre + bordure + survol enrichi (label complet + effectifs)
    hover_extra = "".join(
        f"<br>{col}: %{{customdata[{i + 1}]}}"
        for i, col in enumerate(hover_cols or [])
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
        marker_line=_BAR_LINE,
        hovertemplate="<b>%{customdata[0]}</b><br>"
        + f"{x_title}: %{{x:.1f}}%" + hover_extra + "<extra></extra>",
    )

    if order is None:
        fig.update_yaxes(categoryorder="total ascending")
    else:
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=[_truncate(o) for o in order],
        )

    fig = _apply_theme(fig, height=_dynamic_height(len(data)))
    fig.update_layout(margin_l=_dynamic_left_margin(data["_label"]))
    return fig


_HIGH_CARDINALITY = 10   # au-delà, le titre indique "top N (X au total)"

def _capped_title(base: str, col: str, n_total: int, top_n: int) -> tuple[str, bool]:
    """
    Titre + indicateur de troncature.

    Si ``n_total > _HIGH_CARDINALITY`` : affiche "top N (X catégories au total)"
    et tronque les données à top_n. Sinon : titre simple.
    """
    if n_total > _HIGH_CARDINALITY:
        return f"{base} {col} : top {top_n} ({n_total} catégories au total)", True
    return f"{base} {col}", False


# ---------------------------------------------------------------------------
# 3. Distributions (Partie I)
# ---------------------------------------------------------------------------

def plot_target_balance(
    df: pd.DataFrame,
    target_col: str,
    *,
    labels: Mapping[object, str] | None = None,
) -> go.Figure:
    """Équilibre des classes de la variable cible (barres verticales + %)."""
    labels = labels or {0: "Classe 0", 1: "Classe 1"}
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, "count"]
    counts["label"] = counts[target_col].map(labels)
    counts["pct"] = (counts["count"] / len(df) * 100).round(1)

    color_map = {labels.get(0, "0"): COLORS["paid"],
                 labels.get(1, "1"): COLORS["unpaid"]}

    fig = px.bar(
        counts,
        x="label",
        y="count",
        color="label",
        text=counts["pct"].map(lambda v: f"{v:.1f}%"),
        color_discrete_map=color_map,
        title="Équilibre des classes (variable cible)",
        labels={"label": "", "count": "Nombre d'observations"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig = _apply_theme(fig, height=420)
    fig.update_layout(showlegend=False)
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure | None:
    """Pourcentage de valeurs manquantes par variable (barres horizontales)."""
    miss = (
        pd.DataFrame({
            "variable": df.columns,
            "pct": (df.isnull().mean() * 100).round(1).values,
        })
        .query("pct > 0")
        .sort_values("pct")
    )
    if miss.empty:
        print("Aucune valeur manquante dans le jeu de données.")
        return None
    return _horizontal_bar(
        miss,
        value_col="pct",
        label_col="variable",
        title="Valeurs manquantes par variable",
        x_title="% de valeurs manquantes",
        solid_color=COLORS["warning"],
    )


def plot_numeric_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    """Histogramme + boxplot marginal d'une variable numérique continue."""
    fig = px.histogram(
        df,
        x=col,
        nbins=40,
        marginal="box",
        title=f"Distribution de {col}",
        labels={col: col},
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig = _apply_theme(fig, height=460)
    fig.update_layout(bargap=0.05, yaxis_title="Fréquence")
    return fig


def plot_categorical_distribution(
    df: pd.DataFrame,
    col: str,
    *,
    top_n: int = _DEFAULT_TOP_N,
) -> go.Figure:
    """
    Fréquence (%) de chaque modalité d'une variable catégorielle.
    Tronque automatiquement au top-N si la cardinalité est élevée ;
    garde une épaisseur de barre constante pour les variables binaires.
    """
    n_unique = df[col].nunique()
    title, truncate = _capped_title("Distribution de", col, n_unique, top_n)

    counts = (
        df[col].value_counts(normalize=True).mul(100).round(1)
        .reset_index()
    )
    counts.columns = [col, "proportion"]
    if truncate:
        counts = counts.head(top_n)

    return _horizontal_bar(
        counts,
        value_col="proportion",
        label_col=col,
        title=title,
        x_title="Proportion (%)",
        solid_color=COLORS["primary"],
    )


def plot_volume_over_time(
    df: pd.DataFrame,
    date_col: str,
    *,
    freq: str = "ME",
) -> go.Figure:
    """Volume de souscriptions agrégé par période (courbe)."""
    ts = _resample_rate(df, date_col, target_col=None, freq=freq)
    fig = px.line(
        ts,
        x="periode",
        y="total",
        markers=True,
        text="total",
        title="Volume de souscriptions par mois",
        labels={"periode": "Mois", "total": "Nombre de souscriptions"},
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig.update_traces(textposition="top center")
    return _apply_theme(fig, height=430)


def plot_rate_over_time(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    *,
    freq: str = "ME",
) -> go.Figure:
    """Taux d'impayé agrégé par période de souscription (courbe)."""
    ts = _resample_rate(df, date_col, target_col, freq=freq)
    fig = px.line(
        ts,
        x="periode",
        y="taux",
        markers=True,
        text="taux",
        title="Taux d'impayé par mois de souscription",
        labels={"periode": "Mois", "taux": "Taux d'impayé (%)"},
        color_discrete_sequence=[COLORS["unpaid"]],
        custom_data=["total", "positifs"],
    )
    fig.update_traces(
        textposition="top center",
        texttemplate="%{text:.1f}%",
        hovertemplate="%{x}<br>Taux: %{y:.1f}%"
        "<br>Effectif: %{customdata[0]}<br>Impayés: %{customdata[1]}"
        "<extra></extra>",
    )
    return _apply_theme(fig, height=430)


# ---------------------------------------------------------------------------
# 4. Relation avec la cible (Partie II)
# ---------------------------------------------------------------------------

def plot_numeric_by_target(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    *,
    labels: Mapping[object, str] | None = None,
) -> go.Figure:
    """Distribution d'une variable numérique selon le statut d'impayé (box)."""
    labels = labels or {0: "Payé", 1: "Impayé"}
    tmp = df[[col, target_col]].copy()
    tmp["statut"] = tmp[target_col].map(labels)
    fig = px.box(
        tmp,
        x="statut",
        y=col,
        color="statut",
        color_discrete_map={labels.get(0, "0"): COLORS["paid"],
                            labels.get(1, "1"): COLORS["unpaid"]},
        title=f"Distribution de {col} par statut d'impayé",
        labels={"statut": "", col: col},
    )
    fig = _apply_theme(fig, height=460)
    fig.update_layout(showlegend=False)
    return fig


def plot_rate_by_category(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    *,
    top_n: int = _DEFAULT_TOP_N,
    min_count: int = 1,
    sort: str = "rate",
) -> go.Figure:
    """
    Taux d'impayé par modalité (barres horizontales colorées par risque).

    Convient aux catégorielles classiques, mais aussi aux variables
    géographiques (``zip_code_prefix``) ou ordinales (tranches d'ancienneté).

    Paramètres
    ----------
    min_count : écarte les modalités sous ce seuil d'effectif (taux fiable).
    top_n     : ne montre que les N plus risquées si la cardinalité est élevée.
    sort      : "rate" -> tri par taux (par défaut) ;
                "category" -> ordre naturel conservé (variables ordinales).
    """
    stats = compute_rate(df, col, target_col, min_count=min_count)
    n_kept = stats[col].nunique()
    if sort == "rate":
        stats = stats.sort_values("taux", ascending=False)
        title, truncate = _capped_title("Taux d'impayé —", col, n_kept, top_n)
        if truncate:
            stats = stats.head(top_n)
        order = None
    else:  # ordre catégoriel imposé (ex. ancienneté)
        order = list(stats[col])
        title = f"Taux d'impayé par {col}"

    return _horizontal_bar(
        stats,
        value_col="taux",
        label_col=col,
        title=title,
        x_title="Taux d'impayé (%)",
        color_scale=SCALE_RISK,
        hover_cols=["total", "positifs"],
        order=order,
    )


def plot_rate_vs_volume(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    *,
    min_count: int = 1,
) -> go.Figure:
    """
    Nuage de points taux d'impayé vs volume par modalité.

    Idéal pour les variables à forte cardinalité (zip_code_prefix) : montre
    si les taux élevés reposent sur des effectifs solides ou sur du bruit.
    Une ligne pointillée marque le taux d'impayé moyen global.
    """
    stats = compute_rate(df, group_col, target_col, min_count=min_count)
    global_rate = df[target_col].mean() * 100

    fig = px.scatter(
        stats,
        x="total",
        y="taux",
        size="total",
        log_x=True,
        hover_name=group_col,
        title=f"Taux d'impayé vs volume par {group_col} (effectif ≥ {min_count})",
        labels={"total": "Volume de contrats (échelle log)",
                "taux": "Taux d'impayé (%)"},
        color="taux",
        color_continuous_scale=SCALE_RISK,
    )
    fig.add_hline(
        y=global_rate,
        line_dash="dash",
        line_color=COLORS["muted"],
        annotation_text=f"Moyenne globale ({global_rate:.1f}%)",
        annotation_position="top left",
    )
    return _apply_theme(fig, height=480)


# ---------------------------------------------------------------------------
# 5. Préparation légère (features dérivées pour l'EDA)
# ---------------------------------------------------------------------------

def add_tenure_bins(
    df: pd.DataFrame,
    date_col: str,
    *,
    bins: Sequence[float] = (0, 3, 6, 9, 12, 18, 24, np.inf),
    out_col: str = "tranche_anciennete",
) -> pd.DataFrame:
    """
    Ajoute une colonne d'ancienneté (en mois, en tranches ordonnées) calculée
    par rapport à la date de référence = date max du jeu de données.

    Les bornes sont paramétrables ; les libellés sont générés dynamiquement
    à partir des bornes (rien n'est codé en dur).
    """
    out = df.copy()
    date_ref = out[date_col].max()
    months = ((date_ref - out[date_col]).dt.days / 30.44).round()

    labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        labels.append(f"{int(lo)}-{int(hi)}m" if np.isfinite(hi) else f"{int(lo)}m+")

    out[out_col] = pd.cut(months, bins=list(bins), labels=labels, right=False)
    return out


# ---------------------------------------------------------------------------
# 6. Colinéarité entre variables catégorielles (V de Cramér)
# ---------------------------------------------------------------------------

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    V de Cramér avec correction de biais de Bergsma-Wicher.
    Mesure l'association entre deux variables catégorielles, dans [0, 1].
    """
    from scipy.stats import chi2_contingency  

    confusion = pd.crosstab(x, y)

    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return np.nan
    
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - (r - 1) * 2 / (n - 1)
    kcorr = k - (k - 1) * 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)

    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def plot_cramers_matrix(df: pd.DataFrame, cat_cols: Sequence[str]) -> go.Figure:
    """Matrice d'association (V de Cramér) entre variables catégorielles."""
    n = len(cat_cols)
    matrix = np.zeros((n, n))
    for i, c1 in enumerate(cat_cols):
        for j, c2 in enumerate(cat_cols):
            matrix[i, j] = 1.0 if i == j else cramers_v(df[c1], df[c2])

    fig = px.imshow(
        np.round(matrix, 2),
        x=list(cat_cols),
        y=list(cat_cols),
        text_auto=True,
        zmin=0,
        zmax=1,
        color_continuous_scale=SCALE_NEUTRAL,
        title="Association entre variables catégorielles (V de Cramér)",
        aspect="auto",
    )
    fig = _apply_theme(fig, height=max(420, n * 60))
    fig.update_layout(coloraxis_showscale=True, margin_l=160)
    return fig


def plot_numeric_correlation(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    method: str = "spearman",
) -> go.Figure:
    """
    Matrice de corrélation entre variables numériques, dans le même style
    que ``plot_cramers_matrix``.

    Spearman par défaut : capture les relations monotones non-linéaires,
    plus robuste que Pearson aux outliers et aux WoE bornés.
    """
    corr = df[list(features)].corr(method=method).round(2)
    n = len(features)

    fig = px.imshow(
        corr,
        x=list(features),
        y=list(features),
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title=f"Corrélation entre variables numériques ({method.capitalize()})",
        aspect="auto",
    )
    fig = _apply_theme(fig, height=max(420, n * 55))
    fig.update_layout(coloraxis_showscale=True, margin_l=200)
    return fig


def cramers_with_target(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    target_col: str,
) -> pd.DataFrame:
    """Classe les catégorielles par force d'association avec la cible."""
    rows = [(c, round(cramers_v(df[c], df[target_col]), 3)) for c in cat_cols]
    return (
        pd.DataFrame(rows, columns=["variable", "cramers_v_cible"])
        .sort_values("cramers_v_cible", ascending=False)
        .reset_index(drop=True)
    )


def compute_woe_iv(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    *,
    n_bins: int = 10,
    min_count: int = 5,
) -> pd.DataFrame:
    """
    Calcule le WoE et l'IV pour une variable (catégorielle ou numérique).

    Pour les numériques : discrétisation en quantiles (n_bins).
    Pour les catégorielles : une modalité = un bin.

    Retourne un DataFrame avec, par bin :
      woe, pct_mauvais, pct_bons, total, taux, iv_bin
    et une colonne 'IV' (scalaire répété) pour faciliter les groupby.
    """
    tmp = df[[col, target_col]].copy().dropna(subset=[col])

    # Discrétisation si numérique
    if pd.api.types.is_numeric_dtype(tmp[col]) and tmp[col].nunique() > n_bins:
        tmp["_bin"] = pd.qcut(tmp[col], q=n_bins, duplicates="drop")
    else:
        tmp["_bin"] = tmp[col].astype(str)

    total_bad = tmp[target_col].sum()
    total_good = len(tmp) - total_bad

    stats = (
        tmp.groupby("_bin", observed=True)[target_col]
        .agg(total="count", mauvais="sum")
        .assign(bons=lambda d: d["total"] - d["mauvais"])
        .query("total >= @min_count")
        .copy()
    )

    # Éviter log(0) : clip à un epsilon
    eps = 0.5
    stats["pct_mauvais"] = (stats["mauvais"] + eps) / (total_bad + eps)
    stats["pct_bons"]    = (stats["bons"]    + eps) / (total_good + eps)
    stats["woe"]         = np.log(stats["pct_mauvais"] / stats["pct_bons"])
    stats["iv_bin"]      = (stats["pct_mauvais"] - stats["pct_bons"]) * stats["woe"]
    stats["taux"]        = (stats["mauvais"] / stats["total"] * 100).round(1)
    stats["IV"]          = stats["iv_bin"].sum()

    return stats.reset_index().rename(columns={"_bin": col})


def iv_summary(
    df: pd.DataFrame,
    cols: list[str],
    target_col: str,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Calcule l'IV de chaque variable et retourne un ranking.
    Point d'entrée principal avant la modélisation.
    """
    rows = []
    for col in cols:
        try:
            iv = compute_woe_iv(df, col, target_col, n_bins=n_bins)["IV"].iloc[0]
            rows.append({"variable": col, "IV": round(iv, 4)})
        except Exception:
            rows.append({"variable": col, "IV": np.nan})

    thresholds = [0.02, 0.1, 0.3, 0.5]
    labels = ["Nul", "Faible", "Moyen", "Fort", "Suspect"]

    result = (
        pd.DataFrame(rows)
        .sort_values("IV", ascending=False)
        .reset_index(drop=True)
    )
    result["pouvoir_predictif"] = pd.cut(
        result["IV"], bins=[0] + thresholds + [np.inf], labels=labels
    )
    return result


def plot_woe(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    *,
    n_bins: int = 10,
) -> go.Figure:
    """
    Graphique WoE par bin : barres colorées (rouge = mauvais payeurs,
    vert = bons payeurs) avec le taux d'impayé en annotation.
    """
    stats = compute_woe_iv(df, col, target_col, n_bins=n_bins)
    iv_val = stats["IV"].iloc[0]
    stats["_label"] = stats[col].map(_truncate)

    colors = [COLORS["unpaid"] if w >= 0 else COLORS["paid"] for w in stats["woe"]]

    fig = go.Figure(go.Bar(
        x=stats["woe"],
        y=stats["_label"],
        orientation="h",
        marker_color=colors,
        marker_line=_BAR_LINE,
        text=stats["taux"].map(lambda v: f"{v:.1f}%"),
        textposition="outside",
        customdata=stats[["total", "mauvais"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>WoE: %{x:.3f}"
            "<br>Taux impayé: %{text}"
            "<br>Effectif: %{customdata[0]}"
            "<br>Impayés: %{customdata[1]}"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color=COLORS["muted"], line_width=1)
    title = f"WoE — {col}  (IV = {iv_val:.3f})"
    fig = _apply_theme(fig, height=_dynamic_height(len(stats)))
    fig.update_layout(
        title=title,
        xaxis_title="Weight of Evidence",
        margin_l=_dynamic_left_margin(stats["_label"]),
    )
    return fig