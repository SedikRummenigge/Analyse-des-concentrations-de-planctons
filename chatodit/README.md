# Analyse des Concentrations de Planctons Marins

Projet universitaire — Master 1 DCI, Université Paris Cité
Analyse de données de zooplancton issues de 37 campagnes océanographiques (2008–2019).

---

## Contexte

Ce projet s'appuie sur les données publiées par Panaïotis et al. (2023) dans la base SEANOE.
L'objectif est d'identifier et de caractériser des communautés planctoniques à partir de
concentrations mesurées par imagerie in situ (UVP5), en combinant transformations statistiques,
réductions de dimensionnalité et algorithmes de clustering.

---

## Structure du projet

```
.
├── data/
│   └── 103673.csv             # Données source (SEANOE, Panaïotis et al. 2023)
├── 01_exploration_baseline.ipynb   # Jalon 1 — Exploration & K-means
├── 02_representations.ipynb        # Jalon 2 — AFC, Isomap, LLE, MDS
├── 03_clusterings.ipynb            # Jalon 3 — DBSCAN, HDBSCAN, OPTICS, Spectral, GMM
└── analyse_planctons.ipynb         # Notebook monolithique original (archive)
```

---

## Notebooks

Les trois notebooks sont **autonomes et séquentiels**. Ils doivent être exécutés dans l'ordre.

### Jalon 1 — `01_exploration_baseline.ipynb`
- Chargement et exploration des données (58 variables, ~20 000 observations)
- Transformation de Hellinger pour stabiliser les données de concentration
- Analyse en Composantes Principales (ACP)
- Classification baseline par K-means (k optimal par score de silhouette)
- **Génère** `data/jalon1_outputs.pkl`

### Jalon 2 — `02_representations.ipynb`
- **Requiert** `data/jalon1_outputs.pkl`
- Analyse Factorielle des Correspondances (AFC)
- Réduction non-linéaire : Isomap et LLE (Locally Linear Embedding)
- Multidimensional Scaling métrique et non-métrique (MDS)
- Comparaison des cinq représentations par score de silhouette
- **Génère** `data/jalon2_outputs.pkl`

### Jalon 3 — `03_clusterings.ipynb`
- **Requiert** `data/jalon1_outputs.pkl`
- DBSCAN, HDBSCAN, OPTICS (clustering par densité)
- Spectral Clustering et Gaussian Mixture Models (GMM)
- Comparaison globale de toutes les méthodes de clustering

---

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn hdbscan
```

> Python 3.9+ recommandé.

---

## Données

| Champ | Détail |
|---|---|
| Source | SEANOE — Panaïotis et al. (2023) |
| Fichier | `data/103673.csv` |
| Observations | ~20 000 profils verticaux |
| Campagnes | 37 (2008–2019), océans Atlantique, Pacifique, Indien |
| Variables contextuelles | 7 (position, date, couche, cycle jour/nuit) |
| Variables de concentration | 28 groupes taxonomiques (Acantharea → Trichodesmium) |
| Variables environnementales | 23 (température, salinité, chlorophylle, etc.) |

---

## Référence

Panaïotis, T. et al. (2023). Content-aware segmentation of objects spanning a large size range :
application to plankton images. *Frontiers in Marine Science*, 10.
https://doi.org/10.3389/fmars.2023.1280510
