# Analyse des Concentrations de Planctons Marins

Ce projet de recherche universitaire vise à caractériser les communautés de zooplancton à partir de 37 campagnes océanographiques (2008–2019). L'analyse s'appuie sur le jeu de données public **SEANOE** publié par **Panaïotis et al. (2023)**, comprenant des mesures de concentrations d'organismes et des variables physico-chimiques environnementales.

L'objectif est d'explorer la structure des communautés planctoniques à travers différentes méthodes de réduction de dimension et d'algorithmes de clustering pour identifier des signatures écologiques distinctes.

---

## 🚀 Structure du Projet

L'analyse est découpée en trois jalons séquentiels, chacun correspondant à un notebook Jupyter dédié. Les résultats sont transmis d'un jalon à l'autre via des fichiers de sauvegarde (`.pkl`).

### 1. [Jalon 1] Exploration & Classification Baseline
**Fichier :** `01_exploration_baseline.ipynb`

Ce notebook constitue la fondation du projet. Il traite de la préparation des données et établit une référence (baseline) pour les comparaisons futures.
- **Exploration (EDA) :** Analyse des 4264 observations, gestion des valeurs manquantes et statistiques descriptives sur les groupes taxonomiques (copépodes, appendiculaires, etc.).
- **Prétraitement :** Application de la **transformation de Hellinger**, standard pour les données de concentrations, afin de stabiliser la variance et gérer la forte proportion de zéros.
- **Réduction de dimension :** Mise en œuvre d'une **ACP (Analyse en Composantes Principales)** pour identifier les axes de variance majeurs.
- **Clustering Baseline :** Utilisation de l'algorithme **K-means** avec optimisation du nombre de clusters via la méthode du coude et le score de silhouette.
- **Sortie :** Génère `data/jalon1_outputs.pkl` contenant les données transformées et les résultats de l'ACP.

### 2. [Jalon 2] Nouvelles Représentations de Données
**Fichier :** `02_representations.ipynb`

Ce second volet explore des méthodes de réduction de dimension alternatives, tant linéaires que non-linéaires, pour capturer des structures complexes non détectées par l'ACP.
- **AFC (Analyse Factorielle des Correspondances) :** Étude des relations de contingence entre les échantillons et les taxons.
- **Méthodes Non-Linéaires (Manifold Learning) :**
    - **Isomap :** Conservation des distances géodésiques.
    - **LLE (Locally Linear Embedding) :** Préservation des voisinages locaux.
    - **MDS (Multidimensional Scaling) :** Projection préservant les distances inter-échantillons.
- **Comparaison :** Évaluation visuelle et quantitative de la capacité de chaque méthode à séparer les couches océaniques (Epi, Mesopelagic, etc.).
- **Sortie :** Génère `data/jalon2_outputs.pkl`.

### 3. [Jalon 3] Algorithmes de Clustering Avancés
**Fichier :** `03_clusterings.ipynb`

Le dernier jalon compare la classification baseline (K-means) à des approches plus sophistiquées, notamment basées sur la densité et les modèles probabilistes.
- **Clustering Spatial/Densité :**
    - **DBSCAN & HDBSCAN :** Identification de clusters de forme arbitraire et gestion automatique du bruit (outliers).
    - **OPTICS :** Analyse de la hiérarchie de densité.
- **Approches Probabilistes & Spectrales :**
    - **GMM (Gaussian Mixture Models) :** Clustering "souple" basé sur des distributions normales.
    - **Spectral Clustering :** Utilisation de la structure de graphe des données.
- **Évaluation :** Comparaison rigoureuse à l'aide des indices de Silhouette, Davies-Bouldin et Calinski-Harabasz.

---

## 🛠️ Installation et Dépendances

Le projet nécessite **Python 3.9+** et les bibliothèques suivantes :

```bash
pip install numpy pandas matplotlib seaborn scikit-learn hdbscan
```

---

## 📊 Méthodologie & Conventions

- **Reproductibilité :** Tous les algorithmes stochastiques utilisent `RANDOM_STATE = 42`.
- **Transformation :** La transformation de Hellinger est systématiquement appliquée avant toute réduction de dimension.
- **Visualisation :** Les graphiques utilisent le style `seaborn-whitegrid` pour une clarté optimale.

---

## 📂 Organisation des Fichiers

```text
.
├── 01_exploration_baseline.ipynb  # Jalon 1 : Base et ACP
├── 02_representations.ipynb      # Jalon 2 : AFC, Isomap, LLE, MDS
├── 03_clusterings.ipynb          # Jalon 3 : DBSCAN, HDBSCAN, GMM
├── data/
│   └── 103673.csv                # Données sources (SEANOE)
├── figures/                      # Visualisations exportées
│   └── fig_*.png
└── README.md                     # Documentation du projet
```

---

## 📖 Citation des données
Panaïotis, T., et al. (2023). *Zooplankton communities' concentrations and environmental variables from 37 oceanographic cruises (2008-2019)*. SEANOE.
