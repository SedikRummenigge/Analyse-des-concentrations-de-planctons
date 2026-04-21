# Analyse des Concentrations de Planctons Marins

Ce projet universitaire explore la structure des communautés de zooplancton marin à partir de données collectées lors de **37 campagnes océanographiques mondiales** entre 2008 et 2019. L'objectif : utiliser des techniques d'intelligence artificielle pour identifier automatiquement des **provinces biologiques** — des zones de l'océan où les communautés de plancton se ressemblent — et comprendre comment elles s'organisent en réponse aux conditions environnementales.

Les données proviennent du jeu de données public **SEANOE** publié par **Panaïotis et al. (2023)**, reconnu dans la communauté scientifique comme une référence pour l'étude du zooplancton à l'échelle planétaire.

---

## Contexte scientifique

Le plancton représente la base de la chaîne alimentaire marine et joue un rôle clé dans le cycle du carbone à l'échelle planétaire. Comprendre comment ses communautés s'organisent spatialement et temporellement est essentiel pour surveiller la santé des océans et anticiper les effets du changement climatique.

Le défi : les données de plancton sont **complexes, volumineuses, et multidimensionnelles**. Chaque prélèvement est décrit par des dizaines de variables (concentrations d'espèces, température, profondeur, salinité…), ce qui rend impossible toute analyse visuelle directe. Ce projet applique des méthodes de *machine learning non supervisé* pour extraire automatiquement des structures cohérentes de ces données.

---

## Le jeu de données

**Source :** `data/103673.csv` — Panaïotis et al. (2023), SEANOE

Le fichier contient **4 264 observations** (stations de prélèvement), chacune décrite par trois familles de variables :

| Catégorie | Nombre de variables | Exemples |
|---|---|---|
| Contextuelles | 7 | Latitude, longitude, date, couche océanique (`layer`), heure (jour/nuit) |
| Concentrations biologiques | 28 | Acantharea, Copepoda, Collodaria, Trichodesmium… |
| Variables environnementales | 23 | Température, salinité, oxygène dissous, chlorophylle de surface (`chla_surf`), coefficient d'atténuation (`kd490`)… |

**Points techniques importants :**
- La colonne `prod` (booléen TRUE/FALSE) indique si la station appartient à la couche de production (zone photique).
- La colonne `layer` distingue les couches océaniques : épipélagique (`epi`), mésopélagique (`meso`), etc.
- Certaines colonnes contiennent des valeurs manquantes (`NA`), notamment `strat` et `chla_surf`.
- Les dates s'étendent de juin 2008 à 2019, couvrant plusieurs saisons et hémisphères.

---

## Structure du projet — Les trois jalons

L'analyse est découpée en trois jalons séquentiels. Chaque jalon produit un fichier de sauvegarde (`.pkl`) qui transmet les résultats au suivant, garantissant la cohérence et la reproductibilité de l'ensemble de la chaîne d'analyse.

---

### Jalon 1 — Exploration & Classification Baseline
**Fichier :** `01_exploration_baseline.ipynb`  
**Sortie :** `data/jalon1_outputs.pkl`

Ce premier jalon pose les fondations de tout le projet : préparer les données correctement et établir une référence de comparaison (*baseline*).

#### 1.1 Exploration des données (EDA)
Avant de modéliser quoi que ce soit, on commence par examiner les données brutes : distributions des concentrations par taxon, présence de valeurs aberrantes, variables manquantes, et corrélations entre espèces. Cette étape révèle immédiatement un problème central : les concentrations sont **extrêmement déséquilibrées**. Quelques espèces très abondantes (comme les Copepoda) dominent les chiffres, tandis que des dizaines d'autres espèces affichent des valeurs proches de zéro.

#### 1.2 Transformation de Hellinger
Pour corriger ce déséquilibre, on applique la **transformation de Hellinger**. La formule est :

```
y'_ij = sqrt( y_ij / y_i+ )
```

Où `y_ij` est la concentration de l'espèce `j` dans la station `i`, et `y_i+` est la somme totale des concentrations de cette station. En pratique, on convertit chaque ligne en un *profil de présence relative* (une proportion), puis on prend la racine carrée pour atténuer l'influence des valeurs extrêmes.

**Résultat concret :** après transformation, la somme des carrés de chaque ligne vaut 1. Deux stations prélevées dans des contextes très différents en termes de biomasse totale, mais avec une composition d'espèces similaire, seront désormais reconnues comme proches par les algorithmes.

#### 1.3 Réduction de dimension — ACP
L'**Analyse en Composantes Principales (ACP)** compresse les 19 dimensions biologiques en un nombre réduit d'axes synthétiques. Chaque axe (composante principale) est une combinaison linéaire des espèces originales qui capture le maximum de variance restante.

On retient le nombre de composantes nécessaires pour expliquer **au moins 80 % de la variance totale** (variable `n_80`). Ce seuil garantit que l'information essentielle est préservée tout en réduisant le bruit. Les axes suivants sont ensuite utilisés comme entrée pour le clustering.

#### 1.4 Clustering Baseline — K-Means
Le **K-Means** partitionne les stations en `k` groupes en minimisant la distance entre chaque point et le centre de son groupe (*centroïde*). L'algorithme est rapide et interprétable, mais il impose deux contraintes fortes : les groupes doivent être **sphériques** (même forme dans toutes les directions) et leur **nombre doit être fixé à l'avance**.

Pour choisir le bon `k`, on teste plusieurs valeurs et on retient celle qui maximise le **score de silhouette** (`best_sil_k`) — un indicateur qui mesure à la fois la cohésion interne d'un groupe et sa séparation par rapport aux autres. Un score proche de 1 indique des groupes bien définis ; proche de 0, des groupes qui se chevauchent.

---

### Jalon 2 — Nouvelles Représentations de Données
**Fichier :** `02_representations.ipynb`  
**Sortie :** `data/jalon2_outputs.pkl`

L'ACP est une méthode **linéaire** : elle ne peut détecter que des structures en lignes droites dans les données. Ce second jalon explore des méthodes capables de capturer des formes plus complexes.

#### 2.1 AFC — Analyse Factorielle des Correspondances
L'**AFC** est l'équivalent de l'ACP pour les données de contingence (comptages). Elle analyse les *profils de lignes* et de *colonnes* simultanément, ce qui permet de positionner les stations et les espèces dans le même espace de représentation. Les coordonnées des individus (`F_afc`) et des variables (`G_afc`) permettent d'identifier quelles espèces sont caractéristiques de quels types d'échantillons.

#### 2.2 Isomap
**Isomap** (Isometric Mapping) est une méthode de *manifold learning* : elle cherche à déplier la structure courbe des données comme on déroulerait une feuille de papier enroulée. Plutôt que de calculer des distances en ligne droite (comme l'ACP), elle calcule des **distances géodésiques** — c'est-à-dire des distances mesurées le long de la surface réelle des données. On optimise le paramètre `n_neighbors` (nombre de voisins utilisés pour construire le graphe de proximité) via une validation croisée sur l'erreur de reconstruction.

#### 2.3 LLE — Locally Linear Embedding
La **LLE** adopte une approche locale : chaque point est approximé comme une combinaison linéaire de ses voisins immédiats. L'algorithme cherche ensuite une projection en basse dimension qui préserve ces reconstructions locales. Contrairement à Isomap, elle ne se base pas sur des distances globales, ce qui la rend plus robuste aux données en forme de ruban ou de tube.

#### 2.4 MDS — Multidimensional Scaling
Le **MDS** (métrique et non-métrique) cherche à projeter les données en 2D en préservant au mieux les distances inter-points calculées dans l'espace d'origine. La version *métrique* préserve les distances exactes ; la version *non-métrique* préserve uniquement leur ordre (rang). Pour des raisons de temps de calcul, le MDS est appliqué sur un sous-échantillon de `N_MDS = 2000` stations.

#### 2.5 Bilan comparatif
Chaque méthode est évaluée via son **score de silhouette 2D** (`sil_pca`, `sil_afc`, `sil_isomap`, `sil_lle`, `sil_mds`). Ces scores permettent de comparer objectivement la qualité de séparation obtenue avec chaque représentation, indépendamment de l'algorithme de clustering utilisé par la suite.

---

### Jalon 3 — Algorithmes de Clustering Avancés
**Fichier :** `03_clusterings.ipynb`

Ce dernier jalon compare la baseline K-Means à des méthodes plus sophistiquées, en utilisant les meilleures représentations identifiées au Jalon 2.

#### 3.1 DBSCAN & HDBSCAN — Clustering par densité
**DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) identifie les clusters comme des zones denses dans l'espace des données. Il n'impose ni la forme ni le nombre de groupes : un cluster peut être allongé, courbe, ou de n'importe quelle taille. Les points situés dans des zones peu denses sont automatiquement étiquetés comme **bruit** (outliers) — ce qui est une information utile en soi, car ces points correspondent souvent à des stations de transition ou des événements écologiques ponctuels.

**HDBSCAN** est une extension hiérarchique de DBSCAN : il construit d'abord une hiérarchie de clusters selon la densité, puis sélectionne automatiquement le niveau de découpe optimal. Il est plus robuste aux variations de densité locale.

#### 3.2 OPTICS — Hiérarchie de densité
**OPTICS** génère un profil ordonné de la structure de densité des données (le *reachability plot*), permettant d'explorer la hiérarchie des clusters à différentes résolutions sans avoir à fixer de seuil fixe au préalable.

#### 3.3 Spectral Clustering
Le **Spectral Clustering** transforme le problème de partitionnement en un problème de coupure de graphe. On construit un graphe où chaque station est un nœud et les arêtes sont pondérées par la similarité entre stations. L'algorithme analyse la structure spectrale (valeurs propres) de ce graphe pour trouver des groupes bien connectés en interne et bien séparés entre eux. Il est particulièrement efficace pour des clusters non convexes.

#### 3.4 GMM — Gaussian Mixture Models
Le **GMM** modélise les données comme un mélange de distributions gaussiennes (des "cloches" multidimensionnelles). Chaque cluster est décrit par un centre (moyenne) et une forme (matrice de covariance), ce qui permet des groupes elliptiques de tailles et d'orientations différentes. L'apprentissage se fait via l'algorithme **EM (Expectation-Maximization)** : on alterne entre l'estimation des probabilités d'appartenance de chaque point à chaque gaussienne (E-step), et la mise à jour des paramètres des gaussiennes en fonction de ces probabilités (M-step).

Le GMM produit des **assignations souples** : chaque station reçoit une probabilité d'appartenir à chacun des groupes, plutôt qu'une étiquette rigide. Une station à la frontière de deux groupes peut, par exemple, appartenir à 65 % au groupe A et 35 % au groupe B — ce qui reflète fidèlement la réalité des zones de transition marine.

#### 3.5 Évaluation et comparaison
Tous les algorithmes sont évalués sur trois indices complémentaires :

| Indice | Ce qu'il mesure | Idéal |
|---|---|---|
| **Silhouette** | Cohésion interne + séparation externe | Proche de 1 |
| **Davies-Bouldin** | Rapport entre dispersion interne et distance inter-clusters | Le plus bas possible |
| **Calinski-Harabasz** | Rapport variance inter-cluster / intra-cluster | Le plus élevé possible |

L'utilisation conjointe de ces trois indices permet d'éviter les biais propres à chacun, et d'identifier le modèle le plus robuste sur l'ensemble des critères.

---

## Installation et dépendances

Le projet nécessite **Python 3.9+**. Toutes les bibliothèques nécessaires s'installent via :

```bash
pip install numpy pandas matplotlib seaborn scikit-learn hdbscan
```

| Bibliothèque | Rôle |
|---|---|
| `numpy` / `pandas` | Manipulation des données tabulaires |
| `matplotlib` / `seaborn` | Visualisation des résultats |
| `scikit-learn` | ACP, K-Means, DBSCAN, Isomap, LLE, MDS, GMM, Spectral Clustering, métriques d'évaluation |
| `hdbscan` | Implémentation optimisée de HDBSCAN |

---

## Conventions et reproductibilité

- **Graine aléatoire :** `RANDOM_STATE = 42` — utilisé dans tous les algorithmes stochastiques pour garantir la reproductibilité exacte des résultats.
- **Transformation systématique :** la transformation de Hellinger est appliquée avant toute réduction de dimension ou clustering.
- **Clustering sur ACP :** le clustering est réalisé sur `X_pca[:, :n_80]` — les composantes principales retenues pour expliquer 80 % de la variance.
- **Figures exportées :** toutes les visualisations sont sauvegardées en PNG dans le dossier `figures/`, nommées `fig_01` à `fig_24` selon l'ordre d'apparition dans les notebooks.

---

## Organisation des fichiers

```text
.
├── 01_exploration_baseline.ipynb   # Jalon 1 : EDA, Hellinger, ACP, K-Means
├── 02_representations.ipynb        # Jalon 2 : AFC, Isomap, LLE, MDS
├── 03_clusterings.ipynb            # Jalon 3 : DBSCAN, HDBSCAN, GMM, Spectral
├── analyse_planctons.ipynb         # Notebook principal — livrable unique consolidé
├── presentation_detaillee_planctons.html  # Présentation interactive (Reveal.js)
├── data/
│   ├── 103673.csv                  # Données sources (SEANOE)
│   ├── jalon1_outputs.pkl          # Sorties du Jalon 1 (données transformées, ACP)
│   └── jalon2_outputs.pkl          # Sorties du Jalon 2 (représentations alternatives)
├── figures/
│   └── fig_*.png                   # Visualisations exportées (fig_01 à fig_24)
└── README.md
```

---

## Citation des données

> Panaïotis, T., et al. (2023). *Zooplankton communities' concentrations and environmental variables from 37 oceanographic cruises (2008–2019)*. SEANOE. [https://doi.org/10.17882/103673](https://doi.org/10.17882/103673)
