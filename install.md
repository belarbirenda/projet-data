# 🔬 Youth Smoking and Drug Analysis

> **Analyse exploratoire et stratégique** des facteurs de risque et de protection liés à la consommation de substances chez les jeunes (2020-2024).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)

---

## 📋 Contexte

Ce projet analyse le dataset "Youth Smoking and Drug" pour comprendre les dynamiques de consommation de substances chez les jeunes. L'approche est **purement exploratoire** : aucun modèle prédictif n'est utilisé.

### Problématique

> Quels sont les facteurs de risque et de protection les plus influents sur la consommation de substances chez les jeunes, et comment ces facteurs varient-ils selon les segments démographiques ?

---

## 🎯 Objectifs

1. **Audit qualité** : Évaluer la qualité des données (missing, doublons, outliers)
2. **Feature engineering** : Créer des indices composites (Risk_Index, Protection_Index)
3. **EDA avancée** : Explorer les distributions, corrélations et tendances temporelles
4. **Clustering** : Segmenter la population via KMeans + PCA
5. **Insights** : Extraire 3-7 insights chiffrés et signaux faibles

---

## 📊 Méthodologie

### Axe 1 : Qualité des Données

![Before/After Cleaning](images/before_after_cleaning.png)

- Suppression des doublons stricts
- Winsorization légère des outliers (1er-99e percentiles)
- Conversion des types (Year → int, catégorielles → category)

### Axe 2 : Analyse Exploratoire

![Correlation Matrix](images/correlation_pearson.png)

- Distributions des outcomes (Smoking_Prevalence, Drug_Experimentation)
- Corrélations Pearson/Spearman
- Évolution temporelle 2020-2024
- Comparaisons par segments (Age, Gender, SES)

### Axe 3 : Clustering Non Supervisé

![PCA Clusters](images/pca_clusters.png)

- Réduction dimensionnelle via PCA (2 composantes)
- KMeans avec sélection automatique du k optimal
- Profilage des clusters

---

## 💡 Insights Clés

### Insight 1 : Influence des pairs

> L'influence des pairs (Peer_Influence) montre la corrélation la plus forte avec les outcomes de consommation.

### Insight 2 : Écart socio-économique

> Les jeunes de statut socio-économique "Low" présentent une prévalence significativement plus élevée.

### Insight 3 : Effet protecteur parental

> La supervision parentale (Parental_Supervision) est le facteur de protection le plus influent.

_Voir `artifacts/insights.json` pour les insights complets._

---

## 📂 Structure du Projet

```text
├── Analyse_Youth_Smoking_Drugs.py   # Notebook principal (format percent)
├── README.md                         # Ce fichier
├── requirements.txt                  # Dépendances Python
├── dataset.csv                       # Dataset source (à fournir)
├── dataset_clean.csv                 # Dataset nettoyé (généré)
│
├── src/                              # Code source modulaire
│   ├── config.py                     # Configuration et constantes
│   ├── io_utils.py                   # Chargement et validation
│   ├── cleaning.py                   # Nettoyage des données
│   ├── features.py                   # Feature engineering
│   ├── eda.py                        # Visualisations EDA
│   ├── clustering.py                 # PCA + KMeans
│   └── insights.py                   # Génération d'insights
│
├── scripts/                          # Scripts exécutables
│   ├── run_pipeline.py               # Pipeline complet
│   └── build_report.py               # Génération rapport PDF/HTML
│
├── images/                           # Visualisations (générées)
│   ├── missing_values.png
│   ├── correlation_pearson.png
│   ├── pca_clusters.png
│   └── ...
│
├── reports/                          # Rapports
│   ├── rapport.md                    # Rapport détaillé (Markdown)
│   └── rapport.html                  # Version HTML (générée)
│
├── artifacts/                        # Artefacts d'analyse
│   ├── insights.json                 # Insights structurés
│   ├── variables_dictionary.csv      # Dictionnaire des variables
│   └── pipeline.log                  # Log d'exécution
│
└── data/
    ├── raw/                          # Données brutes
    └── processed/                    # Données traitées
```

---

## 🚀 Installation et Exécution

### Prérequis

- Python 3.8+
- pip ou conda

### Installation

```bash
# Cloner le repo (ou télécharger)
git clone <repo-url>
cd youth-smoking-analysis

# Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Exécution

1. **Placer le dataset** : Copier `dataset.csv` à la racine du projet (ou dans `data/raw/`)

2. **Lancer le pipeline complet** :
```bash
python scripts/run_pipeline.py
```

3. **Générer le rapport** :
```bash
python scripts/build_report.py
```

### Exécution du Notebook

Le fichier `Analyse_Youth_Smoking_Drugs.py` est un notebook au format "percent" (jupytext). Pour l'exécuter :

```bash
# Avec VS Code + extension Python/Jupyter
# Ouvrir le fichier et exécuter les cellules

# OU convertir en .ipynb
pip install jupytext
jupytext --to notebook Analyse_Youth_Smoking_Drugs.py
jupyter notebook Analyse_Youth_Smoking_Drugs.ipynb
```

---

## 📈 Fichiers Générés

Après exécution de `run_pipeline.py` :

| Fichier                              | Description                          |
| ------------------------------------ | ------------------------------------ |
| `dataset_clean.csv`                  | Dataset nettoyé avec features        |
| `images/*.png`                       | Toutes les visualisations            |
| `artifacts/insights.json`            | Insights au format JSON              |
| `artifacts/variables_dictionary.csv` | Description des variables            |
| `reports/rapport.md`                 | Rapport Markdown détaillé            |
| `reports/rapport.html`               | Rapport HTML (après build_report.py) |

---

## ⚠️ Limites et Biais

1. **Pas de prédiction** : Analyse purement exploratoire, pas de modèle supervisé
2. **Causalité non établie** : Les corrélations ne prouvent pas de lien causal
3. **Biais de déclaration** : Données auto-rapportées potentiellement sous-estimées
4. **Période limitée** : 5 années (2020-2024) de données

---

## 🛠️ Stack Technique

- **Langage** : Python 3.8+
- **Data** : pandas, numpy
- **Visualisation** : matplotlib, seaborn
- **Machine Learning** : scikit-learn (KMeans, PCA, StandardScaler)
- **Rapport** : markdown, jinja2 (optionnel)

---

## 📄 Licence

MIT License - Voir le fichier LICENSE pour plus de détails.

---

## 👥 Auteurs

Projet d'analyse de données - 2024

---


