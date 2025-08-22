# Wiki AI

🧠 Un agent basé sur l'Apprentissage par Renforcement (Reinforcement Learning) qui apprend à jouer au [Jeu Wikipédia](https://fr.wikipedia.org/wiki/Wikip%C3%A9dia:Le_plus_court_chemin) : trouver le chemin le plus court entre deux articles en cliquant sur les liens internes.

Ce projet met en œuvre un pipeline complet, de la construction d'un graphe de connaissance intelligent à partir des données brutes de Wikipédia, jusqu'à l'entraînement d'un agent capable de naviguer de manière optimale dans ce graphe, et enfin à sa visualisation dans une interface en ligne de commande.

## ✨ Fonctionnalités

-   **Pipeline de Données Intelligent :** Un importateur avancé qui analyse l'intégralité du dump Wikipédia pour calculer un "score de notoriété" (basé sur les liens entrants, sortants et la taille de l'article) afin de construire un sous-graphe "bac à sable" dense et pertinent pour l'entraînement.
-   **Génération de Graphe "Boule de Neige" :** Pour garantir un terrain de jeu riche, le graphe est généré en sélectionnant un noyau de pages très importantes, puis en explorant leurs voisins les plus pertinents sur plusieurs degrés, avant de nettoyer les pages sans issue.
-   **Apprentissage par Renforcement Guidé par le Graphe :** L'agent utilise l'algorithme `MaskablePPO` et est entraîné avec une fonction de récompense "GPS". À chaque clic, il est récompensé ou pénalisé en fonction de la variation de la distance *réelle* la plus courte le séparant de la cible.
-   **Mécanisme Anti-Cycle :** L'environnement d'entraînement **empêche physiquement l'IA de revisiter une page**, la forçant à explorer et à trouver des chemins efficaces sans jamais tomber dans des boucles.
-   **Générateur de Missions par Marche Aléatoire :** Avant l'entraînement, un script explore le graphe en simulant des "promenades aléatoires" pour générer des dizaines de milliers de missions variées et pertinentes.
-   **Entraînement Itératif et Versioning :** Le script d'entraînement **gère automatiquement le versioning des modèles**, permettant de reprendre un entraînement là où il s'était arrêté et de conserver une généalogie claire des différentes versions de l'IA.
-   **Entraînement Parallélisé :** Le script d'entraînement utilise tous les cœurs CPU disponibles pour accélérer massivement le processus d'apprentissage.
-   **Interface de Jeu en Terminal :** Une interface en ligne de commande simple et robuste pour visualiser l'IA en action, observer son chemin et analyser ses décisions pas à pas.

## 🏛️ Architecture

Le projet est organisé autour d'une préparation de données robuste (hors-ligne) qui alimente une boucle d'entraînement et de jeu.

```
    Phase 1: Préparation des Données
+--------------------------+     +---------------------------+     +-------------------+
|   Dumps Wikipédia      |---->| scripts/01_import_data.py |---->|   Base de Données |
| (pages.sql, pagelinks)   |     | (Score, Snowball, Pruning)|     |   Neo4j (Docker)  |
+--------------------------+     +---------------------------+     +---------^---------+
                                                                             |
                                     +---------------------------------------+
                                     |
    Phase 2: Génération, Entraînement & Jeu
                                     V
+---------------------------+     +---------------------------+
| scripts/00_generate...py  |---->|      missions.json        |
| (Marche Aléatoire)        |     | (Pool de missions variées)|
+---------------------------+     +-------------^-------------+
                                                |
+---------------------------+     +-------------V-------------+     +---------------------+
|  scripts/02_train_agent.py|<--->|    src/environment.py     |<--->|     models/           |
|  (Versioning, Parallèle)  |     | (GPS, Anti-Cycle, Masking)|     | (nouveau_modele_1.zip)|
+---------------------------+     +---------------------------+     +-----------V---------+
                                                                                |
                                                                +---------------V-------------+
                                                                | scripts/03_play_simple.py   |
                                                                | (Jeu en ligne de commande)  |
                                                                +-----------------------------+
```

## 🚀 Getting Started

Suivez ces étapes pour mettre en place, entraîner et utiliser le projet.

### Prérequis

-   [Python 3.10+](https://www.python.org/)
-   [Docker](https://www.docker.com/products/docker-desktop/) et Docker Compose

### Installation

1.  **Clonez le dépôt :** `git clone <votre-url-de-repo>`
2.  **Créez un environnement virtuel :** `python3 -m venv .venv && source .venv/bin/activate`
3.  **Installez les dépendances :** Créez un `requirements.txt` avec le contenu ci-dessous, puis `pip install -r requirements.txt`.
    ```txt
    stable-baselines3[extra]
    sb3-contrib
    gymnasium
    torch
    sentence-transformers
    neo4j
    tqdm
    ```

## 🕹️ Utilisation

### Étape 1 : Création du Graphe de Connaissance

1.  **Lancez Neo4j :** `docker-compose up -d`
2.  **Téléchargez les dumps Wikipédia** dans un dossier `data/`.
3.  **Configurez `src/config.py`** pour ajuster la taille et la densité du graphe (ex: `SNOWBALL_SEED_COUNT`).
4.  **Lancez l'importation :** `python scripts/01_import_data.py`

### Étape 2 : Génération des Missions

Ce script explore le graphe pour créer un fichier `missions.json` qui servira de base à l'entraînement.
```bash
python scripts/00_generate_missions.py
```

### Étape 3 : Entraînement de l'IA

1.  **Configurez l'entraînement** dans `src/config.py` :
    -   Pour un **nouvel entraînement** : `RESUME_TRAINING = False`.
    -   Pour **reprendre** : `RESUME_TRAINING = True` et renseignez `MODEL_NAME_TO_TO_RESUME`.
    -   Ajustez `TOTAL_TIMESTEPS` à votre objectif final.

2.  **Lancez l'entraînement :**
    ```bash
    TOKENIZERS_PARALLELISM=false python scripts/02_train_agent.py
    ```
    Le script gérera automatiquement le nommage des modèles (`nouveau_modele_X.zip` ou `ancien_modele-Y.zip`).

3.  **Suivez la progression** avec TensorBoard : `tensorboard --logdir=logs/`

### Étape 4 : Jouer avec l'IA

Ce script lance une partie dans le terminal, en utilisant le dernier modèle entraîné.
```bash
python scripts/03_play_simple.py
```
Appuyez sur `Entrée` pour faire avancer l'IA pas à pas.

## 🛠️ Stack Technique

-   **Langage :** Python 3.12
-   **Apprentissage par Renforcement :** Stable-Baselines3-Contrib (`MaskablePPO`)
-   **Deep Learning :** PyTorch
-   **NLP / Vecteurs Sémantiques :** Sentence-Transformers
-   **Base de Données Graphe :** Neo4j (via Docker)
-   **Analyse de Données :** TQDM

## 📂 Structure du Projet Finale

```
.
├── data/                     # Dumps Wikipédia téléchargés (.sql.gz)
├── logs/                     # Logs d'entraînement pour TensorBoard
├── models/                   # Modèles IA entraînés (.zip)
├── scripts/                  # Scripts exécutables pour chaque étape
│   ├── 00_generate_missions.py
│   ├── 01_import_data.py
│   ├── 02_train_agent.py
│   └── 03_play_simple.py
├── src/                      # Code source du projet (modules)
│   ├── __init__.py
│   ├── config.py             # Fichier de configuration central
│   └── environment.py        # L'environnement de jeu Gymnasium
├── .gitignore
├── docker-compose.yml        # Configuration pour lancer Neo4j
├── missions.json             # Fichier de missions généré
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
└── STATS.md                  # Guide d'interprétation des statistiques
```

## 💡 Améliorations Possibles

-   **Interface Graphique :** Remplacer le jeu en terminal par une interface construite avec **Textual** pour une expérience plus riche et interactive.
-   **Optimiser l'Observation :** Enrichir le vecteur d'observation avec des données structurelles du graphe (ex: popularité des N meilleurs voisins) en plus des données sémantiques.
-   **Utiliser des Graph Neural Networks (GNN) :** Pour une IA qui apprendrait directement de la topologie du graphe, potentiellement plus performante mais plus complexe à mettre en œuvre.

## 📄 Licence

Distribué sous la licence MIT.