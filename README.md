# Wiki AI

🧠 Un agent basé sur l'Apprentissage par Renforcement (Reinforcement Learning) qui apprend à jouer au [Jeu Wikipédia](https://fr.wikipedia.org/wiki/Wikip%C3%A9dia:Le_plus_court_chemin) : trouver le chemin le plus court entre deux articles en cliquant sur les liens internes.

Ce projet met en œuvre un pipeline complet, de la construction d'un graphe de connaissance intelligent à partir des données brutes de Wikipédia, jusqu'à l'entraînement d'un agent capable de naviguer de manière optimale dans ce graphe.

## ✨ Fonctionnalités

-   **Pipeline de Données Intelligent :** Un importateur avancé qui analyse l'intégralité du dump Wikipédia pour calculer un "score de notoriété" (basé sur les liens entrants, sortants et la taille de l'article) afin de construire un sous-graphe "bac à sable" dense et pertinent pour l'entraînement.
-   **Génération de Graphe "Boule de Neige" :** Pour garantir un terrain de jeu riche, le graphe est généré en sélectionnant un noyau de pages très importantes, puis en explorant leurs voisins les plus pertinents sur plusieurs degrés, avant de nettoyer les pages sans issue.
-   **Apprentissage par Renforcement Guidé par le Graphe :** L'agent utilise l'algorithme `MaskablePPO` et est entraîné avec une fonction de récompense "GPS". À chaque clic, il est récompensé ou pénalisé en fonction de la variation de la distance *réelle* la plus courte le séparant de la cible, l'incitant à une efficacité maximale.
-   **Générateur de Missions Curatées :** Avant l'entraînement, un script analyse le graphe pour générer des milliers de missions (départ, cible, distance) garanties comme étant solvables, assurant un apprentissage stable.
-   **Observation Contextuelle :** Pour prendre ses décisions, l'IA dispose d'une "mémoire à court terme", observant à la fois sa position actuelle, sa destination finale et la page d'où elle vient pour éviter les cycles.
-   **Masquage d'Actions (Action Masking) :** L'agent ne peut choisir que parmi les liens valides sur une page, ce qui rend l'entraînement beaucoup plus stable et efficace.
-   **Entraînement Parallélisé :** Le script d'entraînement est conçu pour utiliser tous les cœurs CPU disponibles, accélérant massivement le processus d'apprentissage.
-   **Interface Terminal (TUI) Prévue :** L'architecture est prête à être connectée à une interface en mode texte construite avec `Textual` pour visualiser l'IA en action.

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
    Phase 2: Génération & Entraînement V
+---------------------------+     +---------------------------+
| scripts/00_generate...py  |---->|      missions.json        |
| (Analyse des chemins)     |     | (Pool de missions valides)|
+---------------------------+     +-------------^-------------+
                                                |
+---------------------------+     +-------------V-------------+     +-------------------+
|  scripts/02_train_agent.py|<--->|    src/environment.py     |<--->| Modèle Entraîné   |
| (Entraînement parallèle)  |     | (Simulateur "GPS", Masking) |     | (wiki_ppo_final.zip)|
+---------------------------+     +---------------------------+     +---------V---------+
                                                                              |
                                                                +-------------V-------------+
                                                                |   scripts/03_play.py      |
                                                                |   (Jeu interactif avec IA)|
                                                                +---------------------------+
```

## 🚀 Getting Started

Suivez ces étapes pour mettre en place, entraîner et utiliser le projet.

### Prérequis

-   [Python 3.10+](https://www.python.org/)
-   [Docker](https://www.docker.com/products/docker-desktop/) et Docker Compose

### Installation

1.  **Clonez le dépôt :**
    ```bash
    git clone <votre-url-de-repo>
    cd WikiAI
    ```

2.  **Créez un environnement virtuel et activez-le :**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Installez les dépendances Python :**
    Créez un fichier `requirements.txt` avec le contenu suivant, puis lancez `pip install -r requirements.txt`.
    ```txt
    stable-baselines3[extra]
    sb3-contrib
    gymnasium
    torch
    sentence-transformers
    neo4j
    tqdm
    textual
    ```

## 🕹️ Utilisation

L'utilisation du projet se fait en 4 étapes séquentielles.

### Étape 0 : Configuration Initiale

Le fichier `src/config.py` est le panneau de contrôle du projet. Vous pouvez y ajuster la stratégie de création du graphe (`SNOWBALL_SEED_COUNT`, `SNOWBALL_DEPTH`, etc.) et le nombre total de pas pour l'entraînement (`TOTAL_TIMESTEPS`).

### Étape 1 : Création du Graphe de Connaissance

1.  **Lancez la base de données Neo4j :**
    ```bash
    docker-compose up -d
    ```

2.  **Téléchargez les dumps officiels de Wikipédia France** dans un dossier `data/` à la racine du projet.
    ```bash
    mkdir data
    cd data
    wget https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-page.sql.gz
    wget https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pagelinks.sql.gz
    cd ..
    ```

3.  **Lancez le script d'importation intelligent :**
    Ce script va analyser les gigaoctets de données, calculer les scores, et construire le graphe "bac à sable" dans Neo4j.
    ```bash
    python scripts/01_import_data.py
    ```

### Étape 2 : Génération des Missions d'Entraînement

Une fois le graphe créé, ce script va l'explorer pour trouver des milliers de chemins solvables qui serviront de base à l'entraînement de l'IA.

```bash
python scripts/00_generate_missions.py
```
Cette commande crée un fichier `missions.json` à la racine du projet.

### Étape 3 : Entraînement de l'IA

1.  **Lancez l'entraînement :**
    Cette commande va lancer l'entraînement sur tous vos cœurs CPU. L'avertissement `tokenizers` est normal et géré.
    ```bash
    TOKENIZERS_PARALLELISM=false python scripts/02_train_agent.py
    ```
    Un modèle `wiki_ppo_final.zip` sera créé dans le dossier `models/`.

2.  **(Optionnel) Suivez la progression :**
    Pendant l'entraînement, ouvrez un second terminal et lancez TensorBoard pour visualiser les courbes d'apprentissage (notamment la récompense moyenne, `ep_rew_mean`).
    ```bash
    tensorboard --logdir=logs/
    ```

### Étape 4 : Jouer avec l'IA (Prochaine étape)

1.  **Créez le script `scripts/03_play.py`**.
2.  Dans ce script, chargez le modèle entraîné (`MaskablePPO.load("models/wiki_ppo_final.zip")`).
3.  Créez une instance de `WikiEnv` et lancez une boucle de jeu interactive où le modèle choisit les actions.
4.  Utilisez `Textual` pour créer une interface utilisateur affichant le chemin, les actions possibles et la décision de l'IA à chaque étape.

## 🛠️ Stack Technique

-   **Langage :** Python 3.12
-   **Apprentissage par Renforcement :** Stable-Baselines3-Contrib (`MaskablePPO`)
-   **Deep Learning :** PyTorch
-   **NLP / Vecteurs Sémantiques :** Sentence-Transformers (Hugging Face)
-   **Base de Données Graphe :** Neo4j (via Docker)
-   **Interface Terminal :** Textual
-   **Manipulation de Données :** TQDM

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
│   └── 03_play.py (à créer)
├── src/                      # Code source du projet (modules)
│   ├── __init__.py
│   ├── config.py             # Fichier de configuration central
│   └── environment.py        # L'environnement de jeu Gymnasium (logique "GPS")
├── .gitignore
├── docker-compose.yml        # Configuration pour lancer Neo4j
├── missions.json             # Fichier de missions généré
├── README.md                 # Ce fichier
└── requirements.txt          # Dépendances Python
```

## 💡 Améliorations Possibles

-   **Interface Web :** Remplacer l'interface Textual par une interface web (avec Flask ou FastAPI) pour une meilleure visualisation.
-   **Optimiser l'Observation :** Enrichir le vecteur d'observation avec des données structurelles du graphe (ex: popularité des N meilleurs voisins) en plus des données sémantiques.
-   **Utiliser des Graph Neural Networks (GNN) :** Pour une IA qui apprendrait directement de la topologie du graphe, potentiellement plus performante mais plus complexe à mettre en œuvre.

## 📄 Licence

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.