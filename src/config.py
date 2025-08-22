# src/config.py
import os

# --- Configuration Générale ---
# Mettre à False pour une importation complète et un entraînement long.
DEBUG_MODE = False
# Limite de lignes à parser dans les fichiers de dump en mode DEBUG.
DEBUG_LINE_LIMIT = 50000

# --- Configuration des Chemins ---
# Le dossier où se trouvent les dumps SQL téléchargés.
WIKI_DUMPS_PATH = "data"
PAGE_SQL_FILE = "frwiki-latest-page.sql.gz"
PAGELINKS_SQL_FILE = "frwiki-latest-pagelinks.sql.gz"
# Chemins complets
PAGE_DUMP_FULL_PATH = os.path.join(WIKI_DUMPS_PATH, PAGE_SQL_FILE)
PAGELINKS_DUMP_FULL_PATH = os.path.join(WIKI_DUMPS_PATH, PAGELINKS_SQL_FILE)

# Pour les logs d'entraînement et les modèles sauvegardés
LOGS_PATH = "logs"
MODELS_PATH = "models"


# --- Configuration de la Base de Données Neo4j ---
NEO4J_AUTH_ENABLED = False

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


# --- Paramètres pour la stratégie "TOP_PAGES" ---
# "FLAT": Garde les N meilleures pages. Simple mais crée un graphe peu dense.
# "SNOWBALL": Prend un noyau de pages et étend le graphe à leurs voisins.
TOP_PAGES_SELECTION_MODE = "SNOWBALL"

# --- Sous-paramètres pour le mode "SNOWBALL" ---
# Nombre de pages "seed" (le noyau de départ).
SNOWBALL_SEED_COUNT = 300

# Profondeur de l'expansion (1 = voisins directs, 2 = voisins des voisins).
SNOWBALL_DEPTH = 3

# À chaque étape de l'expansion, on ne garde que les N meilleurs voisins (basé sur leur score).
SNOWBALL_NEIGHBOR_LIMIT = 500

# Après l'expansion, supprime toutes les pages ayant moins de X connexions au total.
# Cela nettoie le graphe des "feuilles" et des impasses.
PRUNING_THRESHOLD = 10

# --- Sous-paramètres pour le mode "FLAT" (non utilisé si mode="SNOWBALL") ---
NUM_TOP_PAGES_TO_KEEP = 2500

# Poids pour le calcul du score de notoriété (inchangé)
SCORE_WEIGHT_INDEGREE = 0.4
SCORE_WEIGHT_OUTDEGREE = 0.5
SCORE_WEIGHT_PAGELENGTH = 0.1

# --- Configuration de l'Entraînement ---
TOTAL_TIMESTEPS = 1_500_000


# --- Configuration de Reprise d'Entraînement ---
# Mettre à True pour charger un modèle existant et continuer son entraînement.
# Mettre à False pour commencer un nouvel entraînement de zéro.
RESUME_TRAINING = True

# Nom de base du modèle à charger si RESUME_TRAINING est True.
# Le script trouvera automatiquement la dernière version (ex: "mon_super_modele-3")
# Exemple: MODEL_NAME_TO_RESUME = "nouveau_modele_1"
MODEL_NAME_TO_RESUME = "nouveau_modele_1"

# Chemin du modèle à charger si RESUME_TRAINING est True.
# Peut être le modèle final ou un checkpoint spécifique.
# Exemples:
# MODEL_TO_RESUME_PATH = os.path.join(MODELS_PATH, "wiki_ppo_final.zip")
# MODEL_TO_RESUME_PATH = os.path.join(MODELS_PATH, "checkpoints/wiki_ppo_checkpoint_417792_steps.zip")
MODEL_TO_RESUME_PATH = os.path.join(MODELS_PATH, "wiki_ppo_final.zip")


# --- Configuration du Jeu ---
DEFAULT_MODEL_NAME = "wiki_maskable_ppo.zip"
DEFAULT_START_PAGE = "Intelligence artificielle"
DEFAULT_TARGET_PAGE = "Apprentissage par renforcement"