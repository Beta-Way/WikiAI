# src/environment.py (V2.0 - GPS Reward & Short-Term Memory)
import gymnasium as gym
import numpy as np
from neo4j import GraphDatabase, Driver
from sentence_transformers import SentenceTransformer
import random
import json
from typing import Optional, Any, Tuple, Dict, List

from . import config

MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_SIZE = 384


class WikiEnv(gym.Env):
    """
    Environnement Gymnasium pour le jeu Wikipédia, utilisant une récompense "GPS"
    basée sur la distance la plus courte dans le graphe Neo4j.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        print("Initialisation de l'environnement WikiEnv (Mode GPS)...")
        self.driver: Driver = self._connect_to_neo4j()
        print("Connexion à Neo4j établie.")

        self.model = SentenceTransformer(MODEL_NAME)
        print(f"Modèle SentenceTransformer '{MODEL_NAME}' chargé.")

        # Chargement de toutes les missions possibles depuis le fichier JSON
        with open("missions.json", "r", encoding="utf-8") as f:
            self.missions = json.load(f)
        print(f"{len(self.missions)} missions chargées.")

        # ---- Définition des Espaces d'Action et d'Observation ----
        self.max_actions = 100
        self.action_space = gym.spaces.Discrete(self.max_actions)

        # Espace d'observation :
        # - Vecteur de la page actuelle (VECTOR_SIZE)
        # - Vecteur de la page cible (VECTOR_SIZE)
        # - Vecteur de la page précédente (VECTOR_SIZE) -> LA MÉMOIRE À COURT TERME
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3 * VECTOR_SIZE,),
            dtype=np.float32
        )

        self.max_steps = 25  # Sécurité pour éviter les épisodes infinis

        # --- Variables d'état de l'épisode ---
        self.start_page_title: Optional[str] = None
        self.target_page_title: Optional[str] = None
        self.target_vector: Optional[np.ndarray] = None

        self.current_page_title: Optional[str] = None
        self.previous_page_title: Optional[str] = None
        self.current_step = 0
        self.current_distance_to_target = -1

        self.path: List[str] = []
        self.available_actions: List[str] = []

    def _connect_to_neo4j(self) -> Driver:
        auth = None
        if config.NEO4J_AUTH_ENABLED:
            auth = (config.NEO4J_USER, config.NEO4J_PASSWORD)
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=auth)
        driver.verify_connectivity()
        return driver

    def _get_page_vector(self, title: Optional[str]) -> np.ndarray:
        """Convertit un titre de page en vecteur. Gère le cas où le titre est None."""
        if title is None:
            return np.zeros(VECTOR_SIZE, dtype=np.float32)
        return self.model.encode(title, convert_to_numpy=True)

    def _get_shortest_path_distance(self, start_node: str, end_node: str) -> int:
        """Calcule la distance du plus court chemin entre deux nœuds dans Neo4j."""
        with self.driver.session(database="neo4j") as session:
            result = session.run("""
                MATCH (start:Page {title: $start_title}), (end:Page {title: $end_title})
                MATCH p = shortestPath((start)-[:LINKS_TO*]->(end))
                RETURN length(p) AS distance
            """, start_title=start_node, end_title=end_node)

            record = result.single()
            # S'il n'y a pas de chemin, on retourne une très grande distance comme pénalité
            return record["distance"] if record else self.max_steps * 2

    def _get_observation(self) -> np.ndarray:
        """Construit le vecteur d'observation pour l'état actuel."""
        current_vector = self._get_page_vector(self.current_page_title)
        previous_vector = self._get_page_vector(self.previous_page_title)

        obs = np.concatenate([
            current_vector,
            self.target_vector,
            previous_vector
        ]).astype(np.float32)
        return obs

    def _get_available_actions(self) -> List[str]:
        """Récupère les liens sortants, en garantissant la présence de la cible."""
        with self.driver.session(database="neo4j") as session:
            # On récupère tous les voisins et leur popularité (inDegree)
            result = session.run("""
                MATCH (p:Page {title: $title})-[:LINKS_TO]->(next:Page)
                // Le score était une métrique pour l'import, on utilise inDegree pour le tri
                // car il représente la popularité "actuelle" dans le graphe.
                RETURN next.title AS nextPage, next.score AS score
            """, title=self.current_page_title)

            neighbors = {record["nextPage"]: record["score"] for record in result}

        target_is_neighbor = self.target_page_title in neighbors

        # On trie les voisins par score (popularité)
        # On exclut la cible du tri pour la traiter séparément
        sorted_neighbors = sorted(
            [n for n in neighbors if n != self.target_page_title],
            key=lambda n: neighbors.get(n, 0),
            reverse=True
        )

        # On construit la liste d'actions finales
        final_actions = []
        if target_is_neighbor:
            # La cible est le coup gagnant, on la met en premier !
            final_actions.append(self.target_page_title)

        # On ajoute les autres voisins jusqu'à la limite
        final_actions.extend(sorted_neighbors)

        # On tronque à la taille maximale de l'espace d'action
        return final_actions[:self.max_actions]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Réinitialise l'environnement pour un nouvel épisode avec une nouvelle mission."""
        super().reset(seed=seed)

        # 1. Choisir une nouvelle mission au hasard
        mission = random.choice(self.missions)
        self.start_page_title = mission["start"]
        self.target_page_title = mission["target"]

        # 2. Initialiser l'état interne
        self.target_vector = self._get_page_vector(self.target_page_title)
        self.current_page_title = self.start_page_title
        self.previous_page_title = None  # Pas de page précédente au début
        self.current_step = 0
        self.path = [self.start_page_title]

        # 3. Calculer la distance "GPS" initiale
        self.current_distance_to_target = self._get_shortest_path_distance(
            self.current_page_title, self.target_page_title
        )

        self.available_actions = self._get_available_actions()

        return self._get_observation(), {"action_mask": self.action_mask()}


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Exécute une action dans l'environnement."""
        if action >= len(self.available_actions):
            reward = -10.0
            return self._get_observation(), reward, False, True, {"action_mask": self.action_mask()}

        # --- 1. Transition d'état ---
        next_page_title = self.available_actions[action]
        self.previous_page_title = self.current_page_title
        self.current_page_title = next_page_title
        self.path.append(self.current_page_title)
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.max_steps

        # On vérifie si l'agent a gagné AVANT de calculer la distance
        if self.current_page_title == self.target_page_title:
            # L'agent a gagné ! La nouvelle distance est 0.
            new_distance = 0
            terminated = True

            # Récompense de base (se rapprocher) + Bonus de victoire !
            reward = float(self.current_distance_to_target - new_distance) + 20.0

            # Mise à jour de la distance pour la cohérence
            self.current_distance_to_target = new_distance
        else:
            # Si on n'a pas gagné, on calcule la récompense "GPS" normalement
            new_distance = self._get_shortest_path_distance(self.current_page_title, self.target_page_title)

            # Récompense de base : positive si on se rapproche, négative sinon.
            reward = float(self.current_distance_to_target - new_distance)

            # Pénalité pour inciter à finir vite
            reward -= 0.1

            # Grosse pénalité pour le retour en arrière
            if self.current_page_title == self.previous_page_title:
                reward -= 5.0

            # Mise à jour de la distance pour le prochain pas
            self.current_distance_to_target = new_distance

            # Pénalité si on est trop long (uniquement si on n'a pas gagné)
            if truncated:
                reward -= 5.0

        self.available_actions = self._get_available_actions()
        info = {"path": self.path, "action_mask": self.action_mask()}

        return self._get_observation(), reward, terminated, truncated, info

    def action_mask(self) -> np.ndarray:
        """Génère un masque binaire pour les actions valides."""
        mask = np.zeros(self.max_actions, dtype=np.int8)
        num_valid_actions = len(self.available_actions)
        mask[:num_valid_actions] = 1
        return mask

    def close(self):
        print("Fermeture de la connexion Neo4j.")
        self.driver.close()