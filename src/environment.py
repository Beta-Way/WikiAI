# src/environment.py (V3.0 - Anti-Cycle par Filtrage d'Action)
import gymnasium as gym
import numpy as np
from neo4j import GraphDatabase, Driver
from sentence_transformers import SentenceTransformer
import random
import json
from typing import Optional, Tuple, Dict, List

from . import config

MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_SIZE = 384


class WikiEnv(gym.Env):
    """
    Environnement Gymnasium avec récompense GPS et un mécanisme anti-cycle
    qui empêche l'agent de revisiter des pages.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        print("Initialisation de l'environnement WikiEnv (Anti-Cycle)...")
        self.driver: Driver = self._connect_to_neo4j()
        self.model = SentenceTransformer(MODEL_NAME)
        with open("missions.json", "r", encoding="utf-8") as f:
            self.missions = json.load(f)
        print(f"{len(self.missions)} missions chargées.")

        self.max_actions = 100
        self.action_space = gym.spaces.Discrete(self.max_actions)

        # L'observation reste simple, la complexité est dans la logique d'action
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3 * VECTOR_SIZE,),
            dtype=np.float32
        )

        self.max_steps = 25
        # ... (initialisation des variables d'état)
        self.start_page_title: Optional[str] = None
        self.target_page_title: Optional[str] = None
        self.target_vector: Optional[np.ndarray] = None
        self.current_page_title: Optional[str] = None
        self.previous_page_title: Optional[str] = None
        self.current_step = 0
        self.current_distance_to_target = -1
        self.path: List[str] = []
        self.path_set: set[str] = set()  # Pour une vérification rapide des cycles
        self.available_actions: List[str] = []

    def _connect_to_neo4j(self) -> Driver:
        # ... (inchangé)
        auth = None
        if config.NEO4J_AUTH_ENABLED:
            auth = (config.NEO4J_USER, config.NEO4J_PASSWORD)
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=auth)
        driver.verify_connectivity()
        return driver

    def _get_page_vector(self, title: Optional[str]) -> np.ndarray:
        # ... (inchangé)
        if title is None:
            return np.zeros(VECTOR_SIZE, dtype=np.float32)
        return self.model.encode(title, convert_to_numpy=True)

    def _get_shortest_path_distance(self, start_node: str, end_node: str) -> int:
        # ... (inchangé)
        with self.driver.session(database="neo4j") as session:
            result = session.run(
                "MATCH (start:Page {title: $s}), (end:Page {title: $e}) "
                "MATCH p = shortestPath((start)-[:LINKS_TO*]->(end)) "
                "RETURN length(p) AS d", s=start_node, e=end_node
            )
            record = result.single()
            return record["d"] if record else self.max_steps * 2

    def _get_observation(self) -> np.ndarray:
        # ... (inchangé)
        current_vector = self._get_page_vector(self.current_page_title)
        previous_vector = self._get_page_vector(self.previous_page_title)
        obs = np.concatenate([current_vector, self.target_vector, previous_vector]).astype(np.float32)
        return obs

    def _get_available_actions(self) -> List[str]:
        """
        Récupère les liens sortants, en garantissant la présence de la cible
        ET en filtrant les pages déjà visitées.
        """
        with self.driver.session(database="neo4j") as session:
            result = session.run(
                "MATCH (p:Page {title: $title})-[:LINKS_TO]->(next:Page) "
                "RETURN next.title AS nextPage, next.score AS score",
                title=self.current_page_title
            )
            neighbors = {record["nextPage"]: record["score"] for record in result}

        # --- NOUVELLE LOGIQUE ANTI-CYCLE ---
        # On ne considère que les voisins qui ne sont PAS dans le chemin déjà parcouru.
        # On utilise un `set` (self.path_set) pour que cette vérification soit instantanée.
        unvisited_neighbors = {
            title: score for title, score in neighbors.items()
            if title not in self.path_set
        }

        target_is_neighbor = self.target_page_title in unvisited_neighbors

        sorted_neighbors = sorted(
            [n for n in unvisited_neighbors if n != self.target_page_title],
            key=lambda n: unvisited_neighbors.get(n, 0),
            reverse=True
        )

        final_actions = []
        if target_is_neighbor:
            final_actions.append(self.target_page_title)

        final_actions.extend(sorted_neighbors)
        return final_actions[:self.max_actions]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        mission = random.choice(self.missions)
        self.start_page_title = mission["start"]
        self.target_page_title = mission["target"]

        self.target_vector = self._get_page_vector(self.target_page_title)
        self.current_page_title = self.start_page_title
        self.previous_page_title = None
        self.current_step = 0

        # On initialise le chemin et le set pour la vérification des cycles
        self.path = [self.start_page_title]
        self.path_set = {self.start_page_title}

        self.current_distance_to_target = self._get_shortest_path_distance(
            self.current_page_title, self.target_page_title
        )
        self.available_actions = self._get_available_actions()
        return self._get_observation(), {"action_mask": self.action_mask()}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if action >= len(self.available_actions):
            reward = -10.0
            return self._get_observation(), reward, False, True, {"action_mask": self.action_mask()}

        # --- Transition d'état ---
        next_page_title = self.available_actions[action]
        self.previous_page_title = self.current_page_title
        self.current_page_title = next_page_title

        # On met à jour le chemin et le set
        self.path.append(self.current_page_title)
        self.path_set.add(self.current_page_title)
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.max_steps

        # La logique de récompense est maintenant plus simple car les cycles sont impossibles.
        if self.current_page_title == self.target_page_title:
            new_distance = 0
            terminated = True
            reward = float(self.current_distance_to_target - new_distance) + 20.0
        else:
            new_distance = self._get_shortest_path_distance(self.current_page_title, self.target_page_title)
            reward = float(self.current_distance_to_target - new_distance)
            reward -= 0.1  # Pénalité de pas
            if truncated:
                reward -= 5.0

        self.current_distance_to_target = new_distance

        # On met à jour la liste d'actions possibles (qui seront maintenant filtrées)
        self.available_actions = self._get_available_actions()
        info = {"path": self.path, "action_mask": self.action_mask()}
        return self._get_observation(), reward, terminated, truncated, info

    def action_mask(self) -> np.ndarray:
        # ... (inchangé)
        mask = np.zeros(self.max_actions, dtype=np.int8)
        num_valid_actions = len(self.available_actions)
        mask[:num_valid_actions] = 1
        return mask

    def close(self):
        # ... (inchangé)
        print("Fermeture de la connexion Neo4j.")
        self.driver.close()