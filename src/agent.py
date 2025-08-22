# ia.py (Version finale pour le modèle OFFLINE)
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from stable_baselines3 import PPO
from neo4j import GraphDatabase

# On importe la constante de notre nouvel environnement
from environment_offline import MAX_LINKS_TO_CONSIDER


class Agent:
    """
    L'agent IA qui utilise le modèle entraîné sur la base Neo4j.
    Il doit reproduire la logique de l'environnement pour construire son observation.
    """

    def __init__(self, model_path: str, target_page: str):
        self.target_page = target_page

        # 1. Connexion à la base de données Neo4j
        print("🤖 L'agent se connecte à Neo4j...")
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)

        # 2. Chargement du cerveau entraîné
        print(f"🤖 Chargement du modèle depuis '{model_path}'...")
        self.model = PPO.load(model_path, device='cpu')
        print("✅ Modèle chargé.")

        # 3. Chargement du modèle sémantique
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.embedding_dim = self.semantic_model.get_sentence_embedding_dimension()

        # On pré-calcule le vecteur de la cible
        self.target_embedding = self.semantic_model.encode(self.target_page)

    def _get_candidate_links(self, page_title, target_title):
        """
        REPRODUIT LA LOGIQUE DE L'ENVIRONNEMENT :
        Récupère les liens sortants, les trie par popularité (inDegree),
        et applique la règle du "Golden Ticket".
        """
        with self.driver.session(database="neo4j") as session:
            query = """
            MATCH (a:Page {title: $title})-[:LINKS_TO]->(b:Page)
            RETURN b.title AS target_title, b.inDegree AS popularity
            """
            results = session.run(query, title=page_title)
            all_links = [{"title": r["target_title"], "pop": r.get("popularity", 0)} for r in results]

        all_links.sort(key=lambda x: x["pop"], reverse=True)
        top_popular_titles = [link["title"] for link in all_links[:MAX_LINKS_TO_CONSIDER]]

        all_link_titles = {link["title"] for link in all_links}
        if target_title in all_link_titles and target_title not in top_popular_titles:
            if len(top_popular_titles) == MAX_LINKS_TO_CONSIDER:
                top_popular_titles[-1] = target_title
            else:
                top_popular_titles.append(target_title)

        return top_popular_titles

    def choose_next_link(self, current_page_title: str) -> str | None:
        """
        Construit l'observation et utilise le modèle pour choisir le meilleur lien.
        NOTE: n'a plus besoin du 'path' ou de 'available_links' en argument.
        """
        # 1. Construire l'observation EXACTEMENT comme dans l'environnement
        current_embedding = self.semantic_model.encode(current_page_title)

        candidate_links = self._get_candidate_links(current_page_title, self.target_page)

        if not candidate_links:
            print("🤖 Impasse : Aucun lien candidat trouvé.")
            return None

        link_embeddings = self.semantic_model.encode(candidate_links)

        # Padding
        num_links = len(link_embeddings)
        padding_size = MAX_LINKS_TO_CONSIDER - num_links
        padding = np.zeros((padding_size, self.embedding_dim))

        if num_links > 0:
            final_link_vecs = np.vstack([link_embeddings, padding])
        else:  # Ne devrait jamais arriver si on a des candidats
            final_link_vecs = padding

        observation = np.concatenate([self.target_embedding, current_embedding, final_link_vecs.flatten()]).astype(
            np.float32)

        # 2. Demander au modèle de prédire la meilleure action
        action, _ = self.model.predict(observation, deterministic=True)
        action = int(action)

        # 3. Traduire l'action en un nom de lien
        if action >= len(candidate_links):
            print(f"🤖 Le modèle a prédit une action invalide ({action}), on prend la meilleure option.")
            chosen_link = candidate_links[0]
        else:
            chosen_link = candidate_links[action]

        print(f"🤖 L'IA a choisi l'action n°{action} -> '{chosen_link}'")
        return chosen_link

    def __del__(self):
        """S'assure que la connexion à la base de données est bien fermée."""
        if hasattr(self, 'driver'):
            self.driver.close()