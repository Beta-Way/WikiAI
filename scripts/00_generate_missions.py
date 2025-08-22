# scripts/00_generate_missions.py (V2.0 - Marche Aléatoire)
import sys
import os
import json
import random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase, Driver
from src import config

# --- CONFIGURATION ---
NUM_MISSIONS_TO_GENERATE = 100000  # On peut viser plus haut, c'est rapide
MIN_WALK_LENGTH = 4  # Nombre de sauts minimum
MAX_WALK_LENGTH = 11  # Nombre de sauts maximum
OUTPUT_FILE = "missions.json"


def get_random_page(driver: Driver) -> str:
    """Récupère UNE seule page au hasard."""
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (p:Page)
            RETURN p.title AS title, rand() AS r
            ORDER BY r
            LIMIT 1
        """)
        return result.single()["title"]


def perform_random_walk(driver: Driver, start_page: str, length: int) -> str:
    """Effectue une marche aléatoire depuis une page de départ et retourne la page d'arrivée."""
    current_page = start_page
    with driver.session(database="neo4j") as session:
        for _ in range(length):
            # On cherche un voisin au hasard
            result = session.run("""
                MATCH (start:Page {title: $start_page})-[:LINKS_TO]->(neighbor:Page)
                RETURN neighbor.title AS next_page, rand() as r
                ORDER BY r
                LIMIT 1
            """, start_page=current_page)

            record = result.single()
            if record:
                current_page = record["next_page"]
            else:
                # Si on est dans une impasse, on arrête la marche
                break
    return current_page


def get_shortest_path_distance(driver: Driver, start_page: str, target_page: str) -> int:
    """Calcule la distance réelle la plus courte entre les deux pages."""
    if start_page == target_page:
        return 0
    with driver.session(database="neo4j") as session:
        result = session.run(
            "MATCH (s:Page {title: $start}), (t:Page {title: $target}) "
            "MATCH p = shortestPath((s)-[:LINKS_TO*..15]->(t)) "
            "RETURN length(p) as dist",
            start=start_page, target=target_page
        )
        record = result.single()
        return record["dist"] if record else -1  # -1 si aucun chemin n'est trouvé


def main():
    """Script principal pour générer des missions par marche aléatoire."""
    print("--- Générateur de Missions V2.0 (Marche Aléatoire) ---")

    auth = None
    if config.NEO4J_AUTH_ENABLED:
        auth = (config.NEO4J_USER, config.NEO4J_PASSWORD)

    with GraphDatabase.driver(config.NEO4J_URI, auth=auth) as driver:
        driver.verify_connectivity()
        print("Connexion à Neo4j établie.")

        missions = []
        pbar = tqdm(total=NUM_MISSIONS_TO_GENERATE, desc="Génération de missions")

        while len(missions) < NUM_MISSIONS_TO_GENERATE:
            start_page = get_random_page(driver)
            walk_length = random.randint(MIN_WALK_LENGTH, MAX_WALK_LENGTH)
            target_page = perform_random_walk(driver, start_page, walk_length)

            # On s'assure que le départ et la cible sont bien différents
            if start_page == target_page:
                continue

            # On calcule la distance réelle la plus courte pour la stocker
            distance = get_shortest_path_distance(driver, start_page, target_page)

            # On ne garde que les missions valides (chemin existant et pas trop court)
            if distance >= 2:
                missions.append({
                    "start": start_page,
                    "target": target_page,
                    "distance": distance
                })
                pbar.update(1)

        pbar.close()

        if not missions:
            print("\nERREUR: Aucune mission n'a pu être générée.")
            return

        print(f"\nGénération terminée. {len(missions)} missions valides créées.")

        # On mélange une dernière fois pour une bonne mesure
        random.shuffle(missions)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(missions, f, indent=4, ensure_ascii=False)

        print(f"✅ Missions sauvegardées avec succès dans '{OUTPUT_FILE}'.")
        print("Exemple de mission :", missions[0])


if __name__ == "__main__":
    main()