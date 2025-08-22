# scripts/00_generate_missions.py
import sys
import os
import json
import random
from tqdm import tqdm

# Ajoute la racine du projet au path pour permettre l'import de 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase, Driver
from src import config

# --- CONFIGURATION ---
NUM_START_PAGES = 1000  # Nombre de pages de départ à tester
MAX_MISSIONS_TO_FIND = 5000 # On s'arrête quand on a trouvé assez de missions
MIN_DISTANCE = 2       # On veut des missions qui demandent au moins 2 clics
MAX_DISTANCE = 10      # Évite les missions trop longues ou impossibles
OUTPUT_FILE = "missions.json"

def get_random_start_pages(driver: Driver, count: int) -> list[str]:
    """Récupère une liste de titres de pages au hasard depuis la base."""
    with driver.session(database="neo4j") as session:
        # La clause "random()" est coûteuse, mais acceptable pour un script utilitaire.
        # On utilise le score pour biaiser vers des pages plus "intéressantes".
        result = session.run("""
            MATCH (p:Page)
            RETURN p.title AS title, rand() AS r
            ORDER BY r
            LIMIT $count
        """, count=count)
        return [record["title"] for record in result]

def find_missions_from_start_page(driver: Driver, start_page: str) -> list[dict]:
    """Trouve toutes les cibles atteignables entre MIN et MAX distance."""
    with driver.session(database="neo4j") as session:
        # On utilise l'algorithme de plus court chemin (shortestPath)
        # pour trouver la distance à toutes les autres pages atteignables.
        result = session.run("""
            MATCH (start:Page {title: $start_page}), (target:Page)
            WHERE start <> target
            MATCH p = shortestPath((start)-[:LINKS_TO*..10]->(target))
            RETURN target.title AS target_page, length(p) AS distance
        """, start_page=start_page)

        missions = []
        for record in result:
            distance = record["distance"]
            if MIN_DISTANCE <= distance <= MAX_DISTANCE:
                missions.append({
                    "start": start_page,
                    "target": record["target_page"],
                    "distance": distance
                })
        return missions

def main():
    """Script principal pour générer et sauvegarder les missions."""
    print("--- Générateur de Missions pour Wiki AI ---")

    auth = None
    if config.NEO4J_AUTH_ENABLED:
        auth = (config.NEO4J_USER, config.NEO4J_PASSWORD)

    with GraphDatabase.driver(config.NEO4J_URI, auth=auth) as driver:
        driver.verify_connectivity()
        print("Connexion à Neo4j établie.")

        start_pages = get_random_start_pages(driver, NUM_START_PAGES)
        print(f"{len(start_pages)} pages de départ récupérées.")

        all_missions = []

        with tqdm(total=MAX_MISSIONS_TO_FIND, desc="Recherche de missions") as pbar:
            for start_page in start_pages:
                if len(all_missions) >= MAX_MISSIONS_TO_FIND:
                    break

                missions = find_missions_from_start_page(driver, start_page)

                if missions:
                    all_missions.extend(missions)
                    pbar.update(len(missions))

        if not all_missions:
            print("\nERREUR: Aucune mission trouvée. Votre graphe est-il peut-être trop petit ou fragmenté ?")
            print("Essayez d'augmenter NUM_START_PAGES ou de re-générer un graphe plus dense.")
            return

        print(f"\nRecherche terminée. {len(all_missions)} missions trouvées.")

        # On mélange pour ne pas avoir toutes les missions du même départ à la suite
        random.shuffle(all_missions)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_missions, f, indent=4, ensure_ascii=False)

        print(f"✅ Missions sauvegardées avec succès dans '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()