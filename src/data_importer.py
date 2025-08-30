# src/data_importer.py (Version finale, combinant Snowball et la correction du parsing des liens)

import gzip
import re
from collections import defaultdict
from tqdm import tqdm
from neo4j import GraphDatabase, Driver

from . import config

# Regex V2.1 : Plus robuste pour éviter le "gel" du parsing.
# Il capture (page_id, namespace, title, page_latest, page_len)
# En utilisant [^\)]* au lieu de .*?, on contraint la recherche à l'intérieur d'un seul enregistrement,
# ce qui empêche le moteur Regex de se perdre.
PAGE_RECORD_REGEX = re.compile(
    r"\((?P<id>\d+),(?P<ns>\d+),'(?P<title>(?:\\.|[^'])*)',[^\)]*?,(?P<latest>\d+),(?P<len>\d+),[^\)]*?\)")

# ####################################################################
# # CHANGEMENT 1 : RETOUR AU REGEX SIMPLE ET CORRECT POUR LES LIENS (ID -> ID)
# ####################################################################
# Ce Regex est le bon pour le format de données que vous avez.
PAGELINKS_RECORD_REGEX = re.compile(r"\((\d+),(\d+),(\d+)\)")


def parse_pages(filepath: str) -> dict[int, dict]:
    """Parse le dump SQL des pages pour extraire ID, titre et longueur."""
    # Cette fonction est correcte et reste inchangée.
    page_data = {}
    print(f"--- Parsing du fichier de pages : {filepath} ---")
    if config.DEBUG_MODE:
        print(f"!!! MODE DÉBOGAGE ACTIVÉ : Lecture de {config.DEBUG_LINE_LIMIT} lignes maximum. !!!")

    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        from itertools import islice
        line_iterator = f if not config.DEBUG_MODE else islice(f, config.DEBUG_LINE_LIMIT)

        for line in tqdm(line_iterator, desc="Parsing Pages"):
            if not line.startswith("INSERT INTO `page`"):
                continue
            for match in PAGE_RECORD_REGEX.finditer(line):
                try:
                    namespace = int(match.group('ns'))
                    if namespace == 0:
                        page_id = int(match.group('id'))
                        title = match.group('title').replace("\\'", "'")
                        length = int(match.group('len'))
                        page_data[page_id] = {"title": title, "length": length}
                except (ValueError, IndexError):
                    continue
    print(f"--- Parsing des pages terminé. {len(page_data)} articles valides trouvés. ---")
    return page_data


# ####################################################################
# # CHANGEMENT 2 : LA LOGIQUE DE PARSING DES LIENS EST SIMPLIFIÉE
# ####################################################################
# Elle travaille maintenant directement avec les IDs, ce qui est correct pour vos données.
def parse_links_and_count_degrees(filepath: str, page_data: dict) -> tuple:
    """Parse le dump des liens (format ID->ID) et compte les degrés."""
    in_degrees = defaultdict(int)
    out_degrees = defaultdict(int)
    links = []
    print(f"--- Parsing du fichier de liens (format ID -> ID) ---")

    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        from itertools import islice
        line_iterator = f if not config.DEBUG_MODE else islice(f, config.DEBUG_LINE_LIMIT)

        for line in tqdm(line_iterator, desc="Parsing Liens"):
            if not line.startswith("INSERT INTO `pagelinks`"):
                continue
            for match in PAGELINKS_RECORD_REGEX.finditer(line):
                try:
                    # On lit directement les IDs numériques
                    source_id, dest_namespace, dest_id = int(match.group(1)), int(match.group(2)), int(match.group(3))

                    # Le lien est valide si les deux pages (source et destination) existent dans notre dictionnaire de pages.
                    if dest_namespace == 0 and source_id in page_data and dest_id in page_data:
                        links.append((source_id, dest_id))
                        in_degrees[dest_id] += 1
                        out_degrees[source_id] += 1
                except (ValueError, IndexError):
                    continue
    print(f"--- Parsing des liens terminé. {len(links)} liens valides trouvés. ---")
    return links, in_degrees, out_degrees


# ####################################################################
# # CHANGEMENT 3 : PETITE CORRECTION DANS load_into_neo4j
# ####################################################################
# Les deux lignes de création de maps (id_to_title_map et title_to_id_map)
# étaient inutiles et ont été supprimées pour plus de clarté.
# J'ai aussi ajouté une vérification plus robuste dans la création des `relevant_links`.
def load_into_neo4j(driver: Driver, nodes_to_create: list[dict], all_links: list[tuple[int, int]], page_data: dict):
    """Injecte les pages et les liens filtrés dans la base de données Neo4j."""
    print("--- Début de l'injection des données dans Neo4j ---")

    final_titles_to_keep = {node['title'] for node in nodes_to_create}

    relevant_links = [
        {"source": page_data[source_id]['title'], "target": page_data[dest_id]['title']}
        for source_id, dest_id in tqdm(all_links, desc="Filtrage des liens finaux")
        # On ajoute une vérification pour éviter les KeyError si un ID a été filtré
        if page_data.get(source_id) and page_data.get(dest_id) and
           page_data[source_id]['title'] in final_titles_to_keep and
           page_data[dest_id]['title'] in final_titles_to_keep
    ]

    with driver.session(database="neo4j") as session:
        print("1. Nettoyage complet de la base de données...")
        session.run("DROP CONSTRAINT page_title_constraint IF EXISTS")
        while True:
            # J'ai augmenté la limite pour que ce soit un peu plus rapide.
            result = session.run("MATCH (n) WITH n LIMIT 50000 DETACH DELETE n RETURN count(n) AS c")
            if result.single()['c'] == 0: break

        print("2. Création de la nouvelle contrainte d'unicité...")
        session.run("CREATE CONSTRAINT page_title_constraint IF NOT EXISTS FOR (p:Page) REQUIRE p.title IS UNIQUE")

        print(f"3. Création des {len(nodes_to_create)} nœuds :Page...")
        query_nodes = """
        UNWIND $nodes AS node_data
        CREATE (p:Page {title: node_data.title, score: node_data.score})
        """
        for i in tqdm(range(0, len(nodes_to_create), 50000), desc="Injection des Nœuds"):
            batch = nodes_to_create[i:i + 50000]
            session.run(query_nodes, nodes=batch)

        print(f"4. Création des {len(relevant_links)} relations :LINKS_TO...")
        query_links = """
        UNWIND $links AS link
        MATCH (a:Page {title: link.source})
        MATCH (b:Page {title: link.target})
        CREATE (a)-[:LINKS_TO]->(b)
        """
        for i in tqdm(range(0, len(relevant_links), 50000), desc="Injection des Liens"):
            batch = relevant_links[i:i + 50000]
            session.run(query_links, links=batch)

    print("--- Injection Neo4j terminée. ---")


# Le reste du fichier (select_pages_snowball, run_import) est correct et n'a pas besoin de changer.
# Il fonctionne avec une liste de `links` sous forme de tuples d'IDs, ce que la fonction
# `parse_links_and_count_degrees` corrigée lui fournit maintenant.

def select_pages_snowball(page_scores: dict, all_links: list, page_data: dict) -> set:
    """Sélectionne un sous-graphe en utilisant la méthode de la boule de neige (limitée) puis élague."""
    # ... (inchangé)
    sorted_pages = sorted(page_scores.items(), key=lambda item: item[1], reverse=True)
    seed_pages_ids = {id for id, score in sorted_pages[:config.SNOWBALL_SEED_COUNT]}
    print(
        f"Expansion en boule de neige (profondeur: {config.SNOWBALL_DEPTH}, limite: {config.SNOWBALL_NEIGHBOR_LIMIT} voisins/page)...")
    adjacency_list = defaultdict(list)
    for source, target in all_links:
        adjacency_list[source].append(target)
    final_ids_to_keep = set(seed_pages_ids)
    current_frontier = set(seed_pages_ids)
    for i in range(config.SNOWBALL_DEPTH):
        next_frontier = set()
        for page_id in tqdm(current_frontier, desc=f"Expansion niveau {i + 1}/{config.SNOWBALL_DEPTH}"):
            all_neighbors_ids = adjacency_list.get(page_id, [])
            neighbors_with_scores = sorted(
                [(neighbor_id, page_scores.get(neighbor_id, 0.0)) for neighbor_id in all_neighbors_ids],
                key=lambda item: item[1], reverse=True
            )
            top_neighbors = neighbors_with_scores[:config.SNOWBALL_NEIGHBOR_LIMIT]
            for neighbor_id, _ in top_neighbors:
                if neighbor_id not in final_ids_to_keep:
                    next_frontier.add(neighbor_id)
        final_ids_to_keep.update(next_frontier)
        current_frontier = next_frontier
    print(f"Taille du graphe après expansion : {len(final_ids_to_keep)} pages.")
    print(f"Élagage du graphe (seuil de connectivité : {config.PRUNING_THRESHOLD})...")
    subgraph_degrees = defaultdict(int)
    for source, target in tqdm(all_links, desc="Calcul des degrés du sous-graphe"):
        if source in final_ids_to_keep and target in final_ids_to_keep:
            subgraph_degrees[source] += 1
            subgraph_degrees[target] += 1
    pruned_ids = {
        page_id for page_id, degree in subgraph_degrees.items()
        if degree >= config.PRUNING_THRESHOLD
    }
    print(f"Taille du graphe après élagage : {len(pruned_ids)} pages.")
    return pruned_ids


def run_import():
    """Fonction principale orchestrant tout le processus d'importation."""
    # ... (inchangé)
    auth = None
    if config.NEO4J_AUTH_ENABLED:
        auth = (config.NEO4J_USER, config.NEO4J_PASSWORD)
    else:
        print("Connexion à Neo4j sans authentification.")
    with GraphDatabase.driver(config.NEO4J_URI, auth=auth) as driver:
        driver.verify_connectivity()
        print("Connexion à Neo4j établie.")
        page_data = parse_pages(config.PAGE_DUMP_FULL_PATH)
        all_links, in_degrees, out_degrees = parse_links_and_count_degrees(config.PAGELINKS_DUMP_FULL_PATH, page_data)
        if not page_data:
            print("Aucune page trouvée.")
            return
        max_in = max(in_degrees.values()) if in_degrees else 1
        max_out = max(out_degrees.values()) if out_degrees else 1
        max_len = max(p['length'] for p in page_data.values() if p['length'] > 0) or 1
        page_scores = {}
        for pid, data in tqdm(page_data.items(), desc="Calcul des scores de notoriété"):
            score = (
                    (in_degrees.get(pid, 0) / max_in) * config.SCORE_WEIGHT_INDEGREE +
                    (out_degrees.get(pid, 0) / max_out) * config.SCORE_WEIGHT_OUTDEGREE +
                    (data['length'] / max_len) * config.SCORE_WEIGHT_PAGELENGTH
            )
            page_scores[pid] = score
        final_pages_ids_to_import = set()
        if config.TOP_PAGES_SELECTION_MODE == "SNOWBALL":
            print("--- Stratégie de sélection : SNOWBALL ---")
            final_pages_ids_to_import = select_pages_snowball(page_scores, all_links, page_data)
        elif config.TOP_PAGES_SELECTION_MODE == "FLAT":
            print("--- Stratégie de sélection : FLAT ---")
            sorted_pages = sorted(page_scores.items(), key=lambda item: item[1], reverse=True)
            final_pages_ids_to_import = {id for id, score in sorted_pages[:config.NUM_TOP_PAGES_TO_KEEP]}
        else:
            raise ValueError(f"Stratégie de sélection inconnue: {config.TOP_PAGES_SELECTION_MODE}")
        nodes_to_create = [{
            "title": page_data[pid]['title'],
            "score": page_scores.get(pid, 0)
        } for pid in final_pages_ids_to_import]
        print(f"Nombre final de pages à importer dans le graphe : {len(nodes_to_create)}")
        load_into_neo4j(driver, nodes_to_create, all_links, page_data)
    print("\n✅ Importation 'Snowball & Pruning' terminée avec succès !")