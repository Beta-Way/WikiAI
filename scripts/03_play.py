# main.py (version mise à jour pour la démo)

from ui import WikiGameApp

# ==============================================================================
# POINT D'ENTRÉE PRINCIPAL - MODE DÉMONSTRATION
# ==============================================================================
if __name__ == "__main__":
    # --- PARAMÈTRES DE LA DÉMONSTRATION ---

    # Spécifiez le chemin du modèle que vous voulez tester
    # (celui créé par train.py)
    MODEL_TO_TEST = "wiki_ppo_model_offline_test.zip"

    # Choisissez une mission pour l'IA
    # Elle peut être une de celles de l'entraînement, ou une NOUVELLE !

    START_PAGE = input("Entrez le titre de la page de départ : ")
    TARGET_PAGE = input("Entrez le titre de la page cible : ")
    if START_PAGE == "":
        START_PAGE = "France"
    if TARGET_PAGE == "":
        TARGET_PAGE = "Chocolat"

    print(f"Lancement de la démo avec le modèle '{MODEL_TO_TEST}'")
    print(f"Mission : {START_PAGE} -> {TARGET_PAGE}")

    # On crée une instance de notre application UI en lui passant les bons paramètres
    app = WikiGameApp(
        start_page=START_PAGE,
        target_page=TARGET_PAGE,
        model_path=MODEL_TO_TEST
    )

    # On lance l'application
    app.run()