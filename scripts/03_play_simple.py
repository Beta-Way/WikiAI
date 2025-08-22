# scripts/03_play_simple.py (Version 100% stable, basée sur print)
import sys
import os
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

try:
    from sb3_contrib import MaskablePPO
    from src.environment import WikiEnv
    from src import config
except Exception as e:
    print(f"ERREUR CRITIQUE PENDANT L'IMPORTATION : {e}")
    sys.exit(1)

# --- CONFIGURATION DU JEU ---
MODEL_PATH = os.path.join(config.MODELS_PATH, "wiki_ppo_final.zip")
MAX_CLICKS = 20

# --- Couleurs pour le terminal ---
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def clear_screen():
    """Efface le terminal pour un affichage propre."""
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    """Boucle de jeu principale dans le terminal."""

    # --- 1. CHARGEMENT ---
    try:
        print("Chargement de l'environnement et du modèle IA (cela peut prendre un moment)...")
        env = WikiEnv()
        model = MaskablePPO.load(MODEL_PATH)
        print(f"{GREEN}Chargement terminé !{RESET}")
        time.sleep(2)
    except Exception as e:
        print(f"{RED}ERREUR: Impossible de charger le modèle ou l'environnement.{RESET}")
        print(e)
        return

    # --- 2. INITIALISATION DE LA PARTIE ---
    obs, info = env.reset()
    game_over = False
    start_time = time.monotonic()

    # --- 3. BOUCLE DE JEU ---
    while not game_over:
        clear_screen()

        # Affichage des informations
        elapsed_time = int(time.monotonic() - start_time)
        print(f"{BOLD}--- Wiki AI en Action ---{RESET}")
        print(f"Mission   : {YELLOW}{env.start_page_title}{RESET} -> {GREEN}{env.target_page_title}{RESET}")
        print(f"Stats     : Clics {env.current_step}/{MAX_CLICKS} | Temps: {elapsed_time}s")
        print("-" * 25)

        # Affichage du chemin
        print(f"{BOLD}Chemin Actuel :{RESET}")
        for i, page in enumerate(env.path):
            print(f"  {i}. {page}")

        print("\n" + "-" * 25)

        # Affichage de la "vision" de l'IA
        print(f"{BOLD}Vision de l'IA (Actions depuis '{env.current_page_title}') :{RESET}")
        available_actions = env.available_actions
        for i, action_name in enumerate(available_actions[:15]):  # On n'affiche que les 15 premières
            prefix = f"  {i + 1:2d}. "
            if action_name == env.target_page_title:
                print(f"{GREEN}{prefix}{action_name} (CIBLE !){RESET}")
            else:
                print(f"{prefix}{action_name}")
        if len(available_actions) > 15:
            print(f"  ... et {len(available_actions) - 15} autres.")

        print("\n" + "-" * 25)

        # L'IA prend sa décision
        action_masks = info["action_mask"]
        action_index, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
        action_index = int(action_index)
        chosen_link = available_actions[action_index]

        print(f"{BLUE}L'IA réfléchit... et choisit de cliquer sur : {BOLD}{chosen_link}{RESET}")

        # Attente de l'utilisateur
        try:
            input("\nAppuyez sur Entrée pour continuer...")
        except KeyboardInterrupt:
            print("\nPartie interrompue.")
            return

        # L'environnement exécute l'action
        obs, _, terminated, truncated, info = env.step(action_index)

        # Vérification de la fin de partie
        if terminated or truncated or env.current_step >= MAX_CLICKS:
            game_over = True
            clear_screen()
            print("=" * 30)
            if terminated:
                print(f"{GREEN}{BOLD}🎉 VICTOIRE ! 🎉{RESET}")
                print(f"L'IA a atteint la cible en {env.current_step} clics.")
            else:
                print(f"{RED}{BOLD}☠️ DÉFAITE ☠️{RESET}")
                print(f"L'IA n'a pas atteint la cible en {MAX_CLICKS} clics.")
            print("=" * 30)
            print(f"\n{BOLD}Chemin final :{RESET}")
            for i, page in enumerate(env.path):
                print(f"  {i}. {page}")


if __name__ == "__main__":
    main()