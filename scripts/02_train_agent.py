# scripts/02_train_agent.py (Avec Versioning Automatique)
import sys
import os
import re
import time
import multiprocessing

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


# --- NOUVELLES FONCTIONS DE VERSIONING ---
def find_latest_version(path, base_name):
    """Trouve la dernière version d'un modèle (ex: base_name-3)."""
    if not os.path.exists(path):
        return 0, None

    # Regex pour trouver les extensions de version (ex: "-1", "-2", etc.)
    pattern = re.compile(f"^{re.escape(base_name)}(?:-(\d+))?\.zip$")

    latest_version = 0
    latest_file = f"{base_name}.zip"  # Défaut si aucune version "-X" n'est trouvée
    found_base = False

    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            if match.group(1):  # Si une extension de version est trouvée
                version = int(match.group(1))
                if version >= latest_version:
                    latest_version = version
                    latest_file = f
            else:  # Fichier de base sans extension (ex: "modele.zip")
                found_base = True

    # Si on trouve un fichier de base mais aucune version, le fichier de base est la version 0
    if found_base and latest_version == 0:
        return 0, latest_file

    return latest_version, latest_file


def get_next_model_name(path, base_name):
    """Génère le nom pour un tout nouvel entraînement (ex: base_name_3)."""
    if not os.path.exists(path):
        return f"{base_name}_1"

    pattern = re.compile(f"^{re.escape(base_name)}_(\d+)(?:-\d+)?\.zip$")
    max_num = 0
    for f in os.listdir(path):
        match = pattern.match(f)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return f"{base_name}_{max_num + 1}"


# --- Fonction make_env (inchangée) ---
def make_env():
    from src.environment import WikiEnv
    from sb3_contrib.common.wrappers import ActionMasker
    env = WikiEnv()
    env = Monitor(env)
    return ActionMasker(env, action_mask_fn=lambda e: e.action_mask())


def main():
    print("--- Lancement de l'entraînement de l'Agent Wiki AI (avec Versioning) ---")

    os.makedirs(config.MODELS_PATH, exist_ok=True)

    model_base_name = ""
    model_to_load_path = None

    if config.RESUME_TRAINING:
        base_name_to_resume = config.MODEL_NAME_TO_RESUME
        version, filename = find_latest_version(config.MODELS_PATH, base_name_to_resume)

        if filename is None or not os.path.exists(os.path.join(config.MODELS_PATH, filename)):
            print(f"ERREUR: Aucun modèle trouvé pour la base '{base_name_to_resume}'. Arrêt.")
            return

        model_to_load_path = os.path.join(config.MODELS_PATH, filename)
        # Le nouveau nom sera l'ancien avec une version incrémentée
        model_base_name = f"{base_name_to_resume}-{version + 1}"
    else:
        # On génère un nouveau nom de base
        model_base_name = get_next_model_name(config.MODELS_PATH, "nouveau_modele")

    print(f"Nom de base pour cette session : {model_base_name}")

    # Configuration des chemins basée sur le nom de session
    log_dir = os.path.join(config.LOGS_PATH, model_base_name)
    final_model_path = os.path.join(config.MODELS_PATH, f"{model_base_name}.zip")
    checkpoint_model_path = os.path.join(config.MODELS_PATH, "checkpoints", model_base_name)

    # Création de l'environnement
    num_cpu = os.cpu_count()
    print(f"Création d'un environnement vectorisé avec {num_cpu} processus parallèles...")
    env_fns = [make_env for _ in range(num_cpu)]
    env = SubprocVecEnv(env_fns, start_method='spawn')

    # Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoint_model_path,
        name_prefix="checkpoint"
    )

    # Initialisation ou chargement du modèle
    if config.RESUME_TRAINING:
        print(f"Reprise de l'entraînement à partir de : {model_to_load_path}")
        model = MaskablePPO.load(
            model_to_load_path,
            env=env,
            tensorboard_log=log_dir
        )
    else:
        print(f"Création d'un nouveau modèle : {model_base_name}")
        model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device='cpu', n_steps=2048,
                            batch_size=64, gamma=0.99, learning_rate=0.0003)

    # Entraînement
    print(f"\n--- Début de l'entraînement jusqu'à {config.TOTAL_TIMESTEPS} timesteps ---")

    try:
        model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=checkpoint_callback, progress_bar=True,
                    reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\nEntraînement interrompu.")
    finally:
        print(f"Sauvegarde du modèle final dans : {final_model_path}")
        model.save(final_model_path)
        print("Fermeture des environnements...")
        env.close()

    print("\n✅ Entraînenent terminé !")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()