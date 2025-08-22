# scripts/02_train_agent.py (Version Finale - Avec start_method='spawn')
import sys
import os
import time
import multiprocessing

# On met les imports légers ici
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def make_env():
    """
    Fonction utilitaire qui importe et initialise TOUT ce dont l'environnement
    a besoin de manière isolée dans son propre processus.
    """
    from src.environment import WikiEnv
    from sb3_contrib.common.wrappers import ActionMasker

    env = WikiEnv()
    return ActionMasker(env, action_mask_fn=lambda e: e.action_mask())


def main():
    """Script principal pour orchestrer l'entraînement de l'agent IA."""
    print("--- Lancement de l'entraînement de l'Agent Wiki AI ---")

    os.makedirs(config.LOGS_PATH, exist_ok=True)
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    run_name = f"MaskablePPO_{int(time.time())}"
    log_dir = os.path.join(config.LOGS_PATH, run_name)
    final_model_path = os.path.join(config.MODELS_PATH, "wiki_ppo_final.zip")
    checkpoint_model_path = os.path.join(config.MODELS_PATH, "checkpoints")

    num_cpu = os.cpu_count()
    print(f"Création d'un environnement vectorisé avec {num_cpu} processus parallèles (méthode: spawn)...")

    env_fns = [make_env for _ in range(num_cpu)]

    # LA CORRECTION DÉFINITIVE EST ICI : on force la méthode 'spawn'
    env = SubprocVecEnv(env_fns, start_method='spawn')

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoint_model_path,
        name_prefix="wiki_ppo_checkpoint"
    )

    print("Initialisation du modèle MaskablePPO...")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device='cpu',
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.0003
    )

    print(f"\n--- Début de l'entraînement pour {config.TOTAL_TIMESTEPS} timesteps ---")
    print("...")

    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur.")
    finally:
        print(f"Sauvegarde du modèle final dans : {final_model_path}")
        model.save(final_model_path)
        print("Fermeture des environnements...")
        env.close()

    print("\n✅ Entraîtement terminé avec succès !")


if __name__ == "__main__":
    # La protection __main__ est encore plus cruciale avec la méthode 'spawn'
    multiprocessing.set_start_method('spawn', force=True)  # Sécurité supplémentaire
    main()