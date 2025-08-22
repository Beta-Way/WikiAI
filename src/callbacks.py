# callbacks.py
import json
from stable_baselines3.common.callbacks import BaseCallback


class StatsRecorderCallback(BaseCallback):
    """
    Un callback pour enregistrer les statistiques de chaque épisode
    dans un fichier JSON Lines.
    """

    def __init__(self, log_path="training_stats.jsonl", verbose=0):
        super(StatsRecorderCallback, self).__init__(verbose)
        self.log_path = log_path
        # Ouvre le fichier en mode 'append'
        self.log_file = open(self.log_path, "a")

    def _on_step(self) -> bool:
        # Cette méthode est appelée à chaque "pas" de l'environnement.
        # On vérifie si un épisode vient de se terminer.
        if self.locals['dones'][0]:
            # Le dictionnaire 'info' contient les informations de l'environnement.
            # Le Monitor wrapper y ajoute la clé 'episode' à la fin d'une partie.
            info = self.locals['infos'][0]

            if "episode" in info:
                # On a trouvé les stats de fin de partie.
                episode_stats = info["episode"]

                log_entry = {
                    # Infos personnalisées que nous avons ajoutées
                    "start": info.get("start"),
                    "target": info.get("target"),
                    "path": info.get("path"),
                    "success": info.get("path", ["", ""])[-1] == info.get("target"),

                    # Stats fournies par le Monitor wrapper
                    "reward": episode_stats["r"],
                    "steps": episode_stats["l"],

                    # Info générale de l'entraînement
                    "total_timesteps": self.num_timesteps,
                }

                self.log_file.write(json.dumps(log_entry) + "\n")
                self.log_file.flush()  # Pour écrire immédiatement sur le disque

        return True  # On retourne True pour continuer l'entraînement

    def __del__(self):
        # S'assure que le fichier est bien fermé quand l'objet est détruit.
        if hasattr(self, 'log_file'):
            self.log_file.close()