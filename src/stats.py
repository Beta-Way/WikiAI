# stats.py
import pandas as pd
import matplotlib.pyplot as plt

STATS_LOG_PATH = "training_stats_offline_test.jsonl"

def analyze_stats():
    try:
        # Lire le fichier jsonl ligne par ligne dans un DataFrame pandas
        df = pd.read_json(STATS_LOG_PATH, lines=True)
    except FileNotFoundError:
        print(f"Le fichier de log '{STATS_LOG_PATH}' n'a pas été trouvé. Lancez d'abord l'entraînement.")
        return
    except ValueError:
        print("Le fichier de log est vide ou mal formé.")
        return

    print("--- Statistiques Générales ---")
    total_episodes = len(df)
    print(f"Nombre total de parties jouées : {total_episodes}")

    success_rate = (df['success'].sum() / total_episodes) * 100 if total_episodes > 0 else 0
    print(f"Taux de réussite : {success_rate:.2f}%")

    avg_steps = df['steps'].mean()
    print(f"Nombre de clics moyen (toutes parties) : {avg_steps:.2f}")

    avg_steps_success = df[df['success']]['steps'].mean()
    print(f"Nombre de clics moyen (parties réussies) : {avg_steps_success:.2f}")

    print("\n--- Analyse des Progrès ---")
    # On groupe les résultats par tranche de 10% de l'entraînement
    df['training_chunk'] = pd.cut(df['total_timesteps'], bins=10)
    progress = df.groupby('training_chunk')['success'].mean() * 100
    print("Taux de réussite par phase d'entraînement :")
    print(progress)

    # Créer un graphique
    plt.figure(figsize=(12, 6))
    progress.plot(kind='bar', color='skyblue')
    plt.title("Progression du Taux de Réussite de l'IA")
    plt.xlabel("Phase d'entraînement (pas de temps)")
    plt.ylabel("Taux de Réussite (%)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_stats()