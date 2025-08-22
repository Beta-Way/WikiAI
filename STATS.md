# Comprendre les Statistiques d'Entraînement

Ce document vous aide à interpréter les statistiques affichées dans le terminal et sur TensorBoard pendant l'entraînement de votre IA. C'est le "bulletin scolaire" de votre agent.

## Lancer l'Analyse avec TensorBoard

C'est l'outil le plus puissant pour visualiser les progrès.

1.  Ouvrez un **second terminal**.
2.  Activez l'environnement virtuel (`source .venv/bin/activate`).
3.  Lancez la commande en pointant vers le dossier `logs/` :
    ```bash
    tensorboard --logdir=logs/
    ```
4.  Ouvrez l'URL `http://localhost:6006/` dans votre navigateur.

## Les Graphiques Clés à Surveiller sur TensorBoard

Sur TensorBoard, vous verrez de nombreux graphiques. Voici les plus importants, divisés en deux catégories : la **Performance** (est-ce que l'IA s'améliore ?) et l'**Apprentissage** (comment apprend-elle ?).

### Catégorie "Performance" (Les plus importants)

Ces graphiques vous disent si votre IA devient réellement meilleure pour accomplir ses missions. Vous les trouverez sous l'onglet **"SCALARS"** dans la section `rollout`.

-   `rollout/ep_rew_mean` (Récompense Moyenne par Épisode)
    -   **C'est LE GRAPHIQUE LE PLUS IMPORTANT.** Il représente le score moyen que l'IA obtient pour chaque mission terminée (victoire ou défaite).
    -   **Tendance attendue :** Une augmentation **constante**. Au début, elle sera très négative (l'IA se perd et accumule les pénalités). Un bon entraînement la verra monter, franchir le zéro et continuer de grimper. C'est le signe que l'IA gagne de plus en plus souvent et de manière plus efficace.

-   `rollout/ep_len_mean` (Longueur Moyenne par Épisode)
    -   **Ce que c'est :** Le nombre de clics moyen que l'IA effectue avant de terminer une mission.
    -   **Tendance attendue :** Une diminution. Au début, l'IA va errer jusqu'à la limite de clics. En apprenant, elle trouvera des chemins plus directs, faisant baisser cette moyenne.

-   `rollout/success_rate` (Taux de Réussite)
    -   **Ce que c'est :** (Parfois disponible) Le pourcentage de missions que l'IA réussit.
    -   **Tendance attendue :** Une augmentation vers 1.0 (100% de réussite).

### Catégorie "Apprentissage" (Pour les curieux)

Ces graphiques (dans la section `train`) sont plus techniques et décrivent la "santé" du processus d'apprentissage du réseau de neurones.

-   `train/entropy_loss` (Entropie)
    -   **Ce que c'est :** Une mesure de la "curiosité" de l'IA. Une entropie élevée (valeur plus négative) signifie que l'IA explore beaucoup. Une entropie basse signifie qu'elle est très sûre de ses choix.
    -   **Tendance attendue :** Une diminution **lente et régulière**. C'est le signe que l'IA passe d'une exploration chaotique à une stratégie confiante.

-   `train/explained_variance` (Variance Expliquée)
    -   **Ce que c'est :** La capacité de l'IA à prédire si un coup sera bon ou mauvais. Une valeur proche de `1.0` signifie qu'elle prédit très bien ses récompenses futures.
    -   **Tendance attendue :** Une augmentation rapide vers des valeurs élevées (idéalement > 0.8). C'est le signe que l'IA "comprend" les règles du jeu.

-   `train/loss` (Perte)
    -   **Ce que c'est :** Une mesure d'erreur interne.
    -   **Tendance attendue :** Elle doit globalement diminuer, mais elle peut être très "bruyante" (faire des sauts). Ne vous inquiétez pas de ses fluctuations tant que les métriques de performance (`ep_rew_mean`) s'améliorent.

En résumé, pour votre prochain entraînement :
1.  **Modifiez `02_train_agent.py`** comme indiqué ci-dessus.
2.  Lancez l'entraînement.
3.  Lancez TensorBoard.
4.  **Surveillez la courbe `rollout/ep_rew_mean` monter.** C'est votre principal indicateur de succès.