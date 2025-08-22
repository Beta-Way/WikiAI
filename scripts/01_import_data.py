# scripts/01_import_data.py
import sys
import os

# Ajoute le dossier parent (la racine du projet) au path Python
# pour permettre l'import du module 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_importer import run_import

if __name__ == "__main__":
    print("Lancement du script d'importation des données Wikipédia...")
    run_import()