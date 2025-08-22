# scripts/03_play.py (Version de Diagnostic - Stabilité Maximale)
import sys
import os
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
CSS_FILE_PATH = os.path.join(ROOT_DIR, "style.css")

# --- Imports Textual ---
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Footer, Static, Button, DataTable, RichLog
from textual.screen import ModalScreen

# --- Imports de notre projet ---
# On les importe ici pour le bloc try/except
try:
    from sb3_contrib import MaskablePPO
    from src.environment import WikiEnv
    from src import config
except Exception as e:
    print("ERREUR CRITIQUE PENDANT L'IMPORTATION DES MODULES DU PROJET:")
    print(e)
    import traceback

    traceback.print_exc()
    sys.exit(1)

# --- CONFIGURATION DU JEU ---
MODEL_PATH = os.path.join(config.MODELS_PATH, "wiki_ppo_final.zip")
MAX_CLICKS = 20


# --- Écran de Fin ---
class EndScreen(ModalScreen):
    def __init__(self, message: str, classes: str):
        super().__init__()
        self.message = message
        self.classes = classes

    def compose(self) -> ComposeResult:
        with Container(id="end_screen_container", classes=self.classes):
            yield Static(self.message, id="end_screen_message")
            yield Button("Quitter", variant="error", id="quit_game")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.app.exit()


# --- Application Principale ---
class WikiApp(App):
    CSS_PATH = CSS_FILE_PATH
    BINDINGS = [("q", "request_quit", "Quitter")]

    def __init__(self):
        super().__init__()
        # --- BLOC DE DIAGNOSTIC ---
        # Ce bloc va attraper toute erreur qui se produit pendant le chargement
        try:
            print("Chargement du modèle et de l'environnement (cela peut prendre un moment)...")
            self.env = WikiEnv()
            self.model = MaskablePPO.load(MODEL_PATH)
            print("Prêt ! Le jeu va démarrer automatiquement.")

            self.obs = None
            self.info = None
            self.game_over = True
            self.start_time = 0
        except Exception as e:
            print("=" * 60)
            print("   ERREUR CRITIQUE PENDANT L'INITIALISATION DE WIKIAPP   ")
            print("=" * 60)
            print(f"L'erreur est : {e}")
            import traceback
            traceback.print_exc()
            # On force la sortie pour éviter un crash silencieux
            sys.exit(1)
        # --- FIN DU BLOC DE DIAGNOSTIC ---

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main_container"):
            yield Static("Panneau de Contrôle", id="left_panel")
            yield Static("Vision de l'IA", id="right_panel")
        yield Footer()

    def on_mount(self) -> None:
        self.start_new_game()

    def start_new_game(self):
        self.game_over = False
        self.start_time = time.monotonic()

        left_panel = self.query_one("#left_panel")
        right_panel = self.query_one("#right_panel")
        left_panel.remove_children()
        right_panel.remove_children()

        self.obs, self.info = self.env.reset()

        left_panel.mount(Static(f"🎯 Cible : [b]{self.env.target_page_title}[/b]", id="target_display"))
        left_panel.mount(Static(f"📍 Actuel: [b]{self.env.current_page_title}[/b]", id="current_display"))
        left_panel.mount(Static("👣 Chemin :", id="path_title"))
        left_panel.mount(RichLog(id="path_log", wrap=True, highlight=True))
        left_panel.mount(Static("...", id="status_display"))

        right_panel.mount(DataTable(id="actions_table"))
        table = self.query_one(DataTable)
        table.add_column("N°", width=5)
        table.add_column("Page Voisine")

        self.update_ui()
        self.set_timer(1.0, self.run_ai_turn)

    def run_ai_turn(self):
        if self.game_over: return

        action_masks = self.info["action_mask"]
        action, _ = self.model.predict(self.obs, action_masks=action_masks, deterministic=True)
        self.obs, _, terminated, truncated, self.info = self.env.step(int(action))

        if terminated or truncated or self.env.current_step >= MAX_CLICKS:
            self.game_over = True
            self.update_ui()
            message = f"🎉 VICTOIRE ! 🎉\n\nChemin trouvé en {self.env.current_step} clics." if terminated else f"☠️ DÉFAITE ☠️\n\nLimite de {MAX_CLICKS} clics atteinte."
            self.push_screen(EndScreen(message, classes="success" if terminated else "error"))
            return

        self.update_ui()
        self.set_timer(1.0, self.run_ai_turn)

    def update_ui(self):
        self.query_one("#current_display", Static).update(f"📍 Actuel: [b]{self.env.current_page_title}[/b]")

        path_log = self.query_one("#path_log")
        path_log.clear()
        for i, page in enumerate(self.env.path): path_log.write(f"{i}. {page}")

        table = self.query_one(DataTable)
        table.clear()
        for i, action_name in enumerate(self.env.available_actions):
            style = "green" if action_name == self.env.target_page_title else ""
            table.add_row(f"{i + 1}", f"[{style}]{action_name}[/{style}]")

        elapsed = time.monotonic() - self.start_time
        status = (f"Clics: [b]{self.env.current_step}/{MAX_CLICKS}[/b] | Temps: [b]{int(elapsed)}s[/b]")
        self.query_one("#status_display", Static).update(status)


if __name__ == "__main__":
    css_content = """
    #main_container { layout: horizontal; height: 100%; }
    #left_panel, #right_panel { padding: 1; width: 50%; }
    #left_panel { border-right: solid $accent; }
    #end_screen_container { align: center middle; width: 50; height: 10; border: thick $primary-lighten-2; }
    #end_screen_message { text-align: center; }
    #end_screen_container.success { background: $success; }
    #end_screen_container.error { background: $error; }
    """
    with open(CSS_FILE_PATH, "w") as f:
        f.write(css_content)

    app = WikiApp()
    app.run()