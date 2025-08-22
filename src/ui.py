# ui.py

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, SelectionList, Markdown
from textual.containers import VerticalScroll
from textual.binding import Binding

# On importe nos deux autres modules
from api import WikipediaService
from ia import Agent


class WikiGameApp(App):
    """L'application de jeu Wiki en mode texte, pilotée par l'IA."""

    TITLE = "WikiGame IA"
    SUB_TITLE = "L'IA cherche le chemin le plus court !"
    BINDINGS = [Binding(key="q", action="quit", description="Quitter")]

    def __init__(self, start_page: str, target_page: str, model_path: str):
        """
        Initialise l'application avec la mission et le chemin du modèle entraîné.
        """
        super().__init__()
        self.wiki_service = WikipediaService(language='fr')

        # On crée l'agent IA en lui passant le chemin du modèle et sa mission
        self.agent = Agent(model_path=model_path, target_page=target_page)

        # On garde en mémoire l'état du jeu
        self.target_page = target_page
        self.path = [start_page]
        self.current_page_title = start_page
        self.game_over = False

    def compose(self) -> ComposeResult:
        """Crée les widgets de l'interface."""
        yield Header()
        with VerticalScroll():
            yield Static(f"🎯 Cible : [b]{self.target_page}[/b]", id="target_page")
            yield Static(f"📍 Actuel : [b]{self.current_page_title}[/b]", id="current_page_title")
            yield Static(f"👣 Chemin ({len(self.path) - 1} clics) : {' -> '.join(self.path)}", id="path_display")
            yield Static("🤖 L'IA se prépare...", id="ai_status")
            yield Markdown("Chargement du résumé...", id="summary")
            # La liste des liens n'est plus interactive, elle sert juste à l'affichage
            yield SelectionList[str](id="links_list", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        """Appelé une fois que l'application est prête à démarrer."""
        # On charge la première page, ce qui déclenchera la boucle de jeu de l'IA
        self.update_page_display(self.current_page_title)

    def update_page_display(self, page_title: str) -> None:
        """
        Met à jour l'affichage avec les infos de la nouvelle page et prépare le tour de l'IA.
        """
        self.current_page_title = page_title

        # Mise à jour des textes d'information
        self.query_one("#current_page_title", Static).update(f"📍 Actuel : [b]{page_title}[/b]")
        path_str = " -> ".join(self.path)
        self.query_one("#path_display", Static).update(f"👣 Chemin ({len(self.path) - 1} clics) : {path_str}")

        page = self.wiki_service.get_page(page_title)

        if page:
            summary = self.wiki_service.get_page_summary(page)
            self.query_one("#summary", Markdown).update(summary)

            links = self.wiki_service.get_page_links(page)
            link_list = self.query_one("#links_list", SelectionList)
            link_list.clear_options()
            link_list.add_options([(link, link) for link in links])

            # Si le jeu n'est pas fini, on déclenche le prochain tour de l'IA après une pause
            if not self.game_over:
                self.query_one("#ai_status").update("🤖 L'IA réfléchit...")
                # On met une pause de 1.5s pour pouvoir suivre ce qu'il se passe
                self.set_timer(1.5, self.run_ai_turn)
        else:
            self.query_one("#summary", Markdown).update(
                f"❌ ERREUR: La page '{page_title}' n'a pas été trouvée. L'IA est bloquée.")
            self.query_one("#ai_status").update("🤖 L'IA est bloquée.")
            self.game_over = True

    def run_ai_turn(self) -> None:
        """Exécute un tour de jeu de l'IA."""
        # On récupère la liste des liens actuellement affichés sur la page
        link_list_widget = self.query_one("#links_list", SelectionList)
        available_links = [str(option.prompt) for option in link_list_widget._options]

        # On demande à l'agent de choisir un lien en lui donnant son état actuel
        chosen_link = self.agent.choose_next_link(self.current_page_title)

        if chosen_link is None:
            self.query_one("#ai_status").update("🤖 L'IA est dans une impasse (aucun lien) !")
            self.game_over = True
            return

        self.query_one("#ai_status").update(f"🤖 L'IA a choisi : [b]{chosen_link}[/b]")
        self.path.append(chosen_link)

        # On vérifie la condition de victoire
        if chosen_link == self.target_page:
            self.game_over = True
            clicks = len(self.path) - 1
            self.query_one("#summary", Markdown).update(f"🎉 **VICTOIRE !** L'IA a atteint la cible en {clicks} clics.")
            self.query_one("#ai_status").update("🏆 Mission accomplie !")
        else:
            # Sinon, on charge la page suivante, ce qui continue la boucle
            self.update_page_display(chosen_link)