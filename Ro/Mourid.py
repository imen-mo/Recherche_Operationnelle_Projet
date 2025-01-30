import tkinter as tk
from tkinter import messagebox
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.table import Table
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
from io import BytesIO
from PIL import Image, ImageTk
from tkinter import Toplevel
from tkinter import *
from tkinter import simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict






# --- FONCTIONS D'ALGOS ---

def dijkstra():
   # Couleurs pastel
   BG_COLOR = "#ffe4e1"  # Fond rose pastel
   BUTTON_COLOR = "#f9d4e5"  # Rose clair
   TEXT_COLOR = "#4b0082"  # Indigo


   # Algorithme de Dijkstra
   def dijkstra(vertices, edges, source):
       distance = {v: float('inf') for v in range(vertices)}
       predecessor = {v: None for v in range(vertices)}
       distance[source] = 0
       visited = set()
       unvisited = set(range(vertices))

       while unvisited:
           u = min(unvisited, key=lambda v: distance[v])
           unvisited.remove(u)
           visited.add(u)

           for v, w in edges[u]:
               if v not in visited:
                   new_distance = distance[u] + w
                   if new_distance < distance[v]:
                       distance[v] = new_distance
                       predecessor[v] = u

       return distance, predecessor


   # Application Tkinter
   class DijkstraApp:
       def __init__(self, root):
           self.root = root
           self.root.title("Algorithme de Dijkstra")
           self.root.geometry("800x600")
           self.root.configure(bg=BG_COLOR)

           # Titre
           tk.Label(root, text="Algorithme de Dijkstra", bg=BG_COLOR, fg=TEXT_COLOR, font=("Arial", 16, "bold")).pack(pady=10)

           # Saisie des données
           tk.Label(root, text="Nombre de sommets :", bg=BG_COLOR, font=("Arial", 12)).pack()
           self.num_vertices_entry = tk.Entry(root, width=10, font=("Arial", 12))
           self.num_vertices_entry.pack(pady=5)

           tk.Label(root, text="Nombre d'arêtes :", bg=BG_COLOR, font=("Arial", 12)).pack()
           self.num_edges_entry = tk.Entry(root, width=10, font=("Arial", 12))
           self.num_edges_entry.pack(pady=5)

           tk.Label(root, text="Sommet source :", bg=BG_COLOR, font=("Arial", 12)).pack()
           self.source_entry = tk.Entry(root, width=10, font=("Arial", 12))
           self.source_entry.pack(pady=5)

           # Bouton pour générer le graphe
           tk.Button(root, text="Générer le graphe et exécuter", command=self.run_dijkstra, bg=BUTTON_COLOR, font=("Arial", 12, "bold")).pack(pady=10)

           # Cadre pour les résultats avec défilement
           self.result_frame = tk.Frame(root, bg=BG_COLOR)
           self.result_frame.pack(fill=tk.BOTH, expand=True)

           # Ajout d'un défilement
           self.scrollbar = tk.Scrollbar(self.result_frame)
           self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

           self.text_widget = tk.Text(self.result_frame, bg=BG_COLOR, fg="black", font=("Arial", 12, "bold"), wrap=tk.WORD, yscrollcommand=self.scrollbar.set)
           self.text_widget.pack(fill=tk.BOTH, expand=True)
           self.scrollbar.config(command=self.text_widget.yview)

       def run_dijkstra(self):
           try:
               # Récupérer les entrées utilisateur
               vertices = int(self.num_vertices_entry.get())
               edges_count = int(self.num_edges_entry.get())
               source = int(self.source_entry.get())

               # Valider les données
               if edges_count > vertices * (vertices - 1):
                   raise ValueError(f"Nombre d'arêtes trop élevé. Max autorisé : {vertices * (vertices - 1)}")
               if not (0 <= source < vertices):
                   raise ValueError("Le sommet source doit être compris entre 0 et le nombre de sommets - 1.")

               # Générer le graphe
               edges = {i: [] for i in range(vertices)}
               for _ in range(edges_count):
                   u, v = random.sample(range(vertices), 2)
                   w = random.randint(1, 20)
                   edges[u].append((v, w))

               # Exécuter Dijkstra
               distances, predecessor = dijkstra(vertices, edges, source)

               # Afficher les résultats
               self.display_results(vertices, edges, distances, source, predecessor)

           except ValueError as e:
               messagebox.showerror("Erreur", str(e))

       def display_results(self, vertices, edges, distances, source, predecessor):
           self.text_widget.delete("1.0", tk.END)

           # Créer le graphe
           G = nx.DiGraph()
           for u in edges:
               for v, w in edges[u]:
                   G.add_edge(u, v, weight=w)
           pos = nx.spring_layout(G)

           # Tracer le graphe
           fig, ax = plt.subplots(figsize=(8, 6))
           nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, font_weight="bold")
           nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): w for u in edges for v, w in edges[u]})

           # Tracer les chemins les plus courts
           for target in range(vertices):
               if distances[target] < float('inf'):
                   path = []
                   current = target
                   while current is not None:
                       path.insert(0, current)
                       current = predecessor[current]
                   edge_colors = ["red" if (u, v) in zip(path, path[1:]) else "gray" for u, v in G.edges()]
                   nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)

           ax.set_title("Graphe avec chemins les plus courts")
           canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
           canvas.draw()
           canvas.get_tk_widget().pack()

           # Affichage des distances et chemins
           self.text_widget.insert(tk.END, "Distances et Chemins :\n\n")
           for target in range(vertices):
               path = []
               current = target
               while current is not None:
                   path.insert(0, current)
                   current = predecessor[current]

               path_str = " -> ".join(map(str, path))
               distance = distances[target] if distances[target] < float('inf') else "Non Accessible"
               self.text_widget.insert(tk.END, f"Chemin {source} -> {target}: {path_str} (Distance: {distance})\n")

           # Bouton pour exporter les résultats
           tk.Button(self.result_frame, text="Exporter les résultats", command=lambda: self.export_results(distances, predecessor, source, vertices), bg=BUTTON_COLOR, font=("Arial", 12, "bold")).pack(pady=10)

       def export_results(self, distances, predecessor, source, vertices):
           filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Fichiers texte", "*.txt")])
           if filename:
               with open(filename, "w") as f:
                   f.write("Distances et Chemins :\n\n")
                   for target in range(vertices):
                       path = []
                       current = target
                       while current is not None:
                           path.insert(0, current)
                           current = predecessor[current]

                       path_str = " -> ".join(map(str, path))
                       distance = distances[target] if distances[target] < float('inf') else "Non Accessible"
                       f.write(f"Chemin {source} -> {target}: {path_str} (Distance: {distance})\n")
               messagebox.showinfo("Exportation", f"Résultats exportés dans {filename}")


   # Lancer l'application
   if __name__ == "__main__":
       root = tk.Tk()
       app = DijkstraApp(root)
       root.mainloop()



def kruskal():
  # Couleurs pastel
  BG_COLOR = "#ffe4e1"  # Fond rose pastel
  BUTTON_COLOR = "#f9d4e5"  # Rose clair
  TEXT_COLOR = "#4b0082"  # Indigo

  # Fonction pour générer un graphe pondéré aléatoire, toujours connexe
  def generate_connected_weighted_graph(num_nodes, num_edges):
      graph = nx.Graph()
      graph.add_nodes_from(range(num_nodes))

      # Créer une base connexe (arbre)
      for i in range(num_nodes - 1):
          weight = random.randint(1, 20)
          graph.add_edge(i, i + 1, weight=weight)

      # Ajouter des arêtes supplémentaires aléatoires
      while len(graph.edges) < num_edges:
          u, v = random.sample(range(num_nodes), 2)
          if not graph.has_edge(u, v):
              weight = random.randint(1, 20)
              graph.add_edge(u, v, weight=weight)

      return graph

  # Calculer l'arbre couvrant minimum (MST) avec Kruskal
  def kruskal_mst(graph):
      mst = nx.minimum_spanning_tree(graph, algorithm="kruskal")
      return mst

  # Application Tkinter pour poser des questions et afficher les résultats
  class KruskalApp:
      def __init__(self, root):
          self.root = root
          self.root.title("Algorithme de Kruskal")
          self.root.geometry("600x500")
          self.root.configure(bg=BG_COLOR)

          # Titre
          tk.Label(
              root, text="Algorithme de Kruskal", bg=BG_COLOR, fg=TEXT_COLOR, font=("Arial", 16, "bold")
          ).pack(pady=10)

          # Question : Nombre de sommets
          tk.Label(root, text="Nombre de sommets :", bg=BG_COLOR, font=("Arial", 12)).pack()
          self.num_nodes_entry = tk.Entry(root, width=10, font=("Arial", 12))
          self.num_nodes_entry.pack(pady=5)

          # Question : Nombre d'arêtes
          tk.Label(root, text="Nombre d'arêtes :", bg=BG_COLOR, font=("Arial", 12)).pack()
          self.num_edges_entry = tk.Entry(root, width=10, font=("Arial", 12))
          self.num_edges_entry.pack(pady=5)

          # Bouton pour lancer l'algorithme
          tk.Button(
              root, text="Lancer l'algorithme", command=self.run_algorithm, bg=BUTTON_COLOR, font=("Arial", 12, "bold")
          ).pack(pady=15)

          # Cadre pour afficher les résultats
          self.result_frame = tk.Frame(root, bg=BG_COLOR)
          self.result_frame.pack(fill=tk.BOTH, expand=True)

      def run_algorithm(self):
          try:
              # Récupérer les entrées de l'utilisateur
              num_nodes = int(self.num_nodes_entry.get())
              num_edges = int(self.num_edges_entry.get())

              # Vérifier si le nombre d'arêtes est valide
              max_edges = num_nodes * (num_nodes - 1) // 2
              if num_edges > max_edges:
                  raise ValueError(f"Nombre d'arêtes trop élevé. Max autorisé pour {num_nodes} sommets : {max_edges}")

              # Générer le graphe pondéré connexe
              graph = generate_connected_weighted_graph(num_nodes, num_edges)

              # Calculer le MST avec Kruskal
              mst = kruskal_mst(graph)

              # Afficher les résultats
              self.display_results(graph, mst)
          except ValueError as e:
              messagebox.showerror("Erreur", str(e))

      def display_results(self, graph, mst):
          # Effacer les résultats précédents
          for widget in self.result_frame.winfo_children():
              widget.destroy()

          # Résumé
          tk.Label(
              self.result_frame, text="Arbre couvrant minimum calculé avec Kruskal",
              bg=BG_COLOR, font=("Arial", 14, "bold")
          ).pack(pady=10)

          # Affichage des graphes
          pos = nx.spring_layout(graph, seed=42)

          # Graphe pondéré
          fig, ax = plt.subplots(figsize=(6, 4))
          nx.draw(
              graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10
          )
          nx.draw_networkx_edge_labels(
              graph, pos, edge_labels={(u, v): f"{d['weight']}" for u, v, d in graph.edges(data=True)}
          )
          ax.set_title("Graphe Pondéré")

          canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
          canvas.draw()
          canvas.get_tk_widget().pack()

          # Graphe avec MST
          fig, ax = plt.subplots(figsize=(6, 4))
          nx.draw(
              graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10
          )
          nx.draw_networkx_edges(mst, pos, edge_color="red", width=2)
          nx.draw_networkx_edge_labels(
              graph, pos, edge_labels={(u, v): f"{d['weight']}" for u, v, d in mst.edges(data=True)}
          )
          ax.set_title("Arbre Couvrant Minimum (MST)")

          canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
          canvas.draw()
          canvas.get_tk_widget().pack()


  # Main
  if __name__ == "__main__":
      root = tk.Tk()
      app = KruskalApp(root)
      root.mainloop()


def ford_fulkerson():
 # Fonction pour générer un graphe complètement connecté
 def generate_complete_graph(num_nodes, max_capacity=20):
     graph = defaultdict(dict)
     for i in range(num_nodes):
         for j in range(num_nodes):
             if i != j:  # Pas de boucles
                 capacity = random.randint(1, max_capacity)
                 graph[i][j] = capacity
     return graph

 # Algorithme de Ford-Fulkerson
 def bfs(graph, residual_graph, source, sink, parent):
     visited = set()
     queue = [source]
     visited.add(source)
     while queue:
         u = queue.pop(0)
         for v in graph[u]:
             if v not in visited and residual_graph[u][v] > 0:  # Capacité résiduelle > 0
                 queue.append(v)
                 visited.add(v)
                 parent[v] = u
                 if v == sink:
                     return True
     return False

 def ford_fulkerson(graph, source, sink):
     residual_graph = defaultdict(lambda: defaultdict(int))
     for u in graph:
         for v in graph[u]:
             residual_graph[u][v] = graph[u][v]

     parent = {}
     max_flow = 0
     flows = defaultdict(lambda: defaultdict(int))

     while bfs(graph, residual_graph, source, sink, parent):
         path_flow = float('Inf')
         v = sink
         while v != source:
             u = parent[v]
             path_flow = min(path_flow, residual_graph[u][v])
             v = parent[v]

         v = sink
         while v != source:
             u = parent[v]
             residual_graph[u][v] -= path_flow
             residual_graph[v][u] += path_flow
             flows[u][v] += path_flow
             v = parent[v]

         max_flow += path_flow

     return max_flow, flows

 # Fonction pour afficher le graphe dans Tkinter
 def draw_graph(graph, flows, source, sink, max_flow):
     G = nx.DiGraph()

     # Ajouter les nœuds et les arêtes
     for u in graph:
         for v in graph[u]:
             G.add_edge(u, v, capacity=graph[u][v], flow=flows[u][v])

     # Définir la mise en page
     pos = nx.circular_layout(G)

     # Dessiner les nœuds et les arêtes
     edge_labels = {(u, v): f"{d['flow']}/{d['capacity']}" for u, v, d in G.edges(data=True)}
     plt.figure(figsize=(8, 6))
     nx.draw(
         G, pos, with_labels=True, node_color="#ff99cc", edge_color="#ff6699", font_weight="bold", 
         node_size=3000, font_color="#660033", font_size=12
     )
     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="#660033")
     plt.title(f"Flot maximum : {max_flow}", fontsize=14, color="#ff3366", fontweight="bold")
     plt.axis('off')
     return plt.gcf()

 # Interface Tkinter
 def main():
     def calculate_flow():
         try:
             num_nodes = int(node_entry.get())
             if num_nodes < 2:
                 raise ValueError("Le nombre de nœuds doit être au moins 2.")
         except ValueError as e:
             result_label.config(text=f"Erreur : {e}", fg="red")
             return
         
         source = 0
         sink = num_nodes - 1
         graph = generate_complete_graph(num_nodes)
         max_flow, flows = ford_fulkerson(graph, source, sink)

         # Afficher le graphe dans l'interface
         fig = draw_graph(graph, flows, source, sink, max_flow)
         canvas = FigureCanvasTkAgg(fig, master=frame)
         canvas.draw()
         canvas.get_tk_widget().pack()
         result_label.config(text=f"Flot maximum entre le nœud {source} et {sink} : {max_flow}", fg="#990066")

     # Création de la fenêtre principale
     root = tk.Tk()
     root.title("Algorithme de Ford-Fulkerson")
     root.configure(bg="#ffe4e1")

     # Style personnalisé
     style = ttk.Style()
     style.configure("TLabel", background="#ffe4e1", foreground="#990066", font=("Georgia", 12))
     style.configure("TButton", background="#ff99cc", font=("Georgia", 12), padding=5)

     # Widgets
     tk.Label(root, text="Nombre de nœuds :", bg="#ffe4e1", fg="#990066", font=("Georgia", 12)).pack(pady=10)
     node_entry = tk.Entry(root, font=("Georgia", 12), justify="center")
     node_entry.pack(pady=5)
     ttk.Button(root, text="Calculer le flot maximum", command=calculate_flow).pack(pady=10)
     result_label = tk.Label(root, text="", bg="#ffe4e1", font=("Georgia", 12, "bold"))
     result_label.pack(pady=10)

     frame = tk.Frame(root, bg="#ffe4e1")
     frame.pack(fill="both", expand=True)

     root.mainloop()

 if __name__ == "__main__":
     main()


def bellman_ford():
    # Couleurs pastel
    BG_COLOR = "#ffe4e1"  # Fond rose pastel
    BUTTON_COLOR = "#f9d4e5"  # Rose clair
    TEXT_COLOR = "#4b0082"  # Indigo


    def bellman_ford(vertices, edges, source):
        distance = {v: float('inf') for v in range(vertices)}
        predecessor = {v: None for v in range(vertices)}
        distance[source] = 0
        iterations = []

        for _ in range(vertices - 1):
            iteration_distances = distance.copy()
            for u, v, w in edges:
                if distance[u] + w < distance[v]:
                    distance[v] = distance[u] + w
                    predecessor[v] = u
            iterations.append(iteration_distances)

        for u, v, w in edges:
            if distance[u] + w < distance[v]:
                raise ValueError("Le graphe contient un cycle de poids négatif")

        return distance, iterations, predecessor


    def generate_random_graph(vertices, edges_count):
        edges = []
        for _ in range(edges_count):
            u = random.randint(0, vertices - 1)
            v = random.randint(0, vertices - 1)
            while u == v:
                v = random.randint(0, vertices - 1)
            w = random.randint(1, 20)
            edges.append((u, v, w))
        return edges


    class BellmanFordApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Algorithme de Bellman-Ford")
            self.root.geometry("800x600")
            self.root.configure(bg=BG_COLOR)

            # Titre
            tk.Label(
                root, text="Algorithme de Bellman-Ford", bg=BG_COLOR, fg=TEXT_COLOR, font=("Arial", 16, "bold")
            ).pack(pady=10)

            # Questions
            tk.Label(root, text="Nombre de sommets :", bg=BG_COLOR, font=("Arial", 12)).pack()
            self.vertices_entry = tk.Entry(root, width=10, font=("Arial", 12))
            self.vertices_entry.pack(pady=5)

            tk.Label(root, text="Nombre d'arêtes :", bg=BG_COLOR, font=("Arial", 12)).pack()
            self.edges_entry = tk.Entry(root, width=10, font=("Arial", 12))
            self.edges_entry.pack(pady=5)

            tk.Label(root, text="Sommet source :", bg=BG_COLOR, font=("Arial", 12)).pack()
            self.source_entry = tk.Entry(root, width=10, font=("Arial", 12))
            self.source_entry.pack(pady=5)

            tk.Label(root, text="Sommet de destination :", bg=BG_COLOR, font=("Arial", 12)).pack()
            self.destination_entry = tk.Entry(root, width=10, font=("Arial", 12))
            self.destination_entry.pack(pady=5)

            # Bouton pour lancer
            tk.Button(
                root, text="Lancer l'algorithme", command=self.run_algorithm, bg=BUTTON_COLOR, font=("Arial", 12, "bold")
            ).pack(pady=15)

            # Cadre pour les résultats
            self.result_frame = tk.Frame(root, bg=BG_COLOR)
            self.result_frame.pack(fill=tk.BOTH, expand=True)

        def run_algorithm(self):
            try:
                # Récupérer les entrées
                vertices = int(self.vertices_entry.get())
                edges_count = int(self.edges_entry.get())
                source = int(self.source_entry.get())
                destination = int(self.destination_entry.get())

                # Vérifications de validité
                if edges_count > vertices * (vertices - 1):
                    raise ValueError(f"Nombre d'arêtes trop élevé. Max autorisé pour {vertices} sommets : {vertices * (vertices - 1)}")
                if not (0 <= source < vertices) or not (0 <= destination < vertices):
                    raise ValueError("Les sommets source et destination doivent être compris entre 0 et le nombre de sommets - 1.")

                # Générer le graphe
                edges = generate_random_graph(vertices, edges_count)

                # Calculer Bellman-Ford
                distances, iterations, predecessor = bellman_ford(vertices, edges, source)

                # Afficher les résultats
                self.display_results(vertices, edges, iterations, distances, source, destination, predecessor)

            except ValueError as e:
                messagebox.showerror("Erreur", str(e))

        def display_results(self, vertices, edges, iterations, distances, source, destination, predecessor):
            # Effacer les résultats précédents
            for widget in self.result_frame.winfo_children():
                widget.destroy()

            # Graphe
            G = nx.DiGraph()
            G.add_weighted_edges_from(edges)
            pos = nx.spring_layout(G)

            # Graphe avec chemin
            fig, ax = plt.subplots(figsize=(8, 6))
            nx.draw(
                G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold'
            )
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): w for u, v, w in edges})

            # Chemin le plus court
            path = []
            current = destination
            while current is not None:
                path.insert(0, current)
                current = predecessor[current]

            edge_colors = ['red' if (u, v) in zip(path, path[1:]) else 'gray' for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
            ax.set_title("Graphe avec le chemin le plus court (Bellman-Ford)", fontsize=14)

            # Insérer le graphe dans Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

            # Tableau des itérations
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            table = Table(ax, bbox=[0, 0, 1, 1])

            # En-tête du tableau
            table.add_cell(0, 0, text="Itération", loc='center', facecolor='lightgray', width=0.1, height=0.1)
            for i in range(vertices):
                table.add_cell(0, i + 1, text=f"d({chr(65 + i)})", loc='center', facecolor='lightgray', width=0.1, height=0.1)

            # Remplissage des données
            for i, iteration in enumerate(iterations):
                table.add_cell(i + 1, 0, text=f"itr{i + 1}", loc='center', width=0.1, height=0.1)
                for j, dist in iteration.items():
                    table.add_cell(i + 1, j + 1, text=f"{dist if dist != float('inf') else '∞'}", loc='center', width=0.1, height=0.1)

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax.add_table(table)

            # Insérer le tableau dans Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()


    if __name__ == "__main__":
        root = tk.Tk()
        app = BellmanFordApp(root)
        root.mainloop()

def moindre_cout():
  # Couleurs pastel
  BG_COLOR = "#ffe4e1"  # Fond rose pastel
  BUTTON_COLOR = "#f9d4e5"  # Rose clair
  TEXT_COLOR = "#4b0082"  # Indigo

  def afficher_tableau(ax, stocks, demandes, couts, transport, barrer, titre, cout_total):
      ax.clear()
      ax.set_title(f"{titre}\nCoût total : {cout_total}", fontsize=14, color=TEXT_COLOR)
      ax.axis('off')

      n, m = len(stocks), len(demandes)
      data = [["" for _ in range(m + 2)] for _ in range(n + 2)]

      data[0][0] = "Sources/Destinataires"
      for j in range(m):
          data[0][j + 1] = f"M{j + 1}"
      data[0][-1] = "Stocks"

      for i in range(n):
          data[i + 1][0] = f"U{i + 1}"
          data[i + 1][-1] = stocks[i]

      data[-1][0] = "Demandes"
      for j in range(m):
          data[-1][j + 1] = demandes[j]

      for i in range(n):
          for j in range(m):
              if transport[i, j] > 0:
                  data[i + 1][j + 1] = f"{couts[i, j]}\n({transport[i, j]})"
              else:
                  data[i + 1][j + 1] = str(couts[i, j])

      cell_colours = [[BG_COLOR for _ in range(m + 2)] for _ in range(n + 2)]
      for (i, j) in barrer:
          cell_colours[i + 1][j + 1] = "#ffcccc"  # Rouge clair

      table = ax.table(cellText=data, cellColours=cell_colours, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
      for key, cell in table.get_celld().items():
          cell.set_text_props(color=TEXT_COLOR)
          cell.set_fontsize(10)
      plt.draw()

  def principe_moindre_cout(stocks, demandes, couts):
      stocks = stocks.copy()
      demandes = demandes.copy()
      n, m = len(stocks), len(demandes)
      transport = np.zeros((n, m), dtype=int)
      barrer = set()
      cout_total = 0

      fig, ax = plt.subplots(figsize=(10, 6))
      fig.patch.set_facecolor(BG_COLOR)

      while True:
          min_cout = float('inf')
          i_min, j_min = -1, -1
          for i in range(n):
              for j in range(m):
                  if (i, j) not in barrer and couts[i, j] < min_cout and stocks[i] > 0 and demandes[j] > 0:
                      min_cout = couts[i, j]
                      i_min, j_min = i, j

          if i_min == -1 or j_min == -1:
              break

          allocation = min(stocks[i_min], demandes[j_min])
          transport[i_min, j_min] = allocation
          stocks[i_min] -= allocation
          demandes[j_min] -= allocation
          cout_total += allocation * min_cout

          if stocks[i_min] == 0:
              for k in range(m):
                  barrer.add((i_min, k))
          if demandes[j_min] == 0:
              for k in range(n):
                  barrer.add((k, j_min))

          afficher_tableau(ax, stocks, demandes, couts, transport, barrer, "Tableau interactif - Résolution en cours", cout_total)
          plt.pause(1)

      afficher_tableau(ax, stocks, demandes, couts, transport, barrer, "Tableau final - Résolution complète", cout_total)
      plt.show()

  def lancer_calcul():
      try:
          n = int(entry_usines.get())
          m = int(entry_magasins.get())
          stocks = list(map(int, entry_stocks.get().split()))
          demandes = list(map(int, entry_demandes.get().split()))
          couts = []
          for row in tableau_couts.get("1.0", "end").strip().split("\n"):
              couts.append(list(map(int, row.split())))

          if len(stocks) != n or len(demandes) != m or len(couts) != n or any(len(row) != m for row in couts):
              raise ValueError("Les données saisies ne correspondent pas aux dimensions spécifiées.")

          principe_moindre_cout(stocks, demandes, np.array(couts))
      except Exception as e:
          messagebox.showerror("Erreur", f"Erreur dans la saisie des données : {e}")

  # Interface graphique avec Tkinter
  root = tk.Tk()
  root.title("Méthode du moindre coût")
  root.configure(bg=BG_COLOR)

  tk.Label(root, text="Nombre d'usines :", bg=BG_COLOR, fg=TEXT_COLOR).grid(row=0, column=0, sticky="w")
  entry_usines = tk.Entry(root, bg=BUTTON_COLOR, fg=TEXT_COLOR)
  entry_usines.grid(row=0, column=1)

  tk.Label(root, text="Nombre de magasins :", bg=BG_COLOR, fg=TEXT_COLOR).grid(row=1, column=0, sticky="w")
  entry_magasins = tk.Entry(root, bg=BUTTON_COLOR, fg=TEXT_COLOR)
  entry_magasins.grid(row=1, column=1)

  tk.Label(root, text="Stocks (séparés par des espaces) :", bg=BG_COLOR, fg=TEXT_COLOR).grid(row=2, column=0, sticky="w")
  entry_stocks = tk.Entry(root, bg=BUTTON_COLOR, fg=TEXT_COLOR)
  entry_stocks.grid(row=2, column=1)

  tk.Label(root, text="Demandes (séparées par des espaces) :", bg=BG_COLOR, fg=TEXT_COLOR).grid(row=3, column=0, sticky="w")
  entry_demandes = tk.Entry(root, bg=BUTTON_COLOR, fg=TEXT_COLOR)
  entry_demandes.grid(row=3, column=1)

  tk.Label(root, text="Tableau des coûts (ligne par ligne) :", bg=BG_COLOR, fg=TEXT_COLOR).grid(row=4, column=0, sticky="nw")
  tableau_couts = tk.Text(root, width=30, height=10, bg=BUTTON_COLOR, fg=TEXT_COLOR)
  tableau_couts.grid(row=4, column=1)

  tk.Button(root, text="Lancer le calcul", bg=BUTTON_COLOR, fg=TEXT_COLOR, command=lancer_calcul).grid(row=5, column=0, columnspan=2)

  root.mainloop()



def welsh_powell():
    def welsh_powell_coloring(graph):
        sorted_nodes = sorted(graph.nodes, key=lambda x: graph.degree[x], reverse=True)
        colors = {}
        current_color = 0

        for node in sorted_nodes:
            if node not in colors:
                current_color += 1
                colors[node] = current_color
                for other_node in sorted_nodes:
                    if other_node not in colors and all(
                        colors.get(neighbor) != current_color for neighbor in graph.neighbors(other_node)
                    ):
                        colors[other_node] = current_color

        return colors, current_color


    def generate_random_graph(num_nodes, num_edges):
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        edges = set()
        while len(edges) < num_edges:
            u, v = random.sample(range(num_nodes), 2)
            if u != v and (u, v) not in edges and (v, u) not in edges:
                edges.add((u, v))
        graph.add_edges_from(edges)
        return graph


    class WelshPowellApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Algorithme de Welsh-Powell")
            self.root.geometry("600x500")
            self.root.configure(bg="#ffe4e1")

            tk.Label(
                root, text="Algorithme de Welsh-Powell", bg="#ffe4e1", fg="#4b0082", font=("Arial", 16, "bold")
            ).pack(pady=10)

            tk.Label(root, text="Nombre de sommets :", bg="#ffe4e1", font=("Arial", 12)).pack()
            self.num_nodes_entry = tk.Entry(root, width=10, font=("Arial", 12))
            self.num_nodes_entry.pack(pady=5)

            tk.Label(root, text="Nombre d'arêtes :", bg="#ffe4e1", font=("Arial", 12)).pack()
            self.num_edges_entry = tk.Entry(root, width=10, font=("Arial", 12))
            self.num_edges_entry.pack(pady=5)

            tk.Button(
                root, text="Lancer l'algorithme", command=self.run_algorithm, bg="#f9d4e5", font=("Arial", 12, "bold")
            ).pack(pady=15)

            self.result_frame = tk.Frame(root, bg="#ffe4e1")
            self.result_frame.pack(fill=tk.BOTH, expand=True)

        def run_algorithm(self):
            try:
                num_nodes = int(self.num_nodes_entry.get())
                num_edges = int(self.num_edges_entry.get())

                max_edges = num_nodes * (num_nodes - 1) // 2
                if num_edges > max_edges:
                    raise ValueError(f"Nombre d'arêtes trop élevé. Max autorisé pour {num_nodes} sommets : {max_edges}")

                graph = generate_random_graph(num_nodes, num_edges)
                colors, chromatic_number = welsh_powell_coloring(graph)
                self.display_results(graph, colors, chromatic_number)
            except ValueError as e:
                messagebox.showerror("Erreur", str(e))

        def display_results(self, graph, colors, chromatic_number):
            for widget in self.result_frame.winfo_children():
                widget.destroy()

            tk.Label(
                self.result_frame,
                text=f"Nombre chromatique : {chromatic_number}",
                bg="#ffe4e1",
                font=("Arial", 14, "bold"),
            ).pack(pady=10)

            tk.Label(self.result_frame, text="Couleurs des sommets :", bg="#ffe4e1", font=("Arial", 12)).pack(pady=5)
            for node, color in colors.items():
                tk.Label(self.result_frame, text=f"Sommets {node} -> Couleur {color}", bg="#ffe4e1", font=("Arial", 12)).pack()

            pos = nx.spring_layout(graph)
            color_map = [colors[node] for node in graph.nodes]
            fig, ax = plt.subplots(figsize=(6, 4))
            nx.draw(
                graph, pos, with_labels=True, node_color=color_map, cmap=plt.cm.rainbow, node_size=500, font_color="white"
            )
            ax.set_title(f"Graph Coloration (Nombre chromatique : {chromatic_number})")

            canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()


    if __name__ == "__main__":
        root = tk.Tk()
        app = WelshPowellApp(root)
        root.mainloop()


def nord_ouest():
    BG_COLOR = "#ffe4e1"  # Fond rose pastel
    BUTTON_COLOR = "#f9d4e5"  # Rose clair
    TEXT_COLOR = "#4b0082"  # Indigo


    # Afficher le tableau interactif
    def afficher_tableau(ax, stocks, demandes, couts, transport, barrer, titre, cout_total):
        ax.clear()
        ax.set_title(f"{titre}\nCoût total : {cout_total}", fontsize=14)
        ax.axis("off")

        n, m = len(stocks), len(demandes)
        data = [["" for _ in range(m + 2)] for _ in range(n + 2)]

        data[0][0] = "Sources/Destinataires"
        for j in range(m):
            data[0][j + 1] = f"M{j + 1}"
        data[0][-1] = "Stocks"

        for i in range(n):
            data[i + 1][0] = f"U{i + 1}"
            data[i + 1][-1] = stocks[i]

        data[-1][0] = "Demandes"
        for j in range(m):
            data[-1][j + 1] = demandes[j]

        for i in range(n):
            for j in range(m):
                if transport[i, j] > 0:
                    data[i + 1][j + 1] = f"{couts[i, j]}\n({transport[i, j]})"
                else:
                    data[i + 1][j + 1] = str(couts[i, j])

        cell_colours = [["white" for _ in range(m + 2)] for _ in range(n + 2)]
        for (i, j) in barrer:
            cell_colours[i + 1][j + 1] = "red"

        table = ax.table(cellText=data, cellColours=cell_colours, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        plt.draw()


    # Résolution par le principe du moindre coût
    def principe_moindre_cout(stocks, demandes, couts):
        stocks = stocks.copy()
        demandes = demandes.copy()
        n, m = len(stocks), len(demandes)
        transport = np.zeros((n, m), dtype=int)
        barrer = set()
        cout_total = 0

        fig, ax = plt.subplots(figsize=(10, 6))

        while True:
            min_cout = float("inf")
            i_min, j_min = -1, -1
            for i in range(n):
                for j in range(m):
                    if (i, j) not in barrer and couts[i, j] < min_cout and stocks[i] > 0 and demandes[j] > 0:
                        min_cout = couts[i, j]
                        i_min, j_min = i, j

            if i_min == -1 or j_min == -1:
                break

            allocation = min(stocks[i_min], demandes[j_min])
            transport[i_min, j_min] = allocation
            stocks[i_min] -= allocation
            demandes[j_min] -= allocation
            cout_total += allocation * min_cout

            if stocks[i_min] == 0:
                for k in range(m):
                    barrer.add((i_min, k))
            if demandes[j_min] == 0:
                for k in range(n):
                    barrer.add((k, j_min))

            afficher_tableau(ax, stocks, demandes, couts, transport, barrer, "Tableau interactif - Résolution en cours", cout_total)
            plt.pause(1)

        afficher_tableau(ax, stocks, demandes, couts, transport, barrer, "Tableau final - Résolution complète", cout_total)
        plt.show()


    # Application Tkinter pour poser des questions graphiques
    class MoindreCoutApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Principe du moindre coût")
            self.root.geometry("800x600")
            self.root.configure(bg=BG_COLOR)

            # Titre
            tk.Label(root, text="Principe du moindre coût", bg=BG_COLOR, fg=TEXT_COLOR, font=("Arial", 16, "bold")).pack(pady=10)

            # Saisie des données
            tk.Label(root, text="Nombre d'usines :", bg=BG_COLOR, font=("Arial", 12)).pack()
            self.num_usines_entry = tk.Entry(root, width=10, font=("Arial", 12))
            self.num_usines_entry.pack(pady=5)

            tk.Label(root, text="Nombre de magasins :", bg=BG_COLOR, font=("Arial", 12)).pack()
            self.num_magasins_entry = tk.Entry(root, width=10, font=("Arial", 12))
            self.num_magasins_entry.pack(pady=5)

            # Bouton pour valider les données
            tk.Button(root, text="Entrer les stocks et demandes", command=self.ask_values, bg=BUTTON_COLOR, font=("Arial", 12, "bold")).pack(pady=10)

        def ask_values(self):
            try:
                n = int(self.num_usines_entry.get())
                m = int(self.num_magasins_entry.get())

                self.stocks = []
                self.demandes = []
                self.couts = []

                # Fenêtre de saisie
                self.value_window = tk.Toplevel(self.root)
                self.value_window.title("Saisir les données")
                self.value_window.geometry("600x400")
                self.value_window.configure(bg=BG_COLOR)

                # Saisie des stocks
                tk.Label(self.value_window, text="Stocks :", bg=BG_COLOR, font=("Arial", 12)).pack()
                for i in range(n):
                    entry = tk.Entry(self.value_window, width=10, font=("Arial", 12))
                    entry.pack(pady=2)
                    self.stocks.append(entry)

                # Saisie des demandes
                tk.Label(self.value_window, text="Demandes :", bg=BG_COLOR, font=("Arial", 12)).pack()
                for j in range(m):
                    entry = tk.Entry(self.value_window, width=10, font=("Arial", 12))
                    entry.pack(pady=2)
                    self.demandes.append(entry)

                # Bouton pour continuer
                tk.Button(self.value_window, text="Saisir le tableau des coûts", command=self.ask_costs, bg=BUTTON_COLOR, font=("Arial", 12, "bold")).pack(pady=10)

            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

        def ask_costs(self):
            try:
                n = len(self.stocks)
                m = len(self.demandes)

                self.cost_window = tk.Toplevel(self.root)
                self.cost_window.title("Tableau des coûts")
                self.cost_window.geometry("600x400")
                self.cost_window.configure(bg=BG_COLOR)

                tk.Label(self.cost_window, text="Entrez le tableau des coûts :", bg=BG_COLOR, font=("Arial", 12)).pack()

                self.cost_entries = []
                for i in range(n):
                    row_entries = []
                    for j in range(m):
                        entry = tk.Entry(self.cost_window, width=10, font=("Arial", 12))
                        entry.pack(side=tk.LEFT, padx=5, pady=5)
                        row_entries.append(entry)
                    tk.Frame(self.cost_window, bg=BG_COLOR).pack()
                    self.cost_entries.append(row_entries)

                tk.Button(self.cost_window, text="Calculer", command=self.calculate, bg=BUTTON_COLOR, font=("Arial", 12, "bold")).pack(pady=10)

            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

        def calculate(self):
            try:
                stocks = [int(entry.get()) for entry in self.stocks]
                demandes = [int(entry.get()) for entry in self.demandes]
                couts = np.array([[int(entry.get()) for entry in row] for row in self.cost_entries])

                principe_moindre_cout(stocks, demandes, couts)

            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")


    # Lancer l'application
    if __name__ == "__main__":
        root = tk.Tk()
        app = MoindreCoutApp(root)
        root.mainloop()



def stepping_stone():
    def afficher_tableau(ax, stocks, demandes, couts, transport, titre, cout_total):
        ax.clear()
        ax.set_title(f"{titre}\nCoût total: {cout_total}", fontsize=18)
        ax.axis("off")

        n, m = len(stocks), len(demandes)
        data = np.zeros((n + 1, m + 2), dtype=object)

        # Entêtes des colonnes et lignes
        data[0, 0] = "Sources/Destinataires"
        for j in range(m):
            data[0, j + 1] = f"M{j + 1}"
        data[0, -1] = "Stocks"

        for i in range(n):
            data[i + 1, 0] = f"U{i + 1}"
            data[i + 1, -1] = stocks[i]
            for j in range(m):
                if transport[i, j] > 0:
                    data[i + 1, j + 1] = f"{couts[i, j]}\n({transport[i, j]})"
                else:
                    data[i + 1, j + 1] = couts[i, j]

        table = ax.table(cellText=data, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(14)


    # Algorithmes de résolution
    def nord_ouest(stocks, demandes, couts):
        n, m = len(stocks), len(demandes)
        transport = np.zeros((n, m), dtype=int)
        i, j = 0, 0

        while i < n and j < m:
            allocation = min(stocks[i], demandes[j])
            transport[i, j] = allocation
            stocks[i] -= allocation
            demandes[j] -= allocation

            if stocks[i] == 0:
                i += 1
            if demandes[j] == 0:
                j += 1

        cout_total = np.sum(transport * couts)
        return transport, cout_total


    def principe_moindre_cout(stocks, demandes, couts):
        stocks = stocks.copy()
        demandes = demandes.copy()
        n, m = len(stocks), len(demandes)
        transport = np.zeros((n, m), dtype=int)
        cout_total = 0

        while True:
            min_cout = float("inf")
            i_min, j_min = -1, -1
            for i in range(n):
                for j in range(m):
                    if couts[i, j] < min_cout and stocks[i] > 0 and demandes[j] > 0:
                        min_cout = couts[i, j]
                        i_min, j_min = i, j

            if i_min == -1 or j_min == -1:
                break

            allocation = min(stocks[i_min], demandes[j_min])
            transport[i_min, j_min] = allocation
            stocks[i_min] -= allocation
            demandes[j_min] -= allocation
            cout_total += allocation * min_cout

        return transport, cout_total


    def comparer_algos(stocks, demandes, couts):
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))

        # Algorithme Nord-Ouest
        transport_nord_ouest, cout_total_nord_ouest = nord_ouest(stocks.copy(), demandes.copy(), couts)
        afficher_tableau(axes[0], stocks.copy(), demandes.copy(), couts, transport_nord_ouest, "Nord-Ouest", cout_total_nord_ouest)

        # Principe du Moindre Coût
        transport_moindre_cout, cout_total_moindre_cout = principe_moindre_cout(stocks.copy(), demandes.copy(), couts)
        afficher_tableau(axes[1], stocks.copy(), demandes.copy(), couts, transport_moindre_cout, "Principe du Moindre Coût", cout_total_moindre_cout)

        plt.tight_layout()
        plt.show()


    # Interface graphique pour poser les questions
    class TransportApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Méthodes de Transport")
            self.root.geometry("600x400")
            self.root.configure(bg="#ffe4e1")

            # Titre
            tk.Label(root, text="Méthodes de Transport", bg="#ffe4e1", fg="#4b0082", font=("Arial", 16, "bold")).pack(pady=10)

            # Nombre d'usines et de magasins
            tk.Label(root, text="Nombre d'usines :", bg="#ffe4e1", font=("Arial", 12)).pack()
            self.num_usines_entry = tk.Entry(root, width=10, font=("Arial", 12))
            self.num_usines_entry.pack(pady=5)

            tk.Label(root, text="Nombre de magasins :", bg="#ffe4e1", font=("Arial", 12)).pack()
            self.num_magasins_entry = tk.Entry(root, width=10, font=("Arial", 12))
            self.num_magasins_entry.pack(pady=5)

            # Bouton pour passer à la saisie des données
            tk.Button(root, text="Saisir les données", command=self.ask_values, bg="#f9d4e5", font=("Arial", 12, "bold")).pack(pady=15)

        def ask_values(self):
            try:
                n = int(self.num_usines_entry.get())
                m = int(self.num_magasins_entry.get())

                self.stocks = []
                self.demandes = []
                self.couts = []

                # Fenêtre pour saisir les données
                self.data_window = tk.Toplevel(self.root)
                self.data_window.title("Saisir les données")
                self.data_window.geometry("600x400")
                self.data_window.configure(bg="#ffe4e1")

                # Saisie des stocks
                tk.Label(self.data_window, text="Stocks :", bg="#ffe4e1", font=("Arial", 12)).pack()
                for i in range(n):
                    entry = tk.Entry(self.data_window, width=10, font=("Arial", 12))
                    entry.pack(pady=2)
                    self.stocks.append(entry)

                # Saisie des demandes
                tk.Label(self.data_window, text="Demandes :", bg="#ffe4e1", font=("Arial", 12)).pack()
                for j in range(m):
                    entry = tk.Entry(self.data_window, width=10, font=("Arial", 12))
                    entry.pack(pady=2)
                    self.demandes.append(entry)

                # Bouton pour passer à la saisie des coûts
                tk.Button(self.data_window, text="Saisir les coûts", command=self.ask_costs, bg="#f9d4e5", font=("Arial", 12, "bold")).pack(pady=15)

            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

        def ask_costs(self):
            try:
                n = len(self.stocks)
                m = len(self.demandes)

                self.cost_window = tk.Toplevel(self.root)
                self.cost_window.title("Saisir les coûts")
                self.cost_window.geometry("600x400")
                self.cost_window.configure(bg="#ffe4e1")

                tk.Label(self.cost_window, text="Entrez le tableau des coûts :", bg="#ffe4e1", font=("Arial", 12)).pack()

                self.cost_entries = []
                for i in range(n):
                    row_entries = []
                    for j in range(m):
                        entry = tk.Entry(self.cost_window, width=10, font=("Arial", 12))
                        entry.pack(side=tk.LEFT, padx=5, pady=5)
                        row_entries.append(entry)
                    tk.Frame(self.cost_window, bg="#ffe4e1").pack()
                    self.cost_entries.append(row_entries)

                tk.Button(self.cost_window, text="Calculer", command=self.calculate, bg="#f9d4e5", font=("Arial", 12, "bold")).pack(pady=15)

            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

        def calculate(self):
            try:
                stocks = [int(entry.get()) for entry in self.stocks]
                demandes = [int(entry.get()) for entry in self.demandes]
                couts = np.array([[int(entry.get()) for entry in row] for row in self.cost_entries])

                comparer_algos(stocks, demandes, couts)

            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")


    # Lancer l'application
    if __name__ == "__main__":
        root = tk.Tk()
        app = TransportApp(root)
        root.mainloop()


def potentiel_Metra():
   # Fonction pour demander le nombre de tâches à l'utilisateur
   def ask_number_of_tasks():
       root = tk.Tk()
       root.withdraw()  # Masquer la fenêtre principale
       num_tasks = simpledialog.askinteger("Nombre de tâches", 
                                            "Combien de tâches souhaitez-vous ?",
                                            minvalue=1, maxvalue=20)
       return num_tasks

   # Fonction pour générer les tâches et les relations
   def generate_tasks_and_dependencies(num_tasks):
       tasks = {'debut': {'duration': 0, 'precedents': []}}  # La tâche "debut"
       edges = []

       for i in range(num_tasks):
           task_name = chr(65 + i)  # Tâches A, B, C, ...
           duration = random.randint(1, 5)  # Durée aléatoire entre 1 et 5
           precedents = random.sample(list(tasks.keys()), k=random.randint(1, min(3, len(tasks))))
           tasks[task_name] = {'duration': duration, 'precedents': precedents}

           for precedent in precedents:
               edges.append((precedent, task_name))

       # Déterminer les tâches finales et connecter à "fin"
       final_tasks = [task for task in tasks.keys() if task not in [edge[0] for edge in edges]]
       tasks['fin'] = {'duration': 0, 'precedents': final_tasks}
       for task in final_tasks:
           edges.append((task, 'fin'))

       return tasks, edges

   # Fonction pour calculer les heures de début au plus tôt et au plus tard
   def calculate_times(tasks):
       earliest_start = {}
       latest_start = {}
       total_duration = 0

       # Calcul du début au plus tôt (forward pass)
       for task, attrs in tasks.items():
           if not attrs['precedents']:
               earliest_start[task] = 0
           else:
               earliest_start[task] = max(earliest_start[prec] + tasks[prec]['duration'] for prec in attrs['precedents'])

       total_duration = max(earliest_start[task] + attrs['duration'] for task, attrs in tasks.items())

       # Calcul du début au plus tard (backward pass)
       for task in reversed(list(tasks.keys())):
           if task == 'fin':
               latest_start[task] = total_duration - tasks[task]['duration']
           else:
               successors = [succ for succ, attrs in tasks.items() if task in attrs['precedents']]
               latest_start[task] = min(latest_start[succ] - tasks[task]['duration'] for succ in successors)

       return earliest_start, latest_start, total_duration

   # Trouver le chemin critique
   def find_critical_path(earliest_start, latest_start):
       return [task for task in earliest_start if earliest_start[task] == latest_start[task]]

   # Demander le nombre de tâches
   num_tasks = ask_number_of_tasks()

   # Générer les tâches et relations
   tasks, edges = generate_tasks_and_dependencies(num_tasks)

   # Calculer les temps
   earliest_start, latest_start, total_duration = calculate_times(tasks)
   critical_path = find_critical_path(earliest_start, latest_start)

   # Création du graphe
   G = nx.DiGraph()

   # Ajouter les nœuds et arêtes
   for task, attrs in tasks.items():
       G.add_node(task, duration=attrs['duration'])
   for u, v in edges:
       G.add_edge(u, v)

   # Mise en page
   pos = nx.spring_layout(G, seed=42)

   # Dessiner le graphe
   plt.figure(figsize=(12, 8))

   # Coloration des nœuds
   node_colors = []
   for node in G.nodes():
       if node in critical_path:
           node_colors.append("#ff0000")  # Rouge pour le chemin critique
       elif node == 'debut':
           node_colors.append("#b3e5fc")  # Bleu clair pour 'debut'
       elif node == 'fin':
           node_colors.append("#c8e6c9")  # Vert clair pour 'fin'
       else:
           node_colors.append("#f5f5f5")  # Gris clair pour les autres

   nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, edgecolors="#424242", node_shape='s')

   # Dessiner les arêtes
   edge_colors = ["#ff0000" if u in critical_path and v in critical_path else "#b0bec5" for u, v in G.edges()]
   nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowstyle='-|>', arrowsize=20)

   # Ajouter les étiquettes
   labels = {node: f"{node}\nES: {earliest_start.get(node, 'N/A')}\nLS: {latest_start.get(node, 'N/A')}" for node in G.nodes()}
   nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold", font_family="Georgia", font_color="#424242")

   # Ajouter un titre
   plt.title("Graphe MPM - Chemin critique en rouge", fontsize=16, fontweight="bold", fontname="Georgia", color="#ff5722")
   plt.text(-1, -1.5, f"Chemin critique : {' -> '.join(critical_path)}\nDurée totale : {total_duration}", 
            fontsize=12, fontweight="bold", color="#424242")

   plt.axis("off")
   plt.show()



def entree():
    messagebox.showinfo("Bienvenue", "Bienvenue dans l'application des algorithmes!")
    afficher_algorithmes()

def quitter():
    root.quit()

def afficher_algorithmes():
    """Affiche la liste des algorithmes disponibles de manière équilibrée."""
    root_algorithms = Toplevel()
    root_algorithms.title("Choisissez un algorithme")
    root_algorithms.geometry("900x750")
    root_algorithms.configure(bg="#FFE4EC")
    
    header = tk.Label(
        root_algorithms,
        text="🌸 Algorithmes Disponibles 🌸",
        font=("Comic Sans MS", 22, "bold"),
        bg="#FFE4EC",
        fg="#D63384",
    )
    header.pack(pady=20)
    
    algorithms = {
        "Welsh Powell 🚀": welsh_powell,
        "Kruskal 🔗": kruskal,
        "Bellman-Ford 🌐": bellman_ford,
        "Dijkstra 🏁": dijkstra,
        "Nord Ouest 🚛": nord_ouest,
        "Stepping Stone 🔄": stepping_stone,
        "Potentiel Metra ⚖": potentiel_Metra,
        "Ford-Fulkerson 💧": ford_fulkerson,
        "Moindre Coût 💲": moindre_cout,
    }
    
    button_frame = tk.Frame(root_algorithms, bg="#FFE4EC")
    button_frame.pack(pady=10)
    
    cols = 2
    buttons = list(algorithms.items())
    rows = (len(buttons) + cols - 1) // cols  # Calcul pour équilibrer les lignes
    
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(buttons):
                name, func = buttons[index]
                tk.Button(
                    button_frame,
                    text=name,
                    command=func,
                    font=("Comic Sans MS", 14, "bold"),
                    bg="#FFB6C1",
                    fg="#FFFFFF",
                    relief="flat",
                    width=22,
                    height=2,
                    bd=3
                ).grid(row=i, column=j, padx=10, pady=10)
    
    quitter_button = tk.Button(
        root_algorithms,
        text="🌸 Fermer 🌸",
        command=root_algorithms.destroy,
        font=("Comic Sans MS", 14, "bold"),
        bg="#D63384",
        fg="#FFFFFF",
        relief="flat",
        width=15,
    )
    quitter_button.pack(pady=20)

root = tk.Tk()
root.title("🌸 Interface de Bienvenue 🌸")
root.geometry("800x600")
root.config(bg="#FFE4EC")

try:
    my_image = PhotoImage(file=r"C:\Users\hp\Desktop\Formation carriere professionnelle\logo-EMSI.png")
    lbl = Label(root, image=my_image, bg="#FFE4EC")
    lbl.pack(pady=10)
except Exception as e:
    print("Error loading image:", e)



bienvenue_label = tk.Label(
    root,
    text="Bienvenue ! 🌸",
    font=("Comic Sans MS", 28, "bold"),
    bg="#FFE4EC",
    fg="#D63384",
)
bienvenue_label.pack()

info_frame = tk.Frame(root, bg="#FFE4EC")
info_frame.pack()

tk.Label(
    info_frame,
    text="Réalisé par : Imane Mourid ",
    font=("Georgia", 16, "bold"),
    bg="#FFE4EC",
    fg="#4A4A4A",
).grid(row=0, column=0)

tk.Label(
    info_frame,
    text="Encadré par : Mouna El Mkhalet",
    font=("Georgia", 16, "bold"),
    bg="#FFE4EC",
    fg="#4A4A4A",
).grid(row=1, column=0)

button_frame = tk.Frame(root, bg="#FFE4EC")
button_frame.pack()

entree_button = tk.Button(
    button_frame,
    text="✨ Entrée ✨",
    command=entree,
    font=("Comic Sans MS", 16, "bold"),
    bg="#FF69B4",
    fg="#FFFFFF",
    relief="flat",
    width=15,
)
entree_button.grid(row=0, column=0, padx=10, pady=5)

quitter_button = tk.Button(
    button_frame,
    text="❌ Sortie ❌",
    command=quitter,
    font=("Comic Sans MS", 16, "bold"),
    bg="#D63384",
    fg="#FFFFFF",
    relief="flat",
    width=15,
)
quitter_button.grid(row=0, column=1, padx=10, pady=5)

root.mainloop()