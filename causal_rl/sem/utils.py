import networkx as nx
import matplotlib.pyplot as plt

def draw(graph):
    # transpose because nx edges are (from, to) and DAGs are (to, from)
    edges = graph.T.nonzero() 
    edges = zip(*edges)

    G = nx.DiGraph()
    G.add_edges_from(edges)

    plt.subplot(111)
    nx.draw(G, with_labels=True)
    plt.show()