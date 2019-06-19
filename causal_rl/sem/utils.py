import networkx as nx
import matplotlib.pyplot as plt


def draw(graph, path=None):
    # transpose because nx edges are (from, to) and DAGs are (to, from)
    edges = graph.t().nonzero().numpy()

    # initialize graph
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # plot graph
    plt.subplot(111)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos=pos, with_labels=True)

    # save or show based on provided path
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
