import matplotlib.pyplot as plt
from data import Graph
import networkx as nx
import logging


def plot_graphs(g1: Graph, g2: Graph, dataset_name: str = None, save_path: str = None, fixed_layout: bool = False):
    """
    Plot two graphs side by side.
    :param g1: Graph 1
    :param g2: Graph 2
    :param dataset_name: Name of the dataset the graphs are from.
    :param save_path: If specified, save the plot to this path. Else show the plot on screen.
    :param fixed_layout: If True, use a fixed layout for the graphs, instead of a random one. Makes the graphs consistent across runs, but less pretty.
    """

    logging.info("Plotting graphs...")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # plot graph 1
    _plot_single_graph(axs[0], g1, "Graph 1", fixed_layout=fixed_layout)

    # plot graph 2
    _plot_single_graph(axs[1], g2, "Graph 2", fixed_layout=fixed_layout)

    # set figure attributes
    dataset_label = f" ({dataset_name})" if dataset_name is not None else ""
    fig.suptitle(f"Pair {g1.gid()}-{g2.gid()} {dataset_label}")
    plt.show()


def _plot_single_graph(ax, g: Graph, title: str, fixed_layout: bool = False):
    # create a fixed nx layout (to keep plot consistent on different renders)
    pos = nx.spring_layout(g) if fixed_layout else None

    # draw
    nx.draw(g, pos=pos, ax=ax, with_labels=True)
    ax.set_title(title)
