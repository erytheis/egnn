from copy import deepcopy
from numbers import Number

import pandas as pd
import wntr
from matplotlib import pyplot as plt
import matplotlib as mpl
from wntr.graphics import custom_colormap
from wntr.graphics.network import _format_node_attribute, _format_link_attribute
import os


pretty_names = {'l-town': 'L-TOWN',
                'net-3': 'NET-3',
                'zj': 'ZJ',
                'pes': 'PES',
                'mod': 'MOD',
                'marchirural': 'RUR',
                'kl': 'KL',
                'bak': 'BAK',
                'asnet2': 'ASnet2',
                'asnet': 'ASnet2',
                'jilin': 'JILIN',
                'apulia_decreased_demands': 'APULIA',
                'apulia': 'APULIA',
                }


def get_hydraulic_resistance(length, diameter, roughness):
    # loss_coeffient = 1 / (length / ((roughness ** 1.852) * (diameter ** 4.871))) * 10.667

    # for comparison
    other_loss = 10.667 * length / roughness ** 1.852 / diameter ** 4.871
    return other_loss


def derive_hazen_williams_coefficient(length, diameter, hydraulic_resistance, flowrates):
    return (10.667 * length / (hydraulic_resistance * diameter ** 4.871)) ** (1 / 1.852)


def reverse_edge(graph, node1, node2):
    for key, data in graph.get_edge_data(node1, node2).items():
        graph.add_edge(node2, node1, key=key, **(deepcopy(data)))
    graph.remove_edge(node1, node2)


import networkx as nx
# import graphlets
import numpy as np


class CurrentFlowBetweennessCentrality:

    def __str__(self):
        return 'current_flow_betweenness_centrality_subset'

    @classmethod
    def encode(self, G, **kwargs) -> np.ndarray:
        centrality = nx.current_flow_betweenness_centrality_subset(G, **kwargs, solver='lu')
        # centrality = np.array(centrality)
        return centrality


class InvWeightCurrentFlowBetweennessCentrality:

    def __str__(self):
        return 'current_flow_betweenness_centrality_subset'

    @classmethod
    def encode(self, G, **kwargs) -> np.ndarray:
        for u, v in G.edges():
            G[u][v]['new_weight'] = 1 / G[u][v]['weight']
        kwargs['weight'] = 'new_weight'
        centrality = nx.current_flow_betweenness_centrality_subset(G, **kwargs, solver='lu')
        # centrality = np.array(centrality)
        return centrality


class InformationCentrality():

    def __str__(self):
        return 'information_centrality'

    @classmethod
    def encode(self, G, weight, **kwargs) -> np.ndarray:
        centrality = nx.current_flow_closeness_centrality(G, weight)
        return centrality


class HydraulicCosts():

    def __str__(self):
        return 'hydraulic_costs'

    @classmethod
    def encode(self, G, sources, weight, **kwargs):
        costs = nx.multi_source_dijkstra_path_length(G, sources, weight=weight)
        # costs[costs == np.inf] = 0  # replace inf with 0
        # costs[costs == -np.inf] = 0  # replace inf with 0=
        return costs


class EigenvectorCentrality():

    def __str__(self):
        return 'eigenvector_centrality'

    @classmethod
    def encode(self, G, weight, **kwargs) -> np.ndarray:
        centrality = nx.eigenvector_centrality(G, weight=weight)
        return centrality


def plot_network(wn, node_attribute=None, link_attribute=None, title=None,
                 node_size=20, node_range=[None, None], node_alpha=1, node_cmap=None, node_labels=False,
                 link_width=1, link_range=[None, None], link_alpha=1, link_cmap=None, link_labels=None,
                 add_colorbar=True, node_colorbar_label='Node', link_colorbar_label='Link',
                 edge_color=None, scatter_kwargs={}, link_label_kwargs= {},
                 directed=False, ax=None, filename=None, pos=None, show=True, flip_negative_link_attributes=False,
                 flip_links=None):
    """
    Plot network graphic

    Parameters
    ----------
    wn : wntr WaterNetworkModel
        A WaterNetworkModel object

    node_attribute : None, str, list, pd.Series, or dict, optional

        - If node_attribute is a string, then a node attribute dictionary is
          created using node_attribute = wn.query_node_attribute(str)
        - If node_attribute is a list, then each node in the list is given a
          value of 1.
        - If node_attribute is a pd.Series, then it should be in the format
          {nodeid: x} where nodeid is a string and x is a float.
        - If node_attribute is a dict, then it should be in the format
          {nodeid: x} where nodeid is a string and x is a float

	link_attribute : None, str, list, pd.Series, or dict, optional

        - If link_attribute is a string, then a link attribute dictionary is
          created using edge_attribute = wn.query_link_attribute(str)
        - If link_attribute is a list, then each link in the list is given a
          value of 1.
        - If link_attribute is a pd.Series, then it should be in the format
          {linkid: x} where linkid is a string and x is a float.
        - If link_attribute is a dict, then it should be in the format
          {linkid: x} where linkid is a string and x is a float.

    title: str, optional
        Plot title

    node_size: int, optional
        Node size

    node_range: list, optional
        Node range ([None,None] indicates autoscale)

    node_alpha: int, optional
        Node transparency

    node_cmap: matplotlib.pyplot.cm colormap or list of named colors, optional
        Node colormap

    node_labels: bool, optional
        If True, the graph will include each node labelled with its name.

    link_width: int, optional
        Link width

    link_range : list, optional
        Link range ([None,None] indicates autoscale)

    link_alpha : int, optional
        Link transparency

    link_cmap: matplotlib.pyplot.cm colormap or list of named colors, optional
        Link colormap

    link_labels: bool, optional
        If True, the graph will include each link labelled with its name.

    add_colorbar: bool, optional
        Add colorbar

    node_colorbar_label: str, optional
        Node colorbar label

    link_colorbar_label: str, optional
        Link colorbar label

    directed: bool, optional
        If True, plot the directed graph

    ax: matplotlib axes object, optional
        Axes for plotting (None indicates that a new figure with a single
        axes will be used)

    filename : str, optional
        Filename used to save the figure

    Returns
    -------
    ax : matplotlib axes object
    """

    if ax is None:  # create a new figure
        plt.figure(facecolor='w', edgecolor='k')
        ax = plt.gca()

    # Graph
    G = wn.get_graph()
    if not directed:
        G = G.to_undirected()
    else:
        if flip_links is not None:
            edges = [e for e in G.edges()]
            for i, link in enumerate(edges):
                if flip_links[i] == -1:
                    reverse_edge(G, link[0], link[1])

    # Position
    if pos is None:
        pos = nx.get_node_attributes(G, 'pos')
        if len(pos) == 0:
            pos = None

    # Define node properties
    add_node_colorbar = add_colorbar
    if node_attribute is not None:

        if isinstance(node_attribute, list):
            if node_cmap is None:
                node_cmap = ['red', 'red']
            add_node_colorbar = False

        if node_cmap is None:
            node_cmap = plt.get_cmap('Spectral_r')
        elif isinstance(node_cmap, list):
            if len(node_cmap) == 1:
                node_cmap = node_cmap * 2
            node_cmap = custom_colormap(len(node_cmap), node_cmap)

        node_attribute = _format_node_attribute(node_attribute, wn)
        nodelist, nodecolor = zip(*node_attribute.items())

    else:
        nodelist = None
        nodecolor = 'k'

    add_link_colorbar = add_colorbar
    if link_attribute is not None and edge_color is None:

        if isinstance(link_attribute, list):
            if link_cmap is None:
                link_cmap = ['red', 'red']
            add_link_colorbar = False

        if link_cmap is None:
            link_cmap = plt.get_cmap('Spectral_r')
        elif isinstance(link_cmap, list):
            if len(link_cmap) == 1:
                link_cmap = link_cmap * 2
            link_cmap = custom_colormap(len(link_cmap), link_cmap)

        link_attribute = _format_link_attribute(link_attribute, wn)

        # Replace link_attribute dictionary defined as
        # {link_name: attr} with {(start_node, end_node, link_name): attr}
        attr = {}
        for i, (link_name, value) in enumerate(link_attribute.items()):
            link = wn.get_link(link_name)
            # flip direction of link if necessary
            if directed:
                to_flip = ((value < 0) and flip_negative_link_attributes)
                if to_flip:
                    attr[(link.end_node_name, link.start_node_name, link_name)] = - value
                else:
                    attr[(link.start_node_name, link.end_node_name, link_name)] = value
            else:
                attr[(link.start_node_name, link.end_node_name, link_name)] = value

        link_attribute = attr

        linklist, linkcolor = zip(*link_attribute.items())

    else:
        linklist = None
        linkcolor = edge_color or 'k'

    if title is not None:
        ax.set_title(title)

    edge_background = nx.draw_networkx_edges(G, pos, edge_color='grey',
                                             width=0.5, ax=ax)

    nodes = nx.draw_networkx_nodes(G, pos,
                                   nodelist=nodelist, node_color=nodecolor, node_size=node_size,
                                   alpha=node_alpha, cmap=node_cmap, vmin=node_range[0], vmax=node_range[1],
                                   linewidths=0, ax=ax)
    edges = nx.draw_networkx_edges(G, pos, edgelist=linklist,
                                   edge_color=linkcolor, width=link_width, alpha=link_alpha, edge_cmap=link_cmap,
                                   edge_vmin=link_range[0], edge_vmax=link_range[1], ax=ax)
    if node_labels:
        labels = dict(zip(wn.node_name_list, wn.node_name_list))
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    if link_labels is not None:
        labels = {}
        assert len(link_labels) == len(wn.link_name_list)
        for i, link_name in enumerate(wn.link_name_list):
            link = wn.get_link(link_name)
            labels[(link.start_node_name, link.end_node_name)] = link_labels[i]
        nx.draw_networkx_edge_labels(G, pos, labels,  ax=ax,  **link_label_kwargs)
    if add_node_colorbar and node_attribute:
        clb = plt.colorbar(nodes, shrink=0.5, pad=0, ax=ax)
        clb.ax.set_title(node_colorbar_label, fontsize=10)
    if add_link_colorbar and link_attribute:
        if directed:
            vmin = min(map(abs, link_attribute.values()))
            vmax = max(map(abs, link_attribute.values()))
            sm = plt.cm.ScalarMappable(cmap=link_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            clb = plt.colorbar(sm, shrink=0.5, pad=0.05, ax=ax)
        else:
            clb = plt.colorbar(edges, shrink=0.5, pad=0.05, ax=ax)
        clb.ax.set_title(link_colorbar_label, fontsize=10)


    # nicer junctions
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    ax.scatter(*node_xyz.T, s=node_size, ec="w", cmap=node_cmap, **scatter_kwargs)


    # add reservoirs
    reservoir = wn.reservoir_name_list
    pos_reservoir = {node: pos[node] for node in reservoir}
    G_ = G.subgraph(reservoir)
    nx.draw_networkx_nodes(G_, pos_reservoir, node_size=50, node_color='k', node_shape='s', ax=ax)

    ax.axis('off')

    if filename:
        plt.savefig(filename)

    if show:
        plt.show(block=False)

    return ax


def plot_network_3d(wn=None, node_attribute=None, edge_attribute=None, title=None,
                    node_size=20, node_range=[None, None], node_alpha=1, node_cmap=None, node_labels=False,
                    node_color=None,
                    edge_witdth=1, edge_range=[None, None], link_alpha=1, edge_cmap=None, link_labels=False,
                    add_colorbar=True, node_colorbar_label='Node', link_colorbar_label='Link', link_width=1,
                    directed=False, ax=None, filename=None, pos=None, show=True, flip_negative_link_attributes=False,
                    flip_links=None, edge_color=None, G=None, scatter_label=None, lineplot_kwargs={},
                    scatter_kwargs={},
                    link_plot_label=None):
    """
    Plot network graphic

    Parameters
    ----------
    wn : wntr WaterNetworkModel
        A WaterNetworkModel object

    node_attribute : None, str, list, pd.Series, or dict, optional

        - If node_attribute is a string, then a node attribute dictionary is
          created using node_attribute = wn.query_node_attribute(str)
        - If node_attribute is a list, then each node in the list is given a
          value of 1.
        - If node_attribute is a pd.Series, then it should be in the format
          {nodeid: x} where nodeid is a string and x is a float.
        - If node_attribute is a dict, then it should be in the format
          {nodeid: x} where nodeid is a string and x is a float

	link_attribute : None, str, list, pd.Series, or dict, optional

        - If link_attribute is a string, then a link attribute dictionary is
          created using edge_attribute = wn.query_link_attribute(str)
        - If link_attribute is a list, then each link in the list is given a
          value of 1.
        - If link_attribute is a pd.Series, then it should be in the format
          {linkid: x} where linkid is a string and x is a float.
        - If link_attribute is a dict, then it should be in the format
          {linkid: x} where linkid is a string and x is a float.

    title: str, optional
        Plot title

    node_size: int, optional
        Node size

    node_range: list, optional
        Node range ([None,None] indicates autoscale)

    node_alpha: int, optional
        Node transparency

    node_cmap: matplotlib.pyplot.cm colormap or list of named colors, optional
        Node colormap

    node_labels: bool, optional
        If True, the graph will include each node labelled with its name.

    link_width: int, optional
        Link width

    link_range : list, optional
        Link range ([None,None] indicates autoscale)

    link_alpha : int, optional
        Link transparency

    link_cmap: matplotlib.pyplot.cm colormap or list of named colors, optional
        Link colormap

    link_labels: bool, optional
        If True, the graph will include each link labelled with its name.

    add_colorbar: bool, optional
        Add colorbar

    node_colorbar_label: str, optional
        Node colorbar label

    link_colorbar_label: str, optional
        Link colorbar label

    directed: bool, optional
        If True, plot the directed graph

    ax: matplotlib axes object, optional
        Axes for plotting (None indicates that a new figure with a single
        axes will be used)

    filename : str, optional
        Filename used to save the figure

    Returns
    -------
    ax : matplotlib axes object
    """

    # Graph
    if wn is None:
        assert G is not None
    else:
        G = wn.get_graph()

    if not directed:
        G = G.to_undirected()
    else:
        if flip_links is not None:
            edges = [e for e in G.edges()]
            for i, link in enumerate(edges):
                if flip_links[i] == -1:
                    reverse_edge(G, link[0], link[1])

    # Position
    if pos is None:
        pos = nx.get_node_attributes(G, 'pos')
        pos = {name: (*p, 0) for name, p in pos.items()}
        if len(pos) == 0:
            pos = None

    if title is not None:
        ax.set_title(title)

    edge_vmin, edge_vmax = edge_range[0], edge_range[1]

    add_node_colorbar = add_colorbar
    if node_attribute is not None:

        if isinstance(node_attribute, list):
            if node_cmap is None:
                node_cmap = ['red', 'red']
            add_node_colorbar = False

        if node_cmap is None:
            node_cmap = plt.get_cmap('Spectral_r')
        elif isinstance(node_cmap, list):
            if len(node_cmap) == 1:
                node_cmap = node_cmap * 2
            node_cmap = custom_colormap(len(node_cmap), node_cmap)

        node_attribute = _format_node_attribute(node_attribute, wn)
        node_list, node_color = zip(*node_attribute.items())

    add_link_colorbar = add_colorbar

    if edge_attribute is not None:


        if edge_cmap is None:
            edge_cmap = plt.get_cmap('Spectral_r')
        elif isinstance(edge_cmap, list):
            if len(edge_cmap) == 1:
                edge_cmap = edge_cmap * 2
            edge_cmap = custom_colormap(len(edge_cmap), edge_cmap)

        edge_attribute = _format_link_attribute(edge_attribute, wn)

        # Replace link_attribute dictionary defined as
        # {link_name: attr} with {(start_node, end_node, link_name): attr}
        if wn is not None and isinstance(edge_attribute, dict) :
            attr = {}
            for i, (link_name, value) in enumerate(edge_attribute.items()):
                link = wn.get_link(link_name)
                # flip direction of link if necessary
                if directed:
                    to_flip = ((value < 0) and flip_negative_link_attributes)
                    if to_flip:
                        attr[(link.end_node_name, link.start_node_name, link_name)] = - value
                    else:
                        attr[(link.start_node_name, link.end_node_name, link_name)] = value
                else:
                    attr[(link.start_node_name, link.end_node_name, link_name)] = value

            edge_attribute = attr

            edge_list, edge_color = zip(*edge_attribute.items())
        elif edge_color is not None:
            edge_list = [(k[0], k[1], 'n') for k, v in edge_attribute.items()]
            edge_color = [edge_color for k in edge_list]
        else:
            edge_list =  [(k[0], k[1], 'n') for k, v in edge_attribute.items()]
            edge_color = [v for k, v in edge_attribute.items()]

    else:
        edge_list = [(u, v, 'n') for u, v in G.edges()]
        edge_color = [edge_color for k in edge_list]

    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in G])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v, name in edge_list])

    # node_color = [n  i in range(len(node_xyz))]

    # Get colors
    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
            np.iterable(edge_color)
            and (len(edge_color) == len(edge_xyz)) and len(edge_color) > 0
            and np.alltrue([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, mpl.colors.Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_range[0] is None:
            edge_vmin = min(edge_color)
        if edge_range[1] is None:
            edge_vmax = max(edge_color)
        color_normal = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    # Create the 3D figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    node_vmin, node_vmax = node_range[0], node_range[1]

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=node_size,  label=scatter_label, c=node_color, **scatter_kwargs, cmap=node_cmap,
               vmin=node_vmin, vmax=node_vmax)

    # Plot the edges
    for i, vizedge in enumerate(edge_xyz):
        if i == 0:
            label = link_plot_label
        else:
            label = None
        ax.plot(*vizedge.T, color=edge_color[i], alpha=link_alpha, linewidth=link_width, label=label, **lineplot_kwargs)

    if node_labels:
        labels = dict(zip(wn.node_name_list, wn.node_name_list))
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    if link_labels:
        labels = {}
        for link_name in wn.link_name_list:
            link = wn.get_link(link_name)
            labels[(link.start_node_name, link.end_node_name)] = link_name
        nx.draw_networkx_edge_labels(G, pos, labels, font_size=7, ax=ax)

    ax.axis('off')

    if filename:
        plt.savefig(filename)

    if show:
        plt.show(block=False)

    return ax


def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
