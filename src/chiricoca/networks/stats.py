import networkx as nx


def summary(G, root_node=None):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    n_components = (
        nx.number_connected_components(G)
        if not nx.is_directed(G)
        else nx.number_weakly_connected_components(G)
    )

    result = {
        "nodes": n_nodes,
        "edges": n_edges,
        "avg_degree": sum(dict(G.degree()).values()) / n_nodes,
        "density": nx.density(G),
        "is_tree": nx.is_tree(G),
        "is_directed": nx.is_directed(G),
        "n_connected_components": n_components,
    }

    if result["is_tree"] and root_node is not None:
        heights = nx.single_source_shortest_path_length(G, root_node)
        result["height"] = max(heights.values())

        # Contar nodos por nivel para obtener ancho máximo
        levels = {}
        for node, depth in heights.items():
            levels[depth] = levels.get(depth, 0) + 1
        result["max_width"] = max(levels.values())

        # Ratio de hojas
        leaves = sum(1 for node in G.nodes() if G.degree(node) == 1 and node != "ROOT")
        result["leaf_ration"] = leaves / n_nodes

    return result
