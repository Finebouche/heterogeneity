import warnings
import graphviz

# color dictionary for nodes activations and related colors (different for each activation : sigmoid tanh sin gauss relu softplus identity clamped abs hat)
activation_colors = {
    'sigmoid': 'yellow',
    'tanh': 'orange',
    'sin': 'red',
    'gauss': 'purple',
    'relu': 'blue',
    'softplus': 'green',
    'identity': 'black',
    'clamped': 'brown',
    'abs': 'pink',
    'hat': 'cyan'
}

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '10',
        'height': '0.1',
        'width': '0.1',
        'label': '',
        'fixedsize': 'true'
    }

    graph_attrs = {
        'rankdir': 'LR',          # Left to right direction
        'ranksep': '3.0',         # Increase the distance between ranks
        'nodesep': '0.1',         # Increase the distance between nodes
        'outputorder': 'edgesfirst',  # Draw edges first
    }

    net_graph = graphviz.Digraph(format=fmt, node_attr=node_attrs, graph_attr=graph_attrs)


    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        input_attrs = {'style': 'filled', 'fillcolor': 'white', 'xlabel': node_names.get(k, str(k))}
        net_graph.node(str(k), _attributes=input_attrs)  # Move label outside the node using xlabel

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        output_attrs = {'style': 'filled', 'fillcolor': 'lightblue', 'xlabel': node_names.get(k, str(k))}
        net_graph.node(str(k), _attributes=output_attrs)  # Move label outside the node using xlabel

    used_nodes = set(genome.nodes.keys())
    for k in used_nodes:
        if k in inputs or k in outputs:
            continue

        attrs = {
            'style': 'filled',
            'fillcolor': activation_colors[genome.nodes[k].activation],
            'xlabel': genome.nodes[k].activation,
        }
        net_graph.node(str(k), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            attrs = {
                'style': 'solid' if cg.enabled else 'dotted',
                'color': 'lightblue',
                'penwidth': str(0.1 + abs(cg.weight / 3.0)),
                'arrowhead': 'none'  # No arrowhead on the edge
            }
            net_graph.edge(str(cg.key[0]), str(cg.key[1]), _attributes=attrs)

    # Enforce input nodes to be on the first layer and output nodes on the last layer
    input_node_names = [str(k) for k in config.genome_config.input_keys]
    net_graph.body.append('{rank=min; ' + '; '.join(input_node_names) + ';}')

    output_node_names = [str(k) for k in config.genome_config.output_keys]
    net_graph.body.append('{rank=max; ' + '; '.join(output_node_names) + ';}')

    if filename:
        rendering = net_graph.render(filename, view=view, format=fmt)
    else:
        rendering = net_graph.pipe(format=fmt)

    return rendering


if __name__ == '__main__':
    import pickle
    import neat
    import os

    # load best genome from visualisations/CartRacing-v3/best_genome.pickle
    local_dir = os.path.dirname(__file__)
    result_path = os.path.join(local_dir, "visualisations", "CarRacing-v3")
    with open(os.path.join(result_path, 'best_genome.pickle'), 'rb') as f:
        gen_best = pickle.load(f)

    # load config file from config_files/config-CartRacing-v3
    config_path = os.path.join(local_dir, "config_files", "config-car_racing")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    draw_net(config, gen_best, view=False, fmt="svg", filename=result_path + "/win-net.gv")
