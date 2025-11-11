import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast  # for safely parsing architecture strings

if __name__ == '__main__':
    df = pd.read_csv('results_unordered_10.csv')

    # Parse activation function list from the string representation
    df['activation_funcs'] = df['activation_functions'].apply(
        lambda x: x[1:-1].split(', ')
    )

    # Heterogeneity = number of distinct activations in the hidden layer(s)
    df['heterogeneity_score'] = df['activation_funcs'].apply(lambda x: len(set(x)))

    # Parse architecture to count layers (e.g. "[3, 2]" -> [3, 2] -> 2 layers)
    df['num_layers'] = df['architecture'].apply(
        lambda s: len(ast.literal_eval(s)) if isinstance(s, str) else len(s)
    )

    print("Best accuracies for homogeneous networks:")
    print(df[df['heterogeneity_score'] == 1]["best_accuracy"].describe())

    # Jitter settings
    rng = np.random.default_rng(seed=0)
    jitter_scale = 0.04

    # Marker per number of layers
    layer_markers = {
        1: 'o',   # one layer
        2: 's',   # two layers
        3: '^',   # (if ever present)
        4: 'D',   # ...
    }

    unique_layers = sorted(df['num_layers'].unique())

    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot all configurations:
    # x = heterogeneity score (+ jitter), y = best_accuracy, marker = num_layers
    for nl in unique_layers:
        subset = df[df['num_layers'] == nl]
        x = subset['heterogeneity_score'].values.astype(float)
        x = x + jitter_scale * rng.normal(size=len(x))  # horizontal jitter
        y = subset['best_accuracy'].values

        marker = layer_markers.get(nl, 'o')
        label = f'{nl} layer' if nl == 1 else f'{nl} layers'

        ax.scatter(x, y, marker=marker, alpha=0.7, label=label)

    # X-axis ticks: use distinct heterogeneity scores
    hetero_values = sorted(df['heterogeneity_score'].unique())
    ax.set_xticks(hetero_values)
    ax.set_xticklabels(
        ['homogeneous' if k == 1 else f'{k} distinct activations' for k in hetero_values],
        rotation=0
    )

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Activation heterogeneity')
    ax.set_ylabel('Best test accuracy')

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)

    ax.legend(uniq_handles, uniq_labels, title='Number of layers', frameon=False, loc='best')

    plt.tight_layout()
    plt.show()