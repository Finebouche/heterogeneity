import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_literal(x):
    """Parse a Python literal stored as a string in CSV; pass through if already parsed."""
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x

def flatten_layers(comp):
    """Flatten tuple-of-tuples of activation names across layers into a single list."""
    flat = []
    for layer in comp:
        flat.extend(layer)
    return flat

if __name__ == '__main__':
    # Adjust filename if needed
    df = pd.read_csv('results_unordered_10.csv')

    # Parse columns written as stringified Python objects
    df['activation_composition'] = df['activation_composition'].apply(parse_literal)
    df['architecture'] = df['architecture'].apply(parse_literal)

    # Heterogeneity = number of distinct activations across ALL hidden neurons (all layers)
    df['heterogeneity_score'] = df['activation_composition'].apply(
        lambda comp: len(set(flatten_layers(comp)))
    )

    # Number of layers
    df['num_layers'] = df['architecture'].apply(len)

    # Quick sanity checks
    print("Unique heterogeneity scores found:", sorted(df['heterogeneity_score'].unique()))
    print("Unique layer counts found:", sorted(df['num_layers'].unique()))

    # Summary for homogeneous networks (score == 1)
    if (df['heterogeneity_score'] == 1).any():
        print("\nBest accuracies for homogeneous networks (heterogeneity_score = 1):")
        print(df.loc[df['heterogeneity_score'] == 1, "best_test_accuracy"].describe())
    else:
        print("\nNo homogeneous configurations found in this run.")

    # ---------------- Plot: accuracy vs. heterogeneity ----------------
    # ---------------- Plot: accuracy vs. heterogeneity ----------------
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    # --- make everything compact ---
    plt.rcParams.update({
        "figure.figsize": (4, 3),
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelpad": 2,
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
    })

    rng = np.random.default_rng(0)
    jitter = 0.12  # narrower jitter

    hetero_values = sorted(df['heterogeneity_score'].unique())
    labels = ["Homog." if k == 1 else f"{k} acts." for k in hetero_values]
    xpos = {k: i + 1 for i, k in enumerate(hetero_values)}  # 1..K

    fig, ax = plt.subplots(constrained_layout=True)

    # --- boxplots per heterogeneity bucket ---
    data_by_bucket = [df.loc[df['heterogeneity_score'] == k, 'best_test_accuracy'].to_numpy()
                      for k in hetero_values]

    bp = ax.boxplot(
        data_by_bucket,
        positions=[xpos[k] for k in hetero_values],
        widths=0.30,  # thinner boxes
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanline=True,
    )

    # clean styling
    for box in bp['boxes']:
        box.set(facecolor='none', edgecolor='black', linewidth=1.0)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.2)
    for mean in bp['means']:
        mean.set(color='black', linewidth=1.0, linestyle='--')
    for line in bp['whiskers'] + bp['caps']:
        line.set(color='black', linewidth=0.9)

    # --- jittered points (smaller, lighter) ---
    layer_markers = {1: 'o', 2: 's', 3: '^', 4: 'D'}
    unique_layers = sorted(df['num_layers'].unique())

    for nl in unique_layers:
        sub = df[df['num_layers'] == nl]
        xs = sub['heterogeneity_score'].map(xpos).to_numpy(dtype=float)
        xs = xs + rng.uniform(-jitter, jitter, size=len(xs))
        ys = sub['best_test_accuracy'].to_numpy()
        m = layer_markers.get(nl, 'o')
        ax.scatter(xs, ys, marker=m, s=8, alpha=0.35, linewidths=0, rasterized=True)

    # --- axes, ticks, labels ---
    ax.set_xlim(0.5, len(hetero_values) + 0.5)
    ax.set_xticks([xpos[k] for k in hetero_values])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Best test accuracy')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))


    # light grid
    ax.grid(axis='y', which='major', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # compact n annotations
    for k, x in xpos.items():
        n = len(df[df['heterogeneity_score'] == k])
        ax.text(x, ax.get_ylim()[0] + 0.0005, f"n={n}", ha='center', va='bottom', fontsize=7)

    # legend rarely needed; omit unless multiple layer counts exist
    if len(unique_layers) > 1:
        handles = [plt.Line2D([0], [0], marker=layer_markers.get(nl, 'o'),
                              linestyle='None', markersize=4,
                              label=f"{nl} layer" if nl == 1 else f"{nl} layers")
                   for nl in unique_layers]
        ax.legend(handles=handles, title='Layers', frameon=False, loc='lower left', fontsize=7, title_fontsize=7)

    # super-tight save
    plt.savefig('mnist_activation_scatter.pdf', dpi=600, pad_inches=0.01)
    print("Saved tight figures.")