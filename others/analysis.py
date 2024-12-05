import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('results_50.csv')
    # activation_funcs column looks like this: ('relu', 'sigmoid', 'relu', 'relu', 'sigmoid') so we need to convert it to list
    df['activation_funcs'] = df['activation_funcs'].apply(lambda x: x[1:-1].split(', '))
    # compute heterogeneity_score = len(set(activation_funcs)) from activation_funcs column
    df['heterogeneity_score'] = df['activation_funcs'].apply(lambda x: len(set(x)))

    # plot accuracy bin plot were stack colors are heterogeneity score with viridis colormap
    sns.histplot(data=df, x='accuracy', hue='heterogeneity_score', multiple='stack', bins=20, palette='viridis')
    plt.show()