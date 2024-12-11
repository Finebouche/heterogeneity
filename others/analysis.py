import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('results_10.csv')
    # activation_funcs column looks like this: ('relu', 'sigmoid', 'relu', 'relu', 'sigmoid') so we need to convert it to list
    df['activation_funcs'] = df['activation_functions'].apply(lambda x: x[1:-1].split(', '))
    # compute heterogeneity_score = len(set(activation_funcs)) from activation_funcs column
    df['heterogeneity_score'] = df['activation_funcs'].apply(lambda x: len(set(x)))

    #print the row with heterogeneity_score = 1
    print(df[df['heterogeneity_score'] == 1]["mean_accuracy"])

    # plot accuracy bin plot were stack colors are heterogeneity score with viridis colormap
    sns.histplot(data=df, x='mean_accuracy', hue='heterogeneity_score', multiple='stack', bins=20, palette='viridis')
    plt.show()