import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

df = pd.read_csv('./images/Final/dataset-with-workflow-10-labeled.csv')

for i in range(7):
    df_cur = df[df['Labels'] == i]
    df_cur = df_cur[['commits_total', 'contributors_total', 'issues_total',
                     'pulls_total', 'stars_total', 'repository_size', 'cc_mean', 'mi_mean']]

    print(df_cur.shape)
    print('Statistics of the cluster')
    print(f'commits {statistics.mean(df_cur["commits_total"].tolist())}')
    print(f'contributors {statistics.mean(df_cur["contributors_total"].tolist())}')
    print(f'issues {statistics.mean(df_cur["issues_total"].tolist())}')
    print(f'pulls {statistics.mean(df_cur["pulls_total"].tolist())}')
    print(f'stars {statistics.mean(df_cur["stars_total"].tolist())}')
    print(f'repo size {statistics.mean(df_cur["repository_size"].tolist())}')
    print(f'cyclomatic complexity {statistics.mean(df_cur["cc_mean"].tolist())}')
    print(f'MI {statistics.mean(df_cur["mi_mean"].tolist())}')

    fig, axs = plt.subplots(2, 4)

    counts, bins = np.histogram(df_cur["commits_total"].tolist(), 10)
    axs[0, 0].xticks = bins
    axs[0, 0].stairs(counts, bins, fill=True)
    axs[0, 0].set_title('Commits')
    axs[0, 0].set(xlabel='Commits', ylabel='Repositories')

    counts, bins = np.histogram(df_cur["contributors_total"].tolist(), 10)
    axs[0, 1].xticks = bins
    axs[0, 1].stairs(counts, bins, fill=True)
    axs[0, 1].set_title('Contributors')
    axs[0, 1].set(xlabel='Contributors', ylabel='Repositories')

    counts, bins = np.histogram(df_cur["issues_total"].tolist(), 10)
    axs[0, 2].xticks = bins
    axs[0, 2].stairs(counts, bins, fill=True)
    axs[0, 2].set_title('Issues')
    axs[0, 2].set(xlabel='Issues', ylabel='Repositories')

    counts, bins = np.histogram(df_cur["pulls_total"].tolist(), 10)
    axs[0, 3].xticks = bins
    axs[0, 3].stairs(counts, bins, fill=True)
    axs[0, 3].set_title('Pulls')
    axs[0, 3].set(xlabel='Pulls', ylabel='Repositories')

    counts, bins = np.histogram(df_cur["stars_total"].tolist(), 10)
    axs[1, 0].xticks = bins
    axs[1, 0].stairs(counts, bins, fill=True)
    axs[1, 0].set_title('Stars')
    axs[1, 0].set(xlabel='Stars', ylabel='Repositories')

    counts, bins = np.histogram([item // 1024 for item in df_cur["repository_size"].tolist()], 10)
    axs[1, 1].xticks = bins
    axs[1, 1].stairs(counts, bins, fill=True)
    axs[1, 1].set_title('Repository size, mb')
    axs[1, 1].set(xlabel='Size', ylabel='Repositories')

    counts, bins = np.histogram(df_cur["cc_mean"].tolist(), 10)
    axs[1, 2].xticks = bins
    axs[1, 2].stairs(counts, bins, fill=True)
    axs[1, 2].set_title('Cyclomatic complexity')
    axs[1, 2].set(xlabel='CC', ylabel='Repositories')

    counts, bins = np.histogram(df_cur["mi_mean"].tolist(), 10)
    axs[1, 3].xticks = bins
    axs[1, 3].stairs(counts, bins, fill=True)
    axs[1, 3].set_title('Maintainability Index')
    axs[1, 3].set(xlabel='MI', ylabel='Repositories')

    fig.tight_layout(pad=0.1)
    plt.show()
