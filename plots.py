import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./dataset-with-workflow-9.csv')

df_stars = df['stars_total'].tolist()
df_stars = sorted(df_stars)
print(df_stars[0])

counts, bins = np.histogram(df_stars, 25, range=(501, 15000))

plt.xticks(bins)
plt.xlabel('Stars')
plt.ylabel('Number of repositories')
plt.stairs(counts, bins, fill=True)
#Repo-stars distribution. Last hundred omitted

plt.show()