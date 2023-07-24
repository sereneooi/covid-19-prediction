import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# read data from csv files
data = pd.read_csv("Cases.csv").iloc[41:, :]

# calculate the correlation matrix
corr = data.corr()

# plot the heatmap
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap = "Blues", annot = True, linewidths = .5)
plt.show()

c = data.corr().abs()
s = c.unstack()
sort = s.sort_values(kind = "quicksort")
print(sort)