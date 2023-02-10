import pandas as pd
import numpy as np
from numpy import arange
from matplotlib import pyplot as plt


"""
Load data and prepare data rows
"""
# Load data
data = pd.read_excel(
    'results.xlsx')
data = data.drop(['Algorithm'], axis=1)

# Prepare data rows
ols = data.loc[0, :].values.flatten().tolist()
shc = data.loc[1, :].values.flatten().tolist()
sa = data.loc[2, :].values.flatten().tolist()
ils = data.loc[3, :].values.flatten().tolist()
es_coma = data.loc[4, :].values.flatten().tolist()
es_plus = data.loc[5, :].values.flatten().tolist()

row_rows = [ols, shc, sa, ils, es_coma, es_plus]

"""
Make a list for the optimization algorithms
"""
algorithms = ['OLS', 'SHC', 'SA',
              'ILS', 'ES_COMA', 'ES_PLUS']

"""
Make a list for the metrics
"""
metrics = ['mse', 'max_ae', 'mae', 'r2', 'mpd']

"""
Define y-axis range and scale
"""
values = arange(0, 80, 5)

"""
Draw a table chart
"""
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(algorithms)))
index = arange(len(metrics)) + 0.3
bar_width = 0.7

y_offset = np.zeros(len(metrics))
fig, ax = plt.subplots()

cell_text = []
n_rows = len(row_rows)

for row in range(n_rows):
    plot = plt.bar(index,
                   row_rows[row],
                   bar_width,
                   bottom=y_offset,
                   color=colors[row])

    y_offset = row_rows[row]

    cell_text.append([x for x in y_offset])

    i = 0

    # Each iteration of this for loop labels each bar
    # with corresponding value for the given algorithm
    for rect in plot:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2,
                y_offset[i],
                '%d' % int(y_offset[i]),
                ha='center',
                va='bottom')
        i += 1

"""
Add a table to the bottom of the chart, 
and plot the table chart 
"""
the_table = plt.table(cellText=cell_text,
                      rowLabels=algorithms,
                      rowColours=colors,
                      colLabels=metrics,
                      loc='bottom')

plt.ylabel('error_scores')
plt.xticks([])
plt.show()
