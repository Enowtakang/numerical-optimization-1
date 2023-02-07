from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt


"""
Load and Prepare dataset
"""
# Load dataset
data = pd.read_csv('data.csv')

# Define y and x1, through to x9
y = data['ALLSKY_SFC_PAR_TOT']
x1 = data['T2M']
x2 = data['PRECTOTCORR']
x3 = data['WS2M']
x4 = data['PS']
x5 = data['RH2M']
x6 = data['ALLSKY_SFC_UVA']
x7 = data['ALLSKY_SFC_UVB']
x8 = data['ALLSKY_SFC_UV_INDEX']
x9 = data['CLRSKY_SFC_PAR_TOT']

# Make plot

fig, axs = plt.subplots(3, 3)

axs[0, 0].plot(x1, y)
axs[0, 0].set_title('T2M')

axs[0, 1].plot(x2, y, 'green')
axs[0, 1].set_title('PRECTOTCORR')

axs[0, 2].plot(x3, y, 'red')
axs[0, 2].set_title('WS2M')

axs[1, 0].plot(x4, y, 'cyan')
axs[1, 0].set_title('PS')

axs[1, 1].plot(x5, y, 'magenta')
axs[1, 1].set_title('RH2M')

axs[1, 2].plot(x6, y, 'yellow')
axs[1, 2].set_title('ALLSKY_SFC_UVA')

axs[2, 0].plot(x7, y, 'black')
axs[2, 0].set_title('ALLSKY_SFC_UVB')

axs[2, 1].plot(x8, y)
axs[2, 1].set_title('ALLSKY_SFC_UV_INDEX')

axs[2, 2].plot(x9, y, 'green')
axs[2, 2].set_title('CLRSKY_SFC_PAR_TOT')


for ax in axs.flat:
    ax.set(ylabel='ALLSKY_SFC_PAR_TOT')

# Hide x labels and tick labels for top plots and
# y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()


plt.show()


# Some refs
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
