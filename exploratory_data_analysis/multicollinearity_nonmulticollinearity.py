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


"""
Make plot
"""

# Define the axes
fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# add columns to the axes
x_scale_1, y_scale_1, z_scale_1 = x9, x8, x6
x_scale_2, y_scale_2, z_scale_2 = x4, x2, x7

ax1.scatter(
    x_scale_1, y_scale_1, z_scale_1,
    s=50, alpha=0.6, edgecolors='w')

ax2.scatter(
    x_scale_2, y_scale_2, z_scale_2,
    s=50, alpha=0.6, edgecolors='g')

# Add labels to all of the axes
ax1.set_xlabel('CLRSKY_SFC_PAR_TOT')
ax1.set_ylabel('ALLSKY_SFC_UV_INDEX')
ax1.set_zlabel('ALLSKY_SFC_UVA')

ax2.set_xlabel('PS')
ax2.set_ylabel('PRECTOTCORR')
ax2.set_zlabel('ALLSKY_SFC_UVB')

ax1.set_label('A')
ax2.set_label('B')

plt.show()
