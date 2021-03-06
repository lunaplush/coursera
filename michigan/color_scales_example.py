#Colore Scales

#import plotly.plotly as py
#import plotly.graph_objs as go
#import plotly.tools as tls
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
num=1000
num = 1000
s = 121
x1 = np.linspace(-0.5,1,num) + (0.5 - np.random.rand(num))
y1 = np.linspace(-5,5,num) + (0.5 - np.random.rand(num))
x2 = np.linspace(-0.5,1,num) + (0.5 - np.random.rand(num))
y2 = np.linspace(5,-5,num) + (0.5 - np.random.rand(num))
x3 = np.linspace(-0.5,1,num) + (0.5 - np.random.rand(num))
y3 = (0.5 - np.random.rand(num))

ax1 = fig.add_subplot(221)
cb1 = ax1.scatter(x1, y1, c=x1, cmap=plt.cm.get_cmap('Blues'))
plt.colorbar(cb1, ax=ax1)
ax1.set_title('Blues')

ax2 = fig.add_subplot(222)
cb2 = ax2.scatter(x2, y2, c=x2, cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar(cb2, ax=ax2)
ax2.set_title('RdBu')
ax3 = fig.add_subplot(223)
cb3 = ax3.scatter(x3, y3, c=x3, cmap=plt.cm.get_cmap('Dark2'))
plt.colorbar(cb3, ax=ax3)
ax3.set_xlabel('Dark2')
plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)

fig = plt.gcf()
#py.plot_mpl(fig, filename="mpl-colormaps-simple")