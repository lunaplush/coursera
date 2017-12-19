# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:12:14 2017

@author: Inspiron
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
#%%
#%matplotlib 

n = 1000
fig, [[ax1, ax2],[ax3,ax4]] = plt.subplots(2,2)
fig.set_facecolor("w")

x1 = np.random.randn(n)


 
x2 = np.random.standard_gamma(1,size = n)
x3 = np.random.lognormal(mean = 0.0, sigma = 1.0, size = n )
x4 = np.random.normal(loc = 0.0, scale = 10, size = n)

k = 30
def update1(curr):
    if curr > k:        
        a.event_source.stop()
      
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    i = (100 + 30*curr)
    bins1 = np.arange(-4, 4, 0.5)
    ax1.hist(x1[: i], bins = bins1 )  
    ax1.set_title("normal distribution", loc = "center")
    #ax2.ylim(0,300)
    ax2.annotate('n  = {}'.format(i),xycoords='figure fraction', xy = (0.4,0.95))
    bins2 = np.arange(0,8, 0.5)
    ax2.hist(x2[: i], bins = bins2 )
    ax2.set_title("gamma distribution", loc = "center")
    bins3 = np.arange(0,50,5)
    ax3.hist(x3[: i], bins = bins3 )
    ax3.set_title("lognormal distribution")
    bins4 = np.arange(-40,40,5)
    ax4.hist(x4[: i], bins=bins4 )
    ax4.set_title("normal distribution with 10.0 deviation")
    
a = animation.FuncAnimation(fig,update1,interval = 100, frames = np.arange(31), repeat = False)    

 #%%
 #https://ffmpeg.zeranoe.com/
plt.rcParams['animation.ffmpeg_path'] = "c:\\Luna\\Work\\python\\ffmpeg-3.4-win64-static\\bin\\"
#"d:\\Luna\\python\\ffmpeg\\bin\\ffmpeg.exe"
#plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
FFwriter = animation.FFMpegWriter()
#os.chdir("d:\\Luna\\python\\coursera\\michigan\\")
os.chdir("c:\\Luna\\Work\\python\\coursera\\michigan\\")
a.save("assign2.avi",writer = FFwriter)
#%%

n = 100
x = np.random.randn(n)

# create the function that will do the plotting, where curr is the current frame
def update(curr):
    # check if animation is at the last frame, and if so, stop the animation a
    if curr == n: 
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-4, 4, 0.5)
    plt.hist(x[:curr], bins=bins)
    plt.axis([-4,4,0,30])
    plt.gca().set_title('Sampling the Normal Distribution')
    plt.gca().set_ylabel('Frequency')
    plt.gca().set_xlabel('Value')
    plt.annotate('n = {}'.format(curr), [3,27])
    
fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval=100)    

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

freqs = np.arange(2, 20, 3)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
l, = plt.plot(t, s, lw=2)


class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

callback = Index()
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()