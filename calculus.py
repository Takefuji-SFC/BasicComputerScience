#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def df(x):
    return 2*x

def lim(x):
    h = 0.0001
    return (f(x+h) - f(x))/h

x = np.linspace(0,10,100)
y = f(x)
dy = df(x)
limx = lim(x)

#differential coefficient
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(x,y,c='blue',label='f(x)')
ax.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('f(x)')

ax2 = fig.add_subplot(2,1,2)
ax2.plot(x,dy,c='red',label='df(x)/dx')
ax2.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('df(x)/dx')
fig.show()

#tangential line
fig2 = plt.figure()
fig2ax1 = fig2.add_subplot(1,1,1)
def line(x,a,b):
    y = a*x + b
    return y

n = 100
plotx = np.linspace(0,n,5)
for i in np.arange(len(plotx)):
    a = lim(plotx[i])
    b = f(plotx[i]) - a*plotx[i]
    fig2ax1.plot(plotx,line(plotx,a,b),c='red')

fig2ax1.plot(np.linspace(0,n,100),f(np.linspace(0,n,100)),c='blue',label='f(x)')
fig2ax1.scatter(plotx,f(plotx),c='black',marker='+',s=100,label='point of contact')
plt.xlabel('x')
plt.ylabel('f(x)')
fig2ax1.legend(loc='upper left')
fig.show()

