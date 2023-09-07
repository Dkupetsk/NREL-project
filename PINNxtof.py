#%%
import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

geom = dde.geometry.TimeDomain(0,1)

m = 2
b = 3
k = 4

def ode_system(t,y):
    x, F = y[:,0:1], y[:,1:2]
    dx = dde.grad.jacobian(y,t,i=0)
    ddx = dde.grad.hessian(y,t,component=0)
    return m*ddx + b*dx + k*x - F

def boundary(_, on_initial):
    return on_initial

ic = dde.icbc.IC(geom, lambda X: 0, boundary, component = 0)
#return here if issues

#define f0,w (driving freq), w0 (natural frequency)
f0 = 5
w = np.pi/2
w0 = np.sqrt(k/m)

tpoints = np.linspace(0,1,num=50)[:,None]
xamp = (f0/m)/((w0**2 - w**2)**2 + (w**2)*(b/m)**2)
xphase = (b/m)*w/(w0**2 - w**2)
xpoints = xamp*np.cos(w*tpoints - xphase)
#plt.plot(tpoints,xpoints)
# %%
obs = dde.icbc.PointSetBC(tpoints,xpoints,component=0)

data = dde.data.PDE(
    geom,
    ode_system,
    [ic,obs],
    num_domain = 400,
    num_boundary = 2,
    anchors=tpoints
)


# %%

net = dde.nn.FNN([1] + [40]*3 + [2], 'tanh', 'Glorot uniform')

model = dde.Model(data,net)
model.compile('adam', lr=.0001)
losshistory, train_state = model.train(epochs=60000)
# %%

pred = np.ravel(model.predict(tpoints))
plt.plot(tpoints,pred[50:],'r')

from scipy.integrate import odeint
plt.plot(tpoints,f0*np.cos(w*tpoints),'k')
# %%
