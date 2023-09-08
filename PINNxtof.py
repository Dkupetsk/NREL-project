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

def neumannbounds(_, on_initial):
    return on_initial

ic = dde.icbc.IC(geom, lambda X: 0, boundary, component = 0)
icn = dde.icbc.NeumannBC(geom, lambda X: 0, neumannbounds, component=0)

#return here if issues

#define f0,w (driving freq), w0 (natural frequency)
f0 = 5
w = np.pi/2
w0 = np.sqrt(k/m)

t = np.linspace(0,1,num=50)
tpoints = t[:,None]

def forcedosc(y,t,m,b,w,f0):
    x, xi = y
    dydt = [xi, -(b/m)*xi - (k/m)*x + f0*np.cos(w*t)/(f0*m)]
    return(dydt)

from scipy.integrate import odeint

xpoints = odeint(forcedosc,[0,0],t,args=(m,b,w,f0))[:,0]
xpoints = xpoints/max(xpoints)
plt.plot(tpoints,xpoints)
plt.plot(tpoints,f0*np.cos(w*t)/f0)

# %%
obs = dde.icbc.PointSetBC(tpoints,xpoints,component=0)
obs2 = dde.icbc.PointSetBC(tpoints,np.cos(w*t)[:,None],component=1)
data = dde.data.PDE(
    geom,
    ode_system,
    [ic,icn,obs,obs2],
    num_domain = 200,
    num_boundary = 2,
)


# %%

net = dde.nn.FNN([1] + [32]*3 + [2], 'tanh', 'Glorot uniform')

model = dde.Model(data,net)
model.compile('adam', lr=.001)
#model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5,'pgtol':1e-8, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}} 

losshistory, train_state = model.train(epochs=30000)

# %%

pred = model.predict(tpoints)[:,1]
#%%
plt.plot(tpoints,pred,'r')


Ftrue = f0*np.cos(w*t)/(10)
    

plt.plot(tpoints,f0*np.cos(w*t)/(f0),'k')
# %%
