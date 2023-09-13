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
    x, xd, F = y[:,0:1], y[:,1:2], y[2:3]
    ddx = dde.grad.jacobian(y,t,i=1)
    return ddx + (b/m)*xd + (k/m)*x - F

def boundary(X, on_initial):
    return on_initial and np.isclose(X[0],0)

ic1 = dde.icbc.IC(geom, lambda X: 0, boundary, component = 0)
ic2 = dde.icbc.IC(geom, lambda X: 0, boundary, component = 1)
#icn = dde.icbc.IC(geom, lambda X: 0, boundary, component = 0)
#return here if issues

#define f0,w (driving freq), w0 (natural frequency)
f0 = 5
w = np.pi/2
w0 = np.sqrt(k/m)

t = np.linspace(0,1,num=100)
tpoints = t[:,None]
#%%
#Employing solve_ivp instead of odeint to generate training data
#from scipy.integrate import solve_ivp

#def forcedharm(t,y):
    #solution = [y[1],-(b/m)*y[1] - (k/m)*y[0] + f0*np.cos(w*t)/(f0*m)]
    #return solution

#sol = solve_ivp(forcedharm,[0,10],y0=[0,0],t_eval=t)

#plt.plot(t,sol.y[1])
#%%
#Employing SciPy to solve the ODE to generate data
def forcedosc(y,t,m,b,w,f0):
    x, xi = y
    dydt = [xi, -(b/m)*xi - (k/m)*x + f0*np.cos(w*t)/(f0*m)]
    return(dydt)

from scipy.integrate import odeint

xpoints = odeint(forcedosc,[0,0],t,args=(m,b,w,f0))[:,0]
xdpoints = odeint(forcedosc,[0,0],t,args=(m,b,w,f0))[:,1]
xmax = max(xpoints)
xdmax = max(xdpoints)
xpoints = xpoints/xmax
xdpoints = xdpoints/xdmax
#plt.plot(tpoints,xdpoints)
def getvel(x):
    return odeint(forcedosc,[0,0],t,args=(m,b,w,f0))[:,1]

#plt.plot(tpoints,f0*np.cos(w*t)/f0)

# %%
obs1 = dde.icbc.PointSetBC(tpoints,xpoints,component=0)
obs2 = dde.icbc.PointSetBC(tpoints,xdpoints,component=1)
def bound(X, on_boundary):
    return on_boundary

bc = dde.icbc.NeumannBC(geom, lambda X: getvel(X), bound, component=0)
#obs2 = dde.icbc.PointSetBC(tpoints,np.cos(w*t)[:,None],component=1)
data = dde.data.TimePDE(
    geom,
    ode_system,
    [obs1,obs2],
    num_domain = 400,
    num_boundary = 100,
)


# %%

net = dde.nn.FNN([1] + [32]*3 + [3], 'sin', 'Glorot uniform')

model = dde.Model(data,net)
model.compile('adam', lr=1e-5)


losshistory, train_state = model.train(iterations=30000)

# %%

pred = model.predict(tpoints)[:,0]
plt.plot(t,pred,'r')
plt.plot(t,xpoints*xmax,'k')
#%%
pred = model.predict(tpoints)[:,2]
plt.plot(tpoints,pred,'r')


    

plt.plot(tpoints,np.cos(w*t),'k')

plt.savefig('Pinncomparison.png')
# %%
