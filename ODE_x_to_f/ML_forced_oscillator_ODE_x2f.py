#%%

import tensorflow as tf
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import scipy as sp



if dde.backend.backend_name == "paddle":
    import paddle

    transpose = paddle.transpose
elif dde.backend.backend_name == "pytorch":
    import torch

    transpose = lambda x, y: torch.transpose(x, y[0], y[1])
else:
    from deepxde.backend import tf

    transpose = tf.transpose


dde.config.disable_xla_jit()

# PDE
td = dde.geometry.TimeDomain(0, 1)


b = 2
c = 3
from scipy.integrate import odeint


def ode(t,y,auxiliary_var_function=None):
    x, xd, F = y[:,0:1], y[:,1:2], y[:,2:3]
    xd = dde.grad.jacobian(y,t,i=0)
    xdd = dde.grad.hessian(y,t,component=0)
    return xdd + b*xd + c*x - F

#def psbc(y,F,t):
    #def forcedosc(y,t):
        #x1, x2 = y
        #dydt = [x2, -b*x2 - c*x1 + F]
        #return(dydt)
    #sol = odeint(forcedosc,[0,0],t)[:,0]
    #return(sol)


t = np.linspace(0,1,num=50)
newt = t[:,None]

def forcedosc(y,t):
    x1, x2 = y
    dydt = [x2, -b*x2 - c*x1 + 3*np.cos(np.pi*t)]
    return(dydt)


xpoints = odeint(forcedosc,[0,0],t)
x1points = xpoints[:,0]
x2points = xpoints[:,1]

Fpoints = 3*np.cos(np.pi*t)




obs1 = dde.icbc.PointSetBC(newt,x1points,component=0)
obs2 = dde.icbc.PointSetBC(newt,x2points,component=1)


ode = dde.data.TimePDE(td, ode, [obs1,obs2], num_domain=2000, num_boundary=100, num_test=100,auxiliary_var_function=None)

# Function space
func_space = dde.data.GRF(length_scale=0.2,kernel='ExpSineSquared')

# Data
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = dde.data.PDEOperatorCartesianProd(
    ode, func_space, eval_pts, function_variables=[0], num_function=1000, num_test=100, batch_size=100
)

# Net
net = dde.nn.DeepONetCartesianProd(
    [50, 128, 128, 128],
    [1, 128, 128, 128],
    "tanh",
    "Glorot normal",
)


#Hard constraint zero IC
#def zero_ic(inputs, outputs):
    #return outputs * transpose(inputs[1], [1, 0])


#net.apply_output_transform(zero_ic)

model = dde.Model(data, net)
model.compile("adam", lr=0.0005)
losshistory, train_state = model.train(epochs=10)

dde.utils.plot_loss_history(losshistory)

#Testing with scipy ode integrater
# %%
v = np.sin(np.pi * eval_pts).T
x = np.linspace(0, 1, num=50)
u = np.ravel(model.predict((v, x[:, None])))

from scipy.integrate import odeint
def forcedosc(y,t):
    x, xi = y
    dydt = [xi, -b*xi - c*x + np.sin(np.pi*t)]
    return(dydt)

y0 = [0,0]
t = np.linspace(0,1,50)
u_true = odeint(forcedosc,y0,t)[:,0]
print(dde.metrics.l2_relative_error(u_true, u))
plt.figure()
plt.plot(x, u_true, "k")
plt.plot(x, u, "r")
plt.xlabel('t')
plt.ylabel('x(t)')
plt.savefig('comparison_ftox_pi.png')
# %%
