#%%
import tensorflow as tf
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import scipy as sp


import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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
geom = dde.geometry.TimeDomain(0, 1)


b = 2
c = 3

def ode(t,x,F):
    dxdt = dde.grad.jacobian(x,t)
    ddxdt = dde.grad.hessian(x,t)
    return ddxdt + b*dxdt + c*x - F



ic = dde.icbc.IC(geom, lambda _: 0, lambda _, on_initial: on_initial,component=0)
pde = dde.data.PDE(geom, ode, ic, num_domain=20, num_boundary=2, num_test=40)

# Function space
func_space = dde.data.GRF(length_scale=0.2)

# Data
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_pts, 1000, num_test=100, batch_size=100
)

# Net
net = dde.nn.DeepONetCartesianProd(
    [50, 128, 128, 128],
    [1, 128, 128, 128],
    "tanh",
    "Glorot normal",
)


# Hard constraint zero IC
def zero_ic(inputs, outputs):
    return outputs * transpose(inputs[1], [1, 0])


net.apply_output_transform(zero_ic)

model = dde.Model(data, net)
model.compile("adam", lr=0.0005)
losshistory, train_state = model.train(epochs=10000)

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
v = np.sin(np.pi/2 * eval_pts).T
x = np.linspace(0, 1, num=50)
u = np.ravel(model.predict((v, x[:, None])))
from scipy.integrate import odeint
def forcedosc(y,t):
    x, xi = y
    dydt = [xi, -b*xi - c*x + np.sin(np.pi/2*t)]
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
plt.savefig('comparison_ftox_pi/2.png')
# %%
v = 3*np.sin(np.pi * eval_pts).T
x = np.linspace(0, 1, num=50)
u = np.ravel(model.predict((v, x[:, None])))
from scipy.integrate import odeint
def forcedosc(y,t):
    x, xi = y
    dydt = [xi, -b*xi - c*x + 3*np.sin(np.pi*t)]
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
plt.savefig('comparison_ftox_pi_3.png')
# %%
