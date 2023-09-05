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

geom = dde.geometry.TimeDomain(0, 1)


M = np.random.randint(10,size=(2,2))
A = np.random.randint(10,size=(2,2))
B = np.random.randint(10,size=(2,2))
C = np.random.randint(10,size=(2,2)) 

def applymass(Mat,i):
    cordnew = 0
    for j in range(2):
        cordnew += Mat[i][j]
    return(cordnew)

def ode(t,y,F):
    x0,x1 = y[:,0:1], y[:,1:2] 
    F0, F1 = F[:,0:1], F[:,1:2]
    dx0dt = dde.grad.Jacobian(x0,t)
    ddx0dt = dde.grad.Hessian(x0,t)
    dx1dt = dde.grad.Jacobian(x1,t)
    ddx1dt = dde.grad.Hessian(x1,t)
    eq0 = applymass(M+A,0)*ddx0dt + applymass(B,0)*dx0dt + applymass(C,0)*x0 - F0
    eq1 = applymass(M+A,1)*ddx1dt + applymass(B,1)*dx1dt + applymass(C,1)*x1 - F1
    return tf.concat(eq0,eq1)



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
# %%
