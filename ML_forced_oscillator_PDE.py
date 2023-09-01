#%%
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



def ode(t,y,F):
    b = 1.2
    c = 0.5
    dy_t = dde.grad.jacobian(y, t)
    dy_tt = dde.grad.hessian(y, t)
    return dy_tt + b * dy_t + c * y - F


timedomain = dde.geometry.TimeDomain(0,2*np.pi)

ic = dde.icbc.IC(timedomain, lambda _: 0, lambda _, on_initial: on_initial) #set x(0) = 0


pde = dde.data.PDE(timedomain, ode, ic, num_domain=20, num_boundary=2, num_test=40)

func_space = dde.data.GRF(T=2*np.pi,length_scale=0.2,kernel='ExpSineSquared')

eval_points = np.linspace(0,np.pi,num=50)[:,None]
data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_points, 1000, function_variables=[0], num_test=50, batch_size=20
)

bnet, tnet = [50, 16, 16], [1, 16, 16]
lr = .0005
net = dde.nn.DeepONetCartesianProd(
    bnet, tnet,
    'tanh',
    "Glorot normal",
)

#def zero_ic(inputs, outputs):
    #return outputs * tf.transpose(inputs[1], [1, 0])
print(bnet,tnet)
print(lr)
print('batch size = 20')

#%%
#net.apply_output_transform(zero_ic)

model = dde.Model(data, net)
model.compile('adam', lr = lr)

losshistory, train_state = model.train(epochs=50000)


#%%
from scipy.integrate import odeint

dde.utils.plot_loss_history(losshistory)

b = np.random.uniform(0.1,1)
c = np.random.uniform(0.1,3)
fm = np.random.uniform(1,4)
w = np.random.uniform(.5,2)*np.pi
y0 = [0,0]

v = fm*np.sin(w * eval_points[:,0]).T
x = np.linspace(0, 1, num=50)
vmat = []
for i in range(50):
    vmat.append(v)
u = np.ravel(model.predict((vmat, x[:, None])))

def forcedosc(y,t,b,c,w,fm):
    x, xi = y
    dydt = [xi, -b*xi - c*x - fm*np.sin(w*eval_points[:,0])]
    return(dydt)

u_true = odeint(forcedosc,y0,t=eval_points[:,0],args=(b,c,w,fm))
print(dde.metrics.l2_relative_error(u_true, u))
plt.figure()
plt.plot(x, u_true, "k")
plt.plot(x, u, "r")

plt.show()
# %%
