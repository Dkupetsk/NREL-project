#%%
import torch
import deepxde as dde
import numpy as np

d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
d1 = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
X_train,y_train = (d["X"][0].astype(np.float32),d["X"][1].astype(np.float32)), d["y"].astype(np.float32)
X_test,y_test = (d1["X"][0].astype(np.float32),d1["X"][1].astype(np.float32)), d1["y"].astype(np.float32)


#X_train includes both the branch net and trunk net inputs; i.e. a matrix where each row is a function v(x) and the values at which to evaluate u(x)
#y_train is a matrix where each row is a function u(x) corresponding to a function v(x)
#%%
data = dde.data.TripleCartesianProd(X_train=X_train,y_train=y_train, X_test=X_test, y_test=y_test)
m=100
dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [m,40,40],
    [dim_x,40,40],
    'relu',
    'Glorot normal',
)

model = dde.Model(data, net)
model.compile("adam", lr=.001,metrics=["mean l2 relative error"])
losshistory, train_state = model.train(iterations=10000)
#%%
x = np.linspace(-5,5,m)
def f(x):
    return(x**2)

def fint(x):
    return(.33*x**3)

#print(fint(x))
fx = np.array([f(x).astype(np.float32)])
fx = np.repeat(x,150,axis=0)
x = x.reshape((100,1)).astype(np.float32)
testux = dde.data.TripleCartesianProd(X_train=fx,y_train=x,X_test=0,y_test=0)

# %%
