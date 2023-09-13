#%%
#Creating data
import numpy as np
import scipy as sp
import deepxde as dde
import tensorflow as tf
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def forcedosc(y,t,b,c,w,fm):
    x, xi = y
    dydt = [xi, -b*xi - c*x - fm*np.cos(w*t)]
    return(dydt)

y0 = [0,0]
t = np.linspace(0,10,100)
b,c,fm,w = .5,2,2,np.pi
soltest = odeint(forcedosc,y0,t,args=(b/2,np.sqrt(c),fm,w))

plt.plot(t,soltest[:,0],'.')

#%%
y01 = np.random.uniform(0.0,5.0)
y02 = np.random.uniform(0.0,5.0)
y0 = [y01,y02]

funcarray = []
farray = []
    
for i in range(200):
    b = np.random.uniform(0.1,1)
    c = np.random.uniform(0.1,3)
    fm = np.random.uniform(1,4)
    w = np.random.uniform(.5,2)*np.pi
    sol = odeint(forcedosc,y0,t,args=(b,c,w,fm))
    funcarray.append(sol[:,0].tolist())
    f = fm*np.sin(w*t)
    farray.append(f.tolist()) #Create many F functions
m1 = list(np.where(funcarray == np.max(funcarray)))[0][0]
m2 = list(np.where(funcarray == np.max(funcarray)))[1][0]
print(m1,m2)
funcarray = np.array(funcarray)/funcarray[m1][m2]
farray = np.array(farray)/4
newt = []
#%%
for i in range(200):
    plt.plot(t,funcarray[i])
plt.xlabel('t')
plt.ylabel('x')

plt.show()
for i in range(200):
    plt.plot(t,farray[i])

plt.xlabel('t')
plt.ylabel('F')

plt.show()

for time in range(len(t)):
    newt.append([time])

newt = np.array(newt)
#%%


X_train,y_train = (funcarray.astype(np.float32),newt.astype(np.float32)), farray.astype(np.float32)#%%
#Input = (x(t), t*)
#Output = F(t)

newt = np.array(newt)
xtest = []
ytest = []
for j in range(150):
    b = np.random.uniform(0.1,1)
    c = np.random.uniform(0.1,3)
    fm = np.random.uniform(1,4)
    w = np.random.uniform(.5,2)*np.pi
    sol = odeint(forcedosc,y0,t,args=(b,c,w,fm))
    f = fm*np.sin(w*t) + np.random.uniform(-1,1)
    ytest.append(f.tolist())
    xtest.append(sol[:,0].tolist())
m1 = list(np.where(xtest == np.max(xtest)))[0][0]
m2 = list(np.where(xtest == np.max(xtest)))[1][0]
xtest = np.array(xtest)/xtest[m1][m2]
print(xtest)

X_test = (xtest.astype(np.float32),newt.astype(np.float32))

ytest = np.zeros((150,100))
for i in range(150):
    for j in range(100):
        ytest[i][j] = 1
ytest = np.array(ytest)/5
y_test = ytest.astype(np.float32)

#Testing dataset -> (some random numbers, t*), 
#Output -> (some random numbers)
#Hypothesis: The testing dataset is made to be like the training dataset
#This trains the network to emulate G: x(t) -> F(t)

#%%
for i in range(150):
    plt.plot(t,xtest[i])
plt.show()

for i in range(150):
    plt.plot(t,y_test[i])

plt.show()
# %%
print(y_train)
data = dde.data.TripleCartesianProd(X_train=X_train,y_train=y_train, X_test=X_test, y_test=y_test)
m=100
dim_t = 1


print(data)
#%%

net = dde.nn.deeponet.DeepONetCartesianProd(
    [m,750,750,750],
    [dim_t,750,750,750],
    'sin',
    'Glorot normal',
)





model = dde.Model(data, net)

model.compile("adam", lr=1e-4)
losshistory, train_state = model.train(iterations=100000)



import matplotlib.pyplot as plt

dde.utils.plot_loss_history(losshistory)



 #%%
b = np.random.uniform(0.1,1)
c = np.random.uniform(0.1,3)
fm = np.random.uniform(1,4)
w = np.random.uniform(.5,2)*np.pi

sol = odeint(forcedosc,y0,t,args=(b,c,w,fm))
ftrue = fm*np.sin(w*t)
sol = sol[:,0].tolist()
sol = np.array(sol)/max(sol)

plt.plot(t,sol)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()
fpredmax = np.ravel(model.predict(([sol*max(sol).astype(np.float32)],newt.astype(np.float32))))
fmpred = max(np.abs(fpredmax))*2
fpred = np.ravel(model.predict(([sol.astype(np.float32)],newt.astype(np.float32))))*fmpred

plt.plot(t,ftrue,'k')
plt.plot(t,fpred,'r')
plt.xlabel('t')
plt.ylabel('F')
plt.savefig('comparison_tanh.png')




# %%
