# NREL-project

Explanation folders/files:

Direct_x_to_f: - DeepONet mapping of x(t) to F(t) in the equation $m\ddot{x} + b\dot{x} + kx = F(t)$ using a training dataset consisting of inputs $x(t)$ which are derived from SciPy's odeint based on the desired output function $F(t)$. The initial conditions are constant but $m, b, k, f0, w$ are not. The points $t$ in the output and input are the same, so the data is aligned.

ODE_f_to_x: - DeepONet mapping of $F(t)$ to $x(t)$ using the ODE itself. The function inputs $F(t)$ are taken from a Gaussian Random Field (GRF).

ODE_F_to_x_6d: - Still not working, but the idea is to expand ODE_f_to_x to 6 dimensions corresponding to the 6 DOF of FOWT.

ODE_x_to_f: - DeepONet mapping of $x(t)$ to $F(t)$ using the ODE itself. So far, this is not working at all. Seems like the issue is at least partially that DeepXDE is having trouble figuring out how to take a derivative of $x(t)$ when it is not an output of the DeepONet.

PINNxtof.py: - Not using a DeepONet, but mapping $x(t)$ to $F(t)$ using a PINN for a singular ODE case. I am using this as a testing ground for debugging the ODE_x_to_f case. The PINN works fine given that it is fed $x(t)$ data, $F(t)$ corresponding data, and boundary and initial conditions (including $x(0) = 0$ and $\dot{x}(0) = 0)