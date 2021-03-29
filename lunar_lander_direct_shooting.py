import numpy as np
from casadi import *
import matplotlib.pyplot as plt


T = 20.  # Time horizon
N = 150  # number of control intervals
grav = 1.62  # moon gravity

c1 = 44_000

Isp = 311
g0 = 9.81
c2 = Isp * g0
m_0 = 10_000

# ---------Initial Conditions----------
rx0 = -25
rz0 = 100
vx0 = -2.5
vz0 = -1.
m0 = 10_000

# ---------Final Conditions------------
rxf = 0
rzf = 0
vxf = 0
vzf = 0

# -------------Bounds--------------
u1min = 0.
u1max = 1.
u2min = -.5
u2max = .5

rxmin = -50.
rxmax = 50.
rzmin = 0.
rzmax = 250.
vxmin = -50.
vxmax = 50.
vzmin = -50.
vzmax = 50.

# Declare model variables
NUM_STATES = 5
NUM_INPUTS = 2
rx = SX.sym('rx')
rz = SX.sym('rz')
vx = SX.sym('vx')
vz = SX.sym('vz')
m = SX.sym('m')
x = vertcat(rx, rz, vx, vz, m)

u1 = SX.sym('u1')
u2 = SX.sym('u2')
u = vertcat(u1, u2)


def dynamics():
    rxdot = vx
    rzdot = vz
    vxdot = c1 * (u1 / m) * casadi.sin(u2)
    vzdot = c1 * (u1 / m) * casadi.cos(u2) - grav
    mdot = -((c1 / c2) * u1)

    return vertcat(rxdot, rzdot, vxdot, vzdot, mdot)


xdot = dynamics()

# Objective term
L = u1**2 + u2**2  # h**2 + v**2 + u**2

# Formulate discrete time dynamics

# Fixed step Runge-Kutta 4 integrator
M = 4  # RK4 steps per interval
DT = T/N/M
f = Function('f', [x, u], [xdot, L])  # Maps [x, u] -> [xdot, L]
X0 = SX.sym('X0', NUM_STATES)
U = SX.sym('U', NUM_INPUTS)
X = X0
Q = 0
for j in range(M):
    k1, k1_q = f(X, U)
    k2, k2_q = f(X + DT/2 * k1, U)
    k3, k3_q = f(X + DT/2 * k2, U)
    k4, k4_q = f(X + DT * k3, U)
    X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # Integrated state ODE's
    Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
F = Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])  # Maps initial states to final states (also Q)

# Start with an empty NLP
w = []
w0 = []  # Initial Guess for controls
lbw = []  # Lower bound for controls
ubw = []  # Upper bound for controls
J = 0  # Cost function
g = []  # Inequality constraint
lbg = []  # Inequality constraint lower bound
ubg = []  # Inequality constraint upper bound

# Formulate the NLP
Xk = SX([rx0, rz0, vx0, vz0, m0])
for k in range(N):
    # New NLP variable for the control
    Uk = SX.sym('U_' + str(k), NUM_INPUTS)

    w += [Uk[0]]
    lbw += [u1min]  # Control u1 lower bound
    ubw += [u1max]  # Control u1 upper bound
    w0 += [1.]  # Control u1 guess

    w += [Uk[1]]
    lbw += [u2min]  # Control u2 lower bound
    ubw += [u2max]  # Control u2 upper bound
    w0 += [.5]  # Control u2 guess

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk)
    Xk = Fk['xf']
    J = J+Fk['qf']

    if k == N-1:
        lbd_rx = rxf
        upd_rx = rxf

        lbd_rz = rzf
        upd_rz = rzf

        lbd_vx = vxf
        upd_vx = vxf

        lbd_vz = vzf
        upd_vz = vzf

    else:
        lbd_rx = rxmin
        upd_rx = rxmax

        lbd_rz = rzmin
        upd_rz = rzmax

        lbd_vx = vxmin
        upd_vx = vxmax

        lbd_vz = vzmin
        upd_vz = vzmax

    # Add inequality constraint
    g += [Xk[0]]
    lbg += [lbd_rx]  # Lower bound on rx
    ubg += [upd_rx]  # Upper bound on rx

    g += [Xk[1]]
    lbg += [lbd_rz]  # Lower bound on rz
    ubg += [upd_rz]  # Upper bound on rz

    g += [Xk[2]]
    lbg += [lbd_vx]  # Lower bound on vx
    ubg += [upd_vx]  # Upper bound on vx

    g += [Xk[3]]
    lbg += [lbd_vz]  # Lower bound on rx
    ubg += [upd_vz]  # Upper bound on rx

    g += [Xk[4]]
    lbg += [0.]  # Lower bound on m (cant go lower than 0 fuel)
    ubg += [m0]  # Upper bound on m

# 'w' and 'g' are lists of length N (number of control inputs)
# 'w': list of controls
# 'g': list of constraints
# Create an NLP solver

prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
# w0 = Initial Guess for controls (u)
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
u_opt = sol['x']
u_opt_arr = u_opt.full()

# Optimal controls are stored as [u(k)1_opt, u(k)2_opt, u(k+1)1_opt, u(k+1)2_opt, etc...]
u1_arr = u_opt_arr[::NUM_INPUTS].flatten()
u2_arr = u_opt_arr[1::NUM_INPUTS].flatten()


def forward_dynamics(curr_states, throttle, gimbal):
    xp, zp, xv, zv, mass = curr_states

    xp += xv * dt
    zp += zv * dt
    xv += (c1 * (throttle / mass) * np.sin(gimbal)) * dt
    zv += (c1 * (throttle / mass) * np.cos(gimbal) - grav) * dt
    mass += -((c1 / c2) * throttle) * dt

    return np.array([xp, zp, xv, zv, mass])


dt = T / N

rx_hist = []
rz_hist = []
mass_hist = []

states = np.array([rx0, rz0, vx0, vz0, m0])

for i in range(N):
    rx_hist.append(states[0])
    rz_hist.append(states[1])
    mass_hist.append(states[4])
    states = forward_dynamics(states, u1_arr[i], u2_arr[i])


plt.plot([n*dt for n in range(N)], rx_hist, label='X')
plt.plot([n*dt for n in range(N)], rz_hist, label='Z')
plt.plot([n*dt for n in range(N)], mass_hist, label='Mass')
plt.legend()
plt.grid()
plt.show()

plt.plot(rx_hist, rz_hist, label='X-Z')
plt.legend()
plt.grid()
plt.show()

print('FINAL MASS: ', states[4])


