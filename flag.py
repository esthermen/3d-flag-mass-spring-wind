import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================================
# Cloth Simulation (Mass-Spring System)
# Wind Interaction + Gravity
# RK4 Integration
# ==========================================================

# -------------------------
# Physical parameters
# -------------------------
Nx, Ny = 20, 20        # grid resolution
mass = 0.2
k_struct = 150.0
L0 = 0.3
g = 9.81
dt = 0.01
damping = 0.3
wind_strength = 5.0

# -------------------------
# State arrays
# -------------------------
r = np.zeros((Nx, Ny, 3))
v = np.zeros((Nx, Ny, 3))

# Initial flat cloth
for i in range(Nx):
    for j in range(Ny):
        r[i,j] = np.array([i*L0, 5.0, j*L0])

# Fix left edge (flagpole constraint)
fixed = [(0,j) for j in range(Ny)]

# -------------------------
# Spring force
# -------------------------
def spring_force(p1, p2):
    d = p2 - p1
    L = np.linalg.norm(d)
    if L == 0:
        return np.zeros(3)
    return k_struct * (L - L0) * (d / L)

# -------------------------
# Acceleration
# -------------------------
def compute_acceleration(r_local, v_local):
    a = np.zeros_like(r_local)

    for i in range(Nx):
        for j in range(Ny):

            if (i,j) in fixed:
                continue

            F = np.zeros(3)

            # Structural neighbors
            for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < Nx and 0 <= nj < Ny:
                    F += spring_force(r_local[i,j], r_local[ni,nj])

            # Gravity
            F[1] -= mass*g

            # Wind force (simple directional model)
            wind = np.array([wind_strength, 0, 0])
            F += wind

            # Damping
            F -= damping * v_local[i,j]

            a[i,j] = F / mass

    return a

# -------------------------
# RK4 integrator
# -------------------------
def rk4_step(r,v,dt):

    def deriv(r_local,v_local):
        return v_local, compute_acceleration(r_local,v_local)

    k1_r,k1_v = deriv(r,v)
    k2_r,k2_v = deriv(r+0.5*dt*k1_r, v+0.5*dt*k1_v)
    k3_r,k3_v = deriv(r+0.5*dt*k2_r, v+0.5*dt*k2_v)
    k4_r,k4_v = deriv(r+dt*k3_r, v+dt*k3_v)

    r_next = r + dt/6*(k1_r+2*k2_r+2*k3_r+k4_r)
    v_next = v + dt/6*(k1_v+2*k2_v+2*k3_v+k4_v)

    # Apply fixed constraints
    for i,j in fixed:
        r_next[i,j] = r[i,j]
        v_next[i,j] = 0

    return r_next, v_next

# -------------------------
# Visualization
# -------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Cloth Simulation - Wind Interaction")

ax.set_xlim(0, Nx*L0)
ax.set_ylim(0, 6)
ax.set_zlim(0, Ny*L0)

lines = []

for i in range(Nx):
    for j in range(Ny):
        if i < Nx-1:
            line, = ax.plot([],[],[],'b-',lw=0.5)
            lines.append((line,(i,j),(i+1,j)))
        if j < Ny-1:
            line, = ax.plot([],[],[],'b-',lw=0.5)
            lines.append((line,(i,j),(i,j+1)))

def update(frame):
    global r,v
    r,v = rk4_step(r,v,dt)

    for line,a,b in lines:
        x0,y0,z0 = r[a]
        x1,y1,z1 = r[b]
        line.set_data([x0,x1],[y0,y1])
        line.set_3d_properties([z0,z1])

    return [line for line,_,_ in lines]

ani = animation.FuncAnimation(fig, update, frames=600, interval=20)
plt.show()

