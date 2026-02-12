import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from perlin_noise import PerlinNoise
import matplotlib.image as mpimg

# -------------------------
# Parámetros físicos
# -------------------------
N = 20
masas = 10
d = 3
m = np.ones((masas, N))
k = 150*np.ones(masas)
L0 = 0.5*np.ones(masas)
g = 9.81
dt = 0.05
amort_v = 0.5*np.ones(masas)
amort_h = 0.5*np.ones(N)
dist = 10.0
k_h = 100*np.ones(N)
L0_h = dist/(N-1)*np.ones(N)
z_fixed = 4.0*np.ones(N)

# -------------------------
# Posiciones y velocidades
# -------------------------
r = np.zeros((masas, N, d))
v = np.zeros((masas, N, d))
x_fixed = np.linspace(0, dist, N)

r[0, :, 2] = z_fixed
for i in range(1, masas):
    for n in range(N):
        mm = m[i,n]*(masas-i)
        L = L0[i-1] + mm*g/k[i]
        r[i, n, 2] = r[i-1, n, 2] - L

for n in range(N):
    r[:, n, 0] = x_fixed[n]
r[:, :, 1] = 0.0

# -------------------------
# Perlin noise
# -------------------------
noise_gen = PerlinNoise(octaves=3, seed=42)
t_global = 0.0

# -------------------------
# Cargar textura
# -------------------------
img = mpimg.imread("tela.jpg")  # ruta de la imagen
H, W, _ = img.shape

def get_color_from_texture(x, z, r_min, r_max):
    """Interpolar color de la textura para coordenadas x,z de la tela"""
    u = (x - r_min[0]) / (r_max[0] - r_min[0])
    v = (z - r_min[2]) / (r_max[2] - r_min[2])
    i_img = np.clip((u*(W-1)).astype(int), 0, W-1)
    j_img = np.clip(((1-v)*(H-1)).astype(int), 0, H-1)
    return img[j_img, i_img, :3]

# -------------------------
# Figura y artistas
# -------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Tela 3D con textura en muelles y puntos y viento")

spring_lines_v = np.empty((masas, N), dtype=object)
spring_lines_hx = np.empty((masas, N-1), dtype=object)
mass_dots = np.empty((masas, N), dtype=object)

for n in range(N):
    for i in range(masas):
        spring_lines_v[i, n], = ax.plot([], [], [], lw=2)
        mass_dots[i, n], = ax.plot([], [], [], 'o', markersize=4)
for n in range(N-1):
    for i in range(masas):
        spring_lines_hx[i, n], = ax.plot([], [], [], lw=2)

ax.set_xlim(np.min(r[:,:,0])-0.5, np.max(r[:,:,0])+0.5)
ax.set_ylim(np.min(r[:,:,1])-0.5, np.max(r[:,:,1])+0.5)
ax.set_zlim(np.min(r[:,:,2])-0.5, np.max(r[:,:,2])+0.5)

# -------------------------
# Fuerzas internas
# -------------------------
def f_vertical(i, n, r_col):
    F = np.zeros(3)
    if i > 0:
        dr = r_col[i]-r_col[i-1]
        ell = np.linalg.norm(dr)
        if ell != 0:
            F += -k[i]*(ell-L0[i-1])*(dr/ell)
    if i < masas-1:
        dr = r_col[i+1]-r_col[i]
        ell = np.linalg.norm(dr)
        if ell != 0:
            F += k[i]*(ell-L0[i])*(dr/ell)
    return F

def f_horizontal(i, n, r_row):
    F = np.zeros(3)
    if n != 0:
        dr = r_row[n]-r_row[n-1]
        ell = np.linalg.norm(dr)
        if ell != 0:
            F += -k_h[n]*(ell-L0_h[n-1])*(dr/ell)
    if n < N-1:
        dr = r_row[n+1]-r_row[n]
        ell = np.linalg.norm(dr)
        if ell != 0:
            F += k_h[n+1]*(ell-L0_h[n])*(dr/ell)
    return F

# -------------------------
# Wind force
# -------------------------
def wind_force(i,n,r_arr,v_arr,t=0.0):
    W_mean = np.array([5.0,0.0,0.2])
    rho, C_d, area, turb_scale = 1.2, 1.2, 0.05, 0.8
    pos = r_arr[i,n]*0.15
    turbulence_vec = np.array([noise_gen([pos[0],pos[1],t*0.2]),
                               noise_gen([pos[1],pos[2],t*0.2]),
                               noise_gen([pos[2],pos[0],t*0.2])])*turb_scale
    W_local = W_mean + turbulence_vec
    v_rel = W_local - v_arr[i,n]
    # normal local aproximada
    dRi = np.zeros(3)
    dRn = np.zeros(3)
    if i>0 and i<masas-1: dRi = r_arr[i+1,n]-r_arr[i-1,n]
    elif i>0: dRi = r_arr[i,n]-r_arr[i-1,n]
    elif i<masas-1: dRi = r_arr[i+1,n]-r_arr[i,n]
    if n>0 and n<N-1: dRn = r_arr[i,n+1]-r_arr[i,n-1]
    elif n>0: dRn = r_arr[i,n]-r_arr[i,n-1]
    elif n<N-1: dRn = r_arr[i,n+1]-r_arr[i,n]
    normal = np.cross(dRi,dRn)
    area_local = 0.5*np.linalg.norm(normal)
    if np.linalg.norm(normal)>1e-9: normal/=np.linalg.norm(normal)
    else: normal=np.array([0.0,0.0,1.0])
    vrel_dot_n = np.dot(v_rel,normal)
    if vrel_dot_n<=0: return np.zeros(3)
    F_pressure = 0.5*rho*C_d*max(area_local,1e-4)*(np.linalg.norm(v_rel)**2)*normal
    v_tang = v_rel-np.dot(v_rel,normal)*normal
    F_shear = 0.3*rho*area_local*v_tang
    return F_pressure+F_shear

# -------------------------
# Aceleración
# -------------------------
def acceleration(i,n,r_new,v_new,t):
    r_col = r_new[:,n,:].copy()
    r_row = r_new[i,:,:].copy()
    v_col = v_new[:,n,:].copy()
    v_row = v_new[i,:,:].copy()
    Fv = f_vertical(i,n,r_col)
    Fh = f_horizontal(i,n,r_row)
    Fwind = wind_force(i,n,r_new,v_new,t)
    damp = -amort_v[i]*v_col[i] - amort_h[n]*v_row[n]
    F_total = Fv+Fh+Fwind+damp
    a = F_total/m[i,n]
    a[2]-=g
    return a

# -------------------------
# RK4 global
# -------------------------
def deriv_global(r_flat,v_flat,t):
    r_new = r_flat.reshape((masas,N,3))
    v_new = v_flat.reshape((masas,N,3))
    ar = np.zeros_like(r_new)
    av = np.zeros_like(v_new)
    for i in range(masas):
        for n in range(1,N):
            ar[i,n,:]=v_new[i,n,:]
            av[i,n,:]=acceleration(i,n,r_new,v_new,t)
    return ar.flatten(), av.flatten()

def rk4_step(r_flat,v_flat,dt,t):
    k1_r,k1_v=deriv_global(r_flat,v_flat,t)
    k2_r,k2_v=deriv_global(r_flat+0.5*dt*k1_r,v_flat+0.5*dt*k1_v,t+0.5*dt)
    k3_r,k3_v=deriv_global(r_flat+0.5*dt*k2_r,v_flat+0.5*dt*k2_v,t+0.5*dt)
    k4_r,k4_v=deriv_global(r_flat+dt*k3_r,v_flat+dt*k3_v,t+dt)
    r_next = r_flat + dt/6*(k1_r+2*k2_r+2*k3_r+k4_r)
    v_next = v_flat + dt/6*(k1_v+2*k2_v+2*k3_v+k4_v)
    return r_next,v_next

def pack_state(r,v): return r.flatten(), v.flatten()
def unpack_state(r_flat,v_flat): return r_flat.reshape((masas,N,3)), v_flat.reshape((masas,N,3))

r_flat,v_flat=pack_state(r,v)

# -------------------------
# Update (animación) con textura
# -------------------------
def update(frame):
    global r_flat,v_flat,r,v,t_global
    r_flat,v_flat = rk4_step(r_flat,v_flat,dt,t_global)
    t_global += dt
    r,v = unpack_state(r_flat,v_flat)

    artists=[]

    r_min = np.min(r,axis=(0,1))
    r_max = np.max(r,axis=(0,1))

    for n in range(N):
        for i in range(1, masas):
            x0, y0, z0 = r[i-1, n]
            x1, y1, z1 = r[i, n]

            # color de textura interpolado entre extremos
            c0 = get_color_from_texture(x0, z0, r_min, r_max)
            c1 = get_color_from_texture(x1, z1, r_min, r_max)

            spring_lines_v[i, n].set_data([x0, x1], [y0, y1])
            spring_lines_v[i, n].set_3d_properties([z0, z1])

            # Normalizar colores 0-1 para matplotlib
            color_avg = (c0 + c1) / 2 / 255.0
            spring_lines_v[i, n].set_color(color_avg)

            mass_dots[i, n].set_data([x1], [y1])
            mass_dots[i, n].set_3d_properties([z1])
            mass_dots[i, n].set_color(c1 / 255.0)  # normalizar

            artists += [spring_lines_v[i, n], mass_dots[i, n]]

    for n in range(N-1):
        for i in range(masas):
            x0, y0, z0 = r[i, n]
            x1, y1, z1 = r[i, n+1]

            c0 = get_color_from_texture(x0, z0, r_min, r_max)
            c1 = get_color_from_texture(x1, z1, r_min, r_max)

            spring_lines_hx[i, n].set_data([x0, x1], [y0, y1])
            spring_lines_hx[i, n].set_3d_properties([z0, z1])
            spring_lines_hx[i, n].set_color((c0 + c1) / 2 / 255.0)  # normalizar

            artists.append(spring_lines_hx[i, n])



    ax.set_xlim(np.min(r[:,:,0])-0.5,np.max(r[:,:,0])+0.5)
    ax.set_ylim(np.min(r[:,:,1])-0.5,np.max(r[:,:,1])+0.5)
    ax.set_zlim(np.min(r[:,:,2])-0.5,np.max(r[:,:,2])+0.5)

    return artists

ani = animation.FuncAnimation(fig,update,frames=800,interval=20,blit=False)
plt.show()
