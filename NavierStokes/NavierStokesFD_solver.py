"""
============================================================================
Copyright (C) 2025  Andreas Langer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
============================================================================    

GNU GENERAL PUBLIC LICENSE Version 3

Created on Fri Sep  5 12:12:22 2025

@author: Andreas Langer
"""
# Generates data for the data-driven Navier-Stokes example:
#    cylinder_like_periodic_box.mat
#
# Collocated (cell-centered) Backward-Euler finite-difference Navier–Stokes (2D, incompressible)
# - Uniform cell-centered grid for u, v, p
# - Periodic boundary conditions in x and y
# - Backward Euler for diffusion & pressure (projection method)
# - Convection is *linearized* with old velocity: (u^n · ∇) u^{n+1} (no nonlinear iteration)
# - Centered finite differences via Kronecker products
# - Plots the final pressure field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# -------------------- 1D periodic FD matrices --------------------
def D1_per(N, h):
    """1D centered first derivative (periodic)."""
    e = np.ones(N)
    data = np.array([-0.5*e, 0.5*e])
    offsets = np.array([-1, 1])
    D = sp.diags(data, offsets, shape=(N, N), format='lil')
    D[0, -1] = -0.5
    D[-1, 0] = 0.5
    return (1.0/h) * D.tocsr()

def D2_per(N, h):
    """1D second derivative (periodic)."""
    e = np.ones(N)
    data = np.array([e, -2.0*e, e])
    offsets = np.array([-1, 0, 1])
    D2 = sp.diags(data, offsets, shape=(N, N), format='lil')
    D2[0, -1] = 1.0
    D2[-1, 0] = 1.0
    return (1.0/h**2) * D2.tocsr()


# -------------------- Solver --------------------
def solve_ns_collocated_BE_picard(
    Nx=64, Ny=64, Lx=2*np.pi, Ly=2*np.pi,
    dt=2e-3, Nt=200, lam1=0, lam2=1, seed=0,
    picard_max=100, picard_tol=1e-10, verbose=False,
    use_multimode_IC=True, use_forcing=True, F0=0.5, ky_force=1
):
    """
    Collocated (cell-centered) 2D incompressible Navier–Stokes, periodic in x,y.
    Time: Backward Euler. Nonlinearity: Picard iterations per step (fully implicit).
    Space: centered finite differences via Kronecker products.
    Projection: incremental pressure-correction each Picard iterate.

    Returns u, v, p with shape (Ny, Nx, Nt+1), and dx, dy, dt.
    """
    np.random.seed(seed)

    # grid & operators
    dx, dy = Lx/Nx, Ly/Ny
    Ix = sp.eye(Nx, format='csr'); Iy = sp.eye(Ny, format='csr')
    Dx = sp.kron(Iy, D1_per(Nx, dx), format='csr')           # d/dx on centers
    Dy = sp.kron(D1_per(Ny, dy), Ix, format='csr')           # d/dy on centers
    Lap = sp.kron(Iy, D2_per(Nx, dx), format='csr') + sp.kron(D2_per(Ny, dy), Ix, format='csr')
    Ntot = Nx*Ny
    I = sp.eye(Ntot, format='csr')

    nu=lam2

    # Poisson with gauge pin
    Lap_p = Lap.tolil()
    Lap_p[0,:] = 0.0; Lap_p[0,0] = 1.0
    Lap_p = Lap_p.tocsr()
    solve_P = spla.factorized(Lap_p)

    # storage
    u = np.zeros((Ny, Nx, Nt+1))
    v = np.zeros((Ny, Nx, Nt+1))
    p = np.zeros((Ny, Nx, Nt+1))

    # divergence-free periodic IC (Taylor–Green-like)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    x = np.linspace(0, Lx, Nx, endpoint=False)
    X, Y = np.meshgrid(x, y)
    # ---- Initial condition ----
    if use_multimode_IC:
        psi = (1.00*np.sin(1*X)*np.cos(1*Y)
        + 0.30*np.sin(2*X)*np.cos(1*Y)
        + 0.20*np.sin(1*X)*np.cos(2*Y)
        + 0.15*np.sin(2*X)*np.cos(2*Y))
        psi_y = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1))/(2*dy)
        psi_x = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0))/(2*dx)
        u[...,0] = psi_y; v[...,0] = -psi_x; p[...,0] = 0.0
    else:
        u[...,0] = np.sin(X)*np.cos(Y)
        v[...,0] = -np.cos(X)*np.sin(Y)
        p[...,0] = 0.0
        
    # forcing
    if use_forcing:
        fx = F0 * np.sin(ky_force*(2*np.pi/Ly)*Y)
        fy = np.zeros_like(fx)
    else:
        fx = np.zeros_like(X)
        fy = np.zeros_like(X)
        
    flat   = lambda F: F.reshape(-1)
    unflat = lambda f: f.reshape(Ny, Nx)

    for n in range(Nt):
        u_n = flat(u[...,n]); v_n = flat(v[...,n]); p_n = flat(p[...,n])

        # start Picard at old state
        u_k, v_k, p_k = u_n.copy(), v_n.copy(), p_n.copy()

        for it in range(picard_max):
            # convection with current iterate (fully implicit via fixed-point)
            Udiag = sp.diags(u_k, 0, shape=(Ntot,Ntot), format='csr')
            Vdiag = sp.diags(v_k, 0, shape=(Ntot,Ntot), format='csr')
            A = lam1*(Udiag @ Dx + Vdiag @ Dy)  # u^k ∂x + v^k ∂y

            # Helmholtz (Backward Euler in time)
            H = (I - dt*nu*Lap + dt*A)

            # incremental pressure form with p_k
            rhs_u = u_n - dt*(Dx @ p_k) + dt*flat(fx)
            rhs_v = v_n - dt*(Dy @ p_k) + dt*flat(fy)

            u_tilde = spla.spsolve(H, rhs_u)
            v_tilde = spla.spsolve(H, rhs_v)

            # projection
            div_tilde = Dx @ u_tilde + Dy @ v_tilde
            rhs_phi = (1.0/dt) * div_tilde
            rhs_phi[0] = 0.0
            phi = solve_P(rhs_phi)

            u_new = u_tilde - dt*(Dx @ phi)
            v_new = v_tilde - dt*(Dy @ phi)
            p_new = p_k + phi

            # convergence check
            du = np.linalg.norm(u_new - u_k)/np.sqrt(Ntot)
            dv = np.linalg.norm(v_new - v_k)/np.sqrt(Ntot)
            dp = np.linalg.norm(p_new - p_k)/np.sqrt(Ntot)
            if verbose:
                print(f"step {n+1}, picard {it+1}: du={du:.2e}, dv={dv:.2e}, dp={dp:.2e}")
            u_k, v_k, p_k = u_new, v_new, p_new
            if max(du, dv, dp) < picard_tol:
                break

        u[...,n+1] = unflat(u_k)
        v[...,n+1] = unflat(v_k)
        p[...,n+1] = unflat(p_k)
        
        # end-of-step divergence (should match last printed div)
        if (n+1) % max(1, Nt//10) == 0:
            div_end = np.linalg.norm((Dx @ u_k + Dy @ v_k))/np.sqrt(Ntot)
            print(f"tstep {n+1:4d}/{Nt}: div={div_end:.3e}")
            PDE_residual(u_new, u_n, v_new, v_n, dt, Dx, Dy, p_new, Lap, lam1, nu, Ntot)
    return u, v, p, dx, dy, dt


def PDE_residual(u_new, u_n, v_new, v_n, dt, Dx, Dy, p_new, Lap, lam1, nu, Ntot):
    # ---- PDE residuals (fully implicit BE) at the current iterate ----
    # time derivative uses previous time level (u_n, v_n)
    Ut = (u_new - u_n) / dt
    Vt = (v_new - v_n) / dt
    
    # spatial derivatives at t^{n+1} (current iterate)
    ux = Dx @ u_new
    uy = Dy @ u_new
    vx = Dx @ v_new
    vy = Dy @ v_new
    px = Dx @ p_new
    py = Dy @ p_new
    Luu = Lap @ u_new
    Lvv = Lap @ v_new
    
    # fully nonlinear BE residual (this matches your custom loss form)
    Ru_full = Ut + lam1*(u_new * ux + v_new * uy) + px - nu * Luu
    Rv_full = Vt + lam1*(u_new * vx + v_new * vy) + py - nu * Lvv
    
    # norms to print
    ru2 = np.linalg.norm(Ru_full) / np.sqrt(Ntot)
    rv2 = np.linalg.norm(Rv_full) / np.sqrt(Ntot)
    
    
    print(f" ||Ru_full||={ru2:.3e}  ||Rv_full||={rv2:.3e}  ")


# -------------------- run & plot --------------------
Nx, Ny, Nt = 32, 32, 40
Re = 1e1#100.0
lam1=1;lam2=1/Re
dt=1e-1
Lx=2*np.pi; Ly=2*np.pi

u, v, p, dx, dy, dt = solve_ns_collocated_BE_picard(Nx=Nx, Ny=Ny, Lx=Lx, 
                                                        Ly=Ly, dt=dt, Nt=Nt, 
                                                        lam1=lam1, lam2=lam2,
                                                        seed=0,picard_max=100, 
                                                        picard_tol=1e-8,
                                                        use_multimode_IC=True,
                                                        use_forcing=False)


plt.figure(figsize=(6,5))
plt.imshow(p[..., Nt], origin='lower', extent=[0, Nx*dx, 0, Ny*dy], aspect='equal')
plt.colorbar(label='pressure')
plt.title('Pressure at final time')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

# Flatten to (M, T), T=Nt (use times 1..Nt)
M = Nx*Ny
T = Nt
u_flat = u[..., 1:].reshape(M, T)
v_flat = v[..., 1:].reshape(M, T)
p_flat = p[..., 1:].reshape(M, T)

# Coordinates
y = np.linspace(0, Ly, Ny, endpoint=False)
x = np.linspace(0, Lx, Nx, endpoint=False)
Xc, Yc = np.meshgrid(x, y)
X_flat = Xc.reshape(M)
Y_flat = Yc.reshape(M)


U_star = np.zeros((M, 2, T), dtype=np.float64)
U_star[:, 0, :] = u_flat
U_star[:, 1, :] = v_flat
p_star = p_flat
X_star = np.column_stack([X_flat, Y_flat])
t_vec = (np.arange(1, T+1, dtype=np.float64) * dt).reshape(T, 1)

import scipy.io as sio
out_path = "cylinder_like_periodic_box.mat"
sio.savemat(out_path, {
    "U_star": U_star,
    "p_star": p_star,
    "t": t_vec,
    "X_star": X_star,
    "Nx": Nx,
    "Ny": Ny,
    "Nt": Nt,
    "dx": dx,
    "dy": dy,
    "dt": dt,
    "lam1":lam1,"lam2":lam2
})

out_path
