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

Created on Sun Dec  8 15:40:19 2024

@author: Andreas Langer
"""
import tensorflow as tf


class CustomLoss:
    
    @staticmethod
    def custom_loss_FDPINN(pred, lambda1, lambda2, data, model):
        params = data['params']
        Nx = params['Nx']
        Ny = params['Ny']
        Nt = params['Nt']
        hx = params['hx']
        hy = params['hy']
        dt = params['dt']
    
        u_label = data['u_label']
        v_label = data['v_label']
        
        # reshape labels to (Ny,Nx,Nt) and crop to (Ny-2,Nx-2,Nt-1) to match u_int/v_int
        u_lab = tf.reshape(u_label, (Ny, Nx, Nt))#[:, :, :-1]
        v_lab = tf.reshape(v_label, (Ny, Nx, Nt))#[:, :, :-1]
        u_lab = u_lab[1:-1, 1:-1, :]
        v_lab = v_lab[1:-1, 1:-1, :]
        
        
        # u (and v) is only [u_0,....u_{N-1}] in x-direction as u_0 = u_N (truncated)
        psi_pred = pred[0]
        p_pred = pred[1]
        
        
        psi_3d = tf.reshape(psi_pred, (Ny, Nx, Nt))
        p_3d   = tf.reshape(p_pred,  (Ny, Nx, Nt))
        
        u_int = central_y(psi_3d, hy)               # u = ∂ψ/∂y
        v_int = -central_x(psi_3d, hx)              # v = -∂ψ/∂x
        u_int = u_int[:, 1:-1, :]                # remove x-boundaries
        v_int = v_int[1:-1, :, :]                # remove y-boundaries
        # Now u_int, v_int: (Ny-2, Nx-2, Nt)
        
        # time-forward derivatives on that interior grid
        u_t = forward_t(u_int, dt)                 # (Ny-2, Nx-2, Nt-1)
        v_t = forward_t(v_int, dt)                 # (Ny-2, Nx-2, Nt-1)
    
        # First-order:
        u_x = central_x(u_int, hx)#[:, 1:-1, :]     # central x -> (Ny-2, Nx-4, Nt); crop to core x
        u_y = central_y(u_int, hy)#[1:-1, :, :]     # central y -> (Ny-4, Nx-2, Nt); crop to core y
        v_x = central_x(v_int, hx)#[:, 1:-1, :]     # (Ny-2, Nx-4, Nt)
        v_y = central_y(v_int, hy)#[1:-1, :, :]     # (Ny-4, Nx-2, Nt)
        
        # Laplacians of u and v on a common core (Ny-4, Nx-4, Nt)
        u_xx = second_x(u_int, hx)#[:, 1:-1, :]     # (Ny-2, Nx-4, Nt)
        u_yy = second_y(u_int, hy)#[1:-1, :, :]     # (Ny-4, Nx-2, Nt)
        v_xx = second_x(v_int, hx)#[:, 1:-1, :]     # (Ny-2, Nx-4, Nt)
        v_yy = second_y(v_int, hy)#[1:-1, :, :]     # (Ny-4, Nx-2, Nt)

        # bring everything to the same spatial "core" (Ny-4, Nx-4, Nt)
        # core means removing 2 layers in y and x from the original psi grid
        u_core   = interior_crop_xy(u_int)             # (Ny-4, Nx-4, Nt)
        v_core   = interior_crop_xy(v_int)             # (Ny-4, Nx-4, Nt)
        u_t_core = interior_crop_xy(u_t)               # (Ny-4, Nx-4, Nt-1)
        v_t_core = interior_crop_xy(v_t)               # (Ny-4, Nx-4, Nt-1)
    
        u_x_core = u_x[1:-1, :, :]                 # (Ny-4, Nx-4, Nt)
        u_y_core = u_y[:, 1:-1, :]                 # (Ny-4, Nx-4, Nt)
        v_x_core = v_x[1:-1, :, :]                 # (Ny-4, Nx-4, Nt)
        v_y_core = v_y[:, 1:-1, :]                 # (Ny-4, Nx-4, Nt)
    
        
    
        u_lap_core = (u_xx[1:-1, :, :] + u_yy[:, 1:-1, :])    # (Ny-4, Nx-4, Nt)
        v_lap_core = (v_xx[1:-1, :, :] + v_yy[:, 1:-1, :])    # (Ny-4, Nx-4, Nt)
    
    
        # pressure gradients on the same core
        p_int      = interior_crop_xy(p_3d)                        # (Ny-2, Nx-2, Nt)
        p_x_int    = central_x(p_int, hx)#[:, 1:-1, :]           # (Ny-2, Nx-4, Nt)
        p_y_int    = central_y(p_int, hy)#[1:-1, :, :]           # (Ny-4, Nx-2, Nt)
        p_x_core   = p_x_int[1:-1, :, :]                        # (Ny-4, Nx-4, Nt)
        p_y_core   = p_y_int[:, 1:-1, :]                        # (Ny-4, Nx-4, Nt)
    
        
        # match time axis for PDE residuals (use 1 due to forward_t)
        u_core_t0   = u_core[:, :, 1:]   # (Ny-4, Nx-4, Nt-1)
        v_core_t0   = v_core[:, :, 1:]
        u_x_core_t0 = u_x_core[:, :, 1:]
        u_y_core_t0 = u_y_core[:, :, 1:]
        v_x_core_t0 = v_x_core[:, :, 1:]
        v_y_core_t0 = v_y_core[:, :, 1:]
        u_lap_t0    = u_lap_core[:, :, 1:]
        v_lap_t0    = v_lap_core[:, :, 1:]
        p_x_t0      = p_x_core[:, :, 1:]
        p_y_t0      = p_y_core[:, :, 1:]
        
        
        # Navier–Stokes residuals on core × (Nt-1) with backward time
        f_u = u_t_core + lambda1*(u_core_t0*u_x_core_t0 + v_core_t0*u_y_core_t0) + p_x_t0 - lambda2*u_lap_t0
        f_v = v_t_core + lambda1*(u_core_t0*v_x_core_t0 + v_core_t0*v_y_core_t0) + p_y_t0 - lambda2*v_lap_t0
        
        t = tf.linspace(0.0, 1.0, tf.shape(f_u)[-1])
        w_t = tf.exp(-3.0*t)            # β=3 startwert
        f_u = f_u * w_t[None,None,:]
        f_v = f_v * w_t[None,None,:]
        
        loss = (
            tf.reduce_mean(tf.square(u_lab - u_int)) +
            tf.reduce_mean(tf.square(v_lab - v_int)) +
            tf.reduce_mean(tf.square(f_u)) +
            tf.reduce_mean(tf.square(f_v)) 
        )
        div_core = u_x_core + v_y_core
        loss += 1e-3 * tf.reduce_mean(tf.square(div_core))
        return loss
            
def central_y(f, dy):
    # (Ny, Nx, Nt) -> central ∂/∂y at interior y: 1..Ny-2
    return (f[2:, :, :] - f[:-2, :, :]) / (2.0*dy)         # (Ny-2, Nx, Nt)

def central_x(f, dx):
    # (Ny, Nx, Nt) -> central ∂/∂x at interior x: 1..Nx-2
    return (f[:, 2:, :] - f[:, :-2, :]) / (2.0*dx)         # (Ny, Nx-2, Nt)

def second_y(f, dy):
    # (Ny, Nx, Nt) -> ∂²/∂y² at interior y
    return (f[2:, :, :] - 2.0*f[1:-1, :, :] + f[:-2, :, :]) / (dy*dy)  # (Ny-2, Nx, Nt)

def second_x(f, dx):
    # (Ny, Nx, Nt) -> ∂²/∂x² at interior x
    return (f[:, 2:, :] - 2.0*f[:, 1:-1, :] + f[:, :-2, :]) / (dx*dx)  # (Ny, Nx-2, Nt)

def forward_t(f, dt):
    # (.., Nt) -> forward ∂/∂t at t: 0..Nt-2
    return (f[..., 1:] - f[..., :-1]) / dt                 # (..., Nt-1)

def interior_crop_xy(f):
    # take interior y=1..Ny-2 and x=1..Nx-2 from (Ny, Nx, Nt)
    return f[1:-1, 1:-1, :]