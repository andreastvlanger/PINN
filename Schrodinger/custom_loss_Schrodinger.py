"""
============================================================================
Functions related to custom_loss_PINN are based on code from:
Maziar Raissi, "PINNs" (https://github.com/maziarraissi/PINNs), MIT License.

New code and modifications:
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
    
    def custom_loss_PINN(model,pred, data):
        interior_points = data['interior_points']
        boundary_points_lx = data['boundary_points_lX']
        boundary_points_rx = data['boundary_points_rX']
        initial_points = data['boundary_points_T']
        u0 = data['u0']
        
        x0 = tf.reshape(initial_points[:,0],(-1,1))
        t0 = tf.reshape(initial_points[:,1],(-1,1))
        u0_pred, v0_pred, _, _ = net_uv(model, x0,t0)
        x_lx = tf.reshape(boundary_points_lx[:,0],(-1,1))
        t_lx = tf.reshape(boundary_points_lx[:,1],(-1,1))
        x_rx = tf.reshape(boundary_points_rx[:,0],(-1,1))
        t_rx = tf.reshape(boundary_points_rx[:,1],(-1,1))
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = net_uv(model, x_lx, t_lx)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = net_uv(model, x_rx, t_rx)
        f_u_pred, f_v_pred = net_f_uv(model, interior_points)
        
        loss = tf.reduce_mean(tf.square(u0 - u0_pred)) + \
               tf.reduce_mean(tf.square(v0_pred)) + \
               tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
               tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
               tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + \
               tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred)) + \
               tf.reduce_mean(tf.square(f_u_pred)) + \
               tf.reduce_mean(tf.square(f_v_pred))
        return loss

    def custom_loss_FDPINN(model,pred, data):
        params = data['params']
        Nx = params['Nx']
        Nt = params['Nt']
        initial_points = data['boundary_points_T']
        u0 = data['u0']
        x0 = tf.reshape(initial_points[:,0],(-1,1))
        t0 = tf.reshape(initial_points[:,1],(-1,1))
        
        u0_pred, v0_pred, _, _ = net_uv(model, x0,t0)
        u = pred[0]
        v = pred[1]
        
        u_2d = tf.transpose(tf.reshape(u, [Nt,Nx]))
        v_2d = tf.transpose(tf.reshape(v, [Nt,Nx]))
                
        # Time derivative is along rows
        # Forward differences in time
        dt = params['dt']
        dt = tf.cast(dt, dtype=tf.float32)
        u_t = tf.subtract(u_2d[:,1:], u_2d[:,:-1])/dt
        v_t = tf.subtract(v_2d[:,1:], v_2d[:,:-1])/dt
        
        # Spatial derivative is along columns
        # Finite difference stencil with periodic boundary conditions
        h=params['h']
        h = tf.cast(h, dtype=tf.float32)
        u_left = tf.concat([u_2d[-1:,: ], u_2d[:-1,:]], axis=0)  
        u_right = tf.concat([u_2d[1:,:], u_2d[:1,:]], axis=0)  
       
        u_xx = (u_right - 2 * u_2d + u_left ) / h**2
        
        v_left = tf.concat([v_2d[-1:,: ], v_2d[:-1,:]], axis=0)  
        v_right = tf.concat([v_2d[1:,:], v_2d[:1,:]], axis=0)  
        
        v_xx = (v_right  - 2 * v_2d  + v_left ) / h**2
        
        f_v = u_t + 0.5 * v_xx[:,1:] + (u_2d[:,1:]**2 + v_2d[:,1:]**2) * v_2d[:,1:]
        f_u = v_t - 0.5 * u_xx[:,1:] - (u_2d[:,1:]**2 + v_2d[:,1:]**2) * u_2d[:,1:]
        
        loss = tf.reduce_mean(tf.square(f_u)) + \
                tf.reduce_mean(tf.square(f_v)) + \
                tf.reduce_mean(tf.square(u0 - u0_pred)) + \
                tf.reduce_mean(tf.square(v0_pred))

        return loss
 
def net_uv(model, x,t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        X = tf.concat([x, t], axis=1)
        u, v = model(X)
    u_x = tape.gradient(u, x)
    v_x = tape.gradient(v, x)
    del tape
    return u, v, u_x, v_x

def net_f_uv(model, inputs):
    x = tf.reshape(inputs[:,0],(-1,1))
    t = tf.reshape(inputs[:,1],(-1,1))
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(x)
        tape1.watch(t)
        u, v, u_x, v_x = net_uv(model, x, t)
    u_t = tape1.gradient(u, t)
    v_t = tape1.gradient(v, t)
    
    u_xx = tape1.gradient(u_x, x)
    v_xx = tape1.gradient(v_x, x)
    del tape1
    f_v = u_t + 0.5 * v_xx + (u**2 + v**2) * v
    f_u = v_t - 0.5 * u_xx - (u**2 + v**2) * u
    return f_u, f_v   