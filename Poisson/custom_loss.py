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

@author: andreas langer
"""
import tensorflow as tf

class CustomLoss:
    @staticmethod 
    def custom_loss_FDPINN(model, u, data):
        b = data['b']
        A = data['A']
        params = data['params']
 
        if b.ndim == 1:  # (n,) shape has 1 dimension
            b = tf.reshape(b,(-1, 1))
        h=params['hx']
        Nx = params['Nx']
        loss = numerical_integration_2D(tf.square(A@u - b), h,h,Nx,Nx)
        return loss
    
    @staticmethod
    # Define loss function with interior conditions only
    def custom_loss_PINN(model, u, data):
        interior_points = data['interior_points']
        boundary_points = data['boundary_points']
        lambda_boundary = data['lambda_boundary']
        
        # Interior loss: enforce -Δu = 1 in the interior
        laplacian_u = compute_laplacian(model, interior_points)
        interior_residual = -laplacian_u - 1.0
        interior_loss = tf.reduce_mean(tf.square(interior_residual))
        
        # Boundary loss: enforce u = 0 on the boundary
        u_boundary = model(boundary_points)
        boundary_loss = tf.reduce_mean(tf.square(u_boundary))
        
        # Total loss combines interior and boundary losses
        return interior_loss + lambda_boundary * boundary_loss


def numerical_integration_2D(u, hx, hy, m1, m2, rule='AR'):
    if rule == 'TR': # Composite Trapezoid Rule
        u_reshaped = tf.reshape(u, [m1, m2])
        return hx * hy / 4 * (
            u_reshaped[0, 0] + u_reshaped[0, -1] + u_reshaped[-1, 0] + u_reshaped[-1, -1] +
            2 * tf.reduce_sum(u_reshaped[1:-1, 0]) + 2 * tf.reduce_sum(u_reshaped[1:-1, -1]) +
            2 * tf.reduce_sum(u_reshaped[0, 1:-1]) + 2 * tf.reduce_sum(u_reshaped[-1, 1:-1]) +
            4 * tf.reduce_sum(u_reshaped[1:-1, 1:-1]))
    elif rule == 'AR':
        return hx * hy * tf.reduce_sum(u)
    else:
        return None
    
# Compute Laplacian of the model's prediction for interior points
def compute_laplacian(model, inputs):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(inputs)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(inputs)
            u = model(inputs)
        u_x = tape1.gradient(u, inputs)[:, 0]
        u_y = tape1.gradient(u, inputs)[:, 1]
    u_xx = tape2.gradient(u_x, inputs)[:, 0]
    u_yy = tape2.gradient(u_y, inputs)[:, 1]
    laplacian_u = u_xx + u_yy
    del tape1, tape2
    return laplacian_u


    