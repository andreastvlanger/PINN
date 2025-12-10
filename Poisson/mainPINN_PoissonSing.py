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

Created on Sun Dec  8 15:51:47 2024

@author: andreas langer
"""
import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

import neural_network as nn
import custom_loss as cl
import trainer as tr
import PoissonFDM
import time


params = {
    'Problem': 'PoissonSingularity',
    'DiscretizationModel': 'PINN', 
    'epochs': 200000,
    'save_plots': True,  # Whether to save plots or not
    'plot_interval': 500,  # Interval to save plots
    'log_dir': 'logs'
}
params['number_layers'] = 7
params['skip'] = False
params['skip_every_n'] = 2
bc = False # If False boundary conditions are not included in Neural Networks
params['bc'] = bc
l2_reg = 0
params['l2_reg'] = l2_reg
params['lambda_boundary'] = 10000

if params['skip']:
    params['log_dir'] = (params['Problem'] + '/' + params['DiscretizationModel'] 
                         + '/' + 'bc_'+ str(params['bc'])
                         + '/' + 'layers_'+str(params['number_layers']) + '/' +
                         'l2_reg'+str(params['l2_reg']) + '/' +
                         'lambda_boundary_'+str(params['lambda_boundary']) + '/' +
                         'skip_'+str(params['skip'])+'/'+
                         'skip_n'+str(params['skip_every_n']))
else:
    params['log_dir'] = (params['Problem'] + '/' + params['DiscretizationModel'] 
                         + '/' + 'bc_'+ str(params['bc'])
                         + '/' + 'layers_'+str(params['number_layers']) + '/' +
                         'l2_reg'+str(params['l2_reg']) + '/' +
                         'lambda_boundary_'+str(params['lambda_boundary']) + '/' +
                         'skip_'+str(params['skip']))
    
        

if os.path.exists(params['log_dir']):
    shutil.rmtree(params['log_dir'])
os.makedirs(params['log_dir'], exist_ok=True)

hidden_layers = params['number_layers']*[32]
activations = params['number_layers']*['tanh']

params['hidden_layers'] = hidden_layers
params['activations'] = activations



params['use_kernel_constraint'] = False
params['use_bias_constraint'] = False

params['learning_rate'] = 0.001


# Define the boundary mask function
def cut_mask(x, y):
    # Returns 0 if within [0,1) x {0}, else returns 1
    return tf.where((x >= 0) & (x < 1) & (tf.abs(y) == 0), 0.0, 1.0)

@tf.autograph.experimental.do_not_convert
def boundary_mask(x, y):
    #bc = tf.where((tf.abs(x) == 1) | (tf.abs(y) == 1),0.0, 1.0 )
    bc = (1 - x**2) * (1 - y**2) #Works quite good here!
    return bc * cut_mask(x,y)

if bc:
    model = nn.BoundaryConditionNN(
        hidden_layers, 
        activations, 
        boundary_mask,
        neurons_output_layer = 1,
        l2_reg=l2_reg, 
        use_kernel_constraint= params['use_kernel_constraint'], 
        use_bias_constraint= params['use_bias_constraint'],
        skip=params['skip'], skip_every_n=params['skip_every_n']
    )
else:
    model = nn.NeuralNetwork(
        hidden_layers, 
        activations, 
        neurons_output_layer = 1,
        l2_reg=l2_reg, 
        use_kernel_constraint= params['use_kernel_constraint'], 
        use_bias_constraint= params['use_bias_constraint'],
        skip=params['skip'], skip_every_n=params['skip_every_n']
    )
    

optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

lx=-1
rx=1
ly=-1
uy=1

Nx=41
Ny=Nx
params['Nx'] = Nx
params['Ny'] = Ny

x = np.linspace(lx, rx, Nx)
y = np.linspace(ly, uy, Ny)
X, Y = np.meshgrid(x, y)
domain_points = np.stack([X.flatten(), Y.flatten()], axis=-1)

# Define mask for the cut-out region and boundary points
cut_mask_np = (X >= 0) & (X < rx) & (np.abs(Y) == 0)
boundary_mask_np = (X == rx) | (X==lx) | (Y == ly) | (Y == uy) | cut_mask_np
# boundary_mask_np = (np.abs(X) == 1) | (np.abs(Y) == 1) | cut_mask_np
boundary_mask = boundary_mask_np.flatten().astype(np.float32)

# Convert all points and boundary mask to tensors
all_points_tf = tf.convert_to_tensor(domain_points, dtype=tf.float32)
boundary_mask_tf = tf.convert_to_tensor(boundary_mask, dtype=tf.float32)

xy_pad = tf.zeros([all_points_tf.shape[0], hidden_layers[1]])
# Set the first two columns to be the values of xy
xy_pad = tf.tensor_scatter_nd_update(xy_pad, indices=[[i, j] for i in range(all_points_tf.shape[0]) for j in range(all_points_tf.shape[1])], updates=tf.reshape(all_points_tf, [-1]))


# Define interior points excluding both boundary and cut-out regions
interior_points = domain_points[~boundary_mask_np.flatten()]
boundary_points = domain_points[boundary_mask_np.flatten()]
# Convert interior and boundary points to tensors
interior_points_tf = tf.convert_to_tensor(interior_points, dtype=tf.float32)
boundary_points_tf = tf.convert_to_tensor(boundary_points, dtype=tf.float32)

# Pad the coordinates to fit the NN with skip
interior_points_tf_pad = tf.zeros([interior_points_tf.shape[0], hidden_layers[1]])
interior_points_tf_pad = tf.tensor_scatter_nd_update(interior_points_tf_pad, indices=[[i, j] for i in range(interior_points_tf.shape[0]) for j in range(interior_points_tf.shape[1])], updates=tf.reshape(interior_points_tf, [-1]))
boundary_points_tf_pad = tf.zeros([boundary_points_tf.shape[0], hidden_layers[1]])
boundary_points_tf_pad = tf.tensor_scatter_nd_update(boundary_points_tf_pad, indices=[[i, j] for i in range(boundary_points_tf.shape[0]) for j in range(boundary_points_tf.shape[1])], updates=tf.reshape(boundary_points_tf, [-1]))

hx = 2 / (Nx - 1)  # Grid spacing
hy = 2 / (Nx - 1)
params['hx'] = hx
params['hy'] = hy

log_dir = params['log_dir']

#data for the loss function
loss_fc_data={}
loss_fc_data['interior_points'] = interior_points_tf_pad
loss_fc_data['boundary_points'] = boundary_points_tf_pad
loss_fc_data['lambda_boundary'] = params['lambda_boundary']
loss_fc_data['params'] = params 


loss_function = cl.CustomLoss.custom_loss_PINN
trainer = tr.Trainer(model, optimizer, params, loss_function)

params['X'] = X
params['Y'] = Y



start_time = time.time()
###############################################################################
################################ TRAINING #####################################
best_model, best_solution, data = trainer.train(xy_pad, loss_fc_data)
###############################################################################
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
params['elapsed_time'] = elapsed_time


# After training, generate high-resolution grid for final plotting
Nx_hd = 101
params['Nx_hd'] = Nx_hd
tensor_hd = tf.linspace(-1.0, 1.0, Nx_hd)
x_tf_hd = tf.reshape(tensor_hd, (-1, 1))
y_tf_hd = tf.reshape(tensor_hd, (-1, 1))
X_hd, Y_hd = tf.meshgrid(x_tf_hd[:, 0], y_tf_hd[:, 0])
xy_hd = tf.stack([tf.reshape(X_hd, [-1]), tf.reshape(Y_hd, [-1])], axis=1)

xy_pad_hd = tf.zeros([xy_hd.shape[0], hidden_layers[1]])
# Set the first two columns to be the values of xy
xy_pad_hd = tf.tensor_scatter_nd_update(xy_pad_hd, indices=[[i, j] 
                                    for i in range(xy_hd.shape[0]) 
                                    for j in range(xy_hd.shape[1])], 
                                        updates=tf.reshape(xy_hd, [-1]))


best_model.set_weights(best_model.get_weights())
pred = best_model(xy_pad_hd, training=False).numpy()

u_pred = np.reshape(pred,(Nx_hd, Nx_hd))

plt.figure()
plt.pcolormesh(X_hd, Y_hd, u_pred, shading='auto', cmap='rainbow')
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"{params['log_dir']}/solution_hd.png", bbox_inches='tight')
plt.show()
plt.close()



best_loss_list = data['best_loss_list']
best_loss_epoch_list=data['best_loss_epoch_list']
params['best_loss_list'] = best_loss_list
params['best_loss_epoch_list'] = best_loss_epoch_list
params['best_loss'] = best_loss_list[-1]
params['best_loss_epoch'] = best_loss_epoch_list[-1]

PoissonFDM.save_parameters(params['log_dir'], params)
PoissonFDM.save_essential_data(log_dir=params['log_dir'],params=params,
                               u_pred=u_pred, best_loss_list=best_loss_list,
                               best_loss_epoch_list = best_loss_epoch_list,
                               best_solution=best_solution, 
                               elapsed_time = elapsed_time)