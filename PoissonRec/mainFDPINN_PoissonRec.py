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
import PoissonFDM_Rec
import time

params = {
    'Problem': 'PoissonRec',
    'DiscretizationModel': 'FDPINN', 
    'epochs': 200000,
    'save_plots': True,  # Whether to save plots or not
    'plot_interval': 500,  # Interval to save plots
    'log_dir': 'logs'
}
params['number_layers'] = 7
params['number_neurons'] = 32
hidden_layers = params['number_layers']*[params['number_neurons']]
activations = params['number_layers']*['relu']
params['skip'] = False
params['skip_every_n'] = 2
l2_reg = 0
params['l2_reg'] = l2_reg
params['bc'] = True #False (False=default) # If False boundary conditions are not included in Neural Networks

if params['skip']:
    params['log_dir'] = (params['Problem'] + '/' + params['DiscretizationModel'] 
                         + '/' + 'bc_'+ str(params['bc']) + '/' +
                          'layers_'+str(params['number_layers']) + '/' +
                          'neurons_'+str(params['number_neurons']) + '/' +
                          'l2_reg'+str(params['l2_reg']) + '/' +
                          'skip_'+str(params['skip'])+'/'+
                          'skip_n'+str(params['skip_every_n']))
else:
    params['log_dir'] = (params['Problem'] + '/' + params['DiscretizationModel'] 
                          + '/' + 'bc_'+ str(params['bc']) + '/' + 
                          'layers_'+str(params['number_layers']) + '/' +
                          'neurons_'+str(params['number_neurons']) + '/' +
                          'l2_reg'+str(params['l2_reg']) + '/' +
                          'skip_'+str(params['skip']))
    
        

if os.path.exists(params['log_dir']):
    shutil.rmtree(params['log_dir'])
os.makedirs(params['log_dir'], exist_ok=True)




params['hidden_layers'] = hidden_layers
params['activations'] = activations


params['use_kernel_constraint'] = False
params['use_bias_constraint'] = False

params['learning_rate'] = 0.001


@tf.function
def phi(t):
    """C^∞ flat function"""
    return tf.where(t > 0.0, tf.exp(-1.0 / t), tf.zeros_like(t))

@tf.function
def smooth_plateau_tf(x, eps=0.05):
    """
    Smooth plateau:
    - 1 for |x| <= 1-eps
    - 0 for |x| >= 1
    - C^∞ transition on (1-eps, 1)
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    absx = tf.abs(x)

    # Core plateau
    core = absx <= (1.0 - eps)

    # Transition region
    trans = tf.logical_and(absx > (1.0 - eps), absx < 1.0)
    s = (1.0 - absx) / eps

    transition_val = phi(s) / (phi(s) + phi(1.0 - s))

    return tf.where(core, 1.0,
                    tf.where(trans, transition_val, 0.0))

def smooth_cutoff_2d(x, y, eps):
    return smooth_plateau_tf(x, eps) * smooth_plateau_tf(y, eps)

@tf.autograph.experimental.do_not_convert
def boundary_mask1(x, y):
    epsilon = 0.04 #even
    bc = smooth_cutoff_2d(x, y, epsilon)
    return bc

if params['bc']:
    model = nn.BoundaryConditionNN(
        hidden_layers, 
        activations, 
        boundary_mask1,
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

lx=-1.
rx=1.
ly=-1.
uy=1.

Nx=41
Ny=Nx
params['Nx'] = Nx
params['Ny'] = Ny
x = tf.linspace(lx, rx, Nx)
y = tf.linspace(lx, rx, Ny)
x_tf = tf.reshape(x, (-1, 1))
y_tf = tf.reshape(y, (-1, 1))
X, Y = tf.meshgrid(x_tf[:, 0], y_tf[:, 0])
xy = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)

xy_pad = tf.zeros([xy.shape[0], hidden_layers[1]])
# Set the first two columns to be the values of xy
xy_pad = tf.tensor_scatter_nd_update(xy_pad, indices=[[i, j] for i in range(xy.shape[0]) for j in range(xy.shape[1])], updates=tf.reshape(xy, [-1]))

if not params['skip']:
    xy_pad = xy #no padding needed, since we do not skip!!!
    print(params['skip'])
#test = model(xy_pad)
#model.summary()
#
hx = 2 / (Nx - 1)  # Grid spacing
hy = 2 / (Ny - 1)
params['hx'] = hx
params['hy'] = hy

A = PoissonFDM_Rec.SysMatrix(Nx, Ny, hx, hy, x,y).toarray()
A = tf.convert_to_tensor(A, dtype=tf.float32)
b = tf.convert_to_tensor(PoissonFDM_Rec.RHS(Nx,Ny,x.numpy(), y.numpy()), dtype=tf.float32)

log_dir = params['log_dir']

#data for the loss function
loss_fc_data={}
loss_fc_data['b'] = b
loss_fc_data['params'] = params
loss_fc_data['A'] = A

loss_function = cl.CustomLoss.custom_loss_FDPINN
trainer = tr.Trainer(model, optimizer, params, loss_function)

params['X'] = X
params['Y'] = Y

start_time = time.time()
best_model, best_solution, data = trainer.train(xy_pad, loss_fc_data)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
params['elapsed_time'] = elapsed_time


u2D = np.reshape(best_solution,(Nx, Ny))
# Plot the heatmap   
plt.figure()
plt.pcolormesh(X, Y, u2D, shading='auto', cmap='rainbow')
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"{params['log_dir']}/solution.png", bbox_inches='tight')
plt.show()
plt.close()

print("--------")
print("create theoretic solution")
u_exact = PoissonFDM_Rec.poisson_exact(x.numpy(), y.numpy(), N=501)  
plt.pcolormesh(X, Y, u_exact, shading='auto', cmap='rainbow')
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"{params['log_dir']}/theoretic_solution.png", bbox_inches='tight')
plt.show()
plt.close()

# Plot the heatmap
plt.pcolormesh(X, Y, u_exact - u2D, shading='auto', cmap='rainbow')
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"{params['log_dir']}/error.png", bbox_inches='tight')
plt.show()
plt.close()

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
if not params['skip']:
    xy_pad_hd = xy_hd
    
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

print("--------")
print("create theoretic solution")
x_hd = np.linspace(lx, rx, Nx_hd)
y_hd = np.linspace(ly, uy, Nx_hd)
u_exact_hd = PoissonFDM_Rec.poisson_exact(x_hd, y_hd, N=501)  
print("theoretic solution created")
plt.pcolormesh(X_hd, Y_hd, u_exact_hd, shading='auto', cmap='rainbow')
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"{params['log_dir']}/theoretic_solution_hd.png", bbox_inches='tight')
plt.show()
plt.close()

# Plot the heatmap
plt.pcolormesh(X_hd, Y_hd, u_exact_hd - u_pred, shading='auto', cmap='rainbow')
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"{params['log_dir']}/error_hd.png", bbox_inches='tight')
plt.show()
plt.close()

best_loss_list = data['best_loss_list']
best_loss_epoch_list=data['best_loss_epoch_list']
params['best_loss_list'] = best_loss_list
params['best_loss_epoch_list'] = best_loss_epoch_list
params['best_loss'] = best_loss_list[-1]
params['best_loss_epoch'] = best_loss_epoch_list[-1]

PoissonFDM_Rec.save_parameters(params['log_dir'], params)
PoissonFDM_Rec.save_essential_data(log_dir=params['log_dir'],params=params,
                               u_pred=u_pred, best_loss_list=best_loss_list,
                               best_loss_epoch_list = best_loss_epoch_list,
                               best_solution=best_solution, 
                               elapsed_time = elapsed_time)
