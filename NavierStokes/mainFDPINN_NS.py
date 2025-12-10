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

Created on Mon Dec  9 17:35:49 2024

@author: Andreas Langer
"""

import numpy as np
import time
import tensorflow as tf

import Save_Functions
import neural_network_NS as nn
import custom_loss_NS as cl
import trainer_NS

np.random.seed(1234)
tf.random.set_seed(1234)

params = {
    'Problem': 'Navier-Stokes',
    'DiscretizationModel': 'FDPINN', 
    'epochs': 500000,
    'save_plots': True,  # Whether to save plots or not
    'plot_interval': 100,  # Interval to save plots
    'log_dir': 'logs'
}
params['number_layers'] = 9
params['number_neurons'] = 100
hidden_layers = params['number_layers']*[params['number_neurons']]
params['activation_function'] = 'relu' #'tanh'#
activations = params['number_layers']*[params['activation_function']]
params['skip'] = False
params['skip_every_n'] = 2
bc = False # No boundary conditions available
params['bc'] = bc


params['hidden_layers'] = hidden_layers
params['activations'] = activations
params['noise_level'] = 0.01
l2_reg = 0
params['l2_reg'] = l2_reg
if params['skip']:
    params['log_dir'] = (params['Problem'] + '/' + params['DiscretizationModel'] 
                         + '/' + 'bc_'+ str(params['bc']) + '/' +
                         'activation_'+params['activation_function'] + '/' +
                         'layers_'+str(params['number_layers']) + '/' +
                         'neurons_'+str(params['number_neurons']) + '/' +
                         'skip_'+str(params['skip'])+'/'+
                         'skip_n'+str(params['skip_every_n'])+ '/' +
                         'noise_'+str(params['noise_level'])
                         )
else:
    params['log_dir'] = (params['Problem'] + '/' + params['DiscretizationModel'] 
                         + '/' + 'bc_'+ str(params['bc']) + '/' + 
                         'activation_'+params['activation_function'] + '/' +
                         'layers_'+str(params['number_layers']) + '/' +
                         'neurons_'+str(params['number_neurons']) + '/' +
                         'skip_'+str(params['skip']) + '/' +
                         'noise_'+str(params['noise_level'])
                         )
    

import os
import shutil      

if os.path.exists(params['log_dir']):
    shutil.rmtree(params['log_dir'])
os.makedirs(params['log_dir'], exist_ok=True)



params['use_kernel_constraint'] = False
params['use_bias_constraint'] = False

params['learning_rate'] = 0.001




DTYPE = tf.float32  # or tf.float64

model = nn.NeuralNetwork(
    hidden_layers, 
    activations, 
    neurons_output_layer = 2,
    l2_reg=l2_reg, 
    use_kernel_constraint= params['use_kernel_constraint'], 
    use_bias_constraint= params['use_bias_constraint'],
    skip=params['skip'], skip_every_n=params['skip_every_n']
)

optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

import scipy.io
data = scipy.io.loadmat("cylinder_like_periodic_box.mat")
U_star = data['U_star'] # N x 2 x T; N = 5000, T=200
P_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2
# create collocation points on a meshgrid
Nx = data["Nx"] #100
Ny = data["Ny"]#50
Nt = data["Nt"]#200

hx = data["dx"]
hy = data["dy"]
dt = data["dt"]

params['hx'] = hx
params['hy'] = hy
params['dt'] = dt


log_dir = params['log_dir']

#data for the loss function
lambda_boundary=1e2
loss_fc_data={}

loss_fc_data['params'] = params 

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data 
XX = np.tile(X_star[:,0:1], (1,T)) # N x T
YY = np.tile(X_star[:,1:2], (1,T)) # N x T
TT = np.tile(t_star, (1,N)).T # N x T

UU = U_star[:,0,:] # N x T
VV = U_star[:,1,:] # N x T
PP = P_star # N x T

x_unique = np.unique(X_star[:,0])
y_unique = np.unique(X_star[:,1])
Nx, Ny = len(x_unique), len(y_unique)
Nt = T
params['Nx'] = Nx
params['Ny'] = Ny
params['Nt'] = Nt

sample = 1
params['sample'] = sample

# --- Subsample every second in each dimension ---
ix = np.arange(0, Nx, sample)        # x indices we keep
iy = np.arange(0, Ny, sample)        # y indices we keep
it = np.arange(0, Nt, sample)        # time indices we keep

# Spatial rows to keep in the (N, T) matrices:
# build the 2D grid (iy, ix) then map to linear row indices
iy_grid, ix_grid = np.meshgrid(iy, ix, indexing="ij")     # shapes (Ny', Nx')
row_idx = (iy_grid * Nx + ix_grid).ravel()                # shape Ny'*Nx'

# --- Slice the (N, T) arrays, then flatten like before ---
XX_sub = XX[row_idx][:, it]   # (N_sub, Nt_sub)
YY_sub = YY[row_idx][:, it]
TT_sub = TT[row_idx][:, it]

UU_sub = UU[row_idx][:, it]
VV_sub = VV[row_idx][:, it]
PP_sub = PP[row_idx][:, it]

# Flatten to column vectors (same order as your original code)
x = XX_sub.flatten()[:, None]
y = YY_sub.flatten()[:, None]
t = TT_sub.flatten()[:, None]

u = UU_sub.flatten()[:, None]
v = VV_sub.flatten()[:, None]
p = PP_sub.flatten()[:, None]

p_tf = tf.convert_to_tensor(p, dtype=DTYPE)
loss_fc_data['p'] = p_tf

noise_level = params['noise_level']
if noise_level > 0:
    u = u + noise_level*np.std(u)*np.random.randn(u.shape[0], u.shape[1])
    v = v + noise_level*np.std(v)*np.random.randn(v.shape[0], v.shape[1])
u_tf = tf.convert_to_tensor(u, dtype=DTYPE)
v_tf = tf.convert_to_tensor(v, dtype=DTYPE)
loss_fc_data['u_label'] = u_tf
loss_fc_data['v_label'] = v_tf


X_input = np.hstack([x, y, t])  

X_tf = tf.convert_to_tensor(X_input, dtype=DTYPE)
loss_fc_data['X_tf'] = X_tf

lambda1 = tf.Variable(0.0, trainable=True, dtype=DTYPE, name="lambda1") #1.0
lambda2 = tf.Variable(0.0, trainable=True, dtype=DTYPE, name="lambda2") #0.1

dx_sub = x_unique[1] - x_unique[0]   # becomes 2*dx if grid is uniform
dy_sub = y_unique[1] - y_unique[0]
dt_sub = float(t_star[1] - t_star[0])
# After subsampling by 2:
dx_sub *= sample; dy_sub *= sample; dt_sub *= sample
dx_sub = tf.constant(dx_sub, dtype=DTYPE)  
dy_sub = tf.constant(dy_sub, dtype=DTYPE)
dt_sub = tf.constant(dt_sub, dtype=DTYPE)


params['Nx'] = len(ix)
params['Ny'] = len(iy)
params['Nt'] = len(it)
params['hx'] = dx_sub
params['hy'] = dy_sub
params['dt'] = dt_sub


print("Nx_sub, Ny_sub, Nt_sub:", len(ix), len(iy), len(it))
print("X_input:", X_input.shape, "u:", u_tf.shape, "v:", v_tf.shape)

loss_function = cl.CustomLoss.custom_loss_FDPINN

trainer = trainer_NS.Trainer(model, optimizer, params, loss_function, lambda1, lambda2)


start_time = time.time()
###############################################################################
################################ TRAINING #####################################
best_model, info, lambda1, lambda2, pred = trainer.train(X_tf, loss_fc_data)
###############################################################################
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
params['elapsed_time'] = elapsed_time

print(f"lambda1 = {lambda1}")
print(f"lambda2 = {lambda2}")

best_model.set_weights(best_model.get_weights())
pred = best_model(X_tf, training=False)
p_pred=pred[1]

p_3d   = p_pred.numpy().reshape(Ny, Nx, Nt)


best_loss_list = info['best_loss_list']
best_loss_epoch_list=info['best_loss_epoch_list']
params['best_loss_list'] = best_loss_list
params['best_loss_epoch_list'] = best_loss_epoch_list
params['best_loss'] = best_loss_list[-1]
params['best_loss_epoch'] = best_loss_epoch_list[-1]

##### Saving part 
Save_Functions.save_essential_data(log_dir=params['log_dir'],params=params,
                               pred=pred, p_3d=p_3d,
                               best_loss_list=best_loss_list,
                               best_loss_epoch_list = best_loss_epoch_list,
                               elapsed_time = elapsed_time)

Save_Functions.save_parameters(params['log_dir'], params)

