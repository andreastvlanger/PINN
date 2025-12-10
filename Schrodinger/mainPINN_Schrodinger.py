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
import neural_network_Schrodinger as nn
import tensorflow as tf
import custom_loss_Schrodinger as cl
import trainer_Schrodinger
import time
import Save_Functions

# If you want to compare the results you need the data from 
#    https://github.com/maziarraissi/PINNs/blob/master/main/Data/NLS.mat 
# You can run the code without this data (it only effects plotting).  

try:
    import scipy.io
    data = scipy.io.loadmat('./Data/NLS.mat')
except:
    print('CAUTION: Data not found!')


np.random.seed(1234)
tf.random.set_seed(1234)



params = {
    'Problem': 'Schrodinger',
    'DiscretizationModel': 'PINN',
    'epochs': 20000,
    'save_plots': True,  # Whether to save plots or not
    'plot_interval': 100,  # Interval to save plots
    'log_dir': 'logs'
}



params['number_layers'] = 4 
params['skip'] = False
params['skip_every_n'] = 2
bc = False # If False boundary conditions are not included in Neural Networks
params['bc'] = bc
l2_reg = 0
params['l2_reg'] = l2_reg

hidden_layers = params['number_layers']*[100]
activations = params['number_layers']*['tanh']

params['hidden_layers'] = hidden_layers
params['activations'] = activations
if params['skip']:
    params['log_dir'] = (params['Problem'] + '/' + params['DiscretizationModel'] 
                         + '/' + 'bc_'+ str(params['bc'])
                         + '/' + 'layers_'+str(params['number_layers']) + '/' +
                         'l2_reg'+str(params['l2_reg']) + '/' +
                         'skip_'+str(params['skip'])+'/'+
                         'skip_n'+str(params['skip_every_n']))
else:
    params['log_dir'] = (params['Problem'] + '/' + params['DiscretizationModel'] 
                         + '/' + 'bc_'+ str(params['bc'])
                         + '/' + 'layers_'+str(params['number_layers']) + '/' +
                         'l2_reg'+str(params['l2_reg']) + '/' +
                         'skip_'+str(params['skip']))
    
    

import os
import shutil      

if os.path.exists(params['log_dir']):
    shutil.rmtree(params['log_dir'])
os.makedirs(params['log_dir'], exist_ok=True)


params['use_kernel_constraint'] = False
params['use_bias_constraint'] = False

params['learning_rate'] = 0.001

# domain boundaries

lx=-5
rx=5
lt=0
rt=np.pi/2

lb = np.array([lx,lt])
ub=np.array([rx, rt])

if bc:
    @tf.autograph.experimental.do_not_convert
    def boundary_mask(x, y):
        bc = y 
        return bc

    model = nn.BoundaryConditionNN(
        hidden_layers, 
        activations, 
        boundary_mask,
        neurons_output_layer = 2,
        l2_reg=l2_reg, 
        use_kernel_constraint= params['use_kernel_constraint'], 
        use_bias_constraint= params['use_bias_constraint'],
        skip=params['skip'], skip_every_n=params['skip_every_n']
    )
else:
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


# create collocation points on a meshgrid
Nx = 100
Nt = 500
params['Nx'] = Nx
params['Nt'] = Nt
x0 = np.linspace(lx, rx, Nx).reshape(-1, 1)  # 256 x 1
t = np.linspace(lt, rt, Nt).reshape(-1, 1)  # 201 x 1
x_edges = np.linspace(x0.min(), x0.max(), x0.shape[0] + 1)
y_edges = np.linspace(t.min(), t.max(), t.shape[0] + 1)

X, T = np.meshgrid(x0, t)
params['x_edges']=x_edges
params['y_edges']=y_edges
domain_points = np.stack([X.flatten(), T.flatten()], axis=-1)

X0 = np.concatenate((x0, 0*x0), 1) # (x,0) initial points
X_lb = np.concatenate((0 * t + lb[0], t), 1)  # (lb[0], t) lower boundary points
X_ub = np.concatenate((0 * t + ub[0], t), 1)  # (ub[0], t) upper boundary points
Xf, Tf = np.meshgrid(x0[1:-1], t[1:])
X_f = np.stack([Xf.flatten(), Tf.flatten()], axis=-1) # inner points


# Define mask for the boundary points
boundary_mask_np = (X == rx) | (X==lx) | (T == 0)
boundary_mask_rX_np = (X == rx) 
boundary_mask_lX_np = (X==lx)
boundary_mask_T_np = (T == 0)
boundary_mask_rX = boundary_mask_rX_np.flatten().astype(np.float32)
boundary_mask_lX = boundary_mask_lX_np.flatten().astype(np.float32)
boundary_mask_T = boundary_mask_T_np.flatten().astype(np.float32)
boundary_mask = boundary_mask_np.flatten().astype(np.float32)


# Convert all points and boundary mask to tensors
all_points_tf = tf.convert_to_tensor(domain_points, dtype=tf.float32)
boundary_mask_tf = tf.convert_to_tensor(boundary_mask, dtype=tf.float32)

xy_pad = tf.zeros([all_points_tf.shape[0], hidden_layers[1]])
# Set the first two columns to be the values of xy
xy_pad = tf.tensor_scatter_nd_update(xy_pad, indices=[[i, j] for i in range(all_points_tf.shape[0]) for j in range(all_points_tf.shape[1])], updates=tf.reshape(all_points_tf, [-1]))


# Define interior points excluding boundary regions
interior_points = domain_points[~boundary_mask_np.flatten()]
boundary_points_rX = domain_points[boundary_mask_rX_np.flatten()]
boundary_points_lX = domain_points[boundary_mask_lX_np.flatten()]
boundary_points_T = domain_points[boundary_mask_T_np.flatten()]
# Convert interior and boundary points to tensors
interior_points_tf = tf.convert_to_tensor(interior_points, dtype=tf.float32)
boundary_points_rX_tf = tf.convert_to_tensor(boundary_points_rX, dtype=tf.float32)
boundary_points_lX_tf = tf.convert_to_tensor(boundary_points_lX, dtype=tf.float32)
boundary_points_T_tf = tf.convert_to_tensor(boundary_points_T, dtype=tf.float32)

# Pad the coordinates to fit the NN with skip
interior_points_tf_pad = tf.zeros([interior_points_tf.shape[0], hidden_layers[1]])
interior_points_tf_pad = tf.tensor_scatter_nd_update(interior_points_tf_pad, indices=[[i, j] for i in range(interior_points_tf.shape[0]) for j in range(interior_points_tf.shape[1])], updates=tf.reshape(interior_points_tf, [-1]))
boundary_points_rX_tf_pad = tf.zeros([boundary_points_rX_tf.shape[0], hidden_layers[1]])
boundary_points_rX_tf_pad = tf.tensor_scatter_nd_update(boundary_points_rX_tf_pad, indices=[[i, j] for i in range(boundary_points_rX_tf.shape[0]) for j in range(boundary_points_rX_tf.shape[1])], updates=tf.reshape(boundary_points_rX_tf, [-1]))
boundary_points_lX_tf_pad = tf.zeros([boundary_points_lX_tf.shape[0], hidden_layers[1]])
boundary_points_lX_tf_pad = tf.tensor_scatter_nd_update(boundary_points_lX_tf_pad, indices=[[i, j] for i in range(boundary_points_lX_tf.shape[0]) for j in range(boundary_points_lX_tf.shape[1])], updates=tf.reshape(boundary_points_lX_tf, [-1]))

boundary_points_T_tf_pad = tf.zeros([boundary_points_T_tf.shape[0], hidden_layers[1]])
boundary_points_T_tf_pad = tf.tensor_scatter_nd_update(boundary_points_T_tf_pad, indices=[[i, j] for i in range(boundary_points_T_tf.shape[0]) for j in range(boundary_points_T_tf.shape[1])], updates=tf.reshape(boundary_points_T_tf, [-1]))



log_dir = params['log_dir']

#data for the loss function
loss_fc_data={}

loss_fc_data['interior_points'] = tf.convert_to_tensor(X_f, dtype=tf.float32)
loss_fc_data['boundary_points_rX'] = tf.convert_to_tensor(X_ub, dtype=tf.float32)
loss_fc_data['boundary_points_lX'] = tf.convert_to_tensor(X_lb, dtype=tf.float32)
loss_fc_data['boundary_points_T'] = tf.convert_to_tensor(X0, dtype=tf.float32)

loss_fc_data['params'] = params 

u0 = 2/np.cosh(x0)
u0 = tf.convert_to_tensor(u0, dtype=tf.float32)
loss_fc_data['u0'] = u0

loss_function = cl.CustomLoss.custom_loss_PINN
trainer = trainer_Schrodinger.Trainer(model, optimizer, params, loss_function)

start_time = time.time()
###############################################################################
################################ TRAINING #####################################
best_model, best_solution, info = trainer.train(all_points_tf, loss_fc_data)
###############################################################################
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
params['elapsed_time'] = elapsed_time


xy_pad = all_points_tf
best_model.set_weights(best_model.get_weights())
pred = best_model(xy_pad, training=False)
u_pred = pred[0].numpy()
v_pred = pred[1].numpy()
h_pred = np.sqrt(u_pred**2 + v_pred**2)


best_loss_list = info['best_loss_list']
best_loss_epoch_list=info['best_loss_epoch_list']
params['best_loss_list'] = best_loss_list
params['best_loss_epoch_list'] = best_loss_epoch_list
params['best_loss'] = best_loss_list[-1]
params['best_loss_epoch'] = best_loss_epoch_list[-1]

##### Saving part 1 -> part 2 further down
Save_Functions.save_essential_data(log_dir=params['log_dir'],params=params,
                               pred=pred, h_pred = h_pred,
                               best_loss_list=best_loss_list,
                               best_loss_epoch_list = best_loss_epoch_list,
                               best_solution=best_solution, 
                               elapsed_time = elapsed_time)


###############################################################################
##############                    FINISHED                  ###################
###############################################################################


##### Plotting ######

import matplotlib.pyplot as plt

x = params['x_edges']
t = params['y_edges']
H_pred = np.reshape(h_pred, (Nt, Nx))
plt.figure()
plt.imshow(H_pred.T, 
           aspect='auto', 
           extent=[t.min(), t.max(), x.min(), x.max()], 
           cmap='YlGnBu')
plt.colorbar()
plt.xlabel("Time (t)")
plt.ylabel("Space (x)")
plt.savefig(f"{params['log_dir']}/solution.png", bbox_inches='tight')
plt.show()
plt.close()


try:
    #exact solution
    t_data = data['tt'].flatten()[:,None] #201
    x_data = data['x'].flatten()[:,None] #256
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)
    
    u_star = Exact_u.T.flatten()[:,None]
    v_star = Exact_v.T.flatten()[:,None]
    h_star = Exact_h.T.flatten()[:,None]

    X, T = np.meshgrid(x_data, t_data)
    domain_points = np.stack([X.flatten(), T.flatten()], axis=-1)
    domain_points_tf = tf.convert_to_tensor(domain_points, dtype=tf.float32)
    best_model.set_weights(best_model.get_weights())
    pred = best_model(domain_points_tf, training=False)
    u_pred = pred[0].numpy()
    v_pred = pred[1].numpy()
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
    H_pred = np.reshape(h_pred, (t_data.shape[0],  x_data.shape[0]))


    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))
    
    params['error_u'] = error_u
    params['error_v'] = error_v
    params['error_h'] = error_h


##### Saving part 2
    Save_Functions.save_parameters(params['log_dir'], params)




############## slices ################ 

    time1=50
    time2=100
    time3=125
    for time_ in [time1,time2,time3]:
        plt.figure()
        plt.plot(x_data,Exact_h[:,time_], 'b-', linewidth = 3, label = 'Exact') 
        plt.plot(x_data,H_pred[time_,:], 'r--', linewidth = 3, label = 'Prediction')
        plt.xlabel("$x$", fontsize = 12)
        plt.ylabel("$|h(t,x)|$", fontsize = 12)
        plt.title('$t = %.3f$' % (t_data[time_]), fontsize = 12)
        plt.xlim([-5.1,5.1])
        plt.ylim([-0.1,5.1])
        plt.xticks([-5, 0, 5], fontsize=12)  
        plt.yticks([0,1,2,3,4,5], fontsize=12)  
        plt.legend()
        plt.savefig(f"{params['log_dir']}/solution_{t_data[time_,0]:.3f}.png", bbox_inches='tight')
        plt.show()
        plt.close()
    

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    plt.figure()
    plt.plot(t_data[time1]* np.ones(2) , line, 'k--', linewidth=1)
    plt.plot(t_data[time2] * np.ones(2), line, 'k--', linewidth=1)
    plt.plot(t_data[time3]* np.ones(2) , line, 'k--', linewidth=1)
    plt.imshow(H_pred.T, 
               aspect='auto', 
               extent=[t.min(), t.max(), x.min(), x.max()], 
               cmap='YlGnBu')
    plt.colorbar()
    plt.xlabel("Time (t)")
    plt.ylabel("Space (x)")
    plt.savefig(f"{params['log_dir']}/solution_dash.png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    plt.figure()
    plt.imshow(Exact_h, 
               aspect='auto', 
               extent=[t.min(), t.max(), x.min(), x.max()], 
               cmap='YlGnBu')
    plt.colorbar()
    plt.xlabel("Time (t)")
    plt.ylabel("Space (x)")
    plt.savefig(f"{params['log_dir']}/exact_solution.png", bbox_inches='tight')
    plt.show()
    plt.close()
except:
    print('=====================================================')
    print('Comparison plots not available due to missing data!!!')
    print('=====================================================')
    