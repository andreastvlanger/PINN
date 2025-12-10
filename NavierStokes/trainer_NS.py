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

Created on Sun Dec  8 15:33:09 2024

@author: Andreas Langer
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, params, loss_function, lambda1, lambda2):
        self.model = model
        self.optimizer = optimizer
        self.params = params
        self.loss_function = loss_function
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.data_tensors = None

    def _bind_data_tensors(self, data, dtype):
        # cache everything the loss needs as TF tensors
        p = data['params']
        self.data_tensors = {
            'params': {
                'Nx': p['Nx'],                
                'Ny': p['Ny'],
                'Nt': p['Nt'],
                'hx': tf.convert_to_tensor(p['hx'], dtype),
                'hy': tf.convert_to_tensor(p['hy'], dtype),
                'dt': tf.convert_to_tensor(p['dt'], dtype),
            },
            'u_label': tf.convert_to_tensor(data['u_label'], dtype),
            'v_label': tf.convert_to_tensor(data['v_label'], dtype),
            'X_tf': data['X_tf'], 
        }
    
    @tf.function(reduce_retracing=True, jit_compile=True) 
    def train_step(self, x):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            total_loss = self.loss_function( predictions, self.lambda1, 
                                            self.lambda2, self.data_tensors, 
                                            self.model)
        train_vars = self.model.trainable_variables + [self.lambda1, self.lambda2]
        # Compute gradients with respect to the model's trainable variables
        gradients = tape.gradient(total_loss, train_vars)
        # Apply gradients to update model parameters
        self.optimizer.apply_gradients(zip(gradients, train_vars))

        return total_loss
    
    def train(self, x_train, data):
        params = data['params']
        self._bind_data_tensors(data, dtype=x_train.dtype)
        best_loss = float('inf')
        best_weights = None
        
        best_loss_list=[] #List of best losses 
        best_loss_epoch_list=[] #List when the best loss is changed
    
        for epoch in range(self.params['epochs']):
            total_loss = self.train_step(x_train)
            predictions = self.model(x_train, training=False)
            total_loss= self.loss_function(predictions, self.lambda1, 
                                           self.lambda2,self.data_tensors, 
                                           self.model)
    
            current_total_loss = total_loss.numpy()
            if current_total_loss < best_loss:
                best_loss = current_total_loss
                best_loss_list.append(best_loss)
                best_loss_epoch_list.append(epoch)
                best_weights = self.model.get_weights()
                pred = predictions
                self.model.save_weights(f"{params['log_dir']}/best_weights_ReLU_32.weights.h5")
                self.model.save(f"{params['log_dir']}/best_model.keras")
                params['lambda1'] = self.lambda1.numpy()
                params['lambda2'] = self.lambda2.numpy()
                
            if epoch % params['plot_interval'] == 0:
                print(f"Epoch {epoch}, Total Loss: {total_loss.numpy()}")
                
                print(f"  Best Loss so far: {best_loss}")
                print(f" lambda1 = {params['lambda1']}; lambda2 = {params['lambda2']}")
                p_pred = pred[1]
                
                p_exact_flat = data['p']
                t_idx = int(params["Nt"]/2)
                self.plot_pressure_compare(x_train, p_pred, p_exact_flat, params, 
                                           best_loss_epoch_list, 
                                           t_idx)
               
        self.model.set_weights(best_weights)
        info = {
            'best_loss_list': best_loss_list,
            'best_loss_epoch_list': best_loss_epoch_list
        }
        return self.model, info, params['lambda1'], params['lambda2'], pred
    
   
    def plot_pressure_compare(self, X_tf, p_pred, p_exact, params, 
                              best_loss_epoch_list, t_idx=0):
        """
        Compare predicted vs exact pressure at time index t_idx.
    
        X_tf    : tf.Tensor, shape (N'*T', 3)  flattened inputs [x,y,t]
        p_pred  : tf.Tensor, shape (N'*T', 1)  predicted pressure (flattened)
        p_exact : np.ndarray or tf.Tensor, shape (N'*T', 1) or (N'*T',) or (Ny,Nx,Nt)
        params  : dict with keys 'Ny','Nx','Nt'
        t_idx   : int, time index (0..Nt-1)
        """
    
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.gridspec as gridspec
    
        Ny, Nx, Nt = params['Ny'], params['Nx'], params['Nt']
    
        # to numpy
        X_np = X_tf.numpy()
        p_pred_np = p_pred.numpy()
        if hasattr(p_exact, "numpy"):
            p_exact_np = p_exact.numpy()
        else:
            p_exact_np = np.asarray(p_exact)
    
        # reshape to (Ny, Nx, Nt)
        p_pred_3d = p_pred_np.reshape(Ny, Nx, Nt)
        if p_exact_np.ndim == 1 or (p_exact_np.ndim == 2 and min(p_exact_np.shape) == 1):
            p_exact_3d = p_exact_np.reshape(Ny, Nx, Nt)
        elif p_exact_np.ndim == 3 and p_exact_np.shape == (Ny, Nx, Nt):
            p_exact_3d = p_exact_np
        else:
            raise ValueError("p_exact must be flattened (N'*T',1)/(N'*T',) or shaped (Ny,Nx,Nt).")
    
        # coordinates for extent
        x_vals = np.unique(X_np[:, 0])
        y_vals = np.unique(X_np[:, 1])
        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        y_min, y_max = float(y_vals.min()), float(y_vals.max())
        t_vals = np.unique(X_np[:, 2])
    
        # fields at chosen time
        P_pred = p_pred_3d[:, :, t_idx]
        P_true = p_exact_3d[:, :, t_idx]
    
        fig = plt.figure(figsize=(10, 4))
        gs2 = gridspec.GridSpec(1, 2)
        gs2.update(top=0.95, bottom=0.15, left=0.08, right=0.92, wspace=0.4)
    
        # Predicted
        ax = plt.subplot(gs2[:, 0])
        h = ax.imshow(P_pred, interpolation='nearest', cmap='rainbow',
                      extent=[x_min, x_max, y_min, y_max],
                      origin='lower', aspect='auto')
        div = make_axes_locatable(ax); cax = div.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(h, cax=cax)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_aspect('equal', 'box')
        ax.set_title('Predicted pressure', fontsize=10)
    
        # Exact
        ax = plt.subplot(gs2[:, 1])
        h = ax.imshow(P_true, interpolation='nearest', cmap='rainbow',
                      extent=[x_min, x_max, y_min, y_max],
                      origin='lower', aspect='auto')
        div = make_axes_locatable(ax); cax = div.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(h, cax=cax)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_aspect('equal', 'box')
        ax.set_title('Exact pressure', fontsize=10)
        
    
        fig.suptitle(f"Pressure comparison at t = {t_vals[t_idx]:.3f} (index {t_idx})", y=0.995, fontsize=11)
        plt.savefig(f"{params['log_dir']}/pressure{best_loss_epoch_list[-1]}.png", bbox_inches='tight')
        #plt.show() # uncomment to see plot! 
        plt.close(fig) 