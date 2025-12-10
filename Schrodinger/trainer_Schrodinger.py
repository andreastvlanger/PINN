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
    def __init__(self, model, optimizer, params, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.params = params
        self.loss_function = loss_function

    @tf.function
    def train_step(self, x, data):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            total_loss = self.loss_function(self.model, predictions, data)
        # Compute gradients with respect to the model's trainable variables
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Apply gradients to update model parameters
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return total_loss
    
    
    def train(self, x_train, data):
        params = data['params']
        
        best_loss = float('inf')
        best_weights = None
        best_solution = None
        
        best_loss_list=[] #List of best losses 
        best_loss_epoch_list=[] #List when the best loss is changed
    
        for epoch in range(params['epochs']):
            total_loss = self.train_step(x_train, data)
            predictions = self.model(x_train, training=False)
            total_loss= self.loss_function(self.model, predictions, data)
    
            current_total_loss = total_loss.numpy()
            if current_total_loss < best_loss:
                best_loss = current_total_loss
                best_loss_list.append(best_loss)
                best_loss_epoch_list.append(epoch)
                best_weights = self.model.get_weights()
                best_solution = (predictions[0].numpy(),predictions[1].numpy())
                sol= np.sqrt(best_solution[0]**2 + best_solution[1]**2)
                self.model.save_weights(f"{params['log_dir']}/best_weights_ReLU_32.weights.h5")
                self.model.save(f"{params['log_dir']}/best_model.keras")
                
            if epoch % params['plot_interval'] == 0:
                print(f"Epoch {epoch}, Total Loss: {total_loss.numpy()}")
                
                print(f"  Best Loss so far: {best_loss}")
                Nx = params['Nx']
                Nt = params['Nt']
                x = params['x_edges']
                t = params['y_edges']
                                
                plt.figure()
                plt.imshow(np.reshape(sol, (Nt, Nx)).T, 
                           aspect='auto', 
                           extent=[t.min(), t.max(), x.min(), x.max()], 
                           cmap='YlGnBu')
                plt.colorbar()
                plt.xlabel("Time (t)")
                plt.ylabel("Space (x)")
                plt.title(f"Epoch {best_loss_epoch_list[-1]}")
                plt.savefig(f"{params['log_dir']}/solution{best_loss_epoch_list[-1]}.png", bbox_inches='tight')
                plt.show()
                plt.close()
                
        
        self.model.set_weights(best_weights)
        info = {
            'best_loss_list': best_loss_list,
            'best_loss_epoch_list': best_loss_epoch_list
        }
        return self.model, best_solution, info
