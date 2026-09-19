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

@author: andreas langer
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
            #laplacian = compute_laplacian(model, x)
            total_loss = self.loss_function(self.model, predictions, data)
        # Compute gradients with respect to the model's trainable variables
        gradients = tape.gradient(total_loss, self.model.trainable_variables)

        # Check for None gradients
        if any(g is None for g in gradients):
            raise ValueError("Some gradients are None. Check the computational graph.")

        # Apply gradients to update model parameters
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return total_loss
        
    
    def train(self, x_train, data):
        params = data['params']
        #trainer = Trainer(model, optimizer, params, A)  # Initialize Trainer with params
        best_loss = float('inf')
        best_weights = None
        best_solution = None
        
        best_loss_list=[] #List of best losses 
        best_loss_epoch_list=[] #List when the best loss is changed
    
        for epoch in range(params['epochs']):
            total_loss = self.train_step(x_train, data)
            predictions = self.model(x_train, training=False)
            #laplacian = compute_laplacian(model, x_train)
            total_loss= self.loss_function(self.model, predictions, data)
            
            current_total_loss = total_loss.numpy()
            if current_total_loss < best_loss:
                best_loss = current_total_loss
                best_loss_list.append(best_loss)
                best_loss_epoch_list.append(epoch)
                best_weights = self.model.get_weights()
                best_solution = predictions.numpy()
                self.model.save_weights(f"{params['log_dir']}/best_weights_ReLU_32.weights.h5")
                self.model.save(f"{params['log_dir']}/best_model.keras")
                
            if epoch % params['plot_interval'] == 0:
                print(f"Epoch {epoch}, Total Loss: {total_loss.numpy()}")
                
                print(f"  Best Loss so far: {best_loss}")
                Nx = params['Nx']
                X = params['X']
                Y = params['Y']
                
                # Plot the heatmap   
                plt.figure()
                plt.pcolormesh(X, Y, np.reshape(best_solution,(Nx, Nx)), shading='auto', cmap='rainbow')
                plt.colorbar()
                plt.title(f"Epoch {best_loss_epoch_list[-1]}")
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig(f"{params['log_dir']}/solution{best_loss_epoch_list[-1]}.png", bbox_inches='tight')
                plt.show()
                plt.close()
    
        
        self.model.set_weights(best_weights)
        data = {
            'best_loss_list': best_loss_list,
            'best_loss_epoch_list': best_loss_epoch_list
        }
        return self.model, best_solution, data
