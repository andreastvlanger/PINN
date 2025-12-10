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

Created on Sun Dec  8 13:05:20 2024

@author: Andreas Langer
"""

import tensorflow as tf
import numpy as np


@tf.keras.utils.register_keras_serializable()
class NeuralNetwork(tf.keras.Model):
    def __init__(self, hidden_layers, activations, neurons_output_layer = 1, 
                 l2_reg=1e-4, max_value=100, 
                 use_kernel_constraint=True, use_bias_constraint=True,
                 skip=False, skip_every_n=2):
        '''
        Parameters
        ----------
        hidden_layers : list of integers
            List of numbers of neurons per layer.
        activations : list of strings
            Type of activation functions in each layer.
        l2_reg : float, optional
            Parameter regularizing the weights with l2. The default is 1e-4.
        max_value : float, optional
            Bounds the absolute value of the weights and biases. The default is 100.
        use_kernel_constraint : boolean, optional
            True if weights are bounded by max_value, False if not. The default is True.
        use_bias_constraint : boolean, optional
            True if biases are bounded by max_value, False if not. The default is True.
        skip : boolean, optional
            If True, then skip connections are used in the network. The default is False.
        skip_every_n : integer, optional
            Every skip_every_n there is a skip-connection. The default is 2.

        Returns
        -------
        Float. Output of the neural network.

        '''
        super(NeuralNetwork, self).__init__()
        self.hidden_layers_params = {
            "hidden_layers": hidden_layers,
            "activations": activations
        }
        self.hidden_layers = []
        self.neurons_output_layer = neurons_output_layer
        self.l2_reg = l2_reg
        self.max_value = max_value
        self.use_kernel_constraint = use_kernel_constraint
        self.use_bias_constraint = use_bias_constraint
        self.skip_every_n = skip_every_n
        self.skip = skip
        for neurons, activation in zip(hidden_layers, activations):
            kernel_constraint = tf.keras.constraints.MaxNorm(max_value=max_value) if use_kernel_constraint else None
            bias_constraint = tf.keras.constraints.MaxNorm(max_value=max_value) if use_bias_constraint else None
            
            self.hidden_layers.append(tf.keras.layers.Dense(
                neurons, 
                activation=activation, 
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                kernel_constraint=kernel_constraint, 
                bias_constraint=bias_constraint
            ))
        
        kernel_constraint = tf.keras.constraints.MaxNorm(max_value=max_value) if use_kernel_constraint else None
        bias_constraint = tf.keras.constraints.MaxNorm(max_value=max_value) if use_bias_constraint else None
        
        self.output_layer = tf.keras.layers.Dense(
            neurons_output_layer,
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1234),
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )

    
    def call(self, inputs, training=False):
        x = inputs  # Start with inputs
        for layer in self.hidden_layers:
            x = layer(x)  # Apply each hidden layer sequentially
        outputs = self.output_layer(x)  # Return the output
        return tf.split(outputs, num_or_size_splits=outputs.shape[1], axis=1)
    
    @staticmethod
    def xavier_init(size, dtype=None):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def get_config(self):
        return {
            **self.hidden_layers_params,
            "neurons_output_layer": self.neurons_output_layer,
            "l2_reg": self.l2_reg,
            "max_value": self.max_value,
            "use_kernel_constraint": self.use_kernel_constraint,
            "use_bias_constraint": self.use_bias_constraint,
            "skip": self.skip,
            "skip_every_n": self.skip_every_n,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
    
# Inherit from NeuralNetwork
@tf.keras.utils.register_keras_serializable()
class BoundaryConditionNN(NeuralNetwork):
    def __init__(self, hidden_layers, activations, boundary_mask, 
                 neurons_output_layer = 1,l2_reg=1e-4, max_value=100, 
                 use_kernel_constraint=True, use_bias_constraint=True,
                 skip=False, skip_every_n=2):
        # Initialize the parent NeuralNetwork
        super().__init__(
            hidden_layers, 
            activations, 
            neurons_output_layer,
            l2_reg, 
            max_value, 
            use_kernel_constraint, 
            use_bias_constraint, 
            skip, 
            skip_every_n
        )
        self.boundary_mask = boundary_mask

    # Override the call method to apply the boundary condition
    def call(self, inputs, training=False):
        x, y = inputs[:, 0], inputs[:, 1]  # Split the inputs into x and y
        
        # Call the parent class's call method to get the output without boundary conditions
        u_tilde, v_tilde = super().call(inputs, training=training)
        # Apply the boundary mask to enforce u = 0 on the boundary
        g = self.boundary_mask(x, y)
        g_u = tf.reshape(2 / tf.cosh(x),(-1,1))
        g = tf.reshape(y, (-1, 1))
        
        u = g_u + g * u_tilde
        v = g * v_tilde
        
        return [u, v]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "boundary_mask": self.boundary_mask  # Include any boundary_mask logic
        })
        return config

    @classmethod
    def from_config(cls, config):
        boundary_mask = config.pop("boundary_mask")
        # Recreate boundary_mask or pass it directly
        return cls(boundary_mask=boundary_mask, **config)