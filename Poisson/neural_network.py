#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:05:20 2024

@author: andreas langer
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class NeuralNetwork(tf.keras.Model):
    def __init__(self, hidden_layers, activations, neurons_output_layer = 1, l2_reg=1e-4, max_value=100, 
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
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                kernel_constraint=kernel_constraint, 
                bias_constraint=bias_constraint
            ))
        
        kernel_constraint = tf.keras.constraints.MaxNorm(max_value=max_value) if use_kernel_constraint else None
        bias_constraint = tf.keras.constraints.MaxNorm(max_value=max_value) if use_bias_constraint else None
        
        self.output_layer = tf.keras.layers.Dense(
            neurons_output_layer,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )

    def call(self, inputs, training=False):
        
        x = inputs 
        skip_input = x
        hidden_layer_count = 0
        for layer in self.hidden_layers:
            x = layer(x)
            hidden_layer_count += 1
            if self.skip and hidden_layer_count %self.skip_every_n == 0:
                x +=skip_input
                skip_input = x
        return self.output_layer(x)
    
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
        u_tilde = super().call(inputs, training=training)
        # Apply the boundary mask to enforce u = 0 on the boundary
        g = self.boundary_mask(x, y)
        
        g = tf.reshape(g, (-1, 1))
        u = g * u_tilde  # Final output that satisfies the boundary condition
        
        return u
    
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