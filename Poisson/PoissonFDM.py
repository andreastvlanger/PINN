"""
============================================================================
Portions of this file (save_essential_data, save_parameters) are from project
https://github.com/andreastvlanger/DeepTV (GNU GENERAL PUBLIC LICENSE Version 3)

Copyright (C) 2024  Andreas Langer, Sara Behnamian

Rest of the code

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

Created on Mon Nov  4 10:41:49 2024

@author: andreas langer
"""

import os
import pickle

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
params = {
    'Problem': 'PoissonSingularity',
    'DiscretizationModel': 'FDM',
    'log_dir': 'logs'
}

def save_essential_data(log_dir, **kwargs):
    file_path = os.path.join(log_dir, 'essential_data.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(kwargs, f)
    print(f"Saved essential data to {file_path}")
    print(f"Saved variables: {', '.join(kwargs.keys())}")

def save_parameters(log_dir, params):
    # Save parameters to .txt file
    param_file_txt = os.path.join(log_dir, 'parameters.txt')
    with open(param_file_txt, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved parameters to {param_file_txt}")

    # Save parameters to .tex file
    param_file_tex = os.path.join(log_dir, 'parameters.tex')
    with open(param_file_tex, 'w') as f:
        f.write("% Auto-generated parameters file\n")
        f.write("\\providecommand{\\Data}[1]{\n")
        f.write("    \\csname Data/#1\\endcsname\n")
        f.write("}\n\n")
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                value_str = ', '.join(map(str, value))
                f.write(f"\\expandafter\\def\\csname Data/\\DataPrefix/{key}\\endcsname{{\\pgfmathprintnumber{{{value_str}}}}}\n")
            else:
                f.write(f"\\expandafter\\def\\csname Data/\\DataPrefix/{key}\\endcsname{{\\pgfmathprintnumber{{{value}}}}}\n")
    print(f"Saved parameters to {param_file_tex}")

    # Save parameters to .pkl file
    param_file_pkl = os.path.join(log_dir, 'parameters.pkl')
    with open(param_file_pkl, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved parameters to {param_file_pkl}")
    
    
def MaskBoundary(x,y):
    boundary_mask = (x[None,:] == -1) | (x[None,:] == 1) | (y[:,None] == -1) | (y[:,None] == 1)
    cut_region_mask = (x[None,:] >= 0) & (x[None,:] < 1) & (y[:,None] == 0)
    combined_mask = boundary_mask | cut_region_mask
    return combined_mask

def SysMatrix(Nx,Ny,hx,hy,x,y):
    # Step 1: Create main diagonal matrix with 4s inside, 1s on boundary and cut region
    main_diag_matrix = 4 * np.ones((Nx, Ny))
    combined_mask = MaskBoundary(x, y)
    main_diag_matrix[combined_mask] = 1

    # Step 2: Create off-diagonal matrices with -1s inside, 0s on boundary and cut region
    off_diag_matrix = -1 * np.ones((Nx, Ny))
    off_diag_matrix[combined_mask] = 0
    # Flatten matrices to create diagonals for `diags`
    main_diag = main_diag_matrix.ravel()
    left_diag = off_diag_matrix.ravel()[1:]  # Shifted left by 1 (left diagonal)
    right_diag = off_diag_matrix.ravel()[:-1]  # Shifted right by 1 (right diagonal)
    up_diag = off_diag_matrix.ravel()[Nx:]  # Shifted up by N (up diagonal)
    down_diag = off_diag_matrix.ravel()[:-Nx]  # Shifted down by N (down diagonal)

    # Step 3: Assemble sparse matrix using `diags`
    A = sp.diags([main_diag, left_diag, right_diag, up_diag, down_diag],
                 [0, -1, 1, -Nx, Nx], shape=(Nx * Ny, Nx * Ny), format="csr")
    return 1/hx*1/hy*A

def RHS(Nx,Ny,x,y):
    # Initialize the right-hand side (source term) and the solution matrix
    rhs = np.ones((Nx, Ny))  # f(x,y) = 1 everywhere in Omega

    # Mask out the region [0,1) x {0}
    mask = MaskBoundary(x, y)
    rhs[mask] = 0  # Zero out RHS on the cut region
    
    # Flatten the right-hand side and solution arrays
    b = rhs.ravel()
    b[mask.ravel()] = 0  # Set RHS on the cut region to zero for Dirichlet condition
    
    return b
    
if __name__ == "__main__" :
    
    # Set grid parameters
     
    # Number of grid points in each direction
    Nx=41
    Ny=Nx
    params['Nx'] = Nx
    params['Ny'] = Ny
    
    hx = 2 / (Nx - 1)  # Grid spacing
    hy = 2 / (Ny - 1)
    params['hx'] = hx
    params['hy'] = hy
    
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)

    print(x.shape)
    A = SysMatrix(Nx, Ny, hx, hy, x, y)
    b = RHS(Nx,Ny,x,y)
    
    u=sp.linalg.spsolve(A,b)
    
    print(f"residual: {hx * hy * tf.reduce_sum(tf.square(A@u - b))}")
    u2D=np.reshape(u, (Nx,Ny))
    
    params['log_dir'] = params['Problem']+'/'+params['DiscretizationModel']
    log_dir = params['log_dir']
    print(f"Creating folder at: {os.path.abspath(log_dir)}")
    os.makedirs(log_dir, exist_ok=True)
  
    X, Y = np.meshgrid(x, y)
    
    # Plot the heatmap
    plt.pcolormesh(X, Y, u2D, shading='auto', cmap='rainbow')
    plt.colorbar()#label='u')
    #plt.xlabel('X-axis')
    #plt.ylabel('Y-axis')
    #plt.title('Solution u in 2D')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"{params['log_dir']}/solution.png", bbox_inches='tight')
    plt.show()
    
    print(u2D.max())
    max_index = np.unravel_index(np.argmax(u2D), u2D.shape)
    print(max_index, u2D[max_index])
    max_index = np.unravel_index(np.argmax(u), u.shape)
    print(max_index, u[max_index])
    
    save_parameters(log_dir, params)
    save_essential_data(log_dir=log_dir,params=params,u2D=u2D) 