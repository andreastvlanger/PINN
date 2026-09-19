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

@author: andreas langer
"""

import os
import pickle

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
params = {
    'Problem': 'PoissonRec',
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
    return boundary_mask

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

    # Mask the boundary
    mask = MaskBoundary(x, y)
    rhs[mask] = 0  
    
    # Flatten the right-hand side and solution arrays
    b = rhs.ravel()
    b[mask.ravel()] = 0
    
    return b


def poisson_exact(x, y, N=51):
    """
    Exact solution of
        -Δu = 1  in [-1,1]^2
        u = 0    on boundary
    using a truncated sine series.

    Parameters
    ----------
    x, y : floats
        Evaluation point in [-1,1]
    N : int
        Maximum odd mode (use odd numbers only)

    Returns
    -------
    u : float
        Approximate solution u(x,y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Nx = x.size
    Ny = y.size
 
    # Build grid (Nx, Ny)
    X = x[:, None]          # (Nx, 1)
    Y = y[None, :]          # (1, Ny)
 
    pi = np.pi
    U = np.zeros((Nx, Ny), dtype=float)
 
    for k in range(1, N + 1, 2):  # odd k
        sx = np.sin(k * pi * (X + 1.0) / 2.0)  # (Nx, 1) broadcasts over y
        for l in range(1, N + 1, 2):      # odd l
            coeff = 64.0 / (pi**4 * k * l * (k**2 + l**2))
            U += coeff * sx * np.sin(l * pi * (Y + 1.0) / 2.0)  # (1, Ny)
 
    return U

    
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
  
    u_exact = poisson_exact(x, y, N=501)  
    print(f"Error: {np.linalg.norm(u2D-u_exact)}")
    
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
    
    # Plot the heatmap
    plt.pcolormesh(X, Y, u_exact, shading='auto', cmap='rainbow')
    plt.colorbar()#label='u')
    #plt.xlabel('X-axis')
    #plt.ylabel('Y-axis')
    #plt.title('Solution u_exact')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"{params['log_dir']}/theoretic_solution.png", bbox_inches='tight')
    plt.show()
    
    # Plot the heatmap
    plt.pcolormesh(X, Y, u_exact-u2D, shading='auto', cmap='rainbow')
    plt.colorbar()#label='u')
    #plt.xlabel('X-axis')
    #plt.ylabel('Y-axis')
    #plt.title('Difference')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"{params['log_dir']}/error.png", bbox_inches='tight')
    plt.show()
    
    print(u2D.max())
    max_index = np.unravel_index(np.argmax(u2D), u2D.shape)
    print(max_index, u2D[max_index])
    max_index = np.unravel_index(np.argmax(u), u.shape)
    print(max_index, u[max_index])
    
    save_parameters(log_dir, params)
    save_essential_data(log_dir=log_dir,params=params,u2D=u2D) 
    
    Nx_hd = 101
    x_hd = np.linspace(-1, 1, Nx_hd)
    y_hd = np.linspace(-1, 1, Nx_hd)
    u_exact_hd = poisson_exact(x_hd, y_hd, N=501)  
    #print(u2D.shape, u_exact.shape)
    print(f"Error: {np.linalg.norm(u2D-u_exact)}")
    
    X_hd, Y_hd = np.meshgrid(x_hd, y_hd)
    # Plot the heatmap
    plt.pcolormesh(X_hd, Y_hd, u_exact_hd, shading='auto', cmap='rainbow')
    plt.colorbar()#label='u')
    #plt.xlabel('X-axis')
    #plt.ylabel('Y-axis')
    #plt.title('Solution u in 2D')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"{params['log_dir']}/theoretic_solution_hd.png", bbox_inches='tight')
    plt.show()
