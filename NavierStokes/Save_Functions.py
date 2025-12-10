"""
============================================================================
Copyright (C) 2024  Andreas Langer, Sara Behnamian

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

Functions from project https://github.com/andreastvlanger/DeepTV

GNU GENERAL PUBLIC LICENSE Version 3

Created on Sat Jan 11 15:01:20 2025


"""
import os
import pickle


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
    