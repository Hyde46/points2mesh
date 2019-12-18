import os
import subprocess
import numpy as np


def save_pc_to_file(data, file_name, path_to_file):
    '''
    Saves pointcloud data as .xyz file to disk
    data should be in a Nx3 format
    @param path_to_file: where to save the data to
    '''
    np.savetxt(os.path.join(path_to_file, file_name), data, delimiter=" ")


def load_pc_meshlab(path_to_file):
    prg = 'meshlab'
    cmd = [prg, path_to_file]

    return_code = subprocess.check_output(cmd)
    print return_code


if __name__ == '__main__':
    data = np.random.randn(1, 3, 100)
    path_to_files = '/home/heid/Documents/master/utils/'
    file_name = 'test.xyz'
    save_pc_to_file(np.transpose(data[-1]), file_name, path_to_files)

    load_pc_meshlab(os.path.join(path_to_files, file_name))
