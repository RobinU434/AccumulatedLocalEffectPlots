import numpy as np

def poprow(my_array,pr):
    """ row popping in numpy arrays
    Input: my_array - NumPy array, pr: row index to pop out
    Output: [new_array,popped_row] """
    i = pr
    pop = my_array[i]
    new_array = np.vstack((my_array[:i],my_array[i+1:]))
    return [new_array,pop]

def poprows(my_array, index_list):
    res_array = my_array
    
    for index in index_list[::-1]:
        res_array = poprow(res_array, index)
        
    return res_array

def popcol(my_array,pc):
    """ column popping in numpy arrays
    Input: my_array: NumPy array, pc: column index to pop out
    Output: [new_array,popped_col] """
    i = pc
    pop = my_array[:,i]
    new_array = np.hstack((my_array[:,:i],my_array[:,i+1:]))
    return [new_array,pop]

def popcols(my_array, index_list):
    res_array = my_array
    
    for index in index_list[::-1]:
        res_array = popcol(res_array, index)
        
    return res_array
