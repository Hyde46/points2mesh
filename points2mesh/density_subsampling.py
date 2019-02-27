import tensorflow as tf

def density_subsample(data,
        num_points,
        global_density=False):

    # location 
    # [B, Dp, N]

    num_elements = location.shape.as_list()[2] 
    idxs = list(range(num_elements))
    
    cum_density = None

    ret = []


    
