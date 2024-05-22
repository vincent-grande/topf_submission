import numpy as np

def thick_sphere(density,r_inner,r_outer):
    num_samples=int(np.floor(density*4*r_outer**2))
    samples=2*r_outer*np.random.rand(num_samples,2)-r_outer
    samples_refined=[]
    for point in samples:
        cur_r = np.linalg.norm(point)
        if cur_r>r_inner and cur_r<r_outer:
            samples_refined.append(point)
    return np.array(samples_refined)

def filled_rectangle(density,x,y):
    num_samples=int(np.floor(density*x*y))
    samples = [x,y]*np.random.rand(num_samples, 2)
    return np.array(samples)
