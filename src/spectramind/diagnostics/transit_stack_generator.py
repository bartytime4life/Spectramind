# transit_stack_generator.py – simulate + FFT (minimal stub)
import numpy as np

def generate_single_transit(T=1000, depth=0.001, duration=0.1, sigma=0.0005):
    t = np.linspace(0,1,T); f = np.ones(T)
    center = 0.5; in_tr = np.abs(t-center) < (duration/2)
    f[in_tr] -= depth
    return f + np.random.normal(0, sigma, size=T)
