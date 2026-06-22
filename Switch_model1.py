import numpy as np
from scipy.integrate import solve_ivp
from MalCoA_ODE1 import MalCoA_switch
from scipy import integrate

# from scipy.signal import find_peaks

# def oscillation_metric(t, y, transient_time=100):
#     ss = t >= transient_time
#     y_ss = y[ss]

#     amp = np.max(y_ss) - np.min(y_ss)
#     # rel_amp = amp / (np.mean(np.abs(y_ss)) + 1e-12)

#     peaks, _ = find_peaks(y_ss, prominence=0.05 * amp)

#     # first = y_late[:len(y_late)//2]
#     # second = y_late[len(y_late)//2:]

#     # amp_first = np.max(first) - np.min(first)
#     # amp_second = np.max(second) - np.min(second)

#     # persistent = amp_second > 0.5 * amp_first
#     oscillatory = len(peaks) >= 3 #and rel_amp > 0.05 #and persistent

#     return oscillatory #rel_amp 

def model(X_input):
    
    tspan = (0, 300)
    t_eval = np.arange(0, 300, 1)
    y0 = [1, 1, 1, 1, 0, 1, 1, 1, 1]

    n_samples = X_input.shape[0]

    Y = np.zeros((n_samples, 3))
    for i in range(n_samples):

        params = X_input[i, :]  

        sol = solve_ivp(
            MalCoA_switch,
            tspan,
            y0,
            args=(params,),
            t_eval=t_eval,
            method='BDF',
            rtol=1e-6,
            atol=1e-9
        )
        
        if not sol.success:
            raise RuntimeError(f"ODE solver failed at sample {i}")
        if not np.isfinite(sol.y).all():
            raise RuntimeError(f"ODE solver returned non-finite values at sample {i}")
        # score = oscillation_metric(sol.t, sol.y[4])
        # Y[i, 0] = score
        # Y[i,0] = np.mean(sol.y[4, 250:300])   # Final FA concentration
        # Y[i, 0] = np.max(sol.y[4, -50:]) - np.min(sol.y[4, -50:]) 
        # Y[i,0] = integrate.trapz(sol.y[4], sol.t)/integrate.trapz(sol.y[8], sol.t)  
        Y[i,0] = np.mean(sol.y[4, 0:10])
        Y[i,1] = np.mean(sol.y[4, 40:60])
        Y[i,2] = np.mean(sol.y[4, 250:300])


    return Y


