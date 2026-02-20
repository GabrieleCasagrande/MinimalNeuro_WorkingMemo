import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def variable_current_sim(t, I_e, a, b, v0=-65, Iw0=0):
    """
    Compute the open-loop system for different external current amplitudes (I_ext), and parameters a and b
    """
    gL = 10 # leak conductance (nS/mV)
    eL = -62 # membrane potential resting state (mV)
    Cm = 200 # membrane capacitance (pF)
    Vth =  -55 # membrane potential threshold value (mV)
    v_reset =  -70 # after-spike reset value of the membrane potential (mV)
    v_peak = -20 # peak of the membrane potential(mV) 
    Trefr = 0. # refractory period (ms)
    tau_w = 20 # adaptation time constant (ms)

    last_spike = -np.inf 
    V, Iw = v0*np.ones(len(t), dtype=np.float64), Iw0*np.ones(len(t), dtype=np.float64)
    spikes = []
    dt = t[1]-t[0]
    
    for i in range(len(t)-1):

        # If currently inside refractory, keep V clamped to v_reset
        if (t[i] - last_spike) < Trefr:
            V[i+1] = v_reset
            continue 

        # Normal integration step (not refractory)
        V[i+1] = V[i] + dt*(1.0/Cm)*(gL*(eL - V[i]) * (Vth - V[i]) + Iw[i] + I_e[i] )
        Iw[i+1] = Iw[i] + dt*(a * (V[i] - eL) - Iw[i]) / tau_w

        if (V[i+1] > v_peak) and (t[i+1] - last_spike) > Trefr:
            last_spike = t[i+1]
            # set membrane to eL during the reset (refractoriness will hold it there for Trefr)
            V[i+1] = v_reset
            Iw[i+1] = Iw[i] + b
            spikes.append(t[i+1])

    return V, Iw, np.array(spikes)



def hysteresis_boundaries_Iext_vs_a(times, a_range, b_value, I_stim, v0=-65, Iw0=0):

    boundaries_start = []
    boundaries_stop = []

    for a_value in a_range:
        V_sim, Iw_sim, spikes_sim = variable_current_sim(times, I_stim, a_value, b_value, v0, Iw0)

        # Avoid transient period by considering only spikes after 2000 ms
        time_mask = (spikes_sim >= 2000) 
        masked_spikes = spikes_sim[time_mask]

        if len(masked_spikes) > 1:
            isis = np.diff(masked_spikes)  # Compute time differences between spikes
            instantaneous_freq = np.zeros(len(times))  # Initialize with zeros
            for i in range(1, len(masked_spikes)):
                mask = (times >= masked_spikes[i - 1]) & (times < masked_spikes[i])
                instantaneous_freq[mask] = 1 / isis[i - 1]  # Assign frequency
        else:
            instantaneous_freq = np.zeros(len(times))  # No spikes, frequency remains zero
        
        spike_times_mask = (instantaneous_freq > 0.)
        start, stop = I_stim[spike_times_mask][0], I_stim[spike_times_mask][-1]
        #print(start, stop)

        boundaries_start.append((start, a_value))
        boundaries_stop.append((stop, a_value))

    boundaries_start = np.array(boundaries_start)
    boundaries_stop = np.array(boundaries_stop)

    return boundaries_start, boundaries_stop



def compute_rate(spikes, t0=2000, t_end=1000):
    """
    Compute the firing rate from spike times
    """
    if len(spikes) == 0:
        #print(spikes)
        return 0

    # Consider time window from t0 to the end of the simulation in ms
    t_sim = t_end - t0
    time_mask = (spikes >= t0) & (spikes < t_end)
    spikes_sim = spikes[time_mask]
    # Return firing rate in Hz
    rate_Hz = len(spikes_sim) / (t_sim*1e-3)  

    return rate_Hz