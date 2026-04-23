import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def variable_current_sim(t, I_e, a, b, v0=-65, Iw0=0):
    """
   Simulate the eQIF neuron with a non-constant external current amplitudes (I_ext), and parameters a and b
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



def compute_instantaneous_freq(t, spike_times):
    """
    Compute the instantaneous firing frequency from spike times.

    Parameters:
    t : np.ndarray, Time vector (ms)
    spike_times : np.ndarray, Spike times (ms)

    Output:
    instantaneous_freq : np.ndarray, Instantaneous firing frequency (Hz)
    """

    if len(spike_times) > 1:
        isis = np.diff(spike_times)
        # Set 0 before first spike and after last
        rates_per_interval = np.concatenate(([0.0], 1.0 / np.where(isis > 0, isis, np.inf), [0.0]))
        bins = np.digitize(t, spike_times, right=False)  
        instantaneous_freq = rates_per_interval[bins]
    else:
        # No spikes, frequency remains zero
        instantaneous_freq = np.zeros(len(t)) 

    # Convert to Hz (time is in ms)
    return instantaneous_freq*1000 



def perturbation_current(t, Amp, tw, baseAmp=65, sigma_noise=0., random_seed=42):
    """
    Perturbation current for testing state transitions in the eQIF model.

    Parameters:
    t : np.ndarray, Time vector (ms)
    Amp : float, Perturbation amplitude
    tw : float, Perturbation width (ms)
    baseAmp : float, Baseline current
    sigma_noise : float, Standard deviation of noise
    random_seed : int, Seed for random number generator

    Output:
    current_array : np.ndarray, Perturbation current (nA) at each time point
    """
     
    # Define the time window for the perturbation
    p_start = 800
    p_end = p_start + tw
    mask_up = (t > p_start) & (t < p_end)

    rng = np.random.default_rng(seed=random_seed)
    white_noise = rng.normal(0, sigma_noise, len(t))

    # Base value for the current (this depends on the specific case you want to test)
    # NOTE: In this case the value was chose to be in the hysteresis region for the specific choice of parameters (reference to Fig2.b of the paper)
    current_array = np.ones(len(t))*baseAmp + white_noise
    current_array[mask_up] += Amp

    return current_array



def compute_state_matrices(t, amp_values, tw_values, Ibase, sigma=0.0, random_seed=42):
    """
    Compute state transition matrices for UP->DOWN and DOWN->UP perturbations.

    Parameters:
    t : np.ndarray, Time vector (ms)
    amp_values : np.ndarray, Perturbation amplitudes to sweep
    tw_values : np.ndarray, Perturbation widths to sweep (ms)
    Ibase : float, Baseline current
    sigma_noise : float, Standard deviation of noise
    random_seed : int, Seed for random number generator

    Output:
    state_matrix_UP_to_DOWN : np.ndarray, shape (len(amp_values), len(tw_values)), 1 = spiking state, 0 = silence state
    state_matrix_DOWN_to_UP : np.ndarray, shape (len(amp_values), len(tw_values)), 1 = spiking state, 0 = silence state
    shifts_UP_to_DOWN : list of (amp, tw), First entry where a DOWN transition occurred (state = 0)
    shifts_DOWN_to_UP : list of (amp, tw), First entry where an UP transition occurred (state = 1)
    """
    duration = t[-1]

    state_matrix_UP_to_DOWN = np.zeros((len(amp_values), len(tw_values)))
    state_matrix_DOWN_to_UP = np.zeros((len(amp_values), len(tw_values)))

    shifts_UP_to_DOWN = []
    shifts_DOWN_to_UP = []

    for i, amp in enumerate(amp_values):
        up_to_down_fisrt_shift_recorded = False
        down_to_up_fisrt_shift_recorded = False

        for j, tw in enumerate(tw_values):

            # --- UP to DOWN ---
            pert_UP = perturbation_current(t, Amp=-amp, tw=tw, baseAmp=Ibase, sigma_noise=sigma, random_seed=random_seed)
            _, _, spikes_UP = variable_current_sim(t, pert_UP, a=10, b=100, v0=-25, Iw0=100)
            freq_UP = compute_instantaneous_freq(t, spikes_UP)

            # Check the average frequency in the last 100 ms of the simulation to determine if it's in UP (spiking) or DOWN (silent) state
            avg_last_UP = np.mean(freq_UP[t > (duration - 100)])
            state_matrix_UP_to_DOWN[i, j] = 1 if avg_last_UP > 0 else 0

            # Shift = first time point where freq drops to 0 after being non-zero
            if avg_last_UP == 0 and not up_to_down_fisrt_shift_recorded:
                shifts_UP_to_DOWN.append((amp, tw))
                up_to_down_fisrt_shift_recorded = True

            # --- DOWN to UP ---
            pert_DOWN = perturbation_current(t, Amp=amp, tw=tw, baseAmp=Ibase, sigma_noise=sigma, random_seed=random_seed)
            _, _, spikes_DOWN = variable_current_sim(t, pert_DOWN, a=10, b=100, v0=-25, Iw0=-100)
            freq_DOWN = compute_instantaneous_freq(t, spikes_DOWN)

            # Check the average frequency in the last 100 ms of the simulation to determine if it's in UP (spiking) or DOWN (silent) state
            avg_last_DOWN = np.mean(freq_DOWN[t > (duration - 100)])
            state_matrix_DOWN_to_UP[i, j] = 1 if avg_last_DOWN > 0 else 0

            # Shift = first time point where freq becomes non-zero (onset of firing)
            if avg_last_DOWN > 0 and not down_to_up_fisrt_shift_recorded:
                shifts_DOWN_to_UP.append((amp, tw))
                down_to_up_fisrt_shift_recorded = True

    return state_matrix_UP_to_DOWN, state_matrix_DOWN_to_UP, shifts_UP_to_DOWN, shifts_DOWN_to_UP