from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

dtime = 0.01 * ms
duration = 1500 * ms
start_scope()
defaultclock.dt = dtime

# Parameters
C = 0.01 * second         # Cste
tau_w = 0.01 * second       # Adaptation time constant
b = 4            # Spike-triggered adaptation
V_reset = -65      # Reset potential

# Time-varying external current
AmpStep = 20
BaseI = 20
Pert = 10
time_steps = int(duration / dtime)  # Number of time steps
current_array = np.full(time_steps, BaseI)  # Default 90 pA
current_array[int(50*ms/dtime):int(200*ms/dtime)] = BaseI+AmpStep  # 100-600 ms: 130 pA
current_array[int(650*ms/dtime):int(720*ms/dtime)] = BaseI-Pert
current_array[int(780*ms/dtime):int(850*ms/dtime)] = BaseI+Pert  # 100-600 ms: 130 pA
current_array[int(1000*ms/dtime):int(1150*ms/dtime)] = BaseI-AmpStep
current_array[int(1250*ms/dtime):int(1320*ms/dtime)] = BaseI+Pert  # 100-600 ms: 130 pA
current_array[int(1350*ms/dtime):int(1420*ms/dtime)] = BaseI-Pert
I_t = TimedArray(current_array , dt=dtime)  # Ensure units of pA

# Model equations
eqs = '''
dV/dt = (0.04*V**2 + 5*V + 140 + w + I_ext)/C : 1
dw/dt = (0.1 * (0.2*V - w))/tau_w : 1
I_ext = I_t(t) : 1
'''

# Define neuron group
G = NeuronGroup(1, eqs, threshold='V > 30', reset='V = V_reset; w += b', method='rk4')
G.V = -65  # Initial membrane potential
G.w = 0  # Initial adaptation current

# Monitor variables
M_spike = SpikeMonitor(G)
M_FR = PopulationRateMonitor(G)
M_var = StateMonitor(G, ['V', 'w'], record=True)

# Run simulation
run(duration)

RasG_exc = array([M_spike.t/ms, M_spike.i])

plt.rcParams.update({
    'font.size': 18,          # Controls default text size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 18,     # X/Y label font size
    'xtick.labelsize': 16,    # X tick labels
    'ytick.labelsize': 16,    # Y tick labels
    'legend.fontsize': 14,    # Legend font size
})
plt.figure(figsize=(10, 6))

plt.subplot(311)
ax = gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(np.arange(len(current_array)) * dtime / ms, current_array, color='purple')
#xlabel('Time (ms)')
plt.ylabel('$I_{ext}$ (pA)')
plt.xticks([]) 

Vvar = M_var.V[0]

# Plot membrane potential
plt.subplot(312)
ax = gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(M_var.t/ms, Vvar)
plt.ylabel('$V$ (mV)')
plt.xticks([]) 

# Compute interspike intervals (ISI) and instantaneous frequency
spike_times = M_spike.t

if len(spike_times) > 1:
    isis = np.diff(spike_times)  # Compute time differences between spikes
    instantaneous_freq = np.zeros(len(M_var.t))  # Initialize with zeros
    time_bins = M_var.t  # Use full time range
    for i in range(1, len(spike_times)):
        mask = (M_var.t >= spike_times[i - 1]) & (M_var.t < spike_times[i])
        instantaneous_freq[mask] = 1 / isis[i - 1]  # Assign frequency
else:
    instantaneous_freq = np.zeros(len(M_var.t))  # No spikes, frequency remains zero
    time_bins = M_var.t

plt.subplot(313)
ax = plt.gca()
ax2 = plt.twinx()

# Plot on the twin axis (if desired)
ax.plot(M_var.t/ms, M_var.w[0])
ax.set_ylabel('$I_w$ (pA)')

# Set xlabel only on the original axis (or on ax2 if preferred)
ax.set_xlabel('Time (ms)')

# Customize spines: hide top on both axes
for spine in ['top']:
    ax.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)


# Optional: make sure ticks appear only where you want them
ax.yaxis.set_ticks_position('left')
ax2.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('bottom')

plt.plot(time_bins / ms, instantaneous_freq / Hz, color='k')

plt.ylabel('$f$ (Hz)')


plt.tight_layout()
#plt.savefig('Figure_14.svg', dpi=300, format='svg')
plt.show()