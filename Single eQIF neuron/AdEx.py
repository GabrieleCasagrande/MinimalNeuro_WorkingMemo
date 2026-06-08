from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 18,          # Controls default text size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 18,     # X/Y label font size
    'xtick.labelsize': 16,    # X tick labels
    'ytick.labelsize': 16,    # Y tick labels
    'legend.fontsize': 14,    # Legend font size
})

# Define time parameters for the simulation
dtime = 0.01 * ms
duration = 1500 * ms
start_scope()
defaultclock.dt = dtime

# Parameters
C = 180 * pF           # Capacitance
g_L = 10 * nS      # Leak conductance
E_L = -65 * mV         # Resting potential
V_T = -55 * mV         # Threshold sodium activation
D_T = 2 * mV            # Slope factor
tau_w = 20 * ms        # Adaptation time constant
a = 2.5 * nS             # Subthreshold adaptation
b = 60 * pA            # Spike-triggered adaptation
V_reset = -58 * mV     # Reset potential

# Define the time-varying external current
AmpStep = 50
BaseI = 50
Pert = 15
time_steps = int(duration / dtime)  # Number of time steps
current_array = np.full(time_steps, BaseI)  # Default 90 pA
current_array[int(50*ms/dtime):int(200*ms/dtime)] = BaseI+AmpStep  # 100-600 ms: 130 pA
current_array[int(650*ms/dtime):int(720*ms/dtime)] = BaseI-Pert
current_array[int(780*ms/dtime):int(850*ms/dtime)] = BaseI+Pert  # 100-600 ms: 130 pA
current_array[int(1000*ms/dtime):int(1150*ms/dtime)] = BaseI-AmpStep
current_array[int(1250*ms/dtime):int(1320*ms/dtime)] = BaseI+Pert  # 100-600 ms: 130 pA
current_array[int(1350*ms/dtime):int(1420*ms/dtime)] = BaseI-Pert
I_t = TimedArray(current_array * pA, dt=dtime)  # Ensure units of pA

# Model equations
eqs = '''
dV/dt = (g_L * (E_L - V) + g_L * D_T * exp((V - V_T)/D_T) + w + I_ext) / C : volt
dw/dt = (a * (V - E_L) - w) / tau_w : amp
I_ext = I_t(t) : amp
''' 

# Define neuron group
G = NeuronGroup(1, eqs, threshold='V > 0*mV', reset='V = V_reset; w += b', method='euler')
G.V = E_L  # Initial membrane potential
G.w = 0 * pA  # Initial adaptation current

# Monitor variables
M_spike = SpikeMonitor(G)
M_FR = PopulationRateMonitor(G)
M_var = StateMonitor(G, ['V', 'w'], record=True)

# Run simulation
run(duration)

plt.figure(figsize=(12, 6))

# Plot input current
plt.subplot(311)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(np.arange(len(current_array)) * dtime / ms, current_array, color='purple')
plt.ylabel('$I_{ext}$ (pA)')
plt.xticks([]) 

# Plot membrane potential
Vvar = M_var.V[0]/mV
# Set values above -40 mV to 0 for better visualization of spikes
Vm=[0 if Vvar[i] > -40 else Vvar[i] for i in range(len(Vvar))]

plt.subplot(312)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(M_var.t/ms, Vm)
plt.ylabel('$V$ (mV)')
plt.xticks([]) 

# Compute interspike intervals (ISI) and instantaneous frequency
RasG_exc = array([M_spike.t/ms, M_spike.i])
spike_times = M_spike.t

if len(spike_times) > 1:
    isis = np.diff(spike_times)  # Compute time differences between spikes
    instantaneous_freq = np.zeros(len(M_var.t))  # Initialize with zeros
    for i in range(1, len(spike_times)):
        mask = (M_var.t >= spike_times[i - 1]) & (M_var.t < spike_times[i])
        instantaneous_freq[mask] = 1 / isis[i - 1]  # Assign frequency
else:
    instantaneous_freq = np.zeros(len(M_var.t))  # No spikes, frequency remains zero

plt.subplot(313)
ax = plt.gca()
ax2 = plt.twinx()

# Plot self-excitatory current 
ax.plot(M_var.t/ms, M_var.w[0]/pA)
ax.set_ylabel('$I_w$ (pA)')
ax.set_xlabel('Time (ms)')

for spine in ['top']:
    ax.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax2.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('bottom')

# Plot instantaneous frequency
plt.plot(M_var.t/ms, instantaneous_freq / Hz, color='k')
plt.ylabel('$f$ (Hz)')

plt.tight_layout()
plt.savefig('Figure_13.svg', dpi=300, format='svg')
plt.show()