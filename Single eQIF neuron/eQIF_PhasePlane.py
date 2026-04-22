import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from brian2 import *

plt.rcParams.update({
    'font.size': 18,          # Controls default text size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 18,     # X/Y label font size
    'xtick.labelsize': 16,    # X tick labels
    'ytick.labelsize': 16,    # Y tick labels
    'legend.fontsize': 14,    # Legend font size
})

# ----- Parameters -----
parameters = {'C':200, 'gL':10, 'EL':-65, 'V_T':-55, 'V_reset':-58, 'tau_w':20, 'a':12, 'b':50, 'I_ext':100}

##############################################
# Compute the phase diagram and the nullclines
##############################################

C = parameters.get('C')*1e-12           # Farads
g_L = parameters.get('gL')*1e-6         # Siemens
E_L = parameters.get('EL')*1e-3         # Volts
V_T = parameters.get('V_T')*1e-3        # Volts
tau_w = parameters.get('tau_w')*1e-3    # Seconds
a = parameters.get('a')*1e-9            # Siemens
I_ext = parameters.get('I_ext')*1e-12   # External current in Amps

# Model Equations
def dVdt(V, w):
    return (g_L * (E_L - V) * (V_T - V) + w + I_ext) / C

def dwdt(V, w):
    return (a * (V - E_L) - w) / tau_w

# Phase Plane Grid 
V = np.linspace(-80e-3, -15e-3, 100)
w = np.linspace(-100e-12, 500e-12, 100)
V_grid, w_grid = np.meshgrid(V, w)

# Compute derivatives
dV = dVdt(V_grid, w_grid)
dw = dwdt(V_grid, w_grid)

# Nullclines
V_nc = np.linspace(-80e-3, -15e-3, 1000)
w_nullcline = - (g_L * (E_L - V_nc) * (V_T - V_nc) + I_ext)   # dV/dt = 0
w_nc = a * (V_nc - E_L)                                       # dw/dt = 0

# Fixed Points
def fixed_point_solver(guess):
    return fsolve(lambda V: a * (V - E_L) + (g_L * (E_L - V) * (V_T - V) + I_ext), guess)[0]

V_fp1 = fixed_point_solver(-65e-3)
V_fp2 = fixed_point_solver(-55e-3)
w_fp1 = a * (V_fp1 - E_L)
w_fp2 = a * (V_fp2 - E_L)

# Plot Streamlines
plt.figure(figsize=(8, 6))

# Streamplot for vector field
aspect_ratio = (V[-1] - V[0]) / (w[-1] - w[0])  # ratio of V-range to w-range
dw_scaled = dw * aspect_ratio

plt.streamplot(V * 1e3, w * 1e12, dV, dw_scaled,
               color='grey', linewidth=.20, density= 10., arrowsize=.1)

Vg = np.linspace(-80, -15, 40)
wg = np.linspace(-100, 500, 40)
Vgrid, wgrid = np.meshgrid(Vg, wg)

start_points = np.vstack([Vgrid.ravel(), wgrid.ravel()]).T

plt.streamplot(V * 1e3, w * 1e12, dV, dw_scaled, color='grey',start_points=start_points, linewidth=1.2, density=1.0, arrowsize=1.)

# Nullclines
plt.plot(V_nc * 1e3, w_nullcline * 1e12, 'b-', linewidth=3, label='dV/dt = 0')
plt.plot(V_nc * 1e3, w_nc * 1e12, c='darkmagenta', linewidth=3, label='dw/dt = 0')

# Fixed points
plt.plot(V_fp1 * 1e3, w_fp1 * 1e12, 'ko', markersize=12, label='Fixed Point 1')
plt.plot(V_fp2 * 1e3, w_fp2 * 1e12, 'o',markersize=12, markeredgewidth=2.5, markerfacecolor='white', markeredgecolor='black')
plt.vlines(x=-20, ymin=-100, ymax=500, lw=3, color='g', linestyle='dotted')
plt.vlines(x=-58, ymin=-100, ymax=500, lw=3, color='g', linestyle='dotted')
plt.ylim(-100, 500)
plt.xlim(-70, -15)
# Labels and style
plt.xlabel('$V$ (mV)')
plt.ylabel('$I_w$ (pA)')
#plt.grid(True)
#plt.legend()
plt.tight_layout()

###########################
# Compute two trajectories
###########################

# Parameters
C = parameters.get('C') * pF                # Capacitance
g_L = parameters.get('gL') * nS/mV          # Leak conductance
E_L = parameters.get('EL') * mV             # Resting potential
V_T = parameters.get('V_T') * mV            # Threshold potential
tau_w = parameters.get('tau_w') * ms        # Adaptation time constant
a = parameters.get('a') * nS                # Subthreshold adaptation
b = parameters.get('b') * pA            # Spike-triggered adaptation
V_reset = parameters.get('V_reset') * mV     # Reset potential
I_ext = parameters.get('I_ext')*1e-12 *amp  # External current

duration= 400*ms
defaultclock.dt = 0.1*ms

# Model equations
eqs = '''
dVm/dt = (g_L * (E_L - Vm) * (V_T - Vm) + we + I_ext) / C : volt
dwe/dt = (a * (Vm - E_L) - we) / tau_w : amp
'''

# Define neuron group
G = NeuronGroup(1, eqs, threshold='Vm > -20*mV', reset='Vm = V_reset; we += b', method='rk4')

# Define initial condition of the first trajectory
G.Vm = -69*mV  # Initial membrane potential
G.we = 100 * pA  # Initial adaptation current

# Monitor variables
M_spike = SpikeMonitor(G)
M_FR = PopulationRateMonitor(G)
M_var = StateMonitor(G, ['Vm', 'we'], record=True)
spikemon = SpikeMonitor(G)

# Run simulation
run(duration)
plt.plot(M_var.Vm[0]/mV, M_var.we[0]/pA, c='green', lw=2)

# Define initial condition of the second trajectory
G.Vm = -40*mV  # Initial membrane potential
G.we = 200 * pA  # Initial adaptation current

# Monitor variables
M_spike = SpikeMonitor(G)
M_FR = PopulationRateMonitor(G)
M_var = StateMonitor(G, ['Vm', 'we'], record=True)
spikemon = SpikeMonitor(G)

# Run simulation
run(duration)
plt.plot(M_var.Vm[0][3000::]/mV, M_var.we[0][3000::]/pA, c='green', lw=2)
plt.title(f'a = {a}, b = {b}')

# Optional: save the figure
#filename='PhasePlane_a='+str(a)+'_b='+str(b)+'.png'
#plt.savefig(filename, dpi=120, format='png')

plt.show()