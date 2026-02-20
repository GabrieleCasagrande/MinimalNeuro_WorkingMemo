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
C = 200e-12           # Farads
g_L = 10e-6           # Siemens
E_L = -65e-3          # Volts
V_T = -55e-3          # Volts
tau_w = 20e-3         # Seconds
a = 1e-9              # Siemens
I_ext = 100e-12       # External current in Amps

# ----- Model Equations -----
def dVdt(V, w):
    return (g_L * (E_L - V) * (V_T - V) + w + I_ext) / C

def dwdt(V, w):
    return (a * (V - E_L) - w) / tau_w

# ----- Phase Plane Grid -----
V = np.linspace(-80e-3, -15e-3, 100)
w = np.linspace(-100e-12, 500e-12, 100)
V_grid, w_grid = np.meshgrid(V, w)

# Compute derivatives
dV = dVdt(V_grid, w_grid)
dw = dwdt(V_grid, w_grid)

# ----- Nullclines -----
V_nc = np.linspace(-80e-3, -15e-3, 1000)
w_nullcline = - (g_L * (E_L - V_nc) * (V_T - V_nc) + I_ext)   # dV/dt = 0
w_nc = a * (V_nc - E_L)                                       # dw/dt = 0

# ----- Fixed Points -----
def fixed_point_solver(guess):
    return fsolve(lambda V: a * (V - E_L) + (g_L * (E_L - V) * (V_T - V) + I_ext), guess)[0]

V_fp1 = fixed_point_solver(-65e-3)
V_fp2 = fixed_point_solver(-55e-3)
w_fp1 = a * (V_fp1 - E_L)
w_fp2 = a * (V_fp2 - E_L)

# ----- Plot Streamlines -----
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

plt.streamplot(V * 1e3, w * 1e12, dV, dw_scaled,
               color='grey',start_points=start_points, linewidth=1.2, density=1.0, arrowsize=1.)

# Nullclines
plt.plot(V_nc * 1e3, w_nullcline * 1e12, 'b-', linewidth=3, label='dV/dt = 0')
plt.plot(V_nc * 1e3, w_nc * 1e12, c='darkmagenta', linewidth=3, label='dw/dt = 0')

# Fixed points
plt.plot(V_fp1 * 1e3, w_fp1 * 1e12, 'ko', markersize=12, label='Fixed Point 1')
plt.plot(V_fp2 * 1e3, w_fp2 * 1e12, 'o',markersize=12, markeredgewidth=2.5, 
         markerfacecolor='white', markeredgecolor='black')
plt.vlines(x=-20, ymin=-100, ymax=500, lw=3, color='g', linestyle='dotted')
plt.vlines(x=-58, ymin=-100, ymax=500, lw=3, color='g', linestyle='dotted')
plt.ylim(-100, 500)
plt.xlim(-70, -15)
# Labels and style
plt.xlabel('$V$ (mV)')
plt.ylabel('$I_w$ (pA)')
#plt.title('Phase Portrait with Streamlines and Nullclines')
#plt.grid(True)
#plt.legend()
plt.tight_layout()


# Parameters
C = 200 * pF           # Capacitance
g_L = 10 * nS/mV       # Leak conductance
E_L = -65 * mV         # Resting potential
V_T = -55 * mV         # Threshold potential
Delta_T = 2 * mV       # Slope factor
tau_w = 20 * ms        # Adaptation time constant
a = 1 * nS             # Subthreshold adaptation
b = 60 * pA            # Spike-triggered adaptation
V_reset = -58 * mV     # Reset potential
I_ext = 100e-12 *amp 
duration= 400*ms

defaultclock.dt = 0.1*ms
# Model equations
eqs = '''
dVm/dt = (g_L * (E_L - Vm) * (V_T - Vm) + we + I_ext) / C : volt
dwe/dt = (a * (Vm - E_L) - we) / tau_w : amp
'''

# Define neuron group
G = NeuronGroup(1, eqs, threshold='Vm > -20*mV', reset='Vm = V_reset; we += b', method='rk4')

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

G.Vm = -10*mV  # Initial membrane potential
G.we = 100 * pA  # Initial adaptation current

# Monitor variables
M_spike = SpikeMonitor(G)
M_FR = PopulationRateMonitor(G)
M_var = StateMonitor(G, ['Vm', 'we'], record=True)
spikemon = SpikeMonitor(G)

# Run simulation
run(duration)
plt.plot(M_var.Vm[0][3000::]/mV, M_var.we[0][3000::]/pA, c='green', lw=2)

# Opiional: save the figure
#filename='PhasePlane_a='+str(a)+'_b='+str(b)+'.png'
#plt.savefig(filename, dpi=90, format='png')

plt.show()
