import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
from scipy.signal import find_peaks

plt.rcParams.update({
    'font.size': 18,          # Controls default text size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 18,     # X/Y label font size
    'xtick.labelsize': 16,    # X tick labels
    'ytick.labelsize': 16,    # Y tick labels
    'legend.fontsize': 14,    # Legend font size
})
# Constants (unitless, from Brian2 parameters)
g_L = 10.0     # nS/mV
E_L = -65.0    # mV
V_T = -55.0    # mV
a = 4.0        # nS
V_reset = -58 
V_spike = -20

# Function to compute roots for a given I_ext
def compute_roots(I_ext):
    A = -g_L
    B = g_L * (E_L + V_T) + a
    C = - (g_L * E_L * V_T + I_ext + a * E_L)

    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        return np.nan, np.nan  # No real roots

    root1 = (-B + np.sqrt(discriminant)) / (2 * A)
    root2 = (-B - np.sqrt(discriminant)) / (2 * A)
    return root1, root2

# Range of I_ext in pA (or arbitrary units since we're plotting)
I_ext_values = np.linspace(0.0, 600.0, 600)
roots1 = []
roots2 = []

# Compute roots
for I_ext in I_ext_values:
    r1, r2 = compute_roots(I_ext)
    roots1.append(r1)
    roots2.append(r2)

def V_NC(V):
    return - (g_L * (E_L - V) * (V_T - V) + I_ext) 
def w_NC(V):
    return a * (V - E_L) 

wr1 = [w_NC(i) for i in roots1]
wr2 = [w_NC(i) for i in roots2]

current_array, VarV, Varw = np.load('DataIVw2.npy')

VarV[VarV > -30] = -20
# Create a figure with 3 subplots (2 columns), with the first column spanning the 3D plot
fig = plt.figure(figsize=(16, 8))
#gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])  # Left column = 1, Right column = 2
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.8, 1.2],
    wspace=0.45,   # horizontal spacing between columns
    hspace=0.25    # vertical spacing between rows
)
# 3D plot (left column, first row)
ax3d = fig.add_subplot(gs[:, 0], projection='3d')

# Plot root 1
ax3d.plot(I_ext_values, roots1, wr1, lw=3.5, color='darkslategrey', label='Root 1')
# Plot root 2
ax3d.plot(I_ext_values, roots2, wr2, lw=3.5, color='darkslategrey', linestyle='--', label='Root 2')

ax3d.plot(current_array, VarV, Varw, color='blue', linestyle='-', lw=.5)

ax3d.set_xlabel('$I_{ext}$ (pA)', labelpad=10)
ax3d.set_ylabel('$V$ (mV)', labelpad=10)
ax3d.set_zlabel('$I_w$ (pA)', labelpad=10)

ax3d.text2D(0.55, .15, 'SN', transform=ax3d.transAxes,
            ha='center', va='bottom', fontsize=14)
#ax3d.legend()

# 2D plot 1 (right column, first row)
ax2d1 = fig.add_subplot(gs[0, 1])
ax2d1.plot(I_ext_values, wr1,lw=3.5, c='darkslategrey', label='Root 1')
ax2d1.plot(I_ext_values, wr2, lw=3.5, ls='--', c='darkslategrey', label='Root 2')
ax2d1.plot(current_array, Varw, color='blue', linestyle='-', lw=.5)
ax2d1.set_xlabel('$I_{ext}$ (pA)')
ax2d1.set_ylabel('$I_w$ (pA)')
ax2d1.text(0.8, .02, 'SN', transform=ax2d1.transAxes,
           ha='center', va='bottom', fontsize=14)
#ax2d1.legend()

# 2D plot 2 (right column, second row)
ax2d2 = fig.add_subplot(gs[1, 1])
ax2d2.plot(I_ext_values, roots1, lw=3.5, c='darkslategrey', label='Root 1')
ax2d2.plot(I_ext_values, roots2, lw=3.5, ls='--', c='darkslategrey', label='Root 2')
ax2d2.plot(current_array, VarV, color='blue', linestyle='-', lw=.5)
ax2d2.axhline(y=V_reset, lw=2, color='g', linestyle='dotted')
ax2d2.axhline(y=V_spike, lw=2, color='g', linestyle='dotted')
ax2d2.set_xlabel('$I_{ext}$ (pA)')
ax2d2.set_ylabel('$V$ (mV)')
ax2d2.text(0.8, .035, 'SN', transform=ax2d2.transAxes,
           ha='center', va='bottom', fontsize=14)
#ax2d2.legend()

# Adjust layout
plt.tight_layout()
plt.show()
