import matplotlib.pyplot as plt
import numpy as np

def setup_plot_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })

setup_plot_style()

fig, ax = plt.subplots()

eps0 = 8.854e-12
ell = 2e-6
l = 100e-6
m = 1e-6
k = 1
A = l * ell
V0 = 3

a = np.linspace(0, 8, 300)
d0 = np.linspace(1e-6, 4e-6, 300)

a_grid, D0_grid = np.meshgrid(a, d0)

V = (V0 * m**2 / k**2) * (a_grid / D0_grid**2)

hm = ax.contourf(a_grid, D0_grid, V, levels=50, cmap="plasma")
cbar = fig.colorbar(hm, ax=ax)
cbar.set_label(r"Différence de potentiel $\Delta V$ [V]")

ax.set_xlabel(r"Accélération $a$ [m/s$^2$]")
ax.set_ylabel(r"Distance initiale entre les doigts $d_0$ [m]")
#ax.set_title(r"Carte thermique de $\Delta V(a,d_0)$")

fig.tight_layout()
plt.show()