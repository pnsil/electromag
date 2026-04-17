import matplotlib.pyplot as plt
import numpy as np

def setup_plot_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
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

# Données
eps0 = 8.854e-12
ell = 2e-6
l = 100e-6
m = 1e-6
k = 1
A = l * ell
V0 = 3
N = 100

d0 = np.linspace(1e-6, 4e-6, 300)

ke = (2 * eps0 * N * A * V0**2)/d0**3

# Courbe
ax.plot(d0, ke)

# Titres et axes
ax.set_xlabel(r"Distance initiale entre les doigts $d_0$ [m]")
ax.set_ylabel(r"Constante de rappel électrique $k_e$ [N/m]")
#ax.set_title(r"Graphique de base")

# Limites optionnelles
# ax.set_xlim(0, 10)
# ax.set_ylim(-1.5, 1.5)

# Légende
ax.legend()

fig.tight_layout()
plt.show()