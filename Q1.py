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

e0=1
ell = 2 * 10**(-6) # 2 a 5 micro m
l = 100 * 10**(-6) # 100 a 500 micro m
d0 = 2 * 10**(-6) # 1 a 4 micro m
c0 = e0 * ell * l/d0
m = 1 * 10**(-6) # ~ micro m
k = 1 # 0.1 a 10 N/m
a = np.linspace(0, 8, 300)
cl = 2 * (c0/d0) * (m/k) * a
x=m * a/k
c = ((2 * c0 * x)/d0)/(1-((m * a)/(k * d0))**2)

fig, ax = plt.subplots()
ax.plot(a, cl)
ax.set_xlabel("a")
ax.set_ylabel("c")
ax.set_title("lineraire")
ax.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.plot(a, c, label=r"jsp la")
ax.set_xlabel("a")
ax.set_ylabel("c")
ax.set_title("lineraire")
ax.legend()
fig.tight_layout()
plt.show()
