"""
=============================================================
  Condensateur a plaques rectangulaires finies
  Simulation du potentiel electrique et de la charge de surface
=============================================================
  Utilisation : python condensateur_plaques_finies.py
  Les parametres sont saisis interactivement au lancement.
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.integrate import dblquad

def ask(prompt, default, vmin=None):
    while True:
        try:
            raw = input(f"  {prompt} [{default}] : ").strip()
            val = float(raw) if raw else float(default)
            if vmin is not None and val <= vmin:
                print(f"    ! Valeur doit etre > {vmin}. Reessayez.")
                continue
            return val
        except ValueError:
            print("    ! Entrez un nombre valide (ex: 4.0)")

plt.rcParams.update({
    "figure.figsize": (14, 10),
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

print()
print("=" * 58)
print("   Condensateur a plaques rectangulaires finies")
print("   Appuyez sur Entree pour garder la valeur par defaut")
print("=" * 58)
print()

L     = ask("Longueur de la plaque   L   (m)",     100e-6, vmin=0)
W     = ask("Largeur  de la plaque   W   (m)",     2e-6, vmin=0)
d     = ask("Separation entre plaques d  (m)",     1e-6, vmin=0)
sigma = ask("Densite de charge sigma (V/m)",  1.0, vmin=0)

eps0 = 8.854e-12

print()
print(f"  >> L={L} m | W={W} m | d={d} m | sigma/eps0={sigma} V/m")
print()

def V_rect(L, W, z):
    z = np.where(np.abs(z) < 1e-10, 1e-10, z)
    az = np.abs(z)
    a, b = L / 2, W / 2
    R = np.sqrt(a**2 + b**2 + az**2)
    t1 = a * np.log((b + R) / np.maximum(-b + R, 1e-30))
    t2 = b * np.log((a + R) / np.maximum(-a + R, 1e-30))
    t3 = -2 * az * np.arctan((a * b) / (az * R))
    return t1 + t2 + t3

def V_axe(z, L, W, d, sigma):
    return (sigma / (2 * np.pi)) * (V_rect(L, W, z) - V_rect(L, W, d - z))

def V_ideal(z, d, sigma):
    return np.where(z <= 0,  sigma * d / 2,
           np.where(z >= d, -sigma * d / 2,
                             sigma * (d / 2 - z)))

def V_plaque_2D(x, z0, L, W, sigma, sign=1):
    def integrand(yp, xp):
        r = np.sqrt((x - xp)**2 + yp**2 + z0**2)
        return 1.0 / np.maximum(r, 1e-10)
    val, _ = dblquad(integrand, -L/2, L/2, -W/2, W/2,
                     epsabs=1e-4, epsrel=1e-4)
    return sign * (sigma / (4 * np.pi)) * val

print("Calcul de la carte 2D... (quelques secondes)")
Nx, Nz = 60, 60
xs = np.linspace(-L * 0.7, L * 0.7, Nx)
zs = np.linspace(-d * 1.3, d + d * 1.3, Nz)
XX, ZZ = np.meshgrid(xs, zs)
VV = np.zeros_like(XX)

for i, zi in enumerate(zs):
    for j, xj in enumerate(xs):
        z0_p = zi if not (abs(zi) < 1e-3 and abs(xj) < L/2) else 1e-3
        z0_m = (d - zi) if not (abs(d - zi) < 1e-3 and abs(xj) < L/2) else 1e-3
        VV[i, j] = (
            V_plaque_2D(xj, z0_p, L, W, sigma, sign=+1)
            + V_plaque_2D(xj, z0_m, L, W, sigma, sign=-1)
        )

print("Carte 2D calculee.")

z_axis = np.linspace(-d, 2*d, 800)
V_fin = V_axe(z_axis, L, W, d, sigma)
V_id = V_ideal(z_axis, d, sigma)

V_at_0 = float(V_axe(np.array([1e-6]), L, W, d, sigma)[0])
V_at_d = float(V_axe(np.array([d-1e-6]), L, W, d, sigma)[0])
deltaV = V_at_0 - V_at_d

f_centre = float(V_rect(L, W, d/2))
sigma_extrait = (2 * np.pi * deltaV) / (2 * f_centre)
bord_pct = (sigma - deltaV / d) / sigma * 100

print()
print("─" * 54)
print(f"  Delta V (entre plaques)   = {deltaV:.4f}  [sigma/eps0 * m]")
print(f"  E moyen axial             = {deltaV/d:.4f}  [sigma/eps0]")
print(f"  sigma extrait (depuis DV) = {sigma_extrait:.4f}  (attendu : {sigma:.4f})")
print(f"  Erreur relative           = {abs(sigma_extrait-sigma)/sigma*100:.2f} %")
print(f"  Effet de bord estime      = {bord_pct:.1f} %")
print("─" * 54)
print()

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
tkw = dict(fontsize=11)

ax1 = fig.add_subplot(gs[0, 0])
ax1.fill_betweenx([-sigma*d*0.9, sigma*d*0.9], 0, d, alpha=0.10, label='Zone inter-plaques')
ax1.plot(z_axis, V_fin, label='Plaque finie')
ax1.plot(z_axis, V_id, ls='--', alpha=0.85, label='Plaque infinie (ideal)')
ax1.axvline(0, lw=0.8, alpha=0.5)
ax1.axvline(d, lw=0.8, alpha=0.5)
ax1.set_xlabel(r'axe central $z$ [m]')
ax1.set_ylabel('V(z)  [sigma/eps0 * m]')
ax1.set_title("Potentiel sur l'axe central", **tkw)
ax1.legend(fontsize=9)
ax1.set_xlim(z_axis[0], z_axis[-1])
ax1.annotate('z=0\n(+sigma)', xy=(0, V_at_0), xytext=(-0.55*d, V_at_0*0.55),
             fontsize=8, arrowprops=dict(arrowstyle='->'))
ax1.annotate('z=d\n(-sigma)', xy=(d, V_at_d), xytext=(1.25*d, V_at_d*0.55),
             fontsize=8, arrowprops=dict(arrowstyle='->'))
ax1.text(0.97, 0.05,
         f'DV = {deltaV:.3f} * sigma/eps0\nE moy = {deltaV/d:.3f} * sigma/eps0\n'
         f's extrait = {sigma_extrait:.4f}\nBord : {bord_pct:.1f}%',
         transform=ax1.transAxes, fontsize=8,
         va='bottom', ha='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

ax2 = fig.add_subplot(gs[0, 1])
dz_step = z_axis[1] - z_axis[0]
E_fin = -np.gradient(V_fin, dz_step)
E_id = -np.gradient(V_id, dz_step)
ax2.fill_betweenx([0, sigma * 1.2], 0, d, alpha=0.10)
ax2.plot(z_axis, E_fin, label='Plaque finie')
ax2.plot(z_axis, E_id, ls='--', alpha=0.85, label='Plaque infinie')
ax2.axvline(0, lw=0.8, alpha=0.5)
ax2.axvline(d, lw=0.8, alpha=0.5)
ax2.set_xlabel(r'axe central $z$ [m]')
ax2.set_ylabel('E(z)  [sigma/eps0]')
ax2.set_title('Champ electrique axial', **tkw)
ax2.legend(fontsize=9)
ax2.set_xlim(z_axis[0], z_axis[-1])
ax2.set_ylim(bottom=0)

# ── Graphique 3 : carte 2D du potentiel ──────────────────────────────────────
# Mise a l'echelle : x en unites de L, z en unites de d
# Cela evite le rapport L/d >> 1 qui ecrasait l'axe z avec set_aspect('equal')
ax3 = fig.add_subplot(gs[1, :])

x_norm = xs / L        # x adimensionnel : [-0.7, 0.7]
z_norm = zs / d        # z adimensionnel : [-1.3, 2.3]
XX_n, ZZ_n = np.meshgrid(x_norm, z_norm)

vmax = np.percentile(np.abs(VV), 98)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax3.contourf(XX_n, ZZ_n, VV, levels=60, cmap='RdBu_r', norm=norm, alpha=0.92)
cbar = fig.colorbar(im, ax=ax3, pad=0.02, fraction=0.03)
cbar.set_label(r'$V(x,z)$  [$\sigma/\varepsilon_0 \cdot$ m]', fontsize=10)

ax3.contour(XX_n, ZZ_n, VV,
            levels=np.linspace(-vmax * 0.85, vmax * 0.85, 18),
            colors='white', linewidths=0.5, alpha=0.35)

# Plaques representees en coordonnees normalisees
plate_half = (L / 2) / L          # = 0.5
plate_thick = 0.04                 # epaisseur visuelle en unites de z/d

for zp_n, lbl, col in [(0, r'+$\sigma$', 'tab:red'), (1, r'$-\sigma$', 'tab:blue')]:
    ax3.add_patch(Rectangle(
        (-plate_half, zp_n - plate_thick / 2),
        2 * plate_half, plate_thick,
        color=col, zorder=5, linewidth=0
    ))
    ax3.text(plate_half + 0.02, zp_n, lbl,
             fontsize=11, va='center', fontweight='bold', color=col)

# Domaine : x/L in [-0.72, 0.72], z/d in [-1.4, 2.4] avec marges propres
ax3.set_xlim(-0.72, 0.72)
ax3.set_ylim(-1.4, 2.4)

ax3.set_xlabel(r"position transversale normalisee $x/L$", fontsize=11)
ax3.set_ylabel(r"position axiale normalisee $z/d$", fontsize=11)
ax3.set_title(
    f'Potentiel — plan median (y = 0) \n'
    f'   [L = {L*1e6:.0f} μm,  W = {W*1e6:.1f} μm,  d = {d*1e6:.1f} μm]',
    **tkw
)

# Lignes de reference aux plaques
ax3.axhline(0, color='tab:red',  lw=0.8, ls='--', alpha=0.5)
ax3.axhline(1, color='tab:blue', lw=0.8, ls='--', alpha=0.5)

# Annotation des bords de plaque
ax3.axvline( 0.5, color='gray', lw=0.7, ls=':', alpha=0.6)
ax3.axvline(-0.5, color='gray', lw=0.7, ls=':', alpha=0.6)
ax3.text( 0.51, 2.25, 'bord', fontsize=8, color='gray', ha='left')
ax3.text(-0.51, 2.25, 'bord', fontsize=8, color='gray', ha='right')

#fig.suptitle('Condensateur a plaques rectangulaires finies', fontsize=14, y=1.01)

out = 'condensateur_plaques_finies.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Figure sauvegardee : {out}")
plt.show()