"""
PHY-1007 Hiver 2026 — Capteur MEMS capacitif
==============================================
Simulation numérique complète par résolution de l'équation de Laplace (ScalarField).

Physique modélisée
------------------
  Grille 2D (plan xy) :
    x = direction de l'espacement inter-doigts  (direction du déplacement x_mems)
    y = direction le long des doigts             (longueur L)
  Profondeur hors-plan (z) = épaisseur des doigts t.

Conversion d'unités
-------------------
  C_grille est adimensionnel (charge en unités de ε₀, tension en Volts).
  C_phys [F] = C_grille × ε₀ × t × (N/N_sim) × (L / l_phys)

  Dérivation : pour un condensateur plan avec l_phys < L et N_sim < N,
  la capacité est linéaire en longueur et en nombre de paires → mise à l'échelle exacte.

Questions traitées
------------------
  Q1 : C(a), dC/da, ΔV(a) — numérique vs analytique
  Q2 : comparison design de référence vs design ±50g (fusée GAUL)

Utilisation
-----------
  python mems_simulation.py

  Requiert : scalarfield.py (ScalarField), solvers.py (LaplacianSolver), utils.py
  Sorties   : mems_Q1_analysis.png, mems_Q2_comparison.png
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scalarfield import ScalarField

# ── Constantes physiques ───────────────────────────────────────────────────────
EPS0 = 8.854e-12   # Permittivité du vide [F/m]
G    = 9.81        # Accélération gravitationnelle [m/s²]


# ══════════════════════════════════════════════════════════════════════════════
# Paramètres d'un design MEMS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MEMSDesign:
    """
    Paramètres physiques et de discrétisation d'un accéléromètre MEMS capacitif.

    Géométrie simulée
    -----------------
    Le plan xy (grille 2D) coupe les doigts perpendiculairement à leur épaisseur t.
    L'axe x est la direction du mouvement de la masse d'épreuve.
    L'axe y court le long des doigts sur une longueur l_sim = l_fraction × S pixels.

    Paramètres
    ----------
    name        : étiquette du design
    k           : constante de rappel [N/m]
    m           : masse d'épreuve [kg]
    d0          : gap inter-doigts au repos [m]
    N           : nombre total de paires de doigts (design physique réel)
    L           : longueur réelle des doigts [m]
    t           : épaisseur (hauteur hors-plan) des doigts [m]
    V0          : tension de polarisation [V]
    pixel_size  : résolution de la grille [m/pixel]
    N_sim       : paires de doigts simulées (< N pour la rapidité de calcul)
    l_fraction  : fraction de S réservée à la longueur des doigts dans la grille
    """
    name: str

    # — Mécanique —
    k: float           # [N/m]
    m: float           # [kg]

    # — Géométrie physique —
    d0: float          # gap au repos [m]
    N: int             # paires totales
    L: float           # longueur doigts [m]
    t: float           # épaisseur (profondeur z) [m]
    V0: float = 3.0    # tension [V]

    # — Discrétisation —
    pixel_size: float  = 0.25e-6   # [m/pixel]
    N_sim: int         = 6         # paires simulées (suffisant pour capturer la physique)
    l_fraction: float  = 0.90      # fraction de S pour la longueur des doigts


# ── Designs prédéfinis ─────────────────────────────────────────────────────────

DESIGN_Q1 = MEMSDesign(
    name       = "Q1 — Référence",
    k          = 1.0,
    m          = 1e-9,      # 1 ng
    d0         = 2e-6,      # 2 μm
    N          = 50,
    L          = 200e-6,    # 200 μm
    t          = 3e-6,      # 3 μm
    V0         = 3.0,
    pixel_size = 0.25e-6,   # 0.25 μm/pixel → d0 = 8 pixels
    N_sim      = 6,
)

DESIGN_Q2 = MEMSDesign(
    name       = "Q2 — Fusée GAUL (±50 g)",
    k          = 5.0,       # raideur ×5 → 5× moins de déplacement à pleine échelle
    m          = 1e-9,
    d0         = 2e-6,
    N          = 100,       # ×2 paires pour compenser la perte de sensibilité
    L          = 400e-6,    # ×2 longueur
    t          = 3e-6,
    V0         = 3.0,
    pixel_size = 0.25e-6,
    N_sim      = 6,
)


# ══════════════════════════════════════════════════════════════════════════════
# Calcul des paramètres de grille
# ══════════════════════════════════════════════════════════════════════════════

def _grid_params(design: MEMSDesign) -> dict:
    """
    Convertit les paramètres physiques du design en dimensions de grille (pixels).

    Géométrie de la grille
    ----------------------
    Chaque paire de doigts occupe delta pixels en x :
        delta = t_px + 2 × d0_px

    Au repos (d_px = 0), le doigt négatif est centré entre deux doigts positifs
    → gap de d0_px de chaque côté.

    Retourne un dict avec les clés :
        t_px, d0_px, delta_px, S_px, l_px, l_phys
    """
    px = design.pixel_size
    t_px   = max(1, round(design.t  / px))
    d0_px  = max(2, round(design.d0 / px))
    delta  = t_px + 2 * d0_px        # période entre doigts de même signe
    S      = design.N_sim * delta     # taille totale de la grille
    l      = int(design.l_fraction * S)
    return {
        "t_px"   : t_px,
        "d0_px"  : d0_px,
        "delta"  : delta,
        "S"      : S,
        "l_px"   : l,
        "l_phys" : l * px,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Simulation Laplace pour un déplacement donné
# ══════════════════════════════════════════════════════════════════════════════

def simulate_one_point(design: MEMSDesign, d_px: int) -> float | None:
    """
    Résout l'équation de Laplace sur une grille 2D et retourne la capacité
    physique totale C [F] pour un déplacement de la masse d'épreuve de d_px pixels.

    Le déplacement d_px est positif si la masse se rapproche des doigts négatifs
    (gap côté positif réduit, gap côté négatif agrandi).

    Retourne None si les doigts entrent en contact mécanique.

    Conversion physique
    -------------------
    La grille 2D donne C_grille adimensionnel (F/m in z, sans unité de longueur x).
    C_phys = C_grille × ε₀ × t  × (N/N_sim) × (L/l_phys)
    ──────────────────────────────────────────────────────
    ε₀ × t      : intégration sur la profondeur z (épaisseur des doigts)
    N/N_sim     : mise à l'échelle au nombre réel de paires
    L/l_phys    : mise à l'échelle à la longueur réelle des doigts
                  (valide car C ∝ L pour L >> d0)
    """
    g = _grid_params(design)
    t_px, d0_px, delta, S, l_px = g["t_px"], g["d0_px"], g["delta"], g["S"], g["l_px"]

    offset = delta // 2 + d_px   # position x du bord gauche des doigts négatifs

    # Vérification : pas de contact mécanique
    if offset <= t_px or offset >= delta - t_px:
        return None

    # ── Construction de la grille ──────────────────────────────────────────
    pot = ScalarField(shape=(S, S))

    # Bus positif  (bande y = 0:3,   toute la hauteur x)
    # Bus négatif  (bande y = S-3:S, toute la hauteur x)
    pot.add_boundary_condition((slice(None), slice(0,   3)),     design.V0)
    pot.add_boundary_condition((slice(None), slice(S-3, S)),     0.0)

    # Doigts positifs : partent du bord y=0, occupent x = [i×delta, i×delta+t_px]
    for i in range(design.N_sim):
        pot.add_boundary_condition(
            (slice(i * delta, i * delta + t_px), slice(0, l_px)),
            design.V0
        )

    # Doigts négatifs : partent du bord y=S, décalés de d_px en x
    for i in range(design.N_sim):
        x0 = offset + i * delta
        pot.add_boundary_condition(
            (slice(x0, x0 + t_px), slice(S - l_px, S)),
            0.0
        )

    pot.apply_conditions()
    pot.solve_laplace_by_relaxation()   # méthode de relaxation (Jacobi itératif)

    # ── Calcul de la charge par intégration du flux ────────────────────────
    # E = -∇V, sigma = ε₀ E·n  (loi de Gauss sur le contour des électrodes +)
    Ex, Ey = pot.gradient()

    pos_mask = np.zeros((S, S), dtype=bool)
    pos_mask[:, 0:3] = True                                          # bus positif
    for i in range(design.N_sim):
        pos_mask[i * delta: i * delta + t_px, 0:l_px] = True        # doigts +

    outline, nx, ny = pot.boundary_outline(pos_mask)
    sigma   = -(Ex[outline] * nx[outline] + Ey[outline] * ny[outline])
    C_grid  = np.sum(sigma) / design.V0                              # adimensionnel

    # ── Conversion physique ────────────────────────────────────────────────
    C_phys = (
        C_grid
        * EPS0
        * design.t                                 # profondeur hors-plan
        * (design.N   / design.N_sim)              # mise à l'échelle N
        * (design.L   / g["l_phys"])               # mise à l'échelle L
    )
    return C_phys


# ══════════════════════════════════════════════════════════════════════════════
# Balayage C(a) — boucle principale de simulation
# ══════════════════════════════════════════════════════════════════════════════

def sweep_C_vs_acceleration(
    design   : MEMSDesign,
    n_points : int  = 13,
    verbose  : bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Balaye le déplacement d de -(d0 - 1 pixel) à +(d0 - 1 pixel).
    Convertit chaque déplacement d_px en accélération physique :
        a = d_px × pixel_size × k / m

    Retourne
    --------
    a [m/s²], C [F], x [m]  — tableaux NumPy triés par accélération croissante
    """
    g = _grid_params(design)
    d0_px = g["d0_px"]
    d_max = d0_px - 1          # déplacement maximal avant contact [pixels]

    # Génération des points de déplacement entiers
    d_floats   = np.linspace(-d_max, d_max, n_points)
    d_integers = np.unique(np.round(d_floats).astype(int))

    a_list, C_list, x_list = [], [], []

    for d in d_integers:
        x_phys = int(d) * design.pixel_size
        a_phys = x_phys * design.k / design.m

        if verbose:
            print(
                f"  d = {d:+3d} px  |  x = {x_phys*1e9:+7.1f} nm  "
                f"|  a = {a_phys/G:+7.2f} g  ...",
                end=" "
            )

        t0 = time.time()
        C  = simulate_one_point(design, int(d))
        dt = time.time() - t0

        if C is None:
            if verbose: print("contact mécanique — ignoré")
            continue

        a_list.append(a_phys)
        C_list.append(C)
        x_list.append(x_phys)

        if verbose:
            print(f"C = {C*1e15:.3f} fF  [{dt:.1f} s]")

    return np.array(a_list), np.array(C_list), np.array(x_list)


# ══════════════════════════════════════════════════════════════════════════════
# Modèles analytiques (comparaison)
# ══════════════════════════════════════════════════════════════════════════════

def C_analytique(design: MEMSDesign, a: np.ndarray) -> np.ndarray:
    """
    C(a) analytique — modèle condensateur plan :
        C(a) = N ε₀ L t / (d₀ − m·a/k)
    Valable pour |x| = |m·a/k| << d₀.
    """
    x = design.m * a / design.k
    return design.N * EPS0 * design.L * design.t / (design.d0 - x)


def dCda_analytique(design: MEMSDesign) -> float:
    """
    Sensibilité analytique évaluée en a = 0 :
        dC/da|₀ = C₀ × m / (k × d₀)
    """
    C0 = design.N * EPS0 * design.L * design.t / design.d0
    return C0 * design.m / (design.k * design.d0)


def delta_V_analytique(design: MEMSDesign, a: np.ndarray) -> np.ndarray:
    """
    Variation de tension à charge constante (condensateur déconnecté après V₀) :
        Q = C₀ V₀  →  ΔV(a) = V₀ (C₀/C(a) − 1) = −V₀ · (m·a)/(k·d₀)  [exact]
    La relation est linéaire à charge constante (pas d'approximation en x/d₀).
    """
    C0 = design.N * EPS0 * design.L * design.t / design.d0
    return design.V0 * (C0 / C_analytique(design, a) - 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Résumé numérique en console
# ══════════════════════════════════════════════════════════════════════════════

def afficher_resume(design: MEMSDesign, a: np.ndarray, C: np.ndarray) -> None:
    """Affiche les indicateurs clés du design dans la console."""
    idx0       = np.argmin(np.abs(a))
    C0_num     = C[idx0]
    C0_ana     = design.N * EPS0 * design.L * design.t / design.d0
    dCda_num   = np.gradient(C, a)[idx0]
    dCda_ana   = dCda_analytique(design)
    DV_1g      = delta_V_analytique(design, np.array([G]))[0]
    DV_50g     = delta_V_analytique(design, np.array([50*G]))[0]
    x_50g      = design.m * 50 * G / design.k
    contact_g  = design.k * design.d0 / (design.m * G)
    nl_50g     = (x_50g / design.d0)**2 * 100          # non-linéarité à ±50 g [%]

    print(f"\n{'═'*57}")
    print(f"  {design.name}")
    print(f"{'─'*57}")
    print(f"  C₀  (numérique)    = {C0_num*1e15:>9.3f} fF")
    print(f"  C₀  (analytique)   = {C0_ana*1e15:>9.3f} fF")
    print(f"  dC/da (num.)       = {dCda_num/G*1e18:>9.4f} aF/g")
    print(f"  dC/da (ana.)       = {dCda_ana/G*1e18:>9.4f} aF/g")
    print(f"  ΔV à  1g           = {DV_1g*1e3:>9.2f} mV")
    print(f"  ΔV à 50g           = {DV_50g*1e3:>9.1f} mV")
    print(f"  x/d₀ à 50g         = {x_50g/design.d0*100:>9.1f} %")
    print(f"  Non-lin. à 50g     = {nl_50g:>9.2f} %")
    print(f"  Accél. de contact  = {contact_g:>9.1f} g")
    print(f"{'═'*57}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Analyse Question 1
# ══════════════════════════════════════════════════════════════════════════════

def figure_Q1(design: MEMSDesign, a_num: np.ndarray, C_num: np.ndarray) -> None:
    """
    Quatre sous-graphiques :
      (a) C(a)      numérique vs analytique
      (b) dC/da     numérique vs valeur analytique constante
      (c) ΔV(a)     à charge constante
      (d) Écart rel. entre numérique et analytique (capture des effets de frange)
    """
    # Courbes analytiques sur maillage dense
    a_dense = np.linspace(a_num.min() * 1.05, a_num.max() * 1.05, 500)
    C_ana   = C_analytique(design, a_dense)
    DV_ana  = delta_V_analytique(design, a_dense)
    dCda_0  = dCda_analytique(design)

    # Estimations numériques dérivées
    dCda_num = np.gradient(C_num, a_num)
    idx0     = np.argmin(np.abs(a_num))
    C0_num   = C_num[idx0]
    DV_num   = design.V0 * (C0_num / C_num - 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Analyse numérique (Laplace) — {design.name}\n"
        f"k = {design.k} N/m  |  m = {design.m*1e9:.1f} ng  |  "
        f"d₀ = {design.d0*1e6:.1f} μm  |  N = {design.N}",
        fontsize=12, fontweight="bold"
    )

    BLUE, RED, GREEN = "#534AB7", "#D85A30", "#0F6E56"

    # ── (a) C(a) ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(a_dense/G, C_ana*1e15,  "-",  color=BLUE, lw=2,   label="Analytique (plan)")
    ax.plot(a_num/G,   C_num*1e15,  "o",  color=RED,  ms=8,   label="Simulation Laplace",
            zorder=5)
    ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("Accélération a [g]")
    ax.set_ylabel("Capacité C [fF]")
    ax.set_title("(a)  C(a)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (b) dC/da ─────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(a_num/G, dCda_num/G*1e18, "o-", color=RED,  ms=7, label="Numérique (diff. finie)")
    ax.axhline(dCda_0/G*1e18, color=BLUE, ls="--", lw=2,
               label=f"Analytique = {dCda_0/G*1e18:.4f} aF/g")
    ax.set_xlabel("Accélération a [g]")
    ax.set_ylabel("dC/da [aF/g]")
    ax.set_title("(b)  Sensibilité dC/da")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (c) ΔV(a) à charge constante ──────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(a_dense/G, DV_ana*1e3,  "-",  color=BLUE, lw=2,  label="Analytique")
    ax.plot(a_num/G,   DV_num*1e3,  "o",  color=RED,  ms=8,  label="Simulation")
    ax.axvline(0,  color="gray",  lw=0.6, ls="--")
    ax.axhline(0,  color="gray",  lw=0.6, ls="--")
    ax.set_xlabel("Accélération a [g]")
    ax.set_ylabel("ΔV [mV]")
    ax.set_title(f"(c)  ΔV(a) à charge constante  [V₀ = {design.V0:.0f} V]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (d) Écart relatif numérique / analytique ───────────────────────────
    ax = axes[1, 1]
    C_ana_pts = C_analytique(design, a_num)
    err_rel   = (C_num - C_ana_pts) / C_ana_pts * 100
    ax.plot(a_num/G, err_rel, "o-", color=GREEN, ms=7)
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("Accélération a [g]")
    ax.set_ylabel("(C_num − C_ana) / C_ana  [%]")
    ax.set_title(
        "(d)  Écart relatif numérique / analytique\n"
        "(effets de frange et champ non-uniforme)"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mems_Q1_analysis.png", dpi=150, bbox_inches="tight")
    print("→ Sauvegardé : mems_Q1_analysis.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Comparaison Q1 vs Q2
# ══════════════════════════════════════════════════════════════════════════════

def figure_Q2(
    d1: MEMSDesign, a1: np.ndarray, C1: np.ndarray,
    d2: MEMSDesign, a2: np.ndarray, C2: np.ndarray,
) -> None:
    """
    Quatre sous-graphiques comparant les deux designs sur la plage ±50 g :
      (a) C/C₀ normalisée
      (b) Non-linéarité (Δ analytique/analytique linéarisé)
      (c) Déplacement relatif x/d₀  (avec marqueur de contact)
      (d) Sensibilité dC/da
    """
    BLUE, RED, GRAY = "#534AB7", "#D85A30", "#888780"
    LIM_G = 60   # plage d'affichage [g]

    a_range = np.linspace(-LIM_G*G, LIM_G*G, 600)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Comparaison Q1 vs Q2 — Design ±50 g pour la fusée GAUL",
        fontsize=13, fontweight="bold"
    )

    # ── (a) C/C₀ normalisée ───────────────────────────────────────────────
    ax = axes[0, 0]
    for des, a, C, col, label in [
        (d1, a1, C1, BLUE, d1.name),
        (d2, a2, C2, RED,  d2.name),
    ]:
        idx0 = np.argmin(np.abs(a))
        ax.plot(a/G, C/C[idx0], "o-", color=col, ms=6, label=label)

    for s in (-1, 1):
        ax.axvline(s*50, color="red", lw=1.2, ls="--",
                   label="±50 g" if s == 1 else None)
    ax.set_xlabel("Accélération a [g]")
    ax.set_ylabel("C / C₀")
    ax.set_title("(a)  Réponse normalisée C/C₀")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-LIM_G, LIM_G)

    # ── (b) Non-linéarité ─────────────────────────────────────────────────
    # Référence linéaire : C_lin(a) = C₀ (1 + x/d₀)  (Taylor 1er ordre)
    ax = axes[0, 1]
    for des, a, C, col, label in [
        (d1, a1, C1, BLUE, d1.name),
        (d2, a2, C2, RED,  d2.name),
    ]:
        C0    = des.N * EPS0 * des.L * des.t / des.d0
        C_lin = C0 * (1 + des.m * a / (des.k * des.d0))  # linéarisé
        nl    = (C - C_lin) / C_lin * 100
        ax.plot(a/G, nl, "o-", color=col, ms=6, label=label)

    for s in (-1, 1):
        ax.axvline(s*50, color="red", lw=1.2, ls="--")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("Accélération a [g]")
    ax.set_ylabel("(C − C_lin) / C_lin  [%]")
    ax.set_title("(b)  Non-linéarité vs modèle linéarisé")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-LIM_G, LIM_G)

    # ── (c) Déplacement x/d₀ ──────────────────────────────────────────────
    ax = axes[1, 0]
    for des, col, label in [(d1, BLUE, d1.name), (d2, RED, d2.name)]:
        xd0 = des.m * a_range / (des.k * des.d0) * 100   # [%]
        ax.plot(a_range/G, xd0, "-", color=col, lw=2, label=label)

    ax.axhline( 100, color="black", lw=1.5, ls="-",
                label="Contact (x = ±d₀)")
    ax.axhline(-100, color="black", lw=1.5, ls="-")
    ax.axhspan( 100, 115, color="red",  alpha=0.10)
    ax.axhspan(-115,-100, color="red",  alpha=0.10)
    for s in (-1, 1):
        ax.axvline(s*50, color="red", lw=1.2, ls="--")
    ax.set_xlabel("Accélération a [g]")
    ax.set_ylabel("x / d₀  [%]")
    ax.set_title("(c)  Déplacement relatif x/d₀")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-LIM_G, LIM_G)
    ax.set_ylim(-115, 115)

    # ── (d) Sensibilité dC/da ─────────────────────────────────────────────
    ax = axes[1, 1]
    for des, a, C, col, label in [
        (d1, a1, C1, BLUE, d1.name),
        (d2, a2, C2, RED,  d2.name),
    ]:
        s_num = np.gradient(C, a) / G * 1e18                 # [aF/g]
        s_ana = dCda_analytique(des) / G * 1e18              # [aF/g]
        ax.plot(a/G, s_num, "o-", color=col, ms=6,
                label=f"{label}\n(ana. = {s_ana:.4f} aF/g)")

    for s in (-1, 1):
        ax.axvline(s*50, color="red", lw=1.2, ls="--")
    ax.set_xlabel("Accélération a [g]")
    ax.set_ylabel("dC/da  [aF/g]")
    ax.set_title("(d)  Sensibilité dC/da")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-LIM_G, LIM_G)

    plt.tight_layout()
    plt.savefig("mems_Q2_comparison.png", dpi=150, bbox_inches="tight")
    print("→ Sauvegardé : mems_Q2_comparison.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Programme principal
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 60)
    print("  PHY-1007 — Simulation MEMS capacitif par équation de Laplace")
    print("═" * 60)

    # ── Question 1 ──────────────────────────────────────────────────────────
    print(f"\n[ Q1 ]  Design de référence — balayage C(a)")
    g = _grid_params(DESIGN_Q1)
    print(
        f"  Grille : {g['S']}×{g['S']} pixels  |  "
        f"d₀ = {g['d0_px']} px  |  t = {g['t_px']} px  |  "
        f"Δ = {g['delta']} px/pair  |  N_sim = {DESIGN_Q1.N_sim}"
    )
    print(f"  a_max ≈ ±{(g['d0_px']-1)*DESIGN_Q1.pixel_size*DESIGN_Q1.k/(DESIGN_Q1.m*G):.0f} g")
    print()

    t0 = time.time()
    a1, C1, x1 = sweep_C_vs_acceleration(DESIGN_Q1, n_points=13, verbose=True)
    print(f"\n  Temps total Q1 : {time.time()-t0:.1f} s")

    afficher_resume(DESIGN_Q1, a1, C1)
    figure_Q1(DESIGN_Q1, a1, C1)

    # ── Question 2 ──────────────────────────────────────────────────────────
    print(f"\n[ Q2 ]  Design fusée GAUL — balayage C(a)")
    g2 = _grid_params(DESIGN_Q2)
    print(
        f"  Grille : {g2['S']}×{g2['S']} pixels  |  "
        f"d₀ = {g2['d0_px']} px  |  t = {g2['t_px']} px  |  "
        f"Δ = {g2['delta']} px/pair  |  N_sim = {DESIGN_Q2.N_sim}"
    )
    print(f"  a_max ≈ ±{(g2['d0_px']-1)*DESIGN_Q2.pixel_size*DESIGN_Q2.k/(DESIGN_Q2.m*G):.0f} g")
    print()

    t0 = time.time()
    a2, C2, x2 = sweep_C_vs_acceleration(DESIGN_Q2, n_points=13, verbose=True)
    print(f"\n  Temps total Q2 : {time.time()-t0:.1f} s")

    afficher_resume(DESIGN_Q2, a2, C2)
    figure_Q2(DESIGN_Q1, a1, C1, DESIGN_Q2, a2, C2)

    print("\nFigures générées :")
    print("  mems_Q1_analysis.png")
    print("  mems_Q2_comparison.png")