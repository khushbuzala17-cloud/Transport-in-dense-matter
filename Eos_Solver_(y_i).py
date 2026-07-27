import pandas as pd
import numpy as np
from scipy.optimize import approx_fprime, root

# ==========================================================
# 1. Physical Constants & Model Parameters
# ==========================================================
HBAR = 197.3269804      # MeV fm
EPS = 1.0e-8            # Numerical derivative step size
ETA = 0.6               # Model parameter eta

# Particle masses (MeV)
mN = 939.0
mu = 5.0
md = 7.0
ms = 150.0
me = 0.511

# Hadronic model parameters
nsat = 0.16             # Saturation density (fm^-3)
a0 = -96.64
b0 = 58.85
gamma = 1.40
a1 = -26.06
b1 = 7.34
gamma1 = 2.45

# Quark model parameters
a = 0.20                # fm^2
B = (165.0)**4          # Bag constant (MeV^4)


# ==========================================================
# 2. Safe Mathematical Helpers
# ==========================================================
def safe_pow(x, p):
    """Prevents errors when solver tests negative densities."""
    return np.sign(x) * (np.abs(x) ** p)


# ==========================================================
# 3. Physics Steps (Fractions, Momenta, Energies)
# ==========================================================

def calculate_fractions(nb, yn, yp, ys, ye, f):
    """Step 1: Eq. (9) to Eq. (13)"""
    # Prevent division by zero if f approaches 1.0
    f_safe = np.clip(f, 0.0, 0.999999)
    denom = 1.0 - f_safe

    yu = (1.0 + ye - (f * yn) - (2.0 * f * yp)) / (1-f)
    yd = (2.0 - ye - (2.0 * f * yn) - (f * yp) - (ys*(1.0 - f))) / (1-f)
    yeN = yp
    yeQ = (ye - (f * yp)) / (1-f)
    yeG = ye

    return yu, yd, yeN, yeQ, yeG


def calculate_kF(nb, yn, yp, yu, yd, ys, yeN, yeQ, yeG):
    """Step 2: Fermi Momenta"""
    p = 1 / 3.0
    # Hadrons & Leptons (factor of 3)
    kFn = safe_pow(3.0 * np.pi**2 * HBAR**3 * nb * yn, p)
    kFp = safe_pow(3.0 * np.pi**2 * HBAR**3 * nb * yp, p)
    kFeN = safe_pow(3.0 * np.pi**2 * HBAR**3 * nb * yeN, p)
    kFeQ = safe_pow(3.0 * np.pi**2 * HBAR**3 * nb * yeQ, p)
    kFeG = safe_pow(3.0 * np.pi**2 * HBAR**3 * nb * yeG, p)

    # Quarks (factor of 1)
    kFu = safe_pow(np.pi**2 * HBAR**3 * nb * yu, p)
    kFd = safe_pow(np.pi**2 * HBAR**3 * nb * yd, p)
    kFs = safe_pow(np.pi**2 * HBAR**3 * nb * ys, p)

    return kFn, kFp, kFu, kFd, kFs, kFeN, kFeQ, kFeG


def epsilon_hadronic(nb, yn, yp, kFn, kFp):
    """Step 3: Hadronic Energy Density (Eq. 41)"""
    En, Ep = np.sqrt(kFn**2 + mN**2), np.sqrt(kFp**2 + mN**2)

    term_n = kFn * En * (2.0 * kFn**2 + mN**2) - mN**4 * np.log(np.abs(kFn + En) / mN)
    term_p = kFp * Ep * (2.0 * kFp**2 + mN**2) - mN**4 * np.log(np.abs(kFp + Ep) / mN)
    kinetic = (term_n + term_p) / (8.0 * np.pi**2 * HBAR**3)

    rho = nb * (yn + yp)
    int1 = 4.0 * nb**2 * yn * yp * (a0 / nsat + (b0 / (nsat**gamma)) * safe_pow(rho, gamma - 1))
    int2 = nb**2 * (yn - yp)**2 * (a1 / nsat + (b1 / (nsat**gamma1)) * safe_pow(rho, gamma1 - 1))

    return kinetic + int1 + int2


def epsilon_quark(nb, yu, yd, ys, kFu, kFd, kFs):
    """Step 3: Quark Energy Density (Eq. 44-45)"""
    def e_q(kF, mass):
        E = np.sqrt(kF**2 + mass**2)
        term = kF * E * (2.0 * kF**2 + mass**2) - mass**4 * np.log(np.abs(kF + E) / mass)
        return 3.0 * term / (8.0 * np.pi**2 * HBAR**3)

    eps_kin = e_q(kFu, mu) + e_q(kFd, md) + e_q(kFs, ms)
    eps_int = 0.5 * a * (nb * (yu + yd + ys))**2
    eps_bag = B / (HBAR**3)

    return eps_kin + eps_int + eps_bag


def epsilon_lepton(kF, mass):
    """Step 3: Lepton Energy Density (Eq. 48)"""
    E = np.sqrt(kF**2 + mass**2)
    term = kF * E * (2.0 * kF**2 + mass**2) - mass**4 * np.log(np.abs(kF + E) / mass)
    return term / (8.0 * np.pi**2 * HBAR**3)


# ==========================================================
# 4. Chemical Potentials & Pressures (Steps 4 & 5)
# ==========================================================

def mu_hadronic(nb, yn, yp):
    """Step 4: Hadronic Chemical Potential via Numerical Derivative"""
    def func(y):
        kFn = safe_pow(3.0 * np.pi**2 * HBAR**3 * nb * y[0], 1/3)
        kFp = safe_pow(3.0 * np.pi**2 * HBAR**3 * nb * y[1], 1/3)
        return epsilon_hadronic(nb, y[0], y[1], kFn, kFp) / nb

    grad = approx_fprime(np.array([yn, yp]), func, EPS)
    return grad[0], grad[1]


def mu_quark(nb, yu, yd, ys):
    """Step 4: Quark Chemical Potential via Numerical Derivative"""
    def func(y):
        kFu = safe_pow(np.pi**2 * HBAR**3 * nb * y[0], 1/3)
        kFd = safe_pow(np.pi**2 * HBAR**3 * nb * y[1], 1/3)
        kFs = safe_pow(np.pi**2 * HBAR**3 * nb * y[2], 1/3)
        return epsilon_quark(nb, y[0], y[1], y[2], kFu, kFd, kFs) / nb

    grad = approx_fprime(np.array([yu, yd, ys]), func, EPS)
    return grad[0], grad[1], grad[2]


def mu_lepton(kF, mass):
    """Step 4: Analytical Lepton Chemical Potential"""
    return np.sqrt(kF**2 + mass**2)


def pressure_hadronic(nb, yn, yp, mu_n, mu_p, epsN):
    return nb * (yn * mu_n + yp * mu_p) - epsN


def pressure_quark(nb, yu, yd, ys, mu_u, mu_d, mu_s, epsQ):
    return nb * (yu * mu_u + yd * mu_d + ys * mu_s) - epsQ


def pressure_lepton(nb, ye, mu_e, eps_e):
    return nb * ye * mu_e - eps_e


# ==========================================================
# 5. Complete Thermodynamic State Wrapper
# ==========================================================

def calculate_state(nb, yn, yp, ys, ye, f):
    # Step 1: Fractions
    yu, yd, yeN, yeQ, yeG = calculate_fractions(nb, yn, yp, ys, ye, f)

    # Physical safety clipping for intermediate evaluation
    yu_s, yd_s, ys_s = np.maximum(yu, 1e-10), np.maximum(yd, 1e-10), np.maximum(ys, 1e-10)
    yn_s, yp_s = np.maximum(yn, 1e-10), np.maximum(yp, 1e-10)
    yeN_s, yeQ_s, yeG_s = np.maximum(yeN, 1e-10), np.maximum(yeQ, 1e-10), np.maximum(yeG, 1e-10)

    # Step 2: Fermi Momenta
    kFn, kFp, kFu, kFd, kFs, kFeN, kFeQ, kFeG = calculate_kF(
        nb, yn_s, yp_s, yu_s, yd_s, ys_s, yeN_s, yeQ_s, yeG_s
    )

    # Step 3: Energy Densities
    epsN = epsilon_hadronic(nb, yn_s, yp_s, kFn, kFp)
    epsQ = epsilon_quark(nb, yu_s, yd_s, ys_s, kFu, kFd, kFs)
    eps_eN = epsilon_lepton(kFeN, me)
    eps_eQ = epsilon_lepton(kFeQ, me)
    eps_eG = epsilon_lepton(kFeG, me)

    # Step 4: Chemical Potentials
    mu_n, mu_p = mu_hadronic(nb, yn_s, yp_s)
    mu_u, mu_d, mu_s = mu_quark(nb, yu_s, yd_s, ys_s)
    mu_eN, mu_eQ, mu_eG = mu_lepton(kFeN, me), mu_lepton(kFeQ, me), mu_lepton(kFeG, me)

    # Step 5: Pressures
    PN = pressure_hadronic(nb, yn_s, yp_s, mu_n, mu_p, epsN)
    PQ = pressure_quark(nb, yu_s, yd_s, ys_s, mu_u, mu_d, mu_s, epsQ)
    PeN = pressure_lepton(nb, yeN_s, mu_eN, eps_eN)
    PeQ = pressure_lepton(nb, yeQ_s, mu_eQ, eps_eQ)
    PeG = pressure_lepton(nb, yeG_s, mu_eG, eps_eG)

    return {
        "yu": yu, "yd": yd, "yeN": yeN, "yeQ": yeQ, "yeG": yeG,
        "kFn": kFn, "kFp": kFp, "kFu": kFu, "kFd": kFd, "kFs": kFs,
        "kFeN": kFeN, "kFeQ": kFeQ, "kFeG": kFeG,
        "mu_n": mu_n, "mu_p": mu_p, "mu_u": mu_u, "mu_d": mu_d, "mu_s": mu_s,
        "mu_eN": mu_eN, "mu_eQ": mu_eQ, "mu_eG": mu_eG,
        "epsilonN": epsN, "epsilonQ": epsQ,
        "epsilon_eN": eps_eN, "epsilon_eQ": eps_eQ, "epsilon_eG": eps_eG,
        "PN": PN, "PQ": PQ, "PeN": PeN, "PeQ": PeQ, "PeG": PeG
    }


# ==========================================================
# 6. Residual Equations (LHS - RHS = 0)
# ==========================================================

def residuals(x, nb, eta):
    yn, yp, ys, ye, f = x

    # Penalty for unphysical guesses (negative fractions or f > 1)
    penalty = 0.0
    if any(val < -0.001 for val in [yn, yp, ys, ye, f]) or f > 1.001 or f < -0.001:
        penalty = 1e6 * (sum(abs(min(v, 0)) for v in [yn, yp, ys, ye, f]) + abs(max(f - 1.0, 0)))

    try:
        state = calculate_state(nb, yn, yp, ys, ye, f)

        # Residuals from equations:
        # R1: mu_n = mu_u + 2*mu_d
        R1 = state["mu_n"] - (state["mu_u"] + 2.0 * state["mu_d"])

        # R2: mu_p = 2*mu_u + mu_d - eta*(mu_eN - mu_eQ)
        R2 = state["mu_p"] - (2.0 * state["mu_u"] + state["mu_d"] - eta * (state["mu_eN"] - state["mu_eQ"]))

        # R3: mu_d = mu_u + eta*mu_eQ + (1 - eta)*mu_eG
        R3 = state["mu_d"] - (state["mu_u"] + eta * state["mu_eQ"] + (1.0 - eta) * state["mu_eG"])

        # R4: mu_d = mu_s
        R4 = state["mu_d"] - state["mu_s"]

        # R5: PN + eta*PeN = PQ + eta*PeQ
        R5 = (state["PN"] + eta * state["PeN"]) - (state["PQ"] + eta * state["PeQ"])

        return [R1 + penalty, R2 + penalty, R3 + penalty, R4 + penalty, R5 + penalty]

    except Exception:
        # Emergency return for numeric overflow
        return [1e10] * 5


# ==========================================================
# 7. Main Loop & Execution
# ==========================================================

# ----------------------------------------------------------
# Define baryon density range (change as needed)
# ----------------------------------------------------------
nb_values = np.arange(0.05, 2.00, 0.01)   # from 0.15 to 1.20 fm^-3


# ----------------------------------------------------------
# Initial guess for the FIRST density only
# ----------------------------------------------------------
x0 = np.array([
    0.90,   # yn
    0.10,   # yp
    0.01,   # ys
    0.05,   # ye
    0.9    # f
])

results = []

print("Starting EOS solver loop...")

for index, nb_val in enumerate(nb_values):

    # Attempt 1: Powell Hybrid method
    sol = root(
        residuals,
        x0,
        args=(nb_val, ETA),
        method='hybr',
        tol=1e-7
    )

    # Attempt 2: Levenberg-Marquardt fallback
    if not sol.success:
        sol = root(
            residuals,
            x0,
            args=(nb_val, ETA),
            method='lm',
            tol=1e-7
        )

    if sol.success:

        yn_sol, yp_sol, ys_sol, ye_sol, f_sol = sol.x

        state = calculate_state(
            nb_val,
            yn_sol,
            yp_sol,
            ys_sol,
            ye_sol,
            f_sol
        )

        row_res = {
            "nB": nb_val,
            "yn": yn_sol,
            "yp": yp_sol,
            "ys": ys_sol,
            "ye": ye_sol,
            "f": f_sol
        }

        row_res.update(state)
        results.append(row_res)

        print(f"{index+1}/{len(nb_values)} : nB = {nb_val:.4f} solved")

        # --------------------------------------------------
        
        # Current solution becomes the next initial guess
        # --------------------------------------------------
        x0 = sol.x.copy()

    else:

        print(f"{index+1}/{len(nb_values)} : nB = {nb_val:.4f} failed")

        # Keep previous guess if solver fails
        continue


# ----------------------------------------------------------
# Save Results
# ----------------------------------------------------------
if results:
    output_df = pd.DataFrame(results)
    output_df.to_excel("final_eos_solved_output1.xlsx", index=False)
    print(f"\nCompleted! {len(results)} points successfully processed.")
else:
    print("\nNo solution found.")