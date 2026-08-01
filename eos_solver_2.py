
import numpy as np
import pandas as pd
from scipy.optimize import root

# ==========================================================
# 1. Constants & parameters
# ==========================================================
HBAR = 197.3269804      # MeV fm  (= hbar*c)
ETA  = 0.6              # local-to-total electron ratio

# masses (MeV)
mN = 939.0
mu = 5.0                # up-quark mass   (NOTE: 'mu' is a MASS here, not a chem. pot.)
md = 7.0
ms = 150.0
me = 0.511

# Zhao-Lattimer (ZLA) nucleon parameters
nsat   = 0.16
a0, b0, gamma  = -96.64, 58.85, 1.40
a1, b1, gamma1 = -26.06,  7.34, 2.45

# vMIT quark parameters
a = 0.20                # fm^2
B = (165.0) ** 4        # MeV^4


# ==========================================================
# 2. Helpers
# ==========================================================
def safe_pow(x, p):
    return np.sign(x) * (np.abs(x) ** p)

def kF_hadlep(nb, y):            
    arg = 3.0 * np.pi**2 * HBAR**3 * nb * y

    
    if arg > 0:
        return arg ** (1.0 / 3.0)
    else:
        return 0.0


def kF_quark(nb, y):             
    arg = np.pi**2 * HBAR**3 * nb * y

    
    if arg > 0:
        return arg ** (1.0 / 3.0)
    else:
        return 0.0


# ==========================================================
# 3. Mixed-phase fraction relations   (Eqs. 9-13)
# ==========================================================
def calculate_fractions(nb, yn, yp, ys, ye, f):
    denom = 1.0 - np.clip(f, 0.0, 0.999999)
    yu  = (1.0 + ye - f*yn - 2.0*f*yp) / denom
    yd  = (2.0 - ye - 2.0*f*yn - f*yp - ys*denom) / denom
    yeN = yp
    yeQ = (ye - f*yp) / denom
    yeG = ye
    return yu, yd, yeN, yeQ, yeG


# ==========================================================
# 4. Energy densities   (Eqs. 41, 44-45, 48)
# ==========================================================
def epsilon_hadronic(nb, yn, yp, kFn, kFp):
    En, Ep = np.sqrt(kFn**2 + mN**2), np.sqrt(kFp**2 + mN**2)
    term_n = kFn*En*(2.0*kFn**2 + mN**2) - mN**4*np.log(np.abs(kFn + En)/mN)
    term_p = kFp*Ep*(2.0*kFp**2 + mN**2) - mN**4*np.log(np.abs(kFp + Ep)/mN)
    kinetic = (term_n + term_p) / (8.0*np.pi**2*HBAR**3)

    rho = nb * (yn + yp)                        # = nB*(yn+yp) = nucleon number density
    int1 = 4.0*nb**2*yn*yp*(a0/nsat + (b0/nsat**gamma)*safe_pow(rho, gamma-1))
    int2 = nb**2*(yn-yp)**2*(a1/nsat + (b1/nsat**gamma1)*safe_pow(rho, gamma1-1))
    return kinetic + int1 + int2

def epsilon_quark(nb, yu, yd, ys, kFu, kFd, kFs):
    def e_q(kF, mass):
        E = np.sqrt(kF**2 + mass**2)
        term = kF*E*(2.0*kF**2 + mass**2) - mass**4*np.log(np.abs(kF + E)/mass)
        return 3.0*term / (8.0*np.pi**2*HBAR**3)
    eps_kin = e_q(kFu, mu) + e_q(kFd, md) + e_q(kFs, ms)
    eps_int = 0.5 * a * HBAR * (nb*(yu + yd + ys))**2      # <-- HBAR restored (Eq. 44)
    eps_bag = B / (HBAR**3)
    return eps_kin + eps_int + eps_bag

def epsilon_lepton(kF, mass):
    E = np.sqrt(kF**2 + mass**2)
    term = kF*E*(2.0*kF**2 + mass**2) - mass**4*np.log(np.abs(kF + E)/mass)
    return term / (8.0*np.pi**2*HBAR**3)


# ==========================================================
# 5. Chemical potentials   (Eqs. 42, 46, 49)
# ==========================================================
def mu_hadronic(nb, yn, yp, kFn, kFp):
    rho = yn + yp
    common0 = a0*(nb/nsat) + b0*(nb/nsat)**gamma  * rho**(gamma-1)
    common1 = a1*(nb/nsat) + b1*(nb/nsat)**gamma1 * rho**(gamma1-1)
    d0 = b0*(nb/nsat)**gamma *(gamma -1)*rho**(gamma -2)
    d1 = b1*(nb/nsat)**gamma1*(gamma1-1)*rho**(gamma1-2)
    mu_n = (np.sqrt(kFn**2 + mN**2) + 4*yp*common0 + 4*yn*yp*d0
            + 2*(yn-yp)*common1 + (yn-yp)**2*d1)
    mu_p = (np.sqrt(kFp**2 + mN**2) + 4*yn*common0 + 4*yn*yp*d0
            - 2*(yn-yp)*common1 + (yn-yp)**2*d1)
    return mu_n, mu_p

def mu_quark(nb, yu, yd, ys, kFu, kFd, kFs):
    interaction = a * HBAR * nb * (yu + yd + ys)          
    mu_u = np.sqrt(kFu**2 + mu**2) + interaction
    mu_d = np.sqrt(kFd**2 + md**2) + interaction
    mu_s = np.sqrt(kFs**2 + ms**2) + interaction
    return mu_u, mu_d, mu_s

def mu_lepton(kF, mass):
    return np.sqrt(kF**2 + mass**2)


# ==========================================================
# 6. Pressures   (P = nB * sum(y_i mu_i) - eps)
# ==========================================================
def pressure_hadronic(nb, yn, yp, mu_n, mu_p, epsN):
    return nb*(yn*mu_n + yp*mu_p) - epsN

def pressure_quark(nb, yu, yd, ys, mu_u, mu_d, mu_s, epsQ):
    return nb*(yu*mu_u + yd*mu_d + ys*mu_s) - epsQ

def pressure_lepton(nb, ye, mu_e, eps_e):
    return nb*ye*mu_e - eps_e


# ==========================================================
# 7. Full mixed-phase state
# ==========================================================
def calculate_state(nb, yn, yp, ys, ye, f):
    yu, yd, yeN, yeQ, yeG = calculate_fractions(nb, yn, yp, ys, ye, f)

    
    yn_s, yp_s = np.maximum(yn, 1e-12), np.maximum(yp, 1e-12)
    yu_s, yd_s, ys_s = np.maximum(yu, 1e-12), np.maximum(yd, 1e-12), np.maximum(ys, 1e-12)
    yeN_s, yeQ_s, yeG_s = (np.maximum(yeN, 1e-12),
                           np.maximum(yeQ, 1e-12), np.maximum(yeG, 1e-12))

    kFn, kFp = kF_hadlep(nb, yn_s), kF_hadlep(nb, yp_s)
    kFu, kFd, kFs = kF_quark(nb, yu_s), kF_quark(nb, yd_s), kF_quark(nb, ys_s)
    kFeN, kFeQ, kFeG = (kF_hadlep(nb, yeN_s),
                        kF_hadlep(nb, yeQ_s), kF_hadlep(nb, yeG_s))

    epsN = epsilon_hadronic(nb, yn_s, yp_s, kFn, kFp)
    epsQ = epsilon_quark(nb, yu_s, yd_s, ys_s, kFu, kFd, kFs)
    eps_eN, eps_eQ, eps_eG = (epsilon_lepton(kFeN, me),
                              epsilon_lepton(kFeQ, me), epsilon_lepton(kFeG, me))

    mu_n, mu_p = mu_hadronic(nb, yn_s, yp_s, kFn, kFp)
    mu_u, mu_d, mu_s = mu_quark(nb, yu_s, yd_s, ys_s, kFu, kFd, kFs)
    mu_eN, mu_eQ, mu_eG = (mu_lepton(kFeN, me),
                           mu_lepton(kFeQ, me), mu_lepton(kFeG, me))

    PN = pressure_hadronic(nb, yn_s, yp_s, mu_n, mu_p, epsN)
    PQ = pressure_quark(nb, yu_s, yd_s, ys_s, mu_u, mu_d, mu_s, epsQ)
    PeN = pressure_lepton(nb, yeN_s, mu_eN, eps_eN)
    PeQ = pressure_lepton(nb, yeQ_s, mu_eQ, eps_eQ)
    PeG = pressure_lepton(nb, yeG_s, mu_eG, eps_eG)

    return dict(yu=yu, yd=yd, ys=ys, yeN=yeN, yeQ=yeQ, yeG=yeG,
                kFn=kFn, kFp=kFp, kFu=kFu, kFd=kFd, kFs=kFs,
                kFeN=kFeN, kFeQ=kFeQ, kFeG=kFeG,
                mu_n=mu_n, mu_p=mu_p, mu_u=mu_u, mu_d=mu_d, mu_s=mu_s,
                mu_eN=mu_eN, mu_eQ=mu_eQ, mu_eG=mu_eG,
                epsilonN=epsN, epsilonQ=epsQ,
                epsilon_eN=eps_eN, epsilon_eQ=eps_eQ, epsilon_eG=eps_eG,
                PN=PN, PQ=PQ, PeN=PeN, PeQ=PeQ, PeG=PeG)




COLUMNS = [
    "nB", "phase", "f",
    # particle fractions  (y_i = n_i / nB)
    "yn", "yp", "yu", "yd", "ys", "ye", "yeN", "yeQ", "yeG",
    # Fermi momenta  (MeV)
    "kFn", "kFp", "kFu", "kFd", "kFs", "kFeN", "kFeQ", "kFeG",
    # chemical potentials  (MeV)
    "mu_n", "mu_p", "mu_u", "mu_d", "mu_s", "mu_eN", "mu_eQ", "mu_eG",
    # energy densities  (MeV/fm^3)
    "epsilonN", "epsilonQ", "epsilon_eN", "epsilon_eQ", "epsilon_eG", "eps_total",
    # pressures  (MeV/fm^3)
    "PN", "PQ", "PeN", "PeQ", "PeG", "P_total",
]

def blank_record():
    r = {c: 0.0 for c in COLUMNS}
    r["phase"] = ""
    return r



def solve_hadron(nb, yp_guess):
    def eq(v):
        yp = v[0]
        if yp <= 1e-9 or yp >= 1.0:
            return [1e6]
        yn, ye = 1.0 - yp, yp
        kFn, kFp = kF_hadlep(nb, yn), kF_hadlep(nb, yp)
        kFe = kF_hadlep(nb, ye)
        mu_n, mu_p = mu_hadronic(nb, yn, yp, kFn, kFp)
        mu_e = mu_lepton(kFe, me)
        return [mu_n - mu_p - mu_e]

    sol = root(eq, [yp_guess], method='hybr', tol=1e-10)
    if not sol.success or not (0.0 < sol.x[0] < 1.0):
        return None
    yp = sol.x[0]; yn, ye = 1.0 - yp, yp
    kFn, kFp, kFe = kF_hadlep(nb, yn), kF_hadlep(nb, yp), kF_hadlep(nb, ye)
    mu_n, mu_p = mu_hadronic(nb, yn, yp, kFn, kFp)
    mu_e = mu_lepton(kFe, me)
    epsN = epsilon_hadronic(nb, yn, yp, kFn, kFp)
    eps_e = epsilon_lepton(kFe, me)
    PN = pressure_hadronic(nb, yn, yp, mu_n, mu_p, epsN)
    Pe = pressure_lepton(nb, ye, mu_e, eps_e)

    r = blank_record()
    r.update(phase='hadron', f=1.0, yn=yn, yp=yp, ye=ye, yeN=ye,
             kFn=kFn, kFp=kFp, kFeN=kFe,
             mu_n=mu_n, mu_p=mu_p, mu_eN=mu_e,
             epsilonN=epsN, epsilon_eN=eps_e, eps_total=epsN + eps_e,
             PN=PN, PeN=Pe, P_total=PN + Pe)
    r["eps"] = r["eps_total"]; r["x"] = [yp]     
    return r


def solve_quark(nb, guess):

    def eq(v):
        yu, yd, ys, ye = v

        
        penalty = 0.0
        for val in (yu, yd, ys, ye):
            if val < 0.0:
                penalty += 1e6 * (-val)
            elif val > 1.0:
                penalty += 1e6 * (val - 1.0)

        kFu = kF_quark(nb, max(yu, 1e-12))
        kFd = kF_quark(nb, max(yd, 1e-12))
        kFs = kF_quark(nb, max(ys, 1e-12))
        kFe = kF_hadlep(nb, max(ye, 1e-12))

        mu_u, mu_d, mu_s = mu_quark(nb, yu, yd, ys, kFu, kFd, kFs)
        mu_e = mu_lepton(kFe, me)

        return [
            yu + yd + ys - 3.0 + penalty,
            (2.0 * yu - yd - ys) / 3.0 - ye + penalty,
            mu_d - mu_u - mu_e + penalty,
            mu_s - mu_d + penalty
        ]

    sol = root(eq, guess, method='hybr', tol=1e-10)

    if not sol.success:
        sol = root(eq, guess, method='lm', tol=1e-10)
        if not sol.success:
            return None

    yu, yd, ys, ye = sol.x

    
    if (
        yu < 0 or yu > 1 or
        yd < 0 or yd > 1 or
        ys < 0 or ys > 1 or
        ye < 0 or ye > 1
    ):
        return None

    kFu = kF_quark(nb, yu)
    kFd = kF_quark(nb, yd)
    kFs = kF_quark(nb, ys)
    kFe = kF_hadlep(nb, max(ye, 1e-12))

    mu_u, mu_d, mu_s = mu_quark(nb, yu, yd, ys, kFu, kFd, kFs)
    mu_e = mu_lepton(kFe, me)

    epsQ = epsilon_quark(nb, yu, yd, ys, kFu, kFd, kFs)
    eps_e = epsilon_lepton(kFe, me)

    PQ = pressure_quark(nb, yu, yd, ys, mu_u, mu_d, mu_s, epsQ)
    Pe = pressure_lepton(nb, ye, mu_e, eps_e)
    
    r = blank_record()

    r.update(phase='quark', f=0.0, yu=yu, yd=yd, ys=ys, ye=ye, yeQ=ye,

             kFu=kFu, kFd=kFd, kFs=kFs, kFeQ=kFe,

             mu_u=mu_u, mu_d=mu_d, mu_s=mu_s, mu_eQ=mu_e,

             epsilonQ=epsQ, epsilon_eQ=eps_e, eps_total=epsQ + eps_e,

             PQ=PQ, PeQ=Pe, P_total=PQ + Pe)

    r["eps"] = r["eps_total"]; r["x"] = list(sol.x)

    return r



def mixed_residuals(x, nb, eta):

    yn, yp, ys, ye, f = x

    pen = 0.0

    
    for val in (yn, yp, ys, ye):
        if val < 0.0:
            pen += 1e6 * (-val)
        elif val > 1.0:
            pen += 1e6 * (val - 1.0)

    
    if f < 0.0:
        pen += 1e6 * (-f)
    elif f > 1.0:
        pen += 1e6 * (f - 1.0)

    try:
        s = calculate_state(nb, yn, yp, ys, ye, f)

        R1 = s['mu_n'] - (s['mu_u'] + 2.0 * s['mu_d'])
        R2 = s['mu_p'] - (2.0 * s['mu_u'] + s['mu_d']
                          - eta * (s['mu_eN'] - s['mu_eQ']))
        R3 = s['mu_d'] - (s['mu_u']
                          + eta * s['mu_eQ']
                          + (1.0 - eta) * s['mu_eG'])
        R4 = s['mu_d'] - s['mu_s']
        R5 = (s['PN'] + eta * s['PeN']) - (s['PQ'] + eta * s['PeQ'])

        return [
            R1 + pen,
            R2 + pen,
            R3 + pen,
            R4 + pen,
            R5 + pen
        ]

    except Exception:
        return [1e10] * 5


def solve_mixed(nb, eta, guesses):
    """Try several initial guesses; return the first physical converged root."""
    for g in guesses:
        for method in ('hybr', 'lm'):
            sol = root(mixed_residuals, g, args=(nb, eta), method=method, tol=1e-9)
            if not sol.success:
                continue
            yn, yp, ys, ye, f = sol.x
            
            if not (0.0 < f < 1.0):
                continue
            res = np.max(np.abs(mixed_residuals(sol.x, nb, eta)))
            if res > 1e-3:
                continue
            s = calculate_state(nb, yn, yp, ys, ye, f)
            yu, yd = s['yu'], s['yd']
            if min(yn, yp, ys, yu, yd) < -1e-6 or max(yu, yd, ys) > 3.2:
                continue
            
            eps = (f*s['epsilonN'] + (1-f)*s['epsilonQ']
                   + f*eta*s['epsilon_eN'] + (1-f)*eta*s['epsilon_eQ']
                   + (1-eta)*s['epsilon_eG'])
            P = (f*s['PN'] + (1-f)*s['PQ']
                 + f*eta*s['PeN'] + (1-f)*eta*s['PeQ'] + (1-eta)*s['PeG'])

            r = blank_record()
            r.update(phase='mixed', f=f, yn=yn, yp=yp, ys=ys, ye=ye,
                     yu=yu, yd=yd, yeN=s['yeN'], yeQ=s['yeQ'], yeG=s['yeG'],
                     kFn=s['kFn'], kFp=s['kFp'], kFu=s['kFu'], kFd=s['kFd'], kFs=s['kFs'],
                     kFeN=s['kFeN'], kFeQ=s['kFeQ'], kFeG=s['kFeG'],
                     mu_n=s['mu_n'], mu_p=s['mu_p'], mu_u=s['mu_u'], mu_d=s['mu_d'], mu_s=s['mu_s'],
                     mu_eN=s['mu_eN'], mu_eQ=s['mu_eQ'], mu_eG=s['mu_eG'],
                     epsilonN=s['epsilonN'], epsilonQ=s['epsilonQ'],
                     epsilon_eN=s['epsilon_eN'], epsilon_eQ=s['epsilon_eQ'],
                     epsilon_eG=s['epsilon_eG'], eps_total=eps,
                     PN=s['PN'], PQ=s['PQ'], PeN=s['PeN'], PeQ=s['PeQ'], PeG=s['PeG'],
                     P_total=P)
            r["eps"] = eps; r["x"] = list(sol.x)
            return r
    return None


# ==========================================================
# 11. Main loop
# ==========================================================
nb_values = np.arange(0.05, 2.00, 0.01)

g_had = 0.8                                  
g_qrk = [0.01, 0.01, 0.08, 0.02]                  
g_mix_prev = None                              

results = []
print("Starting hybrid EOS solver (eta = %.2f)..." % ETA)

for i, nb in enumerate(nb_values):

    candidates = []

    
    h = solve_hadron(nb, g_had)
    if h:
        candidates.append(h); g_had = h['yp']

    \
    q = solve_quark(nb, g_qrk)
    if q:
        candidates.append(q); g_qrk = q['x']

    
    mix_guesses = []
    if g_mix_prev is not None:
        mix_guesses.append(g_mix_prev)
    mix_guesses += [
        [0.90, 0.06, 0.05, 0.05, 0.80],
        [0.70, 0.10, 0.30, 0.03, 0.50],
        [0.50, 0.12, 0.60, 0.01, 0.30],
        [0.30, 0.10, 0.90, 0.00, 0.15],
    ]
    m = solve_mixed(nb, ETA, mix_guesses)
    if m:
        candidates.append(m); g_mix_prev = m['x']

    if not candidates:
        print(f"{i+1:3d}/{len(nb_values)}  nB={nb:.3f}  NO SOLUTION")
        continue

    
    best = min(candidates, key=lambda c: c['eps'])
    best["nB"] = nb
    results.append({c: best[c] for c in COLUMNS})     

    print(f"{i+1:3d}/{len(nb_values)}  nB={nb:.3f}  -> {best['phase']:6s}"
          f"  f={best['f']:.3f}  ys ={best['ys']:.3f}"
          f"  eps={best['eps_total']:8.2f}  P={best['P_total']:8.2f}")

# ==========================================================
# 12. Save
# ==========================================================
if results:
    df = pd.DataFrame(results, columns=COLUMNS)
    out = "final_eos_solved_output4.xlsx"
    df.to_excel(out, index=False)
    print(f"\nDone. {len(results)} points -> {out}")
    # quick phase-boundary report
    for ph in ('hadron', 'mixed', 'quark'):
        sub = df[df.phase == ph]
        if len(sub):
            print(f"  {ph:6s}: nB in [{sub.nB.min():.3f}, {sub.nB.max():.3f}]  ({len(sub)} pts)")
else:
    print("\nNo solutions found.")
