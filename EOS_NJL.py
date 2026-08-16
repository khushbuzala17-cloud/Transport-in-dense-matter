import numpy as np
import pandas as pd
from scipy.optimize import root, fsolve, least_squares
import os
from openpyxl import load_workbook


HBAR = 197.3269804      # MeV fm  (= hbar*c)
ETA  = 1.0            # local-to-total electron ratio

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



# ==========================================================
# 2. Helpers
# ==========================================================
def safe_pow(x, p):
    return np.sign(x) * (np.abs(x) ** p)

def kF_hadlep(nb, y):            # nucleons / leptons : g = 2
    return safe_pow(3.0 * np.pi**2 * HBAR**3 * nb * y, 1.0/3.0)

def kF_quark(nb, y):             # quarks : g = 6
    return safe_pow(np.pi**2 * HBAR**3 * nb * y, 1.0/3.0)


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



NJL_LAMBDA = 602.3                  # MeV, 3-momentum cutoff
NJL_G      = 1.835 / NJL_LAMBDA**2  # MeV^-2   (G*Lambda^2 = 1.835)
NJL_K      = 12.36 / NJL_LAMBDA**5  # MeV^-5   (K*Lambda^5 = 12.36)
NJL_MQ     = 5.5                    # MeV, current mass of u,d
NJL_MS     = 140.7                  # MeV, current mass of s


def _njl_term(p, m):

    E = np.sqrt(p**2 + m**2)
    return p*E*(2.0*p**2 + m**2) - m**4*np.log(np.abs(p + E)/m)

def _njl_F(p, m):
    
    E = np.sqrt(p**2 + m**2)
    return p*E - m**2*np.log(p + E)

def _njl_condensate(m_star, pF):
   
    if pF >= NJL_LAMBDA:
        return 0.0
    return -(3.0/np.pi**2) * m_star * 0.5 * (_njl_F(NJL_LAMBDA, m_star) - _njl_F(pF, m_star))

def _njl_gap_residuals(m_stars, pF):
    m_u, m_d, m_s = m_stars
    pFu, pFd, pFs = pF
    cu = _njl_condensate(m_u, pFu)
    cd = _njl_condensate(m_d, pFd)
    cs = _njl_condensate(m_s, pFs)
    eq_u = m_u - (NJL_MQ - 4*NJL_G*cu + 2*NJL_K*cd*cs)
    eq_d = m_d - (NJL_MQ - 4*NJL_G*cd + 2*NJL_K*cu*cs)
    eq_s = m_s - (NJL_MS - 4*NJL_G*cs + 2*NJL_K*cu*cd)
    return [eq_u, eq_d, eq_s]

_njl_mass_guess = [80.0, 65.0, 466.0]      # warm-start cache, updated after every solve

def _njl_solve_masses(pFu, pFd, pFs):
   
    global _njl_mass_guess
    pF = [pFu, pFd, pFs]
    guesses_to_try = [_njl_mass_guess, [367.6, 367.6, 549.5], [50.0, 50.0, 460.0], [NJL_MQ, NJL_MQ, NJL_MS]]

    best_sol, best_res = None, np.inf
    for g in guesses_to_try:
        sol = fsolve(_njl_gap_residuals, g, args=(pF,), xtol=1e-10)
        res = np.max(np.abs(_njl_gap_residuals(sol, pF)))
        if res < best_res:
            best_sol, best_res = sol, res
        if res < 1e-6 and np.all(np.array(sol) > -1.0):
            _njl_mass_guess = list(sol)
            return sol
    # nothing converged cleanly -- return the best attempt anyway (best effort)
    _njl_mass_guess = list(best_sol)
    return best_sol

def _njl_B_flavor(m_star, m_current, condensate_val):
    
    return (3.0/(8.0*np.pi**2)) * (_njl_term(NJL_LAMBDA, m_star) - _njl_term(NJL_LAMBDA, m_current)) \
           - 2*NJL_G*condensate_val**2

def _njl_B_total(m_star_u, m_star_d, m_star_s, cu, cd, cs):
  
    Bu = _njl_B_flavor(m_star_u, NJL_MQ, cu)
    Bd = _njl_B_flavor(m_star_d, NJL_MQ, cd)
    Bs = _njl_B_flavor(m_star_s, NJL_MS, cs)
    return Bu + Bd + Bs + 4*NJL_K*cu*cd*cs

# Vacuum reference B0 (eq. 11), computed once at import time.
_vac_masses = fsolve(_njl_gap_residuals, [367.0, 367.0, 549.0], args=([0.0, 0.0, 0.0],), xtol=1e-10)
_vac_cu = _njl_condensate(_vac_masses[0], 0.0)
_vac_cd = _njl_condensate(_vac_masses[1], 0.0)
_vac_cs = _njl_condensate(_vac_masses[2], 0.0)
NJL_B0 = _njl_B_total(_vac_masses[0], _vac_masses[1], _vac_masses[2], _vac_cu, _vac_cd, _vac_cs)
print(f"[NJL] vacuum masses: m*_u={_vac_masses[0]:.1f}  m*_d={_vac_masses[1]:.1f}  m*_s={_vac_masses[2]:.1f} MeV"
      f"  (paper: 367.7, 367.7, 549.5)")
print(f"[NJL] B0^(1/4) = {NJL_B0**0.25:.1f} MeV  (paper: 217.6 MeV)")


_njl_state_cache = {"key": None, "masses": None}

def _njl_masses_for(kFu, kFd, kFs):
    key = (round(float(kFu), 6), round(float(kFd), 6), round(float(kFs), 6))
    if _njl_state_cache["key"] == key:
        return _njl_state_cache["masses"]
    m_stars = _njl_solve_masses(kFu, kFd, kFs)
    _njl_state_cache["key"] = key
    _njl_state_cache["masses"] = m_stars
    return m_stars


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

    m_star_u, m_star_d, m_star_s = _njl_masses_for(kFu, kFd, kFs)
    cu = _njl_condensate(m_star_u, kFu)
    cd = _njl_condensate(m_star_d, kFd)
    cs = _njl_condensate(m_star_s, kFs)

    eps_kin = (3.0*_njl_term(kFu, m_star_u)
               + 3.0*_njl_term(kFd, m_star_d)
               + 3.0*_njl_term(kFs, m_star_s)) / (8.0*np.pi**2)
    B_now = _njl_B_total(m_star_u, m_star_d, m_star_s, cu, cd, cs)
    B_eff = NJL_B0 - B_now

    eps_NJL_natural = eps_kin + B_eff          # MeV^4
    return eps_NJL_natural / (HBAR**3)         # MeV/fm^3 -- same convention as the rest of this file

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

    m_star_u, m_star_d, m_star_s = _njl_masses_for(kFu, kFd, kFs)
    mu_u = np.sqrt(kFu**2 + m_star_u**2)
    mu_d = np.sqrt(kFd**2 + m_star_d**2)
    mu_s = np.sqrt(kFs**2 + m_star_s**2)
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
    r["eps"] = r["eps_total"]; r["x"] = [yp]     # helpers for the main loop
    return r



def solve_quark(nb, guess):
    def eq(v):
        yu, yd, ys, ye = v
        kFu, kFd, kFs = kF_quark(nb, yu), kF_quark(nb, yd), kF_quark(nb, ys)
        kFe = kF_hadlep(nb, max(ye, 1e-12))
        mu_u, mu_d, mu_s = mu_quark(nb, yu, yd, ys, kFu, kFd, kFs)
        mu_e = mu_lepton(kFe, me)
        return [yu + yd + ys - 3.0,
                (2.0*yu - yd - ys)/3.0 - ye,
                mu_d - mu_u - mu_e,
                mu_s - mu_d]

    sol = root(eq, guess, method='hybr', tol=1e-10)
    if not sol.success:
        sol = root(eq, guess, method='lm', tol=1e-10)
        if not sol.success:
            return None
    yu, yd, ys, ye = sol.x
    if min(yu, yd, ys) <= 0 or ye < 0:
        return None
    kFu, kFd, kFs = kF_quark(nb, yu), kF_quark(nb, yd), kF_quark(nb, ys)
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
    if f < 0.0:
        pen += 1e6 * (-f)
    if f > 1.0:
        pen += 1e6 * (f - 1.0)

    try:
        s = calculate_state(nb, yn, yp, ys, ye, f)
        R1 = s['mu_n'] - (s['mu_u'] + 2.0*s['mu_d'])
        R2 = s['mu_p'] - (2.0*s['mu_u'] + s['mu_d'] - eta*(s['mu_eN'] - s['mu_eQ']))
        R3 = s['mu_d'] - (s['mu_u'] + eta*s['mu_eQ'] + (1.0 - eta)*s['mu_eG'])
        R4 = s['mu_d'] - s['mu_s']
        R5 = (s['PN'] + eta*s['PeN']) - (s['PQ'] + eta*s['PeQ'])
        return [R1 + pen, R2 + pen, R3 + pen, R4 + pen, R5 + pen]
    except Exception:
        return [1e10]*5


DEBUG_SOLVER = False        

def _mixed_candidate_from_x(x, nb, eta):

    yn, yp, ys, ye, f = x
    if not (0.0 < f < 1.0):
        return None
    res = np.max(np.abs(mixed_residuals(x, nb, eta)))
    if res > 1e-3:
        return None
    s = calculate_state(nb, yn, yp, ys, ye, f)
    yu, yd = s['yu'], s['yd']
    if min(yn, yp, ys, yu, yd) < -1e-6 or max(yu, yd, ys) > 3.2:
        return None
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
    r["eps"] = eps; r["x"] = list(x)
    return r



_MIX_BOUNDS = ([0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 4.0, 2.0, 1.0])


def solve_mixed(nb, eta, guesses, use_bounded_fallback=False):
    
    for gi, g in enumerate(guesses):
        for method in ('hybr', 'lm'):
            sol = root(mixed_residuals, g, args=(nb, eta), method=method, tol=1e-9)
            if DEBUG_SOLVER:
                res_dbg = np.max(np.abs(mixed_residuals(sol.x, nb, eta)))
                print(f"        try guess#{gi} [{method}]  success={sol.success}"
                      f"  root=[yn={sol.x[0]:.3f} yp={sol.x[1]:.3f} ys={sol.x[2]:.3f}"
                      f" ye={sol.x[3]:.3f} f={sol.x[4]:.3f}]  maxres={res_dbg:.2e}")
            if not sol.success:
                continue
            r = _mixed_candidate_from_x(sol.x, nb, eta)
            if r is not None:
                return r

        if not use_bounded_fallback:
            continue


        g_clipped = np.clip(g, _MIX_BOUNDS[0], _MIX_BOUNDS[1])
        try:
            lsq = least_squares(mixed_residuals, g_clipped, args=(nb, eta),
                                 bounds=_MIX_BOUNDS, xtol=1e-14, ftol=1e-14,
                                 gtol=1e-14, max_nfev=1500)
        except Exception:
            lsq = None
        if lsq is not None:
            if DEBUG_SOLVER:
                res_dbg = np.max(np.abs(mixed_residuals(lsq.x, nb, eta)))
                print(f"        try guess#{gi} [lsq-bounded]  "
                      f"root=[yn={lsq.x[0]:.3f} yp={lsq.x[1]:.3f} ys={lsq.x[2]:.3f}"
                      f" ye={lsq.x[3]:.3f} f={lsq.x[4]:.3f}]  maxres={res_dbg:.2e}")
            r = _mixed_candidate_from_x(lsq.x, nb, eta)
            if r is not None:
                return r
    return None


_MIX_BOUNDARY_EPS = 1e-4


def solve_mixed_continued(nb, nb_prev, eta, g_prev, extra_guesses, depth=0, max_depth=6):

    # already at the edge of the mixed-phase domain -- don't waste time
    # trying to resolve a transition that's numerically already over.
    if g_prev is not None:
        f_prev = g_prev[4]
        if f_prev < _MIX_BOUNDARY_EPS or f_prev > 1.0 - _MIX_BOUNDARY_EPS:
            return None, g_prev, nb_prev

    guesses = ([g_prev] if g_prev is not None else []) + extra_guesses

    m = solve_mixed(nb, eta, guesses, use_bounded_fallback=(depth > 0))
    if m is not None:
        return m, m['x'], nb
    if depth >= max_depth or g_prev is None:
        return None, g_prev, nb_prev
    nb_mid = 0.5*(nb_prev + nb)
    m_mid, g_mid, nb_mid_reached = solve_mixed_continued(
        nb_mid, nb_prev, eta, g_prev, extra_guesses, depth+1, max_depth)
    if g_mid is None:
        return None, g_prev, nb_prev
   
    return solve_mixed_continued(nb, nb_mid_reached, eta, g_mid, extra_guesses,
                                  depth+1, max_depth)



nb_values = np.arange(0.05, 2.00, 0.01)

# warm-start guesses (updated with the last successful solution of each branch)
g_had = 0.9                                   # yp
g_qrk = [.1, .1, 0.1, 0.02]                  # yu, yd, ys, ye
g_mix_prev = None                              # last mixed solution vector
nb_mix_prev = None                             # nB at which g_mix_prev converged

results = []

# Set to True to print every candidate branch (all fractions) at each nB.
VERBOSE = True

print("Starting hybrid EOS solver (eta = %.2f)..." % ETA)

def _print_candidate(nb, c):
    
    ysum_q = c['yu'] + c['yd'] + c['ys']      # should be ~3 wherever quarks exist
    ysum_h = c['yn'] + c['yp']                 # should be ~1 wherever hadrons exist
    print(f"      [{c['phase']:6s}]  f={c['f']:.4f}"
          f"  yn={c['yn']:.4f}  yp={c['yp']:.4f}"
          f"  yu={c['yu']:.4f}  yd={c['yd']:.4f}  ys={c['ys']:.4f}"
          f"  ye={c['ye']:.4f}"
          f"  | (yn+yp)={ysum_h:.3f}  (yu+yd+ys)={ysum_q:.3f}"
          f"  eps={c['eps_total']:8.2f}")

for i, nb in enumerate(nb_values):

    candidates = []

    if VERBOSE:
        print(f"\n=== step {i+1:3d}/{len(nb_values)}  nB={nb:.4f} ===")

    # --- pure hadron ---
    h = solve_hadron(nb, g_had)
    if h:
        candidates.append(h); g_had = h['yp']
        if VERBOSE: _print_candidate(nb, h)

    # --- pure quark ---
    q = solve_quark(nb, g_qrk)
    if q:
        candidates.append(q); g_qrk = q['x']
        if VERBOSE: _print_candidate(nb, q)

    cold_guesses = [
        [0.90, 0.06, 0.05, 0.05, 0.80],
        [0.70, 0.10, 0.30, 0.03, 0.50],
        [0.50, 0.12, 0.60, 0.01, 0.30],
        [0.30, 0.10, 0.90, 0.00, 0.15],
    ]
    if g_mix_prev is not None and nb_mix_prev is not None:
        m, g_from_continuation, nb_reached = solve_mixed_continued(
            nb, nb_mix_prev, ETA, g_mix_prev, cold_guesses)
        if g_from_continuation is not None:
            g_mix_prev = g_from_continuation
            nb_mix_prev = nb_reached
    else:
        m = solve_mixed(nb, ETA, cold_guesses)
    if m:
        candidates.append(m); g_mix_prev = m['x']; nb_mix_prev = nb
        if VERBOSE: _print_candidate(nb, m)

    if not candidates:
        print(f"{i+1:3d}/{len(nb_values)}  nB={nb:.3f}  NO SOLUTION")
        continue

    # thermodynamic selection: lowest energy density wins
    best = min(candidates, key=lambda c: c['eps'])
    best["nB"] = nb
    results.append({c: best[c] for c in COLUMNS})     # full row, fixed column order

    print(f"{i+1:3d}/{len(nb_values)}  nB={nb:.3f}  -> {best['phase']:6s}"
          f"  f={best['f']:.3f}  ys={best['ys']:.3f}"
          f"  eps={best['eps_total']:8.2f}  P={best['P_total']:8.2f}")

# ==========================================================
# 12. Save
# ==========================================================
if results:
    df = pd.DataFrame(results, columns=COLUMNS)

    out = r"/home/khushbu/Desktop/khushbu/phD/EOS codes/max+gibs(paper)/final_eos_solved_output_NJL.xlsx"

    # Give the sheet any name you want
    sheet_name = "eta_1.0_NJL"      # Change to Run2, Run3, ETA06, Bag165, etc.

    if os.path.exists(out):
        # File already exists -> add a new sheet
        with pd.ExcelWriter(out, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # First time creating the file
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nDone. {len(results)} points saved in sheet '{sheet_name}' of {out}")