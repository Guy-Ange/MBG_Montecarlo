import numpy as np
import matplotlib.pyplot as plt

from .simuler_modele import simuler_ST




def payoff_call(ST, K):
    """Calcule le payoff d'un Call Européen."""
    return np.maximum(ST - K, 0)

def simulate_ST(S0, T, r, sigma, Z):
    """Calcule le prix de l'actif à l'échéance S_T selon le modèle B-S."""
    # Z est le tirage aléatoire standard (N(0, 1))
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def price_mc_antithetic_call(S0, K, T, r, sigma, N_pairs,seed = None):
    """Estimateur Variables Antithétiques (VA) - utilise N_pairs."""
    # 1. Tirage et paire antithétique

    rng = np.random.default_rng(seed)
    Z = rng.normal(size=N_pairs)
    Z_anti = -Z
    
    # 2. Simulation S_T
    ST_pos = simulate_ST(S0, T, r, sigma, Z)
    ST_neg = simulate_ST(S0, T, r, sigma, Z_anti)
    
    # 3. Moyenne des paires de Payoff
    P_pos = payoff_call(ST_pos, K)
    P_neg = payoff_call(ST_neg, K)
    
    return np.exp(-r * T) * np.mean((P_pos + P_neg) / 2)

