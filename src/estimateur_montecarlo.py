"""On calcul l'estimateur montecarlo du prix du call 
pour un nombre donnée N  de simulations  puis on quantifie l'erreur
statistique par l'ecart-type"""

# -----------------------------------------------------------------
# 1. Imports des bibliothèques standard et tierces (NumPy, SciPy)
# ------------------------------------------------------------------
import numpy as np
from scipy.stats import norm 
# ------------------------------------------------------------------
# 2. Imports locaux (src)
# -----------------------------------------------------------------
from .simuler_modele import simuler_ST

def prix_montecarlo_call(S0, K, r, sigma, T, N, seed=None):

    # 1. SIMULATION DU PRIX A l'ECHEANCE

    ST = simuler_ST(S0, r, sigma, T, N, seed)

    # 2. CALCUL DU PAYOFF ET DU PRIX ESTIMÉ (Loi des Grands Nombres)
    
    payoff = np.maximum(ST - K, 0)
    
    # Prix Monte Carlo (Moyenne des Payoffs actualisés)
    facteur_actualisation = np.exp(-r * T)
    prix = facteur_actualisation  * np.mean(payoff)
    
    # 3. QUANTIFICATION DE L'ERREUR (Théorème Central Limite - TCL)
    
    # Écart-type du payoff (ddof=1 pour estimateur non biaisé de la variance)
    std_payoff = np.std(payoff, ddof=1)
    
    # Demi-largeur de l'Intervalle de Confiance 95% (basé sur le TCL)
    # La formule est : Z_score * (std_Payoff / sqrt(N)) * Facteur_Actualisation
    Z_score = 1.96  # Pour un IC à 95%
    std_error = std_payoff / np.sqrt(N)
    demi_largeur_IC = Z_score * std_error * facteur_actualisation 
    
    return prix, demi_largeur_IC


#--------------------------------------------------------------------------------------------


def vrai_prix_call(S0, K, r, sigma, T):
    """
    Formule analytique (fermée) de Black-Scholes-Merton pour le prix d'un Call Européen.
    
    Ce prix sert de VRAIE VALEUR THÉORIQUE pour valider la méthode Monte Carlo.
    """
    
    # 1. Condition à l'Échéance
    if T <= 0:
        # Si le temps restant (T) est nul ou passé, la valeur est le payoff intrinsèque immédiat.
        # Cela correspond à la condition aux limites de l'EDP de Black-Scholes.
        return np.maximum(S0 - K, 0)
    
    # --- Calcul des paramètres d1 et d2 ---
    
    # d1 est une mesure de probabilité ajustée par la volatilité. 
    # Le terme (r + 0.5 * sigma**2) est le drift ajusté du processus d'Itô.
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # d2 est lié à d1. d1 et d2 sont les bornes d'intégration des probabilités.
    # d2 = d1 - sigma * np.sqrt(T) est la relation clé dans la formule BSM.
    d2 = d1 - sigma * np.sqrt(T)
    
    # 3. Formule de Pricing de Black-Scholes (Portefeuille de Réplication)
    
    # Le prix du Call est donné par la différence entre :
    # S0 * N(d1) : La valeur actuelle de l'espérance des actions reçues si l'option est exercée.
    # K * exp(-r * T) * N(d2) : La valeur actuelle de l'espérance du prix d'exercice payé.
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)