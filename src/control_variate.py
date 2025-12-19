"""On calcule le prix en utilisant la methode de variable de contrôle pour reduire la 
variance """
import numpy as np
from .simuler_modele import simuler_ST

def prix_mc_control_variate_call(S0, K, r, sigma, T, N_paths, C_BS,seed = None):
    """
    Estimateur Monte Carlo avec Variables de Contrôle (VC).
    
    X : Payoff Call Européen (notre cible)
    Y : Payoff Call Européen (notre variable de contrôle)
    E[Y] : C_BS_ref (le prix analytique)
    """
    # 1. Tirage des variables aléatoires
    
    ST = simuler_ST(S0,r,sigma,T,N_paths,seed = seed)

    # 2. Calcul du Payoff (X) et de la Variable de Contrôle (Y)
    # Dans ce cas simple, X = Y.

    X = np.maximum(ST - K, 0) # la variable cible dont on veut reduire la variance
    Y = np.maximum(ST - K, 0) # la variable de contrôle par laquelle on ajuste l'erreur
    
    facteur_actualisation = np.exp(-r * T)
    
    X_actualise = facteur_actualisation * X
    Y_actualise = facteur_actualisation * Y
    
    # 3. Estimation de l'Estimateur MC standard pour X et Y

    C_mc_X = np.mean(X_actualise)
    C_mc_Y = np.mean(Y_actualise)
    
    # 4. Estimation du Coefficient Optimal Alpha^star
    # Alpha = Cov(X, Y) / Var(Y)
    
    # L'estimation doit se faire sur les réalisations non actualisées (Payoffs)
    # L'actualisation est une constante (np.exp(-r*T)), donc elle ne change pas alpha
    alpha_hat = np.cov(X, Y)[0, 1] / np.var(Y)
    
    # 5. Construction de l'Estimateur Corrigé
    # C_cv = C_hat - alpha_hat * (Y_hat - E[Y])
    C_cv = C_mc_X - alpha_hat * (C_mc_Y - C_BS)
    
    # Si X = Y, théoriquement alpha_star = 1.0. 
    # Le calcul d'alpha_hat doit être très proche de 1.0.
    
    return C_cv, C_mc_X, alpha_hat