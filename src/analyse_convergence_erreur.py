"""Ici on visualise graphiquement la convergence de l'estimateur
précédemment calculé vers la solution analytique 

On teste  egalement la vitesse de convergence de l'erreur entre l'estimateur et la solution 
réelle pour valider la convergence en O(N^{-1/2}) 

On réaffirme encore cette vitesse par l'evolution de la demi-largeur de l'intervalle de confiance
en fonction du nombre de simulation N """

# -----------------------------------------------------------------
# 1. Imports des bibliothèques standard et tierces (NumPy, SciPy)
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------------
# 2. Imports locaux (src)
# -----------------------------------------------------------------
from .estimateur_montecarlo import prix_montecarlo_call, vrai_prix_call 



def plot_convergence_RMSE(S0, K, r, sigma, T, Ns, M_repetitions=1,seed = None):
    """
    on génère les graphiques de convergence de l'estimateur Monte Carlo.
    
    Graphique 1 (Gauche) : Convergence de l'estimateur vers le prix théorique.
    Graphique 2 (Droite) : Validation de la convergence en O(1/sqrt(N)) sur une échelle log-log.
    
    :param Ns: Liste des nombres de simulations N à tester.
    :param M_repetitions: Nombre de répétitions pour lisser l'erreur.
                           Si M_repetitions=1 (défaut), on trace l'Erreur Absolue.
                           Si M_repetitions>1, on trace la RMSE (Root Mean Squared Error).
    """
    
    # Prix de référence analytique
    prix_vrai = vrai_prix_call(S0, K, r, sigma, T)
    
    estimates_moyennes = []  # Pour stocker les moyennes de prix (si M>1) ou le prix unique (si M=1)
    erreurs_a_tracer = []    # Pour stocker l'Erreur Absolue ou la RMSE
    
    # --- 1. Boucle de Simulation (Calculer la métrique d'erreur pour chaque N) ---
    for n in Ns:
        
        erreurs_quadratiques = [] # Stocke les erreurs au carré pour calculer la moyenne (RMSE)
        prix_estimations = []     # Stocke M prix pour calculer la moyenne (Graphique 1)
        
        # Boucle interne pour M_repetitions (Lissage)
        for j in range(M_repetitions):

            specific_seed = seed + n + j # pour s'assurer de l'independance
            
            # Obtient une seule estimation de prix pour ce N
            prix_estime, _ = prix_montecarlo_call(S0, K, r, sigma, T, n, seed=specific_seed) 
            
            # Stocke le prix pour le Graphique 1
            prix_estimations.append(prix_estime)
            
            # Calcule l'erreur quadratique pour le lissage
            erreur_quad = (prix_estime - prix_vrai)**2
            erreurs_quadratiques.append(erreur_quad)
            
        # --- Stockage des résultats pour ce N ---
        
        # Pour le Graphique 1 (Convergence de l'estimateur): 
        # On prend la moyenne des M estimations pour ce point N (plus stable)
        estimates_moyennes.append(np.mean(prix_estimations))
        
        # Pour le Graphique 2 (Ordre de Convergence):
        if M_repetitions > 1:
            # Calcul de la RMSE (Root Mean Squared Error)
            MSE = np.mean(erreurs_quadratiques)
            RMSE = np.sqrt(MSE)
            erreurs_a_tracer.append(RMSE)
            label_erreur = 'RMSE observée'
        else:
            # Si M=1, on utilise l'Erreur Absolue de la seule estimation
            # Note : On prend la racine carrée de la seule erreur_quad, ce qui donne l'erreur absolue.
            # L'erreur_quadratique contient un seul élément à la fin de la boucle si M=1.
            # On prend la première (et unique) valeur stockée.
            erreurs_a_tracer.append(np.sqrt(erreurs_quadratiques[0]))
            label_erreur = 'Erreur absolue'
            
    estimateurs = np.array(estimates_moyennes)
    erreurs_observées = np.array(erreurs_a_tracer)


    # --- 2. Construction des Graphiques (Double Plot) ---
    
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    
    # --- Graphique 1 : Convergence de l'Estimateur (À Gauche) ---
    
    ax[0].plot(Ns, estimateurs, marker='o', linestyle='-')
    ax[0].axhline(prix_vrai, linestyle='--', linewidth=1, label='Black–Scholes', color='red')
    ax[0].set_xscale('log') 
    ax[0].set_xlabel('Nombre de simulations (N)')
    ax[0].set_ylabel('Estimateur MC du prix')
    ax[0].set_title(f'Convergence de l\'estimateur MC (M={M_repetitions} répétitions)')
    ax[0].legend()

    # --- Graphique 2 : Validation de l'Ordre de Convergence (À Droite - Log-Log) ---
    
    # Création de la droite de référence théorique O(N^-1/2)
    # RefN doit couvrir les limites du tracé.
    refN = np.array([Ns[0], Ns[-1]])
    
    # Cref est calé sur la première erreur observée (RMSE ou Erreur Absolue)
    Cref = erreurs_observées[0] * (refN[0]**0.5) 
    
    # Tracé de l'erreur observée (RMSE ou Erreur Absolue)
    ax[1].plot(Ns, erreurs_observées, marker='o', linestyle='-', label=label_erreur)
    
    # Tracé de la référence théorique O(N^-1/2). 
    ax[1].plot(refN, Cref * refN**(-0.5), linestyle='--', linewidth=1, color='red', label=r'Ref $\propto N^{-1/2}$')
    
    ax[1].set_xscale('log') 
    ax[1].set_yscale('log')
    
    ax[1].set_xlabel('N (log scale)')
    ax[1].set_ylabel('Erreur (log scale)')
    ax[1].set_title(f'{label_erreur} vs N (log-log)')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

