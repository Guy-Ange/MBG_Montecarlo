"""On veut s'assurer  ici de la cohérence statistique en utilisant 
l'intervalle de confiance à 95%. On calcule un grand nombre d'intervalle
de confiance et on valide que environ 95% d'eux contiennent le vrai prix
de l'option
"""


# -----------------------------------------------------------------
# 1. Imports des bibliothèques standard
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------------
# 2. Imports locaux (src)
# -----------------------------------------------------------------
from .estimateur_montecarlo import vrai_prix_call,prix_montecarlo_call

def monte_carlo_couverture_IC(S0, K, r, sigma, T, N, M, seed =None):
    """
    Répète la simulation Monte Carlo M fois pour tester le taux de couverture de l'IC.
    
    :param N: Nombre de simulations (trajectoires) par estimation.
    :param M: Nombre de répétitions (expériences) du test.
    :return: Taux de couverture empirique, et le prix analytique.
    """
    # Prix analytique (la "vraie" valeur)
    C_BS = vrai_prix_call(S0, K, r, sigma, T)

    estimateurs= [] # contient les prix estimés pour chaque expérience
    demi_largeur_ICs= [] # contient les demi-largeur estimés pour chaque expérience
    
    # Compteur pour le nombre de fois où l'IC contient le vrai prix C_BS
    compteur_Cbs_dans_IC = 0
    
    # --- Répétition de l'Expérience M fois (Protocole du Test) ---
    for j in range(M):

        seed_j = seed + j # pour rendre chaque expérience independant
        
        prix,demi_largeur_IC = prix_montecarlo_call(S0, K, r, sigma, T, N, seed_j)

        # Stockage des données pour le graphique
        estimateurs.append(prix)
        demi_largeur_ICs.append(demi_largeur_IC)
        
        # 1. Construction de l'Intervalle de Confiance
        IC_lower = prix - demi_largeur_IC
        IC_upper = prix + demi_largeur_IC


        
        # 2. Test de Couverture (Vérifier si le prix analytique est contenu dans l'IC)
        if IC_lower <= C_BS <= IC_upper:
            compteur_Cbs_dans_IC  += 1

            
    # --- Calcul du Taux de Couverture ---
    taux_couverture = compteur_Cbs_dans_IC  / M
    
    return taux_couverture, C_BS, np.array(estimateurs), np.array(demi_largeur_ICs)


def plot_couverture_IC(taux_couverture, C_BS, N, estimateurs, demi_largeur_ICs):
    """
    Trace les résultats du test de couverture de l'IC.
    
    :param N: Nombre de trajectoire simulées.
    :param taux_couverture: Taux de couverture empirique.
    :param C_BS: Prix analytique de référence.
    :param estimateurs: Tableau des prix estimés pour M répétitions.
    :param demi_largeur_ICs: Tableau des demi-largeurs d'IC pour M répétitions.
    """
    M = len(estimateurs)
    indices = np.arange(M)
    niveau_confiance = 0.95 # Niveau théorique (ex: 0.95)
    
    # Déterminer les IC qui NE couvrent PAS la vraie valeur
    # ~ (tilde) est l'opérateur NOT
    non_couvert = ~((estimateurs - demi_largeur_ICs <= C_BS) & (C_BS <= estimateurs + demi_largeur_ICs))# on veut montrer la rarete de ceux qui sont pas couvert
    
    fig, ax = plt.subplots(figsize=(12, 6))

    # Ligne de référence (Prix BSM analytique)
    ax.axhline(C_BS, color='k', linestyle='-', linewidth=3, 
               label=f'Prix Analytique $C_{{0}}$: {C_BS:.4f}')

    # --- 1. Tracé des Succès (Barres d'Erreur vertes) ---
    ax.errorbar(
        indices[~non_couvert],                  # Indices où la couverture est réussie
        estimateurs[~non_couvert],             
        yerr=demi_largeur_ICs[~non_couvert],        # Demi-largeur
        fmt='o',                                   
        color='lightgreen',                                 
        capsize=4,
        label=f'IC Couverture Reussie',
    )
    
    # --- 2. Tracé des Échecs (Barres d'Erreur rouge) ---
    # Ces points (environ 5% du total) sont ceux qui valident le test.
    ax.errorbar(
        indices[non_couvert],                  # Indices où la couverture ÉCHOUE
        estimateurs[non_couvert],              
        yerr=demi_largeur_ICs[non_couvert],         
        fmt='o',
        color='r',                                
        capsize=4,
        label=f'IC Echec de Couverture ({non_couvert.sum()})',
        )
    
    # 3. Mise en forme
    ax.set_title(f"Validation Statistique de l\'Intervalle de Confiance (M={M} Repetitions, N={N})\n Taux de Couverture Observe: {taux_couverture:.2%} vs. Theorique: {niveau_confiance:.2%}")
    ax.set_xlabel("Numero de l\'Experience MC Repetee ($j$)")
    ax.set_ylabel("Estimation de Prix $\hat{C}_N^{(j)}$")
    ax.set_xlim(-1, M)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    #


# # ---  Exécution du Test ---

# # Paramètres de l'Option
# S0 = 100.0  
# K = 100.0   
# r = 0.05    
# sigma = 0.2 
# T = 1.0     

# # Paramètres du Test Statistique
# N_simulations = 10**5   # N: Nombre de simulations par estimation (doit être grand)
# M_repetitions = 1000    # M: Nombre de fois où l'on répète l'expérience (pour avoir une bonne statistique)
# alpha_level = 0.05      # Niveau de signification : 5% (donc IC à 95%)

# print(f"--- Test de Couverture de l'Intervalle de Confiance (IC {int((1-alpha_level)*100)}%) ---")
# print(f"N (Simulations par IC) : {N_simulations}")
# print(f"M (Répétitions du test) : {M_repetitions}")

# taux_obs, C_BS = monte_carlo_couverture_IC(S0, K, r, sigma, T, N_simulations, M_repetitions)

# print(f"\nPrix analytique (C_BS) : {C_BS:.6f}")
# print(f"Taux de Couverture Théorique : {1 - alpha_level:.2%}")
# print(f"Taux de Couverture Empirique observé : {taux_obs:.2%}")

# # --- Interprétation du Résultat ---
# if np.isclose(taux_obs, 1 - alpha_level, atol=0.02): # Tolérance de 2%
#     print("\nCONCLUSION : La validation est réussie. Le taux de couverture est conforme à la théorie.")
# else:
#     print("\nCONCLUSION : Avertissement. Le taux de couverture s'écarte significativement de la théorie (95%).")