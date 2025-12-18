"""Ici on simule le modèle stochastique suivi du marché 
qui est un mouvement Brownien géométrique qui sera ensuite 
utilisé pour calculer le prix d'une option européenne

On s'assure aussi que le générateur de trajectoires simuler_ST produit bien 
une distribution Log-Normale """

import numpy as np

import matplotlib.pyplot as plt

def simuler_ST(S0,r,sigma,T,N,seed = None):
    """Sous la mesure risque neutre, on simule la valeur 
    du sous-jacent à l'échéance T en choisissant un nombre au hasard
    suivant la loi normale"""

    # 1. Création de l'objet générateur de nombres aléatoires
    #    Si 'seed' est None, un seed aléatoire est utilisé.
    rng = np.random.default_rng(seed)
    
    # 2. Utilisation de l'objet générateur pour tirer le nombre selon la loi normale
    Z = rng.normal(size=N) 
    
    # 3. Calcul de S_T 
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    return ST
#---------------------------------------------------------------------------------------------------

def plot_histogram(ST, S0, r, sigma, T):
    """
    Histogramme des prix finaux simulés (S_T) superposé à la densité théorique.
    
    Cette fonction valide visuellement que les trajectoires Monte Carlo suivent 
    la distribution Log-Normale prédite par le modèle GBM.
    """
    fig, ax = plt.subplots(figsize=(8,5))
    
    # 1. Histogramme de l'Échantillon Monte Carlo (Distribution Empirique)
    counts, bins, patches = ax.hist(
        ST,          # Le vecteur des N prix S_T simulés
        bins=120,    # Nombre de barres (assure une bonne granularité)
        density=True, # Normalise l'histogramme pour que l'aire totale soit 1 (crucial pour la comparaison avec la PDF)
        alpha=0.7    # Transparence
    )
    
    # Configuration du graphique
    ax.set_title(r'Histogramme de $S_T$ (simulé) et densité log-normale théorique')
    ax.set_xlabel('$S_T$')
    ax.set_ylabel('Densité (norm.)')

    # 2. Densité Log-Normale Théorique
    
    # Points d'échantillonnage pour un tracé lisse de la courbe théorique
    xs = np.linspace(max(0.001, ST.min()), ST.max(), 400)
    
    # Calcul des deux paramètres de la distribution Log-Normale pour S_T:
    # mu_ln (moyenne logarithmique) : E[ln(S_T)] = ln(S0) + (r - 0.5 * sigma^2) * T
    mu_ln = np.log(S0) + (r - 0.5 * sigma**2) * T
    
    # sigma_ln (écart-type logarithmique) : Std[ln(S_T)] = sigma * sqrt(T)
    sigma_ln = sigma * np.sqrt(T)
    
    # Formule analytique de la Fonction de Densité de Probabilité (PDF) Log-Normale
    # C'est la référence théorique du modèle GBM.
    pdf_log = (1.0 / (xs * sigma_ln * np.sqrt(2 * np.pi))) * \
              np.exp(- (np.log(xs) - mu_ln)**2 / (2 * sigma_ln**2))
              
    # 3. Superposition
    # Trace la courbe théorique sur l'histogramme simulé.
    # Un alignement parfait est la preuve que la simulation est bien implémentée. 
    ax.plot(xs, pdf_log, linewidth=1.5, color='red', label='Densité Théorique')
    ax.legend() 
    
    plt.tight_layout()
    plt.show()