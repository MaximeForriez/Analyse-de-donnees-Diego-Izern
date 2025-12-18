# coding: utf8

print('Bienvenue dans le cours d\'analyse de données en géographie !')

import numpy as np
import pandas as pd
import geopandas as gdp
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from scipy import stats


# Test de pandas
data = pd.DataFrame({'A': [1, 2, 3]})
print(data)

# https://docs.scipy.org/doc/scipy/reference/stats.html


def plot_discrete_distributions():
    # Paramètres pour les distributions
    a = 0
    n = 10
    n_binomial, p_binomial = 20, 0.5
    mu_poisson = 5
    a_zipf = 1.5

    # Création de la figure
    plt.figure(figsize=(15, 10))

    # Loi de Dirac
    x_dirac = [a]
    y_dirac = [1]
    plt.subplot(2, 3, 1)
    plt.stem(x_dirac, y_dirac)
    plt.title('Loi de Dirac')
    plt.xlabel('x')
    plt.ylabel('P(X = x)')

    # Loi uniforme discrète
    x_uniform = np.arange(1, n + 1)
    y_uniform = np.ones(n) / n
    plt.subplot(2, 3, 2)
    plt.stem(x_uniform, y_uniform)
    plt.title('Loi uniforme discrète')
    plt.xlabel('x')
    plt.ylabel('P(X = x)')

    # Loi binomiale
    x_binomial = np.arange(n_binomial + 1)
    y_binomial = stats.binom.pmf(x_binomial, n_binomial, p_binomial)
    plt.subplot(2, 3, 3)
    plt.stem(x_binomial, y_binomial)
    plt.title('Loi binomiale (n=20, p=0.5)')
    plt.xlabel('x')
    plt.ylabel('P(X = x)')

    # Loi de Poisson
    x_poisson = np.arange(0, 15)
    y_poisson = stats.poisson.pmf(x_poisson, mu_poisson)
    plt.subplot(2, 3, 4)
    plt.stem(x_poisson, y_poisson)
    plt.title('Loi de Poisson (λ=5)')
    plt.xlabel('x')
    plt.ylabel('P(X = x)')

    # Loi de Zipf-Mandelbrot
    x_zipf = np.arange(1, 11)
    y_zipf = 1 / (x_zipf ** a_zipf)
    y_zipf /= y_zipf.sum()
    plt.subplot(2, 3, 5)
    plt.stem(x_zipf, y_zipf)
    plt.title('Loi de Zipf-Mandelbrot (a=1.5)')
    plt.xlabel('x')
    plt.ylabel('P(X = x)')

    plt.tight_layout()
    plt.savefig('distributions_discretes.png', dpi=300, bbox_inches='tight')

def plot_continuous_distributions():
    # Paramètres pour les distributions
    mu_poisson_cont, size_poisson_cont = 5, 1000
    mu_normal, sigma_normal = 0, 1
    sigma_lognormal, mu_lognormal, size_lognormal = 0.5, 0, 1000
    a_uniform, b_uniform = 0, 1
    df_chi2 = 5
    b_pareto, size_pareto = 2, 1000

    # Création de la figure
    plt.figure(figsize=(15, 10))

    # Loi de Poisson continue
    x_poisson_cont = np.random.poisson(mu_poisson_cont, size_poisson_cont)
    plt.subplot(2, 3, 1)
    plt.hist(x_poisson_cont, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
    plt.title('Loi de Poisson (λ=5)')
    plt.xlabel('x')
    plt.ylabel('Densité')

    # Loi normale
    x_normal = np.linspace(-4, 4, 1000)
    y_normal = stats.norm.pdf(x_normal, mu_normal, sigma_normal)
    plt.subplot(2, 3, 2)
    plt.plot(x_normal, y_normal, 'r-', lw=2)
    plt.fill_between(x_normal, y_normal, alpha=0.3, color='r')
    plt.title('Loi normale (μ=0, σ=1)')
    plt.xlabel('x')
    plt.ylabel('Densité')

    # Loi log-normale
    x_lognormal = np.random.lognormal(mu_lognormal, sigma_lognormal, size_lognormal)
    plt.subplot(2, 3, 3)
    plt.hist(x_lognormal, bins=30, density=True, alpha=0.6, color='b', edgecolor='black')
    plt.title('Loi log-normale (μ=0, σ=0.5)')
    plt.xlabel('x')
    plt.ylabel('Densité')

    # Loi uniforme
    x_uniform = np.random.uniform(a_uniform, b_uniform, 1000)
    plt.subplot(2, 3, 4)
    plt.hist(x_uniform, bins=30, density=True, alpha=0.6, color='m', edgecolor='black')
    plt.title('Loi uniforme [0, 1]')
    plt.xlabel('x')
    plt.ylabel('Densité')

    # Loi du χ²
    x_chi2 = np.random.chisquare(df_chi2, 1000)
    plt.subplot(2, 3, 5)
    plt.hist(x_chi2, bins=30, density=True, alpha=0.6, color='c', edgecolor='black')
    plt.title('Loi du χ² (df=5)')
    plt.xlabel('x')
    plt.ylabel('Densité')

    # Loi de Pareto
    x_pareto = np.random.pareto(b_pareto, size_pareto) + 1
    plt.subplot(2, 3, 6)
    plt.hist(x_pareto, bins=30, density=True, alpha=0.6, color='y', edgecolor='black')
    plt.title('Loi de Pareto (b=2)')
    plt.xlabel('x')
    plt.ylabel('Densité')

    plt.tight_layout()
    plt.savefig('distributions_continues.png', dpi=300, bbox_inches='tight')

def calculate_mean_std(distribution, *params):
    if distribution == 'dirac':
        a = params[0]
        return a, 0
    
    elif distribution == 'uniform_discrete':
        n = params[0]
        mean = (n + 1) / 2
        variance = (n**2 - 1) / 12
        std = np.sqrt(variance)
        return mean, std
    
    elif distribution == 'binomial':
        n, p = params[0], params[1]
        mean = n * p
        variance = n * p * (1 - p)
        std = np.sqrt(variance)
        return mean, std
    
    elif distribution == 'poisson':
        mu = params[0]
        return mu, np.sqrt(mu)
    
    elif distribution == 'zipf':
        a = params[0]
        return np.nan, np.nan
    
    elif distribution == 'poisson_cont':
        mu = params[0]
        return mu, np.sqrt(mu)
    
    elif distribution == 'normal':
        mu, sigma = params[0], params[1]
        return mu, sigma
    
    elif distribution == 'lognormal':
        mu, sigma = params[0], params[1]
        mean = np.exp(mu + (sigma**2)/2)
        variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
        std = np.sqrt(variance)
        return mean, std
    
    elif distribution == 'uniform':
        a, b = params[0], params[1]
        mean = (a + b) / 2
        variance = (b - a)**2 / 12
        std = np.sqrt(variance)
        return mean, std
    
    elif distribution == 'chi2':
        df = params[0]
        mean = df
        variance = 2 * df
        std = np.sqrt(variance)
        return mean, std
    
    elif distribution == 'pareto':
        b = params[0]
        if b > 1:
            mean = b / (b - 1)
        else:
            mean = np.inf
        if b > 2:
            variance = b / ((b - 1)**2 * (b - 2))
            std = np.sqrt(variance)
        else:
            std = np.inf
        return mean, std


if __name__ == "__main__":   
    # Génération des graphiques
    plot_discrete_distributions()
    plot_continuous_distributions()
    
    # Calcul des statistiques   
    distributions = [
        ("Loi de Dirac (a=0)", 'dirac', [0]),
        ("Loi uniforme discrète (n=10)", 'uniform_discrete', [10]),
        ("Loi binomiale (n=20, p=0.5)", 'binomial', [20, 0.5]),
        ("Loi de Poisson (μ=5)", 'poisson', [5]),
        ("Loi de Zipf-Mandelbrot (a=1.5)", 'zipf', [1.5]),
        ("Loi de Poisson continue (μ=5)", 'poisson_cont', [5]),
        ("Loi normale (μ=0, σ=1)", 'normal', [0, 1]),
        ("Loi log-normale (μ=0, σ=0.5)", 'lognormal', [0, 0.5]),
        ("Loi uniforme (a=0, b=1)", 'uniform', [0, 1]),
        ("Loi du χ² (df=5)", 'chi2', [5]),
        ("Loi de Pareto (b=2)", 'pareto', [2])
    ]
    
    for name, dist_type, params in distributions:
        mean, std = calculate_mean_std(dist_type, *params)
        print(f"{name:40s} → Moyenne: {mean:8.4f}, Écart type: {std:8.4f}")
    
  