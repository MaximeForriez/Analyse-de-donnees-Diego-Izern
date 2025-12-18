#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv","r") as fichier:
    contenu = pd.read_csv(fichier)


# Affichage variable contenu
print (contenu.columns)

# Calcul avec la fonction native len
nb_lignes = len(contenu)
nb_colonnes = len(contenu.columns)
print(f"Nombre de lignes : {nb_lignes}")
print(f"Nombre de colonnes : {nb_colonnes}")

# Liste sur le type de chaque colonne
print("Types de variables")
contenu.info()

print("Liste des types")
print(contenu.dtypes)

# Affichage du nom des colonnes
print("Nom des colonnes : ")
print(contenu.columns)

# Sélection du nombre des inscrits
print("Nombre d'inscrits aux élections présidentielles de 2018 ", contenu['Inscrits'].sum())

# Calcul des données quantitatives

print(contenu.dtypes)

sommes = []
for colonne in contenu.columns:
    if contenu[colonne].dtype in ['int64','float64']:  # Condition pour colonnes quantitatives
        sommes.append(contenu[colonne].sum())

print("Sommes des colonnes quantitatives :", sommes)

# Création de diagramme en barre avec le nombre des inscrits et des votants pour chaque département


# Parcourir chaque département et créer un diagramme
for index, row in contenu.iterrows():
    departement = row['Libellé du département']  # Ajustez le nom si différent
    inscrits = row['Inscrits']
    votants = row['Votants']  # Assurez-vous que cette colonne existe et est numérique
    
    # Création du diagramme en barres
    categories = ['Inscrits', 'Votants']
    valeurs = [inscrits, votants]
    
    plt.figure(figsize=(8, 6))
    plt.bar(categories, valeurs, color=['blue', 'green'])
    plt.title(f'Inscrits et Votants {departement}')
    plt.ylabel('Nombre de personnes')
    plt.xlabel('Catégorie')
    
    # Sauvegarde
    nom_fichier = f'graphiques/{departement}.png'
    plt.savefig(nom_fichier)
    plt.close()

print("Diagrammes en barre créés")

# Réalisation d'un diagramme circulaire
import os

# Création du dossier pour les diagrammes circulaires
os.makedirs('graphiques_circulaires', exist_ok=True)

# Parcourir les départements
for index, row in contenu.iterrows():
    departement = row['Libellé du département']
    
    # Récupération des données
    inscrits = row['Inscrits']
    votants = row['Votants']
    blancs = row['Blancs']
    nuls = row['Nuls']
    exprimes = row['Exprimés']
    
    # Calculer l'abstention
    abstention = inscrits - votants
    
    # Créer les données pour le diagramme circulaire
    categories = ['Abstention', 'Blancs', 'Nuls', 'Exprimés']
    valeurs = [abstention, blancs, nuls, exprimes]
    couleurs = ['lightgray', 'white', 'red', 'green']
    
    # Créer le diagramme circulaire
    plt.figure(figsize=(10, 8))
    plt.pie(valeurs, labels=categories, autopct='%1.1f%%', colors=couleurs, startangle=90)
    plt.title(f'Répartition des votes - {departement}')
    
    # Sauvegarder le diagramme
    nom_fichier = f'graphiques_circulaires/{departement}_circulaire.png'
    plt.savefig(nom_fichier, bbox_inches='tight')
    plt.close()

print("Diagrammes circulaires créés")

# Réalisation d'un histogramme des inscrits
plt.figure(figsize=(10, 6))
plt.hist(contenu['Inscrits'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution des inscrits par département')
plt.xlabel('Nombre d\'inscrits')
plt.ylabel('Fréquence (nombre de départements)')
plt.grid(axis='y', alpha=0.3)

# Sauvegarder l'histogramme
plt.savefig('histogramme_inscrits.png', bbox_inches='tight')
plt.show()

print("Histogramme de la distribution des inscrits créé")








