import json
import math
import string

import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from gensim.models import Word2Vec

import matplotlib.pyplot as plt
import os

# Fonction permettant de récupérer les mots les plus cités dans un corpus
def motsLesPlusUtilisesData(fichier, nbMots, affichage=True, enregistrement=False, nameEnregistrement='graphique.html'):
    """
    :param fichier: fichier à traiter
    :param nbMots: le nombre de mots que l'on veut dans la sortie de la fonction
    :param affichage: boolean qui indique si l'on veut afficher nos résultats à la fin de la fonction
    :param enregistrement: boolean qui indique si l'on veut enregistrer nos résultats à la fin de la fonction
    :param nameEnregistrement: nom du fichier d'enregistrement
    :return: retourne le dictionnaire des mots avec leurs occurrences ainsi que le dictionnaire permettant la création du dataframe
    """

    # name of the lightest file
    file_name = fichier

    # open and load file
    f = open(file_name, 'r', encoding='utf-8')
    data = json.loads(f.read())
    f.close()

    # parcours du fichier
    dico = {}
    for d in data["data-all"]:
        for mois in data["data-all"][d]:
            for article in data["data-all"][d][mois]:
                for etape in data["data-all"][d][mois][article]:
                    for name in etape:
                        if name == 'kws':
                            for mot in etape[name]:
                                if mot.lower() in dico:
                                    dico[mot.lower()] += etape[name][mot]
                                else:
                                    dico[mot.lower()] = etape[name][mot]

    # Triez le dictionnaire par ordre croissant de ses valeurs en place
    dico = dict(sorted(dico.items(), key=lambda item: item[1], reverse=True))

    compteur = 0
    dicoFinal = {}
    # Parcours du dictionnaire
    for cle, valeur in dico.items():
        if compteur < nbMots:
            dicoFinal[cle] = valeur
            compteur += 1
        else:
            break

    # Création d'un dictionnaire prêt pour être transformé en data frame
    dat = {'mots': [], 'nb': []}
    for i in dicoFinal:
        dat['mots'].append(i)
        dat['nb'].append(dicoFinal[i])

    # affichage et enregistrement
    if affichage:
        # Créez un DataFrame à partir des données
        df = pd.DataFrame(dat)

        # Créez un bar chart avec Plotly Express
        fig = px.bar(df, x='mots', y='nb', title='Nombre d’occurences des mots les plus cités')

        # Affichez le graphique
        if enregistrement:
            fig.write_html(nameEnregistrement)
        fig.show()

    return dicoFinal, dat


# Fonction permettant de récupérer les mots les plus cités dans un corpus en fonction des différentes années
def motsLesPlusUtilisesParAnsData(fichier, nbMots, affichage=True):
    """
    :param fichier: fichier à traiter
    :param nbMots: le nombre de mots que l'on veut dans la sortie de la fonction
    :param affichage: boolean qui indique si l'on veut afficher nos résultats à la fin de la fonction
    :return: retourne le dictionnaire des mots avec leurs occurrences ainsi que le dictionnaire permettant la création du dataframe
    """
    # name of the lightest file
    file_name = fichier

    # open and load file
    f = open(file_name, 'r', encoding='utf-8')
    data = json.loads(f.read())
    f.close()
    dico = {}

    # parcours du fichier
    for d in data["data-all"]:
        dico[d] = {}
        for mois in data["data-all"][d]:
            for article in data["data-all"][d][mois]:
                for etape in data["data-all"][d][mois][article]:
                    for name in etape:
                        if name == 'kws':
                            for mot in etape[name]:
                                if mot.lower() in dico[d]:
                                    dico[d][mot.lower()] += etape[name][mot]
                                else:
                                    dico[d][mot.lower()] = etape[name][mot]

    # Triez le dictionnaire par ordre croissant de ses valeurs en place
    for d in dico:
        dico[d] = dict(sorted(dico[d].items(), key=lambda item: item[1], reverse=True))

    compteur = 0
    dicoFinal = {}
    # Parcourez le dictionnaire
    for d in dico:
        dicoFinal[d] = {}
        compteur = 0
        for cle, valeur in dico[d].items():
            if compteur < nbMots:
                dicoFinal[d][cle] = valeur
                compteur += 1
            else:
                break

    # Création d'un dictionnaire prêt pour être transformé en data frame
    i = 0
    dat = {'annee': [], 'mots': [], 'nb': []}
    for d in dicoFinal:
        dat['annee'].append(d)
        dat['mots'].append([])
        dat['nb'].append([])
        for mot in dicoFinal[d]:
            dat['mots'][i].append(mot)
            dat['nb'][i].append(dicoFinal[d][mot])
        i += 1

    # Si jamais l'on souhaite un affichage celui-ci sera fait dans une app dash car cela permettra de choisir l'année que l'on souhaite afficher
    if affichage:
        # mettre les différentes années dans une liste
        options = []
        for d in dicoFinal:
            options.append(d)

        # Création de l'app
        app = Dash(__name__)

        # Structure de la page
        app.layout = html.Div([
            html.H4('Les 10 mots les plus fréquents par année'),
            # Dropdown permettant le choix de l'année
            dcc.Dropdown(
                id="dropdown",
                options=options,
                value=options[0],
                clearable=False,
            ),
            dcc.Graph(id="graph"),
        ])

        # Callback de la page, le graph en output et la valeur du dropdown en input
        @app.callback(
            Output("graph", "figure"),
            Input("dropdown", "value"))

        # Fonction pour update le contenu du dashboard en fonction l'année
        def update_bar_chart(year):
            # choix du dataframe
            df = pd.DataFrame(dat)

            # garder seulement la partie que l'on souhaite du dataframe
            filtered_df = df[df['annee'] == year]

            # Réorganisez les données
            mots = []
            nb = []
            for row in filtered_df.itertuples():
                mots.extend(row.mots)
                nb.extend(row.nb)

            # on recrée un dataframe depuis les nouvelles valeurs
            filtered_df = pd.DataFrame({'mots': mots, 'nb': nb})

            # affichage de l'histogramme en fonction des valeurs
            fig = px.bar(filtered_df, x="mots", y="nb", title=f"Les {nbMots} mots les plus fréquents en {year}")
            return fig

        # lancement de l'app
        app.run_server(debug=True)

    return dicoFinal, dat


'''----------------------------TF-------------------------------'''

""" Ici nous allons utiliser la méthode de term frequency """

def TFGlobalSortOccurrencesToJson(fichier, key, name):
    """
    Fonction qui permet de créer un json depuis un fichier en fonction de la clée souhaité
    L'objectif est de récupérer les mots les plus cités dans un corpus et de les trier avec la méthode de term frequency
    :param fichier: fichier à traiter
    :param key: choix des valeurs à traiter
    :param name: nom du fichier de sortie
    :return: ne retourne rien
    """
    # name of the lightest file
    file_name = fichier

    # open and load file
    f = open(file_name, 'r', encoding='utf-8')
    data = json.loads(f.read())
    f.close()

    # Récupération de la partie souhaitée
    dico = data['metadata-all']['fr']['all'][key]

    dicoFinal = {}
    nbMots = 0
    for mot in dico :
        nbMots += dico[mot]
    for mot in dico :
        dicoFinal[mot] = dico[mot]/nbMots

    dicoFinal = dict(sorted(dicoFinal.items(), key=lambda item: item[1], reverse=True))

    # Chemin du fichier où l'on souhaite enregistrer le JSON
    chemin_fichier = f"./data/TF/jsonTF/globalTF/TFGlobalSortOccurrencesOf----{key}----{name}"

    # Ouverture du fichier en mode écriture
    with open(chemin_fichier, "w") as fichier_json:
        # Utilisation json.dump pour écrire le dictionnaire dans le fichier au format JSON
        json.dump(dicoFinal, fichier_json)


def TFYearSortOccurrencesToJson(fichier, key, name):
    """
    Fonction qui permet de créer un json depuis un fichier en fonction de la clée souhaité
    L'objectif est de récupérer les mots les plus cités dans un corpus et de les trier avec la méthode de term frequency en fonction des années
    :param fichier: fichier à traiter
    :param key: choix des valeurs à traiter
    :param name: nom du fichier de sortie
    :return: ne retourne rien
    """

    # name of the lightest file
    file_name = fichier

    # open and load file
    f = open(file_name, 'r', encoding='utf-8')
    data = json.loads(f.read())
    f.close()

    # Récupération de la partie souhaitée
    dico = data['metadata-all']['fr']['year']
    final_dico = {}

    # Liste pour stocker les clés à supprimer
    cles_a_supprimer = []

    # parcours des années pour remplir le dictionnaire
    for year in dico:
        final_dico[year] = {}
        for elm in dico[year]:
            if elm != key:
                cles_a_supprimer.append(elm)
            else:
                final_dico[year][elm] = dict(sorted(dico[year][elm].items(), key=lambda item: item[1], reverse=True))

    # On supprime les clés en dehors de la boucle en vérifiant d'abord leur existence
    for year in final_dico:
        for elm in cles_a_supprimer:
            final_dico[year].pop(elm, None)


    # On crée le dictionnaire final avec un calcul de term frequency
    dicoFinal = final_dico.copy()
    for dico in final_dico :
        nbMots = 0
        for mot in final_dico[dico][key] :
            nbMots += final_dico[dico][key][mot]
        for mot in final_dico[dico][key] :
            dicoFinal[dico][key][mot] = final_dico[dico][key][mot] / nbMots



    # Chemin du fichier où l'on souhaite enregistrer le JSON
    chemin_fichier = f"./data/TF/jsonTF/yearTF/TFYearSortOccurrencesOf----{key}----{name}"

    # Ouverture le fichier en mode écriture
    with open(chemin_fichier, "w") as fichier_json:
        # Utilisation json.dump pour écrire le dictionnaire dans le fichier au format JSON
        json.dump(dicoFinal, fichier_json)



'''----------------------------TF-IDF-------------------------------'''


""" Ici nous allons utiliser la méthode de Term Frequency * Inverse Document Frequency """


def TFIDFGlobalSortOccurrencesToJson(fichier, key, namefichier) :
    """
    Fonction qui permet de créer un json depuis un fichier en fonction de la clée souhaité
    L'objectif est de récupérer les mots les plus cités dans un corpus et de les trier avec la méthode de Term Frequency * Inverse Document Frequency
    :param fichier: fichier à traiter
    :param key: choix des valeurs à traiter
    :param namefichier: nom du fichier de sortie
    :return: ne retourne rien
    """

    # name of the lightest file
    file_name = fichier

    # open and load file
    f = open(file_name, 'r', encoding='utf-8')
    data = json.loads(f.read())
    f.close()

    # Récupération des éléments souhaités
    nb_total_article = data['metadata-all']['fr']['all']['num']
    nbDocParMot = data['metadata-all']['fr']['all'][key]

    # On remplie le dictionnaire de 0
    for i in nbDocParMot :
        nbDocParMot[i] = 0

    # Parcours des données du corpus
    for d in data["data-all"]:
        for mois in data["data-all"][d]:
            for article in data["data-all"][d][mois]:
                for etape in data["data-all"][d][mois][article]:
                    for name in etape:
                        if name == key :
                            for m in etape[name] :
                                nbDocParMot[m] += 1
                                print(nbDocParMot)



    # On récupère les données que l'on a créées précédemment
    file_name = f'./data/TF/jsonTF/globalTF/TFGlobalSortOccurrencesOf----{key}----{namefichier}'

    # open and load file
    f = open(file_name, 'r', encoding='utf-8')
    data = json.loads(f.read())
    f.close()

    # Calcul du TF*IDF de chacun des mots
    for mot in nbDocParMot :
        nbDocParMot[mot] = (math.log((nb_total_article/nbDocParMot[mot])+1))*data[mot]

    # Trie des éléments
    nbDocParMot = dict(sorted(nbDocParMot.items(), key=lambda item: item[1], reverse=True))

    # Fichier où l'on souhaite enregistrer le JSON
    chemin_fichier = f"./data/TFIDF/jsonTFIDF/globalTFIDF/TFIDFGlobalSortOccurrencesOf----{key}----{namefichier}"

    # Ouverture du fichier en mode écriture
    with open(chemin_fichier, "w") as fichier_json:
        # Utilisation de json.dump pour écrire le dictionnaire dans le fichier au format JSON
        json.dump(nbDocParMot, fichier_json)



'''-------------------------------Word2Vec-------------------------------'''

""" Ici nous allons utiliser tester les différentes fonctionnalités de la méthode Word2Vec"""

def word2vec(fichier, key='kws') :
    """
    Test des fonctionnalités Word2Vec et leur affichage
    :param fichier: fichier à traiter
    :param key: type de données à traiter
    :return: ne retourne rien
    """

    # name of the lightest file
    file_name = fichier

    # open and load file
    f = open(file_name, 'r', encoding='utf-8')
    data = json.loads(f.read())
    f.close()

    # Récupérations des différentes ponctuations possibles
    ponctuations = list(string.punctuation)
    ponctuations.append('’')

    docFinal = []

    # Parcours des données
    for annee in data["data-all"]:
        for mois in data["data-all"][annee]:
            for article in data["data-all"][annee][mois]:
                dat = data["data-all"][annee][mois][article][0]['content'].split()
                dat = [doc.lower() for doc in dat]
                nDat = []
                for mot in dat:
                    nMot = mot
                    for l in mot:
                        if l in ponctuations:
                            if l == mot[0] or l == mot[len(mot) - 1]:
                                nMot = nMot.replace(l, "")
                            else:
                                indice = mot.index(l)
                                nMot = mot.replace(l, "")
                                nMot = mot[:indice - 1] + mot[indice + 1:]

                    if nMot in data["data-all"][annee][mois][article][0][key]:
                        nDat.append(nMot)
                docFinal.append(nDat)

    # permet d'afficher les mots du dictionnaire sous forme de string
    documents = [" ".join(doc) for doc in docFinal]
    print(documents[0])

    # création du modèle Word2Vec
    modele = Word2Vec(docFinal, vector_size=2, window=5)

    # propriété "wv" -> wordvector / mise sous forme de vecteur
    words = modele.wv

    # affichage de la forme du vecteur
    print(words.vectors.shape)

    # similarité entre 2 mots
    print(words.similarity('an','mali'))

    # mots les plus proches de
    print(words.most_similar("macron"))

    # plus proches de la conjonction de "an" et "mali" avec 4 mots
    print(words.most_similar(positive=['an', 'mali'], topn=4))

    # plus proches de "an", loin de ("mali")
    print(words.most_similar(positive=['an'], negative=['mali'], topn=4))


    # récupérer les données dans un data frame
    df = pd.DataFrame(words.vectors, columns=['V1', 'V2'], index=words.key_to_index.keys())
    print(df)

    # quelques mots clés
    mots = ['mali', 'disparition', 'rare', 'grève', 'droit', 'femme', 'homme']
    dfMots = df.loc[mots, :]
    print(dfMots)

    # affichage de ces mots clés sur un graphe
    plt.scatter(dfMots.V1, dfMots.V2, s=0.5)
    for i in range(dfMots.shape[0]):
        plt.annotate(dfMots.index[i], (dfMots.V1[i], dfMots.V2[i]))
    plt.show()

if __name__ == "__main__":

    """C'est ici que nous appelons les différentes fonction pour créer de nouveau fichier ou faire des tests"""

    # motsLesPlusUtilisesData('./jsonBases/topaz-data732--france--fr.sputniknews.africa--20190101--20211231.jsonBases',
    #                         nbMots=30,
    #                         affichage=True,
    #                         enregistrement=True,
    #                         nameEnregistrement='fichierSuptniknewqFr30mots.html')

    # motsLesPlusUtilisesParAnsData('./jsonBases/topaz-data732--france--fr.sputniknews.africa--20190101--20211231.json',
    #                               nbMots=30,
    #                               affichage=True)

    '''--------------------------TF--------------------------'''

    # for nom_fichier in os.listdir("./jsonBases"):
    #     TFGlobalSortOccurrencesToJson(f'./jsonBases/{nom_fichier}',
    #                                 'kws',
    #                                 f'{nom_fichier}')

    # for nom_fichier in os.listdir("./jsonBases"):
    #     TFYearSortOccurrencesToJson(f'./jsonBases/{nom_fichier}',
    #                               'kws',
    #                               f'{nom_fichier}')

    '''--------------------------TF_IDF--------------------------'''

    # TFIDFGlobalSortOccurrencesToJson('./jsonBases/topaz-data732--mali--french.presstv.ir--20190101--20211231.json',
    #                                  'kws',
    #                                  'topaz-data732--mali--french.presstv.ir--20190101--20211231.json')

    '''--------------------------Word2Vec--------------------------'''

    # word2vec('./jsonBases/topaz-data732--mali--french.presstv.ir--20190101--20211231.json')