import json
import os

import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# name of the lightest file
file_name = '../jsonBases/topaz-data732--mali--www.egaliteetreconciliation.fr--20190101--20211231.json'

# open and load file
f = open(file_name, 'r', encoding='utf-8')
data = json.loads(f.read())
f.close()


def graphique1() :
    dico = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0, '6' : 0, '7' : 0, '8' : 0, '9' : 0, '10' : 0, '11' : 0, '12' : 0}
    dat = {'months' : [], 'nb' : []}

    for d in data["data-all"] :
        for mois in data["data-all"][d] :
            for _ in data["data-all"][d][mois] :
                dico[mois] +=1

    for i in dico :
        dat['months'].append(i)
        dat['nb'].append(dico[i])


    # Créez un DataFrame à partir des données
    df = pd.DataFrame(dat)

    # Créez un bar chart avec Plotly Express
    fig = px.bar(df, x='months', y='nb', title='Nombre d’articles par mois')

    # Affichez le graphique
    fig.write_html("graphique.html")
    fig.show()


def graphique2() :
    dico = {}
    for d in data["data-all"] :
        for mois in data["data-all"][d] :
            for article in data["data-all"][d][mois] :
                for etape in data["data-all"][d][mois][article] :
                    for name in etape :
                        if name=='kws' or name=='loc' or name=='mis' or name=='org' or name=='per' :
                            for mot in etape[name] :
                                if mot.lower() in dico :
                                    dico[mot.lower()] += etape[name][mot]
                                else :
                                    dico[mot.lower()] = etape[name][mot]

    # Triez le dictionnaire par ordre croissant de ses valeurs en place
    dico = dict(sorted(dico.items(), key=lambda item: item[1], reverse=True))

    compteur = 0
    dicoFinal ={}
    # Parcourez le dictionnaire
    for cle, valeur in dico.items():
        if compteur < 10:
            dicoFinal[cle] = valeur
            compteur += 1
        else:
            break


    dat = {'mots': [], 'nb': []}
    for i in dicoFinal :
        dat['mots'].append(i)
        dat['nb'].append(dicoFinal[i])


    # Créez un DataFrame à partir des données
    df = pd.DataFrame(dat)

    # Créez un bar chart avec Plotly Express
    fig = px.bar(df, x='mots', y='nb', title='Nombre d’occurences des mots les plus cités')

    # Affichez le graphique
    fig.write_html("graphique2.html")
    fig.show()


def graphique3() :
    dico = {}
    for d in data["data-all"] :
        dico[d] = {}
        for mois in data["data-all"][d] :
            for article in data["data-all"][d][mois] :
                for etape in data["data-all"][d][mois][article] :
                    for name in etape :
                        if name=='kws' or name=='loc' or name=='mis' or name=='org' or name=='per' :
                            for mot in etape[name] :
                                if mot.lower() in dico[d] :
                                    dico[d][mot.lower()] += etape[name][mot]
                                else :
                                    dico[d][mot.lower()] = etape[name][mot]

    # Triez le dictionnaire par ordre croissant de ses valeurs en place
    for d in dico :
        dico[d] = dict(sorted(dico[d].items(), key=lambda item: item[1], reverse=True))

    compteur = 0
    dicoFinal = {}
    # Parcourez le dictionnaire
    for d in dico :
        dicoFinal[d] = {}
        compteur = 0
        for cle, valeur in dico[d].items():
            if compteur < 10:
                dicoFinal[d][cle] = valeur
                compteur += 1
            else:
                break


    i=0
    dat = {'annee' : [], 'mots' : [], 'nb' : []}
    for d in dicoFinal :
        dat['annee'].append(d)
        dat['mots'].append([])
        dat['nb'].append([])
        for mot in dicoFinal[d] :
            dat['mots'][i].append(mot)
            dat['nb'][i].append(dicoFinal[d][mot])
        i +=1



    options = []
    for d in dicoFinal :
        options.append(d)

    app = Dash(__name__)

    app.layout = html.Div([
        html.H4('Les 10 mots les plus fréquents par année'),
        dcc.Dropdown(
            id="dropdown",
            options=options,
            value=options[0],
            clearable=False,
        ),
        dcc.Graph(id="graph"),
    ])

    @app.callback(
        Output("graph", "figure"),
        Input("dropdown", "value"))
    def update_bar_chart(year):
        df = pd.DataFrame(dat)

        filtered_df = df[df['annee'] == year]

        # Réorganisez les données
        mots = []
        nb = []
        for row in filtered_df.itertuples():
            mots.extend(row.mots)
            nb.extend(row.nb)

        filtered_df = pd.DataFrame({'mots': mots, 'nb': nb})

        fig = px.bar(filtered_df, x="mots", y="nb", title=f"Les 10 mots les plus fréquents en {year}")
        return fig


    app.run_server(debug=True)

if __name__ == "__main__":
    graphique1()
    graphique2()
    graphique3()