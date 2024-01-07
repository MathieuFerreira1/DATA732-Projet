import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


# Fonction permettant d'afficher les élements en darkmode
def update_layout_dark_mode(fig):
    fig.update_layout(
        template='plotly_dark',  # Utilisation du template sombre intégré à Plotly
        paper_bgcolor='#161A28',  # Couleur de fond du papier
        plot_bgcolor='#161A28',   # Couleur de fond du graphe
        font=dict(color='white'),  # Couleur du texte
    )
    return fig

# Fonction d'affichage du dashboard
def dashboard1():
    # Les différents articles du corpus
    articles = ['mali--www.egaliteetreconciliation.fr',
                   'mali--french.presstv.ir',
                   'mali--fr.sputniknews.africa',
                   'france--www.fdesouche.com',
                   'france--www.egaliteetreconciliation.fr',
                   'france--french.presstv.ir',
                   'france--fr.sputniknews.africa']

    # Lancement de l'app avec un empêchement des excéption et le darkmode activé
    app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY])

    # Construction de la page
    app.layout = html.Div(children=[
        # Les différents styles
        html.Link(
            rel='stylesheet',
            href='/assets/style.css'
        ),
        # Le titre
        html.H1(
            "DASHBOARD DATA-732",
            id="titre"
        ),
        # Le dropdown permettant de sélectionner l'article dont on veut les données
        dcc.Dropdown(
            id="dropdown",
            options=[{'label': article, 'value': article} for article in articles],
            value=articles[0],
            clearable=False,
            style={'backgroundColor': '#1E2130', 'color': 'white', 'borderColor': 'white'}
        ),
        # Tabs permettant de choisir si on souhaite l'affichage du dashboard des pays, des keywords ou des peoples
        dcc.Tabs([
            dcc.Tab(label='Pays', value='tab_pays'),
            dcc.Tab(label='Keywords', value='tab_keywords'),
            dcc.Tab(label='People', value='tab_people')
        ],
        id='tabs',
        #On choisit la valeur de base comme les pays
        value='tab_pays'),
        html.Div(id='tabs-content')
    ])

    # Callback du choix de tab
    @app.callback(
        Output('tabs-content', 'children'),
        [Input('tabs', 'value')]
    )

    # Structure des différentes tabs
    def update_tab(selected_tab):
        # Tab des pays
        if selected_tab == 'tab_pays':
            return [
                html.Div(id="pays", children=[
                    html.H3(
                        "Occurrences",
                        id="occurrences-title"
                    ),
                    html.Div(id="paysContent", children=[
                        dcc.Graph(
                            id="map"

                        ),
                        dcc.Graph(
                            id="bar"
                        )
                    ]),
                ]),
                html.Div(id="paysWord2Vec", children=[
                    html.H3(
                        "Word2Vec Similarity",
                        id="word2vec-title"
                    ),
                    html.Div(id="paysWord2VecContent", children=[
                        dcc.Graph(
                            id="imshowPays"
                        ),
                        dcc.Graph(
                            id="scatterPays"
                        )
                    ]),
                ]),
            ]
        # Tab des keywords
        elif selected_tab == 'tab_keywords':
            return [
                html.Div(id="kws", children=[
                    html.H3(
                        "Occurrences",
                        id="occurrences-title"
                    ),
                    html.Div(id="kwsContent", children=[
                        dcc.Graph(
                            id="barKws"
                        ),
                        dcc.Graph(
                            id="cloudword"
                        )
                    ]),
                ]),
                html.Div(id="kwsWord2Vec", children=[
                    html.H3(
                        "Word2Vec Similarity",
                        id="word2vec-title-keyword"
                    ),
                    html.Div(id="kwsWord2VecContent", children=[
                        dcc.Graph(
                            id="imshowKws"
                        ),
                        dcc.Graph(
                            id="scatterKws"
                        )
                    ]),
                ]),
            ]
        # Tab des peoples
        elif selected_tab == 'tab_people':
            return [
                html.Div(id="per", children=[
                    html.H3(
                        "Occurrences",
                        id="occurrences-title"
                    ),
                    html.Div(id="perContent", children=[
                        dcc.Graph(
                            id="barPer"
                        ),
                        dcc.Graph(
                            id="cloudpeople"
                        )
                    ]),
                ]),
                html.Div(id="perWord2Vec", children=[
                    html.H3(
                        "Word2Vec Similarity",
                        id="word2vec-title-people"
                    ),
                    html.Div(id="perWord2VecContent", children=[
                        dcc.Graph(
                            id="imshowPer"
                        ),
                        dcc.Graph(
                            id="scatterPer"
                        )
                    ]),
                ]),
            ]


    # Modification du dashboard en fonction du dropdown et de l'article chosie

    # Callback pour les pays
    @app.callback(
        [Output('map', 'figure'),
         Output('bar', 'figure'),
         Output('imshowPays', 'figure'),
         Output('scatterPays', 'figure')],
        [Input('dropdown', 'value')]
    )

    # Update du dashboard pour les pays
    def update_pays(doc):
        # choix du dataframe pour la carte et l'histogramme
        pays = pd.read_csv(f"./data/Dashboard/pays/df_loc_nb_pays_occ_sans_france_mali----{doc}.csv")
        # On ne garde que les 10 premières lignes
        df_subset_pays = pays.head(10)
        # On trie les valeurs
        df_subset_pays = df_subset_pays.sort_values(by='Nombre articles pays', ascending=True)

        # Création de la carte
        carte = px.choropleth(pays,
                              locations="Code_ISO3",
                              color="Nombre articles pays",
                              hover_name="Pays",
                              color_continuous_scale=px.colors.sequential.Plasma)

        # Création de l'histogramme
        histo = px.bar(df_subset_pays,
                       x='Nombre articles pays',
                       y='Pays',
                       orientation='h')

        # Choix du dataframe pour la hitmap
        df_imshowPays = pd.read_csv(f"./data/Dashboard/pays/Word2VecMatrice/df_loc_pays_word2vec_sans_france_mali----{doc}.csv")

        # Création de la hitmap
        imshow = px.imshow(df_imshowPays, x=df_imshowPays.columns, y=df_imshowPays.columns)
        imshow.update_xaxes(side='top')

        # Choix du dataframe pour le graphique
        df_scatter = pd.read_csv(f"./data/Dashboard/pays/Word2VecVector/df_loc_pays_word2vec_vector----{doc}.csv")
        # Rename de la colonne des noms des pays
        df_scatter.columns = ["Pays" if col == "Unnamed: 0" else col for col in df_scatter.columns]
        # On ne garde que les 20 premières lignes
        df_scatter = df_scatter.head(20)

        # Création du graphique
        scatterPays = px.scatter(df_scatter, x='V1', y='V2')
        # Ajout des mots sur le graphique
        scatterPays.update_layout(annotations=[
            dict(x=x_value, y=y_value + 0.05, text=text_value, showarrow=False)
            for x_value, y_value, text_value in zip(df_scatter['V1'], df_scatter['V2'], df_scatter['Pays'])
        ])

        # Appliquer le thème sombre
        update_layout_dark_mode(carte)
        update_layout_dark_mode(histo)
        update_layout_dark_mode(imshow)
        update_layout_dark_mode(scatterPays)

        return carte, histo, imshow, scatterPays


    # Callback pour les keywords
    @app.callback(
        [Output('barKws', 'figure'),
         Output('cloudword', 'figure'),
         Output('imshowKws', 'figure'),
         Output('scatterKws', 'figure')],
        [Input('dropdown', 'value')]
    )

    # Update du dashboard pour les pays
    def update_keywords(doc):
        # Choix du dataframe pour l'histogramme
        kws = pd.read_csv(f"./data/Dashboard/kws/df_kws_occ_sans_france_mali----{doc}.csv")
        # On ne garde que les 15 premières lignes
        df_subset_kws = kws.head(15)

        # Création de l'histogramme
        histo_kws = px.bar(df_subset_kws,
                           x='Keywords',
                           y='Nombre occurrences',
                           title=f'Histo keywords les plus cité dans : {doc} avec la méthode TF')


        # Création du string pour faire le wordcloud
        text_data = ' '.join([keyword + ' ' * int(occurrences * 10000) for keyword, occurrences in
                              zip(df_subset_kws['Keywords'], df_subset_kws['Nombre occurrences'])])

        # Création du wordcloud
        wordcloud = WordCloud(background_color="black").generate(text_data)
        wordcloudKws = go.Figure(go.Image(z=wordcloud.to_array()))

        # Choix du dataframe pour la hitmap
        df_imshowKws = pd.read_csv(f"./data/Dashboard/kws/Word2VecMatrice/df_kws_word2vec_sans_france_mali----{doc}.csv")

        # Création de la hitmap
        imshow = px.imshow(df_imshowKws, x=df_imshowKws.columns, y=df_imshowKws.columns)
        imshow.update_xaxes(side='top')

        # Choix du dataframe pour le graphique
        df_scatter = pd.read_csv(f"./data/Dashboard/kws/Word2VecVector/df_kws_word2vec_vector----{doc}.csv")
        # Rename de la colonne des noms des keywords
        df_scatter.columns = ["Keywords" if col == "Unnamed: 0" else col for col in df_scatter.columns]
        # On supprime les mots non significatifs
        df_scatter = df_scatter.loc[~df_scatter['Keywords'].isin(['burkina', 'afrique', 'faso', 'libre', 'journal', 'mali', 'macron', 'côte'])]
        # On ne garde que les 20 premières lignes
        df_scatter = df_scatter.head(20)

        # Création du graphique
        scatterKws = px.scatter(df_scatter, x='V1', y='V2')
        # Ajout des mots sur le graphique
        scatterKws.update_layout(annotations=[
            dict(x=x_value, y=y_value + 0.08, text=text_value, showarrow=False)
            for x_value, y_value, text_value in zip(df_scatter['V1'], df_scatter['V2'], df_scatter['Keywords'])
        ])

        # Appliquer le thème sombre
        update_layout_dark_mode(histo_kws)
        update_layout_dark_mode(wordcloudKws)
        update_layout_dark_mode(imshow)
        update_layout_dark_mode(scatterKws)



        return histo_kws,wordcloudKws, imshow, scatterKws


    # Callback pour les personnes
    @app.callback(
        [Output('barPer', 'figure'),
         Output('cloudpeople', 'figure'),
         Output('imshowPer', 'figure'),
         Output('scatterPer', 'figure')],
        [Input('dropdown', 'value')]
    )

    # Update du dashboard pour les personnes
    def update_people(doc):
        # Choix du dataframe pour l'histogramme
        per = pd.read_csv(f"./data/Dashboard/per/df_per_occ----{doc}.csv")
        # On ne garde que les 15 premières lignes
        df_subset_per = per.head(15)

        # Création de l'histogramme
        histo_per = px.bar(df_subset_per,
                           x='People',
                           y='Somme des valeurs',
                           title=f'Histo People les plus cité dans : {doc} avec la méthode TF')

        # Création du string pour faire le wordcloud
        text_data = ' '.join([keyword + ' ' * int(occurrences * 10000) for keyword, occurrences in
                              zip(df_subset_per['People'], df_subset_per['Somme des valeurs'])])

        # Création du wordcloud
        wordcloud = WordCloud(background_color="black").generate(text_data)
        wordcloudPer = go.Figure(go.Image(z=wordcloud.to_array()))

        # Choix du dataframe pour la hitmap
        df_imshowPer = pd.read_csv(f"./data/Dashboard/per/Word2VecMatrice/df_per_word2vec----{doc}.csv")

        # Création de la hitmap
        imshow = px.imshow(df_imshowPer, x=df_imshowPer.columns, y=df_imshowPer.columns)
        imshow.update_xaxes(side='top')

        # Choix du dataframe pour le graphique
        df_scatter = pd.read_csv(f"./data/Dashboard/per/Word2VecVector/df_per_word2vec_vector----{doc}.csv")
        # Rename de la colonne des noms des peoples
        df_scatter.columns = ["People" if col == "Unnamed: 0" else col for col in df_scatter.columns]
        # On supprime les mots non significatifs
        df_scatter = df_scatter.loc[~df_scatter['People'].isin([])]
        # On ne garde que les 20 premières lignes
        df_scatter = df_scatter.head(20)

        # Création du graphique
        scatterPer = px.scatter(df_scatter, x='V1', y='V2')
        # Ajout des mots sur le graphique
        scatterPer.update_layout(annotations=[
            dict(x=x_value, y=y_value + 0.05, text=text_value, showarrow=False)
            for x_value, y_value, text_value in zip(df_scatter['V1'], df_scatter['V2'], df_scatter['People'])
        ])

        # Appliquer le thème sombre
        update_layout_dark_mode(histo_per)
        update_layout_dark_mode(wordcloudPer)
        update_layout_dark_mode(imshow)
        update_layout_dark_mode(scatterPer)


        return histo_per,wordcloudPer, imshow, scatterPer

    app.run_server(debug=True)



# Lancement de l'app
if __name__ == "__main__":
    dashboard1()
