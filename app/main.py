# app/main.py
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import nettoyer_tweet
from src.utils import plot_distribution, update_monitoring_log
from sklearn.decomposition import PCA
from datetime import datetime
from wordcloud import WordCloud
import scipy.sparse

# Charger le modèle
model = joblib.load("models/logreg_tfidf_final.pkl")

st.set_page_config(page_title="Analyse de sentiments", layout="wide")

#Navigation
PAGE_ANALYSE = "Analyse de sentiments"
PAGE_MONITORING = "Suivi du modèle"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", [PAGE_ANALYSE, PAGE_MONITORING])

if page == PAGE_ANALYSE:
    
    # Configuration de la page
    st.title("🔍 Surveillez ce que pensent vos clients, en temps réel")
    st.info("Ce site est une démonstration d’analyse automatique des sentiments appliquée aux avis clients.")

    st.markdown("""
    ---
    ### 💡 Pourquoi cette application ?

    Les avis partagés en ligne (réseaux sociaux, Google, plateformes e-commerce) sont une source précieuse d'informations pour toute entreprise.  
    Cette application montre comment une **analyse automatique des sentiments** peut vous aider à :

    - Mesurer la **satisfaction client en temps réel**
    - Détecter des **problèmes récurrents** dans vos produits ou services
    - **Réagir rapidement** à une crise d’image ou une mauvaise publicité
    - **Prioriser les feedbacks négatifs** pour améliorer l’expérience client
    - Suivre l’impact d’une **campagne marketing** ou d’un nouveau lancement

    ---
    ### 🔍 À propos du modèle

    - Entraîné sur un **corpus de 10.000 tweets en anglais dont 5000 positifs et 5000 negatifs**
    - Basé sur une combinaison **Bag of Words avec TF-IDF + régression logistique**
    - Rapide, léger et adapté aux cas simples
    - ⚠️ *Le modèle fonctionne mieux avec des textes en anglais.*

    ---
    ### ✍️ Essayez-le en direct
    """)
    
    with st.form("formulaire"):
        tweet = st.text_area("💬 Entrez un avis ou un tweet :", height=100, placeholder="Ex: I really love this product, it changed my life!")
        submit = st.form_submit_button("Analyser")

    if submit and tweet:
        # Nettoyage + prédiction
        cleaned = nettoyer_tweet(tweet)
        prediction = model.predict([cleaned])[0]
        proba = model.predict_proba([cleaned])[0]
        label = "Positif" if prediction == 1 else "Négatif"

        # Logging pour suivi
        id_tweet = str(uuid.uuid1())
        timestamp = datetime.now().isoformat()
        
        update_monitoring_log(id_tweet, timestamp, tweet, label, proba[1])
        st.session_state["last_cleaned"] = cleaned
        st.session_state["last_label"] = prediction

        # Affichage
        st.success(f"**Sentiment détecté : {label}**")
        st.write("📊 Probabilités :", {"Négatif": round(proba[0], 3), "Positif": round(proba[1], 3)})

        # Call-to-action pour PME
        st.markdown("""
        ---
        🔎 Vous êtes une entreprise ?
        👉 [Contactez-moi](mailto:gpresleyk@gmail.com) pour une version personnalisée, intégrée à vos propres données clients.
        """)
        
        
elif page==PAGE_MONITORING:
    
    # Introduction commerciale
    st.title("📡 Suivi en temps réel des sentiments")
    st.markdown("""
    Cette section vous aide à **surveiller automatiquement la perception de votre produit ou service**, grâce à des indicateurs clés dérivés des prédictions de sentiment.

    ### ✅ Objectifs :
    - Détecter les **problèmes récurrents** évoqués dans les retours clients
    - Suivre l’**évolution de la satisfaction** dans le temps
    - Identifier les **avis négatifs à fort impact émotionnel**
    - Déclencher des **alertes automatiques** en cas de crise d’image

    ---
    > ⚠️ *Les commentaires affichés ici sont **fictifs** et utilisés à des fins **illustratives**. Ils représentent des avis hypothétiques sur un produit ou service générique d'entreprise.*
    """)

    # Bloc d'information technique
    with st.expander("ℹ️ Détails techniques du suivi", expanded=False):
        st.info("""
        Ce tableau de bord est généré à partir des textes soumis dans l’onglet « Analyse ».  
        Il permet un **monitoring continu** du modèle sans nécessiter de vraies étiquettes (`true labels`).  
        Voici ce que vous pouvez suivre :
        - Répartition des classes prédites (`positif` vs `négatif`)
        - Distribution des probabilités (incertitude du modèle)
        - Évolution temporelle des prédictions
        - Nuage de mots basé sur les retours négatifs
        """)
        
    # Chargement du log des prédictions
    df_log = pd.read_csv("data/processed/predictions_log.csv")
    
    #Extraction des probabilités
    x_pos = df_log["proba_positive"] #probabilité de la classe positive
    y_neg = 1 - df_log["proba_positive"] #probabilité des classe negatives
    labels = df_log["label"]
    
    # Préparation des couleurs
    colors = ["green" if lbl == "Positif" else "red" for lbl in labels]
    
    # Création du nuage
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x_pos, y_neg, c=colors, s=100, edgecolor='k', alpha=0.7)
    
    # Diagonale x = y pour visualiser l’incertitude
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Dernier point en bleu
    if not df_log.empty:
        ax.scatter(x_pos.iloc[-1], y_neg.iloc[-1], c="blue", s=150, edgecolor="black", label="Dernier tweet")
        ax.annotate("Dernier tweet", (x_pos.iloc[-1] + 0.01, y_neg.iloc[-1]), fontsize=10, color="blue")

    # Mise en forme
    ax.set_xlabel("Probabilité que ce soit positif")
    ax.set_ylabel("Probabilité que ce soit négatif")
    ax.set_title("📊 Nuage de prédictions – Positif vs Négatif")
    ax.legend()
    st.pyplot(fig)
    
     # 📈 Répartition des classes
    st.subheader("📈 Répartition des prédictions")
    fig_dist = plot_distribution(df_log["label"])
    st.pyplot(fig_dist)
    
    #Évolution du sentiment dans le temps
    st.subheader("🕒 Évolution du sentiment positif dans le temps")
    df_log["timestamp"] = pd.to_datetime(df_log["timestamp"])
    daily_avg = df_log.groupby(df_log["timestamp"].dt.date)["proba_positive"].mean().reset_index()
    daily_avg.columns = ["Date", "Moyenne probabilité positive"] 
    
    # Afficher le graphique
    #st.subheader("📈 Moyenne journalière de la probabilité positive")
    st.line_chart(daily_avg.set_index("Date"))
        
    #Détection des problèmes récurrents
    #st.subheader("Détection des problèmes récurrents")
    
    
    neg_texts = df_log[df_log["label"]=="Négatif"]["tweet"].str.cat(sep=" ")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(neg_texts)
    
    st.subheader("☁️ Problèmes fréquents détectés dans les retours négatifs")
    #st.write("Les commentaires ci-dessous sont **fictifs** et ont été générés à des fins **illustratives uniquement**. Ils simulent des **avis positifs et négatifs** qu’un client pourrait laisser à propos d’un **produit ou service d’une entreprise**. Ces données permettent de **visualiser la perception globale** à travers un nuage de mots.")
    fig_wc, ax_wc = plt.subplots(figsize = (5, 3))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)
    
    # Alerte automatique
    st.subheader("🚨 Alerte en cas de crise d’image")
    neg_ratio = (df_log["label"] == "Négatif").mean()
    if neg_ratio > 0.5:
        st.error(f"⚠️ Attention : Selon les predictions, {round(neg_ratio*100)}% des feedbacks récents sont négatifs. Risque de crise détecté.")
    else:
        st.success(f"👍 Actuellement, {round(neg_ratio*100)}% de feedbacks sont négatifs. Aucun signal critique.")
        
        
    st.subheader("🔍 Feedbacks négatifs à fort impact")
    df_log["uncertainty"] = 1 - abs(df_log["proba_positive"] - 0.5) * 2  # Plus proche de 0.5 = plus incertain
    top_neg = df_log[df_log["label"] == "Négatif"].sort_values(by="uncertainty", ascending=False).head(5)
    st.table(
        top_neg[["tweet", "proba_positive", "uncertainty"]]
        .rename(columns={"tweet":"Commentaire Negatif", "proba_positive": "Proba Positive", "uncertainty": "Incertitude"})
        )
        
        
    #Pied de page
    st.markdown("""
                    <hr style="border: 0.5px solid #ddd;">
                    <div style="text-align:center">
                    <small>© 2025 – Outil de prédiction AVC – Fait avec ❤️ par Presley Koyaweda</small>
                    </div>
        """, unsafe_allow_html=True)