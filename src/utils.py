# src/utils.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_distribution(preds):
    fig, ax = plt.subplots()
    counts = preds.value_counts()
    counts.plot(kind='bar', color=["red", "green"], ax=ax)
    
    #preds.value_counts(normalize=True).plot(kind='bar', color=["red", "green"], ax=ax)
    
    ax.set_title("Distribution des sentiments prédits")
    ax.set_ylabel("Nombre de prédictions")
    
    for i, val in enumerate(counts):
        ax.text(i, val + 0.2, str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    return fig

def update_monitoring_log(id_tweet, timestamp, tweet, label, proba, file="data/processed/predictions_log.csv"):
    row = pd.DataFrame([{
        "id_tweet":id_tweet,
        "timestamp":timestamp,
        "tweet": tweet,
        "label": label,
        "proba_positive": proba
    }])
    try:
        df_old = pd.read_csv(file)
        df = pd.concat([df_old, row], ignore_index=True)
    except FileNotFoundError:
        df = row
    df.to_csv(file, index=False)
