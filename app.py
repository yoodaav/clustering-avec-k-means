from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

# Charger le modèle et le scaler
model = joblib.load('model/kmeans_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoder = LabelEncoder()

# Colonnes attendues par le modèle
COLUMNS = ['status_type', 'num_reactions', 'num_comments', 'num_shares', 
           'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            data = {
                'status_type': request.form['status_type'],
                'num_reactions': float(request.form['num_reactions']),
                'num_comments': float(request.form['num_comments']),
                'num_shares': float(request.form['num_shares']),
                'num_likes': float(request.form['num_likes']),
                'num_loves': float(request.form['num_loves']),
                'num_wows': float(request.form['num_wows']),
                'num_hahas': float(request.form['num_hahas']),
                'num_sads': float(request.form['num_sads']),
                'num_angrys': float(request.form['num_angrys'])
            }
            
            # Créer un DataFrame
            df = pd.DataFrame([data])
            
            # Encoder la variable catégorielle
            df['status_type'] = label_encoder.fit_transform(df['status_type'])
            
            # Normaliser les données
            X = df.values
            X_scaled = scaler.transform(X)
            
            # Prédiction du cluster
            cluster = model.predict(X_scaled)[0]
            
            return render_template('index.html', 
                                prediction=f"Ce post appartient au cluster {cluster}",
                                show_result=True)
        
        except Exception as e:
            return render_template('index.html', 
                                error=f"Erreur: {str(e)}",
                                show_result=True)
    
    return render_template('index.html', show_result=False)

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    app.run(debug=True)
