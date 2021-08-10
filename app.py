# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.write('# Détecteur de faux billets')

# loading dataset
notes_with_components = pd.read_csv('notes_with_components.csv', index_col=0)

X_log = notes_with_components.drop(columns=['is_genuine', 'F1', 'F2', 'F3', 'F4'])
y = notes_with_components.is_genuine

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_log, notes_with_components.is_genuine, test_size = 0.33, random_state=42)


logreg = LogisticRegression(max_iter=400, C=0.1, penalty='l2')

logreg.fit(X_train, y_train)

uploaded_file = st.file_uploader('Soumettez votre liste de billets')
if uploaded_file is not None:
    st.write('## Affichage des billets à détecter :')
    tested_notes = pd.read_csv(uploaded_file)
    st.write(tested_notes)
    
    # Apply detection model on the submitted notes
    tested_notes_into_model = tested_notes.drop(columns=['id'])
    probs = logreg.predict_proba(tested_notes_into_model)
    results = pd.DataFrame(probs, index=tested_notes.id, columns=['Faux', 'Vrai'])
    conditions = [results['Vrai'] >= 0.5, results['Vrai'] < 0.5]
    results['Prédiction'] = np.select(conditions, ['Vrai', 'Faux'])

    st.write('## Prédictions')
    results
