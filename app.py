import streamlit as st
import joblib
import pandas as pd
import random

# Load the trained model and vectorizer
model = joblib.load('specialty_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Medical Specialty Predictor")

tab1, tab2 = st.tabs(["Clinical Notes", "Symptom Checklist (coming soon)"])

with tab1:
    st.write("Paste in a clinical note below, and the model will guess which medical specialty it belongs to.")
    
    if st.button("🎲 Simulate with Random Sample"):
        random_note = sample_notes.sample(1).iloc[0]
        st.session_state.user_note = random_note['note']
    
    user_note = st.text_area("Clinical Note", height=200, key="user_note")
    
    if st.button("Predict Specialty", key="notes_predict"):
        if user_note.strip() == "":
            st.warning("Please paste in a note first.")
        else:
            note_vec = vectorizer.transform([user_note])
            probabilities = model.predict_proba(note_vec)[0]
            specialty_names = model.classes_
            results = sorted(zip(specialty_names, probabilities), key=lambda x: x[1], reverse=True)
            
            st.write("### Most Likely Specialties")
            for specialty, prob in results[:3]:
                st.write(f"**{specialty}**: {prob*100:.1f}% confidence")

with tab2:
    st.write("Coming soon — structured symptom selection.")
