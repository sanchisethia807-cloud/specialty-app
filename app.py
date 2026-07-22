import streamlit as st
import joblib
import pandas as pd
import random

# Load the trained model and vectorizer
model = joblib.load('specialty_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
sample_notes = pd.read_csv('sample_notes.csv')

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

            st.subheader("Most Likely Specialties")
            for specialty, prob in results[:3]:
                pct = prob * 100
                st.markdown(f"""
                <div style="background:#1a1a2e; border:1px solid #2a2a4a; border-radius:12px;
                            padding:16px 20px; margin-bottom:12px;">
                  <div style="display:flex; justify-content:space-between; align-items:center;
                              margin-bottom:10px;">
                    <span style="font-size:1.05rem; font-weight:600; color:#fff;">{specialty}</span>
                    <span style="font-size:1.05rem; font-weight:700; color:#ff6b6b;">{pct:.1f}%</span>
                  </div>
                  <div style="background:#2a2a4a; border-radius:6px; height:10px; overflow:hidden;">
                    <div style="width:{pct}%; height:100%;
                                background:linear-gradient(90deg,#ff6b6b,#ff9e7d);"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.write("Coming soon! Structured symptom selection.")
