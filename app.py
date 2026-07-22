import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('specialty_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Medical Specialty Predictor")

tab1, tab2 = st.tabs(["Clinical Notes", "Symptom Checklist (coming soon)"])

with tab1:
    st.write("Paste in a clinical note below, and the model will guess which medical specialty it belongs to.")
    
    user_note = st.text_area("Clinical Note", height=200)
    
    if st.button("Predict Specialty", key="notes_predict"):
        if user_note.strip() == "":
            st.warning("Please paste in a note first.")
        else:
            # Translate the note into numbers, then predict
            note_vec = vectorizer.transform([user_note])
            
            # Get confidence scores for all specialties
            probabilities = model.predict_proba(note_vec)[0]
            specialty_names = model.classes_
            
            # Pair them up and sort by confidence, highest first
            results = sorted(zip(specialty_names, probabilities), key=lambda x: x[1], reverse=True)
            
            st.write("### Most Likely Specialties")
            for specialty, prob in results[:3]:
                st.write(f"**{specialty}**: {prob*100:.1f}% confidence")

with tab2:
    st.write("Coming soon — structured symptom selection.")
